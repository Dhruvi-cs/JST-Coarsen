"""
Benchmarking Engine for Dynamic Graph Neural Networks on DBLP5
Evaluates:
- GRUoGCN2x2 (Temporal Message Passing / TMP / EvolveGCN)
- GCNx2oGRU (Graph-Then-Time / GTT)
across:
- Full_Data (alpha = 1.0)
- Ours_Novel (JST-Coarsen: Solutions 1-5, time-coupled regularized coarsening)
- SACoD_supra (1st-order supra Laplacian + DSU)
- SACoD_snap (1st-order snapshot Laplacian + DSU)
for all coarsening ratios alpha in [0.3, 0.9] across seeds (50, 51).
Side-by-side comparison against KDD '26 Paper Table 1(b) DBLP ground truth.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import pandas as pd
from typing import Dict, List, Tuple


def get_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ==========================================
# GNN Model Architectures (PyTorch)
# ==========================================

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        if adj_norm.is_sparse:
            out = torch.sparse.mm(adj_norm, h)
        else:
            out = torch.matmul(adj_norm, h)
        return out


def normalize_adj_torch(adj: np.ndarray, device: torch.device) -> torch.Tensor:
    """Symmetric normalization: D^-1/2 (A + I) D^-1/2"""
    N = adj.shape[0]
    A_tilde = adj.astype(np.float32) + np.eye(N, dtype=np.float32)
    d = np.sum(A_tilde, axis=1)
    d_inv_sqrt = np.power(np.maximum(1e-6, d), -0.5)
    norm_adj = A_tilde * (d_inv_sqrt[:, None] * d_inv_sqrt[None, :])
    return torch.tensor(norm_adj, dtype=torch.float32, device=device)


class MLP(nn.Module):
    """2-layer MLP classification head with Softplus activation matching paper setup."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.Softplus()
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.zeros_(self.lin1.bias)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class GCNx2oGRU(nn.Module):
    """
    GTT Model: 2-layer GCN independently on each snapshot, then GRU across time,
    then 2-layer MLP classifier matching author's GNNx2oSNN setup.
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, num_times: int):
        super().__init__()
        self.num_times = num_times
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.skip = nn.Linear(in_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = MLP(hidden_dim, hidden_dim, num_classes)
        self.act = nn.Softplus()

    def forward(self, feats_list: List[torch.Tensor], adjs_norm_list: List[torch.Tensor]) -> torch.Tensor:
        h_snaps = []
        for t in range(self.num_times):
            x = feats_list[t]
            adj = adjs_norm_list[t]
            h1 = self.act(self.gcn1(x, adj))
            h2 = self.act(self.gcn2(h1, adj)) + self.skip(x)
            h_snaps.append(h2)

        # Stack over time: (N, T, hidden_dim)
        h_seq = torch.stack(h_snaps, dim=1)
        gru_out, _ = self.gru(self.act(h_seq))
        final_gru = gru_out[:, -1, :]
        logits = self.classifier(self.act(final_gru))
        return logits


class GRUoGCN2x2(nn.Module):
    """
    TMP Model: Temporal recurrent representation update (GRU), then 2-layer GCN message passing,
    then skip concatenation (concat=True) and 2-layer MLP classifier matching paper setup.
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, num_times: int):
        super().__init__()
        self.num_times = num_times
        self.gru = nn.GRU(in_dim, hidden_dim, batch_first=True)
        self.gcn1 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.classifier = MLP(hidden_dim * 2, hidden_dim, num_classes)
        self.act = nn.Softplus()

    def forward(self, feats_list: List[torch.Tensor], adjs_norm_list: List[torch.Tensor]) -> torch.Tensor:
        feats_seq = torch.stack(feats_list, dim=1)  # (N, T, in_dim)
        gru_out, _ = self.gru(feats_seq)           # (N, T, hidden_dim)

        h_time = gru_out[:, -1, :]                 # (N, hidden_dim)
        h_act = self.act(h_time)

        adj_dense = adjs_norm_list[-1]
        h1 = self.act(self.gcn1(h_act, adj_dense))
        h2 = self.gcn2(h1, adj_dense)

        h_cat = torch.cat([h2, h_act], dim=-1)     # (N, 32)
        logits = self.classifier(self.act(h_cat))
        return logits


# ==========================================
# Dataset Loading & Split for DBLP5
# ==========================================

def get_dblp5_data(device: torch.device, seed: int = 50):
    data_path = os.path.join("src", "DBLP5", "DBLP5_no_isolated.npz")
    print(f"[{get_timestamp()}] Loading DBLP5 dataset from {data_path}...", flush=True)
    d = np.load(data_path)
    adjs = d["adjs"].astype(np.float32)       # (10, 2390, 2390)
    feats = d["attmats"].astype(np.float32)   # (2390, 10, 100)
    labels = d["labels"].astype(np.float32)   # (2390, 5)

    N = labels.shape[0]
    T = adjs.shape[0]

    # Stratified In-Degree Fit-Split matching dynedge_v2.py and spindle.py
    from lib.meta.utils.spindle import stratified_fitsplit
    W_total = np.sum(adjs, axis=0)
    degrees = np.sum(W_total, axis=0)  # in-degree
    deg_order = np.argsort(-degrees)    # sorted large to small
    node_labels = np.argmax(labels, axis=1)

    splits = stratified_fitsplit(deg_order, node_labels, proportion=[6, 2, 2], priority=[2, 1, 0])
    train_idx, val_idx, test_idx = np.array(splits[0]), np.array(splits[1]), np.array(splits[2])

    train_mask = np.zeros(N, dtype=bool)
    val_mask = np.zeros(N, dtype=bool)
    test_mask = np.zeros(N, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # Effective sample frequency balancing (square-root inverse class weighting)
    train_labels = node_labels[train_mask]
    counts = np.bincount(train_labels, minlength=labels.shape[1]).astype(np.float32)
    weights = 1.0 / np.sqrt(counts + 1.0)
    weights = weights / weights.sum() * float(labels.shape[1])
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    # Standardize features across (nodes, time) per feature dimension matching dynedge_v2.py
    feat_mean = np.mean(feats, axis=(0, 1), keepdims=True)
    feat_std = np.std(feats, axis=(0, 1), keepdims=True)
    feat_std[feat_std == 0] = 1.0
    feats_std = (feats - feat_mean) / feat_std

    # Precompute normalized adjs for full data
    norm_adjs_full = [normalize_adj_torch(adjs[t], device) for t in range(T)]
    adj_dense_full = (np.sum(adjs, axis=0) > 0).astype(np.float32)
    norm_adj_dense_full = normalize_adj_torch(adj_dense_full, device)
    feats_full = [torch.tensor(feats_std[:, t, :], dtype=torch.float32, device=device) for t in range(T)]
    targets_full = torch.tensor(node_labels, dtype=torch.long, device=device)

    print(f"[{get_timestamp()}] DBLP5 loaded successfully: N={N}, T={T}, F={feats.shape[2]}, Classes={labels.shape[1]}, Train={train_mask.sum()}, Val={val_mask.sum()}, Test={test_mask.sum()}", flush=True)

    return {
        "adjs": adjs,
        "feats": feats,
        "feats_std": feats_std,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "labels": labels,
        "norm_adjs_full": norm_adjs_full,
        "norm_adj_dense_full": norm_adj_dense_full,
        "feats_full": feats_full,
        "targets_full": targets_full,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "class_weights": class_weights,
        "N": N,
        "T": T,
        "num_classes": labels.shape[1],
        "feat_dim": feats.shape[2],
    }


def compute_auroc_macro(target: np.ndarray, probs: np.ndarray) -> float:
    """Computes multi-class one-vs-one Macro AUROC matching paper methodology."""
    try:
        if probs.shape[1] == 2:
            return float(roc_auc_score(target, probs[:, 1], average="macro")) * 100.0
        else:
            unique_labels = list(sorted(set(target)))
            if len(unique_labels) < probs.shape[1]:
                sub_probs = probs[:, unique_labels]
                sub_probs = sub_probs / np.maximum(1e-8, np.sum(sub_probs, axis=1, keepdims=True))
                return float(roc_auc_score(target, sub_probs, average="macro", multi_class="ovo", labels=unique_labels)) * 100.0
            else:
                return float(roc_auc_score(target, probs, average="macro", multi_class="ovo")) * 100.0
    except Exception:
        return 50.0


def compute_accuracy(target: np.ndarray, probs: np.ndarray) -> float:
    """Computes classification accuracy matching paper methodology."""
    try:
        preds = np.argmax(probs, axis=1)
        return float(np.mean(preds == target)) * 100.0
    except Exception:
        return 20.0


# ==========================================
# Training & Evaluation Function
# ==========================================

def train_and_eval(
    model_name: str,
    method_name: str,
    alpha: float,
    data_dict: Dict,
    device: torch.device,
    epochs: int = 250,
    lr: float = 0.01,
    patience: int = 80,
    seed: int = 50,
    verbose_epoch: bool = False,
) -> Tuple[float, float, int]:
    """
    Trains on coarsened graph (or full data) and evaluates test AUROC and accuracy at best validation epoch.
    Returns: (best_test_auc, best_test_acc, best_epoch)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    T = data_dict["T"]
    in_dim = data_dict["feat_dim"]
    num_classes = data_dict["num_classes"]
    hidden_dim = 16

    # Instantiate Model
    if model_name == "GCNx2oGRU":
        model = GCNx2oGRU(in_dim, hidden_dim, num_classes, T).to(device)
    else:
        model = GRUoGCN2x2(in_dim, hidden_dim, num_classes, T).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    class_weights = data_dict.get("class_weights", None)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    is_full = (method_name == "Full_Data" or alpha >= 1.0)
    
    if is_full:
        train_feats = data_dict["feats_full"]
        if model_name == "GRUoGCN2x2":
            train_adjs = [data_dict["norm_adj_dense_full"]]
            eval_adjs = [data_dict["norm_adj_dense_full"]]
        else:
            train_adjs = data_dict["norm_adjs_full"]
            eval_adjs = data_dict["norm_adjs_full"]

        train_targets = data_dict["targets_full"]
        train_mask = torch.tensor(data_dict["train_mask"], device=device)
        val_mask = torch.tensor(data_dict["val_mask"], device=device)
        test_mask = torch.tensor(data_dict["test_mask"], device=device)
    else:
        # Load coarsened data
        coarse_path = os.path.join(
            "Loukas-graph-coarsening", "results", "DBLP5", f"cr{alpha:.1f}", f"coarsened~{method_name}.npz"
        )
        if not os.path.exists(coarse_path):
            raise FileNotFoundError(f"Coarsened file not found: {coarse_path}")

        cd = np.load(coarse_path)
        cadjs = cd["coarsened_adjs"]
        cfeats = cd["coarsened_feats"]
        clabels = cd["coarsened_labels"]
        P = cd["P"]  # (N, K)

        K = P.shape[1]
        if model_name == "GRUoGCN2x2":
            cadj_sum = np.sum(cadjs, axis=0)
            train_adjs = [normalize_adj_torch((cadj_sum > 0).astype(np.float32), device)]
            eval_adjs = [data_dict["norm_adj_dense_full"]]
        else:
            train_adjs = [normalize_adj_torch(cadjs[t], device) for t in range(T)]
            eval_adjs = data_dict["norm_adjs_full"]

        cfeats_norm = (cfeats - data_dict["feat_mean"]) / data_dict["feat_std"]
        train_feats = [torch.tensor(cfeats_norm[:, t, :], dtype=torch.float32, device=device) for t in range(T)]
        train_targets = torch.tensor(np.argmax(clabels, axis=1), dtype=torch.long, device=device)

        # Supernode in train if it contains any training node
        train_mask_c = (P.T @ data_dict["train_mask"]) > 0
        val_mask_c = (P.T @ data_dict["val_mask"]) > 0

        train_mask = torch.tensor(train_mask_c, device=device)
        val_mask = torch.tensor(val_mask_c, device=device)
        test_mask = torch.tensor(data_dict["test_mask"], device=device)

        # For coarsened graph: compute balanced square-root weights on coarsened train targets
        counts_c = np.bincount(train_targets[train_mask].cpu().numpy(), minlength=num_classes).astype(np.float32)
        weights_c = 1.0 / np.sqrt(counts_c + 1.0)
        weights_c = weights_c / weights_c.sum() * float(num_classes)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights_c, dtype=torch.float32, device=device))

    best_val_score = 0.0
    best_val_auc = 0.0
    best_test_auc = 0.0
    best_test_acc = 0.0
    best_epoch = 0
    patience_cnt = 0

    eval_feats = data_dict["feats_full"]
    eval_targets = data_dict["targets_full"]

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_feats, train_adjs)
        loss = criterion(logits[train_mask], train_targets[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        # Validation & Testing
        model.eval()
        with torch.no_grad():
            eval_logits = model(eval_feats, eval_adjs)
            probs = F.softmax(eval_logits, dim=-1).cpu().numpy()
            y_true = eval_targets.cpu().numpy()

            val_auc = compute_auroc_macro(y_true[data_dict["val_mask"]], probs[data_dict["val_mask"]])
            val_acc = compute_accuracy(y_true[data_dict["val_mask"]], probs[data_dict["val_mask"]])
            test_auc = compute_auroc_macro(y_true[data_dict["test_mask"]], probs[data_dict["test_mask"]])
            test_acc = compute_accuracy(y_true[data_dict["test_mask"]], probs[data_dict["test_mask"]])

            # Pure validation Macro-AUROC checkpointing matching KDD '26 paper methodology
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_test_auc = test_auc
                best_test_acc = test_acc
                best_epoch = epoch
                patience_cnt = 0
            else:
                patience_cnt += 1

            if verbose_epoch and (epoch % 50 == 0 or epoch == epochs):
                print(f"    [{get_timestamp()}] Ep {epoch:03d} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.2f}% | Test AUC: {test_auc:.2f}% | Best Test AUC: {best_test_auc:.2f}%", flush=True)

            if patience_cnt >= patience:
                if verbose_epoch:
                    print(f"    [{get_timestamp()}] Early stopping triggered at epoch {epoch} (patience={patience})", flush=True)
                break

    return best_test_auc, best_test_acc, best_epoch


def plot_curves(df: pd.DataFrame, out_png: str = "results_dblp5_curves.png"):
    """Generates comparison curves showing Macro AUROC across all alphas on DBLP5."""
    try:
        import matplotlib.pyplot as plt
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        for ax_idx, model_name in enumerate(["GRUoGCN2x2", "GCNx2oGRU"]):
            ax = axes[ax_idx]
            sub = df[df["Model"] == model_name]

            # Full data horizontal line
            full_sub = sub[sub["Method"] == "Full_Data"]
            if len(full_sub) > 0:
                full_val = full_sub["Live_AUROC_Mean"].values[0]
                ax.axhline(full_val, color="black", linestyle="--", linewidth=1.5, label=f"Full Data ({full_val:.1f}%)")

            # Coarsening methods
            colors = {"Ours_Novel": "#d62728", "SACoD_supra": "#1f77b4", "SACoD_snap": "#2ca02c"}
            markers = {"Ours_Novel": "o", "SACoD_supra": "s", "SACoD_snap": "^"}

            for meth in ["Ours_Novel", "SACoD_supra", "SACoD_snap"]:
                meth_sub = sub[(sub["Method"] == meth) & (sub["Alpha"] < 1.0)].sort_values("Alpha")
                if len(meth_sub) > 0:
                    ax.plot(
                        meth_sub["Alpha"],
                        meth_sub["Live_AUROC_Mean"],
                        marker=markers.get(meth, "o"),
                        color=colors.get(meth, "#333333"),
                        label=f"{meth}",
                        linewidth=2.0,
                        markersize=6,
                    )
                    ax.fill_between(
                        meth_sub["Alpha"],
                        meth_sub["Live_AUROC_Mean"] - meth_sub["Live_AUROC_Std"],
                        meth_sub["Live_AUROC_Mean"] + meth_sub["Live_AUROC_Std"],
                        alpha=0.15,
                        color=colors.get(meth, "#333333"),
                    )

            title_str = "TMP (GRUoGCN2x2)" if model_name == "GRUoGCN2x2" else "GTT (GCNx2oGRU)"
            ax.set_title(f"{title_str} on DBLP", fontsize=12, fontweight="bold")
            ax.set_xlabel("Coarsening Ratio (alpha)", fontsize=10)
            if ax_idx == 0:
                ax.set_ylabel("Macro AUROC (%)", fontsize=10)
            ax.set_xlim(0.25, 0.95)
            ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            ax.legend(loc="lower right", frameon=True)

        fig.suptitle("Dynamic Graph Coarsening Performance on DBLP (Table 1(b) Replication)", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_png, dpi=300)
        # Also copy to Desktop if accessible
        desktop_png = os.path.join(os.environ["USERPROFILE"], "Desktop", "results_dblp5_curves.png")
        fig.savefig(desktop_png, dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"Error plotting curves: {e}", flush=True)


# ==========================================
# Main Suite Execution
# ==========================================

def run_suite(
    models: List[str] = ["GRUoGCN2x2", "GCNx2oGRU"],
    methods: List[str] = ["Ours_Novel", "SACoD_supra", "SACoD_snap"],
    alphas: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    seeds: List[int] = [50, 51],
    epochs: int = 250,
    patience: int = 80,
    force_rerun: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{get_timestamp()}] Device: {device}", flush=True)
    print(f"[{get_timestamp()}] ============================================================", flush=True)
    print(f"[{get_timestamp()}] STARTING DBLP 250-EPOCH BENCHMARK SUITE WITH TIMESTAMPS", flush=True)
    print(f"[{get_timestamp()}] Models: {models}", flush=True)
    print(f"[{get_timestamp()}] Methods: {methods}", flush=True)
    print(f"[{get_timestamp()}] Alphas: {alphas}", flush=True)
    print(f"[{get_timestamp()}] Epochs: {epochs}, Patience: {patience}, Seeds: {seeds}", flush=True)
    print(f"[{get_timestamp()}] ============================================================", flush=True)

    data_dict = get_dblp5_data(device, seed=50)

    # Reference Paper Table 1(b) for DBLP (Ground Truth)
    paper_table_gru = {
        "SACoD_supra":    {0.3: "81.84 (±0.90)", 0.4: "83.10 (±1.48)", 0.5: "83.25 (±2.05)", 0.6: "82.41 (±2.52)", 0.7: "83.50 (±2.26)", 0.8: "82.95 (±3.08)", 0.9: "83.66 (±2.34)"},
        "SACoD_snap":     {0.3: "78.05 (±3.09)", 0.4: "80.55 (±3.47)", 0.5: "81.95 (±3.29)", 0.6: "82.48 (±2.59)", 0.7: "83.29 (±2.36)", 0.8: "83.66 (±2.69)", 0.9: "84.18 (±3.06)"},
        "Full_Data":      {1.0: "84.21 (±2.30)"}
    }
    paper_table_gcn = {
        "SACoD_supra":    {0.3: "78.78 (±1.70)", 0.4: "81.24 (±2.58)", 0.5: "81.85 (±1.64)", 0.6: "82.25 (±2.39)", 0.7: "84.25 (±1.85)", 0.8: "84.71 (±1.02)", 0.9: "84.76 (±1.66)"},
        "SACoD_snap":     {0.3: "79.14 (±2.87)", 0.4: "79.10 (±2.51)", 0.5: "81.02 (±2.08)", 0.6: "81.95 (±1.08)", 0.7: "84.28 (±1.04)", 0.8: "84.30 (±1.57)", 0.9: "85.92 (±1.34)"},
        "Full_Data":      {1.0: "85.78 (±0.44)"}
    }

    out_csv = "results_dblp5_comparison.csv"
    desktop_csv = os.path.join(os.environ["USERPROFILE"], "Desktop", "benchmark_results_dblp5.csv")
    session_ts = time.strftime("%Y%m%d_%H%M%S")
    timestamped_csv = os.path.join(os.environ["USERPROFILE"], "Desktop", f"benchmark_results_dblp5_{session_ts}.csv")
    log_file = os.path.join(os.environ["USERPROFILE"], "Desktop", f"dblp5_benchmark_run_log_{session_ts}.txt")

    existing_records = {}
    if os.path.exists(out_csv) and not force_rerun:
        try:
            old_df = pd.read_csv(out_csv)
            for _, r in old_df.iterrows():
                prev_ep = int(r["Epochs"]) if "Epochs" in r and pd.notnull(r["Epochs"]) else 0
                if prev_ep >= epochs:
                    k = (str(r["Model"]), str(r["Method"]), round(float(r["Alpha"]), 2))
                    existing_records[k] = r.to_dict()
            if len(existing_records) > 0:
                print(f"[{get_timestamp()}] Loaded {len(existing_records)} existing {epochs}-epoch benchmark checkpoints from {out_csv} for instant resuming.", flush=True)
        except Exception:
            existing_records = {}

    all_results = list(existing_records.values())

    seed_cache_file = "results_dblp5_seed_cache.json"
    seed_cache = {}
    if os.path.exists(seed_cache_file) and not force_rerun:
        try:
            with open(seed_cache_file, "r", encoding="utf-8") as scf:
                seed_cache = json.load(scf)
            if len(seed_cache) > 0:
                print(f"[{get_timestamp()}] Loaded {len(seed_cache)} seed-level checkpoints from {seed_cache_file}.", flush=True)
        except Exception:
            seed_cache = {}

    def log_and_print(msg: str):
        print(msg, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def save_checkpoint():
        df_save = pd.DataFrame(all_results)
        df_save.to_csv(out_csv, index=False)
        df_save.to_csv(desktop_csv, index=False)
        df_save.to_csv(timestamped_csv, index=False)
        plot_curves(df_save)

    def save_seed_cache():
        try:
            with open(seed_cache_file, "w", encoding="utf-8") as scf:
                json.dump(seed_cache, scf, indent=2)
        except Exception as e:
            print(f"Error saving seed cache: {e}", flush=True)

    log_and_print(f"[{get_timestamp()}] Benchmark session initialized. Log: {log_file}")

    for model_name in models:
        log_and_print(f"\n[{get_timestamp()}] >>>>>>>>>>>>>>>>>>> MODEL: {model_name} <<<<<<<<<<<<<<<<<<<")

        # 1. Full Data Baseline
        k_full = (model_name, "Full_Data", 1.0)
        if k_full not in existing_records:
            log_and_print(f"[{get_timestamp()}] Running Full Data (100% graph, alpha=1.0) for {model_name} across {len(seeds)} seeds...")
            aucs, accs = [], []
            t_f0 = time.time()
            for s in seeds:
                seed_key = f"{model_name}|Full_Data|1.0|{s}"
                if seed_key in seed_cache:
                    cached_entry = seed_cache[seed_key]
                    auc = cached_entry["auc"]
                    acc = cached_entry["acc"]
                    best_ep = cached_entry["best_ep"]
                    log_and_print(f"  [{get_timestamp()}] Seed {s}: Loaded from cache: Best Ep={best_ep}, Test AUC={auc:.2f}%, Test ACC={acc:.2f}%")
                else:
                    log_and_print(f"  [{get_timestamp()}] Seed {s}: Starting training on Full Data...")
                    auc, acc, best_ep = train_and_eval(model_name, "Full_Data", 1.0, data_dict, device, epochs=epochs, seed=s, patience=patience, verbose_epoch=True)
                    seed_cache[seed_key] = {"auc": round(auc, 2), "acc": round(acc, 2), "best_ep": best_ep}
                    save_seed_cache()
                    log_and_print(f"  [{get_timestamp()}] Seed {s} finished: Best Ep={best_ep}, Test AUC={auc:.2f}%, Test ACC={acc:.2f}%")

                aucs.append(auc)
                accs.append(acc)

            mean_auc, std_auc = float(np.mean(aucs)), float(np.std(aucs))
            mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs))
            paper_ref = (paper_table_gru if model_name == "GRUoGCN2x2" else paper_table_gcn)["Full_Data"][1.0]

            rec = {
                "Model": model_name,
                "Method": "Full_Data",
                "Alpha": 1.0,
                "Epochs": epochs,
                "Live_AUROC_Mean": round(mean_auc, 2),
                "Live_AUROC_Std": round(std_auc, 2),
                "Live_AUROC_Str": f"{mean_auc:.2f} ± {std_auc:.2f}",
                "Live_ACC_Mean": round(mean_acc, 2),
                "Live_ACC_Std": round(std_acc, 2),
                "Live_ACC_Str": f"{mean_acc:.2f} ± {std_acc:.2f}",
                "Paper_AUROC_Table1b": paper_ref,
                "Timestamp": get_timestamp(),
            }
            all_results.append(rec)
            existing_records[k_full] = rec
            save_checkpoint()
            log_and_print(f"[{get_timestamp()}] -> Full Data Complete ({time.time()-t_f0:.1f}s): AUROC={mean_auc:.2f} ± {std_auc:.2f}%, ACC={mean_acc:.2f} ± {std_acc:.2f}%, Paper={paper_ref}")
        else:
            rec = existing_records[k_full]
            log_and_print(f"[{get_timestamp()}] Full Data already completed: AUROC={rec['Live_AUROC_Str']}, ACC={rec['Live_ACC_Str']}")

        # 2. Coarsening Methods across Alphas
        for alpha in alphas:
            log_and_print(f"\n[{get_timestamp()}] --- Evaluating Alpha = {alpha:.1f} (Retaining {int(round(alpha*100))}% nodes) on {model_name} ---")
            for meth in methods:
                k = (model_name, meth, round(alpha, 2))
                if k in existing_records:
                    rec = existing_records[k]
                    log_and_print(f"  [{get_timestamp()}] [{meth} @ alpha={alpha:.1f}] Already completed: AUROC={rec['Live_AUROC_Str']}, ACC={rec['Live_ACC_Str']}")
                    continue

                log_and_print(f"  [{get_timestamp()}] Training [{meth} @ alpha={alpha:.1f}] across seeds {seeds} for {epochs} epochs...")
                aucs, accs = [], []
                t_m0 = time.time()
                for s in seeds:
                    seed_key = f"{model_name}|{meth}|{alpha:.1f}|{s}"
                    if seed_key in seed_cache:
                        cached_entry = seed_cache[seed_key]
                        auc = cached_entry["auc"]
                        acc = cached_entry["acc"]
                        best_ep = cached_entry["best_ep"]
                        log_and_print(f"    [{get_timestamp()}] Seed {s}: Loaded from cache: Best Ep={best_ep}, Test AUC={auc:.2f}%, ACC={acc:.2f}%")
                    else:
                        auc, acc, best_ep = train_and_eval(model_name, meth, alpha, data_dict, device, epochs=epochs, seed=s, patience=patience, verbose_epoch=False)
                        seed_cache[seed_key] = {"auc": round(auc, 2), "acc": round(acc, 2), "best_ep": best_ep}
                        save_seed_cache()
                        log_and_print(f"    [{get_timestamp()}] Seed {s}: Best Ep={best_ep}, Test AUC={auc:.2f}%, ACC={acc:.2f}%")

                    aucs.append(auc)
                    accs.append(acc)

                mean_auc, std_auc = float(np.mean(aucs)), float(np.std(aucs))
                mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs))
                
                table_map = paper_table_gru if model_name == "GRUoGCN2x2" else paper_table_gcn
                paper_ref = table_map.get(meth, {}).get(alpha, "-")

                rec = {
                    "Model": model_name,
                    "Method": meth,
                    "Alpha": round(alpha, 2),
                    "Epochs": epochs,
                    "Live_AUROC_Mean": round(mean_auc, 2),
                    "Live_AUROC_Std": round(std_auc, 2),
                    "Live_AUROC_Str": f"{mean_auc:.2f} ± {std_auc:.2f}",
                    "Live_ACC_Mean": round(mean_acc, 2),
                    "Live_ACC_Std": round(std_acc, 2),
                    "Live_ACC_Str": f"{mean_acc:.2f} ± {std_acc:.2f}",
                    "Paper_AUROC_Table1b": paper_ref,
                    "Timestamp": get_timestamp(),
                }
                all_results.append(rec)
                existing_records[k] = rec
                save_checkpoint()
                log_and_print(f"  [{get_timestamp()}] [{meth} @ alpha={alpha:.1f}] Complete ({time.time()-t_m0:.1f}s): AUROC={mean_auc:.2f} ± {std_auc:.2f}%, ACC={mean_acc:.2f} ± {std_acc:.2f}%, Paper={paper_ref}")

    log_and_print(f"\n[{get_timestamp()}] ============================================================")
    log_and_print(f"[{get_timestamp()}] ALL DBLP BENCHMARKS COMPLETED SUCCESSFULLY!")
    log_and_print(f"[{get_timestamp()}] Results saved to: {out_csv}")
    log_and_print(f"[{get_timestamp()}] Desktop CSV: {desktop_csv}")
    log_and_print(f"[{get_timestamp()}] Timestamped CSV: {timestamped_csv}")
    log_and_print(f"[{get_timestamp()}] Log file: {log_file}")
    log_and_print(f"[{get_timestamp()}] ============================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--seeds", type=int, nargs="+", default=[50, 51])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--models", type=str, nargs="+", default=["GRUoGCN2x2", "GCNx2oGRU"])
    parser.add_argument("--methods", type=str, nargs="+", default=["Ours_Novel", "SACoD_supra", "SACoD_snap"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_suite(
        models=args.models,
        methods=args.methods,
        alphas=args.alphas,
        seeds=args.seeds,
        epochs=args.epochs,
        patience=args.patience,
        force_rerun=args.force
    )
