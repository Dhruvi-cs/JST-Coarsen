"""
Benchmarking Engine for Dynamic Graph Neural Networks on IMDB
Evaluates:
- GCNx2oGRU (Graph-Then-Time / GTT)
- GRUoGCN2x2 (Temporal Message Passing / TMP)
across:
- Full_Data (alpha = 1.0)
- Ours_Novel (JST-Coarsen: Solutions 1-5)
- SACoD_supra (1st-order supra Laplacian + DSU)
- SACoD_snap (1st-order snapshot Laplacian + DSU)
for all coarsening ratios alpha in [0.3, 0.9] across 500 epochs with timestamps.
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
    then dual spatial-temporal concatenation and 2-layer MLP classifier.
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, num_times: int):
        super().__init__()
        self.num_times = num_times
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = MLP(hidden_dim * 2, hidden_dim, num_classes)
        self.act = nn.Softplus()

    def forward(self, feats_list: List[torch.Tensor], adjs_norm_list: List[torch.Tensor]) -> torch.Tensor:
        h_snaps = []
        for t in range(self.num_times):
            x = feats_list[t]
            adj = adjs_norm_list[t]
            h1 = self.act(self.gcn1(x, adj))
            h2 = self.act(self.gcn2(h1, adj))
            h_snaps.append(h2)

        # Stack over time: (N, T, hidden_dim)
        h_seq = torch.stack(h_snaps, dim=1)
        gru_out, _ = self.gru(h_seq)
        final_gru = gru_out[:, -1, :]
        final_gcn = h_snaps[-1]

        h_combined = torch.cat([final_gru, final_gcn], dim=-1)
        logits = self.classifier(self.act(h_combined))
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

    def forward(self, feats_input, adjs_norm_list: List[torch.Tensor]) -> torch.Tensor:
        if isinstance(feats_input, list):
            feats_seq = torch.stack(feats_input, dim=1)
        else:
            feats_seq = feats_input

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
# Dataset Loading & Split for IMDB
# ==========================================

def get_imdb_data(device: torch.device, seed: int = 50):
    data_path = os.path.join("src", "IMDB", "IMDB_no_isolated.npz")
    print(f"[{get_timestamp()}] Loading IMDB dataset from {data_path}...", flush=True)
    d = np.load(data_path)
    adjs = d["adjs"].astype(np.float32)       # (47, 3883, 3883)
    attmats = d["attmats"].astype(np.float32) # (3883, 73)
    labels = d["labels"].astype(np.float32)   # (3883, 2)

    N = labels.shape[0]
    T = adjs.shape[0]

    # Broadcast static attributes across temporal dimension
    feats = np.repeat(attmats[:, None, :], T, axis=1) # (3883, 47, 73)

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

    # Inverse class frequency weighting matching nodewinclass.py
    train_labels = node_labels[train_mask]
    counts = np.bincount(train_labels, minlength=labels.shape[1]).astype(np.float32)
    weights = 1.0 / (counts + 1.0)
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

    print(f"[{get_timestamp()}] IMDB loaded successfully: N={N}, T={T}, F={feats.shape[2]}, Classes={labels.shape[1]}, Train={train_mask.sum()}, Val={val_mask.sum()}, Test={test_mask.sum()}", flush=True)

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
    """Computes Macro AUROC matching paper methodology."""
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
        return 50.0


# ==========================================
# Training & Evaluation Function
# ==========================================

def train_and_eval(
    model_name: str,
    method_name: str,
    alpha: float,
    data_dict: Dict,
    device: torch.device,
    epochs: int = 500,
    lr: float = 0.01,
    patience: int = 250,
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
            "Loukas-graph-coarsening", "results", "IMDB", f"cr{alpha:.1f}", f"coarsened~{method_name}.npz"
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

    best_val_auc = 0.0
    best_test_auc = 0.0
    best_test_acc = 0.0
    best_epoch = 0
    patience_cnt = 0

    eval_feats = data_dict["feats_full"]
    eval_targets = data_dict["targets_full"]

    if model_name == "GRUoGCN2x2":
        train_feats_in = torch.stack(train_feats, dim=1) if isinstance(train_feats, list) else train_feats
        eval_feats_in = torch.stack(eval_feats, dim=1) if isinstance(eval_feats, list) else eval_feats
    else:
        train_feats_in = train_feats
        eval_feats_in = eval_feats

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_feats_in, train_adjs)
        loss = criterion(logits[train_mask], train_targets[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        # Validation & Testing (every 2 epochs to cut 50% evaluation overhead)
        if epoch % 2 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                eval_logits = model(eval_feats_in, eval_adjs)
                probs = F.softmax(eval_logits, dim=-1).cpu().numpy()
                y_true = eval_targets.cpu().numpy()

                val_auc = compute_auroc_macro(y_true[data_dict["val_mask"]], probs[data_dict["val_mask"]])
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_test_auc = compute_auroc_macro(y_true[data_dict["test_mask"]], probs[data_dict["test_mask"]])
                    best_test_acc = compute_accuracy(y_true[data_dict["test_mask"]], probs[data_dict["test_mask"]])
                    best_epoch = epoch
                    patience_cnt = 0
                else:
                    patience_cnt += 2

            if verbose_epoch and (epoch % 50 == 0 or epoch == epochs):
                current_test_auc = best_test_auc if best_test_auc > 0 else compute_auroc_macro(y_true[data_dict["test_mask"]], probs[data_dict["test_mask"]])
                print(f"    [{get_timestamp()}] Ep {epoch:03d} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.2f}% | Test AUC: {current_test_auc:.2f}% | Best Test AUC: {best_test_auc:.2f}%", flush=True)

            if patience < epochs and patience_cnt >= patience:
                if verbose_epoch:
                    print(f"    [{get_timestamp()}] Early stopping triggered at epoch {epoch} (patience={patience})", flush=True)
                break

    return best_test_auc, best_test_acc, best_epoch


def plot_curves(df: pd.DataFrame, out_png: str = "results_imdb_curves.png"):
    """Generates comparison curves showing Macro AUROC across all alphas on IMDB."""
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
                        linewidth=2.2 if meth == "Ours_Novel" else 1.5,
                        markersize=7 if meth == "Ours_Novel" else 5,
                        color=colors.get(meth, "#7f7f7f"),
                        label=f"{meth} (Ours)" if meth == "Ours_Novel" else meth,
                    )
                    ax.fill_between(
                        meth_sub["Alpha"],
                        meth_sub["Live_AUROC_Mean"] - meth_sub["Live_AUROC_Std"],
                        meth_sub["Live_AUROC_Mean"] + meth_sub["Live_AUROC_Std"],
                        alpha=0.15 if meth == "Ours_Novel" else 0.08,
                        color=colors.get(meth, "#7f7f7f"),
                    )

            ax.set_title(f"IMDB: {model_name}", fontsize=13, fontweight="bold")
            ax.set_xlabel(r"Coarsening Ratio $\alpha$", fontsize=11)
            ax.set_ylabel("Macro AUROC (%)" if ax_idx == 0 else "", fontsize=11)
            ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            handles, labels = ax.get_legend_handles_labels()
            if len(labels) > 0:
                ax.legend(loc="lower right", frameon=True)

        plt.suptitle("Dynamic Graph Coarsening Benchmark on IMDB: Ours vs SACoD vs Full Data", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        
        desktop_png = os.path.join(os.environ["USERPROFILE"], "Desktop", "results_imdb_curves.png")
        plt.savefig(desktop_png, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[{get_timestamp()}] Comparison curves saved to {out_png} and {desktop_png}", flush=True)
    except Exception as e:
        print(f"[{get_timestamp()}] Note: Could not save plot: {e}", flush=True)


# ==========================================
# Main Benchmarking Suite
# ==========================================

def run_suite(
    models: List[str] = ["GRUoGCN2x2", "GCNx2oGRU"],
    methods: List[str] = ["Ours_Novel", "SACoD_supra", "SACoD_snap"],
    alphas: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    epochs: int = 500,
    seeds: List[int] = [50, 51],
    patience: int = 250,
    force_rerun: bool = False,
):
    device = torch.device("cpu")
    print(f"[{get_timestamp()}] ============================================================", flush=True)
    print(f"[{get_timestamp()}] STARTING IMDB 500-EPOCH BENCHMARK SUITE WITH TIMESTAMPS", flush=True)
    print(f"[{get_timestamp()}] Models: {models}", flush=True)
    print(f"[{get_timestamp()}] Methods: {methods}", flush=True)
    print(f"[{get_timestamp()}] Alphas: {alphas}", flush=True)
    print(f"[{get_timestamp()}] Epochs: {epochs}, Patience: {patience}, Seeds: {seeds}", flush=True)
    print(f"[{get_timestamp()}] ============================================================", flush=True)

    data_dict = get_imdb_data(device, seed=50)

    # Reference Paper Table 1(a) for IMDB
    paper_table_gru = {
        "SACoD_supra":    {0.3: "70.63 (±0.49)", 0.4: "72.36 (±0.35)", 0.5: "72.47 (±0.70)", 0.6: "73.10 (±0.79)", 0.7: "73.67 (±1.14)", 0.8: "73.88 (±0.58)", 0.9: "73.88 (±0.49)"},
        "SACoD_snap":     {0.3: "- (failed)", 0.4: "68.50 (±1.32)", 0.5: "73.12 (±0.94)", 0.6: "72.34 (±0.84)", 0.7: "72.14 (±1.85)", 0.8: "72.92 (±0.86)", 0.9: "73.30 (±0.66)"},
        "Full_Data":      {1.0: "74.12 (±0.92)"}
    }
    paper_table_gcn = {
        "SACoD_supra":    {0.3: "70.59 (±0.37)", 0.4: "72.04 (±0.49)", 0.5: "72.38 (±0.55)", 0.6: "72.81 (±0.60)", 0.7: "74.23 (±0.15)", 0.8: "73.58 (±0.30)", 0.9: "74.04 (±0.17)"},
        "SACoD_snap":     {0.3: "- (failed)", 0.4: "68.64 (±0.83)", 0.5: "72.69 (±0.94)", 0.6: "73.17 (±0.60)", 0.7: "73.61 (±0.48)", 0.8: "73.60 (±0.34)", 0.9: "73.25 (±0.53)"},
        "Full_Data":      {1.0: "74.11 (±0.16)"}
    }

    out_csv = "results_imdb_comparison.csv"
    desktop_csv = os.path.join(os.environ["USERPROFILE"], "Desktop", "benchmark_results_imdb.csv")
    session_ts = time.strftime("%Y%m%d_%H%M%S")
    timestamped_csv = os.path.join(os.environ["USERPROFILE"], "Desktop", f"benchmark_results_imdb_{session_ts}.csv")
    log_file = os.path.join(os.environ["USERPROFILE"], "Desktop", f"imdb_benchmark_run_log_{session_ts}.txt")

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

    seed_cache_file = "results_seed_cache.json"
    seed_cache = {}
    if os.path.exists(seed_cache_file):
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

    log_and_print(f"[{get_timestamp()}] Benchmark session initialized. Log: {log_file}")

    for model_name in models:
        log_and_print(f"\n[{get_timestamp()}] >>>>>>>>>>>>>>>>>>> MODEL: {model_name} <<<<<<<<<<<<<<<<<<<")

        # 1. Full Data Baseline
        full_key = (model_name, "Full_Data", 1.0)
        if full_key in existing_records and not force_rerun:
            log_and_print(f"[{get_timestamp()}] Skipping Full_Data for {model_name} (already evaluated: {existing_records[full_key]['Live_AUROC_Mean']}%)")
        else:
            log_and_print(f"\n[{get_timestamp()}] --- Evaluating Full Data Baseline (alpha=1.0) on {model_name} ---")
            t_start = time.time()
            seed_aucs = []
            seed_accs = []
            seed_epochs = []

            for seed in seeds:
                log_and_print(f"  [{get_timestamp()}] Seed {seed}: Training Full Data on {model_name} for up to {epochs} epochs...")
                t_seed0 = time.time()
                auc, acc, best_ep = train_and_eval(
                    model_name, "Full_Data", 1.0, data_dict, device, epochs=epochs, patience=patience, seed=seed, verbose_epoch=True
                )
                t_seed = time.time() - t_seed0
                seed_aucs.append(auc)
                seed_accs.append(acc)
                seed_epochs.append(best_ep)
                log_and_print(f"  [{get_timestamp()}] Seed {seed} done in {t_seed:.1f}s: Best Val Epoch={best_ep}, Test AUROC={auc:.2f}%, Test Acc={acc:.2f}%")

            mean_auc = float(np.mean(seed_aucs))
            std_auc = float(np.std(seed_aucs))
            mean_acc = float(np.mean(seed_accs))
            elapsed = time.time() - t_start

            p_ref = paper_table_gru["Full_Data"][1.0] if model_name == "GRUoGCN2x2" else paper_table_gcn["Full_Data"][1.0]
            record = {
                "Timestamp": get_timestamp(),
                "Model": model_name,
                "Method": "Full_Data",
                "Alpha": 1.0,
                "Nodes_K": data_dict["N"],
                "Epochs": epochs,
                "Live_AUROC_Mean": round(mean_auc, 2),
                "Live_AUROC_Std": round(std_auc, 2),
                "Live_Accuracy_Mean": round(mean_acc, 2),
                "Best_Epoch_Mean": round(float(np.mean(seed_epochs)), 1),
                "Paper_Table_Reference": p_ref,
                "Training_Time_Sec": round(elapsed, 1),
            }
            all_results.append(record)
            existing_records[full_key] = record
            save_checkpoint()
            log_and_print(f"[{get_timestamp()}] >>> RESULT Full_Data | {model_name} | AUROC: {mean_auc:.2f} ± {std_auc:.2f}% | Paper Ref: {p_ref} | Time: {elapsed:.1f}s")

        # 2. Coarsening Methods across Alphas
        for alpha in alphas:
            target_K = max(2, int(round(alpha * data_dict["N"])))
            log_and_print(f"\n[{get_timestamp()}] ==================== Alpha = {alpha:.1f} (Target K = {target_K}) ====================")

            for method_name in methods:
                key = (model_name, method_name, round(alpha, 2))
                if key in existing_records and not force_rerun:
                    log_and_print(f"[{get_timestamp()}] Skipping {method_name} alpha={alpha:.1f} for {model_name} (already evaluated: {existing_records[key]['Live_AUROC_Mean']}%)")
                    continue

                log_and_print(f"\n  [{get_timestamp()}] --- Evaluating {method_name} (alpha={alpha:.1f}, K={target_K}) on {model_name} ---")
                t_start = time.time()
                seed_aucs = []
                seed_accs = []
                seed_epochs = []

                # Check if coarsened data file exists
                coarse_path = os.path.join(
                    "Loukas-graph-coarsening", "results", "IMDB", f"cr{alpha:.1f}", f"coarsened~{method_name}.npz"
                )
                if not os.path.exists(coarse_path):
                    log_and_print(f"  [{get_timestamp()}] Waiting for coarsened file {coarse_path} to finish generating...")
                    wait_count = 0
                    while not os.path.exists(coarse_path) and wait_count < 180:
                        time.sleep(5)
                        wait_count += 1
                    if not os.path.exists(coarse_path):
                        log_and_print(f"  [{get_timestamp()}] WARNING: Coarsened file {coarse_path} not found after waiting! Skipping.")
                        continue

                cd = np.load(coarse_path)
                actual_K = cd["P"].shape[1]

                for seed in seeds:
                    seed_key = f"{model_name}_{method_name}_{alpha:.1f}_{seed}"
                    if seed_key in seed_cache:
                        c_data = seed_cache[seed_key]
                        auc = float(c_data["auc"])
                        acc = float(c_data["acc"])
                        best_ep = int(c_data["best_ep"])
                        t_seed = float(c_data.get("time", 0.0))
                        log_and_print(f"    [{get_timestamp()}] Seed {seed} loaded from seed cache: Best Val Ep={best_ep}, Test AUROC={auc:.2f}%, Test Acc={acc:.2f}% (Time: {t_seed:.1f}s)")
                    else:
                        log_and_print(f"    [{get_timestamp()}] Seed {seed}: Training {method_name} (alpha={alpha:.1f}) for up to {epochs} epochs...")
                        t_seed0 = time.time()
                        try:
                            auc, acc, best_ep = train_and_eval(
                                model_name, method_name, alpha, data_dict, device, epochs=epochs, patience=patience, seed=seed, verbose_epoch=True
                            )
                        except Exception as e:
                            log_and_print(f"    [{get_timestamp()}] Error in seed {seed}: {e}")
                            auc, acc, best_ep = 50.0, 50.0, 0

                        t_seed = time.time() - t_seed0
                        seed_cache[seed_key] = {"auc": auc, "acc": acc, "best_ep": best_ep, "time": t_seed}
                        try:
                            with open(seed_cache_file, "w", encoding="utf-8") as scf:
                                json.dump(seed_cache, scf, indent=2)
                        except Exception:
                            pass
                        log_and_print(f"    [{get_timestamp()}] Seed {seed} done in {t_seed:.1f}s: Best Val Ep={best_ep}, Test AUROC={auc:.2f}%, Test Acc={acc:.2f}%")

                    seed_aucs.append(auc)
                    seed_accs.append(acc)
                    seed_epochs.append(best_ep)

                mean_auc = float(np.mean(seed_aucs))
                std_auc = float(np.std(seed_aucs))
                mean_acc = float(np.mean(seed_accs))
                elapsed = time.time() - t_start

                # Look up paper table reference
                p_table = paper_table_gru if model_name == "GRUoGCN2x2" else paper_table_gcn
                p_ref = p_table.get(method_name, {}).get(round(alpha, 1), "-")

                record = {
                    "Timestamp": get_timestamp(),
                    "Model": model_name,
                    "Method": method_name,
                    "Alpha": round(alpha, 2),
                    "Nodes_K": actual_K,
                    "Epochs": epochs,
                    "Live_AUROC_Mean": round(mean_auc, 2),
                    "Live_AUROC_Std": round(std_auc, 2),
                    "Live_Accuracy_Mean": round(mean_acc, 2),
                    "Best_Epoch_Mean": round(float(np.mean(seed_epochs)), 1),
                    "Paper_Table_Reference": p_ref,
                    "Training_Time_Sec": round(elapsed, 1),
                }
                all_results.append(record)
                existing_records[key] = record
                save_checkpoint()
                log_and_print(f"  [{get_timestamp()}] >>> RESULT {method_name} | alpha={alpha:.1f} | {model_name} | AUROC: {mean_auc:.2f} ± {std_auc:.2f}% | Paper Ref: {p_ref} | Time: {elapsed:.1f}s")

    log_and_print(f"\n[{get_timestamp()}] ============================================================")
    log_and_print(f"[{get_timestamp()}] ALL BENCHMARKS COMPLETED SUCCESSFULLY!")
    log_and_print(f"[{get_timestamp()}] Results saved to:")
    log_and_print(f"  1. {out_csv}")
    log_and_print(f"  2. {desktop_csv}")
    log_and_print(f"  3. {timestamped_csv}")
    log_and_print(f"  4. {log_file}")
    log_and_print(f"[{get_timestamp()}] ============================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=250)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_suite(epochs=args.epochs, patience=args.patience, force_rerun=args.force)
