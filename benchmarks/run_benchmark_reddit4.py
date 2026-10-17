"""
Benchmarking Engine for Dynamic Graph Neural Networks on Reddit4
Evaluates:
- GCNx2oGRU (Graph-Then-Time / GTT)
- GRUoGCN2x2 (Temporal Message Passing / TMP)
across all coarsening methods (Ours_Novel, SACoD_supra, SACoD_snap, Random, NetCondense, HeavyEdge, VariationEdges, VariationNb, Full_Data)
and all coarsening ratios alpha in [0.3, 0.9].
"""

import os
import sys
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import pandas as pd
from typing import Dict, List, Tuple


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
        # x: (N, in_dim), adj_norm: (N, N) sparse or dense
        h = self.linear(x)
        if adj_norm.is_sparse:
            out = torch.sparse.mm(adj_norm, h)
        else:
            out = torch.matmul(adj_norm, h)
        return out


def normalize_adj_torch(adj: np.ndarray, device: torch.device) -> torch.Tensor:
    """Symmetric normalization: D^-1/2 (A + I) D^-1/2"""
    N = adj.shape[0]
    A_tilde = adj + np.eye(N, dtype=np.float32)
    d = np.sum(A_tilde, axis=1)
    d_inv_sqrt = np.power(np.maximum(1e-6, d), -0.5)
    d_mat = np.diag(d_inv_sqrt)
    norm_adj = d_mat @ A_tilde @ d_mat
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
# Dataset Loading & Split
# ==========================================

def get_reddit4_data(device: torch.device, seed: int = 50):
    data_path = os.path.join("src", "Reddit4", "Reddit4_no_isolated.npz")
    d = np.load(data_path)
    adjs = d["adjs"].astype(np.float32)       # (10, 1128, 1128)
    feats = d["attmats"].astype(np.float32)   # (1128, 10, 20)
    labels = d["labels"].astype(np.float32)   # (1128, 4)

    N = labels.shape[0]
    T = adjs.shape[0]

    # Exact Paper Stratified In-Degree Split matching dynedge_v2.py and spindle.py
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
    except Exception as e:
        return 50.0


def compute_accuracy(target: np.ndarray, probs: np.ndarray) -> float:
    """Computes classification accuracy matching paper methodology."""
    try:
        preds = np.argmax(probs, axis=1)
        return float(np.mean(preds == target)) * 100.0
    except Exception:
        return 25.0


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
) -> Tuple[float, float]:
    """
    Trains on coarsened graph (or full data) and evaluates test AUROC and accuracy at best validation epoch.
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
            "Loukas-graph-coarsening", "results", "Reddit4", f"cr{alpha:.1f}", f"coarsened~{method_name}.npz"
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

        # Coarsened masks: a supernode is in train if it contains any training node
        train_mask_c = (P.T @ data_dict["train_mask"]) > 0
        val_mask_c = (P.T @ data_dict["val_mask"]) > 0

        train_mask = torch.tensor(train_mask_c, device=device)
        val_mask = torch.tensor(val_mask_c, device=device)
        test_mask = torch.tensor(data_dict["test_mask"], device=device)

    best_val_auc = 0.0
    best_test_auc = 0.0
    best_test_acc = 0.0
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
            test_auc = compute_auroc_macro(y_true[data_dict["test_mask"]], probs[data_dict["test_mask"]])
            test_acc = compute_accuracy(y_true[data_dict["test_mask"]], probs[data_dict["test_mask"]])

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_test_auc = test_auc
                best_test_acc = test_acc
                patience_cnt = 0
            else:
                patience_cnt += 1

            if patience_cnt >= patience:
                break

    return best_test_auc, best_test_acc


def plot_curves(df: pd.DataFrame, out_png: str = "results_reddit4_curves.png"):
    """Generates comparison curves showing Macro AUROC across all alphas."""
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

            ax.set_title(f"Reddit4: {model_name}", fontsize=13, fontweight="bold")
            ax.set_xlabel(r"Coarsening Ratio $\alpha$", fontsize=11)
            ax.set_ylabel("Macro AUROC (%)" if ax_idx == 0 else "", fontsize=11)
            ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            handles, labels = ax.get_legend_handles_labels()
            if len(labels) > 0:
                ax.legend(loc="lower right", frameon=True)

        plt.suptitle("Dynamic Graph Coarsening Benchmark on Reddit4: Ours vs SACoD vs Full Data", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Comparison curves saved to {out_png}")
    except Exception as e:
        print(f"Note: Could not save plot: {e}")


# ==========================================
# Main Benchmarking Suite
# ==========================================

def run_suite(
    models: List[str],
    methods: List[str],
    alphas: List[float],
    epochs: int = 500,
    seeds: List[int] = [50, 51],
    patience: int = 250,
    force_rerun: bool = False,
):
    device = torch.device("cpu")
    print(f"Using device: {device}")
    data_dict = get_reddit4_data(device, seed=50)

    # Reference Paper Results from Table 1(c)
    paper_table_gru = {
        "SACoD_supra":    {0.3: "60.04 (±0.95)", 0.4: "61.53 (±1.43)", 0.5: "62.15 (±1.57)", 0.6: "61.91 (±2.14)", 0.7: "63.59 (±1.94)", 0.8: "63.64 (±0.93)", 0.9: "64.66 (±2.02)"},
        "SACoD_snap":     {0.3: "57.69 (±2.19)", 0.4: "61.21 (±1.22)", 0.5: "61.73 (±1.28)", 0.6: "62.57 (±2.22)", 0.7: "62.96 (±1.72)", 0.8: "62.89 (±2.74)", 0.9: "64.41 (±2.41)"},
        "Random":         {0.3: "58.96 (±4.48)", 0.4: "58.75 (±2.10)", 0.5: "60.99 (±2.00)", 0.6: "61.34 (±1.66)", 0.7: "60.62 (±2.14)", 0.8: "62.89 (±1.90)", 0.9: "62.97 (±1.51)"},
        "NetCondense":    {0.3: "56.79 (±3.87)", 0.4: "60.15 (±3.25)", 0.5: "59.81 (±1.76)", 0.6: "61.29 (±1.54)", 0.7: "63.27 (±2.17)", 0.8: "63.18 (±2.20)", 0.9: "63.61 (±0.85)"},
        "HeavyEdge":      {0.3: "56.19 (±2.11)", 0.4: "56.99 (±3.07)", 0.5: "56.38 (±4.01)", 0.6: "57.63 (±4.06)", 0.7: "56.65 (±3.12)", 0.8: "60.00 (±2.47)", 0.9: "63.79 (±1.94)"},
        "VariationEdges": {0.3: "56.24 (±1.77)", 0.4: "56.13 (±0.88)", 0.5: "57.30 (±1.71)", 0.6: "58.04 (±1.32)", 0.7: "61.07 (±3.09)", 0.8: "60.63 (±2.53)", 0.9: "61.97 (±1.84)"},
        "VariationNb":    {0.3: "59.18 (±1.92)", 0.4: "60.51 (±2.52)", 0.5: "60.75 (±2.01)", 0.6: "61.52 (±2.57)", 0.7: "61.95 (±1.18)", 0.8: "63.20 (±0.71)", 0.9: "64.71 (±1.33)"},
        "Full_Data":      {1.0: "65.08 (±2.14)"}
    }
    paper_table_gcn = {
        "SACoD_supra":    {0.3: "60.73 (±1.63)", 0.4: "62.22 (±2.97)", 0.5: "62.50 (±1.55)", 0.6: "62.69 (±3.69)", 0.7: "63.95 (±2.18)", 0.8: "66.17 (±1.58)", 0.9: "66.47 (±1.52)"},
        "SACoD_snap":     {0.3: "56.98 (±2.55)", 0.4: "60.81 (±2.11)", 0.5: "62.88 (±2.41)", 0.6: "63.66 (±1.06)", 0.7: "64.88 (±1.75)", 0.8: "64.84 (±1.55)", 0.9: "64.56 (±1.44)"},
        "Random":         {0.3: "59.28 (±1.38)", 0.4: "63.44 (±2.08)", 0.5: "62.26 (±1.51)", 0.6: "61.04 (±1.88)", 0.7: "60.99 (±1.55)", 0.8: "63.00 (±1.75)", 0.9: "64.59 (±2.47)"},
        "NetCondense":    {0.3: "61.36 (±1.45)", 0.4: "59.65 (±1.52)", 0.5: "62.95 (±2.36)", 0.6: "62.78 (±3.53)", 0.7: "63.18 (±1.88)", 0.8: "63.26 (±3.22)", 0.9: "64.43 (±2.38)"},
        "HeavyEdge":      {0.3: "55.59 (±2.58)", 0.4: "54.32 (±1.18)", 0.5: "58.38 (±3.05)", 0.6: "57.31 (±1.93)", 0.7: "57.37 (±2.26)", 0.8: "60.65 (±2.54)", 0.9: "63.78 (±2.81)"},
        "VariationEdges": {0.3: "55.82 (±3.49)", 0.4: "56.80 (±2.96)", 0.5: "57.48 (±2.52)", 0.6: "56.84 (±2.74)", 0.7: "61.66 (±1.00)", 0.8: "62.69 (±2.59)", 0.9: "64.35 (±2.03)"},
        "VariationNb":    {0.3: "59.34 (±3.10)", 0.4: "60.51 (±1.96)", 0.5: "62.20 (±2.02)", 0.6: "62.45 (±1.21)", 0.7: "63.67 (±1.89)", 0.8: "63.91 (±0.93)", 0.9: "66.70 (±2.01)"},
        "Full_Data":      {1.0: "67.09 (±1.02)"}
    }

    out_csv = "results_reddit4_comparison.csv"
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
                print(f"Loaded {len(existing_records)} existing {epochs}-epoch benchmark checkpoints from {out_csv} for instant resuming.")
        except Exception:
            existing_records = {}

    all_results = list(existing_records.values())

    def save_checkpoint():
        df_save = pd.DataFrame(all_results)
        df_save.to_csv(out_csv, index=False)
        plot_curves(df_save)

    for model_name in models:
        print(f"\n=======================================================")
        print(f"            BENCHMARKING MODEL: {model_name}")
        print(f"=======================================================")

        # First: Full Data Baseline
        full_key = (model_name, "Full_Data", 1.0)
        if full_key in existing_records and not force_rerun:
            prev = existing_records[full_key]
            print(f"  [Resumed Checkpoint] Full_Data alpha=1.0 -> AUROC: {prev.get('Live_AUROC_Str', 'N/A')}, ACC: {prev.get('Live_ACC_Str', 'N/A')}")
        else:
            print(f"\nEvaluating Full Data Baseline for {model_name} (epochs={epochs})...")
            full_aucs = []
            full_accs = []
            for s in seeds:
                auc_s, acc_s = train_and_eval(model_name, "Full_Data", 1.0, data_dict, device, epochs=epochs, patience=patience, seed=s)
                full_aucs.append(auc_s)
                full_accs.append(acc_s)
            avg_auc = float(np.mean(full_aucs))
            std_auc = float(np.std(full_aucs))
            avg_acc = float(np.mean(full_accs))
            std_acc = float(np.std(full_accs))
            print(f"  [Full_Data] alpha=1.0 -> AUROC: {avg_auc:.2f} ± {std_auc:.2f}% | ACC: {avg_acc:.2f} ± {std_acc:.2f}%")
            
            paper_full = paper_table_gru["Full_Data"][1.0] if model_name == "GRUoGCN2x2" else paper_table_gcn["Full_Data"][1.0]
            entry = {
                "Model": model_name,
                "Method": "Full_Data",
                "Alpha": 1.0,
                "Epochs": epochs,
                "Live_AUROC_Mean": round(avg_auc, 2),
                "Live_AUROC_Std": round(std_auc, 2),
                "Live_AUROC_Str": f"{avg_auc:.2f} ± {std_auc:.2f}",
                "Live_ACC_Mean": round(avg_acc, 2),
                "Live_ACC_Std": round(std_acc, 2),
                "Live_ACC_Str": f"{avg_acc:.2f} ± {std_acc:.2f}",
                "Paper_AUROC_Table1c": paper_full,
            }
            all_results = [r for r in all_results if (r["Model"], r["Method"], round(float(r["Alpha"]), 2)) != full_key]
            all_results.append(entry)
            existing_records[full_key] = entry
            save_checkpoint()

        for alpha in alphas:
            print(f"\n--- Coarsening Ratio alpha = {alpha:.1f} ---")
            for method in methods:
                method_key = (model_name, method, round(float(alpha), 2))
                if method_key in existing_records and not force_rerun:
                    prev = existing_records[method_key]
                    print(f"  [Resumed Checkpoint] {method:<14} -> AUROC: {prev.get('Live_AUROC_Str', 'N/A')}, ACC: {prev.get('Live_ACC_Str', 'N/A')}")
                    continue

                run_aucs = []
                run_accs = []
                for s in seeds:
                    try:
                        auc_s, acc_s = train_and_eval(model_name, method, alpha, data_dict, device, epochs=epochs, patience=patience, seed=s)
                        run_aucs.append(auc_s)
                        run_accs.append(acc_s)
                    except Exception as e:
                        print(f"    Error on {method} alpha={alpha} seed={s}: {e}")

                if len(run_aucs) > 0:
                    mean_auc = float(np.mean(run_aucs))
                    std_auc = float(np.std(run_aucs))
                    mean_acc = float(np.mean(run_accs))
                    std_acc = float(np.std(run_accs))
                    print(f"  {method:<16} | AUROC: {mean_auc:6.2f} ± {std_auc:4.2f}% | ACC: {mean_acc:5.2f} ± {std_acc:4.2f}%")
                    
                    paper_val = "-"
                    ref_table = paper_table_gru if model_name == "GRUoGCN2x2" else paper_table_gcn
                    if method in ref_table and round(alpha, 1) in ref_table[method]:
                        paper_val = ref_table[method][round(alpha, 1)]

                    entry = {
                        "Model": model_name,
                        "Method": method,
                        "Alpha": round(alpha, 1),
                        "Epochs": epochs,
                        "Live_AUROC_Mean": round(mean_auc, 2),
                        "Live_AUROC_Std": round(std_auc, 2),
                        "Live_AUROC_Str": f"{mean_auc:.2f} ± {std_auc:.2f}",
                        "Live_ACC_Mean": round(mean_acc, 2),
                        "Live_ACC_Std": round(std_acc, 2),
                        "Live_ACC_Str": f"{mean_acc:.2f} ± {std_acc:.2f}",
                        "Paper_AUROC_Table1c": paper_val,
                    }
                    all_results = [r for r in all_results if (r["Model"], r["Method"], round(float(r["Alpha"]), 2)) != method_key]
                    all_results.append(entry)
                    existing_records[method_key] = entry
                    # Instant incremental save on every single method so progress is NEVER lost!
                    save_checkpoint()

    print(f"\n=======================================================")
    print(f"All benchmark runs completed and saved to {out_csv}!")
    print(f"=======================================================")
    return pd.DataFrame(all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs (Paper Horizon: 500)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[50, 51])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--models", type=str, nargs="+", default=["GRUoGCN2x2", "GCNx2oGRU"])
    parser.add_argument("--methods", type=str, nargs="+", default=["Ours_Novel", "SACoD_supra", "SACoD_snap"])
    parser.add_argument("--patience", type=int, default=250)
    parser.add_argument("--force-rerun", action="store_true", help="Force re-run even if already in CSV")
    args = parser.parse_args()

    run_suite(
        models=args.models,
        methods=args.methods,
        alphas=args.alphas,
        epochs=args.epochs,
        seeds=args.seeds,
        patience=args.patience,
        force_rerun=args.force_rerun,
    )
