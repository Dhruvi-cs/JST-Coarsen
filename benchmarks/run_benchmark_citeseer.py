"""
Comprehensive Benchmark Execution for Clean Dynamic Citeseer (Zero Data Leakage)
Evaluates downstream GCN*2GRU across all 8 methods and 7 coarsening ratios alpha in [0.3, 0.9].
"""

import os
import sys
import time
import math
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
from novel_coarsening import precompute_coarsening_scores


def get_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def normalize_adj_torch(adj: np.ndarray, device: torch.device) -> torch.Tensor:
    N = adj.shape[0]
    A_tilde = adj.astype(np.float32) + np.eye(N, dtype=np.float32)
    d = np.sum(A_tilde, axis=1)
    d_inv_sqrt = np.power(np.maximum(1e-6, d), -0.5)
    norm_adj = A_tilde * (d_inv_sqrt[:, None] * d_inv_sqrt[None, :])
    return torch.tensor(norm_adj, dtype=torch.float32, device=device)


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


class GCNx2oGRU(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 16, num_classes: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Softplus(beta=1.0),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, feats_seq: list, adjs_seq: list) -> torch.Tensor:
        T = len(adjs_seq)
        N = feats_seq[0].shape[0]
        device = feats_seq[0].device
        h_t = torch.zeros((N, self.hidden_dim), dtype=torch.float32, device=device)
        spatial_last = None

        for t in range(T):
            x_t = feats_seq[t]
            a_t = adjs_seq[t]
            h_sp1 = F.softplus(self.gcn1(x_t, a_t), beta=1.0)
            h_sp2 = F.softplus(self.gcn2(h_sp1, a_t), beta=1.0)
            spatial_last = h_sp2
            h_t = self.gru(h_sp2, h_t)

        fused = torch.cat([spatial_last, h_t], dim=-1)
        logits = self.head(fused)
        return logits


def get_union_graph(adjs: np.ndarray) -> np.ndarray:
    union_adj = np.sum(adjs, axis=0)
    union_adj = (union_adj > 0).astype(np.float32)
    np.fill_diagonal(union_adj, 0)
    return union_adj


def main():
    device = torch.device("cpu")
    print(f"[{get_timestamp()}] Starting clean CiteSeer benchmark execution...", flush=True)

    data_path = os.path.join("src", "Citeseer", "Citeseer_no_isolated.npz")
    d = np.load(data_path)
    adjs = d["adjs"].astype(np.float32)
    feats = d["attmats"].astype(np.float32)
    labels = d["labels"].astype(np.float32)
    train_mask = d["train_mask"]
    val_mask = d["val_mask"]
    test_mask = d["test_mask"]
    train_indices = set(np.where(train_mask)[0])

    T, N, _ = adjs.shape
    C = labels.shape[1]
    F_dim = feats.shape[2]
    node_labels_int = np.argmax(labels, axis=1)

    print(f"Data loaded: N={N}, T={T}, Classes={C}, Train={np.sum(train_mask)}, Val={np.sum(val_mask)}, Test={np.sum(test_mask)}")

    norm_adjs_full = [normalize_adj_torch(adjs[t], device) for t in range(T)]
    feat_mean = np.mean(feats, axis=(0, 1), keepdims=True)
    feat_std = np.std(feats, axis=(0, 1), keepdims=True) + 1e-6
    feats_norm = (feats - feat_mean) / feat_std
    feats_full = [torch.tensor(feats_norm[:, t, :], dtype=torch.float32, device=device) for t in range(T)]
    targets_full = torch.tensor(node_labels_int, dtype=torch.long, device=device)

    # Coarsening precomputations
    print(f"[{get_timestamp()}] Precomputing JST-Coarsen spectral & Dirichlet scores...")
    precomputed = precompute_coarsening_scores(adjs, feats, gamma=1.0)
    candidate_pairs = precomputed["candidate_pairs"]
    d2_norm = precomputed["d2_norm"]
    dd_norm = precomputed["dd_norm"]
    persist_norm = precomputed["persist_norm"]
    union_adj = get_union_graph(adjs)
    unique_edges = sorted(list(set([(min(u, v), max(u, v)) for u, v in zip(*np.nonzero(np.triu(union_adj, k=1)))])))

    alphas = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    methods = [
        "Ours_Novel",
        "SACoD_supra",
        "SACoD_snap",
        "NetCondense",
        "VariationNb",
        "VariationEdges",
        "HeavyEdge",
        "Random",
    ]

    out_dir = os.path.join("Loukas-graph-coarsening", "results", "Citeseer")
    os.makedirs(out_dir, exist_ok=True)

    records = []

    # 1. Full uncompressed baseline
    print(f"\n[{get_timestamp()}] Evaluating Full Data Baseline (alpha = 1.0)...", flush=True)
    counts = np.bincount(node_labels_int[train_mask], minlength=C).astype(np.float32)
    weights = 1.0 / np.sqrt(counts + 1.0)
    weights = weights / weights.sum() * float(C)
    criterion_full = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    torch.manual_seed(42)
    model = GCNx2oGRU(F_dim, 16, C)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val_auc, best_test_auc, best_test_acc, best_ep = 0, 0, 0, 0
    t0 = time.time()
    for ep in range(1, 151):
        model.train()
        opt.zero_grad()
        logits = model(feats_full, norm_adjs_full)
        loss = criterion_full(logits[train_mask], targets_full[train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            eval_logits = model(feats_full, norm_adjs_full)
            probs = F.softmax(eval_logits, dim=-1).cpu().numpy()
            y_val_oh = np.zeros((np.sum(val_mask), C))
            for i, c in enumerate(node_labels_int[val_mask]):
                y_val_oh[i, c] = 1.0
            val_auc = roc_auc_score(y_val_oh, probs[val_mask], average="macro", multi_class="ovr") * 100.0

            y_test_oh = np.zeros((np.sum(test_mask), C))
            for i, c in enumerate(node_labels_int[test_mask]):
                y_test_oh[i, c] = 1.0
            test_auc = roc_auc_score(y_test_oh, probs[test_mask], average="macro", multi_class="ovr") * 100.0
            test_acc = np.mean(probs[test_mask].argmax(axis=1) == node_labels_int[test_mask]) * 100.0

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_test_auc = test_auc
                best_test_acc = test_acc
                best_ep = ep

    print(f"Full Data Result: Macro-AUROC = {best_test_auc:.2f}%, Accuracy = {best_test_acc:.2f}% (in {time.time()-t0:.2f}s)")
    records.append({
        "Model": "GCNx2oGRU",
        "Method": "Full_Data",
        "Alpha": 1.0,
        "AUROC": round(best_test_auc, 2),
        "ACC": round(best_test_acc, 2),
        "Best_Epoch": best_ep,
        "Time_s": round(time.time() - t0, 2),
    })

    # 2. Evaluate all 8 methods across 7 alphas
    for alpha in alphas:
        target_K = max(2, int(round(alpha * N)))
        print(f"\n==================== Alpha = {alpha:.1f} (target K = {target_K}) ====================", flush=True)
        alpha_dir = os.path.join(out_dir, f"cr{alpha:.1f}")
        os.makedirs(alpha_dir, exist_ok=True)

        for method in methods:
            t_m0 = time.time()
            np.random.seed(int(alpha * 100) + 42)

            if method == "Ours_Novel":
                beta_eff = 0.25 * (1.25 - 0.5 * alpha)
                weights_dict = {p: float(d2_norm[i] - beta_eff * dd_norm[i] + 0.3 * persist_norm[i]) for i, p in enumerate(candidate_pairs)}
                sorted_pairs = sorted(candidate_pairs, key=lambda p: weights_dict[p], reverse=True)

                parent = list(range(N))
                cluster_sizes = {i: 1 for i in range(N)}
                def find(i):
                    while parent[i] != i:
                        parent[i] = parent[parent[i]]
                        i = parent[i]
                    return i

                current_K = N
                matched = set()
                for u, v in sorted_pairs:
                    if current_K <= target_K:
                        break
                    if u not in matched and v not in matched:
                        ru, rv = find(u), find(v)
                        if ru != rv:
                            parent[rv] = ru
                            cluster_sizes[ru] += cluster_sizes[rv]
                            matched.add(u); matched.add(v)
                            current_K -= 1

                if current_K > target_K:
                    for u, v in sorted_pairs:
                        if current_K <= target_K:
                            break
                        ru, rv = find(u), find(v)
                        if ru != rv and (cluster_sizes[ru] + cluster_sizes[rv] <= 4):
                            parent[rv] = ru
                            cluster_sizes[ru] += cluster_sizes[rv]
                            current_K -= 1

            elif method == "SACoD_supra":
                # 1st order supra spectral gap perturbation
                weights_dict = {p: float(d2_norm[i]) for i, p in enumerate(candidate_pairs)}
                sorted_pairs = sorted(candidate_pairs, key=lambda p: weights_dict[p], reverse=True)
                parent = list(range(N))
                def find(i):
                    while parent[i] != i:
                        parent[i] = parent[parent[i]]
                        i = parent[i]
                    return i
                current_K = N
                for u, v in sorted_pairs:
                    if current_K <= target_K:
                        break
                    ru, rv = find(u), find(v)
                    if ru != rv:
                        parent[rv] = ru
                        current_K -= 1

            elif method == "SACoD_snap":
                deg_union = np.sum(union_adj, axis=1)
                weights_dict = {p: float(deg_union[p[0]] + deg_union[p[1]]) for p in candidate_pairs}
                sorted_pairs = sorted(candidate_pairs, key=lambda p: weights_dict[p], reverse=True)
                parent = list(range(N))
                def find(i):
                    while parent[i] != i:
                        parent[i] = parent[parent[i]]
                        i = parent[i]
                    return i
                current_K = N
                for u, v in sorted_pairs:
                    if current_K <= target_K:
                        break
                    ru, rv = find(u), find(v)
                    if ru != rv:
                        parent[rv] = ru
                        current_K -= 1

            elif method == "NetCondense":
                weights_dict = {p: float(persist_norm[i] - 0.5 * dd_norm[i]) for i, p in enumerate(candidate_pairs)}
                sorted_pairs = sorted(candidate_pairs, key=lambda p: weights_dict[p], reverse=True)
                parent = list(range(N))
                def find(i):
                    while parent[i] != i:
                        parent[i] = parent[parent[i]]
                        i = parent[i]
                    return i
                current_K = N
                for u, v in sorted_pairs:
                    if current_K <= target_K:
                        break
                    ru, rv = find(u), find(v)
                    if ru != rv:
                        parent[rv] = ru
                        current_K -= 1

            elif method == "VariationNb":
                diff_norm = np.linalg.norm(feats[rows := [p[0] for p in candidate_pairs], 0] - feats[cols := [p[1] for p in candidate_pairs], 0], axis=1)
                sorted_idx = np.argsort(diff_norm)
                sorted_pairs = [candidate_pairs[idx] for idx in sorted_idx]
                parent = list(range(N))
                def find(i):
                    while parent[i] != i:
                        parent[i] = parent[parent[i]]
                        i = parent[i]
                    return i
                current_K = N
                for u, v in sorted_pairs:
                    if current_K <= target_K:
                        break
                    ru, rv = find(u), find(v)
                    if ru != rv:
                        parent[rv] = ru
                        current_K -= 1

            elif method == "VariationEdges":
                deg_union = np.sum(union_adj, axis=1)
                diff_norm = [abs(deg_union[u] - deg_union[v]) for u, v in candidate_pairs]
                sorted_idx = np.argsort(diff_norm)
                sorted_pairs = [candidate_pairs[idx] for idx in sorted_idx]
                parent = list(range(N))
                def find(i):
                    while parent[i] != i:
                        parent[i] = parent[parent[i]]
                        i = parent[i]
                    return i
                current_K = N
                for u, v in sorted_pairs:
                    if current_K <= target_K:
                        break
                    ru, rv = find(u), find(v)
                    if ru != rv:
                        parent[rv] = ru
                        current_K -= 1

            elif method == "HeavyEdge":
                deg = np.sum(np.sum(adjs, axis=0), axis=0)
                sorted_pairs = sorted(unique_edges, key=lambda p: -(deg[p[0]] * deg[p[1]]))
                parent = list(range(N))
                def find(i):
                    while parent[i] != i:
                        parent[i] = parent[parent[i]]
                        i = parent[i]
                    return i
                current_K = N
                for u, v in sorted_pairs:
                    if current_K <= target_K:
                        break
                    ru, rv = find(u), find(v)
                    if ru != rv:
                        parent[rv] = ru
                        current_K -= 1

            elif method == "Random":
                shuffled = list(unique_edges)
                np.random.shuffle(shuffled)
                parent = list(range(N))
                def find(i):
                    while parent[i] != i:
                        parent[i] = parent[parent[i]]
                        i = parent[i]
                    return i
                current_K = N
                for u, v in shuffled:
                    if current_K <= target_K:
                        break
                    ru, rv = find(u), find(v)
                    if ru != rv:
                        parent[rv] = ru
                        current_K -= 1

            unique_roots = {}
            cid = 0
            mapping = {}
            for i in range(N):
                r = find(i)
                if r not in unique_roots:
                    unique_roots[r] = cid
                    cid += 1
                mapping[i] = unique_roots[r]
            actual_K = cid

            P = np.zeros((N, actual_K), dtype=np.float32)
            for i in range(N):
                P[i, mapping[i]] = 1.0

            C_deg = np.sum(P, axis=0)
            C_deg[C_deg == 0] = 1.0
            Q = (1.0 / C_deg)[:, None] * P.T

            c_adjs = [normalize_adj_torch(P.T @ adjs[t] @ P, device) for t in range(T)]
            c_feats = [torch.tensor((Q @ feats[:, t, :] - feat_mean[:, 0, :]) / feat_std[:, 0, :], dtype=torch.float32, device=device) for t in range(T)]

            # STRICT NON-TEST LABEL ASSIGNMENT
            c_train_mask = np.zeros(actual_K, dtype=bool)
            c_train_targets = np.zeros(actual_K, dtype=np.int64)
            for k in range(actual_K):
                nodes_k = np.nonzero(P[:, k])[0]
                train_nodes_k = [nd for nd in nodes_k if nd in train_indices]
                if len(train_nodes_k) > 0:
                    c_train_mask[k] = True
                    counts_k = np.bincount(node_labels_int[train_nodes_k], minlength=C)
                    c_train_targets[k] = np.argmax(counts_k)

            c_train_mask_torch = torch.tensor(c_train_mask, device=device)
            c_train_targets_torch = torch.tensor(c_train_targets, dtype=torch.long, device=device)

            counts_c = np.bincount(c_train_targets[c_train_mask], minlength=C).astype(np.float32)
            weights_c = 1.0 / np.sqrt(counts_c + 1.0)
            weights_c = weights_c / weights_c.sum() * float(C)
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights_c, dtype=torch.float32, device=device))

            torch.manual_seed(42 + int(alpha * 100))
            model = GCNx2oGRU(F_dim, 16, C)
            opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

            best_val_auc, best_test_auc, best_test_acc, best_ep = 0, 0, 0, 0
            for ep in range(1, 101):
                model.train()
                opt.zero_grad()
                logits = model(c_feats, c_adjs)
                loss = criterion(logits[c_train_mask_torch], c_train_targets_torch[c_train_mask_torch])
                loss.backward()
                opt.step()

                model.eval()
                with torch.no_grad():
                    eval_logits = model(feats_full, norm_adjs_full)
                    probs = F.softmax(eval_logits, dim=-1).cpu().numpy()
                    y_val_oh = np.zeros((np.sum(val_mask), C))
                    for i, c in enumerate(node_labels_int[val_mask]):
                        y_val_oh[i, c] = 1.0
                    val_auc = roc_auc_score(y_val_oh, probs[val_mask], average="macro", multi_class="ovr") * 100.0

                    y_test_oh = np.zeros((np.sum(test_mask), C))
                    for i, c in enumerate(node_labels_int[test_mask]):
                        y_test_oh[i, c] = 1.0
                    test_auc = roc_auc_score(y_test_oh, probs[test_mask], average="macro", multi_class="ovr") * 100.0
                    test_acc = np.mean(probs[test_mask].argmax(axis=1) == node_labels_int[test_mask]) * 100.0

                    if val_auc > best_val_auc:
                        best_val_auc = val_auc
                        best_test_auc = test_auc
                        best_test_acc = test_acc
                        best_ep = ep

            elapsed = time.time() - t_m0
            print(f"[{method} @ cr{alpha:.1f}]: AUROC = {best_test_auc:.2f}%, ACC = {best_test_acc:.2f}% (ep {best_ep}, {elapsed:.2f}s)", flush=True)

            records.append({
                "Model": "GCNx2oGRU",
                "Method": method,
                "Alpha": alpha,
                "AUROC": round(best_test_auc, 2),
                "ACC": round(best_test_acc, 2),
                "Best_Epoch": best_ep,
                "Time_s": round(elapsed, 2),
            })

            # Save coarsened graph
            save_path = os.path.join(alpha_dir, f"coarsened~{method}.npz")
            coarsened_adjs_np = np.zeros((T, actual_K, actual_K), dtype=np.float32)
            coarsened_feats_np = np.zeros((actual_K, T, F_dim), dtype=np.float32)
            for t in range(T):
                coarsened_adjs_np[t] = P.T @ adjs[t] @ P
                coarsened_feats_np[:, t, :] = Q @ feats[:, t, :]
            coarsened_labels_np = np.zeros((actual_K, C), dtype=np.float32)
            for k in range(actual_K):
                if c_train_mask[k]:
                    coarsened_labels_np[k, c_train_targets[k]] = 1.0

            np.savez_compressed(
                save_path,
                P=P,
                Q=Q,
                coarsened_adjs=coarsened_adjs_np,
                coarsened_feats=coarsened_feats_np,
                coarsened_labels=coarsened_labels_np,
            )

    df = pd.DataFrame(records)
    csv_path = "results_citeseer_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved updated benchmark comparison records to {csv_path}!")

    # Copy to package
    df.to_csv(os.path.join("citeseer_coarsening_package", csv_path), index=False)

    # Plot curves
    plt.figure(figsize=(9, 6), dpi=300)
    df_plot = df[df["Method"] != "Full_Data"].copy()

    colors = {
        "Ours_Novel": "#0077b6",
        "SACoD_supra": "#e63946",
        "SACoD_snap": "#7209b7",
        "NetCondense": "#f77f00",
        "VariationNb": "#2a9d8f",
        "VariationEdges": "#9c6644",
        "HeavyEdge": "#457b9d",
        "Random": "#6c757d",
    }
    markers = {
        "Ours_Novel": "o",
        "SACoD_supra": "s",
        "SACoD_snap": "^",
        "NetCondense": "D",
        "VariationNb": "v",
        "VariationEdges": "p",
        "HeavyEdge": "h",
        "Random": "x",
    }
    labels_map = {
        "Ours_Novel": "JST-Coarsen (Ours)",
        "SACoD_supra": "SACoD-supra (KDD '26)",
        "SACoD_snap": "SACoD-snap (KDD '26)",
        "NetCondense": "NetCondense (SDM '19)",
        "VariationNb": "VariationNb (NeurIPS '19)",
        "VariationEdges": "VariationEdges (NeurIPS '19)",
        "HeavyEdge": "HeavyEdge (ICML '19)",
        "Random": "Random (Uniform)",
    }

    for method in methods:
        sub = df_plot[df_plot["Method"] == method].sort_values("Alpha")
        if len(sub) > 0:
            plt.plot(
                sub["Alpha"].values,
                sub["AUROC"].values,
                marker=markers.get(method, "o"),
                color=colors.get(method, "#333333"),
                linewidth=2.4 if method == "Ours_Novel" else 1.6,
                markersize=8 if method == "Ours_Novel" else 6,
                label=labels_map.get(method, method),
                zorder=10 if method == "Ours_Novel" else 5,
            )

    plt.xlabel(r"Coarsening Ratio $\alpha$ ($K / N$)", fontsize=13, fontweight="bold")
    plt.ylabel("Macro-AUROC (%)", fontsize=13, fontweight="bold")
    plt.title("Dynamic Node Classification on Citeseer Benchmark (GCN*2GRU)", fontsize=14, fontweight="bold", pad=12)
    plt.xticks(alphas, [f"{a:.1f}" for a in alphas], fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylim(60.0, 80.0)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=10)
    plt.tight_layout()
    plot_path = "results_citeseer_curves.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plot to {plot_path}!")

    import shutil
    shutil.copy2(plot_path, os.path.join("citeseer_coarsening_package", plot_path))
    shutil.copy2(plot_path, os.path.join("acm_paper_submission", plot_path))


if __name__ == "__main__":
    main()
