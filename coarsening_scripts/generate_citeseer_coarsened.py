"""
Unified Dynamic Graph Coarsening Generator for Citeseer
Generates coarsened graph partitions across all ratios alpha in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] for:
1. Ours_Novel (JST-Coarsen: Solutions 1-5, hierarchical independent matching)
2. SACoD_supra (1st-order supra Laplacian + DSU)
3. SACoD_snap (1st-order snapshot Laplacian + DSU)
4. NetCondense (temporal condensation)
5. VariationNb (neighborhood variation)
6. VariationEdges (spectral edge variation)
7. HeavyEdge (heavy edge degree matching)
8. Random (random edge selection)
"""

import os
import sys
import math
import time
import argparse
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import networkx as nx
from typing import Dict, List, Tuple
from novel_coarsening import precompute_coarsening_scores


def get_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def get_union_graph(adjs: np.ndarray) -> np.ndarray:
    union_adj = np.sum(adjs, axis=0)
    union_adj = (union_adj > 0).astype(np.float32)
    np.fill_diagonal(union_adj, 0)
    return union_adj


def coarsen_jst_novel_citeseer(
    adjs: np.ndarray,
    feats: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    precomputed: Dict,
) -> Dict:
    """
    JST-Coarsen: Solutions 1-5 with Capacitated Hierarchical Independent Matching
    (deg_M(u) <= 1 per pass) to eliminate snowball star collapse.
    """
    T, N, _ = adjs.shape
    target_K = max(2, int(round(alpha * N)))
    candidate_pairs = precomputed["candidate_pairs"]
    d2_norm = precomputed["d2_norm"]
    dd_norm = precomputed["dd_norm"]
    persist_norm = precomputed["persist_norm"]

    # Ratio-adaptive Dirichlet & persistence schedule
    beta_eff = 0.25 * (1.25 - 0.5 * alpha)
    rho_eff = 0.3
    weights_dict = {}
    for i, p in enumerate(candidate_pairs):
        weights_dict[p] = float(d2_norm[i] - beta_eff * dd_norm[i] + rho_eff * persist_norm[i])

    sorted_pairs = sorted(candidate_pairs, key=lambda p: weights_dict[p], reverse=True)

    parent = list(range(N))
    cluster_sizes = {i: 1 for i in range(N)}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    current_K = N

    # Pass 1: Strict independent matching (deg_M(u) <= 1, cluster size <= 2)
    matched = set()
    for u, v in sorted_pairs:
        if current_K <= target_K:
            break
        if u not in matched and v not in matched:
            ru = find(u)
            rv = find(v)
            if ru != rv:
                parent[rv] = ru
                cluster_sizes[ru] += cluster_sizes[rv]
                matched.add(u)
                matched.add(v)
                current_K -= 1

    # Pass 2: Merge bounded supernodes (cluster size <= 4)
    if current_K > target_K:
        matched_pass2 = set()
        for u, v in sorted_pairs:
            if current_K <= target_K:
                break
            ru = find(u)
            rv = find(v)
            if ru != rv and (cluster_sizes[ru] + cluster_sizes[rv] <= 4):
                if ru not in matched_pass2 and rv not in matched_pass2:
                    parent[rv] = ru
                    cluster_sizes[ru] += cluster_sizes[rv]
                    matched_pass2.add(ru)
                    matched_pass2.add(rv)
                    current_K -= 1

    # Pass 3: Finish remainder if needed (cluster size <= 6)
    if current_K > target_K:
        for u, v in sorted_pairs:
            if current_K <= target_K:
                break
            ru = find(u)
            rv = find(v)
            if ru != rv and (cluster_sizes[ru] + cluster_sizes[rv] <= 6):
                parent[rv] = ru
                cluster_sizes[ru] += cluster_sizes[rv]
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

    c_adjs = np.zeros((T, actual_K, actual_K), dtype=np.float32)
    F_dim = feats.shape[2]
    c_feats = np.zeros((actual_K, T, F_dim), dtype=np.float32)
    c_labels = np.zeros((actual_K, labels.shape[1]), dtype=np.float32)

    for t in range(T):
        c_adjs[t] = P.T @ adjs[t] @ P
        c_feats[:, t, :] = Q @ feats[:, t, :]
    for k in range(actual_K):
        nodes = np.nonzero(P[:, k])[0]
        if len(nodes) > 0:
            c_labels[k, np.argmax(np.sum(labels[nodes], axis=0))] = 1.0

    return {
        "P": P,
        "Q": Q,
        "coarsened_adjs": c_adjs,
        "coarsened_feats": c_feats,
        "coarsened_labels": c_labels,
    }


def coarsen_sacod_supra(adjs: np.ndarray, alpha: float) -> np.ndarray:
    """SACoD-Supra: 1st-order Supra-Laplacian spectral gap perturbation with greedy DSU."""
    T, N, _ = adjs.shape
    target_K = max(2, int(round(alpha * N)))

    blocks = [[None for _ in range(T)] for _ in range(T)]
    I_N = sp.identity(N, format="csr", dtype=np.float32)

    for t in range(T):
        A_t = sp.csr_matrix(adjs[t], dtype=np.float32)
        d_t = np.array(A_t.sum(axis=1)).flatten()
        L_t = sp.diags(d_t) - A_t
        if t == 0 or t == T - 1:
            L_block = L_t + I_N
        else:
            L_block = L_t + 2.0 * I_N
        blocks[t][t] = L_block
        if t < T - 1:
            blocks[t][t + 1] = -I_N
            blocks[t + 1][t] = -I_N

    L_supra = sp.bmat(blocks, format="csr")

    try:
        vals, vecs = eigsh(L_supra, k=6, sigma=0.05, which='LM', tol=1e-3)
        vals_real = np.real(vals)
        pos_idx = np.where(vals_real > 1e-4)[0]
        if len(pos_idx) > 0:
            fiedler_idx = pos_idx[np.argmin(vals_real[pos_idx])]
            lambda_2 = float(vals_real[fiedler_idx])
            v_supra = np.real(vecs[:, fiedler_idx])
        else:
            lambda_2 = float(np.max(vals_real))
            v_supra = np.real(vecs[:, -1])
    except Exception:
        vals, vecs = eigsh(L_supra, k=4, which='SM', tol=1e-3)
        vals_real = np.real(vals)
        pos_idx = np.where(vals_real > 1e-4)[0]
        fiedler_idx = pos_idx[0] if len(pos_idx) > 0 else 1
        lambda_2 = float(vals_real[fiedler_idx])
        v_supra = np.real(vecs[:, fiedler_idx])

    V_t = v_supra.reshape(T, N)
    union_adj = get_union_graph(adjs)
    rows, cols = np.nonzero(np.triu(union_adj, k=1))

    intra_terms = np.zeros(len(rows), dtype=np.float32)
    for t in range(T):
        v_diff = V_t[t, rows] - V_t[t, cols]
        w_ab = adjs[t, rows, cols]
        d_a = np.sum(adjs[t, rows, :], axis=1)
        d_b = np.sum(adjs[t, cols, :], axis=1)
        k_val = (d_a + d_b + 2.0 * w_ab) / 4.0
        intra_terms += (v_diff ** 2) * (k_val - lambda_2)

    inter_terms = np.zeros(len(rows), dtype=np.float32)
    for t in range(T - 1):
        v_drift = (V_t[t + 1, rows] - V_t[t + 1, cols]) - (V_t[t, rows] - V_t[t, cols])
        inter_terms -= 0.5 * (v_drift ** 2)

    scores = intra_terms + inter_terms
    sorted_idx = np.argsort(-scores)

    parent = list(range(N))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    current_supernodes = N
    for idx in sorted_idx:
        u = rows[idx]
        v = cols[idx]
        ru = find(u)
        rv = find(v)
        if ru != rv:
            parent[rv] = ru
            current_supernodes -= 1
            if current_supernodes == target_K:
                break

    unique_roots = {}
    cid = 0
    mapping = {}
    for i in range(N):
        r = find(i)
        if r not in unique_roots:
            unique_roots[r] = cid
            cid += 1
        mapping[i] = unique_roots[r]

    P = np.zeros((N, cid), dtype=np.float32)
    for i in range(N):
        P[i, mapping[i]] = 1.0
    return P


def coarsen_sacod_snap(adjs: np.ndarray, alpha: float) -> np.ndarray:
    """SACoD-Snap: 1st-order Snapshot Laplacian perturbation with greedy DSU."""
    T, N, _ = adjs.shape
    target_K = max(2, int(round(alpha * N)))
    scores_sum = None
    union_adj = get_union_graph(adjs)
    rows, cols = np.nonzero(np.triu(union_adj, k=1))

    for t in range(T):
        A_t = sp.csr_matrix(adjs[t], dtype=np.float32)
        d_t = np.array(A_t.sum(axis=1)).flatten()
        L_t = sp.diags(d_t) - A_t
        try:
            vals, vecs = eigsh(L_t, k=4, sigma=0.01, which='LM', tol=1e-3)
            vals_real = np.real(vals)
            pos_idx = np.where(vals_real > 1e-4)[0]
            fiedler_idx = pos_idx[0] if len(pos_idx) > 0 else 1
            lambda_2 = float(vals_real[fiedler_idx])
            v_t = np.real(vecs[:, fiedler_idx])
        except Exception:
            vals, vecs = eigsh(L_t, k=2, which='SM', tol=1e-3)
            vals_real = np.real(vals)
            pos_idx = np.where(vals_real > 1e-4)[0]
            fiedler_idx = pos_idx[0] if len(pos_idx) > 0 else 1
            lambda_2 = float(vals_real[fiedler_idx])
            v_t = np.real(vecs[:, fiedler_idx])

        v_diff = v_t[rows] - v_t[cols]
        w_ab = adjs[t, rows, cols]
        d_a = np.sum(adjs[t, rows, :], axis=1)
        d_b = np.sum(adjs[t, cols, :], axis=1)
        k_val = (d_a + d_b + 2.0 * w_ab) / 4.0
        score_t = (v_diff ** 2) * (k_val - lambda_2)

        if scores_sum is None:
            scores_sum = score_t
        else:
            scores_sum += score_t

    sorted_idx = np.argsort(-scores_sum)

    parent = list(range(N))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    current_supernodes = N
    for idx in sorted_idx:
        u = rows[idx]
        v = cols[idx]
        ru = find(u)
        rv = find(v)
        if ru != rv:
            parent[rv] = ru
            current_supernodes -= 1
            if current_supernodes == target_K:
                break

    unique_roots = {}
    cid = 0
    mapping = {}
    for i in range(N):
        r = find(i)
        if r not in unique_roots:
            unique_roots[r] = cid
            cid += 1
        mapping[i] = unique_roots[r]

    P = np.zeros((N, cid), dtype=np.float32)
    for i in range(N):
        P[i, mapping[i]] = 1.0
    return P


def coarsen_netcondense_python(adjs: np.ndarray, alpha: float) -> np.ndarray:
    """NetCondense baseline implemented directly in Python matching getNodeScores.m."""
    T, N, _ = adjs.shape
    target_K = max(2, int(round(alpha * N)))

    I_N = sp.identity(N, format="csr", dtype=np.float32)
    blocks = [[None for _ in range(T)] for _ in range(T)]
    for t in range(T):
        A_t = sp.csr_matrix(adjs[t], dtype=np.float32)
        next_t = 0 if t == T - 1 else t + 1
        blocks[t][next_t] = A_t + I_N

    fG = sp.bmat(blocks, format="csr")
    flatM = (fG + fG.T) * 0.5

    try:
        vals, vecs = eigsh(flatM, k=1, which='LM', tol=1e-3, maxiter=500)
        u = np.abs(np.real(vecs[:, 0]))
        lambda_val = float(np.abs(np.real(vals[0])))
    except Exception:
        u = np.ones(N * T, dtype=np.float32) / np.sqrt(N * T)
        lambda_val = 1.0

    uv_sum = float(np.sum(u ** 2))
    U_nodes = u.reshape(T, N).T

    union_adj = get_union_graph(adjs)
    rows, cols = np.nonzero(np.triu(union_adj, k=1))

    ui = U_nodes[rows, :]
    uj = U_nodes[cols, :]
    first_term = np.sum(ui ** 2 + uj ** 2, axis=1)

    deg_i = np.array([np.sum(adjs[:, rows[e], :]) for e in range(len(rows))])
    deg_j = np.array([np.sum(adjs[:, cols[e], :]) for e in range(len(cols))])
    degs_sum = deg_i + deg_j

    scores = 2.0 * lambda_val * first_term + degs_sum * (first_term / uv_sum)
    order = np.argsort(scores)

    parent = list(range(N))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    current_supernodes = N
    for idx in order:
        u_node = rows[idx]
        v_node = cols[idx]
        ru = find(u_node)
        rv = find(v_node)
        if ru != rv:
            parent[rv] = ru
            current_supernodes -= 1
            if current_supernodes == target_K:
                break

    unique_roots = {}
    cid = 0
    mapping = {}
    for i in range(N):
        r = find(i)
        if r not in unique_roots:
            unique_roots[r] = cid
            cid += 1
        mapping[i] = unique_roots[r]

    P = np.zeros((N, cid), dtype=np.float32)
    for i in range(N):
        P[i, mapping[i]] = 1.0
    return P


def coarsen_variation_neighborhoods(adjs: np.ndarray, alpha: float) -> np.ndarray:
    """VariationNb baseline: merges neighborhoods with highest structural similarity."""
    T, N, _ = adjs.shape
    target_K = max(2, int(round(alpha * N)))

    W_union = np.sum(adjs, axis=0, dtype=np.float32)
    np.fill_diagonal(W_union, 0.0)

    A_sp = sp.csr_matrix(W_union)
    A_sq = (A_sp @ A_sp).tocsr()
    degs = np.sum(W_union, axis=1)

    rows, cols = np.nonzero(np.triu(W_union > 0, k=1))
    common_nb = np.array(A_sq[rows, cols]).flatten()
    union_nb = degs[rows] + degs[cols] - common_nb
    jaccard = common_nb / np.maximum(1.0, union_nb)

    sorted_idx = np.argsort(-jaccard)
    parent = list(range(N))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    current_supernodes = N
    for idx in sorted_idx:
        u = rows[idx]
        v = cols[idx]
        ru = find(u)
        rv = find(v)
        if ru != rv:
            parent[rv] = ru
            current_supernodes -= 1
            if current_supernodes == target_K:
                break

    unique_roots = {}
    cid = 0
    mapping = {}
    for i in range(N):
        r = find(i)
        if r not in unique_roots:
            unique_roots[r] = cid
            cid += 1
        mapping[i] = unique_roots[r]

    P = np.zeros((N, cid), dtype=np.float32)
    for i in range(N):
        P[i, mapping[i]] = 1.0
    return P


def coarsen_variation_edges(adjs: np.ndarray, alpha: float) -> np.ndarray:
    """VariationEdges baseline: Loukas edge-based spectral variation minimization."""
    T, N, _ = adjs.shape
    target_K = max(2, int(round(alpha * N)))

    W_union = np.sum(adjs, axis=0, dtype=np.float32)
    np.fill_diagonal(W_union, 0.0)

    rows, cols = np.nonzero(np.triu(W_union > 0, k=1))
    degs = np.sum(W_union, axis=1)

    var_scores = np.zeros(len(rows), dtype=np.float32)
    for i in range(len(rows)):
        u = rows[i]
        v = cols[i]
        w_uv = W_union[u, v]
        d_u = degs[u]
        d_v = degs[v]
        var_scores[i] = (1.0 / max(1e-5, d_u) + 1.0 / max(1e-5, d_v) - 2.0 * w_uv / max(1e-5, d_u * d_v))

    sorted_idx = np.argsort(var_scores)

    parent = list(range(N))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    current_supernodes = N
    for idx in sorted_idx:
        u = rows[idx]
        v = cols[idx]
        ru = find(u)
        rv = find(v)
        if ru != rv:
            parent[rv] = ru
            current_supernodes -= 1
            if current_supernodes == target_K:
                break

    unique_roots = {}
    cid = 0
    mapping = {}
    for i in range(N):
        r = find(i)
        if r not in unique_roots:
            unique_roots[r] = cid
            cid += 1
        mapping[i] = unique_roots[r]

    P = np.zeros((N, cid), dtype=np.float32)
    for i in range(N):
        P[i, mapping[i]] = 1.0
    return P


def coarsen_heavy_edge(adjs: np.ndarray, alpha: float) -> np.ndarray:
    """HeavyEdge baseline: contracts edges with heaviest temporal weight."""
    T, N, _ = adjs.shape
    target_K = max(2, int(round(alpha * N)))

    W_union = np.sum(adjs, axis=0, dtype=np.float32)
    np.fill_diagonal(W_union, 0.0)

    rows, cols = np.nonzero(np.triu(W_union > 0, k=1))
    weights = W_union[rows, cols]
    sorted_idx = np.argsort(-weights)

    parent = list(range(N))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    current_supernodes = N
    for idx in sorted_idx:
        u = rows[idx]
        v = cols[idx]
        ru = find(u)
        rv = find(v)
        if ru != rv:
            parent[rv] = ru
            current_supernodes -= 1
            if current_supernodes == target_K:
                break

    unique_roots = {}
    cid = 0
    mapping = {}
    for i in range(N):
        r = find(i)
        if r not in unique_roots:
            unique_roots[r] = cid
            cid += 1
        mapping[i] = unique_roots[r]

    P = np.zeros((N, cid), dtype=np.float32)
    for i in range(N):
        P[i, mapping[i]] = 1.0
    return P


def coarsen_random(adjs: np.ndarray, alpha: float, seed: int = 42) -> np.ndarray:
    """Random baseline: uniformly randomly contracts candidate edges in union graph."""
    np.random.seed(seed)
    T, N, _ = adjs.shape
    target_K = max(2, int(round(alpha * N)))

    union_adj = get_union_graph(adjs)
    rows, cols = np.nonzero(np.triu(union_adj, k=1))
    perm = np.random.permutation(len(rows))

    parent = list(range(N))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    current_supernodes = N
    for idx in perm:
        u = rows[idx]
        v = cols[idx]
        ru = find(u)
        rv = find(v)
        if ru != rv:
            parent[rv] = ru
            current_supernodes -= 1
            if current_supernodes == target_K:
                break

    unique_roots = {}
    cid = 0
    mapping = {}
    for i in range(N):
        r = find(i)
        if r not in unique_roots:
            unique_roots[r] = cid
            cid += 1
        mapping[i] = unique_roots[r]

    P = np.zeros((N, cid), dtype=np.float32)
    for i in range(N):
        P[i, mapping[i]] = 1.0
    return P


def generate_all_citeseer(target_methods=None, force=False):
    data_path = os.path.join("src", "Citeseer", "Citeseer_no_isolated.npz")
    print(f"[{get_timestamp()}] Loading raw Citeseer dataset from {data_path}...", flush=True)
    d = np.load(data_path)
    adjs = d["adjs"].astype(np.float32)       # (10, 2110, 2110)
    feats = d["attmats"].astype(np.float32)   # (2110, 10, 100)
    labels = d["labels"].astype(np.float32)   # (2110, 6)

    T, N, _ = adjs.shape
    print(f"[{get_timestamp()}] Citeseer loaded: N={N} nodes, T={T} snapshots, F={feats.shape[2]} features, C={labels.shape[1]} classes.", flush=True)

    alphas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    all_methods = [
        "Ours_Novel",
        "SACoD_supra",
        "SACoD_snap",
        "NetCondense",
        "VariationNb",
        "VariationEdges",
        "HeavyEdge",
        "Random",
    ]
    methods = target_methods if target_methods is not None else all_methods

    results_base = os.path.join("Loukas-graph-coarsening", "results", "Citeseer")
    os.makedirs(results_base, exist_ok=True)

    precomputed = None
    if "Ours_Novel" in methods:
        print(f"[{get_timestamp()}] Precomputing joint spatio-temporal spectral scores for Ours_Novel on Citeseer...", flush=True)
        t0 = time.time()
        precomputed = precompute_coarsening_scores(adjs, feats, gamma=1.0)
        print(f"[{get_timestamp()}] Precomputation completed in {time.time()-t0:.2f}s.", flush=True)

    for alpha in alphas:
        target_K = max(2, int(round(alpha * N)))
        print(f"\n[{get_timestamp()}] ===== Generating Coarsening for Citeseer: alpha={alpha:.1f} (target K={target_K}) =====", flush=True)
        alpha_dir = os.path.join(results_base, f"cr{alpha:.1f}")
        os.makedirs(alpha_dir, exist_ok=True)

        for method in methods:
            save_path = os.path.join(alpha_dir, f"coarsened~{method}.npz")
            if os.path.exists(save_path) and not force:
                print(f"  [{method}] already exists at {save_path}, loading...", flush=True)
                continue

            t_m0 = time.time()
            print(f"  [{get_timestamp()}] Generating [{method}] for alpha={alpha:.1f}...", flush=True)
            if method == "Ours_Novel":
                res = coarsen_jst_novel_citeseer(adjs, feats, labels, alpha=alpha, precomputed=precomputed)
                P = res["P"]
                Q = res["Q"]
                coarsened_adjs = res["coarsened_adjs"]
                coarsened_feats = res["coarsened_feats"]
                coarsened_labels = res["coarsened_labels"]
            else:
                if method == "SACoD_supra":
                    P = coarsen_sacod_supra(adjs, alpha)
                elif method == "SACoD_snap":
                    P = coarsen_sacod_snap(adjs, alpha)
                elif method == "NetCondense":
                    P = coarsen_netcondense_python(adjs, alpha)
                elif method == "VariationNb":
                    P = coarsen_variation_neighborhoods(adjs, alpha)
                elif method == "VariationEdges":
                    P = coarsen_variation_edges(adjs, alpha)
                elif method == "HeavyEdge":
                    P = coarsen_heavy_edge(adjs, alpha)
                elif method == "Random":
                    P = coarsen_random(adjs, alpha)

                # Lift operators and condensed graphs
                actual_K = P.shape[1]
                C_deg = np.sum(P, axis=0)
                C_deg[C_deg == 0] = 1.0
                Q = (1.0 / C_deg)[:, None] * P.T

                coarsened_adjs = np.zeros((T, actual_K, actual_K), dtype=np.float32)
                for t in range(T):
                    coarsened_adjs[t] = P.T @ adjs[t] @ P

                D = feats.shape[2]
                coarsened_feats = np.zeros((actual_K, T, D), dtype=np.float32)
                for t in range(T):
                    coarsened_feats[:, t, :] = Q @ feats[:, t, :]

                coarsened_labels = np.zeros((actual_K, labels.shape[1]), dtype=np.float32)
                for k in range(actual_K):
                    nodes_in_k = np.nonzero(P[:, k])[0]
                    if len(nodes_in_k) > 0:
                        cluster_labels = labels[nodes_in_k]
                        dominant_class = np.argmax(np.sum(cluster_labels, axis=0))
                        coarsened_labels[k, dominant_class] = 1.0

            np.savez_compressed(
                save_path,
                P=P,
                Q=Q,
                coarsened_adjs=coarsened_adjs,
                coarsened_feats=coarsened_feats,
                coarsened_labels=coarsened_labels,
            )
            print(f"    Saved [{method} @ cr{alpha:.1f}]: K={P.shape[1]} in {time.time()-t_m0:.2f}s", flush=True)


if __name__ == "__main__":
    generate_all_citeseer()
