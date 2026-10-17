"""
Unified Dynamic Graph Coarsening Generator for Cora
Generates coarsened graph partitions across all ratios alpha in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] for:
1. Ours_Novel (JST-Coarsen: Solutions 1-5, time-coupled regularized coarsening)
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
from novel_coarsening import coarsen_dynamic_graph_novel, precompute_coarsening_scores


def get_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def get_union_graph(adjs: np.ndarray) -> np.ndarray:
    union_adj = np.sum(adjs, axis=0)
    union_adj = (union_adj > 0).astype(np.float32)
    np.fill_diagonal(union_adj, 0)
    return union_adj


def coarsen_sacod_supra(adjs: np.ndarray, alpha: float) -> np.ndarray:
    """
    SACoD-Supra: 1st-order Supra-Laplacian spectral gap perturbation with greedy DSU.
    """
    T, N, _ = adjs.shape
    target_K = max(2, int(round(alpha * N)))
    
    # 1. Build standard supra-Laplacian with omega = 1.0
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
    
    # 2. Eigsh for 2nd smallest eigenpair
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
    v_norm_sq = float(np.dot(v_supra, v_supra))
    if v_norm_sq == 0:
        v_norm_sq = 1.0
    
    # 3. Score all edges in union graph
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
        
    scores = intra_terms / v_norm_sq
    
    # 4. Greedy contraction using Disjoint Set Union (DSU)
    order = np.argsort(-scores)
    parent = list(range(N))
    
    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i
        
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i
            return True
        return False

    current_clusters = N
    for idx in order:
        if current_clusters <= target_K:
            break
        u, v = rows[idx], cols[idx]
        if union(u, v):
            current_clusters -= 1
            
    cluster_map = {}
    for i in range(N):
        r = find(i)
        if r not in cluster_map:
            cluster_map[r] = len(cluster_map)
            
    K = len(cluster_map)
    P = np.zeros((N, K), dtype=np.float32)
    for i in range(N):
        P[i, cluster_map[find(i)]] = 1.0
        
    return P


def coarsen_sacod_snap(adjs: np.ndarray, alpha: float) -> np.ndarray:
    """
    SACoD-Snap: 1st-order Snapshot Laplacian spectral gap perturbation with greedy DSU.
    """
    T, N, _ = adjs.shape
    target_K = max(2, int(round(alpha * N)))
    
    union_adj = get_union_graph(adjs)
    rows, cols = np.nonzero(np.triu(union_adj, k=1))
    
    all_snap_scores = []
    
    for t in range(T):
        A_t = sp.csr_matrix(adjs[t], dtype=np.float32)
        d_t = np.array(A_t.sum(axis=1)).flatten()
        L_t = sp.diags(d_t) - A_t
        
        try:
            vals, vecs = eigsh(L_t, k=4, sigma=0.05, which='LM', tol=1e-3)
            vals_real = np.real(vals)
            pos_idx = np.where(vals_real > 1e-4)[0]
            if len(pos_idx) > 0:
                fiedler_idx = pos_idx[np.argmin(vals_real[pos_idx])]
                lambda_2 = float(vals_real[fiedler_idx])
                v_t = np.real(vecs[:, fiedler_idx])
            else:
                lambda_2 = 1.0
                v_t = np.ones(N, dtype=np.float32) / np.sqrt(N)
        except Exception:
            lambda_2 = 1.0
            v_t = np.ones(N, dtype=np.float32) / np.sqrt(N)
            
        v_diff = v_t[rows] - v_t[cols]
        w_ab = adjs[t, rows, cols]
        d_a = np.sum(adjs[t, rows, :], axis=1)
        d_b = np.sum(adjs[t, cols, :], axis=1)
        k_val = (d_a + d_b + 2.0 * w_ab) / 4.0
        delta_lambda = (v_diff ** 2) * (k_val - lambda_2) / (np.dot(v_t, v_t) + 1e-8)
        norm_score = delta_lambda / max(1e-4, lambda_2)
        all_snap_scores.append(norm_score)
        
    all_snap_scores = np.stack(all_snap_scores, axis=0)
    scores = np.max(all_snap_scores, axis=0)
    
    order = np.argsort(-scores)
    parent = list(range(N))
    
    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i
        
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i
            return True
        return False

    current_clusters = N
    for idx in order:
        if current_clusters <= target_K:
            break
        u, v = rows[idx], cols[idx]
        if union(u, v):
            current_clusters -= 1
            
    cluster_map = {}
    for i in range(N):
        r = find(i)
        if r not in cluster_map:
            cluster_map[r] = len(cluster_map)
            
    K = len(cluster_map)
    P = np.zeros((N, K), dtype=np.float32)
    for i in range(N):
        P[i, cluster_map[find(i)]] = 1.0
        
    return P


def coarsen_random(adjs: np.ndarray, alpha: float, seed: int = 42) -> np.ndarray:
    """
    Random baseline: randomly merges pairs connected in the union graph.
    """
    rng = np.random.RandomState(seed)
    T, N, _ = adjs.shape
    target_K = max(2, int(round(alpha * N)))
    
    union_adj = get_union_graph(adjs)
    rows, cols = np.nonzero(np.triu(union_adj, k=1))
    edges = list(zip(rows.tolist(), cols.tolist()))
    rng.shuffle(edges)
    
    parent = list(range(N))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
        
    current_supernodes = N
    for u, v in edges:
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
    """
    HeavyEdgeDegree baseline: merges edges with highest weights and degrees in union graph.
    """
    T, N, _ = adjs.shape
    target_K = max(2, int(round(alpha * N)))
    
    W_union = np.sum(adjs, axis=0, dtype=np.float32)
    np.fill_diagonal(W_union, 0.0)
    degs = np.sum(W_union, axis=1)
    
    rows, cols = np.nonzero(np.triu(W_union > 0, k=1))
    scores = W_union[rows, cols] / np.sqrt(np.maximum(1e-4, degs[rows] * degs[cols]))
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


def coarsen_variation_edges(adjs: np.ndarray, alpha: float) -> np.ndarray:
    """
    VariationEdges baseline: contracts edges that minimize change in Dirichlet energy.
    """
    T, N, _ = adjs.shape
    target_K = max(2, int(round(alpha * N)))
    
    W_union = np.sum(adjs, axis=0, dtype=np.float32)
    np.fill_diagonal(W_union, 0.0)
    L_union = sp.diags(np.sum(W_union, axis=1)) - sp.csr_matrix(W_union)
    
    try:
        vals, vecs = eigsh(L_union, k=min(10, N - 1), sigma=-1e-5, which='LM', tol=1e-3)
    except Exception:
        vals, vecs = eigsh(L_union, k=min(10, N - 1), which='SA', tol=1e-3)
        
    rows, cols = np.nonzero(np.triu(W_union > 0, k=1))
    var_scores = np.zeros(len(rows), dtype=np.float32)
    for k in range(vecs.shape[1]):
        u_k = vecs[:, k]
        var_scores += (u_k[rows] - u_k[cols]) ** 2
        
    sorted_idx = np.argsort(var_scores)  # Ascending to minimize variation
    
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


def coarsen_variation_neighborhoods(adjs: np.ndarray, alpha: float) -> np.ndarray:
    """
    VariationNb baseline: merges neighborhoods with highest structural similarity.
    """
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


def coarsen_netcondense_python(adjs: np.ndarray, alpha: float) -> np.ndarray:
    """
    NetCondense baseline implemented directly in Python matching getNodeScores.m & flattenNetwork.m.
    Constructs the block-cyclic temporal diffusion graph and scores edges by leading eigenvector perturbation.
    """
    T, N, _ = adjs.shape
    target_K = max(2, int(round(alpha * N)))

    # 1. Build flattened block-cyclic network (NT, NT)
    I_N = sp.identity(N, format="csr", dtype=np.float32)
    blocks = [[None for _ in range(T)] for _ in range(T)]
    for t in range(T):
        A_t = sp.csr_matrix(adjs[t], dtype=np.float32)
        next_t = 0 if t == T - 1 else t + 1
        blocks[t][next_t] = A_t + I_N

    fG = sp.bmat(blocks, format="csr")
    flatM = (fG + fG.T) * 0.5

    # 2. Leading eigenvector of flatM
    try:
        vals, vecs = eigsh(flatM, k=1, which='LM', tol=1e-3, maxiter=500)
        u = np.abs(np.real(vecs[:, 0]))
        lambda_val = float(np.abs(np.real(vals[0])))
    except Exception:
        u = np.ones(N * T, dtype=np.float32) / np.sqrt(N * T)
        lambda_val = 1.0

    uv_sum = float(np.sum(u ** 2))
    U_nodes = u.reshape(T, N).T  # (N, T)

    union_adj = get_union_graph(adjs)
    rows, cols = np.nonzero(np.triu(union_adj, k=1))

    # 3. Vectorized NetCondense edge scoring matching getNodeScores.m
    ui = U_nodes[rows, :]  # (E, T)
    uj = U_nodes[cols, :]  # (E, T)
    first_term = np.sum(ui ** 2 + uj ** 2, axis=1)  # (E,)

    # Extract edge weights across all snapshots: (E, T)
    w_t = np.zeros((len(rows), T), dtype=np.float32)
    for t in range(T):
        w_t[:, t] = adjs[t, rows, cols]

    second_term = np.sum(w_t * ui * uj, axis=1)  # (E,)

    # Circular shift across time (circshift(uj, 1))
    uj_shift = np.roll(uj, shift=1, axis=1)
    ui_shift = np.roll(ui, shift=1, axis=1)

    term3_i = (lambda_val * ui) - (w_t * uj_shift)
    term3_j = (lambda_val * uj) - (w_t * ui_shift)
    third_term = 0.5 * np.sum((term3_i + term3_j) * uj, axis=1)

    denom = np.maximum(1e-6, uv_sum - first_term)
    numer = (-lambda_val * first_term) + second_term + third_term
    scores = np.abs(numer / denom)

    # Sort ascending (minimum diffusion distortion)
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


def generate_all_cora(target_methods=None, force=False):
    data_path = os.path.join("src", "Cora", "Cora_no_isolated.npz")
    print(f"[{get_timestamp()}] Loading raw Cora dataset from {data_path}...", flush=True)
    d = np.load(data_path)
    adjs = d["adjs"].astype(np.float32)       # (10, 2492, 2492)
    feats = d["attmats"].astype(np.float32)   # (2492, 10, 100)
    labels = d["labels"].astype(np.float32)   # (2492, 7)

    T, N, _ = adjs.shape
    print(f"[{get_timestamp()}] Cora dataset loaded: N={N} nodes, T={T} snapshots, F={feats.shape[2]} features, C={labels.shape[1]} classes.", flush=True)

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

    results_base = os.path.join("Loukas-graph-coarsening", "results", "Cora")
    os.makedirs(results_base, exist_ok=True)

    precomputed = None
    if "Ours_Novel" in methods:
        print(f"[{get_timestamp()}] Precomputing joint spatio-temporal spectral scores for Ours_Novel on Cora...", flush=True)
        t0 = time.time()
        precomputed = precompute_coarsening_scores(adjs, feats, gamma=1.0)
        print(f"[{get_timestamp()}] Precomputation completed in {time.time()-t0:.2f}s.", flush=True)

    for alpha in alphas:
        target_K = max(2, int(round(alpha * N)))
        print(f"\n[{get_timestamp()}] ===== Generating Coarsening for Cora: alpha={alpha:.1f} (target K={target_K}) =====", flush=True)
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
                res = coarsen_dynamic_graph_novel(adjs, feats, labels, alpha=alpha, precomputed=precomputed, beta=0.05, rho=0.05)
                P = res["P"]
            elif method == "SACoD_supra":
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
            else:
                raise ValueError(f"Unknown method {method}")

            K = P.shape[1]
            C = np.sum(P, axis=0)
            C[C == 0] = 1.0
            Q = (1.0 / C)[:, None] * P.T

            cadjs = np.zeros((T, K, K), dtype=np.float32)
            for t in range(T):
                cadjs[t] = P.T @ adjs[t] @ P

            cfeats = np.zeros((K, T, feats.shape[2]), dtype=np.float32)
            for t in range(T):
                cfeats[:, t, :] = Q @ feats[:, t, :]

            clabels = np.zeros((K, labels.shape[1]), dtype=np.float32)
            for k_idx in range(K):
                nodes_in_k = np.nonzero(P[:, k_idx])[0]
                if len(nodes_in_k) > 0:
                    dominant_c = np.argmax(np.sum(labels[nodes_in_k], axis=0))
                    clabels[k_idx, dominant_c] = 1.0

            np.savez_compressed(
                save_path,
                P=P,
                Q=Q,
                Nprime=K,
                coarsened_adjs=cadjs,
                coarsened_feats=cfeats,
                coarsened_labels=clabels,
                alpha=alpha,
                method=method,
            )
            print(f"  [{get_timestamp()}] -> Saved {method} (K={K}) to {save_path} in {time.time()-t_m0:.2f}s", flush=True)

    print(f"\n[{get_timestamp()}] All 56 coarsened datasets for Cora generated successfully!", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force regenerate files")
    parser.add_argument("--methods", nargs="+", default=None, help="List of methods to generate")
    args = parser.parse_args()
    generate_all_cora(target_methods=args.methods, force=args.force)
