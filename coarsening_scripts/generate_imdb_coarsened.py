"""
Unified Dynamic Graph Coarsening Generator for IMDB
Generates coarsened graph partitions across all ratios alpha in [0.3, 0.9] for:
1. Ours_Novel (JST-Coarsen: Solutions 1-5)
2. SACoD_supra (1st-order supra Laplacian + DSU)
3. SACoD_snap (1st-order snapshot Laplacian + DSU)
"""

import os
import sys
import math
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from typing import Dict, List, Tuple
from novel_coarsening import coarsen_dynamic_graph_novel, precompute_coarsening_scores


def get_union_graph(adjs: np.ndarray) -> np.ndarray:
    union_adj = np.sum(adjs, axis=0)
    union_adj = (union_adj > 0).astype(np.float32)
    np.fill_diagonal(union_adj, 0)
    return union_adj


def precompute_sacod_supra_scores(adjs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Precomputes SACoD-supra perturbation scores once for all candidate edges.
    """
    T, N, _ = adjs.shape
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Building standard supra-Laplacian for SACoD-supra (NT={N*T})...", flush=True)
    t0 = time.time()
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
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] L_supra built in {time.time()-t0:.2f}s. Computing eigsh(which='SM')...", flush=True)
    
    t1 = time.time()
    try:
        vals, vecs = eigsh(L_supra, k=2, which='SM', tol=1e-3, ncv=10, maxiter=800)
        vals_real = np.real(vals)
        pos_idx = np.where(vals_real > 1e-4)[0]
        fiedler_idx = pos_idx[0] if len(pos_idx) > 0 else 1
        lambda_2 = float(vals_real[fiedler_idx])
        v_supra = np.real(vecs[:, fiedler_idx])
    except Exception as e:
        if hasattr(e, 'eigenvalues') and e.eigenvalues is not None and len(e.eigenvalues) > 0:
            vals_real = np.real(e.eigenvalues)
            vecs = np.real(e.eigenvectors)
            pos_idx = np.where(vals_real > 1e-4)[0]
            fiedler_idx = pos_idx[0] if len(pos_idx) > 0 else (len(vals_real) - 1)
            lambda_2 = float(vals_real[fiedler_idx])
            v_supra = np.real(vecs[:, fiedler_idx])
        else:
            print(f"Warning in SACoD-supra eigsh: {e}, using fallback", flush=True)
            vals, vecs = eigsh(L_supra, k=2, which='SA', tol=1e-3)
            lambda_2 = float(np.real(vals[1]))
            v_supra = np.real(vecs[:, 1])
        
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SACoD-supra eigsh completed in {time.time()-t1:.2f}s (lambda_2={lambda_2:.6f}).", flush=True)
    
    V_t = v_supra.reshape(T, N)
    v_norm_sq = float(np.dot(v_supra, v_supra))
    if v_norm_sq == 0:
        v_norm_sq = 1.0
        
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
    order = np.argsort(-scores)
    return order, rows, cols


def precompute_sacod_snap_scores(adjs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Precomputes SACoD-snap perturbation scores across all snapshots.
    """
    T, N, _ = adjs.shape
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Precomputing SACoD-snap scores across {T} snapshots...", flush=True)
    t0 = time.time()
    
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
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SACoD-snap precomputation finished in {time.time()-t0:.2f}s.", flush=True)
    return order, rows, cols


def contraction_dsu(order: np.ndarray, rows: np.ndarray, cols: np.ndarray, N: int, target_K: int) -> np.ndarray:
    """
    Greedy Disjoint Set Union contraction.
    """
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


def generate_all(target_methods=None, force=False):
    data_path = os.path.join("src", "IMDB", "IMDB_no_isolated.npz")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading raw IMDB dataset from {data_path}...", flush=True)
    d = np.load(data_path)
    adjs = d["adjs"].astype(np.float32)       # (47, 3883, 3883)
    attmats = d["attmats"].astype(np.float32) # (3883, 73)
    labels = d["labels"].astype(np.float32)   # (3883, 2)

    T, N, _ = adjs.shape
    feats = np.repeat(attmats[:, None, :], T, axis=1) # (3883, 47, 73)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Dataset loaded: N={N} nodes, T={T} snapshots, F={feats.shape[2]} features, C={labels.shape[1]} classes.", flush=True)
    
    alphas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    methods = ["Ours_Novel", "SACoD_supra", "SACoD_snap"]
    if target_methods is not None:
        methods = [m for m in methods if m in target_methods]
        
    results_base = os.path.join("Loukas-graph-coarsening", "results", "IMDB")
    os.makedirs(results_base, exist_ok=True)
    
    # 1. Precompute Ours_Novel candidate scores
    precomputed_novel = None
    if "Ours_Novel" in methods:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Precomputing joint spatio-temporal spectral scores for Ours_Novel on IMDB...", flush=True)
        t0 = time.time()
        precomputed_novel = precompute_coarsening_scores(adjs, feats, gamma=1.0)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Ours_Novel precomputation done in {time.time()-t0:.2f}s.", flush=True)

    # 2. Precompute SACoD_supra scores
    supra_order, supra_rows, supra_cols = None, None, None
    if "SACoD_supra" in methods:
        supra_order, supra_rows, supra_cols = precompute_sacod_supra_scores(adjs)

    # 3. Precompute SACoD_snap scores
    snap_order, snap_rows, snap_cols = None, None, None
    if "SACoD_snap" in methods:
        snap_order, snap_rows, snap_cols = precompute_sacod_snap_scores(adjs)
    
    for alpha in alphas:
        target_K = max(2, int(round(alpha * N)))
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== Generating Coarsening for IMDB: alpha={alpha:.1f} (target K={target_K}) =====", flush=True)
        alpha_dir = os.path.join(results_base, f"cr{alpha:.1f}")
        os.makedirs(alpha_dir, exist_ok=True)
        
        for method in methods:
            save_path = os.path.join(alpha_dir, f"coarsened~{method}.npz")
            if os.path.exists(save_path) and not force:
                print(f"  [{method}] already exists at {save_path}, skipping...", flush=True)
                continue
                
            t_m0 = time.time()
            print(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')}] Generating [{method}] for alpha={alpha:.1f}...", flush=True)
            if method == "Ours_Novel":
                res = coarsen_dynamic_graph_novel(adjs, feats, labels, alpha=alpha, precomputed=precomputed_novel, beta=0.05, rho=0.05)
                P = res["P"]
            elif method == "SACoD_supra":
                P = contraction_dsu(supra_order, supra_rows, supra_cols, N, target_K)
            elif method == "SACoD_snap":
                P = contraction_dsu(snap_order, snap_rows, snap_cols, N, target_K)
                
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
            print(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')}] -> Saved {method} (K={K}) to {save_path} in {time.time()-t_m0:.2f}s", flush=True)

    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] All coarsened datasets for IMDB generated successfully!", flush=True)


if __name__ == "__main__":
    generate_all(force=True)
