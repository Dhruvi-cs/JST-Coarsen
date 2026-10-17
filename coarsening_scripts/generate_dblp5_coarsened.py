"""
Unified Dynamic Graph Coarsening Generator for DBLP5
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
import networkx as nx
from typing import Dict, List, Tuple
from novel_coarsening import coarsen_dynamic_graph_novel, precompute_coarsening_scores


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
        pos_idx = np.where(vals_real > 1e-3)[0]
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
        pos_idx = np.where(vals_real > 1e-3)[0]
        fiedler_idx = pos_idx[0] if len(pos_idx) > 0 else 1
        lambda_2 = float(vals_real[fiedler_idx])
        v_supra = np.real(vecs[:, fiedler_idx])
        
    V_t = v_supra.reshape(T, N)
    v_norm_sq = float(np.dot(v_supra, v_supra))
    if v_norm_sq == 0: v_norm_sq = 1.0
    
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
            
    # Build P matrix (N, K)
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
            pos_idx = np.where(vals_real > 1e-3)[0]
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
        
    all_snap_scores = np.stack(all_snap_scores, axis=0) # (T, num_edges)
    scores = np.max(all_snap_scores, axis=0) # max over snapshots
    
    # Greedy DSU
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


def generate_all(target_methods=None, force=False):
    data_path = os.path.join("src", "DBLP5", "DBLP5_no_isolated.npz")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading raw DBLP5 dataset from {data_path}...")
    d = np.load(data_path)
    adjs = d["adjs"].astype(np.float32)       # (10, 2390, 2390)
    feats = d["attmats"].astype(np.float32)   # (2390, 10, 100)
    labels = d["labels"].astype(np.float32)   # (2390, 5)

    T, N, _ = adjs.shape
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Dataset loaded: N={N} nodes, T={T} snapshots, F={feats.shape[2]} features, C={labels.shape[1]} classes.")
    
    alphas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    methods = ["Ours_Novel", "SACoD_supra", "SACoD_snap"]
    if target_methods is not None:
        methods = [m for m in methods if m in target_methods]
        
    results_base = os.path.join("Loukas-graph-coarsening", "results", "DBLP5")
    os.makedirs(results_base, exist_ok=True)
    
    precomputed = None
    if "Ours_Novel" in methods:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Precomputing joint spatio-temporal spectral scores for Ours_Novel on DBLP5...")
        t0 = time.time()
        precomputed = precompute_coarsening_scores(adjs, feats, gamma=1.0)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Precomputation done in {time.time()-t0:.2f}s.")
    
    for alpha in alphas:
        target_K = max(2, int(round(alpha * N)))
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ===== Generating Coarsening for DBLP5: alpha={alpha:.1f} (target K={target_K}) =====")
        alpha_dir = os.path.join(results_base, f"cr{alpha:.1f}")
        os.makedirs(alpha_dir, exist_ok=True)
        
        for method in methods:
            save_path = os.path.join(alpha_dir, f"coarsened~{method}.npz")
            if os.path.exists(save_path) and not force:
                print(f"  [{method}] already exists at {save_path}, loading...")
                continue
                
            t_m0 = time.time()
            print(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')}] Generating [{method}] for alpha={alpha:.1f}...")
            if method == "Ours_Novel":
                res = coarsen_dynamic_graph_novel(adjs, feats, labels, alpha=alpha, precomputed=precomputed, beta=0.05, rho=0.05)
                P = res["P"]
            elif method == "SACoD_supra":
                P = coarsen_sacod_supra(adjs, alpha)
            elif method == "SACoD_snap":
                P = coarsen_sacod_snap(adjs, alpha)
                
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
            print(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')}] -> Saved {method} (K={K}) to {save_path} in {time.time()-t_m0:.2f}s")

    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] All coarsened datasets for DBLP5 generated successfully!")


if __name__ == "__main__":
    generate_all()
