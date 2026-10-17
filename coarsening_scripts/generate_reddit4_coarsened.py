"""
Unified Dynamic Graph Coarsening Generator for Reddit4
Generates coarsened graph partitions across all ratios alpha in [0.3, 0.9] for:
1. Ours-Novel (JST-Coarsen: Solutions 1-5)
2. SACoD-supra (1st-order supra Laplacian + DSU)
3. SACoD-snap (1st-order snapshot Laplacian + DSU)
4. Random (random edge selection)
5. HeavyEdge (heavy edge degree matching)
6. VariationEdges (spectral edge variation)
7. VariationNb (neighborhood variation)
8. NetCondense (temporal condensation)
"""

import os
import sys
import math
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import networkx as nx
from typing import Dict, List, Tuple
from novel_coarsening import coarsen_dynamic_graph_novel, precompute_coarsening_scores


def get_union_graph(adjs: np.ndarray) -> np.ndarray:
    T, N, _ = adjs.shape
    union_adj = np.sum(adjs > 0, axis=0) > 0
    np.fill_diagonal(union_adj, False)
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
    degs = np.sum(adjs, axis=2, dtype=np.float32)
    
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
    
    # 2. Eigsh for 2nd smallest eigenpair (true Fiedler algebraic connectivity)
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
        deg_ab = degs[t, rows] + degs[t, cols]
        intra_terms += (v_diff ** 2) * (0.25 * (deg_ab + 2.0 * w_ab) - lambda_2)
        
    inter_terms = np.zeros(len(rows), dtype=np.float32)
    for t in range(T - 1):
        v_drift = (V_t[t + 1, rows] - V_t[t + 1, cols]) - (V_t[t, rows] - V_t[t, cols])
        inter_terms -= 0.5 * (v_drift ** 2)
        
    scores = (intra_terms + inter_terms) / (v_norm_sq * max(1e-4, lambda_2))
    
    # 4. Greedy DSU merging
    sorted_idx = np.argsort(-scores)
    parent = list(range(N))
    
    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
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
                
    # Fallback if needed
    if current_supernodes > target_K:
        for u in range(N):
            for v in range(u + 1, N):
                ru = find(u)
                rv = find(v)
                if ru != rv:
                    parent[rv] = ru
                    current_supernodes -= 1
                    if current_supernodes == target_K:
                        break
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
    """
    SACoD-Snap: 1st-order Snapshot Laplacian spectral gap perturbation with greedy DSU.
    Score = max_t (Delta lambda_snap^(t) / lambda^(t))
    """
    T, N, _ = adjs.shape
    target_K = max(2, int(round(alpha * N)))
    
    # Compute Laplacian eigenpairs per snapshot
    snap_lambdas = []
    snap_vecs = []
    degs = np.sum(adjs, axis=2, dtype=np.float32)
    
    for t in range(T):
        A_sp = sp.csr_matrix(adjs[t], dtype=np.float32)
        d_t = np.array(A_sp.sum(axis=1)).flatten()
        L_t = sp.diags(d_t) - A_sp
        G_snap = nx.from_scipy_sparse_array(A_sp)
        comps = sorted(nx.connected_components(G_snap), key=len, reverse=True)
        if len(comps) > 0 and len(comps[0]) > 2:
            giant = list(comps[0])
            H = G_snap.subgraph(giant)
            L_giant = nx.laplacian_matrix(H).astype(np.float32)
            try:
                vals, vecs = eigsh(L_giant, k=2, which='SM', tol=1e-3)
                vals_sorted = np.sort(np.clip(np.real(vals), 0.0, None))
                l2 = float(vals_sorted[1]) if len(vals_sorted) > 1 and vals_sorted[1] > 1e-4 else 0.05
                fiedler_local = np.real(vecs[:, 1])
                v2 = np.zeros(N, dtype=np.float32)
                v2[giant] = fiedler_local
            except Exception:
                l2 = 0.05
                v2 = np.zeros(N, dtype=np.float32)
        else:
            l2 = 0.05
            v2 = np.zeros(N, dtype=np.float32)
        snap_lambdas.append(max(1e-4, l2))
        snap_vecs.append(v2)
        
    union_adj = get_union_graph(adjs)
    rows, cols = np.nonzero(np.triu(union_adj, k=1))
    
    max_scores = np.full(len(rows), -1e9, dtype=np.float32)
    for t in range(T):
        v_t = snap_vecs[t]
        v_diff_sq = (v_t[rows] - v_t[cols]) ** 2
        w_ab = adjs[t, rows, cols]
        deg_ab = degs[t, rows] + degs[t, cols]
        v_norm_sq = float(np.dot(v_t, v_t)) if np.dot(v_t, v_t) > 0 else 1.0
        delta_l = (v_diff_sq / v_norm_sq) * (0.25 * (deg_ab + 2.0 * w_ab) - snap_lambdas[t])
        norm_score = delta_l / snap_lambdas[t]
        max_scores = np.maximum(max_scores, norm_score)
        
    sorted_idx = np.argsort(-max_scores)
    parent = list(range(N))
    
    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
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
    # Score = w_uv / sqrt(d_u * d_v) or w_uv * (d_u + d_v)
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
    VariationEdges baseline: picks edges to contract that minimize change in Dirichlet energy.
    """
    T, N, _ = adjs.shape
    target_K = max(2, int(round(alpha * N)))
    
    W_union = np.sum(adjs, axis=0, dtype=np.float32)
    np.fill_diagonal(W_union, 0.0)
    L_union = sp.diags(np.sum(W_union, axis=1)) - sp.csr_matrix(W_union)
    
    # Compute low eigenvectors of static union Laplacian
    try:
        vals, vecs = eigsh(L_union, k=min(10, N - 1), sigma=-1e-5, which='LM', tol=1e-3)
    except Exception:
        vals, vecs = eigsh(L_union, k=min(10, N - 1), which='SA', tol=1e-3)
        
    rows, cols = np.nonzero(np.triu(W_union > 0, k=1))
    # Variation of edges: sum_k (U_{k, u} - U_{k, v})^2 / ||U_k||^2
    var_scores = np.zeros(len(rows), dtype=np.float32)
    for k in range(vecs.shape[1]):
        u_k = vecs[:, k]
        var_scores += (u_k[rows] - u_k[cols]) ** 2
        
    # VariationEdges minimizes spectral distortion -> sort ascending!
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


def coarsen_variation_neighborhoods(adjs: np.ndarray, alpha: float) -> np.ndarray:
    """
    VariationNb baseline: merges neighborhoods with highest structural similarity.
    """
    T, N, _ = adjs.shape
    target_K = max(2, int(round(alpha * N)))
    
    W_union = np.sum(adjs, axis=0, dtype=np.float32)
    np.fill_diagonal(W_union, 0.0)
    
    # Neighborhood Jaccard / Cosine similarity
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


def coarsen_netcondense_from_scores(scores_path: str, N: int, alpha: float) -> np.ndarray:
    """
    NetCondense baseline: reads precomputed edgeScores and merges nodes using temporal condensation routine.
    """
    target_K = max(2, int(round(alpha * N)))
    
    scores = []
    if os.path.exists(scores_path):
        with open(scores_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    u, v, s = int(parts[0]), int(parts[1]), float(parts[2])
                    # Note: NetCondense is 1-indexed
                    if 1 <= u <= N and 1 <= v <= N:
                        scores.append((u - 1, v - 1, s))
                        
    # Sort ascending by diffusion error score
    scores.sort(key=lambda x: x[2])
    
    parent = list(range(N))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
        
    current_supernodes = N
    for u, v, s in scores:
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


def generate_all(target_methods=None, force=False):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force overwrite existing coarsened files")
    parser.add_argument("--methods", nargs="+", default=None, help="List of methods to generate")
    args, _ = parser.parse_known_args()
    
    if target_methods is None and args.methods:
        target_methods = args.methods
    if not force and args.force:
        force = True

    data_path = os.path.join("src", "Reddit4", "Reddit4_no_isolated.npz")
    print(f"Loading raw Reddit4 dataset from {data_path}...")
    d = np.load(data_path)
    adjs = d["adjs"]
    feats = d["attmats"]
    labels = d["labels"]
    
    T, N, _ = adjs.shape
    print(f"Reddit4: T={T}, N={N}, Classes={labels.shape[1]}, Features={feats.shape[2]}")
    
    alphas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    all_methods = [
        "Ours_Novel",
        "SACoD_supra",
        "SACoD_snap",
        "Random",
        "HeavyEdge",
        "VariationEdges",
        "VariationNb",
        "NetCondense",
    ]
    methods = target_methods if target_methods is not None else all_methods
    
    results_base = os.path.join("Loukas-graph-coarsening", "results", "Reddit4")
    os.makedirs(results_base, exist_ok=True)
    
    netcondense_scores = os.path.join("NetCondense", "data", "Reddit4", "edgelist.txt_edgeScores")
    
    # Precompute novel scores once if Ours_Novel is in methods
    precomputed = None
    if "Ours_Novel" in methods:
        print("Precomputing joint spatio-temporal spectral scores for Ours_Novel...")
        precomputed = precompute_coarsening_scores(adjs, feats, gamma=1.0)
    
    for alpha in alphas:
        target_K = max(2, int(round(alpha * N)))
        print(f"\n=================== Generating Coarsening for alpha={alpha:.1f} (target K={target_K}) ===================")
        alpha_dir = os.path.join(results_base, f"cr{alpha:.1f}")
        os.makedirs(alpha_dir, exist_ok=True)
        
        for method in methods:
            save_path = os.path.join(alpha_dir, f"coarsened~{method}.npz")
            if os.path.exists(save_path) and not force:
                print(f"  [{method}] already exists at {save_path}, loading...")
                continue
                
            print(f"  Generating [{method}] for alpha={alpha:.1f}...")
            if method == "Ours_Novel":
                res = coarsen_dynamic_graph_novel(adjs, feats, labels, alpha=alpha, precomputed=precomputed, beta=0.05, rho=0.05)
                P = res["P"]
            elif method == "SACoD_supra":
                P = coarsen_sacod_supra(adjs, alpha)
            elif method == "SACoD_snap":
                P = coarsen_sacod_snap(adjs, alpha)
            elif method == "Random":
                P = coarsen_random(adjs, alpha)
            elif method == "HeavyEdge":
                P = coarsen_heavy_edge(adjs, alpha)
            elif method == "VariationEdges":
                P = coarsen_variation_edges(adjs, alpha)
            elif method == "VariationNb":
                P = coarsen_variation_neighborhoods(adjs, alpha)
            elif method == "NetCondense":
                P = coarsen_netcondense_from_scores(netcondense_scores, N, alpha)
                
            K = P.shape[1]
            # Construct coarsened adjs, feats, labels
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
            print(f"    -> Saved {method} (K={K}) to {save_path}")

    print("\nAll coarsened datasets for all methods and all alpha ratios generated successfully!")


if __name__ == "__main__":
    generate_all()
