"""
Preprocesses raw Citeseer files from citeseer.zip into standard dynamic graph format matching Cora/DBLP5/Reddit4/IMDB:
- Number of snapshots T = 10
- Classes C = 6 (Agents, AI, DB, IR, ML, HCI)
- Node features F = 100 (smooth spectral-structural embeddings)
- Output: src/Citeseer/Citeseer_no_isolated.npz with keys: ['attmats', 'adjs', 'labels', 'keep_nodes']
"""

import os
import zipfile
import numpy as np
import scipy.sparse as sp
import networkx as nx
from sklearn.decomposition import TruncatedSVD

def build_dynamic_citeseer():
    zip_path = r"C:\Users\HP\Downloads\citeseer.zip"
    out_dir = r"src\Citeseer"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "Citeseer_no_isolated.npz")

    print(f"Loading raw Citeseer files from {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        edges_raw = z.read('citeseer.edges').decode('utf-8', errors='ignore').strip().splitlines()
        labels_raw = z.read('citeseer.node_labels').decode('utf-8', errors='ignore').strip().splitlines()

    # 1. Parse node labels (1-indexed)
    raw_labels = {}
    for l in labels_raw:
        if l.strip():
            parts = l.strip().split(',')
            node = int(parts[0])
            c = int(parts[1]) - 1  # 0-indexed class (0 to 5)
            raw_labels[node] = c

    # 2. Parse edges
    raw_edges = []
    for l in edges_raw:
        if l.strip():
            parts = l.strip().split(',')
            u, v = int(parts[0]), int(parts[1])
            if u != v and u in raw_labels and v in raw_labels:
                raw_edges.append((u, v))

    print(f"Parsed {len(raw_edges)} valid edges across {len(raw_labels)} labeled nodes.")

    # 3. Extract Giant Connected Component (GCC)
    G = nx.Graph()
    for u, v in raw_edges:
        G.add_edge(u, v)

    largest_cc = max(nx.connected_components(G), key=len)
    H = G.subgraph(largest_cc)
    final_nodes = sorted(list(largest_cc))
    N = len(final_nodes)
    node_to_idx = {p: i for i, p in enumerate(final_nodes)}

    class_names = ["Agents", "AI", "DB", "IR", "ML", "HCI"]
    C = len(class_names)
    class_counts = {class_names[c]: sum(1 for p in final_nodes if raw_labels[p] == c) for c in range(C)}
    print(f"Final connected Citeseer benchmark: N = {N} nodes, {H.number_of_edges()} unique undirected edges.")
    print(f"Class distribution: {class_counts}")

    # 4. Generate 100-dimensional spectral-structural node features (F=100)
    print("Computing 100-dimensional smooth spectral-structural node features...")
    adj_static = nx.to_scipy_sparse_array(H, nodelist=final_nodes, format="csr", dtype=np.float32)
    # Symmetric normalized adjacency
    deg = np.array(adj_static.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(np.maximum(1e-6, deg), -0.5)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    A_norm = D_inv_sqrt @ adj_static @ D_inv_sqrt

    # Truncated SVD for top spectral components
    svd = TruncatedSVD(n_components=70, random_state=42)
    X_svd = svd.fit_transform(A_norm)

    # Multi-hop diffusion features for local structural context
    A2 = A_norm @ A_norm
    svd2 = TruncatedSVD(n_components=30, random_state=42)
    X_diff = svd2.fit_transform(A2)

    feats_static = np.hstack([X_svd, X_diff]).astype(np.float64)  # (N, 100)
    # L2-normalize
    row_norms = np.linalg.norm(feats_static, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    feats_static = feats_static / row_norms
    F_dim = 100

    # 5. One-hot labels (N, 6)
    labels_onehot = np.zeros((N, C), dtype=np.float64)
    for p, idx in node_to_idx.items():
        c = raw_labels[p]
        labels_onehot[idx, c] = 1.0

    # 6. Construct dynamic snapshot sequence (T = 10)
    T = 10
    print(f"Constructing dynamic graph sequence across T = {T} temporal snapshots...")
    unique_undirected = []
    for u, v in H.edges():
        idx_u, idx_v = node_to_idx[u], node_to_idx[v]
        unique_undirected.append((min(idx_u, idx_v), max(idx_u, idx_v)))
    unique_undirected = sorted(list(set(unique_undirected)))

    np.random.seed(42)
    adjs = np.zeros((T, N, N), dtype=np.int32)
    attmats = np.zeros((N, T, F_dim), dtype=np.float64)

    # Assign temporal birth and lifespan to edges with realistic persistence
    for u, v in unique_undirected:
        t_birth = np.random.randint(0, T - 2)
        lifespan = np.random.randint(3, 7)
        for t in range(t_birth, min(T, t_birth + lifespan)):
            adjs[t, u, v] = 1
            adjs[t, v, u] = 1

    for t in range(T):
        edge_cnt = np.sum(adjs[t]) // 2
        print(f"  Snapshot t = {t}: {edge_cnt} undirected edges ({np.sum(adjs[t])} directed)")

    # Repeat features across snapshots with subtle temporal variation (drift)
    for t in range(T):
        temporal_scale = 1.0 + 0.05 * np.sin(t * np.pi / 5.0)
        attmats[:, t, :] = feats_static * temporal_scale

    keep_nodes = np.arange(N, dtype=np.int64)

    # 7. Save to npz
    np.savez_compressed(
        out_path,
        attmats=attmats,
        adjs=adjs,
        labels=labels_onehot,
        keep_nodes=keep_nodes
    )
    print(f"Successfully saved dynamic Citeseer dataset to {out_path}!")
    print(f"Data shapes: adjs={adjs.shape}, attmats={attmats.shape}, labels={labels_onehot.shape}")


if __name__ == "__main__":
    build_dynamic_citeseer()
