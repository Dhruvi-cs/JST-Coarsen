"""
Clean Preprocessing for Dynamic Citeseer Benchmark (Zero Data Leakage)
- Raw Source: Official Planetoid CiteSeer dataset (Yang et al., 2016)
- Node Features: 100-dim TruncatedSVD fitted strictly on the text word matrix X (3,703 bag-of-words dictionary), completely independent of graph topology.
- Number of snapshots: T = 10
- Classes: C = 6
- Nodes: N = 2,120 (Largest Connected Component)
- Standard Semi-Supervised Split: Planetoid official split (20 labels per class)
- Output: src/Citeseer/Citeseer_no_isolated.npz
"""

import os
import numpy as np
import scipy.sparse as sp
import networkx as nx
from torch_geometric.datasets import Planetoid
from sklearn.decomposition import TruncatedSVD


def build_clean_dynamic_citeseer():
    out_dir = os.path.join("src", "Citeseer")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "Citeseer_no_isolated.npz")

    print(f"Loading official Planetoid CiteSeer dataset...")
    dataset = Planetoid(root=os.path.join("src", "Planetoid"), name="CiteSeer")
    data = dataset[0]

    # 1. Authentic document text features (strictly from text word matrix X)
    X_raw = data.x.numpy()  # (3327, 3703) text bag-of-words
    print(f"Raw document text vocabulary matrix: {X_raw.shape}")

    # Dimensionality reduction to F=100 strictly on text words
    svd = TruncatedSVD(n_components=100, random_state=42)
    feats_100 = svd.fit_transform(X_raw)
    row_norms = np.linalg.norm(feats_100, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    feats_100 = (feats_100 / row_norms).astype(np.float32)
    print(f"Extracted clean document text features: {feats_100.shape}")

    # 2. Build graph and extract Largest Connected Component
    G = nx.Graph()
    for i in range(data.num_nodes):
        G.add_node(i)
    for u, v in data.edge_index.t().numpy():
        if u != v:
            G.add_edge(u, v)

    largest_cc = max(nx.connected_components(G), key=len)
    final_nodes = sorted(list(largest_cc))
    N = len(final_nodes)
    node_to_idx = {p: i for i, p in enumerate(final_nodes)}
    y_all = data.y.numpy()

    labels_int = np.array([y_all[p] for p in final_nodes], dtype=np.int64)
    C = 6
    labels_onehot = np.zeros((N, C), dtype=np.float32)
    for i, c in enumerate(labels_int):
        labels_onehot[i, c] = 1.0

    feats_clean = feats_100[final_nodes]

    # Map Planetoid splits to connected component
    train_mask = np.array([data.train_mask[p].item() for p in final_nodes], dtype=bool)
    val_mask = np.array([data.val_mask[p].item() for p in final_nodes], dtype=bool)
    test_mask = np.array([data.test_mask[p].item() for p in final_nodes], dtype=bool)

    print(f"Clean Benchmark Graph: N = {N} nodes, {G.subgraph(final_nodes).number_of_edges()} unique edges")
    print(f"Planetoid Split: Train = {np.sum(train_mask)}, Val = {np.sum(val_mask)}, Test = {np.sum(test_mask)}")

    # 3. Dynamic snapshot sequence (T = 10)
    unique_undirected = set()
    for u, v in G.subgraph(final_nodes).edges():
        unique_undirected.add((min(node_to_idx[u], node_to_idx[v]), max(node_to_idx[u], node_to_idx[v])))
    unique_undirected = sorted(list(unique_undirected))

    T = 10
    np.random.seed(42)
    adjs = np.zeros((T, N, N), dtype=np.int32)
    attmats = np.zeros((N, T, 100), dtype=np.float32)

    # Assign temporal birth and lifespan
    for u, v in unique_undirected:
        t_birth = np.random.randint(0, T - 2)
        lifespan = np.random.randint(3, 7)
        for t in range(t_birth, min(T, t_birth + lifespan)):
            adjs[t, u, v] = 1
            adjs[t, v, u] = 1

    for t in range(T):
        temporal_scale = 1.0 + 0.05 * np.sin(t * np.pi / 5.0)
        attmats[:, t, :] = feats_clean * temporal_scale
        edge_cnt = np.sum(adjs[t]) // 2
        print(f"  Snapshot t = {t}: {edge_cnt} undirected edges")

    keep_nodes = np.arange(N, dtype=np.int64)

    # 4. Save clean dataset
    np.savez_compressed(
        out_path,
        attmats=attmats,
        adjs=adjs,
        labels=labels_onehot,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        keep_nodes=keep_nodes
    )
    print(f"Clean dynamic Citeseer dataset saved to {out_path}!")

    # Copy to package as well
    pkg_path = os.path.join("citeseer_coarsening_package", "src", "Citeseer", "Citeseer_no_isolated.npz")
    os.makedirs(os.path.dirname(pkg_path), exist_ok=True)
    import shutil
    shutil.copy2(out_path, pkg_path)
    print(f"Also copied to {pkg_path}")


if __name__ == "__main__":
    build_clean_dynamic_citeseer()
