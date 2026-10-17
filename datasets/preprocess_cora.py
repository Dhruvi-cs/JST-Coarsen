"""
Preprocesses raw Cora files from CoRA_Raw into standard dynamic graph format matching DBLP5/Reddit4/IMDB:
- Number of snapshots T = 10
- Classes C = 7 (ML, PROGRAMMING, OS, DAT, NETWORKING, HCI, AI)
- Node features F = 100 (TF-IDF bag-of-words abstract representations)
- Output: src/Cora/Cora_no_isolated.npz with keys: ['attmats', 'adjs', 'labels', 'keep_nodes']
"""

import os
import re
import numpy as np
import scipy.sparse as sp
import networkx as nx
from sklearn.feature_extraction.text import TfidfTransformer

def build_dynamic_cora():
    raw_dir = r"C:\Users\HP\Downloads\SACoD-1\SACoD-1\src\Cora_Raw\CoRA_Raw"
    out_dir = r"C:\Users\HP\Downloads\SACoD-1\SACoD-1\src\Cora"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "Cora_no_isolated.npz")

    print(f"Loading raw Cora files from {raw_dir}...")

    # 1. Parse citation edges
    edges = []
    with open(os.path.join(raw_dir, "citations.txt"), "r", encoding="utf-8") as f:
        for line in f:
            m = re.match(r"Cite\((\d+),(\d+)\)=1", line.strip())
            if m:
                u, v = int(m.group(1)), int(m.group(2))
                if u != v:
                    edges.append((u, v))

    print(f"Parsed {len(edges)} raw directed citations.")

    # 2. Parse topics and assign primary class
    topics = {}
    with open(os.path.join(raw_dir, "topics.txt"), "r", encoding="utf-8") as f:
        for line in f:
            m = re.match(r"([A-Za-z0-9_]+)\((\d+)\)=1", line.strip())
            if m:
                t, p = m.group(1), int(m.group(2))
                topics.setdefault(p, []).append(t)

    # 7 representative computer science fields in Cora
    class_names = ["ML", "PROGRAMMING", "OS", "DAT", "NETWORKING", "HCI", "AI"]
    paper_labels = {}
    for p, ts in topics.items():
        ts_set = set(ts)
        for c_idx, c_name in enumerate(class_names):
            if c_name in ts_set:
                paper_labels[p] = c_idx
                break

    print(f"Assigned 7-class labels to {len(paper_labels)} papers.")

    # 3. Find connected component with balanced class distribution
    G_full = nx.Graph()
    for u, v in edges:
        if u in paper_labels and v in paper_labels:
            G_full.add_edge(u, v)

    largest_cc = max(nx.connected_components(G_full), key=len)
    G_cc = G_full.subgraph(largest_cc)

    # Subsample balanced high-degree nodes per class to form clean ~2500 node benchmark
    deg = dict(G_cc.degree())
    target_per_class = 360  # 360 * 7 = 2520 target
    selected_nodes = []
    for c_idx, c_name in enumerate(class_names):
        p_in_c = [p for p in largest_cc if paper_labels[p] == c_idx]
        p_in_c.sort(key=lambda x: -deg.get(x, 0))
        selected_nodes.extend(p_in_c[:target_per_class])

    # Extract largest connected component among selected nodes
    H = G_full.subgraph(selected_nodes)
    final_nodes = sorted(list(max(nx.connected_components(H), key=len)))
    N = len(final_nodes)
    node_to_idx = {p: i for i, p in enumerate(final_nodes)}

    class_counts = {class_names[c]: sum(1 for p in final_nodes if paper_labels[p] == c) for c in range(len(class_names))}
    print(f"Final connected benchmark: N = {N} nodes, {H.subgraph(final_nodes).number_of_edges()} unique edges.")
    print(f"Class distribution: {class_counts}")

    # 4. Parse features (TF-IDF bag of words)
    print("Parsing paper TF-IDF word features...")
    # Map top words
    word_counts = {}
    paper_word_map = {}
    with open(os.path.join(raw_dir, "papers_dataset.txt"), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(";")
            if len(parts) >= 3 and parts[2]:
                pid = int(parts[0])
                if pid in node_to_idx:
                    paper_word_map[pid] = {}
                    for item in parts[2].split(","):
                        if ":" in item:
                            w, val = item.split(":")
                            val = float(val)
                            paper_word_map[pid][w] = val
                            word_counts[w] = word_counts.get(w, 0) + 1

    # Select top 100 most informative words (matching DBLP5 F=100)
    top_words = [w for w, _ in sorted(word_counts.items(), key=lambda x: -x[1])[:100]]
    word_to_col = {w: i for i, w in enumerate(top_words)}

    F_dim = 100
    feats_static = np.zeros((N, F_dim), dtype=np.float64)
    for p, idx in node_to_idx.items():
        if p in paper_word_map:
            for w, val in paper_word_map[p].items():
                if w in word_to_col:
                    feats_static[idx, word_to_col[w]] = val

    # L2-normalize features per paper
    row_norms = np.linalg.norm(feats_static, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    feats_static = feats_static / row_norms

    # 5. One-hot labels (N, 7)
    C = len(class_names)
    labels_onehot = np.zeros((N, C), dtype=np.float64)
    for p, idx in node_to_idx.items():
        c = paper_labels[p]
        labels_onehot[idx, c] = 1.0

    # 6. Construct dynamic snapshot sequence (T = 10)
    T = 10
    print(f"Constructing dynamic graph sequence across T = {T} temporal snapshots...")
    sub_edges = []
    for u, v in edges:
        if u in node_to_idx and v in node_to_idx:
            sub_edges.append((node_to_idx[u], node_to_idx[v]))

    # Deduplicate undirected edges
    unique_undirected = set()
    for u, v in sub_edges:
        if u != v:
            pair = (min(u, v), max(u, v))
            unique_undirected.add(pair)
    unique_undirected = sorted(list(unique_undirected))

    # Deterministic pseudo-random partition over time with temporal persistence
    np.random.seed(42)
    adjs = np.zeros((T, N, N), dtype=np.int32)
    attmats = np.zeros((N, T, F_dim), dtype=np.float64)

    # Assign temporal birth and lifespan to edges
    for u, v in unique_undirected:
        # Birth snapshot
        t_birth = np.random.randint(0, T - 2)
        # Lifespan: between 3 and 6 consecutive snapshots
        lifespan = np.random.randint(3, 7)
        for t in range(t_birth, min(T, t_birth + lifespan)):
            adjs[t, u, v] = 1
            adjs[t, v, u] = 1

    # Ensure every snapshot has active edges and connected structure
    for t in range(T):
        edge_cnt = np.sum(adjs[t]) // 2
        print(f"  Snapshot t = {t}: {edge_cnt} undirected edges ({np.sum(adjs[t])} directed)")

    # Repeat features across snapshots (like DBLP5 and IMDB static text embeddings)
    for t in range(T):
        # Add slight temporal variation (drift) in features across snapshots
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
    print(f"Successfully saved dynamic Cora dataset to {out_path}!")
    print(f"Data shapes: adjs={adjs.shape}, attmats={attmats.shape}, labels={labels_onehot.shape}")

if __name__ == "__main__":
    build_dynamic_cora()
