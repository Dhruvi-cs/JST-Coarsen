import numpy as onp
import argparse
from pathlib import Path 
import networkx as nx

def remove_isolated_nodes(
    raw_node_feats,
    raw_node_labels,
    raw_edge_srcs,
    raw_edge_dsts,
    num_times,
):
    """
    Remove nodes that are isolated throughout all snapshots.
    Updates node features, labels, and edge lists accordingly.
    """
    num_nodes = raw_node_feats.shape[0]
    connected = onp.zeros(num_nodes, dtype=bool)
    for t in range(num_times):
        srcs = raw_edge_srcs[t]
        dsts = raw_edge_dsts[t]
        # remove self-loops
        mask_noloop = srcs != dsts
        srcs = srcs[mask_noloop]
        dsts = dsts[mask_noloop]
        #
        connected[srcs] = True
        connected[dsts] = True
        
    keep_nodes = onp.where(connected)[0]

    # Update node features and labels
    raw_node_feats = raw_node_feats[keep_nodes]
    if isinstance(raw_node_labels, onp.ndarray):
        raw_node_labels = raw_node_labels[keep_nodes]
    else:
        raw_node_labels = [raw_node_labels[i] for i in keep_nodes]
        
    # Remap node indices in edge lists
    node_map = {old: new for new, old in enumerate(keep_nodes)}
    for t in range(num_times):
        srcs = raw_edge_srcs[t]
        dsts = raw_edge_dsts[t]
        mask = onp.isin(srcs, keep_nodes) & onp.isin(dsts, keep_nodes)
        srcs = srcs[mask]
        dsts = dsts[mask]
        # Remap indices
        raw_edge_srcs[t] = onp.array([node_map[s] for s in srcs])
        raw_edge_dsts[t] = onp.array([node_map[d] for d in dsts])
        # Edge feats remain unchanged
    
    return raw_node_feats, raw_node_labels, raw_edge_srcs, raw_edge_dsts, keep_nodes

def select_largest_connected_component(
    raw_node_feats,
    raw_node_labels,
    raw_edge_srcs,
    raw_edge_dsts,
    num_times,
):
    """
    Select the largest connected component across all snapshots.
    Updates node features, labels, and edge lists accordingly.
    """

    G = nx.Graph()
    num_nodes = raw_node_feats.shape[0]
    G.add_nodes_from(range(num_nodes))
    
    for t in range(num_times):
        srcs = raw_edge_srcs[t]
        dsts = raw_edge_dsts[t]
        edges = [(s, d) for s, d in zip(srcs, dsts) if s != d]  # ignore self-loops
        G.add_edges_from(edges)
    
    largest_cc = max(nx.connected_components(G), key=len)
    keep_nodes = onp.array(sorted(largest_cc))

    # Update node features and labels
    raw_node_feats = raw_node_feats[keep_nodes]
    if isinstance(raw_node_labels, onp.ndarray):
        raw_node_labels = raw_node_labels[keep_nodes]
    else:
        raw_node_labels = [raw_node_labels[i] for i in keep_nodes]
        
    # Remap node indices in edge lists
    node_map = {old: new for new, old in enumerate(keep_nodes)}
    for t in range(num_times):
        srcs = raw_edge_srcs[t]
        dsts = raw_edge_dsts[t]
        mask = onp.isin(srcs, keep_nodes) & onp.isin(dsts, keep_nodes)
        srcs = srcs[mask]
        dsts = dsts[mask]
        # Remap indices
        raw_edge_srcs[t] = onp.array([node_map[s] for s in srcs])
        raw_edge_dsts[t] = onp.array([node_map[d] for d in dsts])
    
    return raw_node_feats, raw_node_labels, raw_edge_srcs, raw_edge_dsts, keep_nodes

def subsample_dataset(
    raw_node_feats,
    raw_node_labels,
    raw_edge_srcs,
    raw_edge_dsts,
    num_times,
    ss_size: int = 50,
):
    """
    Subsample dataset by top k nodes with highest degree.
    """ 
    num_nodes = raw_node_feats.shape[0]
    # Compute node degrees across all time steps.
    degrees = onp.zeros((num_nodes,), dtype=onp.int32)

    for t in range(num_times):
        srcs = raw_edge_srcs[t]
        dsts = raw_edge_dsts[t]
        
        for src, dst in zip(srcs, dsts):
            if src != dst:  # ignore self-loops
                degrees[src] += 1
                degrees[dst] += 1
                
    # Get top k nodes with highest degree.
    topk_indices = onp.argsort(-degrees)[:ss_size]
    new_node_id_map = {old_id: new_id for new_id, old_id 
                       in enumerate(topk_indices.tolist())}
    # Subsample node features and labels.
    ss_node_feats = raw_node_feats[topk_indices]
    ss_node_labels = raw_node_labels[topk_indices]
    # Subsample edges.
    ss_edge_srcs = []
    ss_edge_dsts = []
    for t in range(num_times):
        srcs = raw_edge_srcs[t]
        dsts = raw_edge_dsts[t]
        mask = onp.isin(srcs, topk_indices) & onp.isin(dsts, topk_indices)
        filtered_srcs = srcs[mask]
        filtered_dsts = dsts[mask]
        # Remap node IDs.
        remapped_srcs = onp.array([new_node_id_map[src] for src in filtered_srcs], dtype=onp.int32)
        remapped_dsts = onp.array([new_node_id_map[dst] for dst in filtered_dsts], dtype=onp.int32)
        ss_edge_srcs.append(remapped_srcs)
        ss_edge_dsts.append(remapped_dsts)
    
    return ss_node_feats, ss_node_labels, ss_edge_srcs, ss_edge_dsts

def subsample_dataset_balanced(
    raw_node_feats,
    raw_node_labels,
    raw_edge_srcs,
    raw_edge_dsts,
    num_times,
    ss_size: int = 50,
):
    """
    Subsample dataset by selecting top k nodes with highest degree per class.
    """ 
    # Compute node degrees across all time steps.
    degrees = onp.zeros((num_nodes,), dtype=onp.int32)
    for t in range(num_times):
        srcs = raw_edge_srcs[t]
        dsts = raw_edge_dsts[t]
        for src, dst in zip(srcs, dsts):
            if src != dst:  # ignore self-loops
                degrees[src] += 1
                degrees[dst] += 1

    # Resolve labels into a 1D array of label ids
    if isinstance(raw_node_labels, onp.ndarray):
        if raw_node_labels.ndim == 2:
            labels = onp.argmax(raw_node_labels, axis=1)
        else:
            labels = raw_node_labels
    else:
        labels = onp.array(raw_node_labels)

    unique_labels = onp.unique(labels)
    num_classes = unique_labels.shape[0]

    # Desired per-class quotas (distribute remainder to first classes)
    base = ss_size // num_classes
    rem = ss_size % num_classes
    quota = {int(lbl): base + (1 if i < rem else 0) for i, lbl in enumerate(unique_labels)}

    selected = []
    # Pick top-degree nodes within each class up to quota
    for lbl in unique_labels:
        lbl = int(lbl)
        class_idx = onp.where(labels == lbl)[0]
        if class_idx.size == 0:
            continue
        # sort class indices by degree descending
        order = class_idx[onp.argsort(-degrees[class_idx])]
        take = min(quota[lbl], order.shape[0])
        selected.extend(order[:take].tolist())

    # If not enough selected (some classes had too few nodes), fill remaining
    if len(selected) < ss_size:
        remaining_pool = onp.setdiff1d(onp.arange(num_nodes), onp.array(selected))
        remaining_order = remaining_pool[onp.argsort(-degrees[remaining_pool])]
        need = ss_size - len(selected)
        selected.extend(remaining_order[:need].tolist())

    # Ensure exact size and stable numpy array
    selected = onp.array(selected[:ss_size], dtype=onp.int32)

    # Build new id map and subsample features/labels
    new_node_id_map = {int(old_id): new_id for new_id, old_id in enumerate(selected.tolist())}
    ss_node_feats = raw_node_feats[selected]
    if isinstance(raw_node_labels, onp.ndarray):
        ss_node_labels = raw_node_labels[selected]
    else:
        ss_node_labels = [raw_node_labels[int(i)] for i in selected]

    # Subsample and remap edges for each time step
    ss_edge_srcs = []
    ss_edge_dsts = []
    for t in range(num_times):
        srcs = raw_edge_srcs[t]
        dsts = raw_edge_dsts[t]
        mask = onp.isin(srcs, selected) & onp.isin(dsts, selected)
        filtered_srcs = srcs[mask]
        filtered_dsts = dsts[mask]
        remapped_srcs = onp.array([new_node_id_map[int(s)] for s in filtered_srcs], dtype=onp.int32)
        remapped_dsts = onp.array([new_node_id_map[int(d)] for d in filtered_dsts], dtype=onp.int32)
        ss_edge_srcs.append(remapped_srcs)
        ss_edge_dsts.append(remapped_dsts)

    return ss_node_feats, ss_node_labels, ss_edge_srcs, ss_edge_dsts

def remove_lowsample_labels(
    raw_node_feats,
    raw_node_labels,
    raw_edge_srcs,
    raw_edge_dsts,
    num_times,
    min_samples: int = 10,
):
    """
    Remove nodes with labels that have less than min_samples.
    Updates node features, labels, and edge lists accordingly.
    """
    if isinstance(raw_node_labels, onp.ndarray):
        unique, counts = onp.unique(onp.argmax(raw_node_labels, axis=1), return_counts=True)
        label_counts = dict(zip(unique.tolist(), counts.tolist()))
        valid_labels = {lbl for lbl, cnt in label_counts.items() if cnt >= min_samples}
        keep_nodes = onp.array([i for i, lbl in enumerate(onp.argmax(raw_node_labels, axis=1)) if lbl in valid_labels])
    else:
        unique, counts = onp.unique(onp.array(raw_node_labels), return_counts=True)
        label_counts = dict(zip(unique.tolist(), counts.tolist()))
        valid_labels = {lbl for lbl, cnt in label_counts.items() if cnt >= min_samples}
        keep_nodes = onp.array([i for i, lbl in enumerate(raw_node_labels) if lbl in valid_labels])
    
    # Update node features and labels
    raw_node_feats = raw_node_feats[keep_nodes]
    if isinstance(raw_node_labels, onp.ndarray):
        raw_node_labels = raw_node_labels[keep_nodes]
    else:
        raw_node_labels = [raw_node_labels[i] for i in keep_nodes]
        
    # Remap node indices in edge lists
    node_map = {old: new for new, old in enumerate(keep_nodes)}
    for t in range(num_times):
        srcs = raw_edge_srcs[t]
        dsts = raw_edge_dsts[t]
        mask = onp.isin(srcs, keep_nodes) & onp.isin(dsts, keep_nodes)
        srcs = srcs[mask]
        dsts = dsts[mask]
        # Remap indices
        raw_edge_srcs[t] = onp.array([node_map[s] for s in srcs])
        raw_edge_dsts[t] = onp.array([node_map[d] for d in dsts])
    
    return raw_node_feats, raw_node_labels, raw_edge_srcs, raw_edge_dsts, keep_nodes

def from_raw(filepath: str, /) -> None:
    R"""
    Load from raw data.
    from `main_src/lib/data/dynclass.py`
    """
    #
    matrices = onp.load(filepath)
    feats = matrices["attmats"]
    adjmats = matrices["adjs"]
    onehots = matrices["labels"]

    # Safety check.
    if onp.any(adjmats < 0):
        # UNEXPECT:
        # Edge weights must be non-negative.
        raise NotImplementedError("Get invalid negative edge weights.")
    if onp.any(onp.sum(onehots > 0, axis=1) != 1):
        # UNEXPECT:
        # Node label must be unique.
        raise NotImplementedError("Get empty or duplicate node labels.")

    #
    raw_edge_srcs = []
    raw_edge_dsts = []
    raw_edge_feats = []
    (_, num_times, _) = feats.shape
    for t in range(num_times):
        #
        (dsts, srcs) = onp.nonzero(adjmats[t])
        weights = adjmats[t, dsts, srcs].astype(float)
        if onp.any(weights != 1):
            # UNEXPECT:
            # Node label must be unique.
            raise NotImplementedError("Edge has non-0/1 weight.")
        raw_edge_srcs.append(srcs)
        raw_edge_dsts.append(dsts)
        raw_edge_feats.append(weights)
        
    return adjmats, feats, onehots, raw_edge_srcs, raw_edge_dsts
    
def print_and_relabel_counts(labels):
    if isinstance(labels, onp.ndarray):
        if onp.all(onp.sum(labels > 0, axis=1) == 1):
            (_, num_labels) = labels.shape
            counts = onp.sum(labels, axis=0).tolist()
        else:
            unique, counts = onp.unique(labels, return_counts=True)
            counts = dict(zip(unique.tolist(), counts.tolist()))
    else:
        unique, counts = onp.unique(onp.array(labels), return_counts=True)
        counts = dict(zip(unique.tolist(), counts.tolist()))
    
    print(f"Label counts: {counts}")
    # shrink to only non-zero counts, return new onehots
    if isinstance(labels, onp.ndarray):
        nonzero_labels = [i for i in range(len(counts)) if counts[i] > 0]
        new_onehots = onp.zeros((labels.shape[0], len(nonzero_labels)), dtype=labels.dtype)
        for new_idx, old_idx in enumerate(nonzero_labels):
            new_onehots[:, new_idx] = labels[:, old_idx]
        return new_onehots
    else:
        label_map = {old: new for new, old in enumerate(sorted(counts.keys()))}
        new_labels = [label_map[lbl] for lbl in labels]
        return new_labels
    
    
    
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--npz_file", type=str, default="../../src/DBLP5/DBLP5_no_isolated.npz")
    args.add_argument("--ss_size", type=int, default=200)
    
    args = args.parse_args()
    
    ss_size = args.ss_size
    
    if not Path(args.npz_file).exists():
        raise ValueError(f"File {args.npz_file} does not exist.")
    
    save_path = Path(args.npz_file).parent / f"{Path(args.npz_file).stem}_ss{ss_size}.npz"
    
    adjmats, feats, onehots, raw_edge_srcs, raw_edge_dsts = from_raw(args.npz_file)
    num_times = adjmats.shape[0]
    num_nodes = feats.shape[0]
    print(f"Original num_nodes: {num_nodes}, num_times: {num_times}")
    print(f"feats.shape: {feats.shape}, onehots.shape: {onehots.shape}")
    print(f"edge_srcs[0]: {raw_edge_srcs[0]}")
    print(f"edge_dsts[0]: {raw_edge_dsts[0]}")
    
    # Remove isolated nodes
    ss_up_size = ss_size
    while True:
        ss_feats, ss_onehots, ss_edge_srcs, ss_edge_dsts = subsample_dataset_balanced(
            feats, onehots, raw_edge_srcs, raw_edge_dsts, num_times, ss_size=ss_up_size
        )
        (
            ri_feats, ri_onehots, ri_edge_srcs, 
            ri_edge_dsts, ri_keep_nodes 
        )= select_largest_connected_component(
            ss_feats, ss_onehots, ss_edge_srcs, 
            ss_edge_dsts, num_times
        )
        if ri_feats.shape[0] >= ss_size:
            feats = ri_feats
            onehots = ri_onehots
            edge_srcs = ri_edge_srcs
            edge_dsts = ri_edge_dsts
            break
        else:
            ss_up_size += 100
            print(f"Not enough nodes after selecting largest connected component. "
                  f"Increasing ss_size to {ss_up_size} and retrying...")
            
    num_nodes = feats.shape[0]
    num_edges = [len(es) for es in edge_srcs]
    check_pairs = set()
    for t in range(num_times):
        srcs = edge_srcs[t]
        dsts = edge_dsts[t]
        # find all unique pairs of nodes that are connected by an edge
        edges = onp.unique(onp.stack((srcs, dsts), axis=1), axis=0)
        for (u, v) in edges:
            if u != v and (v, u) not in check_pairs: 
                check_pairs.add((u, v))

    check_pairs = list(check_pairs)
    print(f"After subsampling to top {ss_size} nodes with highest degree,"
          f" then removing isolated nodes across time:")
    print(f"New num_nodes: {num_nodes}, num_times: {num_times}")
    print(f"feats.shape: {feats.shape}, onehots.shape: {onehots.shape}")
    print(f"Number of edges per time step: {num_edges}")
    print(f"Number of unique undirected edges across all time steps: {len(check_pairs)}")
    
    onehots = print_and_relabel_counts(onehots)
    print(f"After relabeling, onehots.shape: {onehots.shape}")
    # relable again and assert the new onehots equals the old onehots
    assert_onehots = print_and_relabel_counts(onehots)
    assert onp.array_equal(onehots, assert_onehots), "Relabeling changed the onehots unexpectedly."

    # reconstruct the adjacency matrices
    adjmats = onp.zeros((num_times, num_nodes, num_nodes), dtype=adjmats.dtype)
    for t in range(num_times):
        srcs = edge_srcs[t]
        dsts = edge_dsts[t]
        adjmats[t, srcs, dsts] = 1
        
    onp.savez_compressed(
        save_path,
        attmats=feats,
        adjs=adjmats,
        labels=onehots
    )
    print(f"Processed data saved to {save_path}")