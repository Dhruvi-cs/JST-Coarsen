import numpy as onp
import argparse
from pathlib import Path 

def remove_recent_snapshots(
    raw_node_feats,
    raw_node_labels,
    raw_edge_srcs,
    raw_edge_dsts,
    num_times,
    num_remove,
):
    """
    Remove the last num_remove snapshots.
    Updates node features, labels, and edge lists accordingly.
    """
    if num_remove >= num_times:
        raise ValueError("num_remove must be less than num_times.")
    
    new_num_times = num_times - num_remove
    raw_edge_srcs = raw_edge_srcs[:new_num_times]
    raw_edge_dsts = raw_edge_dsts[:new_num_times]
    
    return raw_node_feats, raw_node_labels, raw_edge_srcs, raw_edge_dsts, new_num_times


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

    keep_nodes = onp.sort(keep_nodes)
    print(f"Removed {num_nodes - len(keep_nodes)} isolated nodes.")
    
    if len(keep_nodes) == num_nodes:
        print("No isolated nodes found.")
        return raw_node_feats, raw_node_labels, raw_edge_srcs, raw_edge_dsts, keep_nodes
    
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

    num_nodes = len(keep_nodes)
    # assert number of unique labels remains the same
    if isinstance(raw_node_labels, onp.ndarray):
        isonehot = onp.all(onp.sum(raw_node_labels > 0, axis=1) == 1)
        if isonehot:
            (_, num_labels) = raw_node_labels.shape
            label_counts = onp.sum(raw_node_labels, axis=0)
            assert onp.sum(label_counts > 0) == num_labels, "Some labels are missing after removing isolated nodes."
        else:
            unique, counts = onp.unique(raw_node_labels, return_counts=True)
            assert len(unique) == max(unique) + 1, "Some labels are missing after removing isolated nodes."
    else:
        unique, counts = onp.unique(onp.array(raw_node_labels), return_counts=True)
        assert len(unique) == max(unique) + 1, "Some labels are missing after removing isolated nodes."
    
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
    #
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
    


### Utility functions for IMDB dataset processing ###
def to_onehot(labels, num_classes):
    onehots = onp.zeros((len(labels), num_classes), dtype=onp.float32)
    for i in range(len(labels)):
        if labels[i] >= 0:
            onehots[i, labels[i]] = 1.0
    return onehots

def get_general_info(dir):
        # step 0: get basic info
        with open(dir + "/graph.info", "r") as f:
            num_tws = int(f.readline())
            num_nodes = int(f.readline())
            num_attrs = int(f.readline())
        return num_tws, num_nodes, num_attrs

def get_labels(dir, num_nodes, default_lab_val=-1):
    with open(dir + "/graph.lab", "r") as f:
        l_labeled_nd = []
        y = onp.zeros([num_nodes], dtype=onp.int64)
        for line in f.readlines():  # assign labels
            token = line.split("::")
            nd = int(token[0])
            lb = int(token[1])
            l_labeled_nd.append(nd)
            y[nd] = lb

        for nd in range(num_nodes):  # put indicators for nodes who do not have labels
            if nd not in l_labeled_nd:
                y[nd] = default_lab_val
    return y, l_labeled_nd

def get_adjmat_from_file(dir, num_nodes, num_times, nodeval2indices=None):
    all_time_vals = set()
    edgeinfo_avail_nodes = set()
    with open(dir + "/graph.tedgelist", "r") as f:
        for line in f.readlines():
            token = line.split("::")
            t = int(token[2])
            s = int(token[0])
            d = int(token[1])
            all_time_vals.add(t)
            edgeinfo_avail_nodes.add(s)
            edgeinfo_avail_nodes.add(d)
    all_time_vals = sorted(all_time_vals)
    # all_node_vals = sorted(all_node_vals)
    timeval2indices = {t: idx for idx, t in enumerate(all_time_vals)}
    # nodeval2indices = {n: idx for idx, n in enumerate(all_node_vals)}
    assert len(timeval2indices) == num_times, 'Inconsistencies found!'
    # assert len(nodeval2indices) == num_nodes, 'Inconsistancies found!' # no need for nodes

    adjmats = onp.zeros((num_times, num_nodes, num_nodes))

    with open(dir + "/graph.tedgelist", "r") as f:
        for line in f.readlines():
            token = line.split("::")
            t = int(token[2])
            s = int(token[0])
            d = int(token[1])
            t = timeval2indices[t]
            # s = nodeval2indices[s]
            # d = nodeval2indices[d]
            adjmats[t, s, d] = 1

    return adjmats, timeval2indices, edgeinfo_avail_nodes

def get_node_static_attrs(dir, num_nodes, num_node_feat):
    with open(dir + "/graph.attr", "r") as f:
        l_attr_nd = set()
        x = onp.zeros([num_nodes, num_node_feat])
        for line in f.readlines():  # assign labels
            token = line.split("::")
            nd = int(token[0])
            attr = token[1:]
            attr = [int(float(a)) for a in attr]
            x[nd] = attr
            l_attr_nd.add(nd)

        for nd in range(num_nodes):  # put indicators for nodes who do not have labels
            if nd not in l_attr_nd:
                x[nd] = [-1] * num_node_feat
    return x, l_attr_nd


def from_raw_imdb( dirname: str) -> None:
    R"""
    Load from raw data.
    """
    #
    num_times, num_nodes, num_node_attrs = get_general_info(dirname)
    num_labels = 2
    #
    feats, feat_avail_nodes = get_node_static_attrs(dirname, num_nodes, num_node_attrs) # (num_nodes, num_feats)
    print(f'Missing feat nodes: {sum(onp.alltrue(feats == -1, axis=1))}')

    # feats = torch.tensor(feats).unsqueeze(1).repeat(1, num_times, 1).numpy()
    adjmats, timeval2indices, edgeinfo_avail_nodes = \
        get_adjmat_from_file(dirname, num_nodes, num_times)
    
    raw_node_feats = feats
    raw_node_labels, lab_avail_nodes = get_labels(dirname, num_nodes, default_lab_val=-1)
    assert len(set(lab_avail_nodes).intersection(set(feat_avail_nodes))) == len(feat_avail_nodes)
    miss_lb_nodes = onp.where(raw_node_labels == -1)[0]
    print(f'Missing-label nodes: {len(miss_lb_nodes)}/{num_nodes}')
    avail_node_missing_lb = set(miss_lb_nodes).intersection(set(edgeinfo_avail_nodes))
    print(f'Edgeinfo avail nodes but missing label: {len(avail_node_missing_lb)}')

    num_labels = num_labels
    label_counts = [sum(raw_node_labels == i) for i in range(num_labels)]
    lb_cnts = onp.array(label_counts)
    print(f'label counts: {label_counts} ({[f"{v:.2f}%" for v in lb_cnts / lb_cnts.sum()]})')
    #
    raw_edge_srcs = []
    raw_edge_dsts = []
    raw_edge_feats = []

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
    return num_times, num_nodes, raw_node_feats, raw_node_labels, num_labels, raw_edge_srcs, raw_edge_dsts

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--npz_file", type=str, default="data/dynclass/")
    args.add_argument('--source', type=str, default='IMDB')
    args = args.parse_args()
    
    source = args.source 
    if source in ['Reddit4', 'DBLP5']:
        if not Path(args.npz_file).exists():
            raise ValueError(f"File {args.npz_file} does not exist.")
        save_path = Path(args.npz_file).parent / f"{Path(args.npz_file).stem}_no_isolated.npz"
        (
            num_times, num_nodes, feats, 
            onehots, raw_edge_srcs, raw_edge_dsts
        ) = from_raw(args.npz_file)
    elif source == 'IMDB':
        dirname = args.npz_file
        save_path = Path(dirname) / f"{source}_no_isolated.npz"
        num_times, num_nodes, feats, raw_node_labels, \
            num_labels, raw_edge_srcs, raw_edge_dsts = from_raw_imdb(dirname)

        onehots = to_onehot(raw_node_labels, num_labels)

    print(f"Original num_nodes: {num_nodes}, num_times: {num_times}")
    print(f"feats.shape: {feats.shape}, onehots.shape: {onehots.shape}")
    print(f"edge_srcs[0]: {raw_edge_srcs[0]}")
    print(f"edge_dsts[0]: {raw_edge_dsts[0]}")
    
    # Remove isolated nodes
    feats, onehots, edge_srcs, edge_dsts, keep_nodes = remove_isolated_nodes(
        feats, onehots, raw_edge_srcs, raw_edge_dsts, num_times
    )
    
    if source == 'IMDB':
        feats, onehots, edge_srcs, edge_dsts, num_times = remove_recent_snapshots(
            feats, onehots, edge_srcs, edge_dsts, num_times, num_remove=14
        )

    num_nodes = feats.shape[0]
    print(f"After removing isolated nodes, num_nodes: {num_nodes}")
    print(f"feats.shape: {feats.shape}, onehots.shape: {onehots.shape}")
    print(f"edge_srcs[0]: {edge_srcs[0]}")
    print(f"edge_dsts[0]: {edge_dsts[0]}")
    
    def print_label_counts(labels):
        # one-hot encoded (fully labeled)
        (_, num_labels) = labels.shape
        counts = onp.sum(labels, axis=0).tolist()
        missing_cnt = labels.shape[0] - onp.sum(counts)
        print(f"Label counts: {counts}")
        if missing_cnt > 0:
            print(f"Missing label count: {missing_cnt}")

    print_label_counts(onehots)

    # reconstruct the adjacency matrices
    adjmats = onp.zeros((num_times, num_nodes, num_nodes))
    for t in range(num_times):
        srcs = edge_srcs[t]
        dsts = edge_dsts[t]
        adjmats[t, srcs, dsts] = 1
    # num edges per snapshot
    print(f"Num edges per snapshot: {[len(edge_srcs[t]) for t in range(num_times)]}")

    onp.savez_compressed(
        save_path,
        attmats=feats,
        adjs=adjmats,
        labels=onehots,
        keep_nodes=keep_nodes,
    )
    print(f"Processed data saved to {save_path}")

