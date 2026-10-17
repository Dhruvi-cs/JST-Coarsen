
R"""
"""
#
import os

# 4. Handle CUBLAS (Crucial for newer PyTorch/CUDA)
# This must be set before certain CUDA functions are called
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ['PYTHONHASHSEED'] = '0' # preconfig

#
import numpy as onp
from typing import Tuple, List, cast
from ..arguments.coarsening import get_coarsen_str
from ...meta.dyngraph.sparse.dynedge import DynamicAdjacencyListDynamicEdge

def seed_everything(seed: int) -> None:
    R"""
    Set random seed for reproducibility.
    """
    #
    os.environ['PYTHONHASHSEED'] = str(seed)

    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print("Random seed set to {:d}.".format(seed))


def datasetIdentifier(**KWARGS) -> str:
    R"""
    Get identifier of given arguments.
    """
    #
    Nprime = cast(float, KWARGS["Nprime"])
    metaset_ss_num = cast(int, KWARGS["metaset_ss_num"])
    args = KWARGS["args"]
    COARS_ALG_STR = get_coarsen_str(args)
    origdset_props = KWARGS["props"]
    props_str = "".join(map(str, origdset_props))
    return (
        "{:s}~{:s}~{:s}".format(
            f"coarsened{COARS_ALG_STR}~{Nprime}~ss{metaset_ss_num}",
            cast(str, KWARGS["target"]),
            props_str,
        )
    )


def identifier(**KWARGS) -> str:
    R"""
    Get identifier of given arguments.
    """
    #
    res = (
        "{:s}~{:s}~{:s}~{:s}_{:s}~{:s}~{:s}_{:s}~{:s}~{:s}~{:s}_{:s}".format(
            "{:s}~{:s}".format(
                cast(str, KWARGS["source"]), cast(str, KWARGS["train_prop"]),
            )
            if "train_prop" in KWARGS else
            cast(str, KWARGS["source"]),
            cast(str, KWARGS["target"]),
            {"transductive": "trans", "inductive": "induc"}
            [cast(str, KWARGS["framework"])],
            cast(str, KWARGS["win_aggr"]), cast(str, KWARGS["model"]),
            cast(str, KWARGS["hidden"]), cast(str, KWARGS["activate"]),
            cast(str, KWARGS["lr"]), cast(str, KWARGS["weight_decay"]),
            cast(str, KWARGS["clipper"]), cast(str, KWARGS["patience"]),
            cast(str, KWARGS["seed"]),
        )
    )
    args = KWARGS["args"]
    if len(args.exp_name) > 0:
        res += "~" + args.exp_name
    return res


def identifier_coarse(**KWARGS) -> str:
    R"""
    Get identifier of given arguments.
    """
    #
    Nprime = cast(float, KWARGS["Nprime"])
    args = KWARGS["args"]
    COARS_ALG_STR = get_coarsen_str(args)
    
    res = (
        "{:s}~{:s}~{:s}~{:s}_{:s}~{:s}~{:s}_{:s}~{:s}~{:s}~{:s}_{:s}".format(
            "{:s}{:s}~{:s}~{:s}".format(
                cast(str, KWARGS["source"]), COARS_ALG_STR, f"{Nprime}", cast(str, KWARGS["train_prop"]),
            )
            if "train_prop" in KWARGS else
            "{:s}{:s}~{:s}".format(cast(str, KWARGS["source"]), COARS_ALG_STR, f"{Nprime}"),
            cast(str, KWARGS["target"]),
            {"transductive": "trans", "inductive": "induc"}
            [cast(str, KWARGS["framework"])],
            cast(str, KWARGS["win_aggr"]), cast(str, KWARGS["model"]),
            cast(str, KWARGS["hidden"]), cast(str, KWARGS["activate"]),
            cast(str, KWARGS["lr"]), cast(str, KWARGS["weight_decay"]),
            cast(str, KWARGS["clipper"]), cast(str, KWARGS["patience"]),
            cast(str, KWARGS["seed"]),
        )
    )
    args = KWARGS["args"]
    if len(args.exp_name) > 0:
        res += "~" + args.exp_name
    return res


def get_train_labels_counts(props: Tuple[int], priorities: Tuple[int],
    metaset: DynamicAdjacencyListDynamicEdge, spindle: str,
    /,
) -> List[int]:
    R"""
    Get train label counts.
    """
    #
    (train_indices, valid_indices, test_indices) = (
        metaset.fitsplit(props, priorities, spindle)
    )
    #
    if metaset.dyn_node_labels:
        if spindle == 'time':
            labels = metaset.node_labels[:, :, train_indices]
        elif spindle == 'node':
            labels = metaset.node_labels[train_indices]
    else:
        labels = metaset.node_labels[train_indices]

    assert onp.all(labels >= 0), "Labels should be non-negative."
    assert onp.all(labels < metaset.node_labels.shape[0]), "Labels out of range."
    #
    # labels = labels[labels >= 0]
    
    unique_labels = onp.unique(labels)
    num_labels = max(len(unique_labels), max(unique_labels) + 1)
    label_counts = onp.zeros((num_labels,), dtype=int)
    #
    if metaset.node_labels.ndim > 2:
        # dynamic node labels
        for t in range(labels.shape[0]):
            # use onp.add.at to handle repeated indices
            onp.add.at(label_counts, labels[t], 1)
    else:
        onp.add.at(label_counts, labels, 1)
    
    train_counts = label_counts.tolist()
    
    return train_counts

# utils function to record running time
# will be called two times
# first time to record the start time
# second time to record the end time 
# and return the time difference
import time
import datetime

_time_dict = dict()

def record_time(tag: str, end: bool = False) -> float:
    R"""
    Record time for given tag.
    """
    #
    global _time_dict
    if not end:
        assert tag not in _time_dict, f"Time for tag {tag} already recorded."
        _time_dict[tag] = time.time()
        return _time_dict[tag]
    else:
        assert tag in _time_dict, f"Time for tag {tag} not recorded."
        start_time = _time_dict[tag]
        elapsed = time.time() - start_time
        del _time_dict[tag]
        return elapsed, start_time

def format_time(timestamp: float) -> str:
    R"""
    Format timestamp to readable string.
    """
    #
    dt_object = datetime.datetime.fromtimestamp(timestamp)
    return dt_object.strftime("%Y-%m-%d %H:%M:%S")

def format_elapsed_time(seconds: float) -> str:
    R"""
    Format time in seconds to hh:mm:ss.
    """
    #
    assert seconds >= 0, "Seconds must be non-negative."
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)
