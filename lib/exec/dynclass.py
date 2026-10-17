R"""
"""
#
import argparse
import os
import numpy as onp
from typing import List, cast, Tuple
from .arguments.dynclass import add_dynclass_arguments
from .arguments.neuralnet import add_neuralnet_arguments
from .arguments.tune import add_tune_arguments
from .utils.floatrep import floatrep
from .utils.commons import (
    identifier, get_train_labels_counts,
    seed_everything, 
    record_time, format_elapsed_time
)
from ..data.dynclass import DynamicClassification, IMDB, DBLP5, DBLP5ss, Reddit4
from .model import encoderize, TIMESTAMPED, NOTEMBEDS, PRETAINABLES
from ..task.nodewinclass import NodeWindowClassification
from ..task.dyncsl import GraphWindowClassification
from ..task.classification import CE, ERR, MACRO, MACRO_REP
from ..framework.dyngraph.dyngraph import FrameworkDynamicGraph
from ..framework.dyngraph.dyngraphev import FrameworkDynamicGraphForEval
from ..meta.dyngraph.sparse.dynedge import DynamicAdjacencyListDynamicEdge

import wandb

from ..framework.indexable import DIR
from .consts import (
    WDB_PROJECT, WDB_MODE, SPLIT_PROPS
)

# Translate arguments.
datasetizes = {
    "Reddit4": Reddit4,
    "IMDB": IMDB, "DBLP5": DBLP5, "DBLP5ss200": DBLP5ss,
}

winsizes = {
    "DynCSL": 8, "Reddit4": 10,
    "IMDB": 61, "DBLP5": 10, "DBLP5ss200": 10,
}

window_future_sizes = {
    "DynCSL": 0, "Reddit4": 0,
    "IMDB": 0, "DBLP5": 0, "DBLP5ss200": 0,
}

spindles = {"transductive": "node", "inductive": "time"}

def get_dataset(
    args: argparse.Namespace, 
    source: str, spindle: str, 
    preprocess: bool = False, 
    extend_edge_time: List[str] = [],
    extend_node_time: List[str] = [],
    attach_edge_time: List[str] = [],
    attach_node_time: List[str] = [],
    win_aggr: str = "dense",
    targeton: List[int] = [0],
) -> Tuple[DynamicClassification, DynamicAdjacencyListDynamicEdge]:
    R"""
    Get dataset and metaset.
    """
    ref_source = source
    winsize = winsizes[ref_source]
    window_future_size = window_future_sizes[ref_source]

    datasetize = datasetizes[ref_source]
    dataset: DynamicClassification = datasetize(os.path.join("src", ref_source))
    # Formalize as future prediction task.
    # Predict 1 future frame (day) by 7 historical frames (past week).
    metaset = (
        dataset.asto_dynamic_adjacency_list_dynamic_edge(
            window_history_size=winsize, window_future_size=window_future_size,
            win_aggr=win_aggr, timestamped_edge_times=extend_edge_time,
            timestamped_node_times=extend_node_time,
            timestamped_edge_feats=attach_edge_time,
            timestamped_node_feats=attach_node_time,
        )
    )
    
    metaset.inputon(["none", "none", "all", "none"]) # (on_edge_feat, on_edge_label, on_node_feat, on_node_label)
    metaset.targeton(["none", "none", "none", targeton]) # (on_edge_feat, on_edge_label, on_node_feat, on_node_label)
    print(metaset)

    return dataset, metaset

def get_neuralnet(
    source: str = None, neuralname: str = None,
    neuralbase: str = None,
    hidden: int = None, activate: str = None,
    pretrain_seq_node: str = None,
    pretrain_seq_edge: str = None,
    spindle: str = None, targeton: List[int] = None,
    dataset: object = None, metaset: DynamicAdjacencyListDynamicEdge = None,
    num_batch_graphs: int = None, seed: int = None,
    label_counts: List[int] = None,
) -> object:
    R"""
    Get neural network.
    """
    tgnn = encoderize(
        neuralbase, metaset.edge_feat_size,
        1 if source == "DynCSL" else metaset.node_feat_size,
        hidden, hidden, activate,
        dyn_edge=True, dyn_node=True,
        num_nodes=metaset.num_nodes * num_batch_graphs, # TODO: TGN requires num_nodes, need to be fixed later when num_nodes changes
        tid_ax=1 if source == "DynCSL" else metaset.node_feat_size,
    )
    
    neuralnet = (
        (
            GraphWindowClassification
            if source == "DynCSL" else
            NodeWindowClassification
        )(
            tgnn, dataset.num_labels, hidden,
            label_counts=label_counts,
            activate=activate,
            notembedon=targeton if neuralbase in NOTEMBEDS else [],
        )
    )

    neuralnet.dynedge()
    neuralnet.initialize(seed)
    if neuralname in PRETAINABLES:
        #
        neuralnet.tgnn.pretrain("node", pretrain_seq_node)
        neuralnet.tgnn.pretrain("edge", pretrain_seq_edge)
    if source == "DynCSL":
        #
        neuralnet.tgnn.SIMPLEST = True
    return neuralnet

def main(*ARGS):
    R"""
    Main.
    """
    #
    parser = argparse.ArgumentParser(description="Dynamic Classification")
    add_dynclass_arguments(parser)
    add_neuralnet_arguments(parser)
    add_tune_arguments(parser)
    args = parser.parse_args() if len(ARGS) == 0 else parser.parse_args(ARGS)
    print("Arguments:", args)
    # Localize arguments.
    source = args.source
    target = args.target
    win_aggr = args.win_aggr

    #
    neuralname = args.model
    hidden = args.hidden
    activate = args.activate
    pretrain_seq_node = args.pretrain_seq_node
    pretrain_seq_edge = args.pretrain_seq_edge

    #
    frame = args.framework
    train_prop = args.train_prop
    max_epochs = args.epoch
    min_epochs = args.min_epoch
    lr = args.lr
    weight_decay = args.weight_decay
    clipper = args.clipper
    patience = args.patience
    seed = args.seed
    device = args.device
    resume_eval = args.resume_eval

    # Constant arguments.
    num_batch_graphs = args.num_batch_graphs
    
    props = SPLIT_PROPS # train, valid, test
    priorities = (2, 1, 0) # train, valid, test
    spindle = spindles[frame]
    
    #
    if len(train_prop) > 0:
        #
        (train_prop_num_str, train_prop_den_str) = train_prop.split("d")
        train_prop_num_str = train_prop_num_str.replace("n", "-")
        train_prop_num = int(train_prop_num_str)
        train_prop_den = int(train_prop_den_str)
        train_prop_neg = train_prop_num < 0
        train_prop_num = -train_prop_num if train_prop_neg else train_prop_num
    else:
        #
        train_prop_num = 0
        train_prop_den = 0
        train_prop_neg = False
    train_prop_tuple = (train_prop_num, train_prop_den, train_prop_neg)

    
    if target == "all":
        #
        targeton = [0]
    else:
        # UNEXPECT:
        # Classification task has fixed prediction.
        raise NotImplementedError("Classification task has fixed prediction.")

    #
    attach_edge_time: List[str]
    attach_node_time: List[str]

    # Get neural network basement.
    neuralitems = neuralname.split("-")
    if len(neuralitems) > 1:
        #
        if len(neuralitems) > 2 or neuralitems[1] != "Seqed":
            # UNEXPECT:
            # Improper neural network suffix.
            raise NotImplementedError(
                "Improper neural network suffix \"{:s}\" for \"{:s}\"."
                .format(neuralitems[0], "-".join(neuralitems[1:])),
            )
        neuralbase = neuralitems[0]
    else:
        #
        neuralbase = neuralname

    #
    if neuralbase in TIMESTAMPED:
        #
        extend_edge_time = TIMESTAMPED[neuralbase]["edge"]
        extend_node_time = TIMESTAMPED[neuralbase]["node"]
    else:
        #
        extend_edge_time = []
        extend_node_time = []
    attach_edge_time = []
    attach_node_time = []

    # Argument description identifier.
    desc = (
        identifier(
            source=source, target=target, framework=frame, win_aggr=win_aggr,
            model=(
                "{:s}-Seqed".format(neuralname)
                if (
                    neuralname in PRETAINABLES
                    and (
                        len(pretrain_seq_node) > 0
                        or len(pretrain_seq_edge) > 0
                    )
                ) else
                neuralname
            ),
            hidden=str(hidden), activate=activate, lr=floatrep(lr),
            weight_decay=floatrep(weight_decay), clipper=clipper,
            patience=str(patience), seed=str(seed),
            args=args,
        )
    )
    if args.exp_name != "":
        os.environ["WANDB_DIR"] = os.path.abspath(f"./wandb/{args.exp_name}")
    
    config = vars(args).copy()
    try:
        wdbrun = wandb.init(
            project=WDB_PROJECT,
            name=desc,
            mode=WDB_MODE,
            config=config
        )
    except Exception as e:
        print("Failed to initialize wandb run.")
        print(e)
        print("Falling back to local mode.")
        wdbrun = wandb.init(
            project=WDB_PROJECT,
            name=desc,
            mode="offline",
            config=config
        )
    
    print("WandB run initialized with id \"{:s}\".".format(wdbrun.id))
    record_time(tag=desc)
    
    seed_everything(seed)
    
    # Prepare COVID dataset.
    print("=" * 10 + " " + "Data & Meta" + " " + "=" * 10)
    dataset, metaset = get_dataset(
        args,
        source, spindle, preprocess=False,
        extend_edge_time=extend_edge_time,
        extend_node_time=extend_node_time,
        attach_edge_time=attach_edge_time,
        attach_node_time=attach_node_time,
        win_aggr=win_aggr,
        targeton=targeton,
    )
    metaset.dset_desc = f"{source}~original"

    print("=" * 10 + " " + "Model & Task" + " " + "=" * 10)
    labels_counts = get_train_labels_counts(
        props, priorities, metaset, spindle,
    )

    neuralnet = get_neuralnet(
        source, neuralname, 
        neuralbase,
        hidden, activate,
        pretrain_seq_node, pretrain_seq_edge,
        spindle, targeton,
        dataset, metaset,
        num_batch_graphs, seed,
        labels_counts
    )
    print(neuralnet)

    # Prepare framework.
    framework = (
        (
            FrameworkDynamicGraphForEval
            if source == "DynCSL" and len(resume_eval) > 0 else
            FrameworkDynamicGraph
        )(
            desc, metaset, neuralnet,
            lr=lr, weight_decay=weight_decay, seed=seed, device=device,
            metaspindle=spindle, gradclip=clipper,
        )
    )
    framework.set_node_batching(True)
    framework.BATCH_PAD = source != "DynCSL"
    
    if len(resume_eval) == 0:
        #
        framework.fit(
            props, priorities, train_prop_tuple,
            batch_size=num_batch_graphs, max_epochs=max_epochs, min_epochs=min_epochs, validon=MACRO,
            validrep=MACRO_REP, patience=patience,
        )
        
        # evaluate on original test set (not preprocessed)
        print("=" * 10 + "Evaluating on original test set. " + "=" * 10)
        _, metaset_orig = get_dataset(
            args,
            source, spindle, preprocess=False,
            extend_edge_time=extend_edge_time,
            extend_node_time=extend_node_time,
            attach_edge_time=attach_edge_time,
            attach_node_time=attach_node_time,
            win_aggr=win_aggr,
            targeton=targeton,
        )
        metaset_orig.dset_desc = f"{source}~original"
        
        neuralnet_test = get_neuralnet(
            source, neuralname, 
            neuralbase,
            hidden, activate,
            pretrain_seq_node, pretrain_seq_edge,
            spindle, targeton,
            dataset, metaset_orig,
            num_batch_graphs, seed,
            labels_counts
        )
        
        test_framework = (
            (
                FrameworkDynamicGraphForEval
                if source == "DynCSL" else
                FrameworkDynamicGraph
            )(
                desc, metaset_orig, neuralnet_test,
                lr=lr, weight_decay=weight_decay, seed=seed, device=device,
                metaspindle=spindle, gradclip=clipper,
            )
        )
        test_framework.set_node_batching(True)
        test_framework.BATCH_PAD = source != "DynCSL"
        
        test_framework.besteval(
            props, priorities, train_prop_tuple,
            batch_size=num_batch_graphs, validon=MACRO, validrep=MACRO_REP,
            resume=DIR,
        )
    else:
        #
        framework.besteval(
            props, priorities, train_prop_tuple,
            batch_size=num_batch_graphs, validon=MACRO, validrep=MACRO_REP,
            resume=resume_eval,
        )
    # 
    runtime, start_time = record_time(desc, end=True)
    wandb.summary["start_time"] = start_time
    print("Total runtime: {:s}".format(format_elapsed_time(runtime)))
    wandb.summary["total_runtime"] = runtime
    wandb.summary["total_runtime_str"] = format_elapsed_time(runtime)
    wandb.finish()
    # 
