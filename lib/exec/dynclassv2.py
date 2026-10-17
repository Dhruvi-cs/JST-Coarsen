R"""
"""
#
import argparse
import os
import numpy as onp
from typing import Tuple, List, cast
from .arguments.dynclass import add_dynclass_arguments
from .arguments.neuralnet import add_neuralnet_arguments
from .arguments.tune import add_tune_arguments
from .arguments.coarsening import add_coarsening_arguments, get_coarsen_str
from .utils.floatrep import floatrep
from .utils.commons import (
    identifier_coarse, get_train_labels_counts,
    datasetIdentifier, seed_everything, 
    record_time, format_elapsed_time
)
from ..data.dynclass import (
    DynamicClassification, 
    IMDB, DBLP5, DBLP5ss, 
    Reddit4, 
)
from ..data.dynclassv2 import (
    DynamicClassificationWithCoarsening, 
    IMDBWithCoarsening, 
    DBLP5WithCoarsening, 
    DBLP5ssWithCoarsening,
    Reddit4WithCoarsening, 
)
from .model import encoderize, TIMESTAMPED, NOTEMBEDS, PRETAINABLES
from ..task.nodewinclass import NodeWindowClassification
from ..task.dyncsl import GraphWindowClassification
from ..task.classification import CE, ERR, MACRO, METRICS_REP
from ..framework.dyngraph.dyngraph import FrameworkDynamicGraph
from ..framework.dyngraph.dyngraphev import FrameworkDynamicGraphForEval
from ..meta.dyngraph.sparse.dynedge import DynamicAdjacencyListDynamicEdge

import wandb

from ..framework.indexable import DIR
from .consts import (
    WDB_PROJECT, WDB_MODE, SPLIT_PROPS
)

datasetizes_orig = {
    "Reddit4": Reddit4,
    "IMDB": IMDB, "DBLP5": DBLP5, "DBLP5ss200": DBLP5ss,
}
datasetizes_coarsen = {
    "IMDB": IMDBWithCoarsening,
    "DBLP5": DBLP5WithCoarsening,
    "DBLP5ss200": DBLP5ssWithCoarsening,
    "Reddit4": Reddit4WithCoarsening,
}

winsizes = {
    "Brain10": 12, "DynCSL": 8, "Reddit4": 10,
    "IMDB": 61, "UIHC": 6, "DBLP5": 10, "DBLP5ss200": 10,
    "SFHH": 7, "LH10": 7, "InVS15": 7, 
    "MOOC": 5,
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
    
    props: Tuple[int, int, int] = (7, 1, 2),
    priorities: Tuple[int, int, int] = (2, 1, 0),
    Nprime: float = None,
    metaset_ss_num: int =None,
    metaset_savepath: str = None,
    metaset_force_reload = False,
    pre_run_results_path: str = None,
    
    spec_gap_method: str = 'approx', 
    coars_op: str = 'PtAP', 
    rw_sl: bool = True, 
    ef_rw_method: str = 'avg',
    refine_every: float = 0.1
) -> Tuple[object, DynamicAdjacencyListDynamicEdge]:
    R"""
    Get dataset and metaset.
    """
    #
    ref_source = source
        
    winsize = winsizes[ref_source]
    window_future_size = window_future_sizes[ref_source]

    if not preprocess:
        datasetize = datasetizes_orig[ref_source]
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
    else:
        datasetize = datasetizes_coarsen[ref_source]
        dataset: DynamicClassificationWithCoarsening = datasetize(os.path.join("src", ref_source), alg_version=args.coars_alg)
        #
        metaset = (
            dataset.asto_dynamic_adjacency_list_dynamic_edge(
                spindle,
                window_history_size=winsize, window_future_size=window_future_size,
                win_aggr=win_aggr, timestamped_edge_times=extend_edge_time,
                timestamped_node_times=extend_node_time,
                timestamped_edge_feats=attach_edge_time,
                timestamped_node_feats=attach_node_time,
                
                props=props, priorities=priorities,
                Nprime=Nprime,
                metaset_ss_num=metaset_ss_num,
                metaset_savepath=metaset_savepath,
                metaset_force_reload=metaset_force_reload,
                pre_run_results_path=pre_run_results_path,
                
                spec_gap_method=spec_gap_method,
                coars_op=coars_op,
                rw_sl=rw_sl,
                ef_rw_method=ef_rw_method,
                refine_every=refine_every
            )
        )
    print("========== Using {:s} metaset. ==========".format(
        "preprocessed" if preprocess else "original"
    ))

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
        num_nodes=metaset.num_nodes * num_batch_graphs,
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
    add_coarsening_arguments(parser)
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
    #
    Nprime = args.n_prime # (1 - coarsening ratio)
    metaset_force_reload = args.metaset_force_reload
    metaset_ss_num = args.metaset_ss_num
    
    # Constant arguments.
    # num_batch_graphs = 1
    num_batch_graphs = args.num_batch_graphs
    pre_run_results_path = args.pre_run_results_path
    prep_only = args.prep_only
    refine_every = args.refine_every

    props = SPLIT_PROPS # train, valid, test
    priorities = (2, 1, 0) # train, valid, test
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

    # Translate arguments.
    spindle = spindles[frame]
    
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
        identifier_coarse(
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
            Nprime=floatrep(Nprime)
        )
    )
    #
    if prep_only: desc = "Prep~" + desc
    
    dsetdesc = datasetIdentifier(
        target=target, Nprime=Nprime, 
        metaset_ss_num=metaset_ss_num,
        args=args, props=props
    )
    if args.exp_name != "":
        exp_name_str = f"~{args.exp_name}"
    else:
        exp_name_str = ""
    metaset_savepath = os.path.join("src", source,  'coarsened', f"metaset{exp_name_str}~{dsetdesc}.pkl")
    # metaset_savepath = os.path.join("src", source,  'coarsened', f"metaset~{dsetdesc}.pkl")
    os.makedirs(os.path.dirname(metaset_savepath), exist_ok=True)
    
    if args.exp_name != "":
        os.environ["WANDB_DIR"] = os.path.abspath(f"./wandb/{args.exp_name}")
        
    # Include constants into the config
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
    record_time(desc)
    
    seed_everything(seed)
    print("=" * 10 + " " + "Data & Meta" + " " + "=" * 10)
    dataset, metaset = get_dataset(
        args,
        source, spindle, preprocess=True,
        extend_edge_time=extend_edge_time,
        extend_node_time=extend_node_time,
        attach_edge_time=attach_edge_time,
        attach_node_time=attach_node_time,
        win_aggr=win_aggr,
        targeton=targeton,

        props=props, priorities=priorities,
        Nprime=Nprime,
        metaset_ss_num=metaset_ss_num,
        metaset_savepath=metaset_savepath,
        metaset_force_reload=metaset_force_reload,
        pre_run_results_path=pre_run_results_path,
        
        spec_gap_method=args.spec_gap_method,
        coars_op=args.coars_op,
        rw_sl=args.rw_sl,
        ef_rw_method=args.ef_rw_method,
        refine_every=refine_every
    )
    metaset.dset_desc = f"{source}~{dsetdesc}"

    if prep_only:
        print("=" * 10 + " " + "Model & Task" + " " + "=" * 10)
        label_counts = get_train_labels_counts(
            props, priorities, metaset, spindle
        )
        print("Label counts on training set:", label_counts)
        wandb.summary["label_counts"] = label_counts
        for i, count in enumerate(label_counts):
            if count < 2:
                print("Warning: Label {:d} has less than 3 samples in training set.".format(i))
                # remove the saved preprocessed data to avoid confusion
                if os.path.exists(metaset_savepath):
                    os.remove(metaset_savepath)
                    print("Removed preprocessed data at \"{:s}\".".format(metaset_savepath))
                    
                raise ValueError("Insufficient label count for training.")
        
        print("Preparation only, exiting now.")
        wandb.finish()
        return
    
    
    print("=" * 10 + " " + "Model & Task" + " " + "=" * 10)
    label_counts = get_train_labels_counts(
        props, priorities, metaset, spindle
    )

    neuralnet = get_neuralnet(
        source, neuralname, 
        neuralbase,
        hidden, activate,
        pretrain_seq_node, pretrain_seq_edge,
        spindle, targeton,
        dataset, metaset,
        num_batch_graphs, seed,
        label_counts
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
            None, None, 
            train_prop_tuple,
            batch_size=num_batch_graphs, max_epochs=max_epochs, min_epochs=min_epochs, validon=MACRO,
            validrep=METRICS_REP[MACRO], patience=patience,
        )
        #
        #
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
            label_counts
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
            batch_size=num_batch_graphs, validon=MACRO, validrep=METRICS_REP[MACRO],
            resume=DIR,
        )
    else:
        #
        framework.besteval(
            props, priorities, train_prop_tuple,
            batch_size=num_batch_graphs, validon=MACRO, validrep=METRICS_REP[MACRO],
            resume=resume_eval,
        )
    #
    runtime, start_time = record_time(desc, end=True)
    wandb.summary["start_time"] = start_time
    print("Total runtime: {:s}".format(format_elapsed_time(runtime)))
    wandb.summary["total_runtime"] = runtime
    wandb.summary["total_runtime_str"] = format_elapsed_time(runtime)
    wandb.finish()