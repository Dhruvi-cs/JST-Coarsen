R"""
"""
#
import numpy as onp
import numpy.typing as onpt
import scipy.sparse as osparse
import os
import torch
from typing import List, Callable, Tuple, cast, Optional, Union
from ..dyngraph import DynamicGraph
from ...graph.sparse.edge import edge_symmetrize, edge_sort, edge_unique
from ...graph.sparse.degree import bins
from ....utils.info import INFO, info5, infounion
from ...utils.repr.setlike import setlike
from ...types import MEMPIN, MEMBAT
from ...utils.normalize import normalize, standardize
from ...utils.repr.distrep import distrep
from ...utils.spindle import fitsplit, stratified_fitsplit, keep_order_in_A
from ...utils.slidewin import sliding_window
from ...vecseq import VectorSequence
import time
from tqdm.auto import tqdm
import pickle 
import wandb

from ....exec.consts import NE_THRESH, CLF_TASK, MAX_CACHE_SIZE
from functools import lru_cache

from .coarsening_utils import *
from collections import Counter

class DynamicAdjacencyListDynamicEdgeWithCoarsening(DynamicGraph):
    R"""
    Temporal graph as dynamic adjacency list meta whose edge connectivities,
    features and labels are also dynamic.
    """
    #
    WINAGG_NONE = 0
    WINAGG_DENSE = 1
    WINAGG_SPARSE = 2
    WINAGGS = {
        "none": WINAGG_NONE,
        "dense": WINAGG_DENSE,
        "sparse": WINAGG_SPARSE,
    }

    #
    WIN_SHIFT = -1

    def set_win_shift(self, num: int, /, data_source=None) -> None:
        R"""
        Change window shifting factor.
        """
        # Should only support DynCSL.
        # if data_source is not None and data_source == 'UIHC':
        #     self.WIN_SHIFT = num
        if num != 8:
            # UNEXPECT:
            # Only support DynCSL.
            raise NotImplementedError(
                "Only support non-default window shifting 8 for DynCSL.",
            )
        self.WIN_SHIFT = num

    def __init__(
        self,
        edge_srcs: List[onpt.NDArray[onp.generic]],
        edge_dsts: List[onpt.NDArray[onp.generic]],
        edge_feats: List[onpt.NDArray[onp.generic]],
        edge_labels: List[onpt.NDArray[onp.generic]],
        node_feats: onpt.NDArray[onp.generic],
        node_labels: onpt.NDArray[onp.generic],
        
        /, *, 
        hetero: bool, symmetrize: bool, sort: bool,
        alg_version: str,  # v2, v3
        task_type: str,
        normalizer: str = 'standardize',
    ) -> None:
        R"""
        Initialize the class.
        """
        self.task_type = task_type
        self.alg_version = alg_version
        self.normalizer = normalizer
        self.dset_desc = None # to be set later outside as cache_id
        # Safety check at each snapshot.
        for (
            t,
            (
                edge_snap_srcs, edge_snap_dsts, edge_snap_feats,
                edge_snap_labels,
            ),
        ) in enumerate(zip(edge_srcs, edge_dsts, edge_feats, edge_labels)):
            #
            if (
                not (
                    len(edge_snap_srcs) == len(edge_snap_dsts)
                    == len(edge_snap_feats)
                )
            ):
                # EXPECT:
                # It is possible to have dirty data.
                raise RuntimeError(
                    "Fail to transform provided data of timestamp {:d} into "
                    "graph as adjacency list since its edge defintion columns "
                    "do not agree on length.".format(t),
                )
            if not len(edge_snap_feats) == len(edge_snap_labels):
                # EXPECT:
                # It is possible to have dirty data.
                raise RuntimeError(
                    "Fail to transform provided data of timestamp {:d} into "
                    "graph as adjacency list since its edge data columns do "
                    "not agree on length.".format(t),
                )
            if (
                len(edge_snap_feats.shape) < 2
                or len(edge_snap_labels.shape) < 2
            ):
                # EXPECT:
                # It is possible to have dirty data.
                raise RuntimeError(
                    "Fail to transform provided data of timestamp {:d} into "
                    "graph as adjacency list since its edge data columns need "
                    "at least 2 dimensions.".format(t),
                )

        # Safety check over all nodes of all timestamps.
        if not len(node_feats) == len(node_labels):
            # EXPECT:
            # It is possible to have dirty data.
            raise RuntimeError(
                "Fail to transform provided data into graph as dynamic "
                "adjacency list since its node data columns do not agree on "
                "length.",
            )
        if len(node_feats.shape) < 2 or len(node_labels.shape) < 2:
            # EXPECT:
            # It is possible to have dirty data.
            raise RuntimeError(
                "Fail to transform provided data into graph as dynamic "
                "adjacency list since its node data columns need at least 2 "
                "dimensions.",
            )

        # Pay attention that symmetrized graph must be symmetric graph, but
        # symmetric graph may not be symmetrized.
        if symmetrize:
            #
            edge_srcs_new = []
            edge_dsts_new = []
            edge_feats_new = []
            edge_labels_new = []
            for (
                edge_snap_srcs, edge_snap_dsts, edge_snap_feats,
                edge_snap_labels,
            ) in zip(edge_srcs, edge_dsts, edge_feats, edge_labels):
                #
                (
                    edge_snap_srcs, edge_snap_dsts, edge_snap_feats,
                    edge_snap_labels,
                ) = (
                    edge_symmetrize(
                        edge_snap_srcs, edge_snap_dsts, edge_snap_feats,
                        edge_snap_labels,
                    )
                )
                edge_srcs_new.append(edge_snap_srcs)
                edge_dsts_new.append(edge_snap_dsts)
                edge_feats_new.append(edge_snap_feats)
                edge_labels_new.append(edge_snap_labels)
            edge_srcs = edge_srcs_new
            edge_dsts = edge_dsts_new
            edge_feats = edge_feats_new
            edge_labels = edge_labels_new
            self.symmetrized = True
        else:
            #
            self.symmetrized = False

        #
        if sort:
            #
            #
            edge_srcs_new = []
            edge_dsts_new = []
            edge_feats_new = []
            edge_labels_new = []
            for (
                edge_snap_srcs, edge_snap_dsts, edge_snap_feats,
                edge_snap_labels,
            ) in zip(edge_srcs, edge_dsts, edge_feats, edge_labels):
                #
                (
                    edge_snap_srcs, edge_snap_dsts, edge_snap_feats,
                    edge_snap_labels,
                ) = (
                    edge_sort(
                        edge_snap_srcs, edge_snap_dsts, edge_snap_feats,
                        edge_snap_labels,
                    )
                )
                edge_srcs_new.append(edge_snap_srcs)
                edge_dsts_new.append(edge_snap_dsts)
                edge_feats_new.append(edge_snap_feats)
                edge_labels_new.append(edge_snap_labels)
            edge_srcs = edge_srcs_new
            edge_dsts = edge_dsts_new
            edge_feats = edge_feats_new
            edge_labels = edge_labels_new
            self.sorted = True
        else:
            #
            self.sorted = False

        # Safety check.
        if not hetero:
            #
            for (t, (edge_snap_srcs, edge_snap_dsts)) in (
                enumerate(zip(edge_srcs, edge_dsts))
            ):
                #
                if (
                    edge_unique(
                        edge_snap_srcs, edge_snap_dsts,
                        sorted=self.sorted,
                    )
                ):
                    # EXPECT:
                    # It is possible to have improper arguments.
                    raise RuntimeError(
                        "Fail to transform provided data into graph as "
                        "dynamic adjacency list since non-heterogeneous graph "
                        "should not have duplicate edges at timestamp {:d}."
                        .format(t),
                    )

        #
        meaninglessall = lambda array: onp.min(array) == onp.max(array)
        meaninglesssnap = (
            lambda lstarray: (
                any(onp.min(array) == onp.max(array) for array in lstarray)
            )
        )
        
        # Collect edge data.
        self.edge_srcs_col = 0
        self.edge_dsts_col = 1
        self.edge_tuples = (
            [
                onp.stack((edge_snap_srcs, edge_snap_dsts))
                for (edge_snap_srcs, edge_snap_dsts) in (
                    zip(edge_srcs, edge_dsts)
                )
            ]
        )
        self.edge_feats = edge_feats
        self.edge_labels = edge_labels

        # Collect node data.
        self.node_feats = node_feats
        self.node_labels = node_labels

        # Collect data shapes.
        self.shapize()

        # Collect meaningless data.
        self.meaningless_node_feats = meaninglessall(self.node_feats)
        self.meaningless_node_labels = meaninglessall(self.node_labels)
        self.meaningless_edge_feats = meaninglesssnap(self.edge_feats)
        self.meaningless_edge_labels = meaninglesssnap(self.edge_labels)

        # Collect essential statistics.
        self.collect()
        

    def save(self, preprocessed_items, path: str, /) -> None:
        R"""
        Save the class to a file.
        """
        # create parent directory if it does not exist
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
            
        # save with pickle 
        with open(path, "wb") as f:
            pickle.dump(preprocessed_items, f)
        
    def load_preprocessed(self, path: str, /) -> None:
        R"""
        Load the class from a file.
        """
        # Load preprocessed items.
        if not os.path.exists(path):
            raise FileNotFoundError(
                "Path \"{:s}\" does not exist.".format(path),
            )
        # load with pickle
        with open(path, "rb") as f:
            preprocessed_items = pickle.load(f)
            
            self.node_feats = preprocessed_items["node_feats"]
            self.node_labels = preprocessed_items["node_labels"]
            self.edge_feats = preprocessed_items["edge_feats"]
            self.edge_labels = preprocessed_items["edge_labels"]
            self.edge_tuples = preprocessed_items["edge_tuples"]
            self.node_times_abs = preprocessed_items["node_times_abs"]
            self.node_times_inc = preprocessed_items["node_times_inc"]
            self.edge_times_abs = preprocessed_items["edge_times_abs"]
            self.edge_times_inc = preprocessed_items["edge_times_inc"]
            self.timestamps = preprocessed_items["timestamps"]
            self.num_nodes = preprocessed_items["num_nodes"]
            self.num_edges = preprocessed_items["num_edges"]
            self.lab_avail_nodes = preprocessed_items["lab_avail_nodes"]
            self.coars_train_indices = preprocessed_items["coars_train_indices"]
            self.coars_valid_indices = preprocessed_items["coars_valid_indices"]
            # self.all_Ps = preprocessed_items["all_Ps"]
        
        
    def preprocess(self, 
        props: Tuple[int, int, int],
        priorities: Tuple[int, int, int],
        spindle: str,
        Nprime: float = 0.95,
        metaset_ss_num: int = 500,
        savepath: Optional[str] = None, 
        force_reload: bool = False,
        pre_run_results_path: str = None, 
        
        spec_gap_method: str = 'approx', 
        coars_op: str = 'PtAP', 
        rw_sl: bool = True, 
        ef_rw_method: str = 'avg',
        refine_every: float = 0.1,
    ) -> None:
        ''' 
        This is called after class initialization (after symmetrization and sorting).
        '''
        if not force_reload:
            assert savepath is not None, "Save path must be provided when not force reloading."
            if os.path.exists(savepath):
                # Load preprocessed items.
                self.load_preprocessed(savepath)

                self.shapize()
                self.collect()
                
                print("Preprocessed data loaded from \"{:s}\".".format(savepath))
                return
            else:
                print("Preprocessed data not found at \"{:s}\", run preprocessing first, existing...".format(savepath))
                exit(0)
                
        elif savepath is None:
            print("Warning: no save path is provided, the preprocessed data will not be saved.")
            raise ValueError("Save path must be provided when force reloading.")

        # Coarsen the graph    
        edge_times_abs = self.edge_times_abs
        edge_times_inc = self.edge_times_inc
        node_times_abs = self.node_times_abs
        node_times_inc = self.node_times_inc

        edge_tuples = self.edge_tuples 
        edge_srcs = self.edge_srcs
        edge_dsts = self.edge_dsts
        edge_feats = self.edge_feats
        edge_labels = self.edge_labels
        #
        node_feats = self.node_feats 
        node_labels = self.node_labels
        timestamps = self.timestamps

        (train_indices, valid_indices, test_indices) = (
            self.fitsplit(props, priorities, spindle)
        )
        
        if spindle == 'node':
            orig_split_indices = (train_indices, valid_indices, test_indices)
        else:
            orig_split_indices = ([], [], [])
        
        if self.alg_version.lower().startswith("random"):
            coarsen_results = self.coarsen_window_dyngraph_random(
                metaset_ss_num,
                timestamps,
                edge_srcs, edge_dsts,
                edge_feats, edge_labels,
                node_feats, node_labels,
                self.alg_version,
                Nprime=Nprime,
                split_indices=orig_split_indices,
                spec_gap_method=spec_gap_method,
                coars_op=coars_op,
                rw_sl=rw_sl,
                ef_rw_method=ef_rw_method,
            )
        elif self.alg_version.startswith("TOPKSupraRefine"):
            coarsen_results = self.coarsen_window_dyngraph_supra_refine(
                metaset_ss_num,
                timestamps,
                edge_srcs, edge_dsts,
                edge_feats, edge_labels,
                node_feats, node_labels, 
                self.alg_version,
                Nprime=Nprime,
                split_indices=orig_split_indices,
                
                spec_gap_method=spec_gap_method, 
                coars_op=coars_op, 
                rw_sl=rw_sl,
                ef_rw_method=ef_rw_method,
                refine_every=refine_every,
            )
        elif self.alg_version.startswith("TOPKSnapRefine"):
            coarsen_results = self.coarsen_window_dyngraph_snap_refine(
                metaset_ss_num,
                timestamps,
                edge_srcs, edge_dsts,
                edge_feats, edge_labels,
                node_feats, node_labels, 
                self.alg_version,
                Nprime=Nprime,
                split_indices=orig_split_indices,
                
                spec_gap_method=spec_gap_method, 
                coars_op=coars_op, 
                rw_sl=rw_sl,
                ef_rw_method=ef_rw_method,
                refine_every=refine_every,
            )
        elif self.alg_version.startswith("TOPKSupra"):
            coarsen_results = self.coarsen_window_dyngraph_supra(
                metaset_ss_num,
                timestamps,
                edge_srcs, edge_dsts,
                edge_feats, edge_labels,
                node_feats, node_labels, 
                self.alg_version,
                Nprime=Nprime,
                split_indices=orig_split_indices,
                
                spec_gap_method=spec_gap_method, 
                coars_op=coars_op, 
                rw_sl=rw_sl,
                ef_rw_method=ef_rw_method,
            )
        elif self.alg_version.lower().startswith("TOPKSnap"):
            coarsen_results = self.coarsen_window_dyngraph_snap(
                metaset_ss_num,
                timestamps,
                edge_srcs, edge_dsts,
                edge_feats, edge_labels,
                node_feats, node_labels, 
                self.alg_version,
                Nprime=Nprime,
                split_indices=orig_split_indices,
                
                spec_gap_method=spec_gap_method, 
                coars_op=coars_op, 
                rw_sl=rw_sl,
                ef_rw_method=ef_rw_method,
            )
        elif self.alg_version.lower().startswith("baseline"):
            coarsen_results = self.coarsen_window_dyngraph_baseline(
                timestamps,
                edge_srcs, edge_dsts,
                edge_feats, edge_labels,
                node_feats, node_labels, 
                self.alg_version,
                split_indices=orig_split_indices,
                pre_run_results_path=pre_run_results_path,
                
                spec_gap_method=spec_gap_method, 
                coars_op=coars_op, 
                rw_sl=rw_sl,
                ef_rw_method=ef_rw_method,
            )
        
        else:
            raise NotImplementedError(
                f"Coarsening algorithm version \"{self.alg_version.lower()}\" is not supported."
            )
        
        edge_times_abs, edge_times_inc, \
        node_times_abs, node_times_inc, \
        edge_tuples, edge_srcs, edge_dsts, \
        edge_feats, edge_labels, \
        node_feats, node_labels, \
        num_nodes, node_label_mask, \
        coars_train_indices, coars_valid_indices, \
        all_Ps = coarsen_results
        
        self.node_times_inc = node_times_inc
        self.node_times_abs = node_times_abs
        self.edge_times_inc = edge_times_inc
        self.edge_times_abs = edge_times_abs
        self.edge_tuples = edge_tuples
        self.edge_feats = edge_feats
        self.edge_labels = edge_labels
        self.node_feats = node_feats
        self.node_labels = node_labels
        self.num_nodes = num_nodes
        self.lab_avail_nodes = onp.where(node_label_mask)[0]
        
        if spindle == 'node':
            self.coars_train_indices = coars_train_indices
            self.coars_valid_indices = coars_valid_indices
        else:
            self.coars_train_indices = train_indices
            self.coars_valid_indices = valid_indices
        # self.all_Ps = all_Ps
        self.shapize()
        self.collect()
        
        
        # Save preprocessed items.
        if savepath is not None:
            preprocessed_items = {
                "node_feats": self.node_feats,
                "node_labels": self.node_labels,
                "edge_feats": self.edge_feats,
                "edge_labels": self.edge_labels,
                "edge_tuples": self.edge_tuples,
                "node_times_abs": self.node_times_abs,
                "node_times_inc": self.node_times_inc,
                "edge_times_abs": self.edge_times_abs,
                "edge_times_inc": self.edge_times_inc,
                "timestamps": self.timestamps,
                "num_nodes": self.num_nodes, 
                "num_edges": self.num_edges,
                "lab_avail_nodes": self.lab_avail_nodes,
                "coars_train_indices": self.coars_train_indices,
                "coars_valid_indices": self.coars_valid_indices,
                "all_Ps": all_Ps,  # save the partition matrices
            }
            
            self.save(preprocessed_items, savepath)
            print("Preprocessed data saved to \"{:s}\".".format(savepath))
            
    
    @property
    def edge_srcs(self, /) -> List[onpt.NDArray[onp.generic]]:
        R"""
        Get edge source indices.
        """
        # Must explicitly do the annotation.
        return (
            [
                cast(
                    onpt.NDArray[onp.generic],
                    edge_snap_tuples[self.edge_srcs_col],
                )
                for edge_snap_tuples in self.edge_tuples
            ]
        )
    

    @property
    def edge_dsts(self, /) -> List[onpt.NDArray[onp.generic]]:
        R"""
        Get edge destination indices.
        """
        # Must explicitly do the annotation.
        return (
            [
                cast(
                    onpt.NDArray[onp.generic],
                    edge_snap_tuples[self.edge_dsts_col],
                )
                for edge_snap_tuples in self.edge_tuples
            ]
        )

    def shapize(self, /) -> None:
        R"""
        Get data shapes.
        """
        # Edge data must be dynamic as a list of arraies.
        num_edges_buf = []
        for (edge_snap_feats, edge_snap_labels) in (
            zip(self.edge_feats, self.edge_labels)
        ):
            #
            (num_edges, self.edge_feat_size) = edge_snap_feats.shape
            (_, self.edge_label_size) = edge_snap_labels.shape
            num_edges_buf.append(num_edges)
        self.num_edges_array = onp.array(num_edges_buf)
        self.num_edges = onp.sum(self.num_edges_array).item()

        # Node data may have time axis which will requires explicitly setting
        # sliding window.
        (self.num_nodes, *self.node_feat_shape) = self.node_feats.shape
        (_, *self.node_label_shape) = self.node_labels.shape

    def collect(self, /) -> None:
        R"""
        Collect essential statistics.
        """
        #
        degree_ins_buf = []
        degree_outs_buf = []
        for (edge_snap_srcs, edge_snap_dsts) in (
            zip(self.edge_srcs, self.edge_dsts)
        ):
            #
            degree_ins = (
                onp.zeros_like(self.num_edges, shape=(self.num_nodes,))
            )
            degree_outs = (
                onp.zeros_like(self.num_edges, shape=(self.num_nodes,))
            )
            onp.add.at(
                degree_ins, cast(onpt.NDArray[onp.int64], edge_snap_dsts), 1,
            )
            onp.add.at(
                degree_outs, cast(onpt.NDArray[onp.int64], edge_snap_srcs), 1,
            )
            degree_ins_buf.append(degree_ins)
            degree_outs_buf.append(degree_outs)
        self.degree_ins = onp.stack(degree_ins_buf, axis=1)
        self.degree_outs = onp.stack(degree_outs_buf, axis=1)

    def dynamicon(
        self,
        /,
        *,
        dyn_node_feats: bool, dyn_node_labels: bool,
    ) -> None:
        R"""
        Specify dynamic data columns.
        """
        #
        if not (dyn_node_feats or dyn_node_labels):
            # EXPECT:
            # It is possible to have improper arguments.
            raise RuntimeError(
                "Fail to transform provided data into temporal graph as "
                "dynamic adjacency list with static edges since none of node"
                "columns is temporal.",
            )

        # Update node shapes.
        if dyn_node_feats:
            #
            (self.node_feat_size, self.num_times) = self.node_feat_shape
        else:
            #
            (self.node_feat_size,) = self.node_feat_shape
        self.dyn_node_feats = dyn_node_feats
        if dyn_node_labels:
            #
            (self.node_label_size, self.num_times) = self.node_label_shape
        else:
            #
            (self.node_label_size,) = self.node_label_shape
        self.dyn_node_labels = dyn_node_labels

    def timestamping(
        self,
        timestamps: onpt.NDArray[onp.generic],
        /,
        *,
        timestamped_edge_times: List[str], timestamped_node_times: List[str],
        timestamped_edge_feats: List[str], timestamped_node_feats: List[str],
    ) -> None:
        R"""
        Set exact timestamps for dynamic graph steps.
        """
        #
        DynamicGraph.timestamping(
            self, timestamps,
            timestamped_edge_times=timestamped_edge_times,
            timestamped_edge_feats=timestamped_edge_feats,
            timestamped_node_times=timestamped_node_times,
            timestamped_node_feats=timestamped_node_feats,
        )

        (
            self.edge_times_abs, self.edge_times_inc, 
            self.node_times_abs, self.node_times_inc
        ) = (
            self._timestamping(
                self.edge_tuples, self.timestamps, self.num_times, self.num_nodes,
            )
        )

    def _timestamping(_, edge_tuples, timestamps, num_times, num_nodes) -> Tuple[
            List[onpt.NDArray[onp.generic]], List[onpt.NDArray[onp.generic]],
            onpt.NDArray[onp.generic], onpt.NDArray[onp.generic],
        ]:
        ''' 
        timestamps: onpt.NDArray[onp.generic] (num_times, ) the actual timestamps of each snapshot
        '''
        # Edge timestamps.
        edge_times_abs = (
            [
                onp.full((edge_snap_tuples.shape[1],), snap_t)
                for (snap_t, edge_snap_tuples) in (
                    zip(timestamps.tolist(), edge_tuples)
                )
            ]
        )
        edge_times_inc = (
            [onp.zeros_like(edge_times_abs[0])]
            + (
                [
                    onp.full((edge_snap_tuples.shape[1],), curr_t - prev_t)
                    for (prev_t, curr_t, edge_snap_tuples) in (
                        zip(
                            timestamps[:-1].tolist(),
                            timestamps[1:].tolist(), edge_tuples[1:],
                        )
                    )
                ]
            )
        )

        # Node timestamps.
        node_times_abs = (
            onp.tile(
                onp.reshape(timestamps, (1, num_times)),
                (num_nodes, 1),
            )
        )
        node_times_inc = (
            onp.concatenate(
                (
                    onp.zeros_like(
                        node_times_abs,
                        shape=(num_nodes, 1),
                    ),
                    node_times_abs[:, 1:] - node_times_abs[:, :-1],
                ),
                axis=1,
            )
        )
        return (
            edge_times_abs, edge_times_inc, node_times_abs, node_times_inc,
        )
        
    def sliding_window(
        self,
        /,
        *,
        window_history_size: Optional[int], window_future_size: Optional[int],
    ) -> None:
        R"""
        Extend dynamic graph with sliding window.
        """
        # Get number of windows.
        (
            self.window_size, self.window_history_size,
            self.window_future_size,
        ) = (
            sliding_window(
                window_history_size, window_future_size, self.num_times,
            )
        )
        if self.WIN_SHIFT < 0:
            #
            self.num_windows = self.num_times - self.window_size + 1
        else:
            # DynCSL.
            self.num_windows = self.num_times // self.WIN_SHIFT

    def sliding_aggregation(self, *, win_aggr: str) -> None:
        R"""
        Set sliding window aggregation method.
        """
        #
        self.winaggr = self.WINAGGS[win_aggr]

    def repr(self, /) -> INFO:
        R"""
        Get essential statistics for representation.
        """
        #
        dict = DynamicGraph.repr(self)
        dict["Basic"]["#Edges"] = (
            "[{:d}, {:d}]({:.1f})".format(
                onp.min(self.num_edges_array), onp.max(self.num_edges_array),
                onp.mean(self.num_edges_array),
            )
        )
        dict["Basic"]["Win.Aggr"] = ["None", "Dense", "Sparse"][self.winaggr]

        # By default, we focus on the degree sum over all timestamps.
        dict["(Deg)ree Max"] = {}
        dict["(Deg)ree Sum"] = {}
        dict["(Deg)ree"] = dict["(Deg)ree Sum"]

        #
        dict["(Deg)ree Max"]["Deg In"] = (
            setlike(onp.max(self.degree_ins, axis=1), axis=0, n=1)[0]
        )
        dict["(Deg)ree Max"]["Deg Out"] = (
            setlike(onp.max(self.degree_outs, axis=1), axis=0, n=1)[0]
        )
        dict["(Deg)ree Max"]["Deg In Dist"] = (
            bins(
                onp.max(self.degree_ins, axis=1),
                num_bins=10, num_bins_per_line=3,
            )[0]
        )
        dict["(Deg)ree Max"]["Deg Out Dist"] = (
            bins(
                onp.max(self.degree_outs, axis=1),
                num_bins=10, num_bins_per_line=3,
            )[0]
        )

        #
        dict["(Deg)ree Sum"]["Deg In"] = (
            setlike(onp.sum(self.degree_ins, axis=1), axis=0, n=1)[0]
        )
        dict["(Deg)ree Sum"]["Deg Out"] = (
            setlike(onp.sum(self.degree_outs, axis=1), axis=0, n=1)[0]
        )
        dict["(Deg)ree Sum"]["Deg In Dist"] = (
            bins(
                onp.sum(self.degree_ins, axis=1),
                num_bins=10, num_bins_per_line=3,
            )[0]
        )
        dict["(Deg)ree Sum"]["Deg Out Dist"] = (
            bins(
                onp.sum(self.degree_outs, axis=1),
                num_bins=10, num_bins_per_line=3,
            )[0]
        )
        return dict

    def argsort_node_degree_in(self, /) -> onpt.NDArray[onp.generic]:
        R"""
        Argsort nodes by in degrees (large to small).
        """
        #
        return onp.argsort(-onp.sum(self.degree_ins, axis=1), kind="stable")

    def argsort_edge_degree_in(
        self,
        f: Callable[
            [onpt.NDArray[onp.generic], onpt.NDArray[onp.generic]],
            onpt.NDArray[onp.generic],
        ],
        /,
    ) -> onpt.NDArray[onp.generic]:
        R"""
        Argsort edges by the maxmimum of its source and destination degree
        scores (large to small).
        In most cases, degree score is the maximum of source and destination
        degrees.
        """
        #
        return (
            onp.argsort(
                -f(
                    self.degree_ins[self.edge_srcs],
                    self.degree_ins[self.edge_dsts],
                ),
            )
        )

    def fitsplit(
        self,
        proportion: Tuple[int, int, int], priority: Tuple[int, int, int],
        spindle: str,
        /,
    ) -> Tuple[
        onpt.NDArray[onp.generic], onpt.NDArray[onp.generic],
        onpt.NDArray[onp.generic],
    ]:
        R"""
        Split data indices into training, validation and test indices along
        dimension identifier by given name.
        """
        # Node split should ensure degree balance.
        # Time split just do the directly cut on spindle axis.
        if self.dset_desc and 'coarsened' in self.dset_desc and hasattr(self, "coars_train_indices"):
            print("Using precomputed coarsened split indices.")
            return (
                self.coars_train_indices, self.coars_valid_indices, []
            )
            
        #
        indices = {
            "node": self.argsort_node_degree_in(),
            "time": onp.arange(self.num_windows),
        }[spindle]
        
        # filter out unlabled nodes for classification task
        if hasattr(self, "lab_avail_nodes") and self.lab_avail_nodes is not None:
            lab_avail_nodes = self.lab_avail_nodes
            indices = keep_order_in_A(indices, lab_avail_nodes)
        
        if self.task_type == CLF_TASK and spindle == "node":
            node_labels = self.node_labels[indices]
            
            (indices_train, indices_valid, indices_test) = (
                stratified_fitsplit(
                    indices, node_labels, list(proportion), list(priority)
                )
            )
        else:
            try:
                (indices_train, indices_valid, indices_test) = (
                    fitsplit(
                        indices, list(proportion), list(priority)
                    )
                )
            except Exception as e:
                print(f'fitsplit error: {e}, fallback to manual split in specified indices order.')
                (train_size, valid_size, test_size) = proportion 
                scale = len(indices) / sum(proportion) 
                train_size = int(train_size * scale)
                valid_size = int(valid_size * scale) + train_size
                test_size = len(indices) # the rest to test
                indices_train = indices[0:train_size]
                indices_valid = indices[train_size:valid_size]
                indices_test = indices[valid_size:test_size]
                assert len(indices_train) + len(indices_valid) + len(indices_test) == len(indices), \
                    "split sizes do not match the total size."

        if self.WIN_SHIFT > 0:
            #
            (train_size, valid_size, test_size) = proportion
            scale = self.num_windows // sum(proportion)
            train_size = train_size * scale
            valid_size = valid_size * scale + train_size
            test_size = test_size * scale + valid_size
            indices_train = onp.arange(0, train_size)
            indices_valid = onp.arange(train_size, valid_size)
            indices_test = onp.arange(valid_size, test_size)
        return (indices_train, indices_valid, indices_test)

    def normalizeby(
        self,
        indices: onpt.NDArray[onp.generic], spindle: str,
        /,
    ) -> List[List[Tuple[float, float]]]:
        R"""
        Normalize data using statistics from indices.
        """
        # We may need to additionally normalize on time dimense for edge
        # features.
        if spindle == "node":
            #
            fromarray_buf = self.edge_feats # Brain10 [(154094, 1), (164190, 1), ...]
        elif spindle == "time":
            #
            fromarray_buf = [self.edge_feats[i] for i in indices.tolist()]
        else:
            # UNEXPECT:
            # Unknown normalization.
            raise NotImplementedError(
                "Unkonw normalization spindle \"{:s}\".".format(spindle),
            )
    
        normalize_func = {
            "normalize": normalize,
            "standardize": standardize,
        }[self.normalizer]
        
        fromarray = onp.concatenate(fromarray_buf, axis=0)
        edge_feats_new = []
        
        for edge_snap_feats in self.edge_feats:
            #
            (edge_snap_feats, edge_factors) = (
                normalize_func(edge_snap_feats, fromarray, axis=0)
            )
            edge_feats_new.append(edge_snap_feats)

        if not self.meaningless_edge_feats:
            #
            self.edge_feats = edge_feats_new

        # We may need to additionally normalize on time dimense for node
        # features.
        vdim = {"node": 0, "time": 2}[spindle]
        axis = (0, 2) if self.dyn_node_feats else (0,)

        if spindle == "node":
            fromarray = self.node_feats
        elif spindle == "time":
            fromarray = onp.take(
                self.node_feats, cast(onpt.NDArray[onp.int64], indices),
                vdim,
            )
        else:
            # UNEXPECT:
            # Unknown normalization.
            raise NotImplementedError(
                "Unkonw normalization spindle \"{:s}\".".format(spindle),
            )
        
        (self.node_feats, node_factors) = (
            normalize_func(
                self.node_feats,
                fromarray,
                axis=axis,
            )
        )

        return [edge_factors, node_factors]


    def distrep(self, /, *, n: int) -> str:
        R"""
        Get data distribution representation of the class.
        """
        #
        axis = lambda flag: (0, 2) if flag else (0,)
        fade = (
            lambda string, flag: (
                "\n".join(
                    "\x1b[90m" + line + "\x1b[0m"
                    for line in string.split("\n")
                )
                if flag else
                string
            )
        )

        #
        return (
            info5(
                {
                    "Data (Dist)ribute": {
                        "Edge Feat": (
                            fade(
                                distrep(
                                    onp.concatenate(self.edge_feats, axis=0),
                                    axis=0, n=n,
                                )[0],
                                self.meaningless_edge_feats,
                            )
                        ),
                        "Edge Label": (
                            fade(
                                distrep(
                                    onp.concatenate(self.edge_labels, axis=0),
                                    axis=0, n=n,
                                )[0],
                                self.meaningless_edge_labels,
                            )
                        ),
                        "Node Feat": (
                            fade(
                                distrep(
                                    self.node_feats,
                                    axis=axis(self.dyn_node_feats), n=n,
                                )[0],
                                self.meaningless_node_feats,
                            )
                        ),
                        "Node Label": (
                            fade(
                                distrep(
                                    self.node_labels,
                                    axis=axis(self.dyn_node_labels), n=n,
                                )[0],
                                self.meaningless_node_labels,
                            )
                        ),
                    }
                },
            )
        )

    def pin(self, batch_size: int, /) -> MEMPIN:
        R"""
        Always-shared data to be pinned into device memory.
        Shared data should not differentiate input and target.
        
        NOTE: current implementation only supports for single-window case.
        """
        # Corner case optimization.
        if self.winaggr == self.WINAGG_NONE:
            #
            return []
        if (
            len(self) == 1 and self.node_feat_size == 20
            and self.num_times == 12
        ):
            #
            corner_case = "Brain10"
        elif (
            len(self) == 1 and self.node_feat_size == 100
            and self.num_times == 10
        ):
            #
            corner_case = "DBLP5"
        elif (
            len(self) == 1 and self.node_feat_size == 20
            and self.num_times == 10
        ):
            #
            corner_case = "Reddit4"
        elif (
            len(self) == 1 and self.node_feat_size == 73
            and 20 <= self.num_times <= 61
        ):
            #
            corner_case = "IMDB"
        else:
            #
            self.cached = []
            return []
        if self.winaggr != self.WINAGG_DENSE:
            # EXPECT:
            # Open to be finisied.
            raise RuntimeError("Unimplemented corner case.")

        #
        corner_case_desc = "DynEdgeAgg~{:s}~T{:d}N{:d}F{:d}".format(
            self.dset_desc,
            self.num_times,
            self.num_nodes,
            self.node_feat_size
        )
        path = ".cache/{:s}.cache".format(corner_case_desc)
        if os.path.isfile(path):
            #
            self.cached = torch.load(path)
            return []
        else:
            print(f'Preparing corner case cache: {corner_case} ...')
            os.makedirs(".cache", exist_ok=True)

        # Get input and target window ties.
        input_tie1 = 0
        input_tie2 = input_tie1 + self.window_history_size
        target_tie1 = input_tie2
        target_tie2 = target_tie1 + self.window_future_size
        input_begin = min(input_tie1, input_tie2)
        input_end = max(input_tie1, input_tie2)
        target_begin = min(target_tie1, target_tie2)
        target_end = max(target_tie1, target_tie2)

        # Get edge timestamp data of current batch.
        edge_times_abs_input = self.edge_times_abs[input_begin:input_end]
        edge_times_rel_input = (
            [
                onp.full_like(
                    edge_times_abs_it,
                    (
                        onp.max(edge_times_abs_input[-1])
                        - onp.min(edge_times_abs_it)
                    ),
                )
                for edge_times_abs_it in edge_times_abs_input
            ]
        )
        edge_times_inc_input = self.edge_times_inc[input_begin:input_end]

        # Get node timestamp data of current batch.
        node_times_abs_input = self.node_times_abs[:, input_begin:input_end]
        node_times_rel_input = (
            onp.reshape(node_times_abs_input[:, -1], (self.num_nodes, 1))
            - node_times_abs_input
        )
        node_times_inc_input = self.node_times_inc[:, input_begin:input_end]

        # Collect unique edge identifiers as integers.
        node_max = (
            max(
                onp.max(edge_tuples_it).item()
                for edge_tuples_it in (
                    self.edge_tuples[input_begin:input_end]
                )
            )
        )
        edge_ints = (
            [
                edge_srcs_it * (node_max + 1) + edge_dsts_it
                for (edge_srcs_it, edge_dsts_it) in (
                    zip(
                        self.edge_srcs[input_begin:input_end],
                        self.edge_dsts[input_begin:input_end],
                    )
                )
            ]
        )
        edge_uniqints = onp.sort(onp.unique(onp.concatenate(edge_ints)))

        # Safety check.
        if self.edge_feat_size > 1:
            # UNEXPECT:
            # Only scalar temporal edge feature is aggregatable.
            raise NotImplementedError(
                "Only scalar temporal edge feature is aggregatable.",
            )
        if (
            any(
                onp.any(onp.less(edge_feats_it, 0.0))
                for edge_feats_it in self.edge_feats
            )
        ):
            # UNEXPECT:
            # Padding value can not exist in raw edge feature.
            raise NotImplementedError(
                "negative values are used as padding values in dense "
                "feature aggregation, thus can not be part of edge "
                "feature.",
            )

        # Generate dense edge feature and appearance aggregation as sparse
        # matrices.
        aggr_eids = onp.concatenate(edge_ints)
        aggr_tids = (
            onp.concatenate(
                [
                    onp.full_like(edge_ints_it, t)
                    for (t, edge_ints_it) in enumerate(edge_ints)
                ],
            )
        )
        aggr_vals_feat = (
            onp.reshape(
                onp.concatenate(self.edge_feats[input_begin:input_end]),
                (len(aggr_eids),),
            )
        )
        aggr_vals_appr = onp.ones((len(aggr_eids)), dtype=bool)
        aggr_spmat_feat = (
            osparse.coo_matrix(
                (aggr_vals_feat, (aggr_eids, aggr_tids)),
                shape=(
                    onp.max(aggr_eids).item() + 1,
                    self.window_history_size,
                ),
            ).tocsc()
        )
        aggr_spmat_appr = (
            osparse.coo_matrix(
                (aggr_vals_appr, (aggr_eids, aggr_tids)),
                shape=(
                    onp.max(aggr_eids).item() + 1,
                    self.window_history_size,
                ),
            ).tocsc()
        )

        #
        aggr_srcs = edge_uniqints // (node_max + 1)
        aggr_dsts = edge_uniqints % (node_max + 1)
        aggr_tuples = onp.stack((aggr_srcs, aggr_dsts))

        # Convert to dense feature matrix of non-trivial feature dynamics.
        # Use -1.0 as default padding value.
        aggr_feats_buf = []
        aggr_apprs_buf = []
        print('converting to dense feature...')
        tic = time.time()
        for eid in tqdm(edge_uniqints.tolist()):
            #
            aggr_feats_buf.append(aggr_spmat_feat[eid].toarray())
            aggr_apprs_buf.append(aggr_spmat_appr[eid].toarray())
            
        # def densify(eid):
        #     return (aggr_spmat_feat[eid].toarray(), aggr_spmat_appr[eid].toarray())
        
        # results = p_map(densify, edge_uniqints.tolist(), **{"num_cpus": 5})
        # aggr_feats_buf = [arri[0] for arri in results]
        # aggr_apprs_buf = [arri[1] for arri in results]

        aggr_feats = (
            onp.reshape(
                onp.stack(aggr_feats_buf),
                (len(edge_uniqints), 1, self.window_history_size),
            )
        )
        aggr_apprs = (
            onp.reshape(
                onp.stack(aggr_apprs_buf),
                (len(edge_uniqints), 1, self.window_history_size),
            )
        )
        aggr_feats[onp.logical_not(aggr_apprs)] = -1.0
        toc = time.time()
        print(f'converting time: {(toc - tic)/60} mins')
        # We should not have any edge labels.
        if not self.meaningless_edge_labels:
            # UNEXPECT
            # We can not aggregate temporal edge labels for now.
            raise NotImplementedError(
                "We can not aggregate temporal edge labels for now.",
            )
        aggr_labels = onp.array(0)

        # Aggregated timestamps will not care the raw appearance since it
        # will regard all edges appearing once as existing all the time.
        aggr_times_abs = (
            onp.concatenate(
                [
                    onp.unique(time_abs)
                    for time_abs in edge_times_abs_input
                ],
            )
        )
        aggr_times_rel = onp.max(aggr_times_abs) - aggr_times_abs
        aggr_times_inc = (
            onp.concatenate(
                (
                    onp.zeros((1,), dtype=aggr_times_abs.dtype),
                    aggr_times_abs[1:] - aggr_times_abs[:-1],
                ),
            )
        )
        aggr_times_abs = onp.tile(aggr_times_abs, (len(edge_uniqints), 1))
        aggr_times_rel = onp.tile(aggr_times_rel, (len(edge_uniqints), 1))
        aggr_times_inc = onp.tile(aggr_times_inc, (len(edge_uniqints), 1))

        # Rename as new edge data.
        edge_tuples_numpy = aggr_tuples
        edge_feats_numpy = aggr_feats
        edge_labels_numpy = aggr_labels
        edge_ranges_numpy = onp.array([0, len(edge_feats_numpy)])

        # Collect edge timestamp data.
        edge_times_numpy_buf = []
        if self.ABS in self.timestamped_edge_times:
            #
            edge_times_numpy_buf.append(aggr_times_abs)
        if self.REL in self.timestamped_edge_times:
            #
            edge_times_numpy_buf.append(aggr_times_rel)
        if self.INC in self.timestamped_edge_times:
            #
            edge_times_numpy_buf.append(aggr_times_inc)
        if len(edge_times_numpy_buf) == 0:
            # Use number of timestamps as place holder.
            edge_times_numpy = onp.array(self.num_times)
        else:
            # Merge all timestamp data into a single array.
            # Put node axis at first for batching.
            edge_times_numpy = onp.stack(edge_times_numpy_buf)
            edge_times_numpy = onp.transpose(edge_times_numpy, (1, 0, 2))

        # A corner case.
        self.cached = (
            [
                edge_tuples_numpy, edge_feats_numpy, edge_labels_numpy,
                edge_ranges_numpy, edge_times_numpy,
            ]
        )
        torch.save(self.cached, path)
        return []

    def __len__(self, /) -> int:
        R"""
        Length of the class.
        """
        #
        return self.num_windows

    def idx_to_timesteps(self, idx: int):
        input_tie1 = idx if self.WIN_SHIFT < 0 else idx * self.WIN_SHIFT
        input_tie2 = input_tie1 + self.window_history_size
        target_tie1 = input_tie2
        target_tie2 = (
            target_tie1 + self.window_future_size
            if self.WIN_SHIFT < 0 else
            target_tie1 - 1
        )

        input_begin = min(input_tie1, input_tie2)
        input_end = max(input_tie1, input_tie2)
        target_begin = min(target_tie1, target_tie2)
        target_end = max(target_tie1, target_tie2)
        return (input_begin, input_end, target_begin, target_end)
    
    def coarsen_window_dyngraph_baseline(self, 
            window_timestamps,
            window_edge_srcs, window_edge_dsts, 
            window_edge_feats, window_edge_labels,
            window_node_feats, window_node_labels,
            alg_version: str,
            split_indices: Tuple[List[int], List[int], List[int]] = None,
            pre_run_results_path: str = None,
            
            spec_gap_method: str = 'approx', 
            coars_op: str = 'PtAP', 
            rw_sl: bool = True, 
            ef_rw_method: str = 'avg'
        ) -> Tuple[
            List[onpt.NDArray[onp.generic]],
            List[onpt.NDArray[onp.generic]],
            List[onpt.NDArray[onp.generic]],
            List[onpt.NDArray[onp.generic]],
            onpt.NDArray[onp.generic],
            onpt.NDArray[onp.generic],
        ]:
        R"""
        Load pre-run results (coarsening matrix), coarsen the dynamic graph.
        """
        
        if pre_run_results_path is None or not os.path.isfile(pre_run_results_path):
            raise ValueError("pre_run_results_path must be a valid file path.", pre_run_results_path)
        print(f"Loading pre-run results from {pre_run_results_path} ...")
        pre_run_results = onp.load(pre_run_results_path, allow_pickle=True)
        P = pre_run_results['P']  # the coarsening matrix
        K = pre_run_results['Nprime']  # the number of nodes to be merged
        #
        num_nodes = P.shape[0]
        num_times = len(window_edge_srcs)
        
        init_specgaps_supra, *_ = \
            get_supra_spectral_gap(
                window_edge_srcs, window_edge_dsts, onp.eye(num_nodes), 
                num_times, num_nodes, num_nodes
            )
        init_specgap_supra = agg_score(init_specgaps_supra, agg_method_cc='min', 
                                    agg_method_temp=None, skip_zero=True)
        #
        init_specgaps_snap, *_ = get_snapshots_spectral_gap(
            window_edge_srcs, window_edge_dsts, onp.eye(num_nodes),
            num_times, num_nodes, num_nodes
        )
        
        init_specgap_snap = agg_score(init_specgaps_snap, agg_method_cc='min', 
                                      agg_method_temp='min', skip_zero=True)
        
        wandb.log({
            "initial_spectral_gap_supra": init_specgap_supra,
            "initial_spectral_gap_snaps": init_specgap_snap,
        })
        #
        all_Ps = [P]
        print(f"Loaded pre-run results from {pre_run_results_path}.")
        # construct the aggregated graph
        if self.dyn_node_feats:
            num_nodes, _, num_times = window_node_feats.shape
        else:
            num_nodes, _ = window_node_feats.shape
        
        (
            window_edge_srcs, window_edge_dsts, window_edge_feats, 
            window_edge_labels, window_node_feats, window_node_labels, 
            supernode_label_mask, coars_train_indices, coars_valid_indices
        ) = coarsen_graph(
            num_times, num_nodes, K, P, split_indices,
            window_edge_srcs, window_edge_dsts,
            window_edge_feats, window_edge_labels,
            window_node_feats, window_node_labels,
            self.dyn_node_feats, self.dyn_node_labels,
            self.node_label_size, self.meaningless_edge_labels, 
            coarsen_op=coars_op,
            reweight_selfloops=rw_sl, 
            edge_feat_reweight_method=ef_rw_method,
            task_type=self.task_type,
        )
        
        num_edges_after = [len(es) for es in window_edge_srcs]
        if any([ne <= NE_THRESH for ne in num_edges_after]):
            print(f"Warning: some snapshots have too few edges "
                  f"after coarsening: {num_edges_after} (threshold: {NE_THRESH}).")
            print("Logging spectral_gap_after_coarsening as -inf and exiting.")
            wandb.log({
                "spectral_gap_after_coarsening": -onp.inf
            })
            exit(0)
            
        spectral_gaps_supra, *_ = get_supra_spectral_gap(
            window_edge_srcs, window_edge_dsts, onp.eye(K), 
            num_times, K, K
        )
        
        spectral_gap_supra = agg_score(spectral_gaps_supra, agg_method_cc='min', 
                                 agg_method_temp=None, skip_zero=True)
        
        # init_spectral_gaps, init_eigenvecs, As, Ls, cc_sizess, cc_nodess
        spectral_gaps_snaps, *_= get_snapshots_spectral_gap(
            window_edge_srcs, window_edge_dsts, onp.eye(K), 
            num_times, K, K
        )

        spectral_gap_snaps = agg_score(spectral_gaps_snaps, agg_method_cc='min', 
                                 agg_method_temp='min', skip_zero=True)
        
        print(f"Super-node sizes: ", P.sum(axis=0).tolist())
        #
        delta_str_supra = "increased" if spectral_gap_supra - init_specgap_supra > 0 else "decreased"
        delta_sg_supra = abs(spectral_gap_supra - init_specgap_supra)
        delta_percent_supra = (
            delta_sg_supra / init_specgap_supra * 100.0
        )
        delta_str_supra = f"{delta_str_supra} by {delta_sg_supra:.8f} ({delta_percent_supra:.2f}%)"
        print(f"Spectral gap (supra) after merging, "
              f"num nodes from {num_nodes} to {K}: {spectral_gap_supra:.8f}, {delta_str_supra}")
        #
        delta_str_snaps = "increased" if spectral_gap_snaps - init_specgap_snap > 0 else "decreased"
        delta_sg_snaps = abs(spectral_gap_snaps - init_specgap_snap)
        delta_percent_snaps = (
            delta_sg_snaps / init_specgap_snap * 100.0
        )
        delta_str_snaps = f"{delta_str_snaps} by {delta_sg_snaps:.8f} ({delta_percent_snaps:.2f}%)"
        print(f"Spectral gap (snaps) after merging, "
              f"num nodes from {num_nodes} to {K}: {spectral_gap_snaps:.8f}, {delta_str_snaps}")
        
        wandb.log({
            "spectral_gap_after_coarsening_supra": spectral_gap_supra,
            "spectral_gap_after_coarsening_snaps": spectral_gap_snaps,
            "spectral_gap_delta_supra": spectral_gap_supra - init_specgap_supra,
            "spectral_gap_delta_percent_supra": delta_percent_supra,
            "spectral_gap_delta_snaps": spectral_gap_snaps - init_specgap_snap,
            "spectral_gap_delta_percent_snaps": delta_percent_snaps,
        })
        window_edge_tuples = (
            [
                onp.stack((edge_snap_srcs, edge_snap_dsts))
                for (edge_snap_srcs, edge_snap_dsts) in (
                    zip(window_edge_srcs, window_edge_dsts)
                )
            ]
        )
        num_times = len(window_edge_tuples)
        window_edge_times_abs, window_edge_times_inc, \
        window_node_times_abs, window_node_times_inc = (
            self._timestamping(
                window_edge_tuples, window_timestamps, num_times, K
            )
        )

        return (
            window_edge_times_abs, window_edge_times_inc,
            window_node_times_abs, window_node_times_inc,
            window_edge_tuples, window_edge_srcs, window_edge_dsts,
            window_edge_feats, window_edge_labels,
            window_node_feats, window_node_labels, 
            K, supernode_label_mask, 
            coars_train_indices, coars_valid_indices, all_Ps
        )
    
    def coarsen_window_dyngraph_random(self,
        metaset_ss_num: int,
        window_timestamps,
        window_edge_srcs, window_edge_dsts,
        window_edge_feats, window_edge_labels,
        window_node_feats, window_node_labels,
        alg_version: str,
        Nprime=0.95,
        split_indices: Tuple[List[int], List[int], List[int]] = None,
        spec_gap_method: str = 'approx',
        coars_op: str = 'PtAP',
        rw_sl: bool = True,
        ef_rw_method: str = 'avg'
    ) -> Tuple[
        List[onpt.NDArray[onp.generic]],
        List[onpt.NDArray[onp.generic]],
        List[onpt.NDArray[onp.generic]],
        List[onpt.NDArray[onp.generic]],
        onpt.NDArray[onp.generic],
        onpt.NDArray[onp.generic],
    ]:
        R"""
        Coarsen the graph using random edge merging.
        """
        # construct the aggregated graph
        if self.dyn_node_feats:
            num_nodes, _, num_times = window_node_feats.shape
        else:
            num_nodes, _ = window_node_feats.shape
        
        if Nprime < 1.0: Nprime = round(num_nodes * Nprime)
        
        print(f"Using coarsen_window_dyngraph_random with Nprime={Nprime} ({Nprime/num_nodes*100:.2f}%)")
        init_specgaps_supra, *_ = \
            get_supra_spectral_gap(
                window_edge_srcs, window_edge_dsts, onp.eye(num_nodes), 
                num_times, num_nodes, num_nodes
            )
        init_specgap_supra = agg_score(init_specgaps_supra, agg_method_cc='min', 
                                    agg_method_temp=None, skip_zero=True)
        #
        init_specgaps_snap, *_ = get_snapshots_spectral_gap(
            window_edge_srcs, window_edge_dsts, onp.eye(num_nodes),
            num_times, num_nodes, num_nodes
        )
        
        init_specgap_snap = agg_score(init_specgaps_snap, agg_method_cc='min', 
                                      agg_method_temp='min', skip_zero=True)
        
        wandb.log({
            "initial_spectral_gap_supra": init_specgap_supra,
            "initial_spectral_gap_snaps": init_specgap_snap,
        })
        #
        L = num_nodes - Nprime
        assert L > 0, "Nprime must be less than the number of nodes in the original graph."
        
        check_pairs = set()
        for t in range(num_times):
            srcs = window_edge_srcs[t]
            dsts = window_edge_dsts[t]
            # find all unique pairs of nodes that are connected by an edge
            edges = onp.unique(onp.stack((srcs, dsts), axis=1), axis=0)
            for (u, v) in edges:
                if u != v and (v, u) not in check_pairs: 
                    check_pairs.add((u, v))

        check_pairs = list(check_pairs)
        check_pairs = [(pair, 0.0) for pair in check_pairs]  # add dummy score for shuffling
        indices = onp.random.permutation(len(check_pairs))
        sorted_edges = [check_pairs[i] for i in indices]
        
        mapping, merged_pairs = map_to_supernodes(sorted_edges, num_nodes, Nprime)
        print(f"Merged pairs: {[(e, round(v, 8)) for (e, v) in merged_pairs[:20]]}", 
              f"... (total {len(merged_pairs)} pairs merged)" if len(merged_pairs) > 20 else "")
        
        supernodes = set(mapping.values())
        if max(supernodes) >= Nprime:
            print("Number of clusters must be less than Nprime.")
            import pdb; pdb.set_trace()
            
        # construct the partition matrix P of shape (N, K)
        K = Nprime
        P = onp.zeros((num_nodes, K))
        for u, cluster_id in mapping.items():
            P[u, cluster_id] = 1.0
        # assert each row of P sums to 1
        assert onp.all(onp.sum(P, axis=1) == 1.0), "Each row of P must sum to 1."
        all_Ps = [P]
        
        (
            coars_window_edge_srcs, coars_window_edge_dsts, coars_window_edge_feats, 
            coars_window_edge_labels, coars_window_node_feats, coars_window_node_labels, 
            supernode_label_mask, coars_train_indices, coars_val_indices
        ) = coarsen_graph(
            num_times, num_nodes, K, P, split_indices,
            window_edge_srcs, window_edge_dsts,
            window_edge_feats, window_edge_labels,
            window_node_feats, window_node_labels,
            self.dyn_node_feats, self.dyn_node_labels,
            self.node_label_size, self.meaningless_edge_labels, 
            coarsen_op=coars_op,
            reweight_selfloops=rw_sl, 
            edge_feat_reweight_method=ef_rw_method,
            task_type=self.task_type,
        )
        
        num_edges_after = [len(es) for es in coars_window_edge_srcs]
        if any([ne <= NE_THRESH for ne in num_edges_after]):
            print(f"Warning: some snapshots have too few edges "
                  f"after coarsening: {num_edges_after} (threshold: {NE_THRESH}).")
            print("Logging spectral_gap_after_coarsening as -inf and exiting.")
            wandb.log({
                "spectral_gap_after_coarsening": -onp.inf
            })
            exit(0)
        #
        spectral_gaps_supra, *_ = get_supra_spectral_gap(
            coars_window_edge_srcs, coars_window_edge_dsts, onp.eye(K), 
            num_times, K, K
        )
        spectral_gap_supra = agg_score(spectral_gaps_supra, agg_method_cc='min', 
                                 agg_method_temp=None, skip_zero=True)
        #
        spectral_gaps_snaps, *_= get_snapshots_spectral_gap(
            coars_window_edge_srcs, coars_window_edge_dsts, onp.eye(K), 
            num_times, K, K
        )
        spectral_gap_snaps = agg_score(spectral_gaps_snaps, agg_method_cc='min', 
                                 agg_method_temp='min', skip_zero=True)
        #
        print(f"Super-node sizes: ", P.sum(axis=0).tolist()[:20], " ..." if K > 20 else "")
        print(f"Super-node sizes: ", Counter(P.sum(axis=0).tolist()))
        #
        delta_str_supra = "increased" if spectral_gap_supra - init_specgap_supra > 0 else "decreased"
        delta_sg_supra = abs(spectral_gap_supra - init_specgap_supra)
        delta_percent_supra = (
            delta_sg_supra / init_specgap_supra * 100.0
        )
        delta_str_supra = f"{delta_str_supra} by {delta_sg_supra:.8f} ({delta_percent_supra:.2f}%)"
        print(f"Spectral gap (supra) after merging {len(merged_pairs)} node pairs, "
              f"num nodes from {num_nodes} to {K}: {spectral_gap_supra:.8f}, {delta_str_supra}")
        #
        delta_str_snaps = "increased" if spectral_gap_snaps - init_specgap_snap > 0 else "decreased"
        delta_sg_snaps = abs(spectral_gap_snaps - init_specgap_snap)
        delta_percent_snaps = (
            delta_sg_snaps / init_specgap_snap * 100.0
        )
        delta_str_snaps = f"{delta_str_snaps} by {delta_sg_snaps:.8f} ({delta_percent_snaps:.2f}%)"
        print(f"Spectral gap (snaps) after merging {len(merged_pairs)} node pairs, "
              f"num nodes from {num_nodes} to {K}: {spectral_gap_snaps:.8f}, {delta_str_snaps}")
        #
        wandb.log({
            "spectral_gap_after_coarsening_supra": spectral_gap_supra,
            "spectral_gap_after_coarsening_snaps": spectral_gap_snaps,
            "spectral_gap_delta_supra": spectral_gap_supra - init_specgap_supra,
            "spectral_gap_delta_percent_supra": delta_percent_supra,
            "spectral_gap_delta_snaps": spectral_gap_snaps - init_specgap_snap,
            "spectral_gap_delta_percent_snaps": delta_percent_snaps,
        })
        
        window_edge_tuples = (
            [
                onp.stack((edge_snap_srcs, edge_snap_dsts))
                for (edge_snap_srcs, edge_snap_dsts) in (
                    zip(coars_window_edge_srcs, coars_window_edge_dsts)
                )
            ]
        )
        num_times = len(window_edge_tuples)
        window_edge_times_abs, window_edge_times_inc, \
        window_node_times_abs, window_node_times_inc = (
            self._timestamping(
                window_edge_tuples, window_timestamps, num_times, K
            )
        )
        
        return (
            window_edge_times_abs, window_edge_times_inc,
            window_node_times_abs, window_node_times_inc,
            window_edge_tuples, 
            coars_window_edge_srcs, coars_window_edge_dsts,
            coars_window_edge_feats, coars_window_edge_labels,
            coars_window_node_feats, coars_window_node_labels, 
            K, supernode_label_mask, 
            coars_train_indices, coars_val_indices, all_Ps
        )
    
    def coarsen_window_dyngraph_supra(self,
        metaset_ss_num: int,
        window_timestamps,
        window_edge_srcs, window_edge_dsts, 
        window_edge_feats, window_edge_labels,
        window_node_feats, window_node_labels,
        alg_version: str,
        Nprime=0.95,
        split_indices: Tuple[List[int], List[int], List[int]] = None,
        
        spec_gap_method: str = 'approx', 
        coars_op: str = 'PtAP', 
        rw_sl: bool = True, 
        ef_rw_method: str = 'avg'
    ) -> Tuple[
        List[onpt.NDArray[onp.generic]],
        List[onpt.NDArray[onp.generic]],
        List[onpt.NDArray[onp.generic]],
        List[onpt.NDArray[onp.generic]],
        onpt.NDArray[onp.generic],
        onpt.NDArray[onp.generic],
    ]:
        R"""
        Coarsen the graph.
        window_timestamps: onpt.NDArray[onp.generic] of shape (num_times, ), the timestamps of each snapshot.
        
        window_edge_srcs: List[onpt.NDArray[onp.generic]] of shape (num_times, num_edges), 
            where each element is the source nodes of edges at time t.
        window_edge_dsts: List[onpt.NDArray[onp.generic]] of shape (num_times, num_edges), 
            where each element is the destination nodes of edges at time t.
        window_edge_feats: List[onpt.NDArray[onp.generic]] of shape (num_times, num_edges, num_features), 
            where each element is the features of edges at time t.
        window_edge_labels: List[onpt.NDArray[onp.generic]] of shape (num_times, num_edges, edge_label_size), 
            where each element is the labels of edges at time t.
        window_node_feats: onpt.NDArray[onp.generic] of shape (num_nodes, num_features, num_times) if self.dyn_node_feats is true, 
            or onpt.NDArray[onp.generic] of shape (num_nodes, num_features) if self.dyn_node_feats is false,
            where each element is the features of nodes at time t.
        window_node_labels: onpt.NDArray[onp.generic] of shape (num_nodes, node_label_size, num_times) if self.dyn_node_labels is true, 
            or onpt.NDArray[onp.generic] of shape (num_nodes, node_label_size) if self.dyn_node_labels is false, 
            where each element is the labels of nodes at time t.
        Nprime: float or int, the number of nodes left in the coarsened graph.
        """
        # construct the aggregated graph
        if self.dyn_node_feats:
            num_nodes, _, num_times = window_node_feats.shape
        else:
            num_nodes, _ = window_node_feats.shape
        
        if Nprime < 1.0: Nprime = round(num_nodes * Nprime)
        
        print(f"Using coarsen_window_dyngraph_supra with Nprime={Nprime} ({Nprime/num_nodes*100:.2f}%)")
        
        L = num_nodes - Nprime
        assert L > 0, "Nprime must be less than the number of nodes in the original graph."
        
        # compute the spectral gap of the current aggregated graph
        # construct the aggregated adjacency matrix A_agg
        init_spectral_gaps, init_eigenvecs, supA, supL, cc_sizess, cc_nodess = \
            get_supra_spectral_gap(
                window_edge_srcs, window_edge_dsts, onp.eye(num_nodes), 
                num_times, num_nodes, num_nodes
            )
        init_spectral_gap = agg_score(init_spectral_gaps, agg_method_cc='min', 
                                    agg_method_temp=None, skip_zero=True)
        print(f"Initial spectral gap: {init_spectral_gap:.8f}")
        wandb.log({"init_spectral_gap": init_spectral_gap})
        
        # loop K times to merge nodes
        all_Ps = []

        check_pairs = set()
        for t in range(num_times):
            srcs = window_edge_srcs[t]
            dsts = window_edge_dsts[t]
            # find all unique pairs of nodes that are connected by an edge
            edges = onp.unique(onp.stack((srcs, dsts), axis=1), axis=0)
            for (u, v) in edges:
                if u != v and (v, u) not in check_pairs: 
                    check_pairs.add((u, v))

        check_pairs = list(check_pairs)

        if len(check_pairs) == 0:
            print("No valid edges found for merging. Stopping early.")
            raise RuntimeError("No valid edges found for merging.")
        
        SS_NUM = metaset_ss_num
        if SS_NUM == -1: # use all pairs
            ss_idx = onp.arange(len(check_pairs))
        else:
            SS_NUM = max(L, metaset_ss_num)
            ss_idx = onp.random.choice(len(check_pairs), size=SS_NUM, replace=False)
            
        print(f"Sampling {len(ss_idx)} pairs from {len(check_pairs)} pairs for checking.")
        ss_check_pairs = [check_pairs[i] for i in ss_idx]
        
        N = num_nodes
        K = num_nodes - 1
        init_info = (init_spectral_gaps, init_eigenvecs, supA, supL)
        links_score = get_links_scores_over_coarsened_supra_dyngraph(
            num_times, N, K, window_edge_srcs, window_edge_dsts,
            init_info, ss_check_pairs, SPEC_GAP_METHOD=spec_gap_method
        )
        sorted_edges = sorted(links_score.items(), key=lambda item: item[1], reverse=True)

        # find clusters of merged nodes
        mapping, merged_pairs = map_to_supernodes(sorted_edges, num_nodes, Nprime)
        
        supernodes = set(mapping.values())
        if max(supernodes) >= Nprime:
            print("Number of clusters must be less than Nprime.")
            import pdb; pdb.set_trace()
            
        # construct the partition matrix P of shape (N, K)
        K = Nprime
        P = onp.zeros((num_nodes, K))
        for u, cluster_id in mapping.items():
            P[u, cluster_id] = 1.0
        # assert each row of P sums to 1
        assert onp.all(onp.sum(P, axis=1) == 1.0), "Each row of P must sum to 1."
        all_Ps.append(P)
        (
            coars_window_edge_srcs, coars_window_edge_dsts, coars_window_edge_feats, 
            coars_window_edge_labels, coars_window_node_feats, coars_window_node_labels, 
            supernode_label_mask, coars_train_indices, coars_val_indices
        ) = coarsen_graph(
            num_times, num_nodes, K, P, split_indices,
            window_edge_srcs, window_edge_dsts,
            window_edge_feats, window_edge_labels,
            window_node_feats, window_node_labels,
            self.dyn_node_feats, self.dyn_node_labels,
            self.node_label_size, self.meaningless_edge_labels, 
            coarsen_op=coars_op,
            reweight_selfloops=rw_sl, 
            edge_feat_reweight_method=ef_rw_method,
            task_type=self.task_type,
        )

        num_edges_after = [len(es) for es in coars_window_edge_srcs]
        if any([ne <= NE_THRESH for ne in num_edges_after]):
            print(f"Warning: some snapshots have too few edges "
                  f"after coarsening: {num_edges_after} (threshold: {NE_THRESH}).")
            print("Logging spectral_gap_after_coarsening as -inf and exiting.")
            wandb.log({
                "spectral_gap_after_coarsening": -onp.inf
            })
            exit(0)
            
        spectral_gaps, *_ = get_supra_spectral_gap(
            coars_window_edge_srcs, coars_window_edge_dsts, onp.eye(K), 
            num_times, K, K
        )
        spectral_gap = agg_score(spectral_gaps, agg_method_cc='min', 
                                agg_method_temp=None, skip_zero=True)
        
        delta_str = "increased" if spectral_gap - init_spectral_gap > 0 else "decreased"
        delta_sg = abs(spectral_gap - init_spectral_gap)
        delta_percent = (
            delta_sg / init_spectral_gap * 100.0
        )
        delta_str = f"{delta_str} by {delta_sg:.8f} ({delta_percent:.2f}%)"
        print(f"Spectral gap after merging {L} node pairs, "
              f"num nodes from {num_nodes} to {K}: {spectral_gap:.8f}, {delta_str}")
        print(f"Merged pairs: {[(e, round(v, 8)) for (e, v) in merged_pairs]}")
        print(f"Super-node sizes: ", P.sum(axis=0).tolist())
        wandb.log({
            "spectral_gap_after_coarsening": spectral_gap,
            "spectral_gap_delta": spectral_gap - init_spectral_gap,
            "spectral_gap_delta_percent": delta_percent,
        })
    
        window_edge_tuples = (
            [
                onp.stack((edge_snap_srcs, edge_snap_dsts))
                for (edge_snap_srcs, edge_snap_dsts) in (
                    zip(coars_window_edge_srcs, coars_window_edge_dsts)
                )
            ]
        )
        num_times = len(window_edge_tuples)
        window_edge_times_abs, window_edge_times_inc, \
        window_node_times_abs, window_node_times_inc = (
            self._timestamping(
                window_edge_tuples, window_timestamps, num_times, K
            )
        )
        
        return (
            window_edge_times_abs, window_edge_times_inc,
            window_node_times_abs, window_node_times_inc,
            window_edge_tuples, 
            coars_window_edge_srcs, coars_window_edge_dsts,
            coars_window_edge_feats, coars_window_edge_labels,
            coars_window_node_feats, coars_window_node_labels, 
            K, supernode_label_mask, 
            coars_train_indices, coars_val_indices, all_Ps
        )
    
    def coarsen_window_dyngraph_supra_refine(self,
        metaset_ss_num: int,
        window_timestamps,
        window_edge_srcs, window_edge_dsts, 
        window_edge_feats, window_edge_labels,
        window_node_feats, window_node_labels,
        alg_version: str,
        Nprime=0.95,
        split_indices: Tuple[List[int], List[int], List[int]] = None,
        
        spec_gap_method: str = 'approx', 
        coars_op: str = 'PtAP', 
        rw_sl: bool = True, 
        ef_rw_method: str = 'avg',
        refine_every: int = 0.1, # refine every 10% of the reduction ratio relative to original graph
    ) -> Tuple[
        List[onpt.NDArray[onp.generic]],
        List[onpt.NDArray[onp.generic]],
        List[onpt.NDArray[onp.generic]],
        List[onpt.NDArray[onp.generic]],
        onpt.NDArray[onp.generic],
        onpt.NDArray[onp.generic],
    ]:
        R"""
        Coarsen the graph.
        window_timestamps: onpt.NDArray[onp.generic] of shape (num_times, ), the timestamps of each snapshot.
        
        window_edge_srcs: List[onpt.NDArray[onp.generic]] of shape (num_times, num_edges), 
            where each element is the source nodes of edges at time t.
        window_edge_dsts: List[onpt.NDArray[onp.generic]] of shape (num_times, num_edges), 
            where each element is the destination nodes of edges at time t.
        window_edge_feats: List[onpt.NDArray[onp.generic]] of shape (num_times, num_edges, num_features), 
            where each element is the features of edges at time t.
        window_edge_labels: List[onpt.NDArray[onp.generic]] of shape (num_times, num_edges, edge_label_size), 
            where each element is the labels of edges at time t.
        window_node_feats: onpt.NDArray[onp.generic] of shape (num_nodes, num_features, num_times) if self.dyn_node_feats is true, 
            or onpt.NDArray[onp.generic] of shape (num_nodes, num_features) if self.dyn_node_feats is false,
            where each element is the features of nodes at time t.
        window_node_labels: onpt.NDArray[onp.generic] of shape (num_nodes, node_label_size, num_times) if self.dyn_node_labels is true, 
            or onpt.NDArray[onp.generic] of shape (num_nodes, node_label_size) if self.dyn_node_labels is false, 
            where each element is the labels of nodes at time t.
        Nprime: float or int, the number of nodes left in the coarsened graph.
        """
        # construct the aggregated graph
        if self.dyn_node_feats:
            NUM_NODES, _, NUM_TIMES = window_node_feats.shape
        else:
            NUM_NODES, _ = window_node_feats.shape
        
        if Nprime < 1.0: 
            Nprime_target = round(NUM_NODES * Nprime)
        else:
            Nprime_target = int(Nprime)
        
        print(f"Using coarsen_window_dyngraph_supra_refine with Nprime={Nprime_target} ({Nprime_target/NUM_NODES*100:.2f}%)")
        
        L = NUM_NODES - Nprime_target
        assert L > 0, "Nprime must be less than the number of nodes in the original graph."
        
        # compute the spectral gap of the current aggregated graph
        # construct the aggregated adjacency matrix A_agg
        init_spectral_gaps, init_eigenvecs, supA, supL, cc_sizess, cc_nodess = \
            get_supra_spectral_gap(
                window_edge_srcs, window_edge_dsts, onp.eye(NUM_NODES), 
                NUM_TIMES, NUM_NODES, NUM_NODES
            )
        init_spectral_gap = agg_score(init_spectral_gaps, agg_method_cc='min', 
                                    agg_method_temp=None, skip_zero=True)
        print(f"Initial spectral gap: {init_spectral_gap:.8f}")
        wandb.log({"init_spectral_gap": init_spectral_gap})
        before_coarsen_spectral_gap = init_spectral_gap
        # loop K times to merge nodes
        all_Ps = []
        merge_pairs_all = []
        
        Np = NUM_NODES
        reduction_step = round(NUM_NODES * refine_every)
        slack = reduction_step // 5  # allow some slack to avoid too small final step
        import pdb;pdb.set_trace()
        while Np > Nprime_target:
            Np_next = Np - reduction_step
            if Np_next <= Nprime_target + slack:
                Np_next = Nprime_target
                
            print(f"Coarsening from {Np} to {Np_next} nodes ...")
            # do coarsening from Np to Np_next
            check_pairs = set()
            for t in range(NUM_TIMES):
                srcs = window_edge_srcs[t]
                dsts = window_edge_dsts[t]
                # find all unique pairs of nodes that are connected by an edge
                edges = onp.unique(onp.stack((srcs, dsts), axis=1), axis=0)
                for (u, v) in edges:
                    if u != v and (v, u) not in check_pairs: 
                        check_pairs.add((u, v))

            check_pairs = list(check_pairs)

            if len(check_pairs) == 0:
                print("No valid edges found for merging. Stopping early.")
                raise RuntimeError("No valid edges found for merging.")
            
            init_info = (init_spectral_gaps, init_eigenvecs, supA, supL)
            links_score = get_links_scores_over_coarsened_supra_dyngraph(
                NUM_TIMES, Np, Np-1, window_edge_srcs, window_edge_dsts,
                init_info, check_pairs, SPEC_GAP_METHOD=spec_gap_method
            )
            # after looping through all edges, pick top L edges with highest spectral gap
            sorted_edges = sorted(links_score.items(), key=lambda item: item[1], reverse=True)
            # construct partion matrix P by merging the top L edges
            # for each merged cluster, map it to a new supernode, 
            # other nodes are mapped to a new unique supernode
            mapping, merged_pairs = map_to_supernodes(sorted_edges, Np, Np_next)
            merge_pairs_all.append(merged_pairs)
            
            supernodes = set(mapping.values())
            if max(supernodes) > Np_next:
                print(f"Warning: Number of clusters {max(supernodes)} must be <= Nprime_next {Np_next}.")
                
            Np_next = len(supernodes) # update Np_next based on actual clusters formed
            # construct the partition matrix P of shape (N, K)
            P = onp.zeros((Np, Np_next))
            for u, cluster_id in mapping.items():
                P[u, cluster_id] = 1.0
            # assert each row of P sums to 1
            assert onp.all(onp.sum(P, axis=1) == 1.0), "Each row of P must sum to 1."
            all_Ps.append(P)
            (
                coars_window_edge_srcs, coars_window_edge_dsts, coars_window_edge_feats, 
                coars_window_edge_labels, coars_window_node_feats, coars_window_node_labels, 
                supernode_label_mask, coars_train_indices, coars_val_indices
            ) = coarsen_graph(
                NUM_TIMES, Np, Np_next, P, split_indices,
                window_edge_srcs, window_edge_dsts,
                window_edge_feats, window_edge_labels,
                window_node_feats, window_node_labels,
                self.dyn_node_feats, self.dyn_node_labels,
                self.node_label_size, self.meaningless_edge_labels, 
                coarsen_op=coars_op,
                reweight_selfloops=rw_sl, 
                edge_feat_reweight_method=ef_rw_method,
                task_type=self.task_type,
            )
            # update for next iteration
            window_edge_srcs = coars_window_edge_srcs
            window_edge_dsts = coars_window_edge_dsts
            window_edge_feats = coars_window_edge_feats
            window_edge_labels = coars_window_edge_labels
            window_node_feats = coars_window_node_feats 
            window_node_labels = coars_window_node_labels
            Np = Np_next
            # update spectral gap info
            # compute the spectral gap of the current (coarsened) graph
            init_spectral_gaps, init_eigenvecs, supA, supL, cc_sizess, cc_nodess = \
                get_supra_spectral_gap(
                    window_edge_srcs, window_edge_dsts, 
                    onp.eye(Np), NUM_TIMES, Np, Np
                )
            init_spectral_gap = agg_score(init_spectral_gaps, agg_method_cc='min', 
                                    agg_method_temp=None, skip_zero=True)
            print(f"Current spectral gap: {init_spectral_gap:.8f}")
            perc = Np / NUM_NODES * 100.0
            wandb.log({f"init_spectral_gap_at_{perc:.2f}": init_spectral_gap})
            # update the split indices
            new_split_indices = []
            for split_set in split_indices:
                new_split_set = set()
                for node in split_set:
                    new_split_set.add(mapping[node])
                new_split_indices.append(list(new_split_set))
            split_indices = tuple(new_split_indices)
            # end of while loop
            
        assert Np == Nprime_target, "Final number of nodes must be equal to Nprime_target."
        # final check
        supernodes = set(mapping.values())
        if max(supernodes) >= Nprime_target:
            print("Number of clusters must be less than Nprime.")
            import pdb; pdb.set_trace()
        
        num_edges_after = [len(es) for es in coars_window_edge_srcs]
        if any([ne <= NE_THRESH for ne in num_edges_after]):
            print(f"Warning: some snapshots have too few edges "
                  f"after coarsening: {num_edges_after} (threshold: {NE_THRESH}).")
            print("Logging spectral_gap_after_coarsening as -inf and exiting.")
            wandb.log({
                "spectral_gap_after_coarsening": -onp.inf
            })
            exit(0)
            
        spectral_gaps, *_ = get_supra_spectral_gap(
            coars_window_edge_srcs, coars_window_edge_dsts, onp.eye(Np), 
            NUM_TIMES, Np, Np
        )
        spectral_gap = agg_score(spectral_gaps, agg_method_cc='min', 
                                agg_method_temp=None, skip_zero=True)
        
        delta_str = (
            "increased" if spectral_gap - before_coarsen_spectral_gap > 0 
                        else "decreased"
        )
        delta_sg = abs(spectral_gap - before_coarsen_spectral_gap)
        delta_percent = (
            delta_sg / before_coarsen_spectral_gap * 100.0
        )
        delta_str = f"{delta_str} by {delta_sg:.8f} ({delta_percent:.2f}%)"
        print(f"Spectral gap after merging {L} node pairs, "
              f"num nodes from {NUM_NODES} to {Np}: {spectral_gap:.8f}, {delta_str}")
        print(f"Merged pairs: {[[(e, round(v, 8)) for (e, v) in merge_pairs] for merge_pairs in merge_pairs_all]}")
        print(f"Super-node sizes: ", P.sum(axis=0).tolist())
        wandb.log({
            "spectral_gap_after_coarsening": spectral_gap,
            "spectral_gap_delta": spectral_gap - before_coarsen_spectral_gap,
            "spectral_gap_delta_percent": delta_percent,
        })

        window_edge_tuples = (
            [
                onp.stack((edge_snap_srcs, edge_snap_dsts))
                for (edge_snap_srcs, edge_snap_dsts) in (
                    zip(coars_window_edge_srcs, coars_window_edge_dsts)
                )
            ]
        )
        NUM_TIMES = len(window_edge_tuples)
        window_edge_times_abs, window_edge_times_inc, \
        window_node_times_abs, window_node_times_inc = (
            self._timestamping(
                window_edge_tuples, window_timestamps, NUM_TIMES, Np
            )
        )
                
        return (
            window_edge_times_abs, window_edge_times_inc,
            window_node_times_abs, window_node_times_inc,
            window_edge_tuples, 
            coars_window_edge_srcs, coars_window_edge_dsts,
            coars_window_edge_feats, coars_window_edge_labels,
            coars_window_node_feats, coars_window_node_labels, 
            Np, supernode_label_mask, 
            coars_train_indices, coars_val_indices, all_Ps
        )
        
    def coarsen_window_dyngraph_snap_refine(self,
        metaset_ss_num: int,
        window_timestamps,
        window_edge_srcs, window_edge_dsts, 
        window_edge_feats, window_edge_labels,
        window_node_feats, window_node_labels,
        alg_version: str,
        Nprime=0.95,
        split_indices: Tuple[List[int]]=None, 
        
        spec_gap_method: str = 'approx', 
        coars_op: str = 'PtAP', 
        rw_sl: bool = True, 
        ef_rw_method: str = 'avg',
        refine_every: int = 0.1, # refine every 10% of the reduction ratio relative to original graph
    ) -> Tuple[
        List[onpt.NDArray[onp.generic]],
        List[onpt.NDArray[onp.generic]],
        List[onpt.NDArray[onp.generic]],
        List[onpt.NDArray[onp.generic]],
        onpt.NDArray[onp.generic],
        onpt.NDArray[onp.generic],
    ]:
        R"""
        Coarsen the graph.
        window_timestamps: onpt.NDArray[onp.generic] of shape (num_times, ), the timestamps of each snapshot.
        
        window_edge_srcs: List[onpt.NDArray[onp.generic]] of shape (num_times, num_edges), 
            where each element is the source nodes of edges at time t.
        window_edge_dsts: List[onpt.NDArray[onp.generic]] of shape (num_times, num_edges), 
            where each element is the destination nodes of edges at time t.
        window_edge_feats: List[onpt.NDArray[onp.generic]] of shape (num_times, num_edges, num_features), 
            where each element is the features of edges at time t.
        window_edge_labels: List[onpt.NDArray[onp.generic]] of shape (num_times, num_edges, edge_label_size), 
            where each element is the labels of edges at time t.
        window_node_feats: onpt.NDArray[onp.generic] of shape (num_nodes, num_features, num_times) if self.dyn_node_feats is true, 
            or onpt.NDArray[onp.generic] of shape (num_nodes, num_features) if self.dyn_node_feats is false,
            where each element is the features of nodes at time t.
        window_node_labels: onpt.NDArray[onp.generic] of shape (num_nodes, node_label_size, num_times) if self.dyn_node_labels is true, 
            or onpt.NDArray[onp.generic] of shape (num_nodes, node_label_size) if self.dyn_node_labels is false, 
            where each element is the labels of nodes at time t.
        Nprime: float or int, the number of nodes left in the coarsened graph.
        """
        # construct the aggregated graph
        if self.dyn_node_feats:
            NUM_NODES, _, NUM_TIMES = window_node_feats.shape
        else:
            NUM_NODES, _ = window_node_feats.shape
        
        if Nprime < 1.0: 
            Nprime_target = round(NUM_NODES * Nprime)
        else: 
            Nprime_target = int(Nprime)
        
        print(f"Using coarsen_window_dyngraph_snap_refine with Nprime={Nprime_target} ({Nprime_target/NUM_NODES*100:.2f}%)")
        
        L = NUM_NODES - Nprime
        assert L > 0, "Nprime must be less than the number of nodes in the original graph."
        
        # compute the spectral gap of the current aggregated graph
        # construct the aggregated adjacency matrix A_agg
        # snaps_spectral_gaps, snaps_eigenvecs, As, Acs, Ls, cc_sizes, cc_to_nodes
        init_spectral_gaps, init_eigenvecs, As, Ls, cc_sizess, cc_nodess = get_snapshots_spectral_gap(
            window_edge_srcs, window_edge_dsts, onp.eye(NUM_NODES),
            NUM_TIMES, NUM_NODES, NUM_NODES
        )
        
        init_spectral_gap = agg_score(init_spectral_gaps, agg_method_cc='min', 
                                      agg_method_temp='min', skip_zero=True)
        before_coarsen_spectral_gap = init_spectral_gap
        
        print(f"Initial spectral gap: {init_spectral_gap:.8f}")
        wandb.log({"init_spectral_gap": init_spectral_gap})
        sgs_logs = {f"init_spectral_gap_sn_{i}": min(sg[sg>0]) for i, sg in enumerate(init_spectral_gaps)}
        wandb.log(sgs_logs)
        # loop K times to merge nodes
        all_Ps = []
        merge_pairs_all = []

        Np = NUM_NODES
        reduction_step = round(NUM_NODES * refine_every)
        slack = reduction_step // 5  # allow some slack to avoid too small final step
        while Np > Nprime_target:
            Np_next = Np - reduction_step
            if Np_next <= Nprime_target + slack:
                Np_next = Nprime_target
                
            print(f"Coarsening from {Np} to {Np_next} nodes ...")
            #
            
            check_pairs = set()
            
            for t in range(NUM_TIMES):
                srcs = window_edge_srcs[t]
                dsts = window_edge_dsts[t]
                # find all unique pairs of nodes that are connected by an edge
                edges = onp.unique(onp.stack((srcs, dsts), axis=1), axis=0)
                for (u, v) in edges:
                    if u != v and (v, u) not in check_pairs:
                        check_pairs.add((u, v))
            
            check_pairs = list(check_pairs)
            if len(check_pairs) == 0:
                print("No valid edges found for merging. Stopping early.")
                raise RuntimeError("No valid edges found for merging.")
            #
            init_info = (init_spectral_gaps, init_eigenvecs, As, Ls, cc_sizess, cc_nodess)
            links_score = get_links_scores_over_coarsened_dyngraph(
                NUM_TIMES, Np, Np-1, window_edge_srcs, window_edge_dsts,
                init_info, check_pairs, SPEC_GAP_METHOD=spec_gap_method
            )
            sorted_edges = sorted(links_score.items(), 
                                key=lambda item: item[1], reverse=True)
            # find clusters of merged nodes
            mapping, merged_pairs = map_to_supernodes(sorted_edges, Np, Np_next)
            merge_pairs_all.append(merged_pairs)
            
            supernodes = set(mapping.values())
            if max(supernodes) > Np_next:
                print(f"Warning: Number of clusters {max(supernodes)} must be <= Nprime_next {Np_next}.")
            
            Np_next = len(supernodes) # update Np_next based on actual clusters formed
            # construct the partition matrix P of shape (N, K)
            P = onp.zeros((Np, Np_next))
            for u, cluster_id in mapping.items():
                P[u, cluster_id] = 1.0
            # assert each row of P sums to 1
            assert onp.all(onp.sum(P, axis=1) == 1.0), "Each row of P must sum to 1."
            all_Ps.append(P)
            (
                coars_window_edge_srcs, coars_window_edge_dsts, coars_window_edge_feats, 
                coars_window_edge_labels, coars_window_node_feats, coars_window_node_labels, 
                supernode_label_mask, coars_train_indices, coars_val_indices
            ) = coarsen_graph(
                NUM_TIMES, Np, Np_next, P, split_indices,
                window_edge_srcs, window_edge_dsts,
                window_edge_feats, window_edge_labels,
                window_node_feats, window_node_labels,
                self.dyn_node_feats, self.dyn_node_labels,
                self.node_label_size, self.meaningless_edge_labels,
                coarsen_op=coars_op,
                reweight_selfloops=rw_sl, 
                edge_feat_reweight_method=ef_rw_method,
                task_type=self.task_type,
            )
            # update for next iteration
            window_edge_srcs = coars_window_edge_srcs
            window_edge_dsts = coars_window_edge_dsts
            window_edge_feats = coars_window_edge_feats
            window_edge_labels = coars_window_edge_labels
            window_node_feats = coars_window_node_feats 
            window_node_labels = coars_window_node_labels
            Np = Np_next
            # update spectral gap info
            # compute the spectral gap of the current (coarsened) graph
            (
                init_spectral_gaps, init_eigenvecs, As, 
                Ls, cc_sizess, cc_nodess
            ) = get_snapshots_spectral_gap(
                window_edge_srcs, window_edge_dsts, 
                onp.eye(Np), NUM_TIMES, Np, Np
            )
            init_spectral_gap = agg_score(
                init_spectral_gaps, agg_method_cc='min',
                agg_method_temp='min', skip_zero=True
            )
            print(f"Current spectral gap: {init_spectral_gap:.8f}") 
            perc = Np / NUM_NODES * 100.0
            wandb.log({f"init_spectral_gap_at_{perc:.2f}": init_spectral_gap})
            sgs_logs = {f"init_spectral_gap_sn_{i}": min(sg[sg>0]) for i, sg in enumerate(init_spectral_gaps)}
            wandb.log(sgs_logs)
            # update the split indices
            new_split_indices = []
            for split_set in split_indices:
                new_split_set = set()
                for node in split_set:
                    new_split_set.add(mapping[node])
                new_split_indices.append(list(new_split_set))
            split_indices = tuple(new_split_indices)
            # end of while loop
            
        assert Np == Nprime_target, "Final number of nodes must be equal to Nprime_target."
        # final check
        supernodes = set(mapping.values())
        if max(supernodes) >= Nprime_target:
            print("Number of clusters must be less than Nprime.")
            import pdb; pdb.set_trace()
        
        num_edges_after = [len(es) for es in window_edge_srcs]
        if any([ne <= NE_THRESH for ne in num_edges_after]):
            print(f"Warning: some snapshots have too few edges after coarsening: {num_edges_after} "
                  f"(threshold: {NE_THRESH}).")
            print("Logging spectral_gap_after_coarsening as -inf and exiting.")
            wandb.log({
                "spectral_gap_after_coarsening": -onp.inf
            })
            exit(0)
            
        # init_spectral_gaps, init_eigenvecs, As, Ls, cc_sizess, cc_nodess
        spectral_gaps, *_= get_snapshots_spectral_gap(
            window_edge_srcs, window_edge_dsts, onp.eye(Np), 
            NUM_TIMES, Np, Np
        )

        spectral_gap = agg_score(spectral_gaps, agg_method_cc='min', 
                                 agg_method_temp='min', skip_zero=True)
        
        delta_str = "increased" if spectral_gap - before_coarsen_spectral_gap > 0 else "decreased"
        delta_sg = abs(spectral_gap - before_coarsen_spectral_gap)
        delta_percent = (
            delta_sg / before_coarsen_spectral_gap * 100.0
        )
        delta_str = f"{delta_str} by {delta_sg:.8f} ({delta_percent:.2f}%)"
        print(f"Spectral gap after merging {L} node pairs, "
              f"num nodes from {NUM_NODES} to {Np}: {spectral_gap:.8f}, {delta_str}")
        print(f"Merged pairs: {[(e, round(v, 4)) for (e, v) in merged_pairs]}")
        print(f"Super-node sizes: ", P.sum(axis=0).tolist())
        wandb.log({
            "spectral_gap_after_coarsening": spectral_gap,
            "spectral_gap_delta": spectral_gap - before_coarsen_spectral_gap,
            "spectral_gap_delta_percent": delta_percent,
        })
        sgs_logs = {f"spectral_gap_after_coarsening_sn_{i}": min(sg[sg>0]) for i, sg in enumerate(spectral_gaps)}
        wandb.log(sgs_logs)
        
        window_edge_tuples = (
            [
                onp.stack((edge_snap_srcs, edge_snap_dsts))
                for (edge_snap_srcs, edge_snap_dsts) in (
                    zip(window_edge_srcs, window_edge_dsts)
                )
            ]
        )
        NUM_TIMES = len(window_edge_tuples)
        window_edge_times_abs, window_edge_times_inc, \
        window_node_times_abs, window_node_times_inc = (
            self._timestamping(
                window_edge_tuples, window_timestamps, NUM_TIMES, Np
            )
        )

        return (
            window_edge_times_abs, window_edge_times_inc,
            window_node_times_abs, window_node_times_inc,
            window_edge_tuples, window_edge_srcs, window_edge_dsts,
            window_edge_feats, window_edge_labels,
            window_node_feats, window_node_labels, 
            Np, supernode_label_mask, 
            coars_train_indices, coars_val_indices, all_Ps
        )
        
    def coarsen_window_dyngraph_snap(self,
        metaset_ss_num: int,
        window_timestamps,
        window_edge_srcs, window_edge_dsts, 
        window_edge_feats, window_edge_labels,
        window_node_feats, window_node_labels,
        alg_version: str,
        Nprime=0.95,
        split_indices: Tuple[List[int]]=None, 
        
        spec_gap_method: str = 'approx', 
        coars_op: str = 'PtAP', 
        rw_sl: bool = True, 
        ef_rw_method: str = 'avg'
    ) -> Tuple[
        List[onpt.NDArray[onp.generic]],
        List[onpt.NDArray[onp.generic]],
        List[onpt.NDArray[onp.generic]],
        List[onpt.NDArray[onp.generic]],
        onpt.NDArray[onp.generic],
        onpt.NDArray[onp.generic],
    ]:
        R"""
        Coarsen the graph.
        window_timestamps: onpt.NDArray[onp.generic] of shape (num_times, ), the timestamps of each snapshot.
        
        window_edge_srcs: List[onpt.NDArray[onp.generic]] of shape (num_times, num_edges), 
            where each element is the source nodes of edges at time t.
        window_edge_dsts: List[onpt.NDArray[onp.generic]] of shape (num_times, num_edges), 
            where each element is the destination nodes of edges at time t.
        window_edge_feats: List[onpt.NDArray[onp.generic]] of shape (num_times, num_edges, num_features), 
            where each element is the features of edges at time t.
        window_edge_labels: List[onpt.NDArray[onp.generic]] of shape (num_times, num_edges, edge_label_size), 
            where each element is the labels of edges at time t.
        window_node_feats: onpt.NDArray[onp.generic] of shape (num_nodes, num_features, num_times) if self.dyn_node_feats is true, 
            or onpt.NDArray[onp.generic] of shape (num_nodes, num_features) if self.dyn_node_feats is false,
            where each element is the features of nodes at time t.
        window_node_labels: onpt.NDArray[onp.generic] of shape (num_nodes, node_label_size, num_times) if self.dyn_node_labels is true, 
            or onpt.NDArray[onp.generic] of shape (num_nodes, node_label_size) if self.dyn_node_labels is false, 
            where each element is the labels of nodes at time t.
        Nprime: float or int, the number of nodes left in the coarsened graph.
        """
        # construct the aggregated graph
        if self.dyn_node_feats:
            num_nodes, _, num_times = window_node_feats.shape
        else:
            num_nodes, _ = window_node_feats.shape
        
        if Nprime < 1.0: 
            Nprime = round(num_nodes * Nprime)
        else: 
            Nprime = int(Nprime)
        
        print(f"Using coarsen_window_dyngraph_snap with Nprime={Nprime} ({Nprime/num_nodes*100:.2f}%)")
        
        L = num_nodes - Nprime
        assert L > 0, "Nprime must be less than the number of nodes in the original graph."
        
        # compute the spectral gap of the current aggregated graph
        # construct the aggregated adjacency matrix A_agg
        # snaps_spectral_gaps, snaps_eigenvecs, As, Acs, Ls, cc_sizes, cc_to_nodes
        I = onp.eye(num_nodes)
        init_spectral_gaps, init_eigenvecs, As, Ls, cc_sizess, cc_nodess = get_snapshots_spectral_gap(
            window_edge_srcs, window_edge_dsts, I, 
            num_times, num_nodes, num_nodes
        )
        
        init_spectral_gap = agg_score(init_spectral_gaps, agg_method_cc='min', 
                                      agg_method_temp='min', skip_zero=True)
        print(f"Initial spectral gap: {init_spectral_gap:.8f}")
        wandb.log({"init_spectral_gap": init_spectral_gap})
        sgs_logs = {f"init_spectral_gap_sn_{i}": min(sg[sg>0]) for i, sg in enumerate(init_spectral_gaps)}
        wandb.log(sgs_logs)
        # loop K times to merge nodes
        all_Ps = []

        check_pairs = set()
        
        for t in range(num_times):
            srcs = window_edge_srcs[t]
            dsts = window_edge_dsts[t]
            # find all unique pairs of nodes that are connected by an edge
            edges = onp.unique(onp.stack((srcs, dsts), axis=1), axis=0)
            for (u, v) in edges:
                # if u != v and (v, u) not in check_pairs \
                #     and (u not in orig_test_indices and v not in orig_test_indices): # 
                #     check_pairs.add((u, v))
                if u != v and (v, u) not in check_pairs:
                    check_pairs.add((u, v))
        
        check_pairs = list(check_pairs)
        if len(check_pairs) == 0:
            print("No valid edges found for merging. Stopping early.")
            raise RuntimeError("No valid edges found for merging.")
        
        SS_NUM = metaset_ss_num
        if SS_NUM == -1: # use all pairs
            ss_idx = onp.arange(len(check_pairs))
        else:
            SS_NUM = max(L, metaset_ss_num)
            ss_idx = onp.random.choice(len(check_pairs), size=SS_NUM, replace=False)
            
        print(f"Sampling {len(ss_idx)} pairs from {len(check_pairs)} pairs for checking.")
        ss_check_pairs = [check_pairs[i] for i in ss_idx]
        # spectral_gaps = {}
        
        N = num_nodes
        K = num_nodes - 1
        init_info = (init_spectral_gaps, init_eigenvecs, As, Ls, cc_sizess, cc_nodess)
        
        links_score = get_links_scores_over_coarsened_dyngraph(
            num_times, N, K,
            window_edge_srcs, window_edge_dsts,
            init_info,
            ss_check_pairs, 
            SPEC_GAP_METHOD=spec_gap_method
        )
        
        # after looping through all edges, pick top L edges with highest spectral gap
        sorted_edges = sorted(links_score.items(), 
                              key=lambda item: item[1], reverse=True)

        # find clusters of merged nodes
        mapping, merged_pairs = map_to_supernodes(sorted_edges, num_nodes, Nprime)
        
        supernodes = set(mapping.values())
        if max(supernodes) >= Nprime:
            print("Number of clusters must be less than Nprime.")
            import pdb; pdb.set_trace()
            
        # construct the partition matrix P of shape (N, K)
        K = Nprime
        P = onp.zeros((num_nodes, K))
        for u, cluster_id in mapping.items():
            P[u, cluster_id] = 1.0
        # assert each row of P sums to 1
        assert onp.all(onp.sum(P, axis=1) == 1.0), "Each row of P must sum to 1."
        all_Ps.append(P)
        
        (
            window_edge_srcs, window_edge_dsts, window_edge_feats, 
            window_edge_labels, window_node_feats, window_node_labels, 
            supernode_label_mask, coars_train_indices, coars_val_indices
        ) = coarsen_graph(
            num_times, num_nodes, K, P, split_indices,
            window_edge_srcs, window_edge_dsts,
            window_edge_feats, window_edge_labels,
            window_node_feats, window_node_labels,
            self.dyn_node_feats, self.dyn_node_labels,
            self.node_label_size, self.meaningless_edge_labels,
            coarsen_op=coars_op,
            reweight_selfloops=rw_sl, 
            edge_feat_reweight_method=ef_rw_method,
            task_type=self.task_type,
        )

        num_edges_after = [len(es) for es in window_edge_srcs]
        if any([ne <= NE_THRESH for ne in num_edges_after]):
            print(f"Warning: some snapshots have too few edges "
                  f"after coarsening: {num_edges_after} (threshold: {NE_THRESH}).")
            print("Logging spectral_gap_after_coarsening as -inf and exiting.")
            wandb.log({
                "spectral_gap_after_coarsening": -onp.inf
            })
            exit(0)
            
        # init_spectral_gaps, init_eigenvecs, As, Ls, cc_sizess, cc_nodess
        spectral_gaps, *_= get_snapshots_spectral_gap(
            window_edge_srcs, window_edge_dsts, onp.eye(K), 
            num_times, K, K
        )

        spectral_gap = agg_score(spectral_gaps, agg_method_cc='min', 
                                 agg_method_temp='min', skip_zero=True)
        
        delta_str = "increased" if spectral_gap - init_spectral_gap > 0 else "decreased"
        delta_sg = abs(spectral_gap - init_spectral_gap)
        delta_percent = (
            delta_sg / init_spectral_gap * 100.0
        )
        delta_str = f"{delta_str} by {delta_sg:.8f} ({delta_percent:.2f}%)"
        print(f"Spectral gap after merging {L} node pairs, "
              f"num nodes from {num_nodes} to {K}: {spectral_gap:.8f}, {delta_str}")
        print(f"Merged pairs: {[(e, round(v, 4)) for (e, v) in merged_pairs]}")
        print(f"Super-node sizes: ", P.sum(axis=0).tolist())
        wandb.log({
            "spectral_gap_after_coarsening": spectral_gap,
            "spectral_gap_delta": spectral_gap - init_spectral_gap,
            "spectral_gap_delta_percent": delta_percent,
        })
        sgs_logs = {f"spectral_gap_after_coarsening_sn_{i}": 
                    min(sg[sg>0]) for i, sg in enumerate(spectral_gaps)}
        wandb.log(sgs_logs)
        
        window_edge_tuples = (
            [
                onp.stack((edge_snap_srcs, edge_snap_dsts))
                for (edge_snap_srcs, edge_snap_dsts) in (
                    zip(window_edge_srcs, window_edge_dsts)
                )
            ]
        )
        num_times = len(window_edge_tuples)
        window_edge_times_abs, window_edge_times_inc, \
        window_node_times_abs, window_node_times_inc = (
            self._timestamping(
                window_edge_tuples, window_timestamps, num_times, K
            )
        )

        # assert train_indices at supernode_label_mask are all True
        if coars_train_indices is not None and len(coars_train_indices) > 0:
            assert onp.all(supernode_label_mask[coars_train_indices]), \
                "All train indices must be valid for labeling."
            assert onp.all(supernode_label_mask[coars_val_indices]), \
                "All val indices must be valid for labeling."
                
        return (
            window_edge_times_abs, window_edge_times_inc,
            window_node_times_abs, window_node_times_inc,
            window_edge_tuples, window_edge_srcs, window_edge_dsts,
            window_edge_feats, window_edge_labels,
            window_node_feats, window_node_labels, 
            K, supernode_label_mask, 
            coars_train_indices, coars_val_indices, all_Ps
        )

    @lru_cache(maxsize=MAX_CACHE_SIZE)
    def __getitem__(self, idx: int, /) -> MEMBAT:
        R"""
        Get an indexable item of the class.
        """
        # Get input and target window ties.
        # DynCSL is special.
        input_tie1 = idx if self.WIN_SHIFT < 0 else idx * self.WIN_SHIFT
        input_tie2 = input_tie1 + self.window_history_size
        target_tie1 = input_tie2
        target_tie2 = (
            target_tie1 + self.window_future_size
            if self.WIN_SHIFT < 0 else
            target_tie1 - 1
        )

        input_begin = min(input_tie1, input_tie2)
        input_end = max(input_tie1, input_tie2)
        target_begin = min(target_tie1, target_tie2)
        target_end = max(target_tie1, target_tie2)

        # Get edge timestamp data of current batch.
        window_edge_times_abs = self.edge_times_abs[input_begin:target_end]
        window_edge_times_inc = self.edge_times_inc[input_begin:target_end]
        window_node_times_abs = self.node_times_abs[:, input_begin:target_end]
        window_node_times_inc = self.node_times_inc[:, input_begin:target_end]

        window_edge_tuples = self.edge_tuples[input_begin:target_end] 
        window_edge_srcs = self.edge_srcs[input_begin:target_end]
        window_edge_dsts = self.edge_dsts[input_begin:target_end]
        window_edge_feats = self.edge_feats[input_begin:target_end]
        window_edge_labels = self.edge_labels[input_begin:target_end]
        #
        if self.dyn_node_feats:
            window_node_feats = self.node_feats[:, :, input_begin:target_end]
        else:
            window_node_feats = self.node_feats
            
        if self.dyn_node_labels:
            window_node_labels = self.node_labels[:, :, input_begin:target_end]
        else:
            window_node_labels = self.node_labels
        
        window_num_nodes = window_node_feats.shape[0]
        
        # get max node id 
        node_max = max(
            onp.max(edge_tuples_it).item()
            for edge_tuples_it in window_edge_tuples
        )
        # check if srcs and dsts are within the range of node_max
        for edge_srcs_it, edge_dsts_it in zip(window_edge_srcs, window_edge_dsts):
            if not onp.all(onp.logical_and(
                onp.logical_and(
                    edge_srcs_it >= 0, edge_srcs_it <= node_max
                ),
                onp.logical_and(
                    edge_dsts_it >= 0, edge_dsts_it <= node_max
                )
            )):
                print("Edge sources or destinations are out of range of node ids.")
                import pdb; pdb.set_trace()
                
        # as we takenout the window information, we need to adjust the
        # input_begin and input_end to match within the slice
        input_end -= input_begin
        target_begin -= input_begin
        target_end -= input_begin
        input_begin -= input_begin 
        
        # Get edge timestamp data of current batch.
        edge_times_abs_input = window_edge_times_abs[input_begin:input_end]
        edge_times_rel_input = (
            [
                onp.full_like(
                    edge_times_abs_it,
                    (
                        onp.max(edge_times_abs_input[-1])
                        - onp.min(edge_times_abs_it)
                    ),
                )
                for edge_times_abs_it in edge_times_abs_input
            ]
        )
        edge_times_inc_input = window_edge_times_inc[input_begin:input_end]

        # Get node timestamp data of current batch.
        node_times_abs_input = window_node_times_abs[:, input_begin:input_end]
        node_times_rel_input = (
            onp.reshape(node_times_abs_input[:, -1], (window_num_nodes, 1))
            - node_times_abs_input
        )
        node_times_inc_input = window_node_times_inc[:, input_begin:input_end]

        #
        if self.winaggr == self.WINAGG_DENSE and len(self) > 1:
            # Collect unique edge identifiers as integers.
            node_max = (
                max(
                    onp.max(edge_tuples_it).item()
                    for edge_tuples_it in (
                        window_edge_tuples[input_begin:input_end]
                    )
                )
            )
            edge_ints = (
                [
                    edge_srcs_it * (node_max + 1) + edge_dsts_it
                    for (edge_srcs_it, edge_dsts_it) in (
                        zip(
                            window_edge_srcs[input_begin:input_end],
                            window_edge_dsts[input_begin:input_end],
                        )
                    )
                ]
            )
            edge_uniqints = onp.sort(onp.unique(onp.concatenate(edge_ints)))

            # Safety check.
            if self.edge_feat_size > 1:
                # UNEXPECT:
                # Only scalar temporal edge feature is aggregatable.
                raise NotImplementedError(
                    "Only scalar temporal edge feature is aggregatable.",
                )
            if (
                any(
                    onp.any(onp.less(edge_feats_it, 0.0))
                    for edge_feats_it in window_edge_feats
                )
            ):
                # UNEXPECT:
                # Padding value can not exist in raw edge feature.
                raise NotImplementedError(
                    "negative values are used as padding values in dense "
                    "feature aggregation, thus can not be part of edge "
                    "feature.",
                )

            # Generate dense edge feature and appearance aggregation as sparse
            # matrices.
            aggr_eids = onp.concatenate(edge_ints)
            aggr_tids = (
                onp.concatenate(
                    [
                        onp.full_like(edge_ints_it, t)
                        for (t, edge_ints_it) in enumerate(edge_ints)
                    ],
                )
            )
            aggr_vals_feat = (
                onp.reshape(
                    onp.concatenate(window_edge_feats[input_begin:input_end]),
                    (len(aggr_eids),),
                )
            )
            aggr_vals_appr = onp.ones((len(aggr_eids)), dtype=bool)
            try:
                aggr_spmat_feat = (
                    osparse.coo_matrix(
                        (aggr_vals_feat, (aggr_eids, aggr_tids)),
                        shape=(
                            onp.max(aggr_eids).item() + 1,
                            self.window_history_size,
                        ),
                    ).tocsc()
                )
            except Exception as e:
                print(f"Failed to create sparse matrix: {e}")
                import pdb; pdb.set_trace()
                
            aggr_spmat_appr = (
                osparse.coo_matrix(
                    (aggr_vals_appr, (aggr_eids, aggr_tids)),
                    shape=(
                        onp.max(aggr_eids).item() + 1,
                        self.window_history_size,
                    ),
                ).tocsc()
            )

            #
            aggr_srcs = edge_uniqints // (node_max + 1)
            aggr_dsts = edge_uniqints % (node_max + 1)
            aggr_tuples = onp.stack((aggr_srcs, aggr_dsts))

            # Convert to dense feature matrix of non-trivial feature dynamics.
            # Use -1.0 as default padding value.
            aggr_feats_buf = []
            aggr_apprs_buf = []
            for eid in edge_uniqints.tolist():
                #
                aggr_feats_buf.append(aggr_spmat_feat[eid].toarray())
                aggr_apprs_buf.append(aggr_spmat_appr[eid].toarray())
            aggr_feats = (
                onp.reshape(
                    onp.stack(aggr_feats_buf),
                    (len(edge_uniqints), 1, self.window_history_size),
                )
            )
            aggr_apprs = (
                onp.reshape(
                    onp.stack(aggr_apprs_buf),
                    (len(edge_uniqints), 1, self.window_history_size),
                )
            )
            aggr_feats[onp.logical_not(aggr_apprs)] = -1.0

            # We should not have any edge labels.
            if not self.meaningless_edge_labels:
                # UNEXPECT
                # We can not aggregate temporal edge labels for now.
                raise NotImplementedError(
                    "We can not aggregate temporal edge labels for now.",
                )
            aggr_labels = onp.array(0)

            # Aggregated timestamps will not care the raw appearance since it
            # will regard all edges appearing once as existing all the time.
            aggr_times_abs = (
                onp.concatenate(
                    [
                        onp.unique(time_abs)
                        for time_abs in edge_times_abs_input
                    ],
                )
            )
            aggr_times_rel = onp.max(aggr_times_abs) - aggr_times_abs
            aggr_times_inc = (
                onp.concatenate(
                    (
                        onp.zeros((1,), dtype=aggr_times_abs.dtype),
                        aggr_times_abs[1:] - aggr_times_abs[:-1],
                    ),
                )
            )
            aggr_times_abs = onp.tile(aggr_times_abs, (len(edge_uniqints), 1))
            aggr_times_rel = onp.tile(aggr_times_rel, (len(edge_uniqints), 1))
            aggr_times_inc = onp.tile(aggr_times_inc, (len(edge_uniqints), 1))

            # Rename as new edge data.
            edge_tuples_numpy = aggr_tuples
            edge_feats_numpy = aggr_feats
            edge_labels_numpy = aggr_labels
            edge_ranges_numpy = onp.array([0, len(edge_feats_numpy)])

            # Collect edge timestamp data.
            edge_times_numpy_buf = []
            if self.ABS in self.timestamped_edge_times:
                #
                edge_times_numpy_buf.append(aggr_times_abs)
            if self.REL in self.timestamped_edge_times:
                #
                edge_times_numpy_buf.append(aggr_times_rel)
            if self.INC in self.timestamped_edge_times:
                #
                edge_times_numpy_buf.append(aggr_times_inc)
            if len(edge_times_numpy_buf) == 0:
                # Use number of timestamps as place holder.
                edge_times_numpy = onp.array(self.num_times)
            else:
                # Merge all timestamp data into a single array.
                # Put node axis at first for batching.
                edge_times_numpy = onp.stack(edge_times_numpy_buf)
                edge_times_numpy = onp.transpose(edge_times_numpy, (1, 0, 2))
        elif self.winaggr == self.WINAGG_NONE:
            # Dynamic graph with dynamic edge has different snapshots with
            # consecutive ranges.
            edge_ranges_numpy = (
                onp.array(
                    [0] + [len(time_abs) for time_abs in edge_times_abs_input],
                )
            )
            edge_ranges_numpy = onp.cumsum(edge_ranges_numpy)
            edge_ranges_numpy = (
                onp.stack((edge_ranges_numpy[:-1], edge_ranges_numpy[1:]))
            )

            # Collect edge timestamp data.
            edge_times_numpy_buf = []
            if self.ABS in self.timestamped_edge_times:
                #
                edge_times_numpy_buf.append(
                    onp.concatenate(edge_times_abs_input),
                )
            if self.REL in self.timestamped_edge_times:
                #
                edge_times_numpy_buf.append(
                    onp.concatenate(edge_times_rel_input),
                )
            if self.INC in self.timestamped_edge_times:
                #
                edge_times_numpy_buf.append(
                    onp.concatenate(edge_times_inc_input),
                )
            if len(edge_times_numpy_buf) == 0:
                # Use number of timestamps as place holder.
                edge_times_numpy = onp.array(self.num_times)
            else:
                # Merge all timestamp data into a single array.
                edge_times_numpy = onp.stack(edge_times_numpy_buf)

            # Collect edge feature and label.
            edge_tuples_numpy = (
                onp.concatenate(
                    window_edge_tuples[input_begin:input_end],
                    axis=1,
                )
            )
            if not self.meaningless_edge_feats:
                #
                edge_feats_numpy = (
                    onp.concatenate(window_edge_feats[input_begin:input_end])
                )
            else:
                #
                edge_feats_numpy = onp.array(0.0)
            if not self.meaningless_edge_labels:
                #
                edge_labels_numpy = (
                    onp.concatenate(window_edge_labels[input_begin:input_end])
                )
            else:
                #
                edge_labels_numpy = onp.array(0)
        elif len(self) > 1:
            # UNEXPECT:
            # Unknown aggregation for given length.
            raise NotImplementedError("Unknown aggregation for given length.")
        # \\ print(edge_tuples_numpy.shape)
        # \\ print(edge_feats_numpy.shape)
        # \\ print(edge_labels_numpy.shape)
        # \\ print(edge_ranges_numpy)
        # \\ print(edge_times_numpy.shape)
        # \\ raise RuntimeError

        def fetch(
            array: onpt.NDArray[onp.generic], dataon: List[int], dynamic: bool,
            tmin: int, tmax: int, null: Union[int, float],
        ) -> onpt.NDArray[onp.generic]:
            R"""
            Fetch online memory from data.
            """
            if dynamic:
                # if dynamic: array.shape = (num_nodes, feat_dim, num_times)
                return (
                    onp.array(null)
                    if len(dataon) == 0 else
                    cast(
                        onpt.NDArray[onp.generic], array[:, dataon, tmin:tmax],
                    )
                )
            else:
                # else: array.shape = (num_nodes, feat_dim)
                return (
                    onp.array(null)
                    if len(dataon) == 0 else
                    cast(onpt.NDArray[onp.generic], array[:, dataon])
                )

        # Fetch inputs.
        # Node inputs will always be a 3D tensor with time axis, thus there is
        # no need to store exact snapshot range.
        node_feats_input_numpy = (
            fetch(
                window_node_feats, self.node_feat_input, self.dyn_node_feats,
                input_begin, input_end, 0.0,
            )
        )
        node_labels_input_numpy = (
            fetch(
                window_node_labels, self.node_label_input, self.dyn_node_labels,
                input_begin, input_end, 0,
            )
        )

        node_feats_target_numpy  = (
            fetch(
                window_node_feats, self.node_feat_target, self.dyn_node_feats,
                target_begin, target_end, 0.0,
            )
        )

        node_labels_target_numpy = (
            fetch(
                window_node_labels, self.node_label_target, self.dyn_node_labels,
                target_begin, target_end, 0,
            )
        )
        # Collect node timestamp data to be pinned.
        node_times_input_numpy_buf = []
        if self.ABS in self.timestamped_node_times:
            #
            node_times_input_numpy_buf.append(node_times_abs_input)
        if self.REL in self.timestamped_node_times:
            #
            node_times_input_numpy_buf.append(node_times_rel_input)
        if self.INC in self.timestamped_node_times:
            #
            node_times_input_numpy_buf.append(node_times_inc_input)
        if len(node_times_input_numpy_buf) == 0:
            # Use number of timestamps as place holder.
            node_times_input_numpy = onp.array(self.num_times)
        else:
            # Merge all timestamp data into a single array.
            # Put node axis at first for batching.
            node_times_input_numpy = onp.stack(node_times_input_numpy_buf)
            node_times_input_numpy = (
                onp.transpose(node_times_input_numpy, (1, 0, 2))
            )
        
        if len(self) == 1 and self.winaggr != self.WINAGG_NONE:
            #
            return_obj = (
                [
                    *self.cached, node_feats_input_numpy,
                    node_labels_input_numpy, node_times_input_numpy,
                ],
                [node_feats_target_numpy, node_labels_target_numpy, window_num_nodes],
            )
        else:
            #
            return_obj = (
                [
                    edge_tuples_numpy, edge_feats_numpy, edge_labels_numpy,
                    edge_ranges_numpy, edge_times_numpy, node_feats_input_numpy,
                    node_labels_input_numpy, node_times_input_numpy,
                ],
                [node_feats_target_numpy, node_labels_target_numpy, window_num_nodes],
            )
        return return_obj

    def to_node_vecseq(self, /) -> VectorSequence:
        R"""
        Transform into node vector sequence metaset.
        """
        #
        raise RuntimeError

    def to_edge_vecseq(self, /) -> VectorSequence:
        R"""
        Transform into edge vector sequence metaset.
        """
        # UNEXPECT:
        # Temporal graph with static edge does not have edge vector sequence
        # data.
        raise NotImplementedError(
            "Temporal graph with static edge does not have edge vector "
            "sequence data.",
        )