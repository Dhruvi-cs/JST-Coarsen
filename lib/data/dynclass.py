R"""
"""
#
import numpy as onp
import numpy.typing as onpt
import torch
import os
from typing import Optional, List, Dict, Tuple, cast
from ..meta.dyngraph.sparse.dynedge import DynamicAdjacencyListDynamicEdge
import pickle
import networkx as nx
from ..exec.consts import CLF_TASK, REG_TASK

class DynamicClassification(object):
    R"""
    Dynamic node classification over the whole window dataset.
    """
    #
    SOURCE: str

    def __init__(
        self,
        dirname: str,
        /,
        filename:str=None,
    ) -> None:
        R"""
        Initialize the class.
        """
        #
        self.from_raw(dirname, filename=filename)
        self.sanitize_edge()

    def from_raw(self, dirname: str, /, filename:str=None) -> None:
        R"""
        Load from raw data.
        """
        if filename is not None:
            print(f'Loading from specified file: {filename}')
            matrices = (
                onp.load(os.path.join(dirname, filename))
            )
        else:
            print(f'Loading from default file: {self.SOURCE}_no_isolated.npz')
            matrices = (
                onp.load(os.path.join(dirname, "{:s}_no_isolated.npz".format(self.SOURCE)))
            )
        feats = matrices["attmats"]
        adjmats = matrices["adjs"]
        onehots = matrices["labels"]
        #
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
        self.raw_node_feats = feats
        (_, self.num_labels) = onehots.shape
        self.label_counts = onp.sum(onehots, axis=0).tolist()
        # lab_avail_nodes are the nodes with one-hot labels are non-zero
        lab_avail_nodes = onp.where(onp.sum(onehots, axis=1) != 0)[0].tolist()
        
        print(f'Number of labeled available nodes: {len(lab_avail_nodes)} '
                f'({len(lab_avail_nodes) / onehots.shape[0] * 100:.2f}%)')
        if len(lab_avail_nodes) < onehots.shape[0]:
            self.lab_avail_nodes = lab_avail_nodes
        
        # (_, self.raw_node_labels) = onp.nonzero(onehots)
        self.raw_node_labels = onp.zeros((feats.shape[0], ), dtype=onp.int64) * -1
        self.raw_node_labels[lab_avail_nodes] = onp.argmax(onehots[lab_avail_nodes], axis=1)

        #
        self.raw_edge_srcs = []
        self.raw_edge_dsts = []
        self.raw_edge_feats = []
        (_, self.num_times, _) = feats.shape
        for t in range(self.num_times):
            #
            (dsts, srcs) = onp.nonzero(adjmats[t])
            weights = adjmats[t, dsts, srcs].astype(float)
            if onp.any(weights != 1):
                # UNEXPECT:
                # Node label must be unique.
                raise NotImplementedError("Edge has non-0/1 weight.")
            self.raw_edge_srcs.append(srcs)
            self.raw_edge_dsts.append(dsts)
            self.raw_edge_feats.append(weights)
            
        self.timestamps = onp.arange(self.num_times, dtype=onp.float64)

    def sanitize_edge(self, /) -> None:
        R"""
        Santiize edge data.
        """
        # All dynamic classification tasks are assumed to be directed graphs.
        self.edge_srcs = self.raw_edge_srcs
        self.edge_dsts = self.raw_edge_dsts
        self.edge_feats = self.raw_edge_feats
        self.edge_hetero = False
        self.edge_symmetric = False

    def asto_dynamic_adjacency_list_dynamic_edge(
        self,
        /,
        *,
        window_history_size: Optional[int], window_future_size: int,
        win_aggr: str, timestamped_edge_times: List[str],
        timestamped_node_times: List[str], timestamped_edge_feats: List[str],
        timestamped_node_feats: List[str],
    ) -> DynamicAdjacencyListDynamicEdge:
        R"""
        Transform dataset as temporal adjacency list (static edge) metaset.
        """
        #
        node_feats = onp.transpose(self.raw_node_feats, (0, 2, 1)) 
        # (num_nodes, num_times, num_feats) -> (num_nodes, num_feats, num_times)
        num_nodes = len(node_feats)
        node_labels = onp.reshape(self.raw_node_labels, (num_nodes, 1))

        #
        edge_feats = []
        edge_labels = []
        for t in range(self.num_times):
            #
            num_edges = len(self.edge_feats[t])
            edge_feats.append(onp.reshape(self.edge_feats[t], (num_edges, 1)))
            edge_labels.append(
                cast(
                    onpt.NDArray[onp.generic],
                    onp.zeros((num_edges, 1)).astype(onp.int64),
                ),
            )

        #
        metaset = (
            DynamicAdjacencyListDynamicEdge(
                self.edge_srcs, self.edge_dsts, edge_feats, edge_labels,
                node_feats, node_labels,
                hetero=self.edge_hetero, symmetrize=self.edge_symmetric,
                sort=True,
                task_type=CLF_TASK,
            )
        )
        metaset.dynamicon(dyn_node_feats=True, dyn_node_labels=False)
        metaset.timestamping(
            self.timestamps,
            timestamped_edge_times=timestamped_edge_times,
            timestamped_node_times=timestamped_node_times,
            timestamped_edge_feats=timestamped_edge_feats,
            timestamped_node_feats=timestamped_node_feats,
        )
        metaset.sliding_window(
            window_history_size=window_history_size,
            window_future_size=window_future_size,
        )
        metaset.sliding_aggregation(win_aggr=win_aggr)
        # customized info
        if hasattr(self, 'lab_avail_nodes'):
            metaset.lab_avail_nodes = self.lab_avail_nodes
        else:
            metaset.lab_avail_nodes = None

        if hasattr(self, 'A_rho'):
            metaset.A_rho = self.A_rho
            metaset.A_list = self.A_list
        return metaset


class Reddit4(DynamicClassification):
    #
    SOURCE = "Reddit4"


class DBLP5(DynamicClassification):
    #
    SOURCE = "DBLP5"

class DBLP5ss(DynamicClassification):
    #
    def __init__(self, dirname, /, filename=None) -> None:
        ss_size = os.path.basename(dirname).split("ss")[-1]
        self.SOURCE = "DBLP5ss{}".format(ss_size)
        super().__init__(dirname, filename=filename)


class IMDB(DynamicClassification):
    #
    SOURCE = "IMDB"

    def from_raw(self, dirname: str, /, filename:str=None) -> None:
        R"""
        Load from raw data.
        """
        if filename is not None:
            print(f'Loading from specified file: {filename}')
            matrices = (
                onp.load(os.path.join(dirname, filename))
            )
        else:
            print(f'Loading from default file: {self.SOURCE}_no_isolated.npz')
            matrices = (
                onp.load(os.path.join(dirname, "{:s}_no_isolated.npz".format(self.SOURCE)))
            )
        feats = matrices["attmats"]
        adjmats = matrices["adjs"]
        onehots = matrices["labels"]
        #
        # Safety check.
        if onp.any(adjmats < 0):
            # UNEXPECT:
            # Edge weights must be non-negative.
            raise NotImplementedError("Get invalid negative edge weights.")
        if onp.any(onp.sum(onehots > 0, axis=1) > 1):
            # UNEXPECT:
            # Node label must be unique.
            raise NotImplementedError("Get empty or duplicate node labels.")

        #
        self.raw_node_feats = feats # (3883,73)
        self.num_times = adjmats.shape[0]

        # duplicate static node features along time dimension
        self.raw_node_feats = onp.tile(self.raw_node_feats[:, None, :], 
                                       (1, self.num_times, 1))

        lab_avail_nodes = onp.where(onp.sum(onehots, axis=1) != 0)[0].tolist()
        
        print(f'Number of labeled available nodes: {len(lab_avail_nodes)} '
                f'({len(lab_avail_nodes) / onehots.shape[0] * 100:.2f}%)')
        if len(lab_avail_nodes) < onehots.shape[0]:
            self.lab_avail_nodes = lab_avail_nodes
        
        # (_, self.raw_node_labels) = onp.nonzero(onehots)
        self.raw_node_labels = onp.zeros((feats.shape[0], ), dtype=onp.int64) * -1
        self.raw_node_labels[lab_avail_nodes] = onp.argmax(onehots[lab_avail_nodes], axis=1)
        #
        (_, self.num_labels) = onehots.shape
        self.label_counts = onp.sum(onehots, axis=0).tolist()

        #
        self.raw_edge_srcs = []
        self.raw_edge_dsts = []
        self.raw_edge_feats = []

        for t in range(self.num_times):
            #
            (dsts, srcs) = onp.nonzero(adjmats[t]) 
            weights = adjmats[t, dsts, srcs].astype(float) 
            if onp.any(weights != 1):
                # UNEXPECT:
                # Node label must be unique.
                raise NotImplementedError("Edge has non-0/1 weight.")
            self.raw_edge_srcs.append(srcs)
            self.raw_edge_dsts.append(dsts)
            self.raw_edge_feats.append(weights)
            
        self.timestamps = onp.arange(self.num_times, dtype=onp.float64)

    def asto_dynamic_adjacency_list_dynamic_edge(
        self,
        /,
        *,
        window_history_size: Optional[int], window_future_size: int,
        win_aggr: str, timestamped_edge_times: List[str],
        timestamped_node_times: List[str], timestamped_edge_feats: List[str],
        timestamped_node_feats: List[str],
    ) -> DynamicAdjacencyListDynamicEdge:
        R"""
        Transform dataset as temporal adjacency list (static edge) metaset.
        """
        #
        node_feats = onp.transpose(self.raw_node_feats, (0, 2, 1)) 
        # (num_nodes, num_times, num_feats) -> (num_nodes, num_feats, num_times)
        num_nodes = len(node_feats)
        node_labels = onp.reshape(self.raw_node_labels, (num_nodes, 1))

        #
        edge_feats = []
        edge_labels = []
        for t in range(self.num_times):
            #
            num_edges = len(self.edge_feats[t])
            edge_feats.append(onp.reshape(self.edge_feats[t], (num_edges, 1)))
            edge_labels.append(
                cast(
                    onpt.NDArray[onp.generic],
                    onp.zeros((num_edges, 1)).astype(onp.int64),
                ),
            )

        #
        metaset = (
            DynamicAdjacencyListDynamicEdge(
                self.edge_srcs, self.edge_dsts, edge_feats, edge_labels,
                node_feats, node_labels,
                hetero=self.edge_hetero, symmetrize=self.edge_symmetric,
                sort=True,
                task_type=CLF_TASK,
                normalizer='normalize'
            )
        )
        metaset.dynamicon(dyn_node_feats=True, dyn_node_labels=False)
        metaset.timestamping(
            self.timestamps,
            timestamped_edge_times=timestamped_edge_times,
            timestamped_node_times=timestamped_node_times,
            timestamped_edge_feats=timestamped_edge_feats,
            timestamped_node_feats=timestamped_node_feats,
        )
        metaset.sliding_window(
            window_history_size=window_history_size,
            window_future_size=window_future_size,
        )
        metaset.sliding_aggregation(win_aggr=win_aggr)
        # customized info
        if hasattr(self, 'lab_avail_nodes'):
            metaset.lab_avail_nodes = self.lab_avail_nodes
        else:
            metaset.lab_avail_nodes = None

        if hasattr(self, 'A_rho'):
            metaset.A_rho = self.A_rho
            metaset.A_list = self.A_list
        return metaset

