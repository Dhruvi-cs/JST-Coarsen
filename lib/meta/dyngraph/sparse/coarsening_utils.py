R"""
"""
#
import numpy as onp
import numpy.typing as onpt
import scipy.sparse as osparse
import os
import torch
from typing import List, Callable, Tuple, cast, Optional, Union
from tqdm.auto import tqdm
    
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix, diags, issparse

import math 
import pickle 
import concurrent.futures
import wandb
import scipy
from scipy.stats import pearsonr
from ....exec.consts import CLF_TASK, REG_TASK, SMALL_NUMB, TRANS, INDUC, TEMP_AGG_METHOD, COARS_SCORE_TAG
import networkx as nx 
import warnings; warnings.filterwarnings('ignore')

GENERIC_NPArray = onpt.NDArray[onp.generic]

NEED_ASSERTION = False  # whether to perform assertions during execution

def get_Q_mat(P: GENERIC_NPArray) -> GENERIC_NPArray:
    # finding the coarsening matrix Q = C^{-1} P^T
    C = onp.sum(P, axis=0)
    C[C == 0] = 1.0  # Avoid division by zero
    C_inv = 1.0 / C  # shape (K, )
    C_inv_mat = onp.diag(C_inv)  # shape (K, K)
    Q = C_inv_mat @ P.T  # shape (K, N)
    return Q

def get_edge_count_mat(P: GENERIC_NPArray, A: GENERIC_NPArray) -> GENERIC_NPArray:
    '''
    Get the edge count matrix C = P^T @ nnz(A) @ P
    where nnz(A) is the adjacency matrix with 1 for non-zero entries.
    '''
    nnz_A = onp.zeros_like(A)
    nnz_A[A != 0] = 1.0
    C = P.T @ nnz_A @ P  # shape (K, K)
    return C


def get_coarsened_feat(
    X: GENERIC_NPArray,
    P: GENERIC_NPArray, 
    Q: GENERIC_NPArray, 
    coarsen_op: str = 'PtAP'
) -> GENERIC_NPArray:
    """
    Coarsen the node features X as average of the features of the nodes in each supernode.
    
    X: Node features matrix of shape (N, F) where N is the number of nodes and F is the number of features.
    P: Partition matrix of shape (N, K) where K is the number of supernodes.
    Q: Coarsening matrix of shape (K, N) where K is the number of supernodes.
        Q_{ki}=1/C_k where C_i = sum_j P_{ji} is the number of nodes in supernode i.
    Returns:
        X_coarse: Coarsened node features matrix of shape (K, F).
    """
    X_coarse = Q @ X  # Shape (K, F)
    return X_coarse


def coarsen_labels(
    Y, P, K, N, max_node_label, label_mask, 
    is_onehot, supernode_label_mask, 
    train_mask, valid_mask,
    coars_train_mask, coars_valid_mask, masked_mix_lb=False
) -> GENERIC_NPArray:
    '''
    P: (N, K) partition matrix
    Y: (N, ) or (N, C) node labels, either int or one-hot
    label_mask: (N, 1) bool mask indicating which nodes have labels
    supernode_label_mask: (K, 1) bool mask indicating which supernodes have labels
    train_mask: (K, 1) bool mask indicating which supernodes are in training set
    valid_mask: (K, 1) bool mask indicating which supernodes are in validation set
    Returns:
        Y_coarse: (K, ) or (K, C) coarsened node labels, either int or one-hot
    ''' 
    if not is_onehot: # convert to one-hot
        Y_onehot = onp.zeros((N, max_node_label+1))
        Y_onehot[onp.arange(N), Y.flatten()] = 1
    else:
        Y_onehot = Y
    
    # only consider the training nodes when doing majority vote
    if masked_mix_lb:
        Yc_multihot_train = (P.T * label_mask.T * train_mask.T) @ Y_onehot
        Y_coarse_int_train = onp.argmax(Yc_multihot_train, axis=1) # (K, )
        Yc_multihot_valid = (P.T * label_mask.T * valid_mask.T) @ Y_onehot
        Y_coarse_int_valid = onp.argmax(Yc_multihot_valid, axis=1) # (K, )
        Y_coarse_int = torch.zeros((K,), dtype=torch.long).numpy()
        Y_coarse_int[coars_train_mask.flatten()] = Y_coarse_int_train[coars_train_mask.flatten()]
        Y_coarse_int[coars_valid_mask.flatten()] = Y_coarse_int_valid[coars_valid_mask.flatten()]  

        # mask out the supernodes that have mixed labels, define new train/valid masks
        mix_label_train = onp.sum(Yc_multihot_train > 0, axis=1)  # (K, )
        mix_mask_train = mix_label_train > 1  # (K, )
        new_train_mask = coars_train_mask.copy()
        new_train_mask[mix_mask_train] = False
        mix_label_valid = onp.sum(Yc_multihot_valid > 0, axis=1)  # (K, )
        mix_mask_valid = mix_label_valid > 1  # (K, )
        new_valid_mask = coars_valid_mask.copy()
        new_valid_mask[mix_mask_valid] = False

        coars_train_mask = new_train_mask
        coars_valid_mask = new_valid_mask
    else:
        Yc_multihot = (P.T * label_mask.T) @ Y_onehot
        Y_coarse_int = onp.argmax(Yc_multihot, axis=1) # (K, )

    Y_coarse_oh = onp.zeros((K, max_node_label+1))
    Y_coarse_oh[onp.arange(K), Y_coarse_int] = 1
    if not is_onehot: # convert back to int labels
        Y_coarse = onp.argmax(Y_coarse_oh, axis=1, keepdims=True)
        Y_coarse[~supernode_label_mask.flatten()] = -1
    else:
        Y_coarse = Y_coarse_oh

    return Y_coarse, coars_train_mask, coars_valid_mask


def get_coarsened_edge_feats_and_labels(
    orig_edge_srcs, orig_edge_dsts, P, Pnorm, Q,
    orig_edge_feats, orig_edge_labels,
    meaningless_edge_labels,
    reweight_selfloops=True,
    edge_feat_reweight_method=None,
    coarsen_op='PtAP',
) -> Tuple[
    List[GENERIC_NPArray], 
    List[GENERIC_NPArray], 
    List[GENERIC_NPArray]
]:
    # get edge features
    super_edges, inv_idx, agg_edge_feats = aggregate_edges_indexadd_np(
        P, onp.stack([orig_edge_srcs, orig_edge_dsts], axis=0), 
        orig_edge_feats, edge_feat_reweight_method,
        reweight_selfloops,
        coarsen_op=coarsen_op
    )
    
    # edge label is the dominant label of the merged edges
    if not meaningless_edge_labels:
        super_edge_labels = aggregate_majority_labels_numpy(
            inv_idx, orig_edge_labels, super_edges.shape[0]
        )
    else:
        super_edge_labels = onp.zeros((super_edges.shape[0],), dtype=orig_edge_labels.dtype)

    return super_edges, agg_edge_feats, super_edge_labels


def aggregate_edges_indexadd_np(
    P, edge_index, edge_attr, 
    edge_feat_reweight_method=None, 
    reweight_selfloops=True, 
    coarsen_op='PtAP'
) -> Tuple[GENERIC_NPArray, GENERIC_NPArray, GENERIC_NPArray]:
    """
    NumPy version: Aggregate edge features into super-edges using index_add_.
    
    Args:
        P: (N, K) NumPy array, hard assignment matrix (rows one-hot)
        edge_index: (2, E) NumPy array, node-level edges
        edge_attr: (E, D) NumPy array, edge features
        edge_feat_reweight_method: str or None, method to reweight edge features, 
            options: None, 'avg', 'cardinality_s', 'cardinality_d', 'cardinality_sd'
            if None, it's summation of edge features
    Returns:
        super_edges: (E_c, 2) NumPy array of unique super-edges (i,j)
        edge_attr_agg: (E_c, D) NumPy array of aggregated features
    """
    # Convert to torch tensors
    P_t = torch.from_numpy(P)
    edge_index_t = torch.from_numpy(edge_index)
    edge_attr_t = torch.from_numpy(edge_attr)

    # Map each node to its super-node
    node2super = P_t.argmax(dim=1)

    # Map each edge to super-edge
    src, dst = edge_index_t
    super_src = node2super[src]
    super_dst = node2super[dst]
    super_edge_index = torch.stack([super_src, super_dst], dim=1)

    # Unique super-edges & inverse mapping
    super_edges, ind2uniqvec = torch.unique(super_edge_index, dim=0, return_inverse=True)
    # super_edges should be symmetric (include both (i,j) and (j,i))

    # Aggregate edge features by super-edge id
    E_c = super_edges.size(0)
    D = edge_attr_t.size(1)
    edge_attr_agg = torch.zeros(E_c, D, dtype=edge_attr_t.dtype)
    edge_attr_agg.index_add_(0, ind2uniqvec, edge_attr_t)
    unique_src = super_edges[:, 0]
    unique_dst = super_edges[:, 1]
    
    if edge_feat_reweight_method == 'avg':
        # average the edge features by the number of edges merged
        counts = torch.bincount(ind2uniqvec, minlength=E_c).unsqueeze(1).to(edge_attr_agg.dtype)
        counts[counts == 0] = 1.0
        edge_attr_agg = edge_attr_agg / counts
    elif edge_feat_reweight_method == 'cardinality_s':
        # reweight the edge features by the cardinality of the source supernode
        src_cardinalities = P_t.sum(dim=0)[unique_src]
        src_cardinalities[src_cardinalities == 0] = 1.0
        edge_attr_agg = edge_attr_agg / src_cardinalities.unsqueeze(1).to(edge_attr_agg.dtype)
    elif edge_feat_reweight_method == 'cardinality_d':
        # reweight the edge features by the cardinality of the destination supernode
        dst_cardinalities = P_t.sum(dim=0)[unique_dst]
        dst_cardinalities[dst_cardinalities == 0] = 1.0
        edge_attr_agg = edge_attr_agg / dst_cardinalities.unsqueeze(1).to(edge_attr_agg.dtype)
    elif edge_feat_reweight_method == 'cardinality_sd':
        # reweight the edge features by the product of the cardinalities of the source and destination supernodes
        src_cardinalities = P_t.sum(dim=0)[unique_src]
        dst_cardinalities = P_t.sum(dim=0)[unique_dst]
        prod_sqrt_card = torch.sqrt(src_cardinalities * dst_cardinalities)
        prod_sqrt_card[prod_sqrt_card == 0] = 1.0
        edge_attr_agg = edge_attr_agg / prod_sqrt_card.unsqueeze(1).to(edge_attr_agg.dtype)
    
    if not edge_feat_reweight_method and reweight_selfloops:
        # reweight self-loops to be the average edge feature self-loops edges
        self_loop_mask = super_edges[:, 0] == super_edges[:, 1]
        # count the number of self-loop edges merged
        counts = torch.bincount(ind2uniqvec, minlength=E_c).unsqueeze(1).to(edge_attr_agg.dtype)
        counts[counts == 0] = 1.0
        avg_self_loop_feats = edge_attr_agg[self_loop_mask] / counts[self_loop_mask]
        edge_attr_agg[self_loop_mask] = avg_self_loop_feats

    # Convert back to NumPy
    return super_edges.numpy(), ind2uniqvec.numpy(), edge_attr_agg.numpy()

def normalize_partition_matrix(P: GENERIC_NPArray, norm_method=None) -> GENERIC_NPArray:
    """
    Normalize the partition matrix P such that: P = P C^{-1/2}, 
    where C is the diagonal matrix with C_ii = sum_j P_ij
    
    P: Partition matrix of shape (N, K) where N is the number of nodes and K is the number of supernodes.
    Returns:
        P_norm: Normalized partition matrix of shape (N, K).
    """
    # C = onp.sum(P, axis=0) # shape (K, )
    # C[C == 0] = 1.0  # Avoid division by zero
    # C_inv_sqrt = onp.diag(1.0 / onp.sqrt(C)) # shape (K, K)
    # P_norm = P @ C_inv_sqrt # shape (N, K)
    if norm_method == 'cardinality_sqrt':
        C = onp.sum(P, axis=0) # shape (K, )
        C[C == 0] = 1.0  # Avoid division by zero
        C_inv_sqrt = onp.diag(1.0 / onp.sqrt(C)) # shape (K, K)
        P_norm = P @ C_inv_sqrt # shape (N, K)
    elif norm_method == 'cardinality':
        C = onp.sum(P, axis=0) # shape (K, )
        C[C == 0] = 1.0  # Avoid division by zero
        C_inv = onp.diag(1.0 / C) # shape (K, K)
        P_norm = P @ C_inv # shape (N, K)
    else:
        P_norm = P
    return P_norm


def coarsen_graph(
    num_times, N, K, P: GENERIC_NPArray, 
    split_indices: Tuple[List[int], List[int], List[int]],
    edge_srcs: List[GENERIC_NPArray], 
    edge_dsts: List[GENERIC_NPArray], 
    edge_feats: List[GENERIC_NPArray], 
    edge_labels: List[GENERIC_NPArray], 
    node_feats: GENERIC_NPArray, 
    node_labels: GENERIC_NPArray, 
    
    dyn_node_feats: bool, dyn_node_labels: bool,
    node_label_size: int, meaningless_edge_labels: bool,
    coarsen_op: str,
    reweight_selfloops: bool = True,
    edge_feat_reweight_method: Optional[str] = None, # None means sum
    
    task_type: str = REG_TASK
) -> Tuple[
    List[GENERIC_NPArray], List[GENERIC_NPArray], 
    List[GENERIC_NPArray], List[GENERIC_NPArray], 
    Union[GENERIC_NPArray, List[GENERIC_NPArray]], 
    Union[GENERIC_NPArray, List[GENERIC_NPArray]]
]:
    ''' 
    Coarsening matrix P: shape (N, K)
        P_ij = 1 if node i belongs to supernode j, else 0
    Pseudo-inverse Q: shape (K, N)
        Q = C^{-1} P^T, where C is the diagonal matrix with C_ii = sum_j P_ji
        C_ii is the number of nodes in supernode i
    Coarsening framework:
        - Adjacency matrix A_coarse = P^T A P - sum of edge weights between supernodes
        - Node features X_coarse = Q X - average of node features in each supernode
        - Node labels Y_coarse: for classification, majority vote; for regression, average
        - Edge features: average edge features of edges between supernodes
    Returns:
        edge_srcs_coarse: List of np arrays of shape (E_c_t, ) for each time t
        edge_dsts_coarse: List of np arrays of shape (E_c_t, ) for each time t
        edge_feats_coarse: List of np arrays of shape (E_c_t, D) for each time t
        edge_labels_coarse: List of np arrays of shape (E_c_t, L) for each time t
        node_feats_coarse: np array of shape (K, F) or List of np arrays of shape (K, F) for each time t
        node_labels_coarse: np array of shape (K, C) or List of np arrays of shape (K, C) for each time t
        
    '''    
    # update the aggregated adjacency matrices
    # new_dyn_adjs = []
    new_edge_srcs = []
    new_edge_dsts = []
    new_edge_feats = []
    new_edges_labels = []
    new_node_feats = []
    new_node_labels = []
    
    #
    max_node_label = onp.max(node_labels)
    is_onehot = (max_node_label == 1) and (node_label_size >= 2)
    Q = get_Q_mat(P)  # shape (K, N)

    train_indices, valid_indices, test_indices = split_indices
    nontest_mask = onp.ones((N,1), dtype=bool) # (N, 1)
    MASKED_MIX_LB = False
    # we don't use any test node labels during merging, 
    # so all supernodes are valid for training/validation
    supernode_label_mask = onp.ones((K,1), dtype=bool) # (K, 1) 
    if len(train_indices) > 0 and len(valid_indices) > 0 and len(test_indices) > 0:
        # Transductive setting
        if len(test_indices) > 0:
            nontest_mask[list(test_indices)] = False

        train_mask = onp.zeros((N,1), dtype=bool) # (N, 1)
        train_mask[list(train_indices)] = True
        valid_mask = onp.zeros((N,1), dtype=bool) # (N, 1) 
        valid_mask[list(valid_indices)] = True
        
        coars_train_mask = (P.T @ train_mask) > 0 # (K, 1)
        coars_valid_mask = (P.T @ valid_mask) > 0 # (K, 1)
        coars_train_indices = onp.nonzero(coars_train_mask)[0]
        coars_valid_indices = onp.nonzero(coars_valid_mask)[0]
    else:
        # Inductive setting
        coars_train_indices = train_indices
        coars_valid_indices = valid_indices


    for t in range(num_times):
        # coarsen the node features
        if dyn_node_feats:
            X = node_feats[:, :, t] # (num_nodes, num_features)
            X_coarse = get_coarsened_feat(X, P, Q, coarsen_op=coarsen_op)
            new_node_feats.append(X_coarse)
        elif t == 0:
            # if node features are static, we only need to compute it once
            X = node_feats # (num_nodes, num_features)
            X_coarse = get_coarsened_feat(X, P, Q, coarsen_op=coarsen_op)
            new_node_feats = X_coarse
            
        # coarsen the node labels
        if dyn_node_labels:
            Y = node_labels[:, :, t]
        elif t == 0:
            Y = node_labels

        if dyn_node_labels or (not dyn_node_labels and t == 0):
            if task_type == CLF_TASK:
                # Majority vote for classification task
                Y_coarse, coars_train_mask, coars_valid_mask = \
                coarsen_labels(
                    Y, P, K, N, max_node_label, 
                    nontest_mask, is_onehot, 
                    supernode_label_mask, 
                    train_mask, valid_mask,
                    coars_train_mask, coars_valid_mask,
                    masked_mix_lb=MASKED_MIX_LB
                )
            else:
                assert task_type == REG_TASK, f"Unknown task type: {task_type}"
                # Average the node labels for regression task
                Y_coarse = get_coarsened_feat(Y, P, Q, coarsen_op=coarsen_op)
        
        if dyn_node_labels:
            new_node_labels.append(Y_coarse)
        elif t == 0:
            new_node_labels = Y_coarse
        
        orig_edge_feats = edge_feats[t]
        orig_edge_labels = edge_labels[t]

        new_edges, new_efeats, new_elabels = \
            get_coarsened_edge_feats_and_labels(
                edge_dsts[t], edge_srcs[t], P, P, Q,
                orig_edge_feats, orig_edge_labels,
                meaningless_edge_labels,
                reweight_selfloops=reweight_selfloops,
                edge_feat_reweight_method=edge_feat_reweight_method,
                coarsen_op=coarsen_op
            )
        
        new_edge_srcs.append(new_edges[:, 0])
        new_edge_dsts.append(new_edges[:, 1])
        
        new_elabels = onp.reshape(new_elabels, (-1, edge_labels[t].shape[1]))
        new_edge_feats.append(new_efeats)
        new_edges_labels.append(new_elabels)
    
    # update the window edge srcs and dsts
    edge_srcs = new_edge_srcs
    edge_dsts = new_edge_dsts
    edge_feats = new_edge_feats
    edge_labels = new_edges_labels
    node_feats = onp.stack(new_node_feats, axis=2) \
        if dyn_node_feats else new_node_feats
    node_labels = onp.stack(new_node_labels, axis=2) \
        if dyn_node_labels else new_node_labels
    # import pdb; pdb.set_trace()

    return (
        edge_srcs, edge_dsts, edge_feats, 
        edge_labels, node_feats, node_labels,
        supernode_label_mask, coars_train_indices, 
        coars_valid_indices
    )

















def sym_norm_laplacian(A, norm=False, is_symmetric: bool = True) -> osparse.csr_matrix:
    if not is_symmetric:
        A = (A + A.T) / 2.0  # Ensure symmetry
    
    # Degree vector
    degrees = onp.array(A.sum(axis=1)).flatten()
    
    # Avoid division by zero for isolated nodes
    degrees[degrees == 0] = 1.0
    if norm:
        # D^{-1/2} as diagonal sparse matrix
        D_inv_sqrt = diags(1.0 / onp.sqrt(degrees))
        
        # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        I = diags(onp.ones(A.shape[0]))
        L = I - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        # Combinatorial Laplacian: L = D - A
        D = diags(degrees)
        L = D - A

    return L


import time

def get_supra_spectral_gap(window_edge_srcs: List[GENERIC_NPArray], 
                     window_edge_dsts: List[GENERIC_NPArray], 
                     P: GENERIC_NPArray,
                     num_times: int,
                     N: int, K: int, symmetrize=True) -> float:
    '''
    Compute the spectral gap of the combinatorial Laplacian of the supra-adjacency matrix
    of the coarsened dynamic graph.
    Parameters
    ----------
    window_edge_srcs : List of np arrays
        List of edge source indices for each snapshot.
    window_edge_dsts : List of np arrays
        List of edge destination indices for each snapshot.
    P : np array
        Partition matrix of shape (N, K) where K is the number of supernodes.
    num_times : int
        Number of time snapshots.
    N : int
        Number of nodes in the original graph.
    K : int
        Number of supernodes in the coarsened graph.
    Returns
    ------- 
    gap : float
        Spectral gap = min positive eigenvalue of combinatorial Laplacian.
    ''' 
    start = time.time() 
    new_adjs = []
    # make P sparse
    _P = csr_matrix(P)
    for t in range(num_times):
        srcs = window_edge_srcs[t]
        dsts = window_edge_dsts[t]
        # create sparse adj, make sure it's symmetric
        data = onp.ones(len(srcs))
        _data = onp.concatenate([data, data])
        _srcs = onp.concatenate([srcs, dsts])
        _dsts = onp.concatenate([dsts, srcs])
        A = csr_matrix((_data, (_srcs, _dsts)), shape=(N, N))
        # remove duplicate edges if already symmetric
        A.data[:] = 1.0
        A_coarse = _P.T @ A @ _P
        new_adjs.append(A_coarse)
    print(f"Time to compute coarsened adjs: {time.time() - start:.2f} seconds.") 
    # compute the spectral gap of the new aggregated graph
    # we only need the second smallest eigenvalue
    # use scipy.sparse.linalg.eigsh for efficiency
    # construct the block diagonal matrix
    block_mat = [] 
    I_sp = osparse.identity(K)
    csI_sp = _P.T @ I_sp @ _P
    Zr_sp = osparse.csr_matrix((K, K))
    
    for i in range(num_times):
        block_row = []
        for j in range(num_times):
            if i == j:
                block_row.append(new_adjs[i])
            elif i == j - 1 or i == j + 1:
                block_row.append(csI_sp)
            else:
                block_row.append(Zr_sp)
        block_mat.append(block_row)
        
    # block_mat = onp.block(block_mat)
    # create sparse matrix
    block_mat = osparse.bmat(block_mat)
    block_mat = csr_matrix(block_mat)
    print(f"Time to construct supra adjacency matrix: {time.time() - start:.2f} seconds.")

    eigvals, eigvecs, Ac_norm, Lc, eigval_comps, \
        cc_sizes_t, cc_nodes = get_Lap_ccs_gap(block_mat)
        
    return (
        eigvals, eigvecs, Ac_norm, 
        Lc, cc_sizes_t, cc_nodes
    )

try:
    import pyamg
except ImportError:
    pyamg = None
from scipy.sparse.linalg import lobpcg

def get_Lap_ccs_gap(A, tol=1e-6):
    """
    Compute the spectral gap of the combinatorial Laplacian of a graph.
    Symmetrize if directed adjacency matrix is given.
    Handles multiple connected components.
    
    Parameters
    ----------
    A : ndarray or sparse matrix
        Adjacency matrix (may be directed).
    tol : float
        Numerical tolerance for eigsh.

    Returns
    -------
    eigvals : ndarray (N,)
        Eigenvalues of the full Laplacian.
    eigvecs : ndarray (N, )
        Corresponding eigenvectors of the full Laplacian.
    Ac_norm : sparse matrix (N, N)
        Normalized adjacency matrix.
    Lc : sparse matrix (N, N)
        Combinatorial Laplacian matrix.
    cc_sizes_t : List[int]
        Sizes of each connected component.
    cc_nodes : List[List[int]]
        Node indices for each connected component.
        
    """
    def sparse_allclose(a, b, rtol=1e-5, atol=1e-8):
        if not (a.shape == b.shape and a.nnz == b.nnz):
            return False

        # Ensure consistent format for comparison (e.g., convert to CSR)
        a_csr = a.tocsr()
        b_csr = b.tocsr()

        # Compare structural elements (pointers and indices)
        if not (onp.array_equal(a_csr.indptr, b_csr.indptr) and
                onp.array_equal(a_csr.indices, b_csr.indices)):
            return False

        # Compare data values using numpy.allclose
        return onp.allclose(a_csr.data, b_csr.data, rtol=rtol, atol=atol)
    
    # Ensure symmetric (for undirected interpretation)
    if not issparse(A):
        A = csr_matrix(A)
    assert sparse_allclose(A, A.T), "Adjacency matrix A must be symmetric."

    n = A.shape[0]
    G = nx.from_scipy_sparse_array(A)
    comps = list(nx.connected_components(G))
    
    eigval_comps, eigvec_comps = [], []
    comp_inv_map = []
    Laps = []
    fL_rows, fL_cols, fL_data = [], [], []
    cc_sizes = []

    for comp in comps:
        nodes = list(comp)
        if len(nodes) <= 1:
            # isolated node, skip
            continue
        
        H = G.subgraph(nodes)
        Lap = nx.laplacian_matrix(H).asfptype()
        Lap = csr_matrix(Lap)

        Laps.append(Lap)
        # extract the data for full Laplacian
        Lap_coo = Lap.tocoo()
        fL_rows.extend(onp.array(nodes)[Lap_coo.row])
        fL_cols.extend(onp.array(nodes)[Lap_coo.col])
        fL_data.extend(Lap_coo.data)

        # Compute second smallest eigenvalues

        if len(nodes) == 2:
            # for 2-node component, directly compute eigenvalues
            vals, vecs = onp.linalg.eigh(Lap.toarray())
        elif len(nodes) >= 50_000:
            # # for very large components, use shift-invert mode
            try:
                ml = pyamg.smoothed_aggregation_solver(Lap)
                M = ml.aspreconditioner()
                X = onp.random.randn(Lap.shape[0], 2)
                # vals, vecs = eigsh(Lap, k=2, sigma=1e-4, which='SM', tol=tol)
                vals, vecs = lobpcg(Lap, X, M=M, tol=1e-6, maxiter=5000, largest=False)
                # compute the solution quality
                residuals = onp.linalg.norm(Lap @ vecs - vals * vecs, axis=0)
                print(f"lobpcg residuals: {residuals}")
            except Exception as e:
                print(f"lobpcg failed with error: {e}")
                import pdb;pdb.set_trace()
        else:
            # use eigsh for larger components
            vals, vecs = eigsh(Lap, k=2, which='SM', tol=tol)

        vals = onp.real(vals)
        vecs = onp.real(vecs)
        vals = onp.sort(onp.clip(vals, 0.0, None))  # clamp tiny negatives to 0
        spectral_gap_idx = 1
        sp_gap = vals[spectral_gap_idx]
        F_vec = vecs[:, spectral_gap_idx] # Fiedler vector
        assert sp_gap > 0, "Spectral gap should be positive."
        
        eigval_comps.append(sp_gap)
        eigvec_comps.append(F_vec)
        cc_sizes.append(len(nodes))
        comp_inv_map.append(nodes)

    # pad eigenvectors to full size
    # note that eigenvalues are the same for all nodes in the same component
    # NOTE: eigenvectors are not supposed to be used concurrently for different components,
    # just for faster accessing the values given the node index in the graph
    eigvals = onp.zeros(n) # 
    eigvecs = onp.zeros(n)
    for i, nodes in enumerate(comp_inv_map):
        eigvals[nodes] = eigval_comps[i]
        eigvecs[nodes] = eigvec_comps[i]
    
    Laplacian_full = csr_matrix((fL_data, (fL_rows, fL_cols)), shape=(n, n))
    #
    return eigvals, eigvecs, A, Laplacian_full, eigval_comps, cc_sizes, comp_inv_map

def get_snapshots_spectral_gap(
        window_edge_srcs: List[GENERIC_NPArray], 
        window_edge_dsts: List[GENERIC_NPArray], 
        P: GENERIC_NPArray,
        num_times: int,
        N: int, K: int
    ) -> float:
    
    snaps_spectral_gaps = []
    snaps_eigenvecs = []
    cc_sizes = [] 
    cc_nodess = []
    
    As, Acs, Lcs = [], [], []
    for t in tqdm(range(num_times)):
        srcs = window_edge_srcs[t]
        dsts = window_edge_dsts[t]
        A = onp.zeros((N, N))
        A[srcs, dsts] = 1.0
        A[dsts, srcs] = 1.0  # make it symmetric
        As.append(A)
        A_coarse = P.T @ A @ P
        if (A.T == A).all():
            assert (A_coarse.T == A_coarse).all(), "A_coarse must be symmetric."
        
        eigvals, eigvecs, Ac_norm, Lc, eigval_comps, \
        cc_sizes_t, cc_nodes = get_Lap_ccs_gap(A_coarse)
        
        Acs.append(Ac_norm)
        Lcs.append(Lc)
        snaps_spectral_gaps.append(eigvals)
        snaps_eigenvecs.append(eigvecs)
        cc_sizes.append(cc_sizes_t)
        cc_nodess.append(cc_nodes)
    
    return snaps_spectral_gaps, snaps_eigenvecs, Acs, Lcs, cc_sizes, cc_nodess

def get_coarsened_adj(A: GENERIC_NPArray,
                    P: GENERIC_NPArray, 
                    Q: int, method='PtAP') -> GENERIC_NPArray:
    """
    Coarsen the adjacency matrix A using the partition matrix P.
    A: Adjacency matrix of shape (N, N) where N is the number of nodes.
    P: Partition matrix of shape (N, K) where K is the number of supernodes.
    method: str, including 'PtAP', 'QAP'
    Returns:
        A_coarse: Coarsened adjacency matrix of shape (K, K).
    """
    if method == 'PtAP':
        # summation of the edges between supernodes
        A_coarse = P.T @ A @ P  # Shape (K, K)
    elif method == 'QAP':
        # Option 2: Summation with weighting: Q = C^{-1} P^T, 
        # where C is the diagonal matrix with C_ii = sum_j P_ij
        A_coarse = Q @ A @ P # shape (K, K)
    elif method == 'CinvPtAP':
        # average of the edges between supernodes
        # need to count number of edges within each supernode
        nnz_A = onp.zeros_like(A)
        nnz_A[A != 0] = 1.0
        C = P.T @ nnz_A @ P  # shape (K, K)
        C[C == 0] = 1.0  # avoid division by zero
        C_inv = 1.0 / C
        A_coarse = C_inv * (P.T @ A @ P)  # shape (K, K)
    else: 
        raise ValueError(f"Unknown coarsening method: {method}")
    
    return A_coarse


def aggregate_majority_labels_numpy(inv_idx, edge_labels, num_super_edges):
    """
    Aggregate edge labels to super-edge labels using majority vote.
    
    Args:
        inv_idx: (E,) NumPy array mapping each edge -> super-edge id
        edge_labels: (E,) NumPy array of integer labels
        num_super_edges: int, number of unique super-edges
        
    Returns:
        super_edge_labels: (num_super_edges,) NumPy array of majority labels
    """
    num_classes = int(edge_labels.max() + 1)  # assume labels are in [0, C-1]

    # Count matrix: rows = super-edges, cols = classes
    counts = onp.zeros((num_super_edges, num_classes), dtype=onp.int64)

    # Add 1 for each edge at its super-edge & class
    onp.add.at(counts, (inv_idx, edge_labels), 1)

    # Majority label for each super-edge = argmax along classes
    super_edge_labels = counts.argmax(axis=1)

    return super_edge_labels

def map_to_supernodes(sorted_edges, num_nodes, Nprime):
    """
    Use union-find to merge nodes according to top edges, until the number of clusters + remaining nodes == Nprime.
    Returns a mapping from node to supernode id (0..Nprime-1).
    
    TODO: there will be the case that number of edges is not enough to merge nodes as many nodes to reach Nprime supernodes.
    """

    def find(u, parent):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v, parent, rank, size):
        u_root = find(u, parent)
        v_root = find(v, parent)
        if u_root == v_root:
            return False
        
        if rank[u_root] < rank[v_root]:
            parent[u_root] = v_root
            size[v_root] += size[u_root]
        else:
            parent[v_root] = u_root
            size[u_root] += size[v_root]
            if rank[u_root] == rank[v_root]:
                rank[u_root] += 1
        return True

    parent = list(range(num_nodes))
    rank = [0] * num_nodes
    size = [1] * num_nodes
    merged_pairs = []
    
    n_super_nodes = num_nodes
    for edge in sorted_edges:
        (u, v), _ = edge
        if union(u, v, parent, rank, size):
            n_super_nodes -= 1
            merged_pairs.append(edge)
            
        if n_super_nodes == Nprime:
            break

    if n_super_nodes > Nprime:
        print(f"!!!!!==Warning==!!!!!: Not enough edges to reach {Nprime} supernodes, "
            f"current supernodes: {n_super_nodes}. Merging supernodes to reach target.")
        # deterministic, size-aware fallback: merge smallest clusters first
        roots = {}
        for u in range(num_nodes):
            r = find(u, parent)
            roots.setdefault(r, size[r])
        # sorted by cluster size ascending
        sorted_roots = sorted(roots.items(), key=lambda x: x[1])
        i = 0
        while n_super_nodes > Nprime:
            active_idx = i % (len(sorted_roots) - 1)
            u_root, _ = sorted_roots[active_idx]
            v_root, _ = sorted_roots[active_idx + 1]
            if union(u_root, v_root, parent, rank, size):
                n_super_nodes -= 1
                merged_pairs.append(((u_root, v_root), -onp.inf))  # mark fallback merge
                # refresh the sorted list locally
                new_root = find(u_root, parent)
                merged_size = size[new_root]
                # replace positions i and i+1 with merged root
                sorted_roots[active_idx] = (new_root, merged_size)
                del sorted_roots[active_idx + 1]
                # as the merged cluster may not be the smallest anymore, we need to re-sort
                sorted_roots = sorted(sorted_roots, key=lambda x: x[1])
                # do not increment i, as we need to check the new cluster again
            else:
                i += 1  # try next pair
            
        if n_super_nodes > Nprime:
            raise RuntimeError(f"Could not reach {Nprime} supernodes; got {n_super_nodes}")

        print(f"After random merging, current supernodes: {n_super_nodes}.")
    
    # Assign cluster ids
    cluster_roots = {}
    cluster_id = 0
    mapping = {}
    for u in range(num_nodes):
        root = find(u, parent)
        if root not in cluster_roots:
            cluster_roots[root] = cluster_id
            cluster_id += 1
        mapping[u] = cluster_roots[root]

    assert len(set(mapping.values())) == n_super_nodes
    return mapping, merged_pairs

def compute_snapshot_eigen_perturbs(nodes_eigvec, nodes_eigval, 
                                    nodes_cc, node_degs, cc_nodes, cc_sizes,
                                    a_idx, b_idx, w_ab):
    """
    Only support one pair of nodes.
    -----------
    
    Compute the change in eigenvalue after merging a pair of nodes
    based on matrix perturbation theory.

    Args:
        - nodes_eigvec (ndarray (N,)): the eigenvector of the 
            connected components arranged in the order of nodes.
        - nodes_eigval (ndarray (N,)): array, the eigenvalue of the 
            connected component that contains corresponding nodes.
        - nodes_cc (ndarray (N,)): array, the connected component index of each node.
        - node_degs (ndarray (N,)): array, the degree of each node
        - cc_nodes: list of lists, each sublist contains the node indices in that connected component.
        - cc_sizes (ndarray (num_cc,)): array, the sizes of each connected component.
        - a_idx (ndarray (k,)): indices of nodes a.
        - b_idx (ndarray (k,)): indices of nodes b.
        - w: list or numpy array, weights w(a_t, b_t) for each t.
    
    Returns:
        - float, the computed Delta lambda.
    """
    # Convert indices and w to numpy arrays if not already
    _lambda_a = nodes_eigval[a_idx] # shape (1, )
    _lambda_b = nodes_eigval[b_idx] # shape (1, )
    w_ab = onp.array(w_ab)  # shape (1, )
    if _lambda_a == 0.0 or _lambda_b == 0.0:
        # connecting to isolated component, no change in eigenvalue
        return 0.0, 1.0, 1.0
    
    _lambda_ab = onp.minimum(_lambda_a, _lambda_b)

    # Extract connected component sizes
    cc_idx_a = nodes_cc[a_idx] # (1,)
    cc_idx_b = nodes_cc[b_idx] # (1,)
    if cc_idx_a == cc_idx_b:
        ccab_nodes = cc_nodes[cc_idx_a] # list of nodes in the connected component
    else:
        ccab_nodes = cc_nodes[cc_idx_a] + cc_nodes[cc_idx_b] # list of nodes in the two connected components
    # Extract v_ccab
    v_ccab = nodes_eigvec[ccab_nodes] # shape (|cc_a| + |cc_b|, )
    # 
    same_cc_idx = (cc_idx_a == cc_idx_b) # shape (1, )
    # Extract v_a and v_b
    v_a = nodes_eigvec[a_idx] # shape (1, )
    v_b = nodes_eigvec[b_idx] # shape (1, )
    
    same_lambda_idx = (_lambda_a == _lambda_b) # shape (1, )
    
    # mask out the different cc and different lambda values pairs
    mask = same_cc_idx | same_lambda_idx # shape (1, )
    if mask:
        # Compute sums using vector operations
        va_sq = v_a ** 2 # shape (1, )
        vb_sq = v_b ** 2
        va_vb = v_a * v_b
        deg_a = node_degs[a_idx]
        deg_b = node_degs[b_idx]
        # symmetric score
        numerator = (va_sq - 2 * va_vb + vb_sq) \
                    * (0.25 * (deg_a + deg_b + 2 * w_ab) - _lambda_ab)
        numerator = onp.sum(numerator) # sum over valid pairs
        denominator = onp.dot(v_ccab, v_ccab)
        if denominator == 0:
            print("Denominator in eigenvalue perturbation is zero. Setting perturbation to 0.")
            eigen_pert = 0.0
        else:
            eigen_pert = numerator / denominator
        orig_lambda = _lambda_ab
    else:
        # Heuristically estimate the eigenvalue change for non-degenerate case (different eigenval)
        deg_a = node_degs[a_idx]
        deg_b = node_degs[b_idx]
        size_a = cc_sizes[cc_idx_a]
        size_b = cc_sizes[cc_idx_b]

        # https://snap.stanford.edu/class/cs224w-2018/handouts/07-spectral.pdf - Slide 24
        eigenbounda = deg_a * (1 / (size_a - 1) + 1 / size_b)
        eigenboundb = deg_b * (1 / size_a + 1 / (size_b -1))
        eigenbound = min(eigenbounda, eigenboundb) 

        switch_eig = eigenbound < _lambda_ab # shape (1, )
        eigen_pert = eigenbound - _lambda_ab if switch_eig else 0.0
        orig_lambda = _lambda_ab 

    return eigen_pert, orig_lambda, len(ccab_nodes)


def agg_score(temporal_cc_scores, agg_method_cc=None, 
              agg_method_temp='min', skip_zero=False) -> float:
    '''
    Aggregate the scores across connected components and snapshots.
    Args:
        - temporal_cc_scores: List[List[float]] or List[float], list of scores for each snapshot, 
            each snapshot has a list of scores for each connected component.
        - method: str, aggregation method, either 'max' or 'min'.
    Returns:
        - score: float, aggregated score.
    '''
    ops = {'max': onp.max, 'min': onp.min, 'mean': onp.mean, 'sum': onp.sum}
    
    # Aggregate scores across connected components for each snapshot
    if agg_method_cc is not None:
        if agg_method_temp is None and not isinstance(temporal_cc_scores[0], (list, onp.ndarray)):
            temporal_cc_scores = [temporal_cc_scores]
        
        if skip_zero:
            temporal_cc_scores = [cc_scores[cc_scores != 0] for cc_scores in temporal_cc_scores]
        
        op1 = ops[agg_method_cc]
        agg_cc_score = [op1(cc_scores) if len(cc_scores) > 0 
                        else 0.0 for cc_scores in temporal_cc_scores]
    else:
        agg_cc_score = temporal_cc_scores
        
    # Aggregate scores across snapshots
    if agg_method_temp is None:
        assert len(agg_cc_score) == 1, "If no temporal aggregation, there should be only one snapshot."
        return agg_cc_score[0]
    else:
        if skip_zero:
            agg_cc_score = onp.array(agg_cc_score)
            agg_cc_score = agg_cc_score[agg_cc_score != 0]
            
        op2 = ops[agg_method_temp]
        agg_temp_score = op2(agg_cc_score) if len(agg_cc_score) > 0 else 0.0
        
    return agg_temp_score

def get_links_scores_over_coarsened_dyngraph(
    num_times, N, K,
    window_edge_srcs: List[GENERIC_NPArray], 
    window_edge_dsts: List[GENERIC_NPArray], 
    init_info: Tuple[List],
    ss_check_pairs: List[Tuple[int, int]],
    SPEC_GAP_METHOD: str,
    agg_method_temp = TEMP_AGG_METHOD  # 'max' or 'sum'
) -> dict:
    '''
    Compute the spectral gap for each pair of nodes in ss_check_pairs,
    when they are merged into a supernode, and all other nodes are mapped to their own
    supernodes.
    Return a dictionary mapping from (u, v) to spectral gap.
    '''
    
    init_eigenvals, init_eigenvecs, As, \
        Ls, cc_sizess, cc_nodess = init_info
    nodes_degs = [onp.array(A.sum(axis=1)).flatten() for A in As]

    nodes_ccs = []
    for t in range(num_times):
        nodes_cc = onp.ones((N,), dtype=int) * -10000 # some nodes are not in any cc (isolated nodes)
        for cc_idx, nodes in enumerate(cc_nodess[t]):
            nodes_cc[nodes] = cc_idx
        nodes_ccs.append(nodes_cc)
    
    init_spectral_gap = agg_score(init_eigenvals, agg_method_cc='min', 
                                  agg_method_temp='min', skip_zero=True)
    
    cache_filepath = f".cache/specgap_snap_{agg_method_temp}_cache_{COARS_SCORE_TAG}_{num_times}_{N}_{len(ss_check_pairs)}"\
                     f"_initsg{round(init_spectral_gap, 6)}_{SPEC_GAP_METHOD}.pkl"
    os.makedirs(os.path.dirname(cache_filepath), exist_ok=True)
    #
    def compute_link_score(pair):
        u, v = pair
        P = onp.zeros((N, K))
        idx = 0
        P[u, idx] = 1.0
        P[v, idx] = 1.0
        other_orignodes = onp.array([n for n in range(N) if n != u and n != v])
        other_supernodes = onp.arange(1, K)
        P[other_orignodes, other_supernodes] = 1.0
        #
        P = normalize_partition_matrix(P)
        #
        assert onp.all(onp.sum(P, axis=1) == 1.0), "Each row of P must sum to 1."
        
        # snaps_spectral_gaps, snaps_eigenvecs, As, Acs, Ls, cc_sizes, cc_to_nodes
        temporal_cc_sp_gaps, *_ = get_snapshots_spectral_gap(
            window_edge_srcs, window_edge_dsts, P, num_times, N, K
        )
        
        score = agg_score(temporal_cc_sp_gaps, agg_method_cc='min', 
                          agg_method_temp='min', skip_zero=True)
        return (u, v), score
    
    def estimate_link_score(pair, As, Ls, init_lambdas, init_eigenvecs, 
                            node_ccs, nodes_degs, cc_nodess, cc_sizess, 
                            agg_method_temp='max'):
        
        a, b = pair
        Va_indices = a
        Vb_indices = b
        temp_scores = []
        N = As[0].shape[0]
        for t in range(num_times):
            Lt = Ls[t]
            At = As[t]
            
            nodes_eigval = init_lambdas[t]
            nodes_eigvec = init_eigenvecs[t]
            nodes_cc = node_ccs[t]
            nodes_deg = nodes_degs[t]
            cc_nodes = cc_nodess[t]
            cc_sizes = cc_sizess[t]
            
            w_ab = At[a, b]
            
            delta_lambda, orig_lambda, cc_size = \
                compute_snapshot_eigen_perturbs(nodes_eigvec, nodes_eigval, 
                                                nodes_cc, nodes_deg, cc_nodes, cc_sizes,
                                                Va_indices, Vb_indices, w_ab)
            score = (delta_lambda / orig_lambda) * (cc_size / N) # relative change
            
            temp_scores.append(score)

        # Aggregate scores across snapshots: maximum change over snapshots
        # score = agg_score(temp_scores, agg_method_cc=None, agg_method_temp='max')
        # Aggregate scores across snapshots: sum of scores across time
        
        score = agg_score(temp_scores, agg_method_cc=None, 
                          agg_method_temp=agg_method_temp, skip_zero=True)
        if abs(score) > 1e5:
            print(f"WARNING: Large spectral score detected for pair ({a}, {b}): {score}")
            import pdb;pdb.set_trace()

        return (a, b), score
    
    if SPEC_GAP_METHOD == 'exact':
        if os.path.exists(cache_filepath):
            print(f"Loading spectral gaps from cache: {cache_filepath}")
            with open(cache_filepath, 'rb') as f:
                exact_spectral_gaps = pickle.load(f)
            if len(exact_spectral_gaps) == len(ss_check_pairs):
                return exact_spectral_gaps
        else:
            exact_spectral_gaps = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                results = list(tqdm(executor.map(compute_link_score, ss_check_pairs), total=len(ss_check_pairs)))
            for (edge, score) in results:
                exact_spectral_gaps[edge] = score
            with open(cache_filepath, 'wb') as f:
                pickle.dump(exact_spectral_gaps, f)
                print(f"Saved spectral gaps to cache: {cache_filepath}")
        
    elif SPEC_GAP_METHOD == 'approx':
        if os.path.exists(cache_filepath):
            print(f"Loading spectral gaps from cache: {cache_filepath}")
            with open(cache_filepath, 'rb') as f:
                approx_spectral_scores = pickle.load(f)
            if len(approx_spectral_scores) == len(ss_check_pairs):
                return approx_spectral_scores
        else:
            # FOR LOOP for debugging
            if len(ss_check_pairs) < 100_000:
                approx_spectral_scores = {}
                for pair in tqdm(ss_check_pairs, total=len(ss_check_pairs)):
                    edge, score = estimate_link_score(pair, As, Ls, init_eigenvals, init_eigenvecs,
                                                    nodes_ccs, nodes_degs, cc_nodess, cc_sizess, 
                                                    agg_method_temp=agg_method_temp)
                    approx_spectral_scores[edge] = score
            else:
                print("Using multithreading to compute approx spectral scores...")
                approx_spectral_scores = {}
                with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                    results = list(tqdm(executor.map(
                        lambda pair: estimate_link_score(pair, As, Ls, init_eigenvals, init_eigenvecs,
                                                        nodes_ccs, nodes_degs, cc_nodess, cc_sizess, 
                                                        agg_method_temp=agg_method_temp), 
                        ss_check_pairs), total=len(ss_check_pairs)))
                for (edge, score) in results:
                    approx_spectral_scores[edge] = score
                    
            with open(cache_filepath, 'wb') as f:
                pickle.dump(approx_spectral_scores, f)
                print(f"Saved spectral gaps to cache: {cache_filepath}")

    else:
        raise ValueError(f"Unknown SPEC_GAP method: {SPEC_GAP_METHOD}")

    if SPEC_GAP_METHOD == 'exact':
        return exact_spectral_gaps
    elif SPEC_GAP_METHOD == 'approx':
        return approx_spectral_scores


def compute_snapshot_eigen_perturbs_supra(nodes_eigvec, nodes_eigval, nodes_degs, 
                                          Va_indices, Vb_indices, w_abs):
    """
    Compute the quantity Delta lambda using vectorized operations.
    
    Parameters:
    - v: numpy array, the vector (eigenvector).
    - lambda_: float, the eigenvalue lambda.
    - Va_indices: list or numpy array, indices of nodes in V_a.
    - Vb_indices: list or numpy array, indices of nodes in V_b.
    - w: list or numpy array, weights w(a_t, b_t) for each t.
    
    Returns:
    - float, the computed Delta lambda.
    """
    # Convert indices and w to numpy arrays if not already
    Va_indices = onp.array(Va_indices)
    Vb_indices = onp.array(Vb_indices)
    _lambda_a = nodes_eigval[Va_indices.flatten()] # shape (T, )
    _lambda_b = nodes_eigval[Vb_indices.flatten()] # shape (T, )
    assert onp.all(_lambda_a == _lambda_b), "Eigenvalues for nodes in different snapshots should be the same."
    assert len(set(_lambda_a.tolist())) == 1, "Eigenvalues for nodes in different snapshots should be the same."
    _lambda_ab = _lambda_a[0]
    
    # Extract v_a and v_b
    v_a = nodes_eigvec[Va_indices]
    v_b = nodes_eigvec[Vb_indices]
    
    # Compute sums using vector operations
    va_sq = v_a ** 2 # shape (T, )
    vb_sq = v_b ** 2
    va_vb = v_a * v_b
    deg_a = nodes_degs[Va_indices]
    deg_b = nodes_degs[Vb_indices]
    
    # symmetric score
    # intra-term: \sum_{t=1}^{T} (v^{t}_a - v^{t}_b)^2 * (0.25 * (d^{t}_a + d^{t}_b + 2 * w^{t}_ab) - \lambda)
    intra_num = (va_sq - 2 * va_vb + vb_sq) \
                * (0.25 * (deg_a + deg_b + 2 * w_abs) - _lambda_ab)
    intra_num = onp.sum(intra_num) # sum over snapshots

    # inter-term: -0.5 * \omega  \sum_{t=1}^{T-1} [(v^{t+1}_a - v^{t+1}_b) - (v^{t}_a - v^{t}_b)]^2
    inter_num = (v_a[1:] - v_b[1:]) - (v_a[:-1] - v_b[:-1]) # shape (T-1, )
    inter_num = inter_num ** 2
    inter_num = -0.5 *onp.sum(inter_num) # sum over snapshots

    numerator = intra_num + inter_num
    # Denominator
    denominator = onp.dot(nodes_eigvec, nodes_eigvec)
    
    # Avoid division by zero
    if denominator == 0:
        print("Denominator in eigenvalue perturbation is zero. Setting perturbation to 0.")
        eigen_pert = 0.0
    else:
        eigen_pert = numerator / denominator

    return eigen_pert, _lambda_ab


def get_links_scores_over_coarsened_supra_dyngraph(
    num_times, N, K,
    window_edge_srcs: List[GENERIC_NPArray], 
    window_edge_dsts: List[GENERIC_NPArray], 
    init_info: Tuple[List],
    ss_check_pairs: List[Tuple[int, int]],
    SPEC_GAP_METHOD: str,
) -> dict:
    '''
    Compute the spectral gap for each pair of nodes in ss_check_pairs,
    when they are merged into a supernode, and all other nodes are mapped to their own
    supernodes.
    Return a dictionary mapping from (u, v) to spectral gap.
    '''
    init_eigenvals, init_eigenvecs, supA, supL = init_info

    init_spectral_gap = agg_score(init_eigenvals, agg_method_cc='min', 
                                  agg_method_temp=None, skip_zero=True)
    
    cache_filepath = f".cache/specgap_supra_cache_{COARS_SCORE_TAG}_{num_times}_{N}_{len(ss_check_pairs)}"\
                     f"_initsg{round(init_spectral_gap, 6)}_{SPEC_GAP_METHOD}.pkl"
    os.makedirs(os.path.dirname(cache_filepath), exist_ok=True) 
    #
    def compute_link_score(pair):
        u, v = pair
        P = onp.zeros((N, K))
        idx = 0
        P[u, idx] = 1.0
        P[v, idx] = 1.0
        other_orignodes = onp.array([n for n in range(N) if n != u and n != v])
        other_supernodes = onp.arange(1, K)
        P[other_orignodes, other_supernodes] = 1.0
        #
        P = normalize_partition_matrix(P)
        #
        assert onp.all(onp.sum(P, axis=1) == 1.0), "Each row of P must sum to 1."
        
        # snaps_spectral_gaps, snaps_eigenvecs, As, Acs, Ls, cc_sizes, cc_to_nodes
        cc_sp_gaps, *_ = get_supra_spectral_gap(
            window_edge_srcs, window_edge_dsts, P, num_times, N, K
        )
        
        score = agg_score(cc_sp_gaps, agg_method_cc='min', 
                          agg_method_temp=None, skip_zero=True)
        return (u, v), score
    
    def estimate_link_score(N, num_times, pair, supA, supL, init_lambdas, init_eigenvecs):
        a, b = pair
        Va_indices = onp.zeros(num_times, dtype=int)
        Vb_indices = onp.zeros(num_times, dtype=int)
        node_degs = onp.array(supA.sum(axis=1)).flatten()
        w_abs = onp.zeros(num_times)  # store w_ab for each snapshot
        acc_idx = 0
        for t in range(num_times):
            a_idx = acc_idx + a
            b_idx = acc_idx + b
            Va_indices[t] = a_idx
            Vb_indices[t] = b_idx
            w_abs[t] = supA[a_idx, b_idx]  # weight between a and b in snapshot t
            acc_idx += N

        nodes_eigval = init_lambdas
        nodes_eigvec = init_eigenvecs

        delta_lambda, orig_lambda = compute_snapshot_eigen_perturbs_supra(
            nodes_eigvec, nodes_eigval, node_degs,
            Va_indices, Vb_indices, w_abs
        )

        score = delta_lambda / orig_lambda

        return (a, b), score, delta_lambda, orig_lambda
    
    if SPEC_GAP_METHOD == 'exact':
        if os.path.exists(cache_filepath):
            print(f"Loading spectral gaps from cache: {cache_filepath}")
            with open(cache_filepath, 'rb') as f:
                exact_spectral_gaps = pickle.load(f)
            if len(exact_spectral_gaps) == len(ss_check_pairs):
                return exact_spectral_gaps
        else:
            exact_spectral_gaps = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                results = list(tqdm(executor.map(compute_link_score, ss_check_pairs), total=len(ss_check_pairs)))
            for (edge, score) in results:
                exact_spectral_gaps[edge] = score
            with open(cache_filepath, 'wb') as f:
                pickle.dump(exact_spectral_gaps, f)
                print(f"Saved spectral gaps to cache: {cache_filepath}")
        
    elif SPEC_GAP_METHOD == 'approx':
        if os.path.exists(cache_filepath):
            print(f"Loading spectral gaps from cache: {cache_filepath}")
            with open(cache_filepath, 'rb') as f:
                approx_spectral_scores = pickle.load(f)
            if len(approx_spectral_scores) == len(ss_check_pairs):
                return approx_spectral_scores
        else:
            # FOR LOOP for debugging
            if len(ss_check_pairs) < 100_000:
                approx_spectral_scores = {}
                
                for pair in tqdm(ss_check_pairs, total=len(ss_check_pairs)):
                    edge, score, _, _ = estimate_link_score(N, num_times, pair, supA, supL, 
                                                    init_eigenvals, init_eigenvecs)
                    approx_spectral_scores[edge] = score
            else:
                print("Using multithreading to compute approx spectral scores...")
                approx_spectral_scores = {}
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    results = list(tqdm(executor.map(
                        lambda pair: estimate_link_score(N, num_times, pair, supA, supL, 
                                                        init_eigenvals, init_eigenvecs), 
                        ss_check_pairs), total=len(ss_check_pairs)))
                for (edge, score, _, _) in results:
                    approx_spectral_scores[edge] = score

            with open(cache_filepath, 'wb') as f:
                pickle.dump(approx_spectral_scores, f)
                print(f"Saved spectral gaps to cache: {cache_filepath}")

    else:
        raise ValueError(f"Unknown SPEC_GAP method: {SPEC_GAP_METHOD}")
    
    if SPEC_GAP_METHOD == 'exact':
        return exact_spectral_gaps
    elif SPEC_GAP_METHOD == 'approx':
        return approx_spectral_scores
