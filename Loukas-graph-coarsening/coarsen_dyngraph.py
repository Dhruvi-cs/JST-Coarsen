import json
import numpy as np
import numpy.typing as onpt
import torch
from scipy import sparse
import pygsp as gsp
import os
from typing import List, Callable, Tuple, cast, Optional, Union
import pandas as pd
import networkx as nx

from graph_coarsening.coarsening_utils import (
    coarsen as coarsen_lib, 
    coarsening_quality,
)

SMALL_NUMBER = 1e-8

def create_union_graph(
    edge_srcs: List[onpt.NDArray[np.generic]], 
    edge_dsts: List[onpt.NDArray[np.generic]],
    edge_weight: Optional[onpt.NDArray[np.generic]],
    N: int,
    symmetrize: bool = False, 
    remove_selfloops: bool = True,
    ) -> Tuple[gsp.graphs.Graph, dict, int]:
    
    num_times = len(edge_srcs)
    assert num_times == len(edge_dsts)
    #
    all_srcs = np.concatenate(edge_srcs, axis=0)
    all_dsts = np.concatenate(edge_dsts, axis=0)
    if edge_weight is not None:
        all_weights = np.concatenate(edge_weight, axis=0)
    else:
        all_weights = np.ones((all_srcs.shape[0],), dtype=np.float32)
    # create union graph, with edge weights being the sum of edge weights over time
    all_adj = sparse.coo_matrix((all_weights, (all_srcs, all_dsts)), shape=(N, N))
    
    if remove_selfloops:
        all_adj.setdiag(0)  # remove self-loops
    
    all_adj.sum_duplicates()
    
    if symmetrize:
        assert np.allclose(all_adj.toarray(), all_adj.T.toarray()) == False, \
        "Graph is already symmetric."
        all_adj = all_adj + all_adj.T
    
    all_adj.eliminate_zeros()
    # create pygsp graph
    G = gsp.graphs.Graph(all_adj)
    G.compute_fourier_basis() # just for silencing warning
    return G



def DynamicClassification_from_raw(filepath: str) -> None:
    R"""
    Load from raw data.
    from `main_src/lib/data/dynclass.py`
    """
    #
    matrices = np.load(filepath)
    feats = matrices["attmats"]
    adjmats = matrices["adjs"]
    onehots = matrices["labels"]
    # Safety check.
    if np.any(adjmats < 0):
        # UNEXPECT:
        # Edge weights must be non-negative.
        raise NotImplementedError("Get invalid negative edge weights.")
    if np.any(np.sum(onehots > 0, axis=1) > 1):
        # UNEXPECT:
        # Node label must be unique.
        raise NotImplementedError("Get empty or duplicate node labels.")

    #
    num_nodes = feats.shape[0]
    #
    raw_edge_srcs = []
    raw_edge_dsts = []
    raw_edge_feats = []
    num_times = adjmats.shape[0]

    for t in range(num_times):
        #
        (dsts, srcs) = np.nonzero(adjmats[t])
        weights = adjmats[t, dsts, srcs].astype(float)
        if np.any(weights != 1):
            # UNEXPECT:
            # Node label must be unique.
            raise NotImplementedError("Edge has non-0/1 weight.")
        raw_edge_srcs.append(srcs)
        raw_edge_dsts.append(dsts)
        raw_edge_feats.append(weights)
        
    #
    edge_feats = []
    edge_labels = []
    for t in range(num_times):
        #
        num_edges = len(raw_edge_feats[t])
        edge_feats.append(np.reshape(raw_edge_feats[t], (num_edges, 1)))
        edge_labels.append(
            cast(
                onpt.NDArray[np.generic],
                np.zeros((num_edges, 1)).astype(np.int64),
            ),
        )
    
    return (raw_edge_srcs, raw_edge_dsts, edge_feats, num_nodes)

def run_coarsening_experiment(
    method: str, ratio: float, coarsen_ratio: float, 
    K_subspace: int,
    G: gsp.graphs.Graph,
    N: int,
    kmax_eval: int,
    output_dir: str 
    ) -> None:
    ''' 
    Run coarsening experiment for (method, ratio, K_subspace) on graph G.
    Save coarsening matrix C, prolongation matrix P, coarsened graph Gc, 
    union graph G, and coarsening quality metrics to output_dir.
    '''
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Running: {method}")
    print(f"  reduction ratio={ratio} (coarsen ratio={coarsen_ratio}), K_subspace={K_subspace}")

    C_sparse, Gc, Call, Gall = coarsen_lib(G, K=K_subspace, r=ratio, method=method, algorithm='greedy')
    metrics = coarsening_quality(G, C_sparse, kmax=kmax_eval)
    
    C = C_sparse.toarray()
    Nprime = Gc.N
    
    P = np.zeros((N, Nprime))
    for i in range(Nprime):
        cluster_fines = np.where(C[i] > SMALL_NUMBER)[0]
        P[cluster_fines, i] = 1.0
    
    savepath = os.path.join(output_dir, method, 
                    f"coarsened~{method}~k{K_subspace}~cr{coarsen_ratio}.npz")
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    # follow `main_src/lib/meta/dyngraph/sparse/dynedge_v2.py:coarsen_window_dyngraph_baseline()``
    np.savez_compressed(
        savepath,
        method=method,
        rr=ratio, # reduction ratio
        cr=coarsen_ratio, # coarsen ratio
        k=K_subspace,
        C=C,
        P=P,
        Nprime=Nprime,
        coarsen_graph=Gc,
        union_graph=G, 
        metrics=metrics,
    )
    print(f"Coarsening matrix saved to {savepath}")
    return metrics

import matplotlib.pyplot as plt
def visualize_metrics(N, k, model_metrics: dict, model_id: str, axes: Optional[plt.Axes] = None) -> None:
    # visualize metrics based on above code snippet
    if axes is None:
        size = 2.04
        fig, axes = plt.subplots(1, 2, figsize=(4*size*3, 3*size))
    
    n = model_metrics['n']
    kmax = len(model_metrics['error_subspace'])
    axes[0].plot(np.arange(kmax), np.abs(model_metrics['error_subspace']), 'o-', label=model_id, alpha=0.6)
    axes[0].set_xlabel('$k$'); axes[0].set_ylabel('$\epsilon$')
    axes[0].plot([k, k], [0, max(model_metrics['error_subspace'])], ':k')
    axes[0].set_title('Subspace Error')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].legend()
    
    # prepare incremental x-positioning and color for multiple boxplots on the same axes
    if not hasattr(axes[1], "_box_x"):
        axes[1]._box_x = 1
    # choose color from the axes color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0'])
    color = prop_cycle[(axes[1]._box_x - 1) % len(prop_cycle)]
    # draw boxplot at current position
    bp = axes[1].boxplot(np.abs(model_metrics['error_eigenvalue']),
                         positions=[axes[1]._box_x],
                         widths=0.6,
                         patch_artist=True,
                         boxprops=dict(facecolor=color, color=color),
                         medianprops=dict(color='white'),
                         whiskerprops=dict(color=color),
                         capprops=dict(color=color),
                         flierprops=dict(markeredgecolor=color))
    # annotate the mean value in the boxplot
    mean_val = np.mean(np.abs(model_metrics['error_eigenvalue']))
    std_val = np.std(np.abs(model_metrics['error_eigenvalue']))
    axes[1].plot(axes[1]._box_x, mean_val, marker='D', 
                 color='black', markersize=8, label='Mean')
    axes[1].text(axes[1]._box_x + 0.1, mean_val, 
                 f"{mean_val:.4f}±{std_val:.4f}",
                 va='center', fontsize=8, rotation=90)
    # create a proxy handle for the legend (one per model)
    axes[1].set_ylabel('relative eigenvalue error')
    axes[1].set_title('Eigenvalue Error')
    axes[1]._box_x += 1  # increment x-position for next boxplot
    
    return axes

def coarsen_and_compute_metrics(
    method: str, k: int,
    /, 
    reduction_ratio: float, coarsen_ratio: float, 
    G_union: gsp.graphs.Graph,
    N: int,
    kmax_eval: int,
    output_dir_cr: str,
    all_metrics: dict,
    ) -> dict:
    ''' 
    Run coarsening experiment for (method, ratio, K_subspace) on graph G.
    Save coarsening matrix C, prolongation matrix P, coarsened graph Gc, 
    union graph G, and coarsening quality metrics to output_dir.
    '''
    model_id = f"{method}~k{k}"
    
    if model_id not in all_metrics:
        print(f"Processing coarsening method: {model_id}, "
                f"reduction ratio={reduction_ratio}, coarsen ratio={coarsen_ratio}")
        metrics = run_coarsening_experiment(
            method, reduction_ratio, coarsen_ratio,
            k, G_union, N, kmax_eval, 
            output_dir_cr
        )
    else:
        print(f"Using existing metrics for coarsening method: {model_id}, "
                f"reduction ratio={reduction_ratio}, coarsen ratio={coarsen_ratio}")
        metrics = all_metrics[model_id]
        
    return model_id, metrics

import argparse
import wandb 
from consts import WDB_PROJECT, WDB_MODE

DSET_CONFIG = {
    'DBLP5': {
        "K_SUBSPACES": [10, 20, 40, 50, 80, 100, 120],
        "kmax_eval": 150,
    },
    'IMDB': {
        "K_SUBSPACES": [40, 80, 100, 150, 200],
        "kmax_eval": 200,
    },
    'Reddit4': {
        "K_SUBSPACES": [10, 20, 40, 50, 80],
        "kmax_eval": 100,
    },
    'DBLP5_ss200': {
        "K_SUBSPACES": [10, 20, 40, 50],
        "kmax_eval": 100,
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Graph Coarsening Experiments on Dynamic Graph Classification Datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Reddit4",
        help="Dataset name (default: Reddit4)",
    )
    args = parser.parse_args()
    dataset = args.dataset

    #
    K_SUBSPACES = DSET_CONFIG[dataset]["K_SUBSPACES"]
    RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # reduction ratios
    CRATIOS = [round(1 - r, 1) for r in RATIOS] # how much left after coarsening

    COARSEN_ALGS = ['variation_edges', 'variation_neighborhood', 
                    'heavy_edge', 'heavy_edge_degree', 
                    'algebraic_JC']
    COARSEN_ALGS = ['variation_edges', 'variation_neighborhood', 'heavy_edge_degree']
    
    SUBSPACE_INSENSITIVE_METHODS = [
        'heavy_edge', 'heavy_edge_degree'
    ]
    
    # Load raw data
    output_dir = f'results/{dataset}/'
    os.makedirs(output_dir, exist_ok=True)
    
    if '_ss' in dataset:
        ssstr = '_' + dataset.split('_')[1]
        orig_dataset = dataset.split('_')[0]
    else:
        ssstr = ''
        orig_dataset = dataset
    #
    npz_filepath = f'../main_src/src/{orig_dataset}/{orig_dataset}_no_isolated{ssstr}.npz'
    raw_edge_srcs, raw_edge_dsts, raw_edge_feats, N = DynamicClassification_from_raw(npz_filepath)
    symmetrize = True
    
    kmax_eval = DSET_CONFIG[dataset]["kmax_eval"] # max number of eigenvalues to evaluate the relative eigenvalue error
    print(f"Loaded raw data from {npz_filepath}: N={N}, num_times={len(raw_edge_srcs)}")
    #
    if raw_edge_feats[0].ndim == 1:
        edge_weights = raw_edge_feats
    else:
        edge_weights = [np.ones((len(raw_edge_feats[t]),), 
                                 dtype=raw_edge_feats[t].dtype) 
                        for t in range(len(raw_edge_feats))]
    
    G_union = create_union_graph(raw_edge_srcs, raw_edge_dsts, edge_weights, N, 
                        symmetrize=symmetrize, remove_selfloops=True)
    #
    print(f"Union graph: |V|={G_union.n_vertices}, |E|={G_union.n_edges} ")
    # get number of connected components
    # components = list(nx.connected_components(G_union.to_networkx()))
    components = G_union.extract_components()
    cc_sizes = [comp.n_vertices for comp in components]
    print(f"  Number of connected components: {len(components)}")
    print(f"  Sizes of connected components: {cc_sizes}")
    #   
    desc = f"{dataset}"
    config = {
        "dataset": dataset,
        "npz_filepath": npz_filepath,
        "output_dir": output_dir,
        "num_nodes": N,
        "num_times": len(raw_edge_srcs),
        "num_edges_per_time": [len(raw_edge_srcs[t]) for t in range(len(raw_edge_srcs))],
        "coarsen_algs": COARSEN_ALGS,
        "reduction_ratios": RATIOS,
        "k_subspaces": K_SUBSPACES,
        "kmax_eval": kmax_eval,
        "G_union": {
            "num_nodes": G_union.n_vertices,
            "num_edges": G_union.n_edges,
            "num_connected_components": len(components),
            "connected_components_sizes": cc_sizes,
        }
    }
    # initialize wandb
    wdbrun = wandb.init(
        project=WDB_PROJECT,
        name=desc,
        mode=WDB_MODE,
        config=config
    )
    #
    size=4    
    PARALLEL_RUN = True
    
    for (reduction_ratio, coarsen_ratio) in zip(RATIOS, CRATIOS):
        output_dir_cr = os.path.join(output_dir, f"cr{coarsen_ratio}")
        os.makedirs(output_dir_cr, exist_ok=True)
        #
        fig, axes = plt.subplots(1, 2, figsize=(4*size*2, 3*size))
        fig_savepath = os.path.join(output_dir_cr, f"coarsening_quality_metrics.png")
        all_metrics_savepath = os.path.join(output_dir_cr, f"all_coarsening_metrics.json")
        best_models_savepath = os.path.join(output_dir_cr, f"best_models_per_method.csv")
        all_eigenerrors_savepath = os.path.join(output_dir_cr, f"all_models_eigenerrors.csv")
        
        #
        all_metrics = {}
        
        if os.path.exists(all_metrics_savepath):
            with open(all_metrics_savepath, 'r') as f:
                all_metrics = json.load(f)
            print(f"Loaded existing coarsening metrics from {all_metrics_savepath}")
            for key in all_metrics:
                for mkey in all_metrics[key]:
                    if isinstance(all_metrics[key][mkey], list):
                        all_metrics[key][mkey] = np.array(all_metrics[key][mkey])
        #
        model_ids = []
        best_models_id = {}
        all_models_eigenerrors = {}
        if PARALLEL_RUN:
            # use multiprocessing to run coarsening experiments in parallel
            from multiprocessing import Pool
            from functools import partial
            coarsen_func = partial(
                coarsen_and_compute_metrics,
                reduction_ratio=reduction_ratio,
                coarsen_ratio=coarsen_ratio,
                G_union=G_union,
                N=N,
                kmax_eval=kmax_eval,
                output_dir_cr=output_dir_cr,
                all_metrics=all_metrics,
            )
            args_list = []
            for method in COARSEN_ALGS:
                for k in K_SUBSPACES:
                    args_list.append( (method, k) )
                    if method in SUBSPACE_INSENSITIVE_METHODS:
                        print(f"  Method {method} is insensitive to k, skipping other k values.")
                        break
                    
            with Pool(10) as pool:
                results = pool.starmap(coarsen_func, args_list)
                
            for model_id, metrics in results:
                model_ids.append(model_id)
                all_metrics[model_id] = metrics
                # visualize metrics
                axes = visualize_metrics(G_union.n_vertices, k, model_metrics=metrics, 
                            model_id=model_id, axes=axes)
                # track best model
                avg_eigen_err = np.mean(np.abs(metrics['error_eigenvalue']))
                std_eigen_err = np.std(np.abs(metrics['error_eigenvalue']))
                all_models_eigenerrors[model_id] = (avg_eigen_err, std_eigen_err)
                
                method = model_id.split('~')[0]
                if method not in best_models_id:
                    best_models_id[method] = (model_id, avg_eigen_err)
                else:
                    if avg_eigen_err < best_models_id[method][1]:
                        best_models_id[method] = (model_id, avg_eigen_err)
        else:
            for method in COARSEN_ALGS:
                for k in K_SUBSPACES:
                    model_id, metrics = coarsen_and_compute_metrics(
                        method, k,
                        reduction_ratio=reduction_ratio,
                        coarsen_ratio=coarsen_ratio,
                        G_union=G_union,
                        N=N,
                        kmax_eval=kmax_eval,
                        output_dir_cr=output_dir_cr,
                        all_metrics=all_metrics,
                    )
                    
                    model_ids.append(model_id)
                    #
                    all_metrics[model_id] = metrics
                    # visualize metrics
                    axes = visualize_metrics(G_union.n_vertices, k, model_metrics=metrics, 
                                model_id=model_id, axes=axes)
                    # track best model
                    avg_eigen_err = np.mean(np.abs(metrics['error_eigenvalue']))
                    std_eigen_err = np.std(np.abs(metrics['error_eigenvalue']))
                    all_models_eigenerrors[model_id] = (avg_eigen_err, std_eigen_err)
                    
                    if method not in best_models_id:
                        best_models_id[method] = (model_id, avg_eigen_err)
                    else:
                        if avg_eigen_err < best_models_id[method][1]:
                            best_models_id[method] = (model_id, avg_eigen_err)
                    # skip other k if method is insensitive to k
                    if method in SUBSPACE_INSENSITIVE_METHODS:
                        print(f"  Method {method} is insensitive to k, skipping other k values.")
                        break
                    
        print("Best models per method (based on average eigenvalue error):")
        for method in best_models_id:
            print(f"  Method: {method}, Best Model: {best_models_id[method][0]}, "
                  f"Avg. Eigenvalue Error: {best_models_id[method][1]:.6f}")
        # save best_models_id to csv file 
        df = pd.DataFrame.from_dict(best_models_id, orient='index', columns=['Best Model ID', 'Avg Eigenvalue Error'])
        df.to_csv(best_models_savepath)
        print(f"Best models per method saved to {best_models_savepath}")
        # save all_models_eigenerrors to csv file
        df_all = pd.DataFrame.from_dict(all_models_eigenerrors, orient='index', columns=['Avg Eigenvalue Error', 'Std Eigenvalue Error'])
        df_all.to_csv(all_eigenerrors_savepath)
        print(f"All models eigenvalue errors saved to {all_eigenerrors_savepath}")
        
        # save figure
        axes[1].set_xticks(np.arange(1, len(model_ids)+1))
        axes[1].set_xticklabels(model_ids, rotation=45, ha='right')
        plt.suptitle(f"Coarsening Quality Metrics on {dataset} Union Graph")
        plt.tight_layout()
        plt.savefig(fig_savepath)
        print(f"Coarsening quality metrics figure saved to {fig_savepath}")
        # save all metrics
        with open(all_metrics_savepath, 'w') as f:
            for key in all_metrics:
                for mkey in all_metrics[key]:
                    if isinstance(all_metrics[key][mkey], np.ndarray):
                        all_metrics[key][mkey] = all_metrics[key][mkey].tolist()
            json.dump(all_metrics, f, indent=4)
        print(f"All coarsening metrics saved to {all_metrics_savepath}")