import argparse

def add_coarsening_arguments(parser: argparse.ArgumentParser) -> None:
    R"""
    Add other problem-specific arguments, including coarsening method.
    """
    parser.add_argument(
        "--method", type=str, help="Coarsening method (e.g., variation_edges, heavy_edge). NOTE: this is the same as coars_alg, keep it for backward compatibility."
    )
    parser.add_argument(
        "--coars_alg", type=str, required=True, help="Coarsening method (e.g., TOPKSupra, TOPKSnap)."
    )
    parser.add_argument(
        "--spec_gap_method", type=str, default='approx', help="Computing/estimating spectral gap (e.g., approx, exact)."
    )
    parser.add_argument(
        "--coars_op", type=str, default="PtAP", help="Coarsening operator (e.g., 'QAP' or 'PtAP')."
    )
    parser.add_argument(
        "--rw_sl", action="store_true", default=False, help="reweight self-loops of super-nodes"
    )
    parser.add_argument(
        "--ef_rw_method", type=str, default='avg', help="Coarsened edge feature reweight method 'avg', 'cardinality_s', 'cardinality_d', 'cardinality_sd'"
    )
    parser.add_argument(
        "--others", type=str, default="cc", help="Notes for other coarsening method specific arguments (e.g., 'cc' for connected components)."
    )
    #
    parser.add_argument(
        "--n_prime", type=float, required=True, help="1 - coarsening ratio."
    )
    parser.add_argument(
        "--msfr", dest="metaset_force_reload", 
            action="store_true", 
            default=False,
            help="Force reload metaset."
    )
    parser.add_argument(
        "--metaset_ss_num", type=int, default=-1,
        help="Number of subsampling for metaset. -1 means no subsampling."
    )
    
    parser.add_argument(
        "--pre_run_results_path", type=str, default=None,
        help="Path to pre-run results containing coarsening matrix from baselines"
    )
    parser.add_argument(
        "--refine_every", type=float, default=0.1,
        help="Recompute the spectral gap after every this proportion of number of nodes"
    )
    

def get_coarsen_str(args) -> str:
    COARS_ALG = args.coars_alg
    SPEC_GAP = args.spec_gap_method
    COARS_OP = args.coars_op
    RW_SL = args.rw_sl
    EF_RW_METHOD = args.ef_rw_method
    OTHERS = args.others
    REFINE_EVERY = args.refine_every
    coarsen_str = f"_{COARS_ALG}_{SPEC_GAP}_{COARS_OP}_{RW_SL}_{EF_RW_METHOD}_{OTHERS}_{REFINE_EVERY}"
    return coarsen_str