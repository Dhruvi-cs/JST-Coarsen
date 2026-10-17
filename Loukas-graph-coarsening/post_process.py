# load "results/Reddit4/cr0.1/best_models_per_method.csv"
# ,Best Model ID,Avg Eigenvalue Error
# variation_edges,variation_edges~k10,4.534136682184124
# variation_neighborhood,variation_neighborhood~k50,2.174904173810101
# heavy_edge,heavy_edge~k10,5.741009009440848
# heavy_edge_degree,heavy_edge_degree~k10,0.8100861692882896
# algebraic_JC,algebraic_JC~k10,15.570181339834829
# - get k values from model IDs
# copy "results/Reddit4/cr{coarsen_ratio}/{method}/coarsened~{method}~k{k}~cr{coarsen_ratio}.npz"
# to "results/Reddit4/cr{coarsen_ratio}/coarsened~{method}~cr{coarsen_ratio}.npz"
import os
import pandas as pd
import shutil
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process coarsened graph files based on best models.")
    parser.add_argument('--dataset', type=str, default="Reddit4", help='Directory containing results.')
    args = parser.parse_args()
    
    dataset = args.dataset
    
    COARSEN_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # coarsen ratios, how much left after coarsening

    # COARSEN_RATIOS = [0.879518] # coarsen ratios, how much left after coarsening
    
    for coarsen_ratio in COARSEN_RATIOS:
        results_dir = f"results/{dataset}/cr{coarsen_ratio}"
        best_models_path = os.path.join(results_dir, "best_models_per_method.csv")
        best_models_df = pd.read_csv(best_models_path)
        for _, row in best_models_df.iterrows():
            method = row['Best Model ID'].split('~')[0]
            k = row['Best Model ID'].split('~')[1][1:]  # extract k value
            src_path = os.path.join(results_dir, method, f"coarsened~{method}~k{k}~cr{coarsen_ratio}.npz")
            deldst_path = os.path.join(results_dir, f"coarsened~{method}~cr{coarsen_ratio}.npz")
            if os.path.exists(deldst_path):
                os.remove(deldst_path)
                print(f"Removed existing file {deldst_path}")
                
            dst_path = os.path.join(results_dir, f"coarsened~{method}.npz")
            shutil.copyfile(src_path, dst_path)
            print(f"Copied {src_path} to {dst_path}")
        print("Post-processing completed.")