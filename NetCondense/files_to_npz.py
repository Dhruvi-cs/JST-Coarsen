import numpy as np
import argparse
import os
from pathlib import Path

def convert_supernodes_to_python(supernodes_path, output_npz_path, n_nodes, n_prime):
    # Reads _superNodes file
    # Format: superNodeID:\t node1, node2, ...
    # Creates P matrix (N, K)
    # Output NPZ with keys 'P' and 'Nprime'=K
    
    # create parent directory if it doesn't exist
    output_dir = Path(output_npz_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        supernodes_map = {}
        with open(supernodes_path, 'r') as f:
            for line in f:
                parts = line.strip().split(':')
                if len(parts) < 2: continue
                sid = int(parts[0])
                nodes_str = parts[1].strip()
                if not nodes_str: continue
                # Split by comma or tab? Code says "write(str(i)+',\t')"
                # So split by ','
                nodes = [int(x) for x in nodes_str.split(',') if x.strip()]
                supernodes_map[sid] = nodes
        
        K = len(supernodes_map)
        P = np.zeros((n_nodes, K), dtype=np.float32)
        
        # Supernodes likely 1-indexed IDs, but positions need mapping
        # Let's just enumerate them 0 to K-1
        
        for k_idx, (sid, nodes) in enumerate(supernodes_map.items()):
            for node in nodes:
                # node is 1-based
                if 1 <= node <= n_nodes:
                    P[node-1, k_idx] = 1.0
                    
        # Verify partition property
        # Check unassigned nodes?
        
        # np.savez(output_npz_path, P=P, Nprime=n_prime)
        np.savez(output_npz_path, P=P, Nprime=K)
        print(f"Saved partition matrix to {output_npz_path}")
        
    except Exception as e:
        print(f"Error converting supernodes: {e}")

def get_n_from_edgelist(edgelist_path):
    n = 0
    try:
        with open(edgelist_path, 'r') as f:
            for line in f:
                parts = line.split()
                if parts:
                    # parts[1] and parts[2] are nodes
                    n = max(n, int(parts[1]), int(parts[2]))
    except Exception as e:
        print(f"Error reading edgelist: {e}")
    return n

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--supernodes_file', type=str, required=True)
    parser.add_argument('--edgelist_file', type=str, required=True, help="To determine N")
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--n_prime', type=float, required=True)
    args = parser.parse_args()
    
    n = get_n_from_edgelist(args.edgelist_file)
    if n == 0:
        print("Could not determine number of nodes from edgelist.")
        return

    convert_supernodes_to_python(args.supernodes_file, args.output_file, n, args.n_prime)

if __name__ == "__main__":
    main()
