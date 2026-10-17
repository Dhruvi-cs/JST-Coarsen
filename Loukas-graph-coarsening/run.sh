# !/bin/bash
eval "$(conda shell.bash hook)"
conda activate sacod

source="Reddit4"
echo "Coarsening dataset: $source"

taskset --cpu-list 10-30 python coarsen_dyngraph.py --dataset $source &&
python post_process.py --dataset $source && 
echo "Done"