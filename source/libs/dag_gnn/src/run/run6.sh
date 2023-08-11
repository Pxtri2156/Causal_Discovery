#PATHFINDER
CUDA_VISIBLE_DEVICES=1  python train.py --data_type='discrete_benchmark' \
--data_filename='PATHFINDER_DATA.csv' \
--data_dir="/dataset/Bayesian_Data/PATHFINDER" \
--gt_path="/dataset/Bayesian_Data/PATHFINDER/DAGtrue_PATHFINDER_bi.csv"

