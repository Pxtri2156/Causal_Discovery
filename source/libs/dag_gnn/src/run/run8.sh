#SPORTS_DATA 
CUDA_VISIBLE_DEVICES=2 python train.py --data_type='discrete_benchmark' \
--data_filename='SPORTS_DATA.CSV' \
--data_dir="/dataset/Bayesian_Data/SPORTS" \
--gt_path="/dataset/Bayesian_Data/SPORTS/DAGtrue_SPORTS_bi.csv"

