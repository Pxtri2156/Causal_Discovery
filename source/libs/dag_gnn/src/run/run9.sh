#SPORTS_real
CUDA_VISIBLE_DEVICES=3 python train.py --data_type='discrete_benchmark' \
--data_filename='SPORTS_real.CSV' \
--data_dir="/dataset/Bayesian_Data/SPORTS" \
--gt_path="/dataset/Bayesian_Data/SPORTS/DAGtrue_SPORTS_bi.csv"
