#FORMED
CUDA_VISIBLE_DEVICES=1 python train.py --data_type='discrete_benchmark' \
--data_filename='FORMED_DATA.csv' \
--data_dir="/dataset/Bayesian_Data/FORMED" \
--gt_path="/dataset/Bayesian_Data/FORMED/DAGtrue_FORMED_bi.csv"
