#PROPERTY 
CUDA_VISIBLE_DEVICES=2 python train.py --data_type='discrete_benchmark' \
--data_filename='PROPERTY_DATA.csv' \
--data_dir="/dataset/Bayesian_Data/PROPERTY" \
--gt_path="/dataset/Bayesian_Data/PROPERTY/DAGtrue_PROPERTY_bi.csv"

