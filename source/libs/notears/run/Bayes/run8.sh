#SPORTS_DATA 
DATA_PATH="/dataset/Bayesian_Data/SPORTS/SPORTS_DATA.CSV"
GT_PATH="/dataset/Bayesian_Data/SPORTS/DAGtrue_SPORTS_bi.csv"
RESULT_PATH="/workspace/tripx/projects/causality/notears/run/Bayes/W_est_sports_data.csv"

python notears/linear_custom.py --dataset="bayes" \
--data_path=$DATA_PATH \
--gt_path=$GT_PATH \
--result_path=$RESULT_PATH \