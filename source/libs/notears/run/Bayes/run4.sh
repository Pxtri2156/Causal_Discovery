#FORMED
DATA_PATH="/dataset/Bayesian_Data/FORMED/FORMED_DATA.csv"
GT_PATH="/dataset/Bayesian_Data/FORMED/DAGtrue_FORMED_bi.csv"
RESULT_PATH="/workspace/tripx/projects/causality/notears/run/Bayes/W_est_formed.csv"

python notears/linear_custom.py --dataset="bayes" \
--data_path=$DATA_PATH \
--gt_path=$GT_PATH \
--result_path=$RESULT_PATH \