#PATHFINDER
DATA_PATH="/dataset/Bayesian_Data/PATHFINDER/PATHFINDER_DATA.csv"
GT_PATH="/dataset/Bayesian_Data/PATHFINDER/DAGtrue_PATHFINDER_bi.csv"
RESULT_PATH="/workspace/tripx/projects/causality/notears/run/Bayes/W_est_pathfinder.csv"

python notears/linear_custom.py --dataset="bayes" \
--data_path=$DATA_PATH \
--gt_path=$GT_PATH \
--result_path=$RESULT_PATH \