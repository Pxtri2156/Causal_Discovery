#PROPERTY 
DATA_PATH="/dataset/Bayesian_Data/PROPERTY/PROPERTY_DATA.csv"
GT_PATH="/dataset/Bayesian_Data/PROPERTY/DAGtrue_PROPERTY_bi.csv"
RESULT_PATH="/workspace/tripx/projects/causality/notears/run/Bayes/W_est_property.csv"

python notears/linear_custom.py --dataset="bayes" \
--data_path=$DATA_PATH \
--gt_path=$GT_PATH \
--result_path=$RESULT_PATH \