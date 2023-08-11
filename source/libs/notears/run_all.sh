#ALARM
DATA_PATH="/dataset/Bayesian_Data/ALARM/ALARM_DATA.csv"
GT_PATH="/dataset/Bayesian_Data/ALARM/DAGtrue_ALARM_bi.csv"
python notears/linear_custom.py --dataset="bayes" \
--data_path=$DATA_PATH \
--gt_path=$GT_PATH \

#ASIA
DATA_PATH="/dataset/Bayesian_Data/ASIA/ASIA_DATA.csv"
GT_PATH="/dataset/Bayesian_Data/ASIA/DAGtrue_ASIA_bi.csv"
python notears/linear_custom.py --dataset="bayes" \
--data_path=$DATA_PATH \
--gt_path=$GT_PATH \

#DIARRHOEA
DATA_PATH="/dataset/Bayesian_Data/DIARRHOEA/DIARRHOEA_DATA.csv"
GT_PATH="/dataset/Bayesian_Data/DIARRHOEA/DAGtrue_DIARRHOEA_bi.csv"
python notears/linear_custom.py --dataset="bayes" \
--data_path=$DATA_PATH \
--gt_path=$GT_PATH \

#FORMED
DATA_PATH="/dataset/Bayesian_Data/FORMED/FORMED_DATA.csv"
GT_PATH="/dataset/Bayesian_Data/FORMED/DAGtrue_FORMED_bi.csv"
python notears/linear_custom.py --dataset="bayes" \
--data_path=$DATA_PATH \
--gt_path=$GT_PATH \

#FORMED-REAL
DATA_PATH="/dataset/Bayesian_Data/FORMED/FORMED_real.csv"
GT_PATH="/dataset/Bayesian_Data/FORMED/DAGtrue_FORMED_real_bi.csv"
python notears/linear_custom.py --dataset="bayes" \
--data_path=$DATA_PATH \
--gt_path=$GT_PATH \

#PATHFINDER
DATA_PATH="/dataset/Bayesian_Data/PATHFINDER/PATHFINDER_DATA.csv"
GT_PATH="/dataset/Bayesian_Data/PATHFINDER/DAGtrue_PATHFINDER_bi.csv"
python notears/linear_custom.py --dataset="bayes" \
--data_path=$DATA_PATH \
--gt_path=$GT_PATH \

#PROPERTY 
DATA_PATH="/dataset/Bayesian_Data/PROPERTY/PROPERTY_DATA.csv"
GT_PATH="/dataset/Bayesian_Data/PROPERTY/DAGtrue_PROPERTY_bi.csv"
python notears/linear_custom.py --dataset="bayes" \
--data_path=$DATA_PATH \
--gt_path=$GT_PATH \

