import numpy as np
import pandas as pd
from sklearn import preprocessing 
import argparse
import yaml
import os
import sys
sys.path.append("./")
from libs.notears.notears.linear import notears_linear
from libs.notears.notears.nonlinear import NotearsMLP, notears_nonlinear
from libs.dag_gnn.src.dag_gnn import dag_gnn, train_dag_gnn
from libs.golem.src.golem import golem
from libs.golem.src.utils.dir import get_datetime_str
from libs.notears.notears import utils as notears_utils
from libs.bngpu.nobears_benchmark.benchmark_nobears import run_nobear
from utils.evaluation import count_accuracy
from utils.load_data import DataLoader 
from utils.processing import Processing

class DAG_Agorithm:
    def __init__(self, X, ground_truth, configs):
        self.name = configs.ALGORITHM['NAME']
        self.configs = configs
        self.X = X 
        self.gt = ground_truth
        
    def run(self):
        W_est = None
        if self.name == "NOTEAR":
            W_est = notears_linear(self.X, \
                    lambda1=self.configs.ALGORITHM['LAMPDA_1'], \
                    loss_type=self.configs.ALGORITHM['LOSS'])
            
        elif self.name == "GOLEM":
            output_dir = 'libs/golem/output/{}'.format(get_datetime_str())
            W_est = golem(self.X, self.configs.ALGORITHM['LAMBDA_1'], self.configs.ALGORITHM['LAMBDA_2'], self.configs.ALGORITHM['EQUAL_VARIANCE'], \
                         self.configs.ALGORITHM['NUM_ITER'], self.configs.ALGORITHM['LR'], self.configs.ALGORITHM['SEED'], \
                            self.configs.ALGORITHM['CHECKPOINT_ITER'], output_dir, None)
        elif self.name == "NOBEAR":
            W_est = run_nobear(self.X)
            
        elif self.name == "DAG-GNN":
            print("Running DAG-GNN")
            W_est = dag_gnn(self.X, self.gt)
            
        elif self.name == "Non_parametric":
            print("Running Non Parametric")
            model = NotearsMLP(self.configs.MODEL['dims'], self.configs.MODEL['bias'])
            W_est = notears_nonlinear(model, self.X, self.configs.ALGORITHM['LAMPDA_1'], \
                self.configs.ALGORITHM['LAMPDA_2'])
        else:
            raise Exception ("Algorithm haven't defined or wrong name")
        return W_est
    

def load_config(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return argparse.Namespace(**config)
    
def main(configs):
    # Load data 
    if configs.DATASET['NAME'] != "synthetic":
        data = DataLoader(configs)
        X, gt = data.load_data(configs.DATASET['DATA_PATH'], configs.DATASET['GT_PATH'])
    else: 
        data = DataLoader(configs)
        X, gt = data.gen_data()
        
    print("X: ", X.shape)
    print("X[0]: ", X[0])
    # print("Type X: ", type(X))
    print("Ground Truth: \n", gt.shape)
    print("gt[0]: ", gt)
    # Preprocessing 

    # Algorithms 
    dag_algorithm = DAG_Agorithm(X, gt, configs)
    W_est = dag_algorithm.run()
    print("W_est: ", W_est)
    
    # Thresolding
    post_processing = Processing(W_est, configs.THRESOLDING)
    W_est = post_processing.run()
    print("W_est_post-processing: ", W_est)

    # Save results
    np.savetxt('W_est.csv', W_est, delimiter=',')

    # Evaluate       
    acc = count_accuracy(gt, W_est != 0)
    print(acc)
    
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithms",
        default='notear',
        type=str,
        help="The algorithms you want to use",
    )
    parser.add_argument(
        "--config",
        default='/workspace/causal_discovery/source/configs/golem/default.yaml',
        type=str,
        help="The file contain all of configs and hyper parameters.",
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = get_parser()
    config = load_config(args.config)
    main(config)
