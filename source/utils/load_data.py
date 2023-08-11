import pandas as pd
import numpy as np
import os

from libs.notears.notears import utils as notears_utils

class DataLoader:
    
    def __init__(self, data_name):
        self.data_name = data_name
        self.X = None # Dataset matrix
        self.GT = None # Ground truth matrix
        
    def load_data(self, dataset_path, gt_path):
        """ Load data 
        Args:
            dataset_path (string): the path contain dataset with csv format
            gt_path (string): the path contain ground truth with csv format

        Returns:
            X (np.ndarray): [n, d] dataset matrix with n is no. samples and d is no. nodes (features)
            GT (np.ndarray): [d, d] ground truth graph with values {0, 1}
        """

        if self.data_name == "default":
            print(f"[INFO] Start loading dataset {self.data_name}")
            X = pd.read_csv(dataset_path) 
            # convert catagories to numberic values
            cat_columns = X.select_dtypes(['object']).columns 
            X[cat_columns] = X[cat_columns].apply(lambda x: pd.factorize(x)[0])
            # convert boolean to numberic values
            bool_columns = X.select_dtypes(['boolean']).columns
            X[bool_columns] = X[bool_columns].replace({True: 1, False: 0})
            X = X.to_numpy()
            self.X = X
            print(f"[INFO] Finishing loading dataset, shape data {self.X.shape}")
            
            print(f"[INFO] Start loading ground truth")
            gt = pd.read_csv(gt_path, header=None).to_numpy()
            self.GT = gt
            print(f"[INFO] Finishing loading ground truth, shape gt {self.GT.shape}")

        elif self.data_name == "sachs":
            print(f"[INFO] Start loading dataset {self.data_name}")
            X = None   
            X = pd.read_csv(dataset_path)
            X.columns = [x.lower() for x in X.columns]
            X=X.to_numpy() 
            self.X = X            
            print(f"[INFO] Finishing loading dataset, shape data {self.X.shape}")
            gt = pd.read_csv(gt_path,header=None)
            gt = gt.to_numpy() 
            gt = np.asfarray(gt) # Need to review
            self.GT = gt
            print(f"[INFO] Start loading ground truth")
            print(f"[INFO] Finishing loading ground truth, shape gt {self.GT.shape}")
            
        elif self.data_name == "synthesis":
            self.gen_syn_data()
            print(f"[INFO] Finished generate synthesis data with shape data {self.X.shape}, shape gt {self.GT.shape}")

        else: 
            raise Exception ("Dataset haven't defined")
            
        return self.X, self.GT
    
    def gen_syn_data(self):
        print(f"[INFO] Start generate synthesis data")
        n, d, s0, graph_type, sem_type = 100, 20, 20, 'ER', 'gauss'
        B_true = notears_utils.simulate_dag(d, s0, graph_type)
        W_true = notears_utils.simulate_parameter(B_true)
        X = notears_utils.simulate_linear_sem(W_true, n, sem_type)
        self.X = X
        self.GT = B_true
