
import  sys

sys.path.append("./")
sys.path.append("/workspace/causal_discovery/non_parametrics/notears/notears")


from notears.locally_connected_Binh import LocallyConnected
from notears.lbfgsb_scipy import LBFGSBScipy
from notears.trace_expm import trace_expm
# from torchsummary import summary

import torch
import torch.nn as nn
import numpy as np
import math
from tqdm import tqdm
from torch.autograd.variable import Variable
from scipy.stats import qmc
import notears.utils as ut
from notears.visualize import Visualization
from notears.orthogonality import latin_hyper, orthogonality
from notears.log_causal import LogCausal
import random

class ScalableDAG_V2(nn.Module):
    def __init__(self, dims, bias=True):
        super(ScalableDAG_V2, self).__init__()
        assert len(dims) >= 2
        # assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias) # d*dims[1] => 
        self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()

        # fc2: local linear layers
        layers = []
        layers.append(LocallyConnected(1, d * dims[1], dims[2], bias=bias))

        for l in range(1,len(dims) - 2):
            layers.append(LocallyConnected(1, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)
        self.fc3 = nn.Linear(dims[-1], 1, bias=bias)


    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (None, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x):  # [n, d] -> [n, k]
        x_forward = torch.empty((x.size()))
        for j in range(x.shape[1]):
            x_j = torch.clone(x)   
            x_j[:,j] = 0 
            x_j = self.fc1_pos(x_j) - self.fc1_neg(x_j)  # [n, d * m1]
            # print(x.shape)
            for fc in self.fc2:
                x_j = torch.sigmoid(x_j) # [n, d * m1]
                x_j = fc(x_j)  # [n, m2]
                # print(x.shape)
            x_j = self.fc3(x_j)
            x_forward[:,j] =  x_j[:,0]
        return x_forward

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        h = trace_expm(A) - d  # (Zheng et al. 2018)
        # A different formulation, slightly faster at the cost of numerical stability
        # M = torch.eye(d) + A / d  # (Yu et al. 2019)
        # E = torch.matrix_power(M, d - 1)
        # h = (E.t() * M).sum() - d
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg
    
    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W

  
def squared_loss(output, target):
    n = target.shape[0]
    loss = 1 / n * torch.sum((output - target) ** 2)
    return loss

def dual_ascent_step(model, X, B_true, w_threshold, lambda1, lambda2, lambda3, rho, alpha, h, rho_max, X_latin, log):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    optimizer = LBFGSBScipy(model.parameters()) 
    while rho < rho_max:
        def closure():
            loss = 0.0
            optimizer.zero_grad()
            #transform to tensor X_phi, X_alpha
            X_torch = torch.from_numpy(X)
            #get X_hat
            X_hat = model(X_torch)            
            #loss 
            loss = squared_loss(X_hat, X_torch) 
            ortho = lambda3*orthogonality(model(X_latin))
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            
            primal_obj = loss + ortho + penalty + l2_reg + l1_reg
            #backwward
            primal_obj.backward()
            return primal_obj  #primal_obj
        optimizer.step(closure)  # NOTE: updates model in-place
        
        with torch.no_grad():
            h_new = model.h_func().item()
            X_torch = torch.from_numpy(X)

            #get X_hat
            X_hat = model(X_torch)            

            #loss 
            loss = squared_loss(X_hat, X_torch) 
            ortho = lambda3*orthogonality(model(X_latin))
            penalty = 0.5 * rho * h_new * h_new + alpha * h_new
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            
            primal_obj = loss + ortho + penalty + l2_reg + l1_reg

        if h_new > 0.25 * h:
            rho *= 10
        else:
            break

    alpha += rho * h_new
    return rho, alpha, h_new, primal_obj, loss, ortho

def ScalableDAG_V2_nonlinear(model: nn.Module,
                      X: np.ndarray,log,B_true,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      lambda3: float = 0.,
                      max_iter: int = 100,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3):
    rho, alpha, h = 1.0, 0.0, np.inf
    X_latin = latin_hyper(X, n=20)
    for _ in tqdm(range(max_iter)):
        rho, alpha, h, primal_obj, loss, ortho = dual_ascent_step(model, X, B_true, w_threshold, lambda1, lambda2, lambda3,
                                        rho, alpha, h, rho_max, X_latin, log)   
        log.step_update(primal_obj, loss, ortho, h)
        num_epoch = _ + 1
        if h <= h_tol or rho >= rho_max:
            break
        

    log.log['num_epoch'][log.log['random_seed'][-1]] = num_epoch

    W_est = model.fc1_to_adj()
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


def main():
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    #LOGGING ----
    lambda1 = 0.01
    lambda2 = 0.01
    lambda3 = 0.01
    lambda_ = [lambda1, lambda2, lambda3]
    w_threshold = 0.3
    name = 'test_1'
    log = LogCausal(name, lambda_)
    #AVERAG ---- 

    random_numbers = [random.randint(1, 10000) for _ in range(3)]#[702,210,1536]
    print(random_numbers)
    for r in random_numbers: #[2,3,5,6,9,15,19,28,2000,2001]
        # print('Binh')
        print("\n-----------------------------------------")
        print('random seed:',r)
        ut.set_random_seed(r)

        #LOGGING----
        log.random_seed_update(r)

        n, d, s0, graph_type, sem_type = 200, 5, 9, 'ER', 'mim'
        B_true = ut.simulate_dag(d, s0, graph_type)
        # np.savetxt(f'ScalableDAG_v2/W_true_{r}.csv', B_true, delimiter=',')

        X = ut.simulate_nonlinear_sem(B_true, n, sem_type)
        # np.savetxt(f'ScalableDAG_v2/X_{r}.csv', X, delimiter=',')

        k = 3
        # model = ScalableDAG_V2(dims=[d, 10, k], bias=True)
        model = ScalableDAG_V2(dims=[d , 10, k], bias=True)

        
        W_est = ScalableDAG_V2_nonlinear(model, X, log, B_true, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, max_iter=50, w_threshold=w_threshold) #ADD LOG 
        assert ut.is_dag(W_est)
        acc = ut.count_accuracy(B_true, W_est != 0)
        # np.savetxt(f'ScalableDAG_v2/W_est_{r}.csv', W_est, delimiter=',')
        
        print(W_est)
        print(B_true)
        print(acc)
        log.acc_update(acc) 

    # VISUALIZTION
    save_path = 'ScalableDAG_v2/visualization'
    vis = Visualization(log.log, save_path)
    vis.visualize()

    print("\n-----------------------------------------")
    log.print_acc_average()

if __name__ == '__main__':
    main()

