
import  sys

sys.path.append("./")


from notears.locally_connected_Binh import LocallyConnected
from notears.lbfgsb_scipy import LBFGSBScipy
from notears.trace_expm import trace_expm
# from torchsummary import summary

import torch
import torch.nn as nn
import numpy as np
import math
from tqdm import tqdm
import notears.utils as ut
from notears.orthogonality import latin_hyper, orthogonality

import torch.nn.functional as F 
import random
import os
import wandb
from numpy.linalg import norm

class ScalableDAG_V2_1(nn.Module):
    def __init__(self, dims, bias=True):
        super(ScalableDAG_V2_1, self).__init__()
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
        
        self.fc3 = [] 
        for _ in range(d):
            self.fc3.append(nn.Linear(dims[-1], 1, bias=False))
        self.fc3 = nn.ModuleList(self.fc3)


    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
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
            x_j = self.fc3[j](x_j)
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

def cal_vae_loss(target, reconstructed1, reconstructed2, mean, log_var): # loss vae
    # reconstruction loss
    RCL = F.mse_loss(reconstructed1, target, reduction='sum') + \
                F.mse_loss(reconstructed2, target, reduction='sum') # Loss 1 + 2 
    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) # RegLoss
    return RCL + KLD

def algo_2_rho(mu, mu_max, beta, kappa, eta, lamd):
    if norm(kappa) < eta: 
        lamd +=mu*eta 
    else: 
        mu = min(beta*mu,mu_max)
    eta = norm(kappa)
    return mu, lamd, eta  #rho, alpha, h

def algo_3_rho(mu, v, kappa, lamd):
    gamma = 1e-2
    alpha = 0.99
    epsilon = 1e-8
    v = alpha*v + (1-alpha)*(kappa**2)
    mu = gamma* (1/(math.sqrt(v)+epsilon))
    lamd += mu*kappa
    return mu, v, lamd  #rho, alpha, h


def dual_ascent_step(model, X, lambda1, lambda2, lambda3, lambda4, rho, alpha, h, rho_max, X_latin, beta):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    #transform to tensor X_phi, X_alpha
    X_torch = torch.from_numpy(X)
    optimizer = LBFGSBScipy(model.parameters()) 

    while rho < rho_max:
        def closure():
            optimizer.zero_grad()

            #get X_hat
            X_hat = model(X_torch)   

            #loss 
            loss = lambda4*squared_loss(X_hat, X_torch) 

            ortho = lambda3*orthogonality(model(X_latin))
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            
            primal_obj = loss + ortho + penalty + l2_reg + l1_reg

            #log
            result_dict = {'obj_func': primal_obj, 
                           'sq_loss': loss, 
                           'orth':ortho, 
                           'h_func': h_val, 
                           'penalty': penalty, 
                           'l1': l1_reg, 
                           'l2': l2_reg, 
                           'rho': rho, 
                           'alpha': alpha} 
            wandb.log(result_dict)
            
            # backwward
            primal_obj.backward()
            return primal_obj  #primal_obj
        optimizer.step(closure)  # NOTE: updates model in-place
        
        with torch.no_grad():
            h_new = model.h_func().item()
        if h_new > 0.025 * h:
            # rho, alpha, h_new = algo_2_rho(rho, rho_max, beta, h_new, h, alpha)    
            rho, alpha, h_new = algo_3_rho(rho, alpha, h_new, h)  

        else:
            break
    return rho, alpha, h_new

def dual_ascent_step_gradient(model, X, lambda1, lambda2, lambda3, lambda4, rho, alpha, h, rho_max, X_latin, beta):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    #transform to tensor X_phi, X_alpha
    X_torch = torch.from_numpy(X)
    optimizer = torch.optim.Adam() 

    while rho < rho_max:
        optimizer.zero_grad()

        #get X_hat
        X_hat = model(X_torch)   

        #loss 
        loss = lambda4*squared_loss(X_hat, X_torch) 

        ortho = lambda3*orthogonality(model(X_latin))
        h_val = model.h_func()
        penalty = 0.5 * rho * h_val * h_val + alpha * h_val
        l2_reg = 0.5 * lambda2 * model.l2_reg()
        l1_reg = lambda1 * model.fc1_l1_reg()
        
        primal_obj = loss + ortho + penalty + l2_reg + l1_reg

        #log
        result_dict = {'obj_func': primal_obj, 
                        'sq_loss': loss, 
                        'orth':ortho, 
                        'h_func': h_val, 
                        'penalty': penalty, 
                        'l1': l1_reg, 
                        'l2': l2_reg, 
                        'rho': rho, 
                        'alpha': alpha} 
        wandb.log(result_dict)
        
        # backwward
        primal_obj.backward()
        optimizer.step()  # NOTE: updates model in-place
        
        with torch.no_grad():
            h_new = model.h_func().item()
            
        if h_new > 0.25 * h:
            rho *= beta
        else:
            break

    alpha += rho * h_new
    return rho, alpha, h_new

def ScalableDAG_V2_nonlinear(model: nn.Module,
                      X: np.ndarray,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      lambda3: float = 0.,
                      lambda4: float = 0.,
                      max_iter: int = 100,
                      h_tol: float = 1e-7,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3,
                      beta = 10):
    rho, alpha, h = 1.0, 0.0, np.inf
    X_latin = latin_hyper(X, n=20)
    for _ in tqdm(range(max_iter)):
        rho, alpha, h = dual_ascent_step(model, X, lambda1, lambda2, lambda3, lambda4,
                                        rho, alpha, h, rho_max, X_latin, beta)   
        if h <= h_tol or rho >= rho_max:
            break
        
    W_est = model.fc1_to_adj()
    print(f'Before threshold:\n{W_est}')
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


def main():
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    lambda1 = 0.01
    lambda2 = 0.01
    lambda3 = 0.15
    lambda4 = 1

    random_numbers = range(10)
    print(random_numbers)
    for r in random_numbers: 
        # print('Binh')
        print("\n-----------------------------------------")
        print('random seed:',r)
        #log 
        name_seed = "seed_" + str(r)
        save_foler = 'ScalableDAG_v2/logs/'

        wandb.init(
            project="ScalableDAG_v2_1_algo_3",
            name=name_seed,
            config={
            "name": name_seed,
            "dataset": "Synthetic",
            "lambda1": lambda1,
            "lambda2": lambda2,
            "lambda3": lambda3,
            "lambda4": lambda4,},
            dir=save_foler)

        n, d, s0, graph_type, sem_type = 200, 5, 9, 'ER', 'mim'
        k = 3
        #--------
        B_true = ut.simulate_dag(d, s0, graph_type)
        # np.savetxt(f'ScalableDAG_v2/W_true_{r}.csv', B_true, delimiter=',')

        X = ut.simulate_nonlinear_sem(B_true, n, sem_type)
        # np.savetxt(f'ScalableDAG_v2/X_{r}.csv', X, delimiter=',')

        
        model = ScalableDAG_V2_1(dims=[d , 10, k], bias=True)
        
        
        W_est = ScalableDAG_V2_nonlinear(model, X, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, lambda4=lambda4, max_iter=50, w_threshold=0.3) #ADD LOG 
        assert ut.is_dag(W_est)
        acc = ut.count_accuracy(B_true, W_est != 0)
        # np.savetxt(f'ScalableDAG_v2/W_est_{r}.csv', W_est, delimiter=',')
        
        print(f'After threshold:\n {W_est}')
        print(f'Ground truth:\n {B_true}')
        print(acc)
        wandb.log({'acc': acc})
        wandb.finish()

if __name__ == '__main__':
    main()

