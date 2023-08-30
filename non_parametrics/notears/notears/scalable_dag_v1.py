import  sys
sys.path.append("../")
sys.path.append("/workspace/causal_discovery/non_parametrics/notears/notears")

from tqdm import tqdm
import os
import torch
import torch.nn as nn
import numpy as np
import math
import wandb
import random

from notears.locally_connected import LocallyConnected
from notears.lbfgsb_scipy import LBFGSBScipy
from notears.trace_expm import trace_expm
import notears.utils as ut


class ScalableDAG_v1(nn.Module):
    
    def __init__(self, dims, d, k, bias=True):
        super(ScalableDAG_v1, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        self.d = d
        self.dims = dims
        self.K = k
        # fc1: variable splitting for l1
        self.fc1_pos = []
        self.fc1_neg = []
        bounds = self._bounds()
        for j in range(d):
            self.fc1_pos.append(nn.Linear(d, dims[1], bias=bias))
            self.fc1_neg.append(nn.Linear(d, dims[1], bias=bias))
            self.fc1_pos[j].weight.bounds = bounds[j]
            self.fc1_neg[j].weight.bounds = bounds[j]
            # print('weight fc1: ', self.fc1_pos[j].weight.size())
        # fc2: local linear layers
        self.fc1_pos = nn.ModuleList(self.fc1_pos)
        self.fc1_neg = nn.ModuleList(self.fc1_neg)
        
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(self.K, dims[l + 1], dims[l + 2], bias=bias)) # input [d, 10, 1]
            
        self.fc2 = nn.ModuleList(layers)
        self.fc3 = [] 
        for _ in range(d):
            self.fc3.append(nn.Linear(self.K, 1,  bias=False))
        self.fc3 = nn.ModuleList(self.fc3)        
        # summary(self.fc2, (200,10, 1))

    def _bounds(self): #DAG
        bounds = []
        for j in range(self.d):
            sub_bounds = []
            for m in range(self.dims[1]):
                for i in range(self.d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    sub_bounds.append(bound)
                    
            bounds.append(sub_bounds)
        # print("bounds in function: ", len(bounds))
        return bounds

    def forward(self, x):  # [n, d] -> [n, d]
        last_x = torch.empty((x.size()))
        for j, (fc1_pos_iter, fc1_neg_iter, fc3_iter) in enumerate(zip(self.fc1_pos, self.fc1_neg, self.fc3)):
            x_phi = torch.clone(x)   
            x_phi[:,j] = 0  #[n, d]
            x_phi = fc1_pos_iter(x_phi) - fc1_neg_iter(x_phi)  # [n, m1]
            x_phi = x_phi[:, None, :].expand(-1, self.K, -1)  # [n, k, m1]
            for fc in self.fc2:
                x_phi = torch.sigmoid(x_phi)  # [n, k, m1]
                x_phi = fc(x_phi)  # [n, k, 1]
                # print('fc2 weight: ', fc.weight.size())
            # print(x_phi.shape)
            x_phi = x_phi.squeeze(dim=2)  # [n, k]
            x_phi = fc3_iter(x_phi) # [n, 1]
            # print('x of forward: ', x.shape)
            last_x[:,j] = x_phi[:, 0]
        return last_x

    def get_fc1_weight(self):
        fc1_weight = torch.empty((self.d*self.dims[1], self.d))
        for j, (fc1_pos_iter, fc1_neg_iter) in enumerate(zip(self.fc1_pos, self.fc1_neg)):
            weight = fc1_pos_iter.weight - fc1_neg_iter.weight # [m1, d]
            start_ix = j*self.dims[1]
            end_ix = start_ix + self.dims[1] 
            fc1_weight[start_ix:end_ix, :] = weight
        return fc1_weight # [d*m1, d]
    
    def get_sum_fc1_weight(self):
        fc1_weight = torch.empty((self.d*self.dims[1], self.d))
        for j, (fc1_pos_iter, fc1_neg_iter) in enumerate(zip(self.fc1_pos, self.fc1_neg)):
            weight = fc1_pos_iter.weight + fc1_neg_iter.weight # [m1, d]
            start_ix = j*self.dims[1]
            end_ix = start_ix + self.dims[1] 
            fc1_weight[start_ix:end_ix, :] = weight
        return fc1_weight # [d*m1, d]
    
    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        fc1_weight = self.get_fc1_weight()  # # [d*m1, d]
        fc1_weight = fc1_weight.view(self.d, -1, self.d)  # [d, m1, d]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [d, d]
        h = trace_expm(A) - self.d  # (Zheng et al. 2018)
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.get_fc1_weight()  # # [d*m1, d]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.get_sum_fc1_weight())
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        fc1_weight = self.get_fc1_weight()  # # [d*m1, d]
        fc1_weight = fc1_weight.view(self.d, -1, self.d)  # [d, m1, d]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W

def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2) # Request norm 2
    return loss

def dual_ascent_step(model, X, wandb, lambda1, lambda2, rho, alpha, h, rho_max):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    optimizer = LBFGSBScipy(model.parameters())
    X_torch = torch.from_numpy(X)
    
    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            X_hat = model(X_torch)
            loss = squared_loss(X_hat, X_torch)
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj = loss + penalty + l2_reg + l1_reg
            result_dict = {'obj_func': primal_obj, 
                                        'sq_loss': loss, 
                                        'penalty': penalty,
                                        'h_func': h_val.item(), 
                                        'l2_reg': l2_reg,
                                        'l1_reg': l1_reg}
            wandb.log(result_dict)
            primal_obj.backward()
            return primal_obj
        
        optimizer.step(closure)  # NOTE: updates model in-place
        with torch.no_grad():
            h_new = model.h_func().item()
        if h_new > 0.25 * h:
            rho *= 10
        else:
            break
        wandb.log({'rho': rho})  
        
    alpha += rho * h_new
    wandb.log({'alpha': alpha})  
    return rho, alpha, h_new

def scalable_dag_v1(model: nn.Module,
                      X: np.ndarray, wandb,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      max_iter: int = 100,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3):
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in tqdm(range(max_iter)):
        print(f"{'='*30} Iter {_} {'='*30}")
        rho, alpha, h = dual_ascent_step(model, X, wandb, lambda1, lambda2,
                                         rho, alpha, h, rho_max)
        if h <= h_tol or rho >= rho_max:
            print(h)
            print(rho)
            break
        
    W_est = model.fc1_to_adj() # convert the matrix
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est

def main():
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)
    root_path = '../results/scalable_dag_v1'
    lambda1 = 0.01
    lambda2 = 0.01
    #LOGGING ----    

    for r in tqdm(range(2, 3)): 
        ut.set_random_seed(r)
        name_seed = 'seed_' + str(r)
        save_foler = root_path + f"/{name_seed}"
        if not os.path.isdir(save_foler):
            os.mkdir(save_foler)
            
        wandb.init(
            project="ScalableDAG_v1",
            name=name_seed,
            config={
            "name": name_seed,
            "dataset": "Synthetic",
            "lambda1": lambda1,
            "lambda2": lambda2,},
            dir=save_foler)
        
        n, d, s0, graph_type, sem_type = 200, 5, 9, 'ER', 'mim'
        B_true = ut.simulate_dag(d, s0, graph_type)
        np.savetxt(f'{save_foler}/W_true.csv', B_true, delimiter=',')

        X = ut.simulate_nonlinear_sem(B_true, n, sem_type)
        np.savetxt(f'{save_foler}/X.csv', X, delimiter=',')

        model = ScalableDAG_v1(dims=[d, 10, 8, 1], d=5, k=6, bias=True)
        sum_param = 0 
        # for name, param in model.named_parameters():
        #     print(f"Parameter name: {name}")
        #     print(param)
        #     print("=" * 20)

        #     # sum_param += param.size 
        # input("Stop")
        # params = sum([np.prod(p.size()) for _, p in model.named_parameters()])
        params = sum(p.numel() for p in model.parameters())
        print("params: ", params)
        input("Stop")
        W_est = scalable_dag_v1(model, X, wandb, lambda1, lambda2)
        # print(W_est)
        assert ut.is_dag(W_est)
        np.savetxt(f'{save_foler}/W_est.csv', W_est, delimiter=',')
        acc = ut.count_accuracy(B_true, W_est != 0)
        print("acc: ", acc)
        wandb.log({'acc': acc})
        wandb.finish()
        break


if __name__ == '__main__':
    main()
