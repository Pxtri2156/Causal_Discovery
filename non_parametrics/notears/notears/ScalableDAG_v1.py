
import  sys
from tqdm import tqdm

sys.path.append("./")
sys.path.append("/workspace/causal_discovery/non_parametrics/notears/notears")

from notears.locally_connected import LocallyConnected
from notears.lbfgsb_scipy import LBFGSBScipy
from notears.trace_expm import trace_expm
from torchsummary import summary

import torch
import torch.nn as nn
import numpy as np
import math

import notears.utils as ut
from notears.visualize import Visualization
from notears.orthogonality import latin_hyper, orthogonality
# from notears.log import Logging
import random


class ScalableDAG_v1_1(nn.Module): # Based on NotearsSobolev and non Mask
    
    def __init__(self, dims, d, k, bias=True):
        '''
            dims = [h1, h2, d]
        '''
        super(ScalableDAG_v1_1, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == d
        self.d = d
        self.dims = dims
        self.K = k
        # fc_phi: layers of K phi
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(self.K, dims[l], dims[l + 1], bias=bias)) # input [3, 10, d]
        layers.append(LocallyConnected(self.K, dims[-2], self.d , bias=bias)) 
        self.fc_phi = nn.ModuleList(layers)
        
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(self.d *self.K, self.d, bias=False) # d*dims[1] => 
        self.fc1_neg = nn.Linear(self.d *self.K, self.d, bias=False)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        nn.init.zeros_(self.fc1_pos.weight)
        nn.init.zeros_(self.fc1_neg.weight)
        self.l2_reg_store = None

    def _bounds(self):
        bounds = []
        for j in range(self.d):
            for m in range(self.d):
                for i in range(self.K):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        # print("bounds in function: ", len(bounds))
        return bounds

    def forward(self, x):  # [n, d] -> [n, d] 
        #[n, d] => [n, k, d]
        #  [[1,2], 
        #   [3,4]] => 
        #  [[[1,2], [1,2]],[[3,4], [3,4] ] ]
        x  = x.unsqueeze(1).repeat(1, self.K, 1)
        for phi_layer in self.fc_phi:
            x = phi_layer(x) # [n, k, d]
            x = torch.sigmoid(x)
        phi_values = x.view(-1, self.K * self.d) # [n, k*d]
        x = self.fc1_pos(phi_values) - self.fc1_neg(phi_values)  # [n, d]
        self.l2_reg_store = torch.sum(x ** 2) / x.shape[0]
        return x

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(self.d, self.d, self.K)  # [j, i, k]
        A = torch.sum(fc1_weight * fc1_weight, dim=2).t()  # [i, j]
        h = trace_expm(A) - self.d  # (Zheng et al. 2018)
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = self.l2_reg_store
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(self.d, self.d, self.K)  # [j, i, k]
        A = torch.sum(fc1_weight * fc1_weight, dim=2).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W


class ScalableDAG_v1_2(nn.Module): # Based on NotearsSobolev and Zero mask
    
    def __init__(self, dims, d, k, bias=True):
        '''
            dims = [h1, h2, d]
        '''
        super(ScalableDAG_v1_2, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        assert dims[0] == d
        self.d = d
        self.dims = dims
        self.K = k
        # fc_phi: layers of K phi
        layers = []
        for l in range(len(dims) - 1):
            layers.append(LocallyConnected(self.K, dims[l], dims[l + 1], bias=bias)) # input [3, 10, d]
        self.fc_phi = nn.ModuleList(layers)
        
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(self.d *self.K, self.d, bias=False) # d*dims[1] => 
        self.fc1_neg = nn.Linear(self.d *self.K, self.d, bias=False)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        nn.init.zeros_(self.fc1_pos.weight)
        nn.init.zeros_(self.fc1_neg.weight)
        self.l2_reg_store = None

    def _bounds(self):
        bounds = []
        for j in range(self.d):
            for m in range(self.d):
                for i in range(self.K):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        # print("bounds in function: ", len(bounds))
        return bounds

    def forward(self, x):  # [n, d] -> [n, d] 
        phi_values = []
        for j in range(self.d):
            x_phi = torch.clone(x)   
            x_phi[:,j] = 0  #[n, d]
            x_phi  = x_phi.unsqueeze(1).repeat(1, self.K, 1) # Convert #[n, d] => [n, k, d]
            for phi_layer in self.fc_phi:
                x_phi = phi_layer(x_phi) # [n, k, 1]
                x_phi = torch.sigmoid(x_phi)
            phi_values.append(x_phi.squeeze(dim=2)) # [n, k]*d
        phi_values = torch.stack(phi_values, dim=1) # [n, d, k ]
        phi_values = phi_values.view(-1, self.d * self.K) # [n, d*k]
        x = self.fc1_pos(phi_values) - self.fc1_neg(phi_values)  # [n, d]
        self.l2_reg_store = torch.sum(x ** 2) / x.shape[0]
        return x

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(self.d, self.d, self.K)  # [j, i, k]
        A = torch.sum(fc1_weight * fc1_weight, dim=2).t()  # [i, j]
        h = trace_expm(A) - self.d  # (Zheng et al. 2018)
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = self.l2_reg_store
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(self.d, self.d, self.K)  # [j, i, k]
        A = torch.sum(fc1_weight * fc1_weight, dim=2).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W


class ScalableDAG_v1_3(nn.Module): # Based on NotearsSobolev and drop mask
    
    def __init__(self, dims, d, k, bias=True):
        '''
            dims = [h1, h2, d]
        '''
        super(ScalableDAG_v1_3, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        assert dims[0] == d-1
        self.d = d
        self.dims = dims
        self.K = k
        # fc_phi: layers of K phi
        layers = []
        for l in range(len(dims) - 1):
            layers.append(LocallyConnected(self.K, dims[l], dims[l + 1], bias=bias)) # input [3, 10, d]
        self.fc_phi = nn.ModuleList(layers)
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(self.d *self.K, self.d, bias=False) # d*dims[1] => 
        self.fc1_neg = nn.Linear(self.d *self.K, self.d, bias=False)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        nn.init.zeros_(self.fc1_pos.weight)
        nn.init.zeros_(self.fc1_neg.weight)
        self.l2_reg_store = None

    def _bounds(self):
        bounds = []
        for j in range(self.d):
            for m in range(self.d):
                for i in range(self.K):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        # print("bounds in function: ", len(bounds))
        return bounds

    def forward(self, x):  # [n, d] -> [n, d] 
        phi_values = []
        for j in range(self.d):
            x_phi = torch.cat([x[:, :j], x[:, j+1:]], dim=1) # [n, d-1]
            x_phi  = x_phi.unsqueeze(1).repeat(1, self.K, 1) # Convert #[n, d] => [n, k, d-1]
            for phi_layer in self.fc_phi:
                x_phi = phi_layer(x_phi) # [n, k, 1]
                x_phi = torch.sigmoid(x_phi)
            phi_values.append(x_phi.squeeze(dim=2)) # [n, k]*d
        phi_values = torch.stack(phi_values, dim=1) # [n, d, k ]
        phi_values = phi_values.view(-1, self.d * self.K) # [n, d*k]
        x = self.fc1_pos(phi_values) - self.fc1_neg(phi_values)  # [n, d]
        self.l2_reg_store = torch.sum(x ** 2) / x.shape[0]
        return x

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(self.d, self.d, self.K)  # [j, i, k]
        A = torch.sum(fc1_weight * fc1_weight, dim=2).t()  # [i, j]
        h = trace_expm(A) - self.d  # (Zheng et al. 2018)
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = self.l2_reg_store
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(self.d, self.d, self.K)  # [j, i, k]
        A = torch.sum(fc1_weight * fc1_weight, dim=2).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W
    
class ScalableDAG_v1_4(nn.Module):
    
    def __init__(self, dims, d, k, bias=True):
        super(ScalableDAG_v1_4, self).__init__()
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
        # print("h fc1_weight: ", fc1_weight.shape)
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [d, d]
        # print("A: ", A)
        # print('trace_expm(A):', trace_expm(A))
        h = trace_expm(A) - self.d  # (Zheng et al. 2018)
        # print("h_func: ", h.item())
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.get_fc1_weight()  # # [d*m1, d]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        # l2 of alpha ??????????
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

class ScalableDAG_v1(nn.Module):
    
    def __init__(self, dims, k, bias=True):
        super(ScalableDAG_v1, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        self.K = k
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(d, self.K * dims[1], bias=bias) # d*dims[1] => 
        self.fc1_neg = nn.Linear(d, self.K * dims[1], bias=bias)
        self.fc1_pos.weight.bounds = self._bounds()
        # print('weight of fc1 pos: ', self.fc1_pos.weight.shape)
        # print("Size weight of bounds: ", len(self.fc1_pos.weight.bounds))
        # print("bound: ", len(self._bounds()))
        self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(self.K, dims[l + 1], dims[l + 2], bias=bias)) # input [d, 10, 1]
            
        self.fc2 = nn.ModuleList(layers)
        self.fc3 = nn.Linear(self.K, d,  bias=False)
        # print(type( self.fc2))
        
        # summary(self.fc2, (200,10, 1))

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(self.K):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        # print("bounds in function: ", len(bounds))
        return bounds

    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        # print("after fc1: ", x.shape)
        x = x.view(-1, self.K, self.dims[1])  # [n, d, m1]
        # print("after view: ", x.shape)
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        x =self.fc3(x)
        # print('x of forward: ', x.shape)
        return x

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        # print("h fc1_weight: ", fc1_weight.shape)
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        h = trace_expm(A) - d  # (Zheng et al. 2018)
        print("h_func: ", h)
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
    loss = 0.5 / n * torch.sum((output - target) ** 2) # Request norm 2
    return loss


def dual_ascent_step(model, X, lambda1, lambda2, rho, alpha, h, rho_max):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    # for param in model.parameters():
    #     print(type(param), param.size())
    # input("Hi")
    optimizer = LBFGSBScipy(model.parameters())
    X_torch = torch.from_numpy(X)
    primal_obj = 0.0
    loss = 0.0
    h_val = 0.0
    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            X_hat = model(X_torch)
            # print("X_hat: ", X_hat.shape)
            loss = squared_loss(X_hat, X_torch)
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj = loss + penalty + l2_reg + l1_reg
            primal_obj.backward()
            # print('primal_obj: ', primal_obj.item())
            return primal_obj
        
        optimizer.step(closure)  # NOTE: updates model in-place
        with torch.no_grad():
            h_new = model.h_func().item()
        if h_new > 0.25 * h:
            rho *= 10
        else:
            break
    alpha += rho * h_new
    return rho, alpha, h_new, primal_obj, loss, h_val


def scalable_dag_v1(model: nn.Module,
                      X: np.ndarray,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      max_iter: int = 100,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3):
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in tqdm(range(max_iter)):
        print(f"{'='*30} Iter {_} {'='*30}")
        rho, alpha, h, obj_func, loss, h_val = dual_ascent_step(model, X, lambda1, lambda2,
                                         rho, alpha, h, rho_max)
        num_epoch = _

        ortho = 0
        acc={'fdr':0, "fpr":0, 'tpr':0, 'shd':0}
        
        if h <= h_tol or rho >= rho_max:
            print(h)
            print(rho)
            print("hú")
            break
        
        
    W_est = model.fc1_to_adj() # convert the matrix
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


def main():
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    import notears.utils as ut
    ut.set_random_seed(1234)

    #LOGGING ----
    name = 'Scalable_v1'
    #AVERAG ---- 
    for r in [2,3,5,6,9,15,19,28,2000,2001]: #

    #LOGGING----
        
        n, d, s0, graph_type, sem_type = 200, 5, 9, 'ER', 'mim'
        B_true = ut.simulate_dag(d, s0, graph_type)
        np.savetxt('ScalableDAG_v1/W_true.csv', B_true, delimiter=',')

        X = ut.simulate_nonlinear_sem(B_true, n, sem_type)
        np.savetxt('ScalableDAG_v1/X.csv', X, delimiter=',')

        # model = ScalableDAG_v1_1(dims=[d, 10, d], d=d, k=10, bias=True)
        # model = ScalableDAG_v1_2(dims=[d, 10, 1], d=d, k=6, bias=True)
        # model = ScalableDAG_v1_3(dims=[d-1, 10, 1], d=d, k=6, bias=True)
        # model = ScalableDAG_v1(dims=[d, 10, 1], k=6, bias=True)
        model = ScalableDAG_v1_4(dims=[d, 10, 8, 1], d=5, k=6, bias=True)

        W_est = scalable_dag_v1(model, X, lambda1=0.01, lambda2=0.01)
        # print(W_est)
        assert ut.is_dag(W_est)
        np.savetxt('ScalableDAG_v1/W_est.csv', W_est, delimiter=',')
        acc = ut.count_accuracy(B_true, W_est != 0)
        print(acc)
        


if __name__ == '__main__':
    main()
