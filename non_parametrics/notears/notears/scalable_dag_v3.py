import  sys
from tqdm import tqdm

sys.path.append("./")
sys.path.append("/workspace/causal_discovery/non_parametrics/notears/notears")

from notears.locally_connected import LocallyConnected
from notears.lbfgsb_scipy import LBFGSBScipy
from notears.trace_expm import trace_expm

import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import math

import notears.utils as ut
from notears.visualize import Visualization
from notears.orthogonality import latin_hyper, orthogonality
from notears.vae import VAE
import random

   
class ScalableDAGv3(nn.Module):
    
    def __init__(self, dims, pivae_dims, d, k, batch_size, bias=True ):
        super(ScalableDAGv3, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        self.d = d
        self.dims = dims
        self.K = k
        self.pivae_dims = pivae_dims
        self.batch_size = batch_size
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
              
        # Define pivae layers
        self.betas = nn.ModuleList()
        for _ in range(self.batch_size):
            self.betas.append(nn.Linear(k, 1)) 
                   
        self.vae = VAE(input_dim=k, hidden_dim1=pivae_dims[0], 
                        hidden_dim2=pivae_dims[1], latent_dim=pivae_dims[2])
    
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
        last_phi_x = []
        for j, (fc1_pos_iter, fc1_neg_iter) in enumerate(zip(self.fc1_pos, self.fc1_neg)):
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
            last_phi_x.append(x_phi)
            
        last_phi_x = torch.stack(last_phi_x) # [d, n, k]
        last_phi_x = last_phi_x.transpose(0, 1) # [n, d, k]

        # feed for forward of VAE
        y1 = torch.stack([self.betas[i](last_phi_x[i]) for i in range(self.batch_size)
                            ]).flatten(1)
        beta = torch.stack([self.betas[i].weight for i in range(self.batch_size)
                            ]).flatten(0,1)
        beta_vae = self.vae(beta)
        y2 = torch.stack([last_phi_x[i]@beta_vae[0][i] + self.betas[i].bias for i in 
                            range(self.batch_size)]) # y2 = X_hat 
        print(y1.shape, y2.shape)
        return y1, y2, beta_vae[1], beta_vae[2]

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

def cal_vae_loss(target, reconstructed1, reconstructed2, mean, log_var): # loss vae
    # reconstruction loss
    RCL = F.mse_loss(reconstructed1, target, reduction='sum') + \
                F.mse_loss(reconstructed2, target, reduction='sum') # Loss 1 + 2 
    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) # RegLoss
    return RCL + KLD


def dual_ascent_step(model, X, lambda1, lambda2, lambda3, rho, alpha, h, rho_max):
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
            # X_hat = model(X_torch)
            y1, y2, z_mu, z_sd = model(X_torch)
            # print("X_hat: ", X_hat.shape)
            loss = squared_loss(y2, X_torch)
            vae_loss = cal_vae_loss(X_torch, y1, y2, z_mu, z_sd)
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj = loss + penalty + l2_reg + l1_reg + vae_loss*lambda3
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


def scalable_dag_v3(model: nn.Module,
                      X: np.ndarray,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      lambda3: float = 0.,
                      max_iter: int = 100,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3):
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in tqdm(range(max_iter)):
        print(f"{'='*30} Iter {_} {'='*30}")
        rho, alpha, h, obj_func, loss, h_val = dual_ascent_step(model, X, lambda1, lambda2, lambda3,
                                         rho, alpha, h, rho_max)
        num_epoch = _

        ortho = 0
        acc={'fdr':0, "fpr":0, 'tpr':0, 'shd':0}
        
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

    import notears.utils as ut
    ut.set_random_seed(1234)
    for r in [15]: #
    #LOGGING----
        
        n, d, k, s0, graph_type, sem_type = 200, 5, 6, 9, 'ER', 'mim'
        hidden_dims1 = 128
        hidden_dims2 = 64
        z_dim = 20
        dims = [d, 10, 8, 1]
        batch_size = n
        pivae_dims = [hidden_dims1, hidden_dims2, z_dim]
        
        B_true = ut.simulate_dag(d, s0, graph_type)
        # np.savetxt('scalable_dag_v3/W_true.csv', B_true, delimiter=',')

        X = ut.simulate_nonlinear_sem(B_true, n, sem_type)
        # np.savetxt('scalable_dag_v3/X.csv', X, delimiter=',')
        print("[INFO]: Done gen and save dataset!!!")
        model = ScalableDAGv3(dims, pivae_dims, d, k, batch_size, bias=True )

        W_est = scalable_dag_v3(model, X, lambda1=0.01, lambda2=0.01, lambda3=0.01)
        # print(W_est)
        assert ut.is_dag(W_est)
        # np.savetxt('scalable_dag_v3/W_est.csv', W_est, delimiter=',')
        acc = ut.count_accuracy(B_true, W_est != 0)
        print(acc)
        

if __name__ == '__main__':
    main()