
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
from torch.autograd.variable import Variable
from scipy.stats import qmc
import notears.utils as ut
from notears.visualize import Visualization
from notears.orthogonality import latin_hyper, orthogonality
from notears.log_causal import LogCausal
from notears.vae import VAE
import torch.nn.functional as F 
import random

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

class ScalableDAG_V2_2(nn.Module):
    def __init__(self, dims, bias=True):
        super(ScalableDAG_V2_2, self).__init__()
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

class ScalableDAG_V2_3(nn.Module):
    def __init__(self, dims, pivae_dims, d, k, batch_size, bias=True):
        super(ScalableDAG_V2_3, self).__init__()
        assert len(dims) >= 2
        # assert dims[-1] == 1
        d = dims[0]
        self.d = d
        self.dims = dims
        self.K = k
        self.pivae_dims = pivae_dims
        self.batch_size = batch_size
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
        
        # Define pivae layers
        self.betas = nn.ModuleList()
        for _ in range(self.batch_size):
            self.betas.append(nn.Linear(k, 1)) 
                   
        self.vae = VAE(input_dim=k, hidden_dim1=pivae_dims[0], 
                        hidden_dim2=pivae_dims[1], latent_dim=pivae_dims[2])



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
        x_phi = []
        for j in range(x.shape[1]):
            x_j = torch.clone(x)   
            x_j[:,j] = 0 
            x_j = self.fc1_pos(x_j) - self.fc1_neg(x_j)  # [n, d * m1]
            for fc in self.fc2:
                x_j = torch.sigmoid(x_j) # [n, d * m1]
                x_j = fc(x_j)  # [n, m2]
            x_phi.append(x_j)
        x_phi = torch.stack(x_phi)
        x_phi = x_phi.transpose(0, 1)

        # feed for forward of VAE
        print(len(self.betas))
        y1 = torch.stack([self.betas[i](x_phi[i]) for i in range(self.batch_size)
                            ]).flatten(1)
        
        beta = torch.stack([self.betas[i].weight for i in range(self.batch_size)
                            ]).flatten(0,1)
        beta_vae = self.vae(beta)
        y2 = torch.stack([x_phi[i]@beta_vae[0][i] + self.betas[i].bias for i in 
                            range(self.batch_size)])
        print(y1.shape, y2.shape)
        return y1, y2, beta_vae[1], beta_vae[2]

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

def dual_ascent_step(model, X, lambda1, lambda2, lambda3, lambda4, rho, alpha, h, rho_max, X_latin):
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

            # y1, X_hat, z_mu, z_sd = model(X_torch)   

            #loss 
            loss = squared_loss(X_hat, X_torch) 
            # vae_loss = cal_vae_loss(X_torch, y1, X_hat, z_mu, z_sd)

            ortho = lambda3*orthogonality(model(X_latin))
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            
            primal_obj = loss + ortho + penalty + l2_reg + l1_reg
            # primal_obj = loss + vae_loss + ortho + penalty + l2_reg + l1_reg
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
            loss = lambda4*squared_loss(X_hat, X_torch) 
            

            # get with piVAE   
            # y1, X_hat, z_mu, z_sd = model(X_torch)   
            #loss 
            # loss = squared_loss(X_hat, X_torch) 
            # vae_loss = cal_vae_loss(X_torch, y1, X_hat, z_mu, z_sd)

            ortho = lambda3*orthogonality(model(X_latin))
            penalty = 0.5 * rho * h_new * h_new + alpha * h_new
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            
            primal_obj = loss + ortho + penalty + l2_reg + l1_reg
            result_dict = {'obj_func': primal_obj, 'sq_loss': loss, 'orth':ortho, 'h_func': h_new, 'pen': penalty, 'l1': l1_reg, 'l2': l2_reg} #THE SAME KEY AS INIT IN MAIN FUNCTION
            # primal_obj = loss + vae_loss + ortho + penalty + l2_reg + l1_reg
            # result_dict = {'obj_func': primal_obj, 'sq_loss': loss, 'vae_loss': vae_loss, 'orth':ortho, 'h_func': h_new} #THE SAME KEY AS INIT IN MAIN FUNCTION
            
        if h_new > 0.25 * h:
            rho *= 10
        else:
            break

    alpha += rho * h_new
    return rho, alpha, h_new, result_dict

def ScalableDAG_V2_nonlinear(model: nn.Module,
                      X: np.ndarray,log,B_true,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      lambda3: float = 0.,
                      lambda4: float = 0.,
                      max_iter: int = 100,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3):
    rho, alpha, h = 1.0, 0.0, np.inf
    X_latin = latin_hyper(X, n=20)
    for _ in tqdm(range(max_iter)):
        rho, alpha, h, result_dict = dual_ascent_step(model, X, lambda1, lambda2, lambda3, lambda4,
                                        rho, alpha, h, rho_max, X_latin)   
        log.step_update(result_dict)
        num_epoch = _ + 1
        if h <= h_tol or rho >= rho_max:
            break
        
    W_est = model.fc1_to_adj()
    print(W_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est, num_epoch


def main():
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    #LOGGING ----
    lambdas = []
    # check = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    # for lambda1 in [0.01]: 
    #     for lambda2 in [0.02]: 
    #         for lambda3 in [0.02]: 
    #             for lambda4 in [0.03]: 
    #                 lambdas.append([lambda1, lambda2, lambda3, lambda4])

    # 0.1 0.1 0.05 0.1

    max_average_tpr = 0.0 
    min_average_tpr = 1.0
    
    for indx, lambda_ in enumerate([0.01, 0.02, 0.02, 0.03]): 
        lambda1 = lambda_[0]
        lambda2 = lambda_[1]
        lambda3 = lambda_[2]
        lambda4 = lambda_[3]
        name = 'test_new' + str(indx)

        key_list = ['obj_func', 'sq_loss', 'orth', 'h_func', 'pen', 'l1', 'l2']
        # key_list = ['obj_func', 'sq_loss', 'vae_loss','orth', 'h_func']
        log = LogCausal(name, lambda_, key_list)

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
            k = 3
            #for v2_3------
            hidden_dims1 = 128
            hidden_dims2 = 64
            z_dim = 20
            dims = [d , 10, k]
            batch_size = n
            pivae_dims = [hidden_dims1, hidden_dims2, z_dim]
            #--------
            B_true = ut.simulate_dag(d, s0, graph_type)
            # np.savetxt(f'ScalableDAG_v2/W_true_{r}.csv', B_true, delimiter=',')

            X = ut.simulate_nonlinear_sem(B_true, n, sem_type)
            # np.savetxt(f'ScalableDAG_v2/X_{r}.csv', X, delimiter=',')

            
            # model = ScalableDAG_V2(dims=[d, 10, k], bias=True)
            model = ScalableDAG_V2_2(dims=[d , 10, k], bias=True)
            # model = ScalableDAG_V2_3(dims, pivae_dims, d, k, batch_size, bias=True )
            
            
            W_est, num_epoch = ScalableDAG_V2_nonlinear(model, X, log, B_true, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, lambda4=lambda4, max_iter=50, w_threshold=0.3) #ADD LOG 
            assert ut.is_dag(W_est)
            acc = ut.count_accuracy(B_true, W_est != 0)
            # np.savetxt(f'ScalableDAG_v2/W_est_{r}.csv', W_est, delimiter=',')
            print(W_est)
            print(B_true)
            print(acc)


            #LOGGING-----
            log.acc_update(acc,num_epoch) 

        # VISUALIZTION
        save_path = 'ScalableDAG_v2/visualization'
        vis = Visualization(log.log, save_path)
        vis.visualize()

        print("\n-----------AVERAGE SCORE---------------------")
        log.print_acc_average()
        if log.acc_average['tpr'] >= max_average_tpr: 
            max_average_tpr =  log.acc_average['tpr']
            max_lamda = lambda_
        if log.acc_average['tpr'] <= min_average_tpr: 
            min_average_tpr =  log.acc_average['tpr']
            min_lamda = lambda_
    
    print("\n-----------MAX SCORE---------------------")
    print(f'Max average: {max_average_tpr}, max lambda: {max_lamda}')
    print(f'Min average: {min_average_tpr}, min lambda: {min_lamda}')


if __name__ == '__main__':
    main()

