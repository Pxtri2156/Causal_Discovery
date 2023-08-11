
import  sys

sys.path.append("./")

from notears.locally_connected import LocallyConnected
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

# from torchsummary import summary

class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self, d):
        super(DiscriminatorNet, self).__init__()
        n_features = d
        n_out = 1
        
        self.hidden0 = nn.Sequential( 
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(256, n_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

def ones_target(size):
    # Tensor containing ones, with shape = size
    data = Variable(torch.ones(size, 1))
    return data

def zeros_target(size):
    # Tensor containing zeros, with shape = size
    data = Variable(torch.zeros(size, 1))
    return data
    
def train_discriminator(optimizer, real_data, fake_data):
    discriminator = DiscriminatorNet(real_data.size(1))
    loss = nn.BCELoss()
    N = real_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    
    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, ones_target(N))
    error_real.backward(retain_graph=True)

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward(retain_graph=True)
    
    # 1.3 Update weights with gradients
    optimizer.step()
    
    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake

class NotearsMLP(nn.Module):
    def __init__(self, dims, bias=True):
        super(NotearsMLP, self).__init__()
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
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(1, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)


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

    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        # print(f'before: {x.shape}\n{x[0]}')
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        # print(f'after: {x.shape}\n{x[0]}')
        # input('binh')
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        # x = x.squeeze(dim=2)  # [n, d]
        return x

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
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(1, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)

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

    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
            # x = torch.sigmoid(x) # [n, d, m2]
        # x = x.squeeze(dim=2)  # [n, d]
        return x

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

def count_accuracy(B_true, B_est):
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}

    
def squared_loss(output, target):
    n = target.shape[0]
    loss = 1 / n * torch.sum((output - target) ** 2)

    return loss

def latin_hyper(X, n=10):
    sampler = qmc.LatinHypercube(d=X.shape[1])
    sample = sampler.random(n) 
    l_bounds = list(np.min(X,axis=0))
    u_bounds = list(np.max(X,axis=0))
    X_latin = torch.from_numpy(qmc.scale(sample, l_bounds, u_bounds))
    return X_latin 

def orthogonality(X): 
    ortho = 0
    X = X.transpose(1,2)
    for i in range(X.shape[1]): 
        for j in range(i+1,X.shape[1]):
            if (i!=j): 
                ortho += torch.sum(abs(torch.sum(X[:,i,:] * X[:,j,:],dim=1)),dim=0)
    return float(ortho)

def dual_ascent_step(model, X, lambda1, lambda2, rho, alpha, h, rho_max, j, X_latin):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    optimizer = LBFGSBScipy(model.parameters())
    

    X_phi = X.copy()
    X_alpha = np.zeros(X.shape) 

    X_phi[:,j] = np.zeros(X.shape[0]) 
    X_alpha[:,j] = X[:,j]

    X_phi_torch = torch.from_numpy(X_phi)
    X_alpha_torch = torch.from_numpy(X_alpha)
    X_torch = torch.from_numpy(X)
    # while rho < rho_max:
    def closure():
        optimizer.zero_grad()

        X_phi_hat = model(X_phi_torch)
        X_alpha_hat = model(X_alpha_torch)[:,[j]]

        X_hat = torch.matmul(X_phi_hat,X_alpha_hat.transpose(1,2))
        X_hat = X_hat.squeeze(dim=2)

        loss = squared_loss(X_hat, X_torch)
        ortho = orthogonality(model(X_latin))
        h_val = model.h_func()

        penalty = 0.5 * rho * h_val * h_val + alpha * h_val
        l2_reg = 0.5 * lambda2 * model.l2_reg()
        l1_reg = lambda1 * model.fc1_l1_reg()
        primal_obj = loss + penalty + l1_reg  + l2_reg + ortho
        primal_obj.backward()
        return primal_obj
    optimizer.step(closure)  # NOTE: updates model in-place
    with torch.no_grad():
        h_new = model.h_func().item()

        X_phi_hat = model(X_phi_torch)
        X_alpha_hat = model(X_alpha_torch)[:,[j]]

        X_hat = torch.matmul(X_phi_hat,X_alpha_hat.transpose(1,2))
        X_hat = X_hat.squeeze(dim=2)

        loss = squared_loss(X_hat, X_torch)
        ortho = orthogonality(model(X_latin))
        h_val = model.h_func()

        penalty = 0.5 * rho * h_val * h_val + alpha * h_val
        l2_reg = 0.5 * lambda2 * model.l2_reg()
        l1_reg = lambda1 * model.fc1_l1_reg()
        primal_obj = loss + penalty  + l1_reg + l2_reg  + ortho
    # if h_new > 0.25 * h:
    #     rho *= 10
    # # else:
    # #     break

    if h_new == None:
        h_new = h
        print('there is something with h_new')
    alpha += rho * h_new
    return rho, alpha, h_new, primal_obj, loss, ortho, h_val


def notears_nonlinear(model: nn.Module,
                      X: np.ndarray,log,B_true,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      max_iter: int = 100,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3):
    rho, alpha, h = 1.0, 0.0, np.inf
    X_latin = latin_hyper(X, n=10)

    for _ in tqdm(range(max_iter)):
        for j in range(X.shape[1]): 
            rho, alpha, h, obj_func, loss, ortho, h_val = dual_ascent_step(model, X, lambda1, lambda2,
                                        rho, alpha, h, rho_max, j, X_latin)
            
        W_est = model.fc1_to_adj()
        W_est[np.abs(W_est) < w_threshold] = 0
        acc = count_accuracy(B_true, W_est != 0)

        log['obj_func'][log['random_seed'][-1]].append(obj_func)
        log['loss'][log['random_seed'][-1]].append(loss)
        log['ortho'][log['random_seed'][-1]].append(ortho)
        log['h_func'][log['random_seed'][-1]].append(h_val)
        log['score']['fdr'][log['random_seed'][-1]].append(acc['fdr']) 
        log['score']['fpr'][log['random_seed'][-1]].append(acc['fpr']) 
        log['score']['tpr'][log['random_seed'][-1]].append(acc['tpr']) 
        log['score']['shd'][log['random_seed'][-1]].append(acc['shd']) 
            # if h <= h_tol or rho >= rho_max:
            #     break

    log['num_epoch'][log['random_seed'][-1]] = max_iter


    W_est = model.fc1_to_adj()
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


def main():
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    #   LOGGING ---- 
    log = {}
    log['name'] = 'with_orth_8'
    log['random_seed'] = []
    log['obj_func'] = {}
    log['loss'] = {}
    log['ortho'] = {}
    log['h_func'] = {} 
    log['num_epoch'] = {} 
    log['score'] = {}
    log['score']['fdr']= {} 
    log['score']['fpr']= {}
    log['score']['tpr']= {}
    log['score']['shd']= {}

    #AVERAG ---- 
    fdr = 0.0 
    fpr = 0.0
    tpr = 0.0
    shd = 0

    random_seed = [702,210,1536,123]#[702,210,1536]
    for r in random_seed: #[2,3,5,6,9,15,19,28,2000,2001]
        print("\n-----------------------------------------")
        print('random seed:',r)
        ut.set_random_seed(r)

        #LOGGING----
        log['random_seed'].append(r)

        log['obj_func'][r] = []
        log['loss'][r] = []
        log['ortho'][r] = []
        log['h_func'][r] = []

        log['score']['fdr'][r]= []
        log['score']['fpr'][r]= []
        log['score']['tpr'][r]= []
        log['score']['shd'][r]= []

        n, d, s0, graph_type, sem_type = 200, 5, 9, 'ER', 'mim'
        B_true = ut.simulate_dag(d, s0, graph_type)
        np.savetxt('ScalableDAG_v2/W_true.csv', B_true, delimiter=',')

        X = ut.simulate_nonlinear_sem(B_true, n, sem_type)
        np.savetxt('ScalableDAG_v2/X.csv', X, delimiter=',')

        k = 3
        # model = ScalableDAG_V2(dims=[d, 10, k], bias=True)
        model = ScalableDAG_V2(dims=[d, 10, k], bias=True)

        
        W_est = notears_nonlinear(model, X, log, B_true, lambda1=0.01, lambda2=0.01, max_iter=50) #ADD LOG 
        assert ut.is_dag(W_est)
        acc = ut.count_accuracy(B_true, W_est != 0)
        np.savetxt('ScalableDAG_v2/W_est.csv', W_est, delimiter=',')
        print(acc)
        fdr += acc['fdr'] 
        fpr += acc['fpr'] 
        tpr += acc['tpr'] 
        shd += acc['shd'] 

    # VISUALIZTION
    save_path = 'ScalableDAG_v2/visualization'
    vis = Visualization(log, save_path)
    vis.visualize()

    print("\n-----------------------------------------")
    print(f'average: |FDR: {fdr/len(random_seed)} |TPR: {tpr/len(random_seed)} |FPR: {fpr/len(random_seed)} |SHD: {shd/len(random_seed)} |')

if __name__ == '__main__':
    main()
