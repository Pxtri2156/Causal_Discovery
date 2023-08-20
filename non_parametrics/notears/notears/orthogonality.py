from notears.lbfgsb_scipy import LBFGSBScipy
from notears.trace_expm import trace_expm
import math
import torch
import torch.nn as nn
from scipy.stats import qmc
import numpy as np
import notears.utils as ut

class LocallyConnected(nn.Module):
    """Local linear layer, i.e. Conv1dLocal() with filter size 1.

    Args:
        num_linear: num of local linear layers, i.e.
        in_features: m1
        out_features: m2
        bias: whether to include bias or not

    Shape:
        - Input: [n, d, m1]
        - Output: [n, d, m2]

    Attributes:
        weight: [d, m1, m2]
        bias: [d, m2]
    """

    def __init__(self, num_linear, input_features, output_features, bias=True):
        super(LocallyConnected, self).__init__()
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(num_linear,
                                                input_features,
                                                output_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_linear, output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        k = 1.0 / self.input_features
        bound = math.sqrt(k)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor):
        # [n, d, 1, m2] = [n, d, 1, m1] @ [1, d, m1, m2]
        # print("input.unsqueeze(dim=2): ", input.unsqueeze(dim=2).shape)
        # print("self.weight.unsqueeze(dim=0): ", self.weight.unsqueeze(dim=0).shape)

        
        out = torch.matmul(input, self.weight.squeeze(dim=0))
        out = out.squeeze(dim=-2)
        if self.bias is not None:
            # [n, d, m2] += [d, m2]
            out += self.bias
        return out

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'num_linear={}, in_features={}, out_features={}, bias={}'.format(
            self.num_linear, self.in_features, self.out_features,
            self.bias is not None
        )

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
        # x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
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

def latin_hyper(X, n=10):
    sampler = qmc.LatinHypercube(d=X.shape[1])
    sample = sampler.random(n) 
    l_bounds = list(np.min(X,axis=0))
    u_bounds = list(np.max(X,axis=0))
    X_latin = torch.from_numpy(qmc.scale(sample, l_bounds, u_bounds))
    return X_latin 

def orthogonality(X): 
    ortho = 0.0
    X = X.transpose(0,1)
    for i in range(X.shape[0]): 
        for j in range(i+1,X.shape[0]):
            if (i!=j): 
                ortho += float(abs(X[[i],:] @ X[[j],:].transpose(0,1)))
                # torch.dot
                # print(X[[i],:].shape)
                # print(X[[j],:].shape)
                # print(torch.sum(X[[i],:] * X[[j],:]))
                # print(X[[i],:] @ X[[j],:].transpose(0,1))
                # print(float(abs(X[[i],:] @ X[[j],:].transpose(0,1))))

                # input('Binh')
    return ortho

def main(): 
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    ut.set_random_seed(123)

    n, d, s0, graph_type, sem_type = 200, 5, 9, 'ER', 'mim'
    B_true = ut.simulate_dag(d, s0, graph_type)
    np.savetxt('ScalableDAG_v2/W_true.csv', B_true, delimiter=',')

    X = ut.simulate_nonlinear_sem(B_true, n, sem_type)
    np.savetxt('ScalableDAG_v2/X.csv', X, delimiter=',')

    X_latin = latin_hyper(X, n=10)

    k = 3
    model = ScalableDAG_V2(dims=[d, 10, k], bias=True)
    ortho = orthogonality(model(X_latin))

    print(ortho)
    return 

if __name__ == '__main__':
    main()