import logging

import igraph as ig
import networkx as nx
import numpy as np

import os
import pandas as pd 
from utils.utils import is_dag


class ProteinSignalDataset:
    """Generate synthetic data.

    Key instance variables:
        X (numpy.ndarray): [n, d] data matrix.
        B (numpy.ndarray): [d, d] weighted adjacency matrix of DAG.
        B_bin (numpy.ndarray): [d, d] binary adjacency matrix of DAG.

    Code modified from:
        https://github.com/xunzheng/notears/blob/master/notears/utils.py
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, n, d, graph_type, degree, noise_type, B_scale, seed=1):
        """Initialize self.

        Args:
            n (int): Number of samples.
            d (int): Number of nodes.
            graph_type ('ER' or 'SF'): Type of graph.
            degree (int): Degree of graph.
            noise_type ('gaussian_ev', 'gaussian_nv', 'exponential', 'gumbel'): Type of noise.
            B_scale (float): Scaling factor for range of B.
            seed (int): Random seed. Default: 1.
        """
        self.n = n
        self.d = d
        self.graph_type = graph_type
        self.degree = degree
        self.noise_type = noise_type
        self.B_ranges = ((B_scale * -2.0, B_scale * -0.5),
                         (B_scale * 0.5, B_scale * 2.0))
        self.rs = np.random.RandomState(seed)    # Reproducibility

        self._setup()
        self._logger.debug("Finished setting up dataset class.")

    def _setup(self):
        """Generate B_bin, B and X."""
        # self.B_bin = ProteinSignalDataset.simulate_random_dag(self.d, self.degree,
        #                                                   self.graph_type, self.rs)
        self.B_bin = ProteinSignalDataset.load_ground_truth()
        # print(f'B_bin: {self.B_bin}')
        self.B = ProteinSignalDataset.simulate_weight(self.B_bin, self.B_ranges, self.rs)
        # print(f'B: {self.B}')
        self.X = ProteinSignalDataset.simulate_linear_sem(self.B, self.n, self.noise_type, self.rs)
        assert is_dag(self.B)

    @staticmethod
    def simulate_er_dag(d, degree, rs=np.random.RandomState(1)):
        """Simulate ER DAG using NetworkX package.

        Args:
            d (int): Number of nodes.
            degree (int): Degree of graph.
            rs (numpy.random.RandomState): Random number generator.
                Default: np.random.RandomState(1).

        Returns:
            numpy.ndarray: [d, d] binary adjacency matrix of DAG.
        """
        def _get_acyclic_graph(B_und):
            return np.tril(B_und, k=-1)

        def _graph_to_adjmat(G):
            return nx.to_numpy_matrix(G)

        p = float(degree) / (d - 1)
        G_und = nx.generators.erdos_renyi_graph(n=d, p=p, seed=rs)
        B_und_bin = _graph_to_adjmat(G_und)    # Undirected
        B_bin = _get_acyclic_graph(B_und_bin)
        return B_bin

    @staticmethod
    def simulate_sf_dag(d, degree):
        """Simulate ER DAG using igraph package.

        Args:
            d (int): Number of nodes.
            degree (int): Degree of graph.

        Returns:
            numpy.ndarray: [d, d] binary adjacency matrix of DAG.
        """
        def _graph_to_adjmat(G):
            return np.array(G.get_adjacency().data)

        m = int(round(degree / 2))
        # igraph does not allow passing RandomState object
        G = ig.Graph.Barabasi(n=d, m=m, directed=True)
        B_bin = np.array(G.get_adjacency().data)
        return B_bin

    @staticmethod
    def simulate_random_dag(d, degree, graph_type, rs=np.random.RandomState(1)):
        """Simulate random DAG.

        Args:
            d (int): Number of nodes.
            degree (int): Degree of graph.
            graph_type ('ER' or 'SF'): Type of graph.
            rs (numpy.random.RandomState): Random number generator.
                Default: np.random.RandomState(1).

        Returns:
            numpy.ndarray: [d, d] binary adjacency matrix of DAG.
        """
        def _random_permutation(B_bin):
            # np.random.permutation permutes first axis only
            P = rs.permutation(np.eye(B_bin.shape[0]))
            return P.T @ B_bin @ P

        if graph_type == 'ER':
            B_bin = ProteinSignalDataset.simulate_er_dag(d, degree, rs)
        elif graph_type == 'SF':
            B_bin = ProteinSignalDataset.simulate_sf_dag(d, degree)
        else:
            raise ValueError("Unknown graph type.")
        return _random_permutation(B_bin)

    @staticmethod
    def simulate_weight(B_bin, B_ranges, rs=np.random.RandomState(1)):
        """Simulate the weights of B_bin.

        Args:
            B_bin (numpy.ndarray): [d, d] binary adjacency matrix of DAG.
            B_ranges (tuple): Disjoint weight ranges.
            rs (numpy.random.RandomState): Random number generator.
                Default: np.random.RandomState(1).

        Returns:
            numpy.ndarray: [d, d] weighted adjacency matrix of DAG.
        """
        B = np.zeros(B_bin.shape)
        S = rs.randint(len(B_ranges), size=B.shape)  # Which range
        for i, (low, high) in enumerate(B_ranges):
            U = rs.uniform(low=low, high=high, size=B.shape)
            B += B_bin * (S == i) * U
        return B

    @staticmethod
    def simulate_linear_sem(B, n, noise_type, rs=np.random.RandomState(1)):
        """Simulate samples from linear SEM with specified type of noise.

        Args:
            B (numpy.ndarray): [d, d] weighted adjacency matrix of DAG.
            n (int): Number of samples.
            noise_type ('gaussian_ev', 'gaussian_nv', 'exponential', 'gumbel'): Type of noise.
            rs (numpy.random.RandomState): Random number generator.
                Default: np.random.RandomState(1).

        Returns:
            numpy.ndarray: [n, d] data matrix.
        """
        def _simulate_single_equation(X, B_i):
            """Simulate samples from linear SEM for the i-th node.

            Args:
                X (numpy.ndarray): [n, number of parents] data matrix.
                B_i (numpy.ndarray): [d,] weighted vector for the i-th node.

            Returns:
                numpy.ndarray: [n,] data matrix.
            """
            if noise_type == 'gaussian_ev':
                # Gaussian noise with equal variances
                N_i = rs.normal(scale=1.0, size=n)
            elif noise_type == 'gaussian_nv':
                # Gaussian noise with non-equal variances
                scale = rs.uniform(low=1.0, high=2.0)
                N_i = rs.normal(scale=scale, size=n)
            elif noise_type == 'exponential':
                # Exponential noise
                N_i = rs.exponential(scale=1.0, size=n)
            elif noise_type == 'gumbel':
                # Gumbel noise
                N_i = rs.gumbel(scale=1.0, size=n)
            else:
                raise ValueError("Unknown noise type.")
            return X @ B_i + N_i
        

        ############
        dataset_path = '/dataset/Causal_Protein_Signaling/DataFiles/'
        ############
        d = B.shape[0]
        X = ProteinSignalDataset.load_dataset(dataset_path)
        G = nx.DiGraph(B)
        ordered_vertices = list(nx.topological_sort(G))
        assert len(ordered_vertices) == d
        for i in ordered_vertices:
            parents = list(G.predecessors(i))
            X[:, i] = _simulate_single_equation(X[:, parents], B[parents, i])
        return X

    def load_dataset(dataset_path):
        X = pd.DataFrame(columns=['praf','pmek','plcg','pip2','pip3','p44/42','pakts473','pka','pkc','p38','pjnk'])
        dirs = os.listdir(dataset_path)
        for file in dirs:
            if '.xls' in file and file != 'gt.xls': 
                file = dataset_path+file
                print(file)
                df = pd.read_excel(file)
                df.columns = [x.lower() for x in df.columns]
                X = X.append(df)
        X=X.to_numpy()  
        return X   

    def load_ground_truth(): 
        path = '/dataset/Causal_Protein_Signaling/DataFiles/gt.xls'
        B_bin = pd.read_excel(path,header=None)
        B_bin = B_bin.to_numpy()
        B_bin = np.asfarray(B_bin)
        # print(f'excel{B_bin}')
        # if graph_type == 'ER':
        #     # B_bin = nx.from_numpy_array(B_bin)
        #     # B_bin = nx.to_numpy_matrix(B_bin) 
        #     # B_bin = np.tril(B_bin, k=-1)
        #     B_bin = B_bin
        # elif graph_type == 'SF':
        #     B_bin = B_bin
        # else:
        #     raise ValueError("Unknown graph type.")
        return B_bin


if __name__ == '__main__':
    # n, d = 1000, 20
    n, d = 11672, 11
    graph_type, degree = 'ER', 4    # ER2 graph
    B_scale = 1.0
    noise_type = 'gaussian_ev'

    dataset = ProteinSignalDataset(n, d, graph_type, degree,noise_type, B_scale, seed=1)
    print("dataset.X.shape: {}".format(dataset.X.shape))
    print("dataset.B.shape: {}".format(dataset.B.shape))
    print("dataset.B_bin.shape: {}".format(dataset.B.shape))
