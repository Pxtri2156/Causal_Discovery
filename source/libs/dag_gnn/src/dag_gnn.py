
''''
Main function for traininng DAG-GNN

'''


from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os
import datetime

# import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import math

import numpy as np
import sys
sys.path.append("./")
from libs.dag_gnn.src.dag_gnn_utils import *
from libs.dag_gnn.src.modules import *

def get_parser():

    parser = argparse.ArgumentParser()

    # -----------data parameters ------
    # configurations
    # parser.add_argument('--data_type', type=str, default= 'synthetic',
    #                     choices=['synthetic', 'discrete', 'real', 'discrete_benchmark'],
    #                     help='choosing which experiment to do.')
    # parser.add_argument('--data_filename', type=str, default= 'alarm',
    #                     help='data file name containing the discrete files.')
    # parser.add_argument('--data_dir', type=str, default= 'data/',
    #                     help='data file name containing the discrete files.')
    # parser.add_argument('--data_sample_size', type=int, default=5000,
    #                     help='the number of samples of data')
    parser.add_argument('--data_variable_size', type=int, default=10,
                        help='the number of variables in synthetic generated data')
    parser.add_argument('--config', type=str, default="test",
                        help='tried')
    # parser.add_argument('--graph_type', type=str, default='erdos-renyi',
    #                     help='the type of DAG graph by generation method')
    # parser.add_argument('--graph_degree', type=int, default=2,
    #                     help='the number of degree in generated DAG graph')
    # parser.add_argument('--graph_sem_type', type=str, default='linear-gauss',
    #                     help='the structure equation model (SEM) parameter type')
    # parser.add_argument('--graph_linear_type', type=str, default='nonlinear_2',
    #                     help='the synthetic data type: linear -> linear SEM, nonlinear_1 -> x=Acos(x+1)+z, nonlinear_2 -> x=2sin(A(x+0.5))+A(x+0.5)+z')
    parser.add_argument('--edge-types', type=int, default=2,
                        help='The number of edge types to infer.')
    parser.add_argument('--x_dims', type=int, default=1, #changed here
                        help='The number of input dimensions: default 1.')
    parser.add_argument('--z_dims', type=int, default=1,
                        help='The number of latent variable dimensions: default the same as variable size.')
    # parser.add_argument('--gt_path', type=str, default= '/workspace/tripx/projects/causality/notears/gt.xls',
    #                     help='data path of ground truth')
    
    # -----------training hyperparameters
    parser.add_argument('--optimizer', type = str, default = 'Adam',
                        help = 'the choice of optimizer used')
    parser.add_argument('--graph_threshold', type=  float, default = 0.3,  # 0.3 is good, 0.2 is error prune
                        help = 'threshold for learned adjacency matrix binarization')
    parser.add_argument('--tau_A', type = float, default=0.0,
                        help='coefficient for L-1 norm of A.')
    parser.add_argument('--lambda_A',  type = float, default= 0.,
                        help='coefficient for DAG constraint h(A).')
    parser.add_argument('--c_A',  type = float, default= 1,
                        help='coefficient for absolute value h(A).')
    parser.add_argument('--use_A_connect_loss',  type = int, default= 0,
                        help='flag to use A connect loss')
    parser.add_argument('--use_A_positiver_loss', type = int, default = 0,
                        help = 'flag to enforce A must have positive values')


    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default= 300,
                        help='Number of epochs to train.')
    parser.add_argument('--batch-size', type=int, default = 100, # note: should be divisible by sample size, otherwise throw an error
                        help='Number of samples per batch.')
    parser.add_argument('--lr', type=float, default=3e-3,  # basline rate = 1e-3
                        help='Initial learning rate.')
    parser.add_argument('--encoder-hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--decoder-hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--temp', type=float, default=0.5,
                        help='Temperature for Gumbel softmax.')
    parser.add_argument('--k_max_iter', type = int, default = 1e2,
                        help ='the max iteration number for searching lambda and c')

    parser.add_argument('--encoder', type=str, default='mlp',
                        help='Type of path encoder model (mlp, or sem).')
    parser.add_argument('--decoder', type=str, default='mlp',
                        help='Type of decoder model (mlp, or sim).')
    parser.add_argument('--no-factor', action='store_true', default=False,
                        help='Disables factor graph model.')
    parser.add_argument('--suffix', type=str, default='_springs5',
                        help='Suffix for training data (e.g. "_charged".')
    parser.add_argument('--encoder-dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--decoder-dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--save-folder', type=str, default='logs',
                        help='Where to save the trained model, leave empty to not save anything.')
    parser.add_argument('--load-folder', type=str, default='',
                        help='Where to load the trained model if finetunning. ' +
                            'Leave empty to train from scratch')


    parser.add_argument('--h_tol', type=float, default = 1e-8,
                        help='the tolerance of error of h(A) to zero')
    parser.add_argument('--prediction-steps', type=int, default=10, metavar='N',
                        help='Num steps to predict before re-using teacher forcing.')
    parser.add_argument('--lr-decay', type=int, default=200,
                        help='After how epochs to decay LR by a factor of gamma.')
    parser.add_argument('--gamma', type=float, default= 1.0,
                        help='LR decay factor.')
    parser.add_argument('--skip-first', action='store_true', default=False,
                        help='Skip first edge type in decoder, i.e. it represents no-edge.')
    parser.add_argument('--var', type=float, default=5e-5,
                        help='Output variance.')
    parser.add_argument('--hard', action='store_true', default=False,
                        help='Uses discrete samples in training forward pass.')
    parser.add_argument('--prior', action='store_true', default=False,
                        help='Whether to use sparsity prior.')
    parser.add_argument('--dynamic-graph', action='store_true', default=False,
                        help='Whether test with dynamically re-computed graph.')

    return parser.parse_args()

def choose_encoder(encoder_type, adj_A, args):
    if encoder_type == 'mlp':
        encoder = MLPEncoder(args.data_variable_size * args.x_dims, args.x_dims, args.encoder_hidden,
                            int(args.z_dims), adj_A,
                            batch_size = args.batch_size,
                            do_prob = args.encoder_dropout, factor = args.factor).double()
    elif encoder_type == 'sem':
        encoder = SEMEncoder(args.data_variable_size * args.x_dims, args.encoder_hidden,
                            int(args.z_dims), adj_A,
                            batch_size = args.batch_size,
                            do_prob = args.encoder_dropout, factor = args.factor).double()

    return encoder

def choose_decoder(decoder_type, encoder, args):
    if decoder_type == 'mlp':
        decoder = MLPDecoder(args.data_variable_size * args.x_dims,
                            args.z_dims, args.x_dims, encoder,
                            data_variable_size = args.data_variable_size,
                            batch_size = args.batch_size,
                            n_hid=args.decoder_hidden,
                            do_prob=args.decoder_dropout).double()
    elif decoder_type == 'sem':
        decoder = SEMDecoder(args.data_variable_size * args.x_dims,
                            args.z_dims, 2, encoder,
                            data_variable_size = args.data_variable_size,
                            batch_size = args.batch_size,
                            n_hid=args.decoder_hidden,
                            do_prob=args.decoder_dropout).double()
    return decoder

def load_encode_decode(encoder, decoder, load_folder):
    encoder_file = os.path.join(load_folder, 'encoder.pt')
    encoder.load_state_dict(torch.load(encoder_file))
    decoder_file = os.path.join(load_folder, 'decoder.pt')
    decoder.load_state_dict(torch.load(decoder_file))

    return encoder, decoder

def set_optimizer(optimizer_type, encoder, decoder, lr):
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),lr=lr)
    elif optimizer_type == 'LBFGS':
        optimizer = optim.LBFGS(list(encoder.parameters()) + list(decoder.parameters()),
                            lr=lr)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(list(encoder.parameters()) + list(decoder.parameters()),
                            lr=lr)

    return optimizer

def _h_A(A, m):
    expm_A = matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A

def stau(w, tau):
    prox_plus = torch.nn.Threshold(0.,0.)
    w1 = prox_plus(torch.abs(w)-tau)
    return torch.sign(w)*w1

def update_optimizer(optimizer, original_lr, c_A):
    '''related LR to c_A, whenever c_A gets big, reduce LR proportionally'''
    MAX_LR = 1e-2
    MIN_LR = 1e-4

    estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
    if estimated_lr > MAX_LR:
        lr = MAX_LR
    elif estimated_lr < MIN_LR:
        lr = MIN_LR
    else:
        lr = estimated_lr

    # set LR
    for parame_group in optimizer.param_groups:
        parame_group['lr'] = lr

    return optimizer, lr


def train(epoch, best_val_loss, ground_truth_G, lambda_A, c_A, optimizer, \
    encoder, decoder, scheduler, args, train_loader, rel_rec, rel_send, \
        encoder_file, decoder_file, log):
    
    t = time.time()
    nll_train = []
    kl_train = []
    mse_train = []
    shd_trian = []
    encoder.train()
    decoder.train()
    scheduler.step()

    # update optimizer
    optimizer, lr = update_optimizer(optimizer, args.lr, c_A)

    for batch_idx, (data, relations) in enumerate(train_loader):
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = Variable(data).double(), Variable(relations).double()
        # reshape data
        relations = relations.unsqueeze(2)

        optimizer.zero_grad()
        enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(data, rel_rec, rel_send)  # logits is of size: [num_sims, z_dims]
        edges = logits
        dec_x, output, adj_A_tilt_decoder = decoder(data, edges, args.data_variable_size * args.x_dims, rel_rec, rel_send, origin_A, adj_A_tilt_encoder, Wa)

        if torch.sum(output != output):
            print('nan error\n')
        target = data
        preds = output
        variance = 0.
        # reconstruction accuracy loss
        loss_nll = nll_gaussian(preds, target, variance)
        
        # KL loss
        loss_kl = kl_gaussian_sem(logits)
        
        # ELBO loss:
        loss = loss_kl + loss_nll
        
        # add A loss
        one_adj_A = origin_A # torch.mean(adj_A_tilt_decoder, dim =0)
        sparse_loss = args.tau_A * torch.sum(torch.abs(one_adj_A))

        # other loss term
        if args.use_A_connect_loss:
            connect_gap = A_connect_loss(one_adj_A, args.graph_threshold, z_gap)
            loss += lambda_A * connect_gap + 0.5 * c_A * connect_gap * connect_gap

        if args.use_A_positiver_loss:
            positive_gap = A_positive_loss(one_adj_A, z_positive)
            loss += .1 * (lambda_A * positive_gap + 0.5 * c_A * positive_gap * positive_gap)

        # compute h(A)
        h_A = _h_A(origin_A, args.data_variable_size)
        loss += lambda_A * h_A + 0.5 * c_A * h_A * h_A + 100. * torch.trace(origin_A*origin_A) + sparse_loss #+  0.01 * torch.sum(variance * variance)

        loss.backward()
        loss = optimizer.step()
        myA.data = stau(myA.data, args.tau_A*lr)

        if torch.sum(origin_A != origin_A):
            print('nan error\n')

        # compute metrics
        graph = origin_A.data.clone().cpu().numpy()
        graph[np.abs(graph) < args.graph_threshold] = 0
        # input("Stop")
        fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))

        mse_train.append(F.mse_loss(preds, target).item())
        nll_train.append(loss_nll.item())
        kl_train.append(loss_kl.item())
        shd_trian.append(shd)

    print(h_A.item())
    nll_val = []

    print('Epoch: {:04d}'.format(epoch),
        'nll_train: {:.10f}'.format(np.mean(nll_train)),
        'kl_train: {:.10f}'.format(np.mean(kl_train)),
        'ELBO_loss: {:.10f}'.format(np.mean(kl_train)  + np.mean(nll_train)),
        'mse_train: {:.10f}'.format(np.mean(mse_train)),
        'shd_trian: {:.10f}'.format(np.mean(shd_trian)),
        'time: {:.4f}s'.format(time.time() - t))
    if args.save_folder and np.mean(nll_val) < best_val_loss:
        torch.save(encoder.state_dict(), encoder_file)
        torch.save(decoder.state_dict(), decoder_file)
        print('Best model so far, saving...')
        print('Epoch: {:04d}'.format(epoch),
            'nll_train: {:.10f}'.format(np.mean(nll_train)),
            'kl_train: {:.10f}'.format(np.mean(kl_train)),
            'ELBO_loss: {:.10f}'.format(np.mean(kl_train)  + np.mean(nll_train)),
            'mse_train: {:.10f}'.format(np.mean(mse_train)),
            'shd_trian: {:.10f}'.format(np.mean(shd_trian)),
            'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()

    if 'graph' not in vars():
        print('error on assign')


    return np.mean(np.mean(kl_train)  + np.mean(nll_train)), np.mean(nll_train), np.mean(mse_train), graph, origin_A

def set_prior():
    prior = np.array([0.91, 0.03, 0.03, 0.03])  # hard coded for now
    print("Using prior")
    # print(prior)
    log_prior = torch.DoubleTensor(np.log(prior))
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = Variable(log_prior)
    return log_prior
    
def dag_gnn(X, gt):
    if len(X.shape) != 3:
        print('Extend')
        X = np.expand_dims(X, axis=2)
        gt = pd.DataFrame(gt)
    # print("X shape: ", X.shape)
    # print("gt shape: ", gt.shape)
    # print("gt: ", gt.shape)

    args = get_parser()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.factor = not args.no_factor
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    if args.dynamic_graph:
        print("Testing with dynamically re-computed graph.")

    # Save model and meta-data. Always saves in a new sub-folder.
    if args.save_folder:
        exp_counter = 0
        now = datetime.datetime.now()
        timestamp = now.isoformat()
        save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
        # safe_name = save_folder.text.replace('/', '_')
        os.makedirs(save_folder)
        meta_file = os.path.join(save_folder, 'metadata.pkl')
        encoder_file = os.path.join(save_folder, 'encoder.pt')
        decoder_file = os.path.join(save_folder, 'decoder.pt')

        log_file = os.path.join(save_folder, 'log.txt')
        log = open(log_file, 'w')

        pickle.dump({'args': args}, open(meta_file, "wb"))
    else:
        print("WARNING: No save_folder provided!" +
            "Testing (within this script) will throw an error.")
    # ================================================
    # get data: experiments = {synthetic SEM, ALARM}
    # ================================================
    args.data_variable_size=gt.shape[0]
    train_loader, ground_truth_G = transform_data(X, gt, args, batch_size=1000)
    
    # Generate off-diagonal interaction graph
    off_diag = np.ones([args.data_variable_size, args.data_variable_size]) - np.eye(args.data_variable_size)
    rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float64)
    rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float64)
    rel_rec = torch.DoubleTensor(rel_rec)
    rel_send = torch.DoubleTensor(rel_send)

    # add adjacency matrix A
    adj_A = np.zeros((args.data_variable_size, args.data_variable_size))
    # chosse encoder and decoder
    encoder = choose_encoder(args.encoder, adj_A, args)
    decoder = choose_decoder(args.decoder, encoder, args)
    ## Load encoder and decoder
    if args.load_folder:
        encoder, decoder = load_encode_decode(encoder, decoder, args.load_folder)
        args.save_folder = False
    #===================================
    # set up training parameters
    #===================================
    optimizer = set_optimizer(args.optimizer, encoder, decoder, args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                    gamma=args.gamma)
    # Linear indices of an upper triangular mx, used for acc calculation
    triu_indices = get_triu_offdiag_indices(args.data_variable_size)
    tril_indices = get_tril_offdiag_indices(args.data_variable_size)

    if args.prior:
        log_prior = set_prior()            
        if args.cuda:
            log_prior = log_prior.cuda()

    rel_rec = Variable(rel_rec)
    rel_send = Variable(rel_send)
    
    best_ELBO_loss = np.inf
    best_NLL_loss = np.inf
    best_MSE_loss = np.inf
    best_epoch = 0
    best_ELBO_graph = []
    best_NLL_graph = []
    best_MSE_graph = []
    # optimizer step on hyparameters
    c_A = args.c_A
    lambda_A = args.lambda_A
    h_A_new = torch.tensor(1.)
    h_tol = args.h_tol
    h_A_old = np.inf
    
    if args.cuda:
        h_A_new.cuda()
        encoder.cuda()
        decoder.cuda()
        rel_rec = rel_rec.cuda()
        rel_send = rel_send.cuda()
        triu_indices = triu_indices.cuda()
        tril_indices = tril_indices.cuda()
    try:
        for step_k in range(int(args.k_max_iter)):
            while c_A < 1e+20:
                for epoch in range(args.epochs):
                    ELBO_loss, NLL_loss, MSE_loss, graph, origin_A  = train(epoch, best_ELBO_loss, ground_truth_G, lambda_A, c_A, optimizer, \
                                                                            encoder, decoder, scheduler, args, train_loader, rel_rec, rel_send, \
                                                                            encoder_file, decoder_file, log)
                  
                    if ELBO_loss < best_ELBO_loss:
                        best_ELBO_loss = ELBO_loss
                        best_epoch = epoch
                        best_ELBO_graph = graph

                    if NLL_loss < best_NLL_loss:
                        best_NLL_loss = NLL_loss
                        best_epoch = epoch
                        best_NLL_graph = graph

                    if MSE_loss < best_MSE_loss:
                        best_MSE_loss = MSE_loss
                        best_epoch = epoch
                        best_MSE_graph = graph
                        
                print("Optimization Finished!")
                print("Best Epoch: {:04d}".format(best_epoch))
                if ELBO_loss > 2 * best_ELBO_loss:
                    break
                # update parameters
                A_new = origin_A.data.clone()
                h_A_new = _h_A(A_new, args.data_variable_size)
                if h_A_new.item() > 0.25 * h_A_old:
                    c_A*=10
                else:
                    break
                # update parameters
                # h_A, adj_A are computed in loss anyway, so no need to store
                break
            h_A_old = h_A_new.item()
            lambda_A += c_A * h_A_new.item()
            if h_A_new.item() <= h_tol:
                break
            break
        if args.save_folder:
            print("Best Epoch: {:04d}".format(best_epoch), file=log)
            log.flush()
                
        # test()
        return best_ELBO_graph


    except KeyboardInterrupt:
        # print the best anway
        print(best_ELBO_graph)
        print(nx.to_numpy_array(ground_truth_G))
        fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_ELBO_graph))
        print('Best ELBO Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)
        
    if log is not None:
        print(save_folder)
        log.close()


def train_dag_gnn(X, gt):
    args = get_parser()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.factor = not args.no_factor
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    if args.dynamic_graph:
        print("Testing with dynamically re-computed graph.")

    # Save model and meta-data. Always saves in a new sub-folder.
    if args.save_folder:
        exp_counter = 0
        now = datetime.datetime.now()
        timestamp = now.isoformat()
        save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
        # safe_name = save_folder.text.replace('/', '_')
        os.makedirs(save_folder)
        meta_file = os.path.join(save_folder, 'metadata.pkl')
        encoder_file = os.path.join(save_folder, 'encoder.pt')
        decoder_file = os.path.join(save_folder, 'decoder.pt')

        log_file = os.path.join(save_folder, 'log.txt')
        log = open(log_file, 'w')

        pickle.dump({'args': args}, open(meta_file, "wb"))
    else:
        print("WARNING: No save_folder provided!" +
            "Testing (within this script) will throw an error.")
    # ================================================
    # get data: experiments = {synthetic SEM, ALARM}
    # ================================================
    train_loader, ground_truth_G = transform_data(X, gt, args, batch_size=1000)
    
    # Generate off-diagonal interaction graph
    off_diag = np.ones([args.data_variable_size, args.data_variable_size]) - np.eye(args.data_variable_size)
    rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float64)
    rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float64)
    rel_rec = torch.DoubleTensor(rel_rec)
    rel_send = torch.DoubleTensor(rel_send)

    # add adjacency matrix A
    adj_A = np.zeros((args.data_variable_size, args.data_variable_size))
    # chosse encoder and decoder
    encoder = choose_encoder(args.encoder, adj_A, args)
    decoder = choose_decoder(args.decoder, encoder, args)
    ## Load encoder and decoder
    if args.load_folder:
        encoder, decoder = load_encode_decode(encoder, decoder, args.load_folder)
        args.save_folder = False
    #===================================
    # set up training parameters
    #===================================
    optimizer = set_optimizer(args.optimizer, encoder, decoder, args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                    gamma=args.gamma)
    # Linear indices of an upper triangular mx, used for acc calculation
    triu_indices = get_triu_offdiag_indices(args.data_variable_size)
    tril_indices = get_tril_offdiag_indices(args.data_variable_size)

    if args.prior:
        log_prior = set_prior()            
        if args.cuda:
            log_prior = log_prior.cuda()

    rel_rec = Variable(rel_rec)
    rel_send = Variable(rel_send)
    
    best_ELBO_loss = np.inf
    best_NLL_loss = np.inf
    best_MSE_loss = np.inf
    best_epoch = 0
    best_ELBO_graph = []
    best_NLL_graph = []
    best_MSE_graph = []
    # optimizer step on hyparameters
    c_A = args.c_A
    lambda_A = args.lambda_A
    h_A_new = torch.tensor(1.)
    h_tol = args.h_tol
    h_A_old = np.inf
    
    if args.cuda:
        h_A_new.cuda()
        encoder.cuda()
        decoder.cuda()
        rel_rec = rel_rec.cuda()
        rel_send = rel_send.cuda()
        triu_indices = triu_indices.cuda()
        tril_indices = tril_indices.cuda()
    try:
        for step_k in range(int(args.k_max_iter)):
            while c_A < 1e+20:
                for epoch in range(args.epochs):
                    ELBO_loss, NLL_loss, MSE_loss, graph, origin_A  = train(epoch, best_ELBO_loss, ground_truth_G, lambda_A, c_A, optimizer, \
                                                                            encoder, decoder, scheduler, args, train_loader, rel_rec, rel_send, \
                                                                            encoder_file, decoder_file, log)
                  
                    if ELBO_loss < best_ELBO_loss:
                        best_ELBO_loss = ELBO_loss
                        best_epoch = epoch
                        best_ELBO_graph = graph

                    if NLL_loss < best_NLL_loss:
                        best_NLL_loss = NLL_loss
                        best_epoch = epoch
                        best_NLL_graph = graph

                    if MSE_loss < best_MSE_loss:
                        best_MSE_loss = MSE_loss
                        best_epoch = epoch
                        best_MSE_graph = graph
                        
                print("Optimization Finished!")
                print("Best Epoch: {:04d}".format(best_epoch))
                if ELBO_loss > 2 * best_ELBO_loss:
                    break
                # update parameters
                A_new = origin_A.data.clone()
                h_A_new = _h_A(A_new, args.data_variable_size)
                if h_A_new.item() > 0.25 * h_A_old:
                    c_A*=10
                else:
                    break
                # update parameters
                # h_A, adj_A are computed in loss anyway, so no need to store
                break
            h_A_old = h_A_new.item()
            lambda_A += c_A * h_A_new.item()
            if h_A_new.item() <= h_tol:
                break
            break
        if args.save_folder:
            print("Best Epoch: {:04d}".format(best_epoch), file=log)
            log.flush()
                
        # test()
        return best_ELBO_graph


    except KeyboardInterrupt:
        # print the best anway
        print(best_ELBO_graph)
        print(nx.to_numpy_array(ground_truth_G))
        fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_ELBO_graph))
        print('Best ELBO Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)
        
    if log is not None:
        print(save_folder)
        log.close()

def main():
    X = read_data_csv("/dataset/Bayesian_Data/ASIA/ASIA_DATA.csv")
    # Read ground truth 
    gt = pd.read_csv("/dataset/Bayesian_Data/ASIA/DAGtrue_ASIA_bi.csv", header=None)
    # gt = nx.from_pandas_adjacency(gt)
    W_est = dag_gnn(X, gt)

if __name__ == '__main__':
    main()