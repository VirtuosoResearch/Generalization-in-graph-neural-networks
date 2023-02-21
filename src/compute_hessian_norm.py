import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset
from sklearn.metrics import roc_auc_score
from torch_geometric.data import DataLoader
from tqdm import tqdm

from loader import MoleculeDataset
from model import GNN, GNN_graphpred
from splitters import scaffold_split
from utils.bypass_bn import disable_running_stats, enable_running_stats
from utils.sam import SAM
from utils.nsm import NSM
from utils.constraint import deep_copy

def get_weights(model):
    layers = []; names = []
    for name, module in model.named_modules():
        if type(module) == torch.nn.Linear or type(module) == torch.nn.Embedding:
            names.append(name)
            layers.append(module.weight)
    return layers, names

def compute_hessians_quantity(model, loss, state_dict = None):
    weights, names = get_weights(model)
    model.zero_grad()
    gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)
    
    vs = []
    for i, weight in enumerate(weights):
        name = names[i]
        if "graph_pred_linear" in name:
            v = weight.detach().clone()
        else:
            v = weight.detach().clone() - state_dict[name+".weight"]
        vs.append(v)

    model.zero_grad()    
    Hvs = torch.autograd.grad(gradients, weights, grad_outputs=vs, retain_graph=True)

    layer_hessian_quantities = [torch.sum(Hv*v).cpu().item() for (Hv, v) in zip(Hvs, vs)]
    
    return np.array(layer_hessian_quantities)


def compute_hessians_trace(model, loss, maxIter=100, tol=1e-3):
    # Get parameters and gradients of corresponding layer
    weights, name = get_weights(model)
    model.zero_grad()
    gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)

    layer_traces = []
    trace_vhv = []
    trace = 0.

    # Start Iterations
    for _ in range(maxIter):
        vs = [torch.randint_like(weight, high=2) for weight in weights]
            
        # generate Rademacher random variables
        for v in vs:
            v[v == 0] = -1

        model.zero_grad()  
        Hvs = torch.autograd.grad(gradients, weights, grad_outputs=vs, retain_graph=True)
        tmp_layer_traces = np.array([torch.sum(Hv*v).cpu().item() for (Hv, v) in zip(Hvs, vs)])

        layer_traces.append(tmp_layer_traces)
        trace_vhv.append(np.sum(tmp_layer_traces))

        if abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6) < tol:
            break
        else:
            trace = np.mean(trace_vhv)
    return np.mean(np.array(layer_traces), axis=0)

criterion = nn.BCEWithLogitsLoss(reduction = "none")

def main(args):
    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

    print(dataset)
    
    if args.split == "scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    # np.random.seed(args.seed)
    # indices = np.random.permutation(len(train_dataset))[:args.train_size]
    # train_dataset = Subset(train_dataset, indices)
    # print(len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    #set up model
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    assert args.input_model_file != ""    
    model.to(device)
    model.from_pretrained(args.input_model_file, device=device)
    source_state_dict = deep_copy(model.state_dict())
    finetuned_file = "./finetuned_models/{}".format(
            args.checkpoint_name
        ) + ".pth"
    print(finetuned_file)
    model.load_state_dict(torch.load(finetuned_file, map_location=device))

    hessian_norms = np.zeros(shape=(23, )) 

    max_loss = torch.tensor([0.0], device=device)
    model.eval()
    for step, batch in enumerate(train_loader):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
        loss = torch.sum(loss_mat)/torch.sum(is_valid)

        layer_hessian_quantities = compute_hessians_quantity(model, loss, source_state_dict)
        hessian_norms = np.maximum(hessian_norms, layer_hessian_quantities)
        max_loss = torch.maximum(max_loss, loss)
        print(hessian_norms)
        print(max_loss)
    max_loss = max_loss.to("cpu").item()
    print("Hessian-based norms: {}".format(hessian_norms))

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'tox21', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--checkpoint_name', type=str)

    args = parser.parse_args()
    main(args)