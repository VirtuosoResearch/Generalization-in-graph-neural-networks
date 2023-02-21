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
from utils.constraint import LInfLipschitzConstraint, FrobeniusConstraint, add_penalty, deep_copy
from torch.optim.swa_utils import AveragedModel, SWALR

criterion = nn.BCEWithLogitsLoss(reduction = "none")

def update_bn(args, model, device, loader):
    '''
    take a forward pass on every element of training dataset
    '''
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

def train(args, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        optimizer.step()

def train_reg(args, model, device, loader, optimizer, penalties, constraints):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        ''' Apply penalties '''
        for penalty in penalties:
            loss += add_penalty(
                model, 
                penalty["norm"], 
                penalty["_lambda"], 
                excluding_key = penalty["excluding_key"],
                including_key = penalty["including_key"],
                state_dict=penalty["state_dict"]
            )

        loss.backward()
        optimizer.step()

        ''' Apply constraints '''
        for constraint in constraints:
            model.apply(constraint)

def train_ls(args, model, device, loader, optimizer, alpha = 0.2):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix: smoothed labels y = (1-alpha)y + alpha(1-y)
        loss_mat = criterion(pred.double(), (1-alpha)*(y+1)/2 + alpha*(1-y)/2 )
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
        
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        optimizer.step()

def train_sam(args, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        # first forward-backward step
        enable_running_stats(model)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        is_valid = y**2 > 0 # Whether y is non-null or not.
        loss_mat = criterion(pred.double(), (y+1)/2) # Loss matrix
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype)) # loss matrix after removing null target     
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # second forward-backward step
        disable_running_stats(model)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        loss_mat = criterion(pred.double(), (y+1)/2) # Loss matrix
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype)) # loss matrix after removing null target     
        (torch.sum(loss_mat)/torch.sum(is_valid)).backward()
        optimizer.second_step(zero_grad=True)

def train_nsm(args, model, device, loader, optimizer, num_perturbs=10, lam=0.5, use_neg=False, penalties=[], constraints=[]):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        # first forward-backward step
        enable_running_stats(model)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)
        is_valid = y**2 > 0 # Whether y is non-null or not.
        loss_mat = criterion(pred.double(), (y+1)/2) # Loss matrix
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype)) # loss matrix after removing null target     
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        ''' Apply penalties '''
        for penalty in penalties:
            loss += add_penalty(
                model, 
                penalty["norm"], 
                penalty["_lambda"], 
                excluding_key = penalty["excluding_key"],
                including_key = penalty["including_key"],
                state_dict=penalty["state_dict"]
            )

        loss.backward()
        optimizer.store_gradients(zero_grad=True, store_weights=True, update_weight=lam)

        # forward-backward step for num_perturbs
        disable_running_stats(model)
        update_weight = (1-lam)/(2*num_perturbs) if use_neg else (1-lam)/(num_perturbs)
        for _ in range(num_perturbs):
            optimizer.first_step(zero_grad=True, store_perturb=True)

            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss_mat = criterion(pred.double(), (y+1)/2) # Loss matrix
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype)) # loss matrix after removing null target     
            (torch.sum(loss_mat)/torch.sum(is_valid)).backward()
            
            optimizer.store_gradients(zero_grad=True, store_weights=False, update_weight=update_weight)
            if use_neg:
                optimizer.first_step(zero_grad=True, store_perturb=False)
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss_mat = criterion(pred.double(), (y+1)/2) # Loss matrix
                loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype)) # loss matrix after removing null target     
                (torch.sum(loss_mat)/torch.sum(is_valid)).backward()
                optimizer.store_gradients(zero_grad=True, store_weights=False, update_weight=update_weight)
        
        optimizer.second_step(zero_grad=True)

        ''' Apply constraints '''
        for constraint in constraints:
            model.apply(constraint)


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0)
    y_scores = torch.cat(y_scores, dim = 0)

    ''' Evaluate loss '''
    y_true = y_true.to(torch.float64)
    #Whether y is non-null or not.
    is_valid = y_true**2 > 0
    #Loss matrix
    loss_mat = criterion(y_scores.double(), (y_true+1)/2)
    #loss matrix after removing null target
    loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
    loss = (torch.sum(loss_mat)/torch.sum(is_valid)).item()

    y_true = y_true.to(torch.int64).cpu().numpy()
    y_scores = y_scores.cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list), loss #y_true.shape[1]

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

    np.random.seed(args.seed)
    indices = np.random.permutation(len(train_dataset))[:args.train_size]
    train_dataset = Subset(train_dataset, indices)
    print(len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    #set up model
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    assert args.input_model_file != ""    
    model.to(device)

    valid_aucs = []; test_aucs = []; final_valid_aucs = []; final_test_aucs = []
    for run in range(1, args.runs+1):
        # reset parameters of model
        model.reset_parameters()
        model.from_pretrained(args.input_model_file, device=device)
        source_state_dict = deep_copy(model.state_dict())
        # set up optimizer
        # different learning rate for different part of GNN
        model_param_group = []
        model_param_group.append({"params": model.gnn.parameters()})
        if args.graph_pooling == "attention":
            model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
        model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
        
        if args.train_nsm:
            base_optimizer = torch.optim.Adam
            optimizer = NSM(model_param_group, base_optimizer, sigma=args.nsm_sigma, lr=args.lr, weight_decay=args.weight_decay)
        elif args.train_sam:
            base_optimizer = torch.optim.Adam
            optimizer = SAM(model_param_group, base_optimizer, rho=args.sam_rho, adaptive=False, lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.weight_decay)
        print(optimizer)

        if args.train_swa:
            swa_model = AveragedModel(model)
            swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)

        train_acc_list = []
        val_acc_list = []; val_loss_list = []
        test_acc_list = []; test_loss_list = []
        
        penalties = []; constraints = []
        if args.reg_method == 'penalty':
            penalties.append(
                {"norm": "frob", 
                "_lambda": args.lam_gnn,
                "excluding_key": None,
                "including_key": "gnns",
                "state_dict": source_state_dict}
            )
            penalties.append(
                {"norm": "frob", 
                "_lambda": args.lam_pred,
                "excluding_key": None,
                "including_key": "graph_pred_linear",
                "state_dict": None}
            )
        elif args.reg_method == 'constraint':
            constraints.append(
                FrobeniusConstraint(type(model), args.lam_gnn, 
                state_dict = source_state_dict, including_key= "gnns")
            )
            constraints.append(
                FrobeniusConstraint(type(model), args.lam_pred, 
                including_key="graph_pred_linear")
            )

        for epoch in range(1, args.epochs+1):

            if args.train_nsm:
                train_nsm(args, model, device, train_loader, optimizer, 
                    num_perturbs=args.num_perturbs, lam=args.nsm_lam, use_neg=args.use_neg, penalties=penalties, constraints=constraints)
            elif args.train_sam:
                train_sam(args, model, device, train_loader, optimizer)
            elif args.train_ls:
                train_ls(args, model, device, train_loader, optimizer, alpha=args.ls_alpha)
            elif args.train_swa:
                train(args, model, device, train_loader, optimizer)
                if epoch > args.swa_epoch:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()
            else:            
                train_reg(args, model, device, train_loader, optimizer, penalties=penalties, constraints=constraints)

            train_acc, train_loss = eval(args, model, device, train_loader)
            val_acc, val_loss = eval(args, model, device, val_loader)
            test_acc, test_loss = eval(args, model, device, test_loader)
            print("====epoch %d train loss: %f val loss: %f test loss: %f" %(epoch, train_loss, val_loss, test_loss))
            print("====         train auc: %f val auc: %f test auc: %f" %(train_acc, val_acc, test_acc))

            val_acc_list.append(val_acc)
            test_acc_list.append(test_acc)
            train_acc_list.append(train_acc)

        if args.train_swa:
            update_bn(args, swa_model, device, train_loader)
            swa_val_acc, _ = eval(args, swa_model, device, val_loader)
            swa_test_acc, _ = eval(args, swa_model, device, test_loader)

        output_model_file = "./finetuned_models/{}_{}_run_{}_rand_init_{}".format(
                            args.gnn_type, args.dataset, args.runseed, args.input_model_file == ""
                        ) + \
                        ("_ls" if args.train_ls else "")  + \
                        ("_swa" if args.train_swa else "")  + \
                        ("_sam" if args.train_sam else "")  + \
                        ("_nsm" if args.train_nsm else "")
        torch.save(model.state_dict(), output_model_file + ".pth")

        # record the best accuracy
        best_idx = np.argmax(val_acc_list)

        if args.train_swa:
            valid_aucs.append(swa_val_acc); test_aucs.append(swa_test_acc)
            print(f"Run: {run:2.0f} train: {train_acc_list[best_idx]:.4f} valid: {swa_val_acc:.4f} test: {swa_test_acc:.4f}")
        else:
            valid_aucs.append(val_acc_list[best_idx]); test_aucs.append(test_acc_list[best_idx])
            print(f"Run: {run:2.0f} train: {train_acc_list[best_idx]:.4f} valid: {val_acc_list[best_idx]:.4f} test: {test_acc_list[best_idx]:.4f}")
        final_valid_aucs.append(val_acc_list[-1]); final_test_aucs.append(test_acc_list[-1])
    print(f"Valid AUC-ROC: {np.mean(valid_aucs):.4f}+/-{np.std(valid_aucs):.4f} Test AUC-ROC: {np.mean(test_aucs):.4f}+/-{np.std(test_aucs):.4f}")
    print(f"Final valid AUC-ROC: {np.mean(final_valid_aucs):.4f}+/-{np.std(final_valid_aucs):.4f} Final test AUC-ROC: {np.mean(final_test_aucs):.4f}+/-{np.std(final_test_aucs):.4f}")

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
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--train_size', type=int, default=10000)

    ''' Algorithms '''
    parser.add_argument('--reg_method', type=str, default="none")
    parser.add_argument('--lam_gnn', type=float, default=0)
    parser.add_argument('--lam_pred', type=float, default=0)

    parser.add_argument('--train_ls', action="store_true")
    parser.add_argument('--ls_alpha', type=float, default=0.2)

    parser.add_argument('--train_swa', action="store_true")
    parser.add_argument('--swa_epoch', type=int, default=75)
    parser.add_argument('--swa_lr', type=float, default=0.001)

    parser.add_argument('--train_sam', action="store_true")
    parser.add_argument('--sam_rho', type=float, default=0.05)

    parser.add_argument('--train_nsm', action="store_true")
    parser.add_argument('--use_neg', action="store_true")
    parser.add_argument('--nsm_sigma', type=float, default=0.05)
    parser.add_argument('--num_perturbs', type=int, default=10)
    parser.add_argument('--nsm_lam', type=float, default=0.6)

    args = parser.parse_args()
    main(args)
