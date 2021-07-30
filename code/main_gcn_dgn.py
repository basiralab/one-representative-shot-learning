# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 23:58:29 2020
@author: Mohammed Amine
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pickle
from torch.autograd import Variable
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

import cross_val
from models_gcn import GCN
from dgn.model import DGN
import time
import random

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import mlab

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device('cpu')



def evaluate(dataset, model_GCN, args, threshold_value):
    """
    Parameters
    ----------
    dataset : dataloader (dataloader for the validation/test dataset).
    model_GCN : nn model (GCN model).
    args : arguments
    threshold_value : float (threshold for adjacency matrices).
    
    Description
    ----------
    This methods performs the evaluation of the model on test/validation dataset
    
    Returns
    -------
    test accuracy.
    """
    model_GCN.eval()
    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).to(device)
        labels.append(data['label'].long().numpy())
        
        adj = torch.squeeze(adj)
        
        features = np.identity(adj.shape[0])
        
        features = Variable(torch.from_numpy(features).float(), requires_grad=False).cpu()
        if args.threshold in ["median", "mean"]:
            adj = torch.where(adj > threshold_value, torch.tensor([1.0]), torch.tensor([0.0]))
        
        ypred = model_GCN(features, adj)

        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

    labels = np.hstack(labels)
    preds = np.hstack(preds)
    
    #https://en.wikipedia.org/wiki/Confusion_matrix
    tn, fp, fn, tp = metrics.confusion_matrix(labels, preds).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1_score = (2*tp) / (2*tp + fp + fn)
    auc_score = metrics.roc_auc_score(labels, preds)
    
    result = {'acc': accuracy,
              'sens': sensitivity,
              'spec': specificity,
              'F1': f1_score,
              'auc' : auc_score
              }
    if args.evaluation_method == 'model assessment':
        name = 'Test'
    if args.evaluation_method == 'model selection':
        name = 'Validation'
    print(name, " accuracy:", result['acc'])
    return result

def minmax_sc(x):
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    return x

def train(args, train_dataset, val_dataset, model_GCN, threshold_value):
    """
    Parameters
    ----------
    args : arguments
    train_dataset : dataloader (dataloader for the validation/test dataset).
    val_dataset : dataloader (dataloader for the validation/test dataset).
    model_GCN : nn model (GCN model).
    threshold_value : float (threshold for adjacency matrices).
    
    Description
    ----------
    This methods performs the training of the model on train dataset and calls evaluate() method for evaluation.
    
    Returns
    -------
    test accuracy.
    """
    params = list(model_GCN.parameters()) 
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    test_accs = []
    for epoch in range(args.num_epochs):
        print("Epoch ",epoch)
        
        model_GCN.train()
        total_time = 0
        avg_loss = 0.0
        
        preds = []
        labels = []
        for batch_idx, data in enumerate(train_dataset):
            begin_time = time.time()

            adj = Variable(data['adj'].float(), requires_grad=False).to(device)
            label = Variable(data['label'].long()).to(device)
            adj_id = Variable(data['id'].int()).to(device)
            
            adj = torch.squeeze(adj)

            features = np.identity(adj.shape[0])
            features = Variable(torch.from_numpy(features).float(), requires_grad=False).cpu()
            if args.threshold in ["median", "mean"]:
                adj = torch.where(adj > threshold_value, torch.tensor([1.0]), torch.tensor([0.0]))
            
            ypred = model_GCN(features, adj)
            
            _, indices = torch.max(ypred, 1)
            preds.append(indices.cpu().data.numpy())
            labels.append(data['label'].long().numpy())
            loss = model_GCN.loss(ypred, label)
            model_GCN.zero_grad()
            loss.backward()
            #nn.utils.clip_grad_norm_(model_DIFFPOOL.parameters(), args.clip)
            optimizer.step()
            
            avg_loss += loss
            elapsed = time.time() - begin_time
            total_time += elapsed
        preds = np.hstack(preds)
        labels = np.hstack(labels)
        print("Train accuracy : ", np.mean( preds == labels ))
        test_acc = evaluate(val_dataset, model_GCN, args, threshold_value)
        print('Avg loss: ', avg_loss, '; epoch time: ', total_time)
        test_accs.append(test_acc)
    return test_acc

def load_data(args):
    """
    Parameters
    ----------
    args : arguments
    Description
    ----------
    This methods loads the adjacency matrices representing the args.view -th view in dataset
    
    Returns
    -------
    List of dictionaries{adj, label, id}
    """
    #Load graphs and labels
    with open('../data/classification/edges','rb') as f:
        multigraphs = pickle.load(f)        
    with open('../data/classification/labels','rb') as f:
        labels = pickle.load(f)
    adjacencies = [multigraphs[i][:,:,args.view] for i in range(len(multigraphs))]
    #Normalize inputs
    if args.NormalizeInputGraphs==True:
        for subject in range(len(adjacencies)):
            adjacencies[subject] = minmax_sc(adjacencies[subject])
    
    #Create List of Dictionaries
    G_list=[]
    for i in range(len(labels)):
        G_element = {"adj":   adjacencies[i],"label": labels[i],"id":  i,}
        G_list.append(G_element)
    return G_list

def arg_parse(dataset, view):
    """
    arguments definition method
    """
    parser = argparse.ArgumentParser(description='Graph Classification')
    parser.add_argument('--dataset', type=str, default=dataset,
                        help='Dataset')
    parser.add_argument('--view', type=int, default=view,
                        help = 'view index in the dataset')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Training Epochs')
    parser.add_argument('--cv_number', type=int, default=5,
                        help='number of validation folds.')
    parser.add_argument('--NormalizeInputGraphs', default=False, action='store_true',
                        help='Normalize Input adjacency matrices of graphs')
    parser.add_argument('--evaluation_method', type=str, default='model selection',
                        help='evaluation method, possible values : model selection, model assessment')
    ##################
    parser.add_argument('--threshold', dest='threshold', default='mean',
            help='threshold the graph adjacency matrix. Possible values: no_threshold, median, mean')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
    parser.add_argument('--num-classes', dest='num_classes', type=int, default=2,
                        help='Number of label classes')
    parser.add_argument('--lr', type=float, default=0.0001,
                    help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
    
    return parser.parse_args()

def benchmark_task(args):
    """
    Parameters
    ----------
    args : Arguments
    Description
    ----------
    Initiates the model and performs train/test or train/validation splits and calls train() to execute training and evaluation.
    Returns
    -------
    test_accs : test accuracies (list)

    """
    G_list = load_data(args)
    num_nodes = G_list[0]['adj'].shape[0]
    test_accs = []
    folds = cross_val.stratify_splits(G_list,args)
    
    MODEL_PARAMS = {
        "N_ROIs": num_nodes,
        "learning_rate" : 0.0005,
        "n_attr": 2,
        "Linear1" : {"in": 2, "out": 36},
        "conv1": {"in" : 1, "out": 36},
        
        "Linear2" : {"in": 2, "out": 36*24},
        "conv2": {"in" : 36, "out": 24},
        
        "Linear3" : {"in": 2, "out": 24*5},
        "conv3": {"in" : 24, "out": 5} 
    }
    
    
    [random.shuffle(folds[i]) for i in range(len(folds))]
    for i in range(args.cv_number):
        train_set, validation_set, test_set = cross_val.datasets_splits(folds, args, i)
        if args.evaluation_method =='model selection':
            train_dataset, val_dataset, threshold_value = cross_val.model_selection_split(train_set, validation_set, args)
        
        if args.evaluation_method =='model assessment':
            train_dataset, val_dataset, threshold_value = cross_val.model_assessment_split(train_set, validation_set, test_set, args)
        print("CV : ",i)
        
        fewshot_dataset = []
        
        class_0 = torch.zeros((0, num_nodes, num_nodes))
        class_1 = torch.zeros((0, num_nodes, num_nodes))
        
        for id_x, data in enumerate(train_dataset):
            if data['label'] == 0:
                class_0 = torch.cat((class_0, data['adj']), axis = 0)

            elif data['label'] == 1:
                class_1 = torch.cat((class_1, data['adj']), axis = 0)

        
        class_0_cbt = DGN.train_model(
                                        np.concatenate([class_0.unsqueeze(3), class_0.unsqueeze(3)], axis=3),
                                        model_params=MODEL_PARAMS,
                                        n_max_epochs= 100,
                                        n_folds= 5,
                                        random_sample_size=10,
                                        early_stop=True,
                                        model_name="DGN_test")
                                        
        class_1_cbt = DGN.train_model(
                                        np.concatenate([class_1.unsqueeze(3), class_1.unsqueeze(3)], axis=3),
                                        model_params=MODEL_PARAMS,
                                        n_max_epochs= 100,
                                        n_folds= 5,
                                        random_sample_size=10,
                                        early_stop=True,
                                        model_name="DGN_test")
                                        
        class_0_cbt = torch.tensor((class_0_cbt[0] + class_0_cbt[1] + class_0_cbt[2] + class_0_cbt[3] + class_0_cbt[4]) / 5.0).unsqueeze(0)
        class_1_cbt = torch.tensor((class_1_cbt[0] + class_1_cbt[1] + class_1_cbt[2] + class_1_cbt[3] + class_1_cbt[4]) / 5.0).unsqueeze(0)
        
        fewshot_dataset.append({'adj': class_0_cbt, 
                                'label': torch.tensor([0]),
                                'id': torch.tensor([0]) 
                                })
                                
        fewshot_dataset.append({'adj': class_1_cbt, 
                                'label': torch.tensor([1]), 
                                'id': torch.tensor([1]) 
                                })
        
        
        model_GCN = GCN(nfeat = num_nodes,
                        nhid = args.hidden,
                        nclass = args.num_classes,
                        dropout = args.dropout)
                        
        
        test_acc = train(args, fewshot_dataset, val_dataset, model_GCN, threshold_value)
        test_accs.append(test_acc)
    return test_accs

def test_scores(dataset, view):
    
    args = arg_parse(dataset, view)
    print("Main : ",args)
    test_accs = benchmark_task(args)
    
    result = {'acc': 0,
              'sens': 0,
              'spec': 0,
              'F1': 0,
              'auc' : 0
              }
    
    for fold in test_accs:
        for metric in fold:
          result[metric] += fold[metric] / len(test_accs)
    
    print("GCN DGN")
    for metric in result:
        print(metric + ': ' + str(result[metric]))
    
    return test_accs
    
