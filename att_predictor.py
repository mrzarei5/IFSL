#The main code to train att predictor network f_h

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchvision import transforms

from PIL import Image
import json
import numpy as np
import os
import torch.nn.functional as F
from torch.distributions import Bernoulli
import argparse
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from models import AttPredictor
from datasets import AttDataset
from utils import parse_args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def atts_weight_calculate(labels, att_size): #Sample-wise weighting for cross entropy 
    weight_1 = torch.count_nonzero(labels, dim = 1)
    weight_0 = att_size-weight_1
    weight_1[weight_1==0] = 1
    weight_0[weight_0==0] = 1
    weight_0 = 1/weight_0[:,None].expand(labels.size())
    weight_1 = 1/weight_1[:,None].expand(labels.size())
    zeros = torch.ones(labels.size()).to(device) - labels
    weight_matrix = labels * weight_1 + zeros * weight_0
    return weight_matrix

def calculate_acc_atts_class_wise(predicted,true): 
    acc_all_0 = []
    acc_all_1 = []
    acc_all = []
    for i in range(len(predicted)):
        matrix = confusion_matrix(true[i], predicted[i])
        per_class_num = matrix.sum(axis=1)
        if len(per_class_num) == 2 and  per_class_num.min() > 0:
            vec = matrix.diagonal()/per_class_num
            acc_all_0.append(vec[0])
            acc_all_1.append(vec[1])
            acc_all.append(matrix.diagonal().sum()/matrix.sum())
    return np.mean(acc_all_0), np.mean(acc_all_1), np.mean(acc_all)

def evaluate_attributes(dataloader,model): #Evaluate the performance of the model in attribute prediction
    model.eval()
    
    acc_all_0 = []
    acc_all_1 = []
    acc_all = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            #start = time.time()
            
            images = data[0].to(device)
            atts = data[1].to(device)
            
            atts_predicted = model(images)
            
            atts_predicted_round = atts_predicted.detach().round() + atts_predicted - atts_predicted.detach()

            acc_0, acc_1, acc = calculate_acc_atts_class_wise(atts_predicted_round.detach().cpu().numpy().round(),atts.detach().cpu().numpy().round())
            acc_all_0.append(acc_0)
            acc_all_1.append(acc_1)
            acc_all.append(acc)
        return np.mean(acc_all_0),np.std(acc_all_0), np.mean(acc_all_1),np.std(acc_all_1), np.mean(acc_all),np.std(acc_all)

if __name__=='__main__':
    np.random.seed(10)
    args = parse_args('att_predictor')

    lr_backbone_network = args.lr_backbone_network
    n_iter = args.n_iter
    num_workers = args.num_workers
    dataset_dir = args.dataset_dir
    dataset = args.dataset
    save_dir = args.save_dir_predictor

    if args.dataset == 'CUB':
        att_size = 312
        hid_dim_list = [64,128,256]
        input_size = (84, 84)
    elif args.dataset == 'AWA2':
        att_size = 85
        hid_dim_list = [64,64,64]
        input_size = (84, 84)
    elif args.dataset == 'SUN':
        att_size = 102
        hid_dim_list = [64,64,64]
        input_size = (84, 84)
    elif args.dataset == 'APY':
        att_size = 64
        hid_dim_list = [64,64,64]
        input_size = (84, 84)
    elif args.dataset == 'CIFAR100':
        att_size = 235
        hid_dim_list = [64,64,128]
        input_size = (32, 32)


    backboneNetwork = AttPredictor(hid_dim_list= hid_dim_list, att_size= att_size)

    backboneNetwork.to(device)

    optimizer_backboneNetwork = optim.Adam(backboneNetwork.parameters(), lr=lr_backbone_network) 

    att_criterion = nn.BCELoss(reduction='none')

    save_dir = save_dir + '/' + dataset 

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    train_datafile = dataset_dir + '/base.json'

    val_datafile = dataset_dir + '/val.json'

    test_datafile = dataset_dir + '/novel.json'

    train_dataset_att = AttDataset(dataset_dir, train_datafile, aug = True, input_size= input_size)
    train_loader_att = torch.utils.data.DataLoader(train_dataset_att, batch_size=128, shuffle=True,num_workers = num_workers, pin_memory = True)

    val_dataset_att = AttDataset(dataset_dir, val_datafile, aug = False, input_size= input_size)
    val_loader_att = torch.utils.data.DataLoader(val_dataset_att, batch_size=128, shuffle=False,num_workers = num_workers, pin_memory = True)

    test_dataset_att = AttDataset(dataset_dir, test_datafile, aug = False, input_size= input_size)
    test_loader_att = torch.utils.data.DataLoader(test_dataset_att, batch_size=128, shuffle=False,num_workers = num_workers, pin_memory = True)


    best_att_perf = 0
    for e in range(n_iter):
        class_loss = 0
        backboneNetwork.train()
        for i,data in enumerate(train_loader_att):
        
            images = data[0].to(device)
            atts = data[1].to(device)
            atts_predicted = backboneNetwork(images)
            tt_weights = atts_weight_calculate(atts, att_size)
            atts_zeros = torch.clone(atts)
            atts_zeros[atts_zeros==-1] = 0
            att_weights = atts_weight_calculate(atts_zeros, att_size)
            loss = (att_criterion(atts_predicted, atts) * att_weights).mean()
            backboneNetwork.zero_grad()
            class_loss += loss.item()
            loss.backward()
            optimizer_backboneNetwork.step()

        acc_mean_val_0, acc_std_val_0, acc_mean_val_1, acc_std_val_1, acc_mean_val, acc_std_val = evaluate_attributes(val_loader_att,backboneNetwork) #Evaluate attribute selector on validation classes attribute prediction

        with open(save_dir+'/att_perf.txt', 'a') as f:
            f.write('Iter {:d} | Val 0 Mean Acc {:f} | Val 0 Std {:f} | Val 1 Mean Acc {:f} | Val 1 Std {:f} | Val Mean Acc {:f} | Val Std {:f}\n'.format(e, acc_mean_val_0, acc_std_val_0, acc_mean_val_1, acc_std_val_1, acc_mean_val, acc_std_val))

        if acc_mean_val > best_att_perf: #Saving model with its complete performance reports
            acc_mean_train_0, acc_std_train_0, acc_mean_train_1, acc_std_train_1, acc_mean_train, acc_std_train = evaluate_attributes(train_loader_att,backboneNetwork)
            acc_mean_test_0, acc_std_test_0, acc_mean_test_1, acc_std_test_1, acc_mean_test, acc_std_test = evaluate_attributes(test_loader_att,backboneNetwork)
            with open(save_dir+'/best_att_perf.txt', 'a') as f:
                f.write('Iter {:d} | Train 0 Mean Acc {:f} | Train 0 Std {:f} | Train 1 Mean Acc {:f} | Train 1 Std {:f} | Train Mean Acc {:f} | Train Std {:f} \n'.format(e, acc_mean_train_0, acc_std_train_0, acc_mean_train_1, acc_std_train_1, acc_mean_train, acc_std_train))
                f.write('Iter {:d} | Val 0 Mean Acc {:f} | Val 0 Std {:f} | Val 1 Mean Acc {:f} | Val 1 Std {:f} | Val Mean Acc {:f} | Val Std {:f} \n'.format(e, acc_mean_val_0, acc_std_val_0, acc_mean_val_1, acc_std_val_1, acc_mean_val, acc_std_val))
                f.write('Iter {:d} | Test 0 Mean Acc {:f} | Test 0 Std {:f} | Test 1 Mean Acc {:f} | Test 1 Std {:f} | Test Mean Acc {:f} | Test Std {:f} \n'.format(e, acc_mean_test_0, acc_std_test_0, acc_mean_test_1, acc_std_test_1, acc_mean_test, acc_std_test))
                f.write('\n')
            best_att_perf = acc_mean_val
            torch.save({
                'epoch' : e,
                'model_state_dict' : backboneNetwork.state_dict(),
                }, os.path.join(save_dir,'backboneNetwork-best.pth.tar'))
        