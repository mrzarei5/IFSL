#The main code to achieve unknown attributes using training f_u [Experiment related to automatically balancing accuracy and interpretability]
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
import argparse
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from utils import parse_args
from datasets import FSLDataset
from models import AttPredictor, MutualHelper


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ml_estimate(ml_model, concepts_real, concepts_fake, concepts_fake_marginal, eval_mode = 0):
    if eval_mode == 1:
        ml_model.eval()
    
    concept_cat = torch.cat((concepts_real, concepts_fake), dim = 1)
    concept_cat_marginal = torch.cat((concepts_real, concepts_fake_marginal), dim = 1)
    ml1 = ml_model(concept_cat).mean()
    ml2 = torch.log((torch.exp(ml_model(concept_cat_marginal))).mean())
    return ml1 - ml2

def train_mutual_network(loader, images, backboneNetwork, mutualNetwork, optimizer_mutualNetwork, atts_predicted_real):
    for param in backboneNetwork.parameters():
        param.requires_grad = False

    atts_predicted_fake = backboneNetwork(images)


    for param in mutualNetwork.parameters(): 
        param.requires_grad = True
    
    mutualNetwork.train()
        
    for i,data in enumerate(loader):
        images = data[0][0,:].to(device)
        atts = data[1][0,:].to(device)

        images_reshape = images.contiguous().view(images_size[0]*images_size[1],images_size[2],images_size[3],images_size[4])


        atts_predicted_fake2 = backboneNetwork(images_reshape)


        loss = -1 * ml_estimate(mutualNetwork, atts_predicted_real, atts_predicted_fake, atts_predicted_fake2)

        mutualNetwork.zero_grad()
    
        loss.backward(retain_graph=True)
        
        optimizer_mutualNetwork.step()
        
    return mutualNetwork, optimizer_mutualNetwork



def fsl_dists(prototypes,query_features): #Estimate distances between query samples and prototypes of different classes
    query_features_ = torch.unsqueeze(query_features.view(query_features.size(0)*query_features.size(1),query_features.size(2)),1)
    prototypes_ = torch.unsqueeze(prototypes,0)

    size_0 = query_features_.size(0)
    size_1 = prototypes_.size(1)
    size_2 = query_features_.size(2)

    query_features_ = query_features_.expand(size_0,size_1,size_2)   
    
    prototypes_ = prototypes_.expand(size_0,size_1,size_2)

    dist  = (((query_features_-prototypes_)**2)).sum(2)
    return dist 
def calculate_acc(prototypes,query_features, n_query, att_size_real, att_size_fake): #Calculate accuracy in one episode
    query_features_ = torch.unsqueeze(query_features,1)
    prototypes_ = torch.unsqueeze(prototypes,0)

    size_0 = query_features_.size(0)
    size_1 = prototypes_.size(1)
    size_2 = query_features_.size(2)
    
    query_features_ = query_features_.expand(size_0,size_1,size_2)   
    prototypes_ = prototypes_.expand(size_0,size_1,size_2)
    

    dist  = (((query_features_- prototypes_)**2)).sum(2)

    min_args = torch.argmin(dist,1)

    query_labels = torch.tensor([[i] for i in range(prototypes.size(0))]).to(device)
    query_labels = query_labels.repeat(1,n_query).flatten()
    
    sum = (min_args == query_labels).sum().item()
    overall = len(query_labels)
    return sum / overall * 100

def evaluate(data_list, backboneNetwork, n_way, n_support, n_query, att_size_real, att_size_fake): #Evaluate the framework on a set of episodes 
    backboneNetwork.eval()
    acc_list = []
    
    index_selected_list = []
    with torch.no_grad():
        for i, data in enumerate(data_list):
            #start = time.time()
            
            images = data[0][0,:].to(device)
            atts = data[1][0,:].to(device)


            images_size = images.size()



            images_reshape = images.contiguous().view(images_size[0]*images_size[1],images_size[2],images_size[3],images_size[4])


            atts_predicted_reshape = backboneNetwork(images_reshape)

            atts_predicted = atts_predicted_reshape.contiguous().view(n_way,n_support + n_query,-1)

            
            support_features = atts_predicted[:,:n_support,:]
            support_features_protos = support_features.mean(1).view(-1,att_size_fake)

            query_features_reshape = atts_predicted[:,n_support:,:].contiguous().view(n_way*n_query,att_size_fake)
            
            
            acc = calculate_acc(support_features_protos,query_features_reshape,n_query, att_size_real, att_size_fake)
     
    
            acc_list.append(acc)
    index_selected_list = [ind.sum().item() for ind in index_selected_list]
    
    acc_all  = np.array(acc_list)
    return np.mean(acc_all), 1.96 * np.std(acc_all)/np.sqrt(len(data_list)), np.mean(index_selected_list), np.std(index_selected_list)
if __name__=='__main__':
    np.random.seed(10)
    args = parse_args('att_predictor_unk')

    lr_backbone_network = args.lr_backbone_network

    n_episode_train = args.n_episode_train
    n_iter = args.n_iter
    num_workers = args.num_workers

    dataset_dir = args.dataset_dir
    dataset = args.dataset
    save_dir = args.save_dir_predictor_unknown
    save_dir_real = args.save_dir_predictor
    n_way = args.n_way
    n_query = args.n_query
    n_support = args.n_support

    lr_mi_helper = args.lr_mi_helper
    decoupling_weight = args.decoupling_weight
    n_mi_learner = args.n_mi_learner

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

    att_size_fake = att_size
    att_size_overall = att_size + att_size_fake

    cross_entropy_criterioin = nn.CrossEntropyLoss(reduction='mean')

    save_dir_fake = save_dir + '/' + dataset + '/' + str(n_support)+'_shot_unknown/'+'n_mi_learner_'+str(n_mi_learner)+'_decoupling_weight_'+str(decoupling_weight)
    save_dir_att = save_dir_real + '/' + dataset #directory of trained attribute predictor
    

    if not os.path.isdir(save_dir_fake):
        os.makedirs(save_dir_fake)


    n_episode_test = 600

    train_datafile = dataset_dir + '/base.json'
    train_dataset_fsl = FSLDataset(dataset_dir, train_datafile, n_episode_train, n_way, n_support, n_query, aug = True, input_size= input_size)
    train_loader_fsl = torch.utils.data.DataLoader(train_dataset_fsl, batch_size=1, shuffle=True,num_workers = num_workers, pin_memory = True)

    val_datafile = dataset_dir + '/val.json'
    val_dataset_fsl = FSLDataset(dataset_dir,val_datafile, n_episode_test, n_way, n_support, n_query, aug = False, input_size= input_size)
    val_loader_fsl = torch.utils.data.DataLoader(val_dataset_fsl, batch_size=1, shuffle=False,num_workers = num_workers, pin_memory = True)

    test_datafile = dataset_dir + '/novel.json'
    test_dataset_fsl = FSLDataset(dataset_dir, test_datafile, n_episode_test, n_way, n_support, n_query, aug = False, input_size= input_size)
    test_loader_fsl = torch.utils.data.DataLoader(test_dataset_fsl, batch_size=1, shuffle=False,num_workers = num_workers, pin_memory = True)



    mi_dataset_fsl = FSLDataset(dataset_dir, train_datafile, n_mi_learner, n_way, n_support, n_query, aug = True, input_size= input_size)
    mi_loader_fsl = torch.utils.data.DataLoader(mi_dataset_fsl, batch_size=1, shuffle=True,num_workers = num_workers, pin_memory = True)

    

    backboneNetwork_real = AttPredictor(hid_dim_list= hid_dim_list, att_size= att_size)  
    backboneNetwork_saved = torch.load(os.path.join(save_dir_att,'backboneNetwork-best.pth.tar'))  
    backboneNetwork_real.load_state_dict(backboneNetwork_saved['model_state_dict']) 
    backboneNetwork_real.to(device)

    for param in backboneNetwork_real.parameters():
        param.requires_grad = False


    backboneNetwork = AttPredictor(hid_dim_list= hid_dim_list, att_size= att_size_fake)
    backboneNetwork.to(device)
    optimizer_backboneNetwork = optim.Adam(backboneNetwork.parameters(), lr=lr_backbone_network) 

    mutualNetwork = MutualHelper(att_size_overall)
    mutualNetwork.to(device)
    optimizer_mutualNetwork = optim.Adam(mutualNetwork.parameters(), lr = lr_mi_helper)


    best_acc = 0
    for e in range(n_iter):
        epoch_loss = 0
        class_loss = 0
        mi_loss = 0
        mi_value = 0

        backboneNetwork.train()
        for i,data in enumerate(train_loader_fsl):
            # n_way x (n_query+n_support) X C x H x W
            
            images = data[0][0,:].to(device)
            atts = data[1][0,:].to(device)

            images_size = images.size()
            
            images_reshape = images.contiguous().view(images_size[0]*images_size[1],images_size[2],images_size[3],images_size[4])

            
            atts_predicted_real = backboneNetwork_real(images_reshape)
         
            mutualNetwork, optimizer_mutualNetwork = train_mutual_network(mi_loader_fsl, images_reshape, backboneNetwork, mutualNetwork, optimizer_mutualNetwork, atts_predicted_real)

            for param in backboneNetwork.parameters():
                param.requires_grad = True
            
            for param in mutualNetwork.parameters(): 
                param.requires_grad = False

            atts_predicted_fake = backboneNetwork(images_reshape)

            atts_predicted_reshape = atts_predicted_fake

            atts_predicted = atts_predicted_reshape.contiguous().view(n_way,n_support + n_query,-1)
            
            support_features = atts_predicted[:,:n_support,:]
            

            query_labels_np = np.repeat(np.array([[i] for i in range(n_way)]),n_query,axis=1) 
            query_labels = torch.from_numpy(query_labels_np).to(device)
            query_labels_reshape = query_labels.contiguous().view(n_way*n_query)
            
            support_features_protos = support_features.mean(1).view(n_way,att_size_fake)

            query_features = atts_predicted[:,n_support:,:].contiguous().view(n_way,n_query,att_size_fake)
            
            dists = fsl_dists(support_features_protos,query_features)
            
            
            loss1 = cross_entropy_criterioin(-dists,query_labels_reshape)

            
            for j,data2 in enumerate(mi_loader_fsl):
                # n_way x (n_query+n_support) X C x H x W
                
                images2 = data2[0][0,:].to(device)
                atts2 = data2[1][0,:].to(device)
                
                images_reshape2 = images2.contiguous().view(images_size[0]*images_size[1],images_size[2],images_size[3],images_size[4])

                atts_predicted_fake2 = backboneNetwork(images_reshape2)
                
                mi_estimation = ml_estimate(mutualNetwork, atts_predicted_real, atts_predicted_fake, atts_predicted_fake2)
    
                loss2 = decoupling_weight * mi_estimation
                break
            
            loss = loss1 + loss2

            backboneNetwork.zero_grad()
            loss.backward()
            optimizer_backboneNetwork.step()

            epoch_loss += loss.item()
            class_loss += loss1.item()
            mi_loss += loss2.item()
            mi_value += mi_estimation.item()

        val_acc, val_acc_std, val_mean, val_std = evaluate(val_loader_fsl, backboneNetwork, n_way, n_support, n_query, att_size, att_size_fake)
        
        
        with open(save_dir_fake+'/model_perf.txt', 'a') as f:  
            f.write('Iter {:d} | Loss: {:f} | Class Loss {:f} | MI Loss {:f} | MI Value {:f} | Val Acc {:f} | Val Std {:f} \n'.format(e, epoch_loss/len(train_loader_fsl), class_loss/len(train_loader_fsl), mi_loss/len(train_loader_fsl), mi_value/len(train_loader_fsl), val_acc, val_acc_std))
            
            f.write('\n')
        if val_acc > best_acc:
            
            test_acc, test_acc_std, test_mean, test_std = evaluate(test_loader_fsl, backboneNetwork, n_way, n_support, n_query, att_size, att_size_fake)
            with open(save_dir_fake+'/best_perf.txt', 'a') as f:
                f.write('Iter {:d} | Loss: {:f} | Class Loss {:f} | MI Loss {:f} | MI Value {:f} | Val Acc {:f} | Val Std {:f} | Test Acc {:f} | Test Std {:f} \n'.format(e, epoch_loss/len(train_loader_fsl), class_loss/len(train_loader_fsl), mi_loss/len(train_loader_fsl), mi_value/len(train_loader_fsl), val_acc, val_acc_std, test_acc, test_acc_std))
                
                f.write('\n')
            torch.save({
                'epoch' : e,
                'model_state_dict' : backboneNetwork.state_dict(),
                'val_acc' : val_acc,
                'train_loss' : epoch_loss,
                'best_acc' : best_acc,
                }, os.path.join(save_dir_fake,'backboneNetwork-best.pth.tar'))
            torch.save({
                'epoch' : e,
                'model_state_dict' : mutualNetwork.state_dict(),
                'val_acc' : val_acc,
                'train_loss' : epoch_loss,
                'best_acc' : best_acc,
                }, os.path.join(save_dir_fake,'mutualNetwork-best.pth.tar'))
            best_acc = val_acc
