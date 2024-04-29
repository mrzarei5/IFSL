# The main code to train unknown attribute participation detector network (g_u)  [Experiment related to automatically balancing accuracy and interpretability]
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

from models import AttSelector, AttPredictor, AttSelectorUnknown
from utils import parse_args
from datasets import FSLDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fsl_dists(prototypes,query_features,prototypes_fake, query_features_fake, selected_index, selected_index_real): #Estimate distances between query samples and prototypes of different classes

    query_features_ = torch.unsqueeze(query_features.view(query_features.size(0)*query_features.size(1),query_features.size(2)),1)
    prototypes_ = torch.unsqueeze(prototypes,0)

    size_0 = query_features_.size(0)
    size_1 = prototypes_.size(1)
    size_2 = query_features_.size(2)

    query_features_ = query_features_.expand(size_0,size_1,size_2)   
    
    prototypes_ = prototypes_.expand(size_0,size_1,size_2)

    query_features_fake_ = torch.unsqueeze(query_features_fake.view(query_features_fake.size(0)*query_features_fake.size(1),query_features_fake.size(2)),1)
    prototypes_fake_ = torch.unsqueeze(prototypes_fake,0)

    size_0_fake = query_features_fake_.size(0)
    size_1_fake = prototypes_fake_.size(1)
    size_2_fake = query_features_fake_.size(2)
    
    
    query_features_fake_ = query_features_fake_.expand(size_0_fake,size_1_fake,size_2_fake)   
    
    prototypes_fake_ = prototypes_fake_.expand(size_0_fake,size_1_fake,size_2_fake)

    dist  = (((query_features_ - prototypes_)**2)*selected_index_real).sum(2)
    dist2 = (((query_features_fake_ - prototypes_fake_)**2)*selected_index[0]).sum(2)

    return dist+dist2
def calculate_acc(prototypes,query_features, prototypes_fake, query_features_fake, selected_index, selected_index_real, n_query): #Calculate accuracy in one episode
    query_features_ = torch.unsqueeze(query_features,1)
    prototypes_ = torch.unsqueeze(prototypes,0)

    size_0 = query_features_.size(0)
    size_1 = prototypes_.size(1)
    size_2 = query_features_.size(2)


    query_features_ = query_features_.expand(size_0,size_1,size_2)   
    prototypes_ = prototypes_.expand(size_0,size_1,size_2)

    query_features_fake_ = torch.unsqueeze(query_features_fake,1)
    prototypes_fake_ = torch.unsqueeze(prototypes_fake,0)

    size_0_fake = query_features_fake_.size(0)
    size_1_fake = prototypes_fake_.size(1)
    size_2_fake = query_features_fake_.size(2)

    query_features_fake_ = query_features_fake_.expand(size_0_fake,size_1_fake,size_2_fake)   
    prototypes_fake_ = prototypes_fake_.expand(size_0_fake,size_1_fake,size_2_fake)


    dist  = (((query_features_ - prototypes_)**2)*selected_index_real).sum(2)
    dist2 = (((query_features_fake_ - prototypes_fake_)**2)*selected_index[0]).sum(2)
    
    dist = dist + dist2

    min_args = torch.argmin(dist,1)

    query_labels = torch.tensor([[i] for i in range(prototypes.size(0))]).to(device)
    query_labels = query_labels.repeat(1,n_query).flatten()
    
    sum = (min_args == query_labels).sum().item()
    overall = len(query_labels)
    return sum / overall * 100


def evaluate(data_list, backboneNetwork, backboneNetwork_fake, attSelector, attSelector_real, att_size, att_size_fake, n_way, n_support, n_query): #Evaluate on a set of episodes
    backboneNetwork.eval()
    backboneNetwork_fake.eval()
    attSelector.eval()
    attSelector_real.eval()
    acc_list = []
    
    selected_index_list = []

    with torch.no_grad():
        for i, data in enumerate(data_list):
            
            images = data[0][0,:].to(device)
            atts = data[1][0,:].to(device)


            images_size = images.size()

            images_reshape = images.contiguous().view(images_size[0]*images_size[1],images_size[2],images_size[3],images_size[4])


            atts_predicted_reshape = backboneNetwork(images_reshape)
            atts_predicted_reshape_fake = backboneNetwork_fake(images_reshape)
          
            atts_predicted = atts_predicted_reshape.contiguous().view(n_way,n_support + n_query,-1)
            atts_predicted_fake = atts_predicted_reshape_fake.contiguous().view(n_way,n_support + n_query,-1)
    
            support_features = atts_predicted[:,:n_support,:]
            support_features_protos = support_features.mean(1).view(-1,att_size)

            support_features_fake = atts_predicted_fake[:,:n_support,:]
            support_features_protos_fake = support_features_fake.mean(1).view(-1,att_size_fake)


            support_features_protos_mixed = torch.cat((support_features_protos,support_features_protos_fake),dim=1)

            index_selected_real, _ = attSelector_real(support_features_protos)

            index_selected, _ = attSelector(support_features_protos_mixed)
    
            index_selected = torch.ravel(index_selected)
        
            query_features_reshape = atts_predicted[:,n_support:,:].contiguous().view(n_way*n_query,att_size)
            query_features_reshape_fake = atts_predicted_fake[:,n_support:,:].contiguous().view(n_way*n_query,att_size_fake)

            selected_index_list.append(index_selected.detach())
            
            acc = calculate_acc(support_features_protos,query_features_reshape, support_features_protos_fake, query_features_reshape_fake, index_selected,index_selected_real,n_query)
     
    
            acc_list.append(acc)
    index_selected_list_all = [ind.sum().item() for ind in selected_index_list]

    
    acc_all  = np.array(acc_list)
    return np.mean(acc_all), 1.96 * np.std(acc_all)/np.sqrt(len(data_list)), np.mean(index_selected_list_all), np.std(index_selected_list_all)


if __name__=='__main__':
    args = parse_args('unk_att_participation')


    lr_unk_part_detector = args.lr_unk_part_detector
    n_episode_train = args.n_episode_train
    n_iter = args.n_iter
    num_workers = args.num_workers
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    dataset_dir = args.dataset_dir
    dataset = args.dataset
    save_dir = args.save_dir_unknown_participation
    n_way = args.n_way
    n_query = args.n_query
    n_support = args.n_support
    tau = args.tau    



    if args.dataset == 'CUB':
        att_size = 312
        hid_dim_list = [64,128,256]
        att_size_fake = 312
        hid_dim_list_fake = [64,128,256]
        input_size = (84, 84)
    elif args.dataset == 'AWA2':
        att_size = 85
        hid_dim_list = [64,64,64]
        att_size_fake = 85
        hid_dim_list_fake = [64,64,64]
        input_size = (84, 84)
    elif args.dataset == 'SUN':
        att_size = 102
        hid_dim_list = [64,64,64]
        att_size_fake = 102
        hid_dim_list_fake = [64,64,64]
        input_size = (84, 84)
    elif args.dataset == 'APY':
        att_size = 64
        hid_dim_list = [64,64,64]
        att_size_fake = 64
        hid_dim_list_fake = [64,64,64]
        input_size = (84, 84)
    elif args.dataset == 'CIFAR100':
        att_size = 235
        hid_dim_list = [64,64,128]
        att_size_fake = 235
        hid_dim_list_fake = [64,64,128]
        input_size = (32, 32)

    att_size_fake = att_size
    att_size_overall = att_size + att_size_fake

    cross_entropy_criterioin = nn.CrossEntropyLoss(reduction='mean')


    save_dir_att = save_dir + '/' + dataset
    save_dir_att_fake = save_dir + '/' + dataset + '/' + str(n_support)+'_shot_unknown/'+'n_mi_learner_10_decoupling_weight_2.0'

    save_dir_att_selector = save_dir + '/' + dataset + '/'+str(n_way)+'_way_'+str(n_support)+'_shot'+ '/'+ 'l1_' + str(alpha) + '_l2_' + str(gamma)

    #save_dir = save_dir + '/' + dataset + '/'+str(n_way)+'_way_'+str(n_support)+'_unknown_participation_updated'+ '/'+ 'l1_' + str(alpha) + '_l2_' + str(beta)
    save_dir = save_dir + '/' + dataset + '/'+str(n_way)+'_way_'+str(n_support)+'_unknown_participation'+ '/'+ 'l1_' + str(alpha) + '_l2_' + str(beta)
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)


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



    backboneNetwork = AttPredictor(hid_dim_list= hid_dim_list, att_size= att_size)   #load attribute predictor network for human friendly attributes
    backboneNetwork_saved = torch.load(os.path.join(save_dir_att,'backboneNetwork-best.pth.tar'))
    backboneNetwork.load_state_dict(backboneNetwork_saved['model_state_dict'])
    backboneNetwork.to(device)


    backboneNetwork_fake = AttPredictor(hid_dim_list= hid_dim_list_fake, att_size= att_size_fake) #load unknown attirubte predictor network
    backboneNetwork_saved_fake = torch.load(os.path.join(save_dir_att_fake,'backboneNetwork-best.pth.tar'))
    backboneNetwork_fake.load_state_dict(backboneNetwork_saved_fake['model_state_dict'])
    backboneNetwork_fake.to(device)


    attSelector_real = AttSelector(tau=tau, att_size=att_size)  #load human-friendly attribute selector network
    attSelector_real_saved = torch.load(os.path.join(save_dir_att_selector,'attSelector-best.pth.tar'))
    attSelector_real.load_state_dict(attSelector_real_saved['model_state_dict'])
    attSelector_real.to(device)

    attSelector = AttSelectorUnknown(tau=tau, att_size=att_size+att_size_fake, index_size=1)
    attSelector.to(device)
    optimizer_attSelector = optim.Adam(attSelector.parameters(), lr=lr_unk_part_detector) 
    
    for param in backboneNetwork.parameters():
        param.requires_grad = False

    for param in backboneNetwork_fake.parameters():
        param.requires_grad = False

    for param in attSelector_real.parameters():
        param.requires_grad = False


    best_acc = 0
    for e in range(n_iter):
        epoch_loss = 0
        class_loss = 0
        selected_index_fake_loss = 0

        selected_index_list = []
        attSelector.train()
        for i,data in enumerate(train_loader_fsl):

            images = data[0][0,:].to(device)
            atts = data[1][0,:].to(device)

            images_size = images.size()
            
            images_reshape = images.contiguous().view(images_size[0]*images_size[1],images_size[2],images_size[3],images_size[4])


            atts_predicted_reshape = backboneNetwork(images_reshape)
            
            atts_predicted_reshape_fake = backboneNetwork_fake(images_reshape)

            
            atts_predicted = atts_predicted_reshape.contiguous().view(n_way,n_support + n_query,-1)

            atts_predicted_fake = atts_predicted_reshape_fake.contiguous().view(n_way,n_support + n_query,-1)

            
        
            atts_support = atts_predicted[:,:n_support,:]
            atts_prototypes = atts_support.mean(1).view(-1,att_size)

            atts_support_fake = atts_predicted_fake[:,:n_support,:]
            atts_prototypes_fake = atts_support_fake.mean(1).view(-1,att_size_fake)


            atts_prototypes_mixed = torch.cat((atts_prototypes,atts_prototypes_fake),dim=1)  #stack human-friendly and unknown prototypes


            index_selected, _ = attSelector(atts_prototypes_mixed)  #The output will be a scalar which determines the particiaption of unknown attributes in the current episode

            index_selected_real, _ = attSelector_real(atts_prototypes)
            
        
            selected_index_list.append(index_selected.detach())
            
            support_features = atts_predicted[:,:n_support,:]
            support_features_fake = atts_predicted_fake[:,:n_support,:]


            query_labels_np = np.repeat(np.array([[i] for i in range(n_way)]),n_query,axis=1) 
            query_labels = torch.from_numpy(query_labels_np).to(device)
            query_labels_reshape = query_labels.contiguous().view(n_way*n_query)
            
            support_features_protos = support_features.mean(1).view(n_way,att_size)
            support_features_protos_fake = support_features_fake.mean(1).view(n_way,att_size_fake)


            query_features = atts_predicted[:,n_support:,:].contiguous().view(n_way,n_query,att_size)
            query_features_fake = atts_predicted_fake[:,n_support:,:].contiguous().view(n_way,n_query,att_size_fake)

            
            dists = fsl_dists(support_features_protos,query_features,support_features_protos_fake, query_features_fake,index_selected, index_selected_real)
            
            
            loss1 = cross_entropy_criterioin(-dists,query_labels_reshape)
            
            loss2 = index_selected.sum()

        
            attSelector.zero_grad()
            
            loss1 = alpha*loss1
            loss2 = beta*loss2 
            

            loss = loss1 + loss2 
        
            loss.backward()
            
            optimizer_attSelector.step()

            epoch_loss += loss.item()
            class_loss += loss1.item()
            selected_index_fake_loss += loss2.item()
            
        selected_index_list_all = [ind.sum().item() for ind in selected_index_list]
    
        val_acc, val_acc_std, val_mean, val_std = evaluate(val_loader_fsl, backboneNetwork, backboneNetwork_fake, attSelector, attSelector_real, att_size, att_size_fake, n_way, n_support, n_query)
        
        with open(save_dir+'/model_perf.txt', 'a') as f:
            f.write('Iter {:d} | Loss: {:f} | Class Loss {:f} | Selected Index Fake Loss {:f} | Val Acc {:f} | Val Std {:f} | Temp {:f}\n'.format(e, epoch_loss/len(train_loader_fsl), class_loss/len(train_loader_fsl), selected_index_fake_loss/len(train_loader_fsl), val_acc, val_acc_std, attSelector.tau))
            f.write('Iter {:d} | Train Index mean: {:f} | Train Index std {:f} \n'.format(e, np.mean(selected_index_list_all), np.std(selected_index_list_all)))
            f.write('Iter {:d} | Val Index mean {:f} | Val Index std {:f} \n'.format(e, val_mean, val_std))
            
            f.write('\n')
        if val_acc > best_acc and val_mean < 1 and val_mean != 0 and val_mean >= 0.1 and val_mean <= 0.9:
            
            test_acc, test_acc_std, test_mean, test_std = evaluate(test_loader_fsl, backboneNetwork, backboneNetwork_fake, attSelector, attSelector_real, att_size, att_size_fake, n_way, n_support, n_query)
            
            with open(save_dir+'/best_perf.txt', 'a') as f:
                f.write('Iter {:d} | Loss: {:f} | Class Loss {:f} | Selected Index Fake Loss {:f}| Val Acc {:f} | Val Std {:f} | Test Acc {:f} | Test Std {:f} | Temp {:f}\n'.format(e, epoch_loss/len(train_loader_fsl), class_loss/len(train_loader_fsl), selected_index_fake_loss/len(train_loader_fsl), val_acc, val_acc_std, test_acc, test_acc_std, attSelector.tau))
                #f.write('Iter {:d} | Train Index mean: {:f} | Train Index std {:f} | Val Index mean {:f} | Val Index std {:f}| Test Index mean {:f} | Test Index std {:f}\n'.format(e, train_mean, train_std, val_mean, val_std, test_mean, test_std))
                f.write('Iter {:d} | Train Index mean: {:f} | Train Index std {:f} \n'.format(e, np.mean(selected_index_list_all), np.std(selected_index_list_all)))
                f.write('Iter {:d} | Val Index mean {:f} | Val Index std {:f} | Test Index mean {:f} | Test Index std {:f} \n'.format(e, val_mean, val_std, test_mean, test_std))

                f.write('\n')
            torch.save({
                'epoch' : e,
                'model_state_dict' : attSelector.state_dict(),
                'val_acc' : val_acc,
                'train_loss' : epoch_loss,
                'best_acc' : best_acc,
                }, os.path.join(save_dir,'attSelector-best.pth.tar'))
            best_acc = val_acc

        if e > 0 and e % 25 == 0 and attSelector.tau > 0.5:
            attSelector.tau = attSelector.tau / 2