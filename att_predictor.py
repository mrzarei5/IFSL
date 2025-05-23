import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from models import AttPredictor
from datasets import AttDataset
from utils import parse_args, setup_logger
from config import dataset_config

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Weight calculation function
def atts_weight_calculate(labels, att_size): 
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
    acc_all_0, acc_all_1, acc_all = [], [], []
    for pred, gt in zip(predicted, true):
        matrix = confusion_matrix(gt, pred)
        per_class_num = matrix.sum(axis=1)
        if len(per_class_num) == 2 and  per_class_num.min() > 0:
            vec = np.diag(matrix) / per_class_num
            acc_all_0.append(vec[0])
            acc_all_1.append(vec[1])
            acc_all.append(np.sum(np.diag(matrix)) / np.sum(matrix))
    return np.mean(acc_all_0), np.mean(acc_all_1), np.mean(acc_all)

# Evaluate the performance of the model in attribute prediction
@torch.no_grad()
def evaluate_attributes(dataloader, model): 
    model.eval()

    acc_all = []
    for images, atts in dataloader:
        images, atts = images.to(device), atts.to(device)
        atts_pred = model(images)
        atts_pred_rounded = atts_pred.round()
        acc_0, acc_1, acc = calculate_acc_atts_class_wise(atts_pred_rounded.cpu().numpy(), atts.cpu().numpy().round())
        acc_all.append((acc_0, acc_1, acc))
    return np.mean([x[0] for x in acc_all]), np.std([x[0] for x in acc_all]), np.mean([x[1] for x in acc_all]), np.std([x[1] for x in acc_all]), np.mean([x[2] for x in acc_all]), np.std([x[2] for x in acc_all])

    

if __name__=='__main__':
    np.random.seed(10)
    args = parse_args('att_predictor')

    # Define log directory based on save_dir and dataset
    save_dir = os.path.join(args.save_dir_predictor, args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    # Initialize loggers
    logger = setup_logger('att_perf', os.path.join(save_dir, 'att_perf.txt'))
    best_logger = setup_logger('best_att_perf', os.path.join(save_dir, 'best_att_perf.txt'))

    num_workers = args.num_workers
    dataset_dir = args.dataset_dir

    if args.dataset not in dataset_config:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    config = dataset_config[args.dataset]
    att_size = config['att_size']
    hid_dim_list = config['hid_dim_list']
    input_size = config['input_size']

    model = AttPredictor(hid_dim_list= hid_dim_list, att_size= att_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr_backbone_network) 
    att_criterion = nn.BCELoss(reduction='none')

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    train_dataset = AttDataset(dataset_dir, 'base', aug = True, input_size= input_size)
    val_dataset = AttDataset(dataset_dir, 'val', aug = False, input_size= input_size)
    test_dataset = AttDataset(dataset_dir, 'novel', aug = False, input_size= input_size)
    
    train_loader_att = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers = num_workers, pin_memory = True)
    val_loader_att = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers = num_workers, pin_memory = True)
    test_loader_att = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers = num_workers, pin_memory = True)

    best_att_perf = 0
    for e in tqdm(range(args.n_iter)):
        class_loss = 0
        model.train()
        
        for images, atts in train_loader_att:    
            images, atts = images.to(device), atts.to(device)
            atts_predicted = model(images)
            atts_zeros = torch.clone(atts)
            atts_zeros[atts_zeros==-1] = 0
            att_weights = atts_weight_calculate(atts_zeros, att_size)
            loss = (att_criterion(atts_predicted, atts) * att_weights).mean()
            
            optimizer.zero_grad()
            class_loss += loss.item()
            loss.backward()
            optimizer.step()

        acc_val_data = evaluate_attributes(val_loader_att,model) #Evaluate attribute selector on validation classes attribute prediction
        
        logger.info('Iter {:d} | Val 0 Mean Acc {:f} | Val 0 Std {:f} | Val 1 Mean Acc {:f} | Val 1 Std {:f} | Val Mean Acc {:f} | Val Std {:f}'.format(e+1, acc_val_data[0], acc_val_data[1], acc_val_data[2], acc_val_data[3], acc_val_data[4], acc_val_data[5]))

        if acc_val_data[4] > best_att_perf: #Saving model with its complete performance reports
                        
            acc_train_data = evaluate_attributes(train_loader_att,model)
            acc_test_data = evaluate_attributes(test_loader_att,model)
            with open(save_dir+'/best_att_perf.txt', 'a') as f:
                best_logger.info('Iter {:d} | Train 0 Mean Acc {:f} | Train 0 Std {:f} | Train 1 Mean Acc {:f} | Train 1 Std {:f} | Train Mean Acc {:f} | Train Std {:f}'.format(e, acc_train_data[0], acc_train_data[1], acc_train_data[2], acc_train_data[3], acc_train_data[4], acc_train_data[5]))
                best_logger.info('Iter {:d} | Test 0 Mean Acc {:f} | Test 0 Std {:f} | Test 1 Mean Acc {:f} | Test 1 Std {:f} | Test Mean Acc {:f} | Test Std {:f}'.format(e, acc_test_data[0], acc_test_data[1], acc_test_data[2], acc_test_data[3], acc_test_data[4], acc_test_data[5]))
                
            best_att_perf = acc_val_data[4]
            torch.save({
                'epoch' : e,
                'model_state_dict' : model.state_dict(),
                }, os.path.join(save_dir,'backboneNetwork-best.pth.tar'))
        