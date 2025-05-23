import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
import numpy as np

from models import AttSelector, AttPredictor
from utils import parse_args, setup_logger
from datasets import FSLDataset
from config import dataset_config

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_fsl_distances(prototypes,query_features,selected_attribute_mask): 
    """
    Compute distance between query features and prototypes,
    considering only the selected attributes (via a binary mask).
    
    Args:
        prototypes: Prototype vectors for each class
        query_features: Query feature vectors
        selected_attribute_mask: Mask indicating which attributes are selected

    Returns:
        Distance matrix between queries and prototypes. 
    """
    
    query_features_ = torch.unsqueeze(query_features.view(query_features.size(0)*query_features.size(1),query_features.size(2)),1)
    prototypes_ = torch.unsqueeze(prototypes,0)

    query_features_ = query_features_.expand(-1,  prototypes_.size(1), -1)   
    
    prototypes_ = prototypes_.expand(query_features_.size(0), -1, -1)
    
    diff = (query_features_ - prototypes_)**2

    masked = diff * selected_attribute_mask

    return masked.sum(2)

def compute_accuracy(prototypes,query_features,selected_attribute_mask, n_query): 

    """
    Computes accuracy for a given few-shot task.

    Args:
        prototypes: Prototype vectors for classes
        query_features: Query feature vectors
        selected_attribute_mask: Mask for selected attributes
        n_query: Number of query samples per class

    Returns:
        Accuracy as a percentage
    """


    query_features_ = torch.unsqueeze(query_features,1)
    prototypes_ = torch.unsqueeze(prototypes,0)
    
    query_features_ = query_features_.expand(-1,prototypes_.size(1),-1)   
    prototypes_ = prototypes_.expand(query_features_.size(0),-1,-1)

    diff = (query_features_ - prototypes_)**2
    masked = diff * selected_attribute_mask
    dist  = masked.sum(2)
  
    min_args = torch.argmin(dist,1)

    query_labels = torch.tensor([[i] for i in range(prototypes.size(0))]).to(device)
    query_labels = query_labels.repeat(1,n_query).flatten()    
    correct = (min_args == query_labels).sum().item()
    return correct / len(query_labels) * 100

class AttributeSelectorTrainer:
    def __init__(self, config):
        self.args = config
        self.device = device
        self._initialize_dataset_config()
        self._prepare_data()
        self._build_models()

    def _initialize_dataset_config(self):
        config = dataset_config[self.args.dataset]
        self.att_size = config['att_size']
        self.hid_dim_list = config['hid_dim_list']
        self.input_size = config['input_size']

    def _prepare_data(self):
        self.train_loader = self._get_loader('base', self.args.n_episode_train, aug=True)
        self.val_loader = self._get_loader('val', 600)
        self.test_loader = self._get_loader('novel', 600)


    def _get_loader(self, split, n_episode, aug=False):
        dataset = FSLDataset(self.args.dataset_dir, split, n_episode,
                             self.args.n_way, self.args.n_support, self.args.n_query, aug, self.input_size)
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle='base' in split,
                                           num_workers=self.args.num_workers, pin_memory=True)

    def _build_models(self):
        self.attribute_predictor = AttPredictor(hid_dim_list=self.hid_dim_list, att_size=self.att_size).to(self.device)
        state = torch.load(os.path.join(self.args.save_dir_predictor, self.args.dataset, 'backboneNetwork-best.pth.tar'))
        self.attribute_predictor.load_state_dict(state['model_state_dict'])
        for p in self.attribute_predictor.parameters():
            p.requires_grad = False

        self.selector = AttSelector(tau=self.args.tau, att_size=self.att_size).to(self.device)
        self.optimizer = optim.Adam(self.selector.parameters(), lr=self.args.lr_att_selector)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        best_acc = 0
        selector_save_dir = os.path.join(self.args.save_dir_selector, self.args.dataset,
                                f"{self.args.n_way}_way_{self.args.n_support}_shot",
                                f"l1_{self.args.alpha}_l2_{self.args.gamma}")
        os.makedirs(selector_save_dir, exist_ok=True)

        model_logger = setup_logger('model_perf', os.path.join(selector_save_dir, 'model_perf.txt'), console=True)
        best_logger = setup_logger('best_perf', os.path.join(selector_save_dir, 'best_perf.txt'), console=False)

        for epoch in tqdm(range(self.args.n_iter), desc="Training Epochs"):
            best_acc = self._train_one_epoch(epoch, selector_save_dir, best_acc, model_logger, best_logger)
            if epoch > 0 and epoch % 25 == 0 and self.selector.tau > 0.5:
                self.selector.tau /= 2

    def _train_one_epoch(self, epoch, save_dir, best_acc, model_logger, best_logger):
        self.selector.train()
        epoch_loss = class_loss = l1_regularization_loss = 0
        selected_attribute_list = []

        for i, data in enumerate(self.train_loader):
            images, _atts = data[0][0].to(self.device), data[1][0].to(self.device)
            images = images.contiguous().view(-1, *images.shape[2:])
            predicted_atts = self.attribute_predictor(images).contiguous().view(self.args.n_way, self.args.n_support + self.args.n_query, -1)

            atts_support = predicted_atts[:, :self.args.n_support, :]
            prototypes = atts_support.mean(1).view(-1, self.att_size)
            selected_attribute_mask, _ = self.selector(prototypes)
            selected_attribute_list.append(selected_attribute_mask.detach())

            query_labels = torch.arange(self.args.n_way).repeat_interleave(self.args.n_query).to(self.device)
            query_features = predicted_atts[:, self.args.n_support:, :].contiguous().view(self.args.n_way, self.args.n_query, -1)

            dists = compute_fsl_distances(prototypes, query_features, selected_attribute_mask)
            classification_loss = self.args.alpha * self.criterion(-dists, query_labels)
            l1_regularization_loss = self.args.gamma * selected_attribute_mask.sum()
            loss = classification_loss + l1_regularization_loss

            self.selector.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            class_loss += classification_loss.item()
            l1_regularization_loss += l1_regularization_loss.item()

        val_acc, val_std, val_mean, val_std_idx = self.evaluate(self.val_loader)
        selected_vals = [x.sum().item() for x in selected_attribute_list]
        
        self._log_epoch(model_logger, epoch, epoch_loss, class_loss, l1_regularization_loss, selected_vals, val_acc, val_std, val_mean, val_std_idx)

        if val_acc > best_acc:
            test_acc, test_std, test_mean, test_std_idx = self.evaluate(self.test_loader)
            self._log_best(best_logger, epoch, epoch_loss, class_loss, l1_regularization_loss, val_acc, val_std, test_acc, test_std,
                           selected_vals, val_mean, val_std_idx, test_mean, test_std_idx)
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.selector.state_dict(),
                'val_acc': val_acc,
                'train_loss': epoch_loss,
                'best_acc': best_acc
            }, os.path.join(save_dir, 'attSelector-best.pth.tar'))
            best_acc = val_acc

        return best_acc
    
    def evaluate(self, loader):
        self.attribute_predictor.eval()
        self.selector.eval()
        acc_list = []
        index_selected_list = []
        with torch.no_grad():
            for i, data in enumerate(loader):
                images, _atts = data[0][0].to(self.device), data[1][0].to(self.device)
                images = images.contiguous().view(-1, *images.shape[2:])
                predicted_atts = self.attribute_predictor(images).contiguous().view(self.args.n_way, self.args.n_support + self.args.n_query, -1)
                
                
                prototypes = predicted_atts[:, :self.args.n_support, :].mean(1).view(-1,self.att_size)
                selected_attribute_mask, _ = self.selector(prototypes)
                query_features = predicted_atts[:, self.args.n_support:, :].contiguous().view(self.args.n_way * self.args.n_query, -1)
                acc = compute_accuracy(prototypes, query_features, selected_attribute_mask, self.args.n_query)
                acc_list.append(acc)
                index_selected_list.append(selected_attribute_mask.sum().item())
        acc_arr = np.array(acc_list)
        return np.mean(acc_arr), 1.96 * np.std(acc_arr) / np.sqrt(len(loader)), np.mean(index_selected_list), np.std(index_selected_list)
    
    def _log_epoch(self, logger, epoch, loss, class_loss, l1_loss, index_list, val_acc, val_std, val_mean, val_std_idx):
        logger.info(f"Iter {epoch} | Loss: {loss:.4f} | Class Loss {class_loss:.4f} | Selected Index Loss {l1_loss:.4f} | Val Acc {val_acc:.2f} | Val Std {val_std:.2f} | Temp {self.selector.tau:.2f}")
        logger.info(f"Iter {epoch} | Train Index mean: {np.mean(index_list):.2f} | Train Index std {np.std(index_list):.2f}")
        logger.info(f"Iter {epoch} | Val Index mean {val_mean:.2f} | Val Index std {val_std_idx:.2f}")

    def _log_best(self, logger, epoch, loss, class_loss, l1_loss, val_acc, val_std, test_acc, test_std, index_list, val_mean, val_std_idx, test_mean, test_std_idx):
        logger.info(f"Iter {epoch} | Loss: {loss:.4f} | Class Loss {class_loss:.4f} | Selected Index Loss {l1_loss:.4f} | Val Acc {val_acc:.2f} | Val Std {val_std:.2f} | Test Acc {test_acc:.2f} | Test Std {test_std:.2f} | Temp {self.selector.tau:.2f}")
        logger.info(f"Iter {epoch} | Train Index mean: {np.mean(index_list):.2f} | Train Index std {np.std(index_list):.2f}")
        logger.info(f"Iter {epoch} | Val Index mean {val_mean:.2f} | Val Index std {val_std_idx:.2f} | Test Index mean {test_mean:.2f} | Test Index std {test_std_idx:.2f}")



if __name__=='__main__':
    np.random.seed(10)
    
    config = parse_args('att_selector')
    trainer = AttributeSelectorTrainer(config)
    trainer.train()