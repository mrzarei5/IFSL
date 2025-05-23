import torch
from torchvision import transforms
import json
import os
from PIL import Image
import numpy as np

class AttDataset(torch.utils.data.Dataset):
    def __init__(self, att_dir, split, aug, input_size):

        data_file = os.path.join(att_dir, split + '.json')
        with open(data_file, 'r') as f:
            self.meta = json.load(f)

        attr_file = os.path.join(att_dir, '{}_attr.pt'.format(split))
        attr_data = torch.load(attr_file)
        self.attr_labels = attr_data['attr_labels']
        self.aug = aug

        if aug:
            self.transform = transforms.Compose([
                transforms.Resize(input_size), 
                transforms.ColorJitter(brightness=0.4, 
                                       contrast=0.4,
                                       saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(input_size),  
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
        self.img_labels = np.array(self.meta['image_labels'])

    def get_img(self, idx):
        img_path = self.meta['image_names'][idx]
        return self.transform(Image.open(img_path).convert('RGB'))

    def __getitem__(self,idx):
        return self.get_img(idx), self.attr_labels[idx]

    def __len__(self):
        return len(self.img_labels)

class FSLDataset(torch.utils.data.Dataset):
    def __init__(self, att_dir, split, n_episode, n_way, n_support, n_query,
                 aug, input_size):
        
        data_file = os.path.join(att_dir, split + '.json')
        
        self.n_episode = n_episode
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query

        with open(data_file, 'r') as f:
            self.meta = json.load(f)

        attr_file = os.path.join(att_dir, '{}_attr.pt'.format(split))
        attr_data = torch.load(attr_file)
        self.attr_labels = attr_data['attr_labels']
        self.aug = aug

        if aug:
            self.transform = transforms.Compose([

                transforms.Resize(input_size), 
                transforms.ColorJitter(brightness=0.4, 
                                       contrast=0.4,
                                       saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(input_size),  
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
        self.img_labels = np.array(self.meta['image_labels'])
        self.idxs = np.arange(len(self.img_labels))
        self.classes = np.unique(self.img_labels)
     


    def get_img(self, idx):
        img_path = self.meta['image_names'][idx]
        return self.transform(Image.open(img_path).convert('RGB'))

    def __getitem__(self,idx):
        # sample n_way classes
        classes = self.classes[torch.randperm(len(self.classes))[:self.n_way]]
        examples = []
        attributes = []

        for cl in classes:
            curr_idxs = self.idxs[self.img_labels == cl]
            curr_idxs = curr_idxs[torch.randperm(
                len(curr_idxs))[:(self.n_query+self.n_support)]]
            curr_examples = []
            curr_attributes = []
            for curr_idx in curr_idxs:
                curr_examples.append(self.get_img(curr_idx))
                curr_attributes.append(self.attr_labels[curr_idx])
            curr_examples = torch.stack(curr_examples, axis=0)
            curr_attributes = torch.stack(curr_attributes, axis=0)
            examples.append(curr_examples)
            attributes.append(curr_attributes)

        examples = torch.stack(examples, axis=0)
        attributes = torch.stack(attributes, axis = 0)

        return examples,attributes 
    def __len__(self):
        return self.n_episode