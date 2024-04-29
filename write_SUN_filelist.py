import os
import random
from os import listdir
from os.path import isfile, isdir, join

import numpy as np
import argparse
from csv import reader
import torch

parser = argparse.ArgumentParser(description='write_AwA2_filelist')


parser.add_argument('--dataset', default='C:/My Files/Carleton/Thesis/research/datasets/AwA2', help='dataset location')

args = parser.parse_args()
cwd = args.dataset
#cwd = os.getcwd() 

text_files = join(cwd,'SUNAttributeDB') 

data_path = join(cwd,'images') 
savedir = cwd + '/'

dataset_list = ['base','val','novel']

#classes = []
#images = []
classes_to_images = {}
image_to_image_index = {} #to find attributes
#580,65,72
with open(join(text_files, 'images.csv'),'r') as f:
    csv_reader = reader(f)

    for i,line in enumerate(csv_reader):
        image_address = line[1][2:-2]
        class_name = '/'.join(image_address.split('/')[:-1])
        
        images = classes_to_images.get(class_name,[])
        image_address = join(data_path,image_address)
        images.append(image_address)
        classes_to_images[class_name] = images
        image_to_image_index[image_address] = i

attribute_values = np.zeros((len(image_to_image_index.keys()),102)) 
with open(join(text_files, 'attributeLabels_continuous.csv'),'r') as f:
    csv_reader = reader(f)
    for i,line in enumerate(csv_reader):
        attributes = line[1:]
        attributes = [round(float(x)) for x in attributes]
        attribute_values[i] = np.array(attributes)
classes = list(classes_to_images.keys())


classes_base = classes[:580]
classes_val = classes[580:580+65]
classes_novel = classes[580+65:]
classes_dic = {'base':classes_base,'val':classes_val,'novel':classes_novel}



for dataset in dataset_list:
    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item  for item in classes])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')
    
    class_names_this = classes_dic[dataset]
    
    image_labels_this = []
    classfile_this = []
    attribute_labels = []
    for i, class_name in enumerate(class_names_this):

        folder_path = join(data_path, class_name)
        classfile_this.append(classes_to_images[class_name])
        random.shuffle(classfile_this[i])

        class_index = classes.index(class_name)
        image_labels_this = image_labels_this + np.repeat(class_index, len(classfile_this[i])).tolist()
        
        atts_this_class = []
        for imagge_add in classfile_this[i]:
            atts = torch.tensor(attribute_values[image_to_image_index[imagge_add]])
            atts_this_class.append(atts)
        atts_this_class = torch.cat(atts_this_class).view(-1,102)
        attribute_labels.append(atts_this_class)
    attribute_labels = torch.cat(attribute_labels,0).float()
    
    torch.save({
        'attr_labels' : attribute_labels
    }, savedir +'{}_attr.pt'.format(dataset))


    image_names_this = []
 
    for classfile_list in classfile_this:
        image_names_this = image_names_this + classfile_list
        
    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item  for item in image_names_this])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')
    
    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item  for item in image_labels_this])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')

    fo.close()

