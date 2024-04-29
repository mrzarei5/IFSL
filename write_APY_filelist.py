import os
import random
from os import listdir
from os.path import isfile, isdir, join

import numpy as np
import argparse
from csv import reader
import torch

parser = argparse.ArgumentParser(description='write_APY_filelist')


parser.add_argument('--dataset', default='C:/My Files/Carleton/Thesis/research/datasets/aPY/dataset files', help='dataset location')

args = parser.parse_args()
cwd = args.dataset
#cwd = os.getcwd() 

text_files = join(cwd,'attribute_data') 

data_path = join(cwd,'images') 

pascal_images = join(cwd,'VOCImages_bb') 
yahoo_images = join(cwd, 'ayahoo_test_images_bb')

savedir = cwd + '/'

dataset_list = ['base','val','novel']

classes_base_val = []
classes_base = []
classes_val = []
classes_novel = []

with open(join(text_files, 'classes_base_val.txt'),'r') as f:
    lines = f.readlines()
    for line in lines:
        class_name = line.strip().split()[0]
        #classes.append(class_name_list[0]+'.'+class_name_list[1])    
        classes_base_val.append(class_name) 

classes_val_index = [5, 0, 14, 6, 8]
classes_base_index = [1, 2, 3, 4, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]

for val_index in classes_val_index:
    classes_val.append(classes_base_val[val_index])

for base_index in classes_base_index:
    classes_base.append(classes_base_val[base_index])
    
with open(join(text_files, 'classes_novel.txt'),'r') as f:
    lines = f.readlines()
    for line in lines:
        class_name = line.strip().split()[0]
        #classes.append(class_name_list[0]+'.'+class_name_list[1])    
        classes_novel.append(class_name) 

classes_to_images = {}
image_to_image_index = {} #to find attributes
atts_list = []

image_index = 0
with open(join(text_files, 'ayahoo_test_ed.txt'),'r') as f:
    lines = f.readlines()
    for line in lines:
        lst = line.split()
        image_name = lst[0]
        class_name = lst[1]
        boundings = [int(j) for j in lst[2:6]]
        atts =  [float(j) for j in lst[6:]]
        image_address = join(yahoo_images,image_name)
        images = classes_to_images.get(class_name,[])
        images.append(image_address)
        classes_to_images[class_name] = images
        image_to_image_index[image_address] = image_index      
        atts_list.append(atts)
        image_index += 1

with open(join(text_files, 'apascal_train_ed.txt'),'r') as f:
    lines = f.readlines()
    for line in lines:
        lst = line.split()
        image_name = lst[0]
        class_name = lst[1]
        boundings = [int(j) for j in lst[2:6]]
        atts =  [float(j) for j in lst[6:]]
        image_address = join(pascal_images,image_name)
        images = classes_to_images.get(class_name,[])
        images.append(image_address)
        classes_to_images[class_name] = images
        image_to_image_index[image_address] = image_index 
        atts_list.append(atts)
        image_index += 1

with open(join(text_files, 'apascal_test_ed.txt'),'r') as f:
    lines = f.readlines()
    for line in lines:
        lst = line.split()
        image_name = lst[0]
        class_name = lst[1]
        boundings = [int(j) for j in lst[2:6]]
        atts =  [float(j) for j in lst[6:]]
        image_address = join(pascal_images,image_name)
        images = classes_to_images.get(class_name,[])
        images.append(image_address)
        classes_to_images[class_name] = images
        image_to_image_index[image_address] = image_index
        atts_list.append(atts)
        image_index += 1

attribute_values = np.zeros((len(image_to_image_index.keys()),64)) 
for i,attributes in enumerate(atts_list):
    attribute_values[i] = np.array(attributes)



classes = list(classes_to_images.keys())
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
        atts_this_class = torch.cat(atts_this_class).view(-1,64)
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

'''

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

    fo.close()'''

