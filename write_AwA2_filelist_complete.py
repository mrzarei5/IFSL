import os
import random
from os import listdir
from os.path import isfile, isdir, join

import numpy as np
import argparse
import torch

parser = argparse.ArgumentParser(description='write_AwA2_filelist')


parser.add_argument('--dataset', default='C:/My Files/Carleton/Thesis/research/datasets/AwA2', help='dataset location')

args = parser.parse_args()


cwd = join(args.dataset, 'Animals_with_Attributes2')
#cwd = join(os.getcwd(), 'Animals_with_Attributes2') 

text_files = cwd

data_path = join(cwd,'JPEGImages') #should be changed
savedir = cwd + '/'

dataset_list = ['base','val','novel']

classes = []

with open(join(text_files, 'classes.txt'),'r') as f:
    lines = f.readlines()
    for line in lines:
        class_name_list = line.strip().split()
        #classes.append(class_name_list[0]+'.'+class_name_list[1])    
        classes.append(class_name_list[1]) 
    
attribute_values =  np.loadtxt(join(text_files,'predicate-matrix-binary.txt'), dtype='i', delimiter=' ')
#pd.read_csv(join(text_files,'predicate-matrix-binary.txt'),header=None)




classes_base = classes[:30]
classes_val = classes[30:40]
classes_novel = classes[40:]

classes_dic = {'base':classes_base,'val':classes_val,'novel':classes_novel}

folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]



for dataset in dataset_list:
    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item  for item in classes])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')
    
    class_names_this = classes_dic[dataset]
    
    attribute_labels = []
    image_labels_this = []
    classfile_this = []
    for i, class_name in enumerate(class_names_this):
        folder_path = join(data_path, class_name)
        classfile_this.append( [ join(folder_path, cf) for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.')])
        random.shuffle(classfile_this[i])
        
        class_index = classes.index(class_name)
        image_labels_this = image_labels_this + np.repeat(class_index, len(classfile_this[i])).tolist()
        attr_torch = torch.tensor(attribute_values[class_index]).expand(len(classfile_this[i]),attribute_values.shape[1])
        attribute_labels.append(attr_torch)
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



'''folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
folder_list.sort()
label_dict = dict(zip(folder_list,range(0,len(folder_list))))

classfile_list_all = []

for i, folder in enumerate(folder_list):
    folder_path = join(data_path, folder)
    classfile_list_all.append( [ join(folder_path, cf) for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.')])
    random.shuffle(classfile_list_all[i])

print('Writing filelist...')
for dataset in dataset_list:
    file_list = []
    label_list = []
    for i, classfile_list in enumerate(classfile_list_all):
        if 'base' in dataset:
            if (i%2 == 0):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        if 'val' in dataset:
            if (i%4 == 1):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        if 'novel' in dataset:
            if (i%4 == 3):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()

    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item  for item in folder_list])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item  for item in file_list])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item  for item in label_list])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" %dataset)'''
