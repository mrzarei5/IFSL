import os
import random
import csv
from os import listdir
from os.path import isfile, isdir, join
import numpy as np
import torch
import json
import argparse
parser = argparse.ArgumentParser(description='write_CUB_filelist')


parser.add_argument('--dataset_files', default='./datasets/CUB/', help='dataset location')

args = parser.parse_args()

dataset_dir = args.dataset_files

images_data_dir = join(dataset_dir,'CUB_200_2011','images')

dataset_list = ['base','val','novel']
folder_list = [f for f in listdir(images_data_dir) if isdir(join(images_data_dir, f))]

print('Found %d folders' % len(folder_list))


folder_list.sort()
label_dict = dict(zip(folder_list,range(0,len(folder_list))))

classfile_list_all = []

for i, folder in enumerate(folder_list):
    folder_path = join(images_data_dir, folder)
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

    fo = open(join(dataset_dir, dataset + ".json"), "w")
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
    print("%s -OK" %dataset)

img_name_file = join(dataset_dir,'CUB_200_2011','images.txt')

name_idx_map = {}
with open(img_name_file, 'r') as f:
    rd = csv.reader(f, delimiter=' ')
    for row in rd:
        name_idx_map[row[1]] = int(row[0])-1


num_imgs = len(name_idx_map)
num_attr = 312
attr_labels = torch.zeros(num_imgs, num_attr)

attr_file = join(dataset_dir,'CUB_200_2011','attributes','image_attribute_labels.txt')

with open(attr_file, 'r') as f:
    rd = csv.reader(f, delimiter=' ')
    for row in rd:
        img_idx = int(row[0])-1
        attr_idx = int(row[1])-1
        attr_labels[img_idx, attr_idx] = int(row[2])

print('Getting image attributes...')
for split in dataset_list:
    print('Processing {}'.format(split))
    with open(join(dataset_dir,'{}.json'.format(split)), 'r') as f:
        meta = json.load(f)
    img_idxs = []
    for img_name in meta['image_names']:
        curr_idx = name_idx_map['/'.join(img_name.split('/')[-2:])]
        img_idxs.append(curr_idx)
    img_idxs = torch.LongTensor(img_idxs)
    torch.save({
        'attr_labels' : attr_labels[img_idxs],
    }, join(dataset_dir,'{}_attr.pt'.format(split)))      