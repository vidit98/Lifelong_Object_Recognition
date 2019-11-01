import sys
import time
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms



class BatchData(Dataset):

    def format_images(self, path, datatype, batch_index):
        path_prefix = '{}/{}/batch{}/'.format(path, datatype, batch_index)
        path_prefix = os.path.join(path,datatype,'task'+str(batch_index))+'/'
        table = pd.read_csv(path_prefix + 'labels.csv')
        table = table.sample(frac=1)
        data_list = [path +"/" + filename for filename in table['file name'].tolist()]
        label_list = table['label'].tolist()

        return data_list, label_list

    def __init__(self, path, datatype, batch_index, transforms):
        self.transforms = transforms
        self.data_list, self.label_list = self.format_images(path, datatype, batch_index)

        # print a summary
        print('Load {} batch {} have {} images '.format(datatype, batch_index, len(self.data_list)))

    def __getitem__(self, idx):
        img = self.data_list[idx]
        img = Image.open(img)
        label = int(self.label_list[idx])
        img = self.transforms(img)
        return img, label #self.data_list[idx].split('/')[-1].split('.')[0]

    def __len__(self):
        return len(self.data_list)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = 0
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

