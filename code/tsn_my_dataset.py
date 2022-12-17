from PIL import Image
import torch
from torch.utils.data import Dataset
from mytransformer import RandomResizedCrop
import pandas as pd
import numpy as np
from torchvision import transforms
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import os

from tsn_utils import TSN_yuchuli


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list,frames_per_clip,num_segments,jitter_mode,name,ratio=0.1):
        self.images_path = images_path
        self.images_class = images_class
        self.frames_per_clip= frames_per_clip
        self.num_segments= num_segments
        self.jitter_mode = jitter_mode
        self.name = name
        self.ratio = ratio

    def __len__(self):
        return len(self.images_path)
 
    def __getitem__(self, item):
        img = TSN_yuchuli(self.images_path[item],self.frames_per_clip,self.num_segments,self.jitter_mode,self.name,self.ratio)
         
        label = self.images_class[item]
        path = self.images_path[item]
        return img, label, path

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels ,path = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels,path
