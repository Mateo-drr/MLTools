# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:04:55 2024

@author: Mateo-drr
"""

from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
    #JUST THE LENGTH OF THE DATASET
        return len(self.data)

    def __getitem__(self, idx):    
    #TAKE ONE ITEM FROM THE DATASET
        data = self.data[idx]
        data = torch.tensor(data)

        return {'img':data,
                }