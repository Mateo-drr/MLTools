# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 17:13:55 2024

@author: Mateo-drr
"""

import torch
import torch.nn as nn

class SampleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 5)
        self.fc2 = nn.Linear(5, 16)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    