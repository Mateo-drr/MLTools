# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 17:09:10 2024

@author: Mateo-drr
"""

from dataset import CustomDataset
from torch.utils.data import DataLoader
from model import SampleNet
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

#PARAMS
lr=0.01
numEpochs=100

data=[]

train_ds = CustomDataset(data)

train_dl = DataLoader(train_ds)

# Instantiate the model
model = SampleNet()
# Define a loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(numEpochs):    
    trainLoss=0
    for sample in tqdm(train_dl, desc=f"Epoch {epoch+1}/{numEpochs}"):
        lbl,inp = sample
        out = model(sample)
        
        optimizer.zero_grad()    
        loss = criterion(out, lbl)  # Compute loss
        loss.backward()             # Backward pass
        optimizer.step()        

        trainLoss += loss.item()
        
    avg_loss = trainLoss / len(train_dl)    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')


