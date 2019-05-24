#!/usr/bin/env python

import numpy as np
import os, sys
import time
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import math
import gpytorch # currently unused
from model import lstm
from pytorch_data_loader import Dataset
from driver import save_challenge_predictions
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

#TODO: add more args, including train, etc.
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available() and False: #remove false
    args.device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    args.device = torch.device('cpu')
    torch.set_default_tensor_type('torch.DoubleTensor')

'''remove all of these once data_loader is working '''
#train_data, train_labels = load_from_file('/home/wanglab/Osvald/CinC_data/setA')
#train_data, train_labels = load_from_file('/home/wanglab/Osvald/Sepsis/small_train')
#train_data, train_labels = load_from_file('D:\Sepsis Challenge\setA')
#train_data, train_labels = load_from_file(r'C:\Users\Osvald\Sepsis_ML\small_train')
#train_data, train_labels = load_from_file(r'C:\Users\Osvald\Sepsis_ML\test')

partition = dict([])
partition['train'] = list(range(1500))
partition['validation'] = list(range(1500,2000))

epochs = 10
embedding = 40
hidden_size = 64
num_layers = 2
batch_size = 4
save_rate = 10

train_data = Dataset(partition['train'])
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

val_data = Dataset(partition['validation'])
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

ratio = 10 # TODO: manually find ratio of sepsis occurence

model = lstm(embedding, hidden_size, num_layers, batch_size)
#model.load_state_dict(torch.load('/home/wanglab/Osvald/Sepsis/Models/lstm40_2_64/model_epoch4_A'))
#model.eval()

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.DoubleTensor([ratio]).to(args.device))
optimizer = optim.SGD(model.parameters(), lr=0.001)

train_losses = np.zeros(epochs)
val_losses = np.zeros(epochs)

# TODO: Figure out batching with different sizes w/o excessive padding
# TODO: edit loss so that it ignores -1s
start = time.time()
for epoch in range(epochs):
    # Training
    train_loss = 0
    for batch, labels in train_loader:
        # pass to GPU if available
        batch, labels = batch.to(args.device), labels.to(args.device)

        optimizer.zero_grad()
        outputs, hidden_state = model(batch, None)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_losses[epoch] += loss.data / len(train_loader)
    
    # Validation
    val_loss = 0
    with torch.set_grad_enabled(False):
        for batch, labels in val_loader:
            # pass to GPU if available
            batch, labels = batch.to(args.device), labels.to(args.device)
            
            outputs, hidden_state = model(batch, None)
            loss = criterion(outputs, labels)

            val_losses[epoch] += loss.data / len(val_loader)

    print('Epoch', epoch+1, 'train loss:', train_losses[epoch], 'validation loss:', val_losses[epoch])
    print('total runtime:', str(round(time.time() - start, 2)))

    np.save('C:/Users/Osvald/Sepsis_ML/Models/lstm_batch/', train_losses)
    np.save('C:/Users/Osvald/Sepsis_ML/Models/lstm_batch/', val_losses)
    if (epoch+1) % save_rate ==0:
       torch.save(model.state_dict(), 'C:/Users/Osvald/Sepsis_ML/Models/lstm_batch/model_epoch%s' % (epoch+1))
        
    #np.save('/home/wanglab/Osvald/Sepsis/Models/lstm40_2_64/losses', losses)
    #if (epoch+1) % save_rate ==0:
    #   torch.save(model.state_dict(), '/home/wanglab/Osvald/Sepsis/Models/lstm40_2_64/model_epoch%s_A' % (epoch+1))
    
plt.plot(train_losses)
plt.plot(val_losses)
plt.show()