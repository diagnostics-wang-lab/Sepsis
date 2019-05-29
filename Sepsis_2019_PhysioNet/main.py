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
from model import lstm
from pytorch_data_loader import Dataset, collate_fn
from driver import save_challenge_predictions
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

#TODO: make this a bit nicer
#data_path = 'C:/Users/Osvald/Sepsis_ML/'
data_path = '/home/osvald/Projects/Diagnostics/CinC_data/tensors/'
save_path = '/home/osvald/Projects/Diagnostics/Sepsis/Models/'
model_name = 'lstm/lr0025_ratio50'

#TODO: add more args, including train/test, etc.
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    args.device = torch.device('cpu')
    torch.set_default_tensor_type('torch.DoubleTensor')

partition = dict([])
#TODO: fix batching so that partions don't need to be multiples of batch size
#TODO: add random split somehow
#TODO: integrate data from training set B
partition['train'] = list(range(14336))
partition['validation'] = list(range(14336,19456))

#TODO: control with args
#       be careful since some parameters are model specic!
epochs = 100
embedding = 40
hidden_size = 64
num_layers = 2
batch_size = 128
save_rate = 10
l_r = 0.0025

train_data = Dataset(partition['train'], data_path)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

val_data = Dataset(partition['validation'], data_path)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

ratio = 50 # TODO: manually find ratio of sepsis occurence

model = lstm(embedding, hidden_size, num_layers, batch_size, args.device)
''' for loading previous model'''
#TODO: make this controled by an arg when calling
#TODO: also load losses and accuracy for graphing and add ability to continue them
#model.load_state_dict(torch.load('/home/osvald/Projects/Diagnostics/Sepsis/Models/lstm/model_epoch20'))
#model.eval()

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.DoubleTensor([ratio]).to(args.device))
optimizer = optim.SGD(model.parameters(), lr=l_r)

train_losses = np.zeros(epochs)
val_losses = np.zeros(epochs)
# train accuracy
train_pos_acc = np.zeros(epochs)
train_neg_acc = np.zeros(epochs)
# val accuracy
val_pos_acc = np.zeros(epochs)
val_neg_acc = np.zeros(epochs)

start = time.time()
for epoch in range(epochs):
    # Training
    running_loss = 0
    pos_total, pos_correct = 0, 0
    neg_total, neg_correct = 0, 0
    for batch, labels, seq_len in train_loader:
        # pass to GPU if available
        batch, labels = batch.to(args.device), labels.to(args.device)
        max_len = labels.shape[1]

        optimizer.zero_grad()
        outputs = model(batch, seq_len, max_len, batch_size)
        outputs = outputs.view(-1, max_len)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.cpu().data.numpy()/seq_len.cpu().sum().numpy()
    
        # Train Accuracy
        for i in range(labels.shape[0]):
            targets = labels.data[i,:int(seq_len[i])].cpu().numpy()
            predictions = torch.round(torch.sigmoid(outputs.data[i, :int(seq_len[i])])).cpu().numpy()
            match = targets == predictions
            pos_total += (targets == 1).sum()
            neg_total += (targets == 0).sum()
            pos_correct += (match * (targets == 1)).sum()
            neg_correct += (match * (targets == 0)).sum()

    train_losses[epoch] = running_loss/len(train_loader)
    train_pos_acc[epoch] = pos_correct/pos_total
    train_neg_acc[epoch] = neg_correct/neg_total

    # Validation
    running_loss = 0
    pos_total, pos_correct = 0, 0
    neg_total, neg_correct = 0, 0
    with torch.set_grad_enabled(False):
        for batch, labels, seq_len in val_loader:
            # pass to GPU if available
            batch, labels = batch.to(args.device), labels.to(args.device)
            max_len = labels.shape[1]

            outputs = model(batch, seq_len, max_len, batch_size)
            outputs = outputs.view(-1, max_len)
            loss = criterion(outputs, labels)
            running_loss += loss.cpu().data.numpy()/seq_len.cpu().sum().numpy()
            
            # Validation Accuracy
            for i in range(labels.shape[0]):
                targets = labels.data[i,:int(seq_len[i])].cpu().numpy()
                predictions = torch.round(torch.sigmoid(outputs.data[i, :int(seq_len[i])])).cpu().numpy()
                match = targets == predictions
                pos_total += (targets == 1).sum()
                neg_total += (targets == 0).sum()
                pos_correct += (match * (targets == 1)).sum()
                neg_correct += (match * (targets == 0)).sum()

        val_losses[epoch] = running_loss/len(val_loader)
        val_pos_acc[epoch] = pos_correct/pos_total
        val_neg_acc[epoch] = neg_correct/neg_total

    print('Epoch', epoch+1, 'train avg loss:', train_losses[epoch], 'validation avg loss:', val_losses[epoch])
    print('Epoch', epoch+1, 'train pos acc:', train_pos_acc[epoch], 'validation pos acc:', val_pos_acc[epoch])
    print('Epoch', epoch+1, 'train neg acc:', train_neg_acc[epoch], 'validation neg acc:', val_neg_acc[epoch])
    print('total runtime:', str(round(time.time() - start, 2)))

    np.save(save_path + model_name +'/train_losses', train_losses)
    np.save(save_path + model_name +'/val_losses', val_losses)
    np.save(save_path + model_name +'/train_pos_acc', train_pos_acc)
    np.save(save_path + model_name +'/train_neg_acc', train_neg_acc)
    np.save(save_path + model_name +'/val_pos_acc', val_pos_acc)
    np.save(save_path + model_name +'/val_neg_acc', val_neg_acc)

    if (epoch+1) % save_rate ==0: #save model dict
       torch.save(model.state_dict(), save_path + model_name + '/model_epoch%s' % (epoch+1))

#TODO: update this, shouldn't be hard coded
plt.subplot(1,2,1)   
plt.plot(train_losses, label='train')
plt.plot(val_losses, label='val')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Weighted BCE Loss')
plt.subplot(1,2,2)
plt.plot(train_pos_acc, label='train_pos')
plt.plot(train_neg_acc, label='train_neg')
plt.plot(val_pos_acc, label='val_pos')
plt.plot(val_neg_acc, label='val_neg')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.suptitle('LSTM model')
plt.show()