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

#data_path = 'C:/Users/Osvald/Sepsis_ML/'
data_path = '/home/osvald/Projects/Diagnostics/CinC_data/tensors/'
save_path = '/home/osvald/Projects/Diagnostics/Sepsis/Models/'
model_name = 'lstm01'

#TODO: add more args, including train, etc.
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

def sort_by_seq_len(labels, pad_val=-1):
    '''
    returns descending order of array lengths ignoring pad_val entries
    '''
    seq_len = np.array([])
    for l in labels:
        arr = l.data.numpy()
        unique, counts = np.unique(arr, return_counts=True)
        counts = dict(zip(unique, counts))
        try: # will fail on max_len sequence since no -1 dict entries
            seq_len = np.append(seq_len, labels.shape[1] - counts[pad_val])
        except: seq_len = np.append(seq_len, len(l.data))
    # sort by sequence lengths
    order = torch.from_numpy(np.argsort(seq_len*-1))
    seq_len = torch.from_numpy(seq_len[order])
    return order, seq_len


partition = dict([])
#partition['train'] = list(range(12288))
#partition['validation'] = list(range(12288,15360))
partition['train'] = list(range(14400))
partition['validation'] = list(range(14400,20320))

epochs = 20
embedding = 40
hidden_size = 64
num_layers = 2
batch_size = 32
save_rate = 1
l_r = 0.001

train_data = Dataset(partition['train'], data_path)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

val_data = Dataset(partition['validation'], data_path)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

ratio = 4 # TODO: manually find ratio of sepsis occurence

model = lstm(embedding, hidden_size, num_layers, batch_size, args.device)
model.load_state_dict(torch.load('/home/osvald/Projects/Diagnostics/Sepsis/Models/lstm/model_epoch20'))
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


# TODO: Figure out batching with different sizes w/o excessive padding
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
        #seq_len = torch.LongTensor(seq_len.cpu()).to('cpu')
        #seq_len = seq_len.type(torch.int64).to('cpu')
        '''
        if labels.shape[0] != 1:
            order, seq_len = sort_by_seq_len(labels) #TODO: Fix this inefficient method of counting sequence lengths -> move to data loader (in val loop too)            
            labels = labels[order, :]
            batch = batch[order, :]
        else: # if final batch is size 1
            seq_len = torch.Tensor([max_len])'''

        optimizer.zero_grad()
        outputs = model(batch, seq_len, max_len, batch_size)
        outputs = outputs.view(-1, max_len)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.cpu().data.numpy()/seq_len.sum().numpy()
    
        
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
            
            '''
            if labels.shape[0] != 1:  
                order, seq_len = sort_by_seq_len(labels)  
                labels = labels[order, :]
                batch = batch[order, :]
            else: # if final batch is size 1
                seq_len = torch.Tensor([max_len])'''

            outputs = model(batch, seq_len, max_len, batch_size)
            outputs = outputs.view(-1, max_len)
            loss = criterion(outputs, labels)
            running_loss += loss.cpu().data.numpy()/seq_len.sum().numpy()
            
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

    if (epoch+1) % save_rate ==0:
       torch.save(model.state_dict(), save_path + model_name + '/model_epoch%s' % (epoch+1))

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