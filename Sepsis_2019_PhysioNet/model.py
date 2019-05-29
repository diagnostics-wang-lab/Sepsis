#!/usr/bin/env python

import numpy as np
import os, sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import math
# for hotfix to pack_padded_sequence
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence

class lstm(nn.Module):
    '''
    lstm prototype
    input -> [40, n] physiological variable time series tensor
    output -> [n,] sepsis label tensor
    '''
    def __init__(self, embedding, hidden_size, num_layers=2, batch_size=1 ,device='cpu'):
        super(lstm, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lstm_layers = num_layers

        self.inp = nn.Linear(40, embedding) # input embedding - can be changed, potentially to small TCN
        self.rnn = nn.LSTM(embedding, hidden_size, num_layers=num_layers, batch_first=True) # RNN structure
        self.out = nn.Linear(hidden_size, 1) # output linear

        for m in self.modules():
            if isinstance(m, nn.Conv2d): #not used yet
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d): #not used yet either
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def init_hidden(self):
        hidden_a = torch.randn(self.lstm_layers, self.batch_size, self.hidden_size).to(self.device)
        hidden_b = torch.randn(self.lstm_layers, self.batch_size, self.hidden_size).to(self.device)

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)
    
    def forward(self, X, seq_len, max_len, hidden_state=None): 
        self.hidden = self.init_hidden()
        X = self.inp(X)
        X = hotfix_pack_padded_sequence(X, seq_len, batch_first=True, enforce_sorted=False)
        #X = torch.nn.utils.rnn.pack_padded_sequence(X, seq_len, batch_first=True, enforce_sorted=False)
        X, self.hidden = self.rnn(X, self.hidden)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True, padding_value=-1, total_length=max_len)
        X = self.out(X)
        return X.squeeze()
        
def hotfix_pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True):
    '''
    #TODO: if ever fixed just go back to original
    GPU errors with orig func:
    torch.nn.utils.rnn.pack_padded_sequence()
    this fix was provided on pytorch board
    ''' 
    lengths = torch.as_tensor(lengths, dtype=torch.int64)
    lengths = lengths.cpu()
    if enforce_sorted:
        sorted_indices = None
    else:
        lengths, sorted_indices = torch.sort(lengths, descending=True)
        sorted_indices = sorted_indices.to(input.device)
        batch_dim = 0 if batch_first else 1
        input = input.index_select(batch_dim, sorted_indices)

    data, batch_sizes = \
        torch._C._VariableFunctions._pack_padded_sequence(input, lengths, batch_first)
    return PackedSequence(data, batch_sizes, sorted_indices)
