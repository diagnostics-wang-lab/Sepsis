#!/usr/bin/env python

import numpy as np
import os, sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import math

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
        X = torch.nn.utils.rnn.pack_padded_sequence(X, seq_len, batch_first=True, enforce_sorted=False)
        X, self.hidden = self.rnn(X, self.hidden)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True, padding_value=-1, total_length=max_len)
        X = self.out(X)
        return X.squeeze()
        

    ''' #Old code w/o packing - loss includes padding
    def step(self, time_step, hidden_state=None):
        time_step = self.inp(time_step) 
        output, hidden_state = self.rnn(time_step.view(time_step.shape[0], 1, time_step.shape[1]), hidden_state) # unsqeeze not working
        output = self.out(output.squeeze(1)).squeeze(1)
        return output, hidden_state


    def forward(self, in_states, hidden_state=None): 
        steps = in_states.shape[1]
        outputs = Variable(torch.zeros(steps, self.batch_size))
        for i in range(steps):
            time_step = in_states[:,i,:]
            output, hidden_state = self.step(time_step, hidden_state)
            outputs[i] = output
        return torch.t(outputs), hidden_state #TODO:is hidden_state necessary?
    '''
