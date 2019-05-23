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
from data_loader import load_from_file
from driver import save_challenge_predictions
import matplotlib.pyplot as plt

#TODO: add more args, including train, etc.print([pt.shape for pt in train_data])
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

train_data, train_labels = load_from_file('/home/wanglab/Osvald/CinC_data/setA')
#train_data, train_labels = load_from_file('/home/wanglab/Osvald/Sepsis/small_train')
#train_data, train_labels = load_from_file('D:\Sepsis Challenge\setA')

epochs = 10
embedding = 40
hidden_size = 64
num_layers = 2

def pt_count(labels, display=True):
    sepsis = 0
    total = 0
    for pt in train_labels:
        sepsis += np.count_nonzero(pt)
        total += len(pt)
    if display:
        print('total data points: ', total)
        print('sepsis data points:', sepsis)
    n = len(train_data)
    return (total/sepsis), n

ratio, n = pt_count(train_labels)

model = lstm(embedding, hidden_size, num_layers)
model.load_state_dict(torch.load('/home/wanglab/Osvald/Sepsis/Models/lstm40_2_64/model_epoch4_A'))
model.train()

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.DoubleTensor([ratio]).to(args.device))
optimizer = optim.SGD(model.parameters(), lr=0.001)

losses = np.zeros(epochs)

# TODO: Figure out batching with different sizes
start = time.time()
for epoch in range(epochs):
    loss = 0
    for i in range(n):
        # TODO: -> move outside of loop - can't just call torch.from_numpy() on entire train_data - it is type numpy.object_
        #          might have to loop through all seperately to convert to tensor
        inputs = Variable(torch.from_numpy(train_data[i]).to(args.device))
        targets = Variable(torch.from_numpy(train_labels[i]).to(args.device)).view(-1,1,1)

        optimizer.zero_grad()
        outputs, hidden_state = model(inputs, None)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses[epoch] += loss.data
        if i == 0:
            print('beginning epoch', i)
        if i == int(n/20):
            print('epoch progress: ' + str(int(n/i*100)) + '%')

    losses[epoch] = losses[epoch]/n
    print('Epoch', epoch, 'loss:',losses[epoch])
    print('total runtime:', str(round(time.time() - start, 2)))
    torch.save(model.state_dict(), '/home/wanglab/Osvald/Sepsis/Models/lstm40_2_64/model_epoch%s_A' % (epoch+1+4)) #remove +4
    np.save('/home/wanglab/Osvald/Sepsis/Models/lstm40_2_64/losses', losses)

plt.plot(losses)
plt.show()

def get_sepsis_score(data, model):
    x_mean = np.array([
        83.8996, 97.0520,  36.8055,  126.2240, 86.2907,
        66.2070, 18.7280,  33.7373,  -3.1923,  22.5352,
        0.4597,  7.3889,   39.5049,  96.8883,  103.4265,
        22.4952, 87.5214,  7.7210,   106.1982, 1.5961,
        0.6943,  131.5327, 2.0262,   2.0509,   3.5130,
        4.0541,  1.3423,   5.2734,   32.1134,  10.5383,
        38.9974, 10.5585,  286.5404, 198.6777])
    x_std = np.array([
        17.6494, 3.0163,  0.6895,   24.2988, 16.6459,
        14.0771, 4.7035,  11.0158,  3.7845,  3.1567,
        6.2684,  0.0710,  9.1087,   3.3971,  430.3638,
        19.0690, 81.7152, 2.3992,   4.9761,  2.0648,
        1.9926,  45.4816, 1.6008,   0.3793,  1.3092,
        0.5844,  2.5511,  20.4142,  6.4362,  2.2302,
        29.8928, 7.0606,  137.3886, 96.8997])
    c_mean = np.array([60.8711, 0.5435, 0.0615, 0.0727, -59.6769, 28.4551])
    c_std = np.array([16.1887, 0.4981, 0.7968, 0.8029, 160.8846, 29.5367])

    x = data[-1, 0:34]
    c = data[-1, 34:40]
    x_norm = np.nan_to_num((x - x_mean) / x_std)
    c_norm = np.nan_to_num((c - c_mean) / c_std)

    beta = np.array([
        0.1806,  0.0249, 0.2120,  -0.0495, 0.0084,
        -0.0980, 0.0774, -0.0350, -0.0948, 0.1169,
        0.7476,  0.0323, 0.0305,  -0.0251, 0.0330,
        0.1424,  0.0324, -0.1450, -0.0594, 0.0085,
        -0.0501, 0.0265, 0.0794,  -0.0107, 0.0225,
        0.0040,  0.0799, -0.0287, 0.0531,  -0.0728,
        0.0243,  0.1017, 0.0662,  -0.0074, 0.0281,
        0.0078,  0.0593, -0.2046, -0.0167, 0.1239])
    rho = 7.8521
    nu = 1.0389

    xstar = np.concatenate((x_norm, c_norm))
    exp_bx = np.exp(np.dot(xstar, beta))
    l_exp_bx = pow(4 / rho, nu) * exp_bx

    score = 1 - np.exp(-l_exp_bx)
    label = score > 0.45

    return score, label