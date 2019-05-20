#!/usr/bin/env python

import numpy as np
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gpytorch
from driver import save_challenge_predictions
# load_challenge_data(file) returns data
# save_challenge_predictions(file, scores, labels)

train_dir = "D:/Sepsis Challenge/training"

def load_challenge_data(file):
    with open(file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')

    # Ignore SepsisLabel column if present.
    '''
    if column_names[-1] != 'SepsisLabel':
        print(file, ' does not have sepsis label')
        return
        labels = data[:, -1]
        column_names = column_names[:-1]
        data = data[:, :-1]'''
        
    return data

def load_data(input_directory):

    # Find files.
    files = []
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('psv'):
            files.append(f)

    # TODO: implement output
    #if not os.path.isdir(output_directory):
    #    os.mkdir(output_directory)

    data_arr = []
    label_arr = []
    # Iterate over files.
    for f in files:
        # Load data.
        input_file = os.path.join(input_directory, f)
        data = load_challenge_data(input_file)
        data_arr.append(np.transpose(data))
        # TODO: remove this length constraint
        if len(data_arr) == 500:
            break

    return data_arr #, label_arr

train_data = load_data(train_dir) #list of np arrays w 41 vars

def data_process(dataset):
    '''
    preprocessing - expand dims to match largest entry
    output is shape [n, 40, max] np array
    each row is an hours worth of data
    TODO: edit labels to match utility funciton
          currently: 1 if past t_sepsis - 6, 0 otherwise
    '''
    n = len(dataset) # number of patients
    max_len = 0
    for pt in dataset: #get max_len and remove NaN
        if pt.shape[1] > max_len:
            max_len = pt.shape[1]
        np.nan_to_num(pt, copy=False) #replaces NaN with zeros

    for i, pt in enumerate(dataset): # expand dimensions to match largest input
        diff = max_len - pt.shape[1]
        if diff:
            pt = np.append(pt, np.ones((41, diff)) * -1, axis=1)
        pt = np.expand_dims(pt, axis=0)
        #TODO: fix this very ugly workaround, here because of np flattening tendency
        if i == 0: 
            output = pt
        else:
            output = np.append(output, pt, axis=0)
    
    data, labels = output[:,:-1,:], output[:,-1,:]
    return data, labels

'''Save Data as .npy'''
#train_data, train_labels = data_process(train_data) #currently (n, 40, max_time)
#print(train_data.shape)
#print(train_labels.shape)
#np.save('nan0_miss1_setA_data', train_data)
#np.save('nan0_miss1_setA_labels', train_labels)

'''Load .npy Data''' #currently (500, 40, 258) and (500, 258)
data = np.load('nan0_miss1_setA_data.npy')
labels = np.load('nan0_miss1_setA_labels.npy')
print(data.shape)
print(labels.shape)


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