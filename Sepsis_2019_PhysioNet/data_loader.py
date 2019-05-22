#!/usr/bin/env python

import numpy as np
import os, sys
from driver import save_challenge_predictions

#train_dir = "D:/Sepsis Challenge/training"
train_dir = '/home/wanglab/Osvald/CinC_data/training_setB'

def load_challenge_data(file, split=True):
    with open(file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')
    
    if column_names[-1] != 'SepsisLabel':
        print(file, ' does not have sepsis label')
        return
    elif split:
        labels = data[:, -1]
        column_names = column_names[:-1]
        data = data[:, :-1]
    else:
        data, labels = data, None
    
    return data, labels

def load_data(input_directory, limit=100, split=True):

    # Find files.
    files = []
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('psv'):
            files.append(f)

    # TODO: implement output (Maybe scrap)
    #if not os.path.isdir(output_directory):
    #    os.mkdir(output_directory)

    data_arr = []
    label_arr = []
    # Iterate over files.
    for f in files:
        # Load data.
        input_file = os.path.join(input_directory, f)
        data,labels = load_challenge_data(input_file, split)
        data_arr.append(np.transpose(data))
        label_arr.append(labels)
        if len(data_arr) == limit:
            break

    return data_arr, label_arr

def data_process(dataset, expand_dims=False):
    '''
    preprocessing - expand dims to match largest entry
    output is shape [n, time_steps, 40] np array
    each row is an hours worth of data
    TODO: edit labels to match utility funciton
          currently: 1 if past t_sepsis - 6, 0 otherwise
    '''
    n = len(dataset) # number of patients
    max_len = 0
    for i,pt in enumerate(dataset): #get max_len and remove NaN
        if pt.shape[1] > max_len:
            max_len = pt.shape[1]
        np.nan_to_num(pt, copy=False) #replaces NaN with zeros
        dataset[i] = pt.T

    if not expand_dims:
        return dataset

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

def save_to_file(name, data, labels):
    np.save(name + '_data', data)
    np.save(name + '_labels', labels)

def load_from_file(name):
    data = np.load(name + '_data.npy', allow_pickle=True)
    labels = np.load(name + '_labels.npy', allow_pickle=True)
    print('\nLoaded data of shape:', data.shape)
    print('Loaded labels of shape:', labels.shape, '\n')
    return data, labels

''' Load with no resizing example '''
#train_data, train_labels = load_data(train_dir, limit=None, split=True)
#train_data = data_process(train_data, expand_dims=False) # only tuns NaNs to zeros
#save_to_file('/home/wanglab/Osvald/CinC_data/setB', train_data, train_labels)

'''Load with resizing example'''
#train_data, _= load_data(train_dir, limit=500, split=False)
#train_data, train_labels = data_process(train_data, expand_dims=True)

'''padding'''
train_data, train_labels = load_data(train_dir, limit=None, split=True)
train_data = data_process(train_data, expand_dims=False) # only tuns NaNs to zeros
lengths = [len(label) for label in train_labels]