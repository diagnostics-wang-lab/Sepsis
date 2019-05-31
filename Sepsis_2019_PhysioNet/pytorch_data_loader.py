#!/usr/bin/env python

import numpy as np
import torch
from torch.utils import data
import os, sys
from driver import save_challenge_predictions
import matplotlib.pyplot as plt

#raw_dir = "D:/Sepsis Challenge/training"
#pth = 'C:/Users/Osvald/Sepsis_ML/'
raw_dir = '/home/osvald/Projects/Diagnostics/CinC_data/training'
pth = '/home/osvald/Projects/Diagnostics/CinC_data/tensors/'
pos_pth = '/home/osvald/Projects/Diagnostics/CinC_data/A_pos/'
neg_pth = '/home/osvald/Projects/Diagnostics/CinC_data/A_neg/'

def load_challenge_data(file):
    with open(file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')
    
    if column_names[-1] != 'SepsisLabel':
        print(file, ' does not have sepsis label')
        return
    
    return data

def load_data(input_directory, limit=100):

    # Find files.
    files = []
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('psv'):
            files.append(f)

    data_arr = []
    # Iterate over files.
    for f in files:
        # Load data.
        input_file = os.path.join(input_directory, f)
        data = load_challenge_data(input_file)
        data_arr.append(np.transpose(data))
        if len(data_arr) == limit:
            break

    return data_arr

def data_process(dataset, length_bins=True, class_count=True):
    '''
    turns NaN to zero and
    TODO: edit labels to match utility funciton
          currently: 1 if past t_sepsis - 6, 0 otherwise
    '''
    lengths = {} 
    label_counts = [0,0]
    time_step_counts = [0,0]
    class_vec = np.zeros(len(dataset))
    for i,pt in enumerate(dataset): #get max_len and remove NaN  
        if length_bins:
            decade = (pt.shape[1] // 10) * 10
            if decade in lengths.keys():
                lengths[decade] += 1
            else:
                lengths[decade] = 1

        #TODO: change this to turn NaN to something else?
        np.nan_to_num(pt, copy=False) #replaces NaN with zeros

        if class_count:
            time_step_counts[0] += int(pt.shape[1] - pt[-1,:].sum())
            time_step_counts[1] += int(pt[-1,:].sum())
            label_counts[bool(pt[-1,:].sum())] += 1
        
        class_vec[i] = bool(pt[-1,:].sum())
        dataset[i] = pt.T


    return dataset, class_vec, lengths, label_counts, time_step_counts

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, IDs, path):
        self.IDs = IDs #list of IDs in dataset
        self.path = path

  def __len__(self):
        return len(self.IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = str(self.IDs[index])

        # Load data and get label
        x = torch.load(self.path + ID + '.pt')
        y = x[:,-1]
        x = x[:,:-1]

        return x, y

def collate_fn(data):
    ''' Creates mini-batch tensors from the list of tuples (data, label). '''

    data.sort(key=lambda x: len(x[1]), reverse=True) #sort by descending length w/in mini-batch
    inputs, labels = zip(*data)
    seq_len = torch.as_tensor([inputs[i].shape[0] for i in range(len(inputs))], dtype=torch.double).cpu()

    out_data= torch.zeros((len(inputs), inputs[0].shape[0], inputs[0].shape[1])) # (B, max, 40) tensor of zeros
    out_labels = torch.zeros((len(inputs), labels[0].shape[0]))                  # (B, 40) tensor of zreos

    for i in range(len(inputs)): # fill in available data
        out_data[i, :inputs[i].shape[0], :] = inputs[i]
        out_labels[i, :labels[i].shape[0]] = labels[i]

    return out_data, out_labels, seq_len

#TODO: clean up and put into functions

#train_data = load_data(raw_dir, limit=14336)
#train_data, classes, lengths, l_c, ts_c = data_process(train_data)
#print(l_c)
#print(ts_c)
#for i,pt in enumerate(train_data):
#    torch.save(torch.from_numpy(pt), pth + str(i)+ '.pt') 

'''
pos = 0
neg = 0
for i,pt in enumerate(train_data):
    if classes[i] == 0 and pt.shape[0] <= 60:
        torch.save(torch.from_numpy(pt), neg_pth + str(neg)+ '.pt')
        neg += 1
    elif classes[i] == 1 and pt.shape[0] <= 60:
        torch.save(torch.from_numpy(pt), pos_pth + str(pos)+ '.pt')
        pos += 1
print(neg)
print(pos)
'''

#train_dataB = load_data('/home/osvald/Projects/Diagnostics/CinC_data/training_setB', limit=None)
#train_dataB, lengths, l_cB, ts_cB = data_process(train_data)
#print(l_cB)
#print(ts_cB)
#print('no sepsis label',l_c[0]+l_cB[0])
#print('sepsis label',l_c[1]+l_cB[1])
#print('no sepsis step',ts_c[0]+ts_cB[0])
#print('sepsis step',ts_c[1]+ts_cB[1])

#plt.bar(lengths.keys(), lengths.values(), width=5, linewidth=2, edgecolor='k',color='b')
#plt.show()

#for i,pt in enumerate(train_data):
#    torch.save(torch.from_numpy(pt), pth + str(i)+ '.pt') 
