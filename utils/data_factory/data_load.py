from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split 


class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = pd.read_csv(path)
        X = df.iloc[:, 0:100]  #1-100th column is dictionary operation
        Y = df.iloc[:, 100]  #101th column is ground truth
        self.x = X.values
        self.y = Y.values
        # ensure input data is floats
        self.x = self.x.astype('float32')
        self.y = self.y.astype('float32')
       
    # number of rows in the dataset
    def __len__(self):
        return len(self.x)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.x[idx], self.y[idx]]

    def get_splits(self, n_test=0.1):
        # determine sizes
        test_size = round(n_test * len(self.x))
        train_size = len(self.x) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])

#prepare dataset
def prepare_data(path):
    # load the dataset
    dataset = CSVDataset(path)
    # calculate split
    train, test = dataset.get_splits()
    #print(train.__getitem__)
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=1, shuffle=True)
    test_dl = DataLoader(test, batch_size=1, shuffle=True)
    return train_dl, test_dl

# # prepare the data
path = 'data10000_qubitOp_details.csv'
train_loader, test_loader = prepare_data(path)
print(len(train_loader))
print(len(test_loader))
print(train_loader.shape())