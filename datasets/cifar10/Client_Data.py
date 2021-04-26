import torch
import numpy as np
import flearn.servers.server
from datasets.cifar10.preprocess import client_idxs, num_clients

IDs = np.arange(num_clients)

class Client_Data(object):
    def __init__(self, dataset, id):
        idx = client_idxs[id]
        self.X = dataset.data[idx]
        self.y = np.array(dataset.targets, dtype = np.int64)[idx]
    
    # must define len() method for Client Data object
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]
        
flearn.servers.server.Client_Data = Client_Data
