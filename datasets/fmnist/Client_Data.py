import torch
import flearn.servers.server
from datasets.fmnist.preprocess import mu, sigma

IDs = ['Shirt', 'Pullover', 'T-shirt/top']
ID2idx = {'Shirt': 0,
      'Pullover': 1,
      'T-shirt/top': 2}

class Client_Data(object):
    def __init__(self, dataset, id):
        data = dataset.data
        targets = dataset.targets
        idx = dataset.class_to_idx[id]
        new_idx = ID2idx[id]
        idx_diff = new_idx - idx
        raw_X = data[dataset.targets == idx].to(dtype = torch.float32)
        self.X = (raw_X - mu) / 500
        self.y = targets[dataset.targets == idx] + idx_diff
    
    def __len__(self):
        return len(self.y)
        
flearn.servers.server.Client_Data = Client_Data
