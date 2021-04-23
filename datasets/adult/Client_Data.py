import numpy as np
import flearn.servers.server

IDs = ('doctor', 'nondoctor')

class Client_Data(object):
    def __init__(self, dataset, id):
        raw_X, raw_y = dataset
        if id == 'doctor':
            doc_ind = (raw_X[:,18] == 1)
            X = raw_X[doc_ind]
            y = raw_y[doc_ind]
        else:
            ndoc_ind = (raw_X[:,18] == 0)
            X = raw_X[ndoc_ind]
            y = raw_y[ndoc_ind]
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
    
    def __len__(self):
        return len(self.y)
        
flearn.servers.server.Client_Data = Client_Data
