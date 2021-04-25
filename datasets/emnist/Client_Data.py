import numpy as np

class Client_Data(object):
    def __init__(self, dataset, id):
        data = dataset.create_tf_dataset_for_client(id)
        pixels = np.array([example['pixels'].numpy().reshape(-1) for example in data])
        labels = np.array([example['label'].numpy() for example in data])
        self.X = pixels.astype(np.float32)
        self.y = labels.astype(np.int64)
    
    def __len__(self):
        return len(self.y)
