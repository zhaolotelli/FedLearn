import torch
import torchvision
import torchvision.transforms as transforms

train_dataset = torchvision.datasets.FashionMNIST(root = "./datasets/fmnist/data", train = True, download = True, transform = transforms.ToTensor())
test_dataset = torchvision.datasets.FashionMNIST(root = "./datasets/fmnist/data", train = False, download = True, transform = transforms.ToTensor())

raw_X = train_dataset.data.to(dtype = torch.float32)
mu = raw_X.mean(dim = 0)
sigma = raw_X.std(dim = 0)

def preprocess():
    
    return train_dataset, test_dataset