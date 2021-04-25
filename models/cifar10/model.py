import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, BatchSampler, RandomSampler
from datasets.cifar10.preprocess import transform

class Client_Dataset(Dataset):
    def __init__(self, X, y, transform = transform):
        self.X = X
        self.y = y
        self.transform = transform
    
    # custom Dataset object must define __len__ and __getitem__ methods
    # len method
    def __len__(self):
        return len(self.y)
    
    # getitem method
    def __getitem__(self, idx):
        img = Image.fromarray(self.X[idx])
    
        if self.transform is not None:
            img = self.transform(img)
    
        return img, torch.tensor(self.y[idx])
        
# check if device is gpu or not
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
        
class CNN(nn.Module):
    def __init__(self, initial_params, learning_rate):
        """ CNN model for CIFAR10 dataset
        the CNN model structure is from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        
        Args:
            initial_params (:obj:'list' of :obj:'np.array): a list contains shape(-1) arrays of parameters of model. 
            if initial_params is None, it will generate random initial parameters. 
            learning_rate: learning rate
        
        Attributes:
            loss_func: loss function
            lr: learning rate
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.lr = learning_rate
        self.loss_func = F.cross_entropy
        if initial_params is not None:
            self.assign_params(initial_params)
        
    def forward(self, xb):
        xb = self.pool(F.relu(self.conv1(xb)))
        xb = self.pool(F.relu(self.conv2(xb)))
        xb = xb.view(-1, 16 * 5 * 5)
        xb = F.relu(self.fc1(xb))
        xb = F.relu(self.fc2(xb))
        xb = self.fc3(xb)
        return xb
    
    def train(self, client_data, epoch, batch_size):
        """gradient descent on client data
        
        Args:
            client_data (:obj:'Client_Data'): data to train model on
            epoch: epochs for training
            batch_size: batch size for training
        """
        #X, y = map(torch.tensor, (client_data.X, client_data.y))
        X, y = client_data.X, client_data.y
        
        if dev == torch.device("cuda"):
            # training with GPU
            self.to(dev)
        
        train_ds = Client_Dataset(X, y)
        train_dl = DataLoader(train_ds, batch_size = batch_size)
        opt = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        
        for _ in range(epoch):
            for xb, yb in train_dl:
                if dev == torch.device("cuda"):
                    # training with GPU
                    xb, yb = xb.to(dev), yb.to(dev)
                pred = self.forward(xb)
                loss = self.loss_func(pred, yb)
                loss.backward()
                opt.step()
                opt.zero_grad()
    
    def sgd(self, client_data, batch_size):
        """stochastic gradient descent
        
        Args:
            client_data (:obj:'Client_Data'): data to train model on
            batch_size: batch size for training
        
        Returns:
            loss_value: loss value
            grads (:obj:'list' of :obj:'np.array'): stochastic gradients on mini-batch data
        """
        #X, y = map(torch.tensor, (client_data.X, client_data.y))
        X, y = client_data.X, client_data.y
        
        if dev == torch.device("cuda"):
            # training with GPU
            self.to(dev)
        
        train_ds = Client_Dataset(X, y)
        train_dl = DataLoader(train_ds, 
            sampler = BatchSampler(RandomSampler(train_ds), 
            batch_size = batch_size, drop_last = False
        ))
        opt = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        
        xb, yb = next(iter(train_dl))
        yb = yb.view(-1)
        if dev == torch.device("cuda"):
            # training with GPU
            xb, yb = xb.to(dev), yb.to(dev)
        pred = self.forward(xb)
        loss = self.loss_func(pred, yb)
        loss.backward()
        
        if dev == torch.device("cuda"):
            grads = [p.grad.view(-1).cpu().detach().numpy() for p in self.parameters()]
        else:
            grads = [p.grad.view(-1).detach().numpy() for p in self.parameters()]
        loss_value = self.solve_loss(client_data)
        
        return loss_value, grads
    
    def assign_params(self, params):
        self.conv1.weight = nn.Parameter(torch.tensor(params[0].reshape(6, 3, 5, 5), dtype=torch.float32))
        self.conv1.bias = nn.Parameter(torch.tensor(params[1], dtype=torch.float32))
        self.conv2.weight = nn.Parameter(torch.tensor(params[2].reshape(16, 6, 5, 5), dtype=torch.float32))
        self.conv2.bias = nn.Parameter(torch.tensor(params[3], dtype=torch.float32))
        self.fc1.weight = nn.Parameter(torch.tensor(params[4].reshape(120, 400), dtype=torch.float32))
        self.fc1.bias = nn.Parameter(torch.tensor(params[5], dtype=torch.float32))
        self.fc2.weight = nn.Parameter(torch.tensor(params[6].reshape(84, 120), dtype=torch.float32))
        self.fc2.bias = nn.Parameter(torch.tensor(params[7], dtype=torch.float32))
        self.fc3.weight = nn.Parameter(torch.tensor(params[8].reshape(10, 84), dtype=torch.float32))
        self.fc3.bias = nn.Parameter(torch.tensor(params[9], dtype=torch.float32))
    
    def print_params(self):
        """print model parameters
        
        Returns:
            model parameters
        """
        if dev == torch.device("cuda"):
            params = [p.cpu().detach().numpy().reshape(-1) for p in self.parameters()]
        else:
            params = [p.detach().numpy().reshape(-1) for p in self.parameters()]
        return params
        
    def solve_loss(self, client_data):
        """return the loss value on given data
        
        Args:
            client_data (:obj:'Client_Data'): data to compute loss value on
        
        Returns:
            loss value on given data
        """
        X, y = client_data.X, client_data.y
        train_ds = Client_Dataset(X, y)
        train_dl = DataLoader(train_ds, batch_size = 1)
        
        total_loss = 0
        with torch.no_grad():
            for xb, yb in train_dl:
                if dev == torch.device("cuda"):
                    xb, yb = xb.to(dev), yb.to(dev)
                pred = self.forward(xb)
                loss = self.loss_func(pred, yb)
                total_loss += loss.item()
        
        return total_loss
    
    def predict_accu(self, client_data):
        """return the loss value on given data
        
        Args:
            client_data (:obj:'Client_Data'): data to compute loss value on
        
        Returns:
            predict accuracy on given data
        """
        X, y = client_data.X, client_data.y
        train_ds = Client_Dataset(X, y)
        train_dl = DataLoader(train_ds, batch_size = 10)
        
        correct = 0
        total = 0
    
        with torch.no_grad():
            for xb, yb in train_dl:
                if dev == torch.device("cuda"):
                    xb, yb = xb.to(dev), yb.to(dev)
                pred = self.forward(xb)
                y_pred = F.softmax(pred, dim = 1).detach().argmax(axis = 1)
                total += yb.size(0)
                correct += (y_pred == yb).sum().item()
        
        return correct / total
