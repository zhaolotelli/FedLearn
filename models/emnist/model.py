import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, BatchSampler, RandomSampler
        
# check if device is gpu or not
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
        
class CNN(nn.Module):
    ''' 
    the CNN model structure is from https://pytorch.org/tutorials/beginner/nn_tutorial.html
    '''
    def __init__(self, initial_params, learning_rate):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size = 3, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size = 3, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size = 3, stride = 2, padding = 1)

        self.lr = learning_rate
        self.loss_func = F.cross_entropy
        if initial_params is not None:
            self.assign_params(initial_params)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))

    def train(self, client_data, epoch, batch_size):
        X, y = map(torch.tensor, (client_data.X, client_data.y))

        if dev == torch.device("cuda"):
            # training with GPU
            X = X.to(dev)
            y = y.to(dev)
            self.to(dev)

        train_ds = TensorDataset(X, y)
        train_dl = DataLoader(train_ds, batch_size = batch_size)
        opt = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)

        for _ in range(epoch):
            for xb, yb in train_dl:
                pred = self.forward(xb)
                loss = self.loss_func(pred, yb)
                loss.backward()
                opt.step()
                opt.zero_grad()

  def sgd(self, client_data, batch_size):
        X, y = map(torch.tensor, (client_data.X, client_data.y))

        if dev == torch.device("cuda"):
            # training with GPU
            X = X.to(dev)
            y = y.to(dev)
            self.to(dev)

        train_ds = TensorDataset(X, y)
        train_dl = DataLoader(train_ds, 
            sampler = BatchSampler(RandomSampler(train_ds), 
            batch_size = batch_size, drop_last = False
        ))
        opt = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)

        xb, yb = next(iter(train_dl))
        yb = yb.view(-1)
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
        self.conv1.weight = nn.Parameter(torch.tensor(params[0].reshape(16, 1, 3, 3), dtype=torch.float32))
        self.conv1.bias = nn.Parameter(torch.tensor(params[1], dtype=torch.float32))
        self.conv2.weight = nn.Parameter(torch.tensor(params[2].reshape(16, 16, 3, 3), dtype=torch.float32))
        self.conv2.bias = nn.Parameter(torch.tensor(params[3], dtype=torch.float32))
        self.conv3.weight = nn.Parameter(torch.tensor(params[4].reshape(10, 16, 3, 3), dtype=torch.float32))
        self.conv3.bias = nn.Parameter(torch.tensor(params[5], dtype=torch.float32))

    def print_params(self):
        if dev == torch.device("cuda"):
          params = [p.cpu().detach().numpy().reshape(-1) for p in self.parameters()]
        else:
          params = [p.detach().numpy().reshape(-1) for p in self.parameters()]
        return params
    
    def solve_loss(self, client_data):
        X = torch.tensor(client_data.X)
        y_true = torch.tensor(client_data.y)

        if dev == torch.device("cuda"):
            # compute with GPU
            X = X.to(dev)
            y_true = y_true.to(dev)

        y_pred = self.forward(X)
        return self.loss_func(y_pred, y_true).item()

    def predict_accu(self, client_data):
        X = torch.tensor(client_data.X)
        y_true = client_data.y

        if dev == torch.device("cuda"):
            X = X.to(dev)
            self.to(dev)
            y_pred = F.softmax(self.forward(X), dim = 1).cpu().detach().numpy().argmax(axis = 1)
        else:
            y_pred = F.softmax(self.forward(X), dim = 1).detach().numpy().argmax(axis = 1)

        accuracy = sum(y_pred == y_true) / len(client_data)
        return accuracy
