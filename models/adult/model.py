import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, BatchSampler, RandomSampler

class LogR(nn.Module):
    dev = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    def __init__(self, initial_params, learning_rate):
        super().__init__()

        self.lr = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()

        self.linear = nn.Linear(99, 2)

        if initial_params is not None:
            self.assign_params(initial_params)

    def forward(self, xb):
        logits = self.linear(xb)
        return logits

    def train(self, client_data, epoch, batch_size):
        X, y = map(torch.tensor, (client_data.X, client_data.y))
        train_ds = TensorDataset(X, y)
        train_dl = DataLoader(train_ds, batch_size = batch_size)
        opt = optim.Adagrad(self.parameters(), lr=self.lr)

        for _ in range(epoch):
            for xb, yb in train_dl:
                logits = self.forward(xb)
                loss = self.loss_fn(logits, yb)
                #loss_value = loss.item()

                loss.backward()
                opt.step()
                opt.zero_grad()

    def sgd(self, client_data, batch_size):
        X, y = map(torch.tensor, (client_data.X, client_data.y))
        train_ds = TensorDataset(X, y)
        train_dl = DataLoader(train_ds, 
            sampler = BatchSampler(RandomSampler(train_ds), 
            batch_size = batch_size, drop_last = False
        ))
        xb, yb = next(iter(train_dl))
        xb = xb.view(-1, 99)
        yb = yb.view(-1)

        #opt = optim.Adagrad((self.linear.weight, 
        #          self.linear.bias), lr=self.lr)

        logits = self.forward(xb)
        loss = self.loss_fn(logits, yb)

        loss.backward()
        grads = []
        grads.append(self.linear.weight.grad.view(-1).detach().numpy())
        grads.append(self.linear.bias.grad.detach().numpy())

        loss_value = self.solve_loss(client_data)

        return loss_value, grads

    def assign_params(self, params):
        self.linear.weight = nn.Parameter(torch.tensor(params[0].reshape(2, 99), dtype=torch.float32))
        self.linear.bias = nn.Parameter(torch.tensor(params[1], dtype=torch.float32))

    def print_params(self):
        params = [self.linear.weight.detach().numpy().reshape(-1),
                self.linear.bias.detach().numpy()]
        return params
  
    def solve_loss(self, client_data):
        X = torch.tensor(client_data.X)
        y_true = torch.tensor(client_data.y)

        y_pred = self.forward(X)
        return self.loss_fn(y_pred, y_true).item()

    def predict_accu(self, client_data):
        X = torch.tensor(client_data.X)
        #y_true = torch.tensor(client_data.y)

        y_pred = F.softmax(self.forward(X), dim = 1).detach().numpy().argmax(axis = 1)

        accuracy = sum(y_pred == client_data.y) / len(client_data)
        return accuracy
