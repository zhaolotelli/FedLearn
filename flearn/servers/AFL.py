from flearn.servers.server import Server
from flearn.utils.project import project
import numpy as np

class AFL(Server):
    def __init__(self, train_data, ids, Learner, initial_params, learning_rate, lambda_learning_rate):
        super(AFL, self).__init__(train_data, ids, Learner, initial_params, learning_rate)
        self.lambdas = np.ones(len(self.clients)) / len(self.clients)
        self.lambda_lr = lambda_learning_rate

    def train(self, epoch, batch_size, select_rate):
        self.send_model()
        losses = []
        grads = []
        for client in self.clients:
            client_num, client_loss, client_grads = client.sgd(batch_size)
            losses.append(client_loss)
            grads.append(client_grads)
            # print('Client: {}, Local_loss: {:f}'.format(client.id, client_loss))
        self.aggregate(losses, grads)
        return np.sum(losses)

    def aggregate(self, losses, grads):
        lambdas_new = self.lambdas + self.lambda_lr * np.array(losses)
        self.lambdas = project(lambdas_new)

        total_grad = [np.zeros(len(g)) for g in grads[0]]
        for lambda_, grad in zip(self.lambdas, grads):
            for i in range(len(grad)):
                total_grad[i] = total_grad[i] + grad[i] * lambda_
    
        total_params = [param for param in self.model.print_params()]
        for i in range(len(total_params)):
            total_params[i] = total_params[i] - self.model.lr * total_grad[i]
        self.model.assign_params(total_params)
        return total_params