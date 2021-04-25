from flearn.servers.server import Server
import numpy as np

class qFFL(Server):
    def __init__(self, q, L, train_data, ids, Learner, initial_params, learning_rate):
        self.L = L
        self.q = q
        super(qFFL, self).__init__(train_data, ids, Learner, initial_params, learning_rate)

    def train(self, epoch, batch_size, select_rate = 1):
        self.send_model()
        self.select_clients = self.select_client(select_rate)
        self.start_losses = []
        losses = []
        for client in self.select_clients:
            start_loss = client.model.solve_loss(client.client_data)
            self.start_losses.append(start_loss)
            _, client_loss = client.train(epoch, batch_size)
            losses.append(client_loss)
            # print('Client: {}, Local_loss: {:f}'.format(client.id, client_loss))
        self.aggregate()
        return np.sum(losses)
  
    def aggregate(self):
        total_params = [np.zeros(len(param)) for param in self.model.print_params()]
        delta_ = [np.zeros(len(param)) for param in self.model.print_params()]
        start_params = [param for param in self.model.print_params()]
        h_ = 0
        for k, c in enumerate(self.select_clients):
            loss = self.start_losses[k]
            client_params = c.model.print_params()
            for i in range(len(total_params)):
              delta_[i] += np.power(loss, self.q) * (start_params[i] - client_params[i])
            flatten_deltas = np.concatenate(delta_).ravel().tolist()
            h_ += self.q * np.power(loss, self.q - 1) * np.sum(np.square(flatten_deltas)) + self.L * np.power(loss, self.q)
        for i in range(len(total_params)):
            total_params[i] = start_params[i] - delta_[i] / h_
        self.model.assign_params(total_params)
        return total_params