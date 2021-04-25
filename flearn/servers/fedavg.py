from flearn.servers.server import Server
import numpy as np

class fedavg(Server):
    def __init__(self, train_data, ids, Learner, initial_params, learning_rate):
        super(fedavg, self).__init__(train_data, ids, Learner, initial_params, learning_rate)

    def train(self, epoch, batch_size, select_rate=1):
        self.send_model()
        self.select_clients = self.select_client(select_rate)
        losses = []
        self.client_nums = []
        for client in self.select_clients:
            client_num, client_loss = client.train(epoch, batch_size)
            losses.append(client_loss)
            self.client_nums.append(client_num)
            # print('Client: {}, Local_loss: {:f}'.format(client.id, client_loss))
        self.aggregate()
        return np.sum(losses)
    
    def aggregate(self):
        total_params = [np.zeros(len(param)) for param in self.model.print_params()]
        total_num = sum(self.client_nums)
        t = 0
        for c in self.select_clients:
            for i in range(len(total_params)):
                total_params[i] = total_params[i] + self.client_nums[t] / total_num * c.model.print_params()[i]
            t += 1
        self.model.assign_params(total_params)
        return total_params