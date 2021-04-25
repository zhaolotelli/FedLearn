from flearn.utils.ClientData import ClientData
from flearn.clients.client import Client
import numpy as np

Client_Data = ClientData

class Server(object):
    def __init__(self, train_data, ids, Learner, initial_params, learning_rate):
        self.ids = ids
        self.learner = Learner
        self.initial_params = initial_params
        self.learning_rate = learning_rate
        self.clients = self.set_clients(train_data)
        self.model = self.learner(initial_params, learning_rate)
        
    def set_clients(self, train_data):
        clients = []
        for id in self.ids:
            client_data = Client_Data(train_data, id)
            c = Client(id, client_data)
            c.create_model(self.learner, self.initial_params, self.learning_rate)
            clients.append(c)
        return clients

    def send_model(self):
        params = self.model.print_params()
        for c in self.clients:
            c.update_model(params)

    def select_client(self, select_rate):
        self.num_clients = np.maximum(1, np.int(np.floor(len(self.ids) * select_rate)))
        select_ids = np.random.choice(self.ids, self.num_clients, replace=False)
        select_clients = []
        for id in select_ids:
            loc_id = np.array([id == idx for idx in self.ids])
            ind = np.int(np.array(range(len(self.ids)))[loc_id])
            select_client = self.clients[ind]
            select_clients.append(select_client)
        return select_clients
