from flearn.utils.sub import boot, boot_agg
from flearn.clients.client import Client

class SubClient(Client):
    def __init__(self, id, client_data):
        super(SubClient, self).__init__(id, client_data)

    def create_model(self, Learner, initial_params, learning_rate):
        self.model = Learner(initial_params, learning_rate)
        self.sub_model = Learner(initial_params, learning_rate)

    def update_model(self, params):
        self.model.assign_params(params)
        self.sub_model.assign_params(params)

    def sub_train(self, epoch, batch_size, sub_rate):
        self.model.train(self.client_data, epoch, batch_size)
        self.sub_data = boot(self.client_data, sub_rate)
        self.sub_model.train(self.sub_data, epoch, batch_size)
        params = boot_agg(self.model.print_params(), self.sub_model.print_params(), sub_rate)
        self.model.assign_params(params)
        loss = self.model.solve_loss(self.client_data)
        num_example = len(self.client_data)
        return num_example, loss