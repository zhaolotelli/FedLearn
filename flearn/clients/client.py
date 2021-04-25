class Client(object):
    def __init__(self, id, client_data):
        self.id = id
        self.client_data = client_data

    def create_model(self, Learner, initial_params, learning_rate):
        self.model = Learner(initial_params, learning_rate)

    def update_model(self, params):
        self.model.assign_params(params)

    def train(self, epoch, batch_size):
        self.model.train(self.client_data, epoch, batch_size)
        loss = self.model.solve_loss(self.client_data)
        num_example = len(self.client_data)
        return num_example, loss

    def sgd(self, batch_size):
        loss, grads = self.model.sgd(self.client_data, batch_size)
        num_example = len(self.client_data)
        return num_example, loss, grads
