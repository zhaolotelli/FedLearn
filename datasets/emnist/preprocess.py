import tensorflow_federated as tff

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

def preprocess():
    return emnist_train, emnist_test