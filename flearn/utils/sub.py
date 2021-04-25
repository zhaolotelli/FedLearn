import copy
import numpy as np

def boot(client_data, sub_rate):
    n = len(client_data)
    rand_ind = np.random.choice(n, np.int(np.floor(n * sub_rate)))
    sub_data = copy.deepcopy(client_data)
    X = client_data.X
    y = client_data.y
    sub_data.X = X[rand_ind]
    sub_data.y = y[rand_ind]
    return sub_data

def boot_agg(params, sub_params, sub_rate):
    final_params = [np.zeros(len(p)) for p in params]
    for i, (param, sub_param) in enumerate(zip(params, sub_params)):
        final_params[i] = (param - sub_rate * sub_param) / (1 - sub_rate)
    return final_params