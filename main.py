import argparse
import random
import importlib
import numpy as np
import torch

OPTIMIZERS = ['fedavg', 'AFL', 'qFFL', 'SFL']
DATASETS = ['adult', 'fmnist', 'emnist', 'cifar10', 'shakespeare']

def read_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--optimizer',
                        help='name of optimizer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default='fedavg')
    parser.add_argument('-d', '--dataset',
                        help='name of dataset;',
                        type=str,
                        choices=DATASETS,
                        default='adult')
    parser.add_argument('-m', '--model',
                        help='name of model;',
                        type=str,
                        default='LogR')
    parser.add_argument('-i', '--iter_rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=-1)
    parser.add_argument('-e', '--epochs', 
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=1)
    parser.add_argument('-b', '--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=10)
    parser.add_argument('-l', '--learning_rate',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.001)
    parser.add_argument('--lambda_learning_rate',
                        help='learning rate for afl weights;',
                        type=float,
                        default=0.001)
    parser.add_argument('-q', '--fair_q',
                        help='q for qFFL;',
                        type=float,
                        default=1.0)
    parser.add_argument('--fair_L',
                        help='L for qFFL;',
                        type=float,
                        default=1.0)
    parser.add_argument('--sub_rate',
                        help='subsampling rate for sfl;',
                        type=float,
                        default=0.1)
    parser.add_argument('--select_rate',
                        help='rate of clients selected for training per round;',
                        type=float,
                        default=1.0)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)


    try: parsed = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))

    # Set seeds
    random.seed(1 + parsed['seed'])
    np.random.seed(12 + parsed['seed'])
    torch.manual_seed(123 + parsed['seed'])


    # load model
    model_path = '%s.%s.%s' % ('models', parsed['dataset'], 'model')
    mod = importlib.import_module(model_path)
    learner = getattr(mod, parsed['model'])
	
	# preprocess data
    pre_path = '%s.%s.%s' % ('datasets', parsed['dataset'], 'preprocess')
    mod = importlib.import_module(pre_path)
    preprocess = getattr(mod, 'preprocess')
	
	# load Client_Data and IDs
    data_path = '%s.%s.%s' % ('datasets', parsed['dataset'], 'Client_Data')
    mod = importlib.import_module(data_path)
    Client_Data = getattr(mod, 'Client_Data')
    IDs = getattr(mod, 'IDs')
    
    # load selected server
    opt_path = 'flearn.servers.%s' % parsed['optimizer']
    mod = importlib.import_module(opt_path)
    server = getattr(mod, parsed['optimizer'])

    return parsed, preprocess, Client_Data, IDs, learner, server

def main():
    INITIAL_PARAMETERS = None #
    
    # parse command line arguments
    options, preprocess, Client_Data, IDs, learner, server = read_options()
	
    LEARNING_RATE = options['learning_rate']
    LAMBDA_LEARNING_RATE = options['lambda_learning_rate']
    SUB_RATE = options['sub_rate']
    q = options['fair_q']
    L = options['fair_L']
    EPOCH = options['epochs']
    BATCH_SIZE = options['batch_size']
    ITER = options['iter_rounds']
    SELECT_RATE = options['select_rate']
    train_dataset, test_dataset = preprocess()
	
    if str(server)[-5:-2] == 'AFL':
        fit = server(train_dataset, IDs, learner, INITIAL_PARAMETERS, LEARNING_RATE, LAMBDA_LEARNING_RATE)
    elif str(server)[-6:-2] == 'qFFL':
        fit = server(q, L, train_dataset, IDs, learner, INITIAL_PARAMETERS, LEARNING_RATE)
    else:
        fit = server(train_dataset, IDs, learner, INITIAL_PARAMETERS, LEARNING_RATE)
    
    for i in range(ITER):
        if str(server)[-5:-2] == 'SFL':
            loss = fit.train(EPOCH, BATCH_SIZE, SUB_RATE, SELECT_RATE)
        else:
            loss = fit.train(EPOCH, BATCH_SIZE, SELECT_RATE)
        print('----------iter: {:d}/{:d}, loss: {:f}----------'.format(i+1, ITER, loss))
    	
    final_model = fit.model
    accs = []
    for id in IDs:
        id_testdata = Client_Data(test_dataset, id)
        id_acc = final_model.predict_accu(id_testdata)
        print('{} data prediction accuracy: {:2f} \n'.format(id, id_acc*100))
    	
    return dict(zip(IDs, accs))

if __name__ == '__main__':
    main()