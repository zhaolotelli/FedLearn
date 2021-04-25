# FedLearn
demo federated learning scenario

## three baselines

- FedAvg: [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
- AFL: [Agnostic Federated Learning](https://arxiv.org/abs/1902.00146)
- qFFL: [Fair Resource Allocation in Federated Learning](https://openreview.net/forum?id=ByexElSYDr)

## some example cmd commands

```
python main.py -i 10 -e 5 --seed 13
python main.py -d fmnist -i 100 -b 20 --seed 13
python main.py -o AFL -d fmnist -i 100 -b 20 --lambda_learning_rate 0.01 --seed 13
python main.py -o SFL -d fmnist -i 100 -b 20 --sub_rate 0.05 --seed 13
python main.py -o qFFL -d fmnist -i 1000 -b 20 -q 5 --fair_L 5 --seed 13
```

## full example 

a full FedAvg example is in [Fed_CIFAR10.ipynb](https://github.com/zhaolotelli/FedLearn/blob/master/Fed_CIFAR10.ipynb)
