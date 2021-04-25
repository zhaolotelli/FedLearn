import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./datasets/fmnist/data', train=True, 
                download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./datasets/fmnist/data', train=False,
                download=True, transform=transform)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# generate non-iid clients from Dirichlet prior distribution
# the idea of this part comes from https://github.com/IBM/probabilistic-federated-neural-matching
K = 10
alpha = 0.5
num_clients = 100
min_size = 0
y_train = np.array(trainset.targets)
np.random.seed(13)

while min_size < 10:
    client_idxs = [[] for _ in range(num_clients)]
    for k in range(K):
        ps = np.random.dirichlet(np.repeat(alpha, num_clients))
        idx_k = np.where(y_train == k)[0]
        np.random.shuffle(idx_k)
        ps = (np.cumsum(ps)*len(idx_k)).astype(int)[:-1]
        client_idx_k = np.split(idx_k, ps)
        client_idxs = [client_idx + idx.tolist() for client_idx, idx in zip(client_idxs, client_idx_k)]
        min_size = min([len(client_idx) for client_idx in client_idxs])

def preprocess():
    
    return trainset, testset