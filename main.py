import random
from torchvision import datasets, transforms
from splits import *
from utils import get_client_labels

if __name__ == '__main__':
    # set params (iid, shard, dirichlet_equal, dirichlet_unequal)
    #TODO: implement Dirichlet unequal
    SPLIT = 'dirichlet_equal'
    NUM_USERS = 20
    DIRICHLET_ALPHA = 0.1

    # load CIFAR10 DS
    data_dir = '../data/'

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    validation_dataset = datasets.CIFAR10(data_dir, train=True, download=False, transform=transform_test)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=False, transform=transform_test)

    if SPLIT == 'iid':
        user_groups = iid_split(train_dataset, NUM_USERS)
    elif SPLIT == 'shard':
        user_groups = noniid_fedavg_split(train_dataset, NUM_USERS, client_shards=3)
    elif SPLIT == 'dirichlet_equal':
        user_groups = noniid_dirichlet_equal_split(train_dataset, DIRICHLET_ALPHA, NUM_USERS, 10)
    elif SPLIT == 'dirichlet_unequal':
        print('dirichlet_unequal split not implemented yet')
        exit(1)
    else:
        print("invalid split option")
        exit(1)

    # select a user at random and inspect class label proportions
    user_idx = random.randint(0, NUM_USERS - 1)
    user_labels = get_client_labels(train_dataset, user_groups, 0, 10, proportions=True)
    print(f'User {user_idx} class proportions: {user_labels[user_idx]}')
