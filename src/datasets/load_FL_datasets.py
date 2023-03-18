import os
import random
import copy
import numpy as np
import torchtext
from torchvision import datasets, transforms
import torch
import pickle

from src.datasets.dataset_L import Language, caption_collate_fn
from src.utils.color_lib import RGBmean, RGBstdv


def get_FL_trainloader(dataset, data_root, num_clients, partition, alpha, batch_size):
    if dataset == 'cifar100':
        data_transforms = transforms.Compose([transforms.Resize(int(256 * 1.1)),
                                              transforms.RandomRotation(10),
                                              transforms.RandomCrop(256),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(RGBmean['Cifar100'], RGBstdv['Cifar100'])])
        train_set = datasets.CIFAR100(data_root, train=True, download=True,
                                      transform=data_transforms)
        test_set = datasets.CIFAR100(data_root, train=False, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize(RGBmean['Cifar100'], RGBstdv['Cifar100'])]
                                     ))
    elif dataset == 'cifar10':
        train_set = datasets.CIFAR10(data_root, train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                                     ]))

        test_set = datasets.CIFAR10(data_root, train=False, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                                    ]))
    elif dataset == 'AG_NEWS':
        train_set = Language(name="AG_NEWS", train=True, transform=None, is_iid=False)
        test_set = Language(name="AG_NEWS", train=False, transform=None, is_iid=False)
    elif dataset == 'YelpReviewPolarity':
        train_set = Language(name="YelpReviewPolarity", train=True, transform=None)
        test_set = Language(name="YelpReviewPolarity", train=False, transform=None)

    train_set.data = train_set.data  # [:int(50000*0.7)]
    train_set.targets = train_set.targets  # [:int(50000*0.7)]
    if dataset == 'AG_NEWS' or dataset == 'YelpReviewPolarity':
        num_samples = train_set.targets.shape[0]
    else:
        num_samples = train_set.data.shape[0]
    net_dataidx_map = data_partitioner(dataset, num_samples, num_clients, partition=partition,
                                       check_dir="./data_partition/", alpha=alpha,
                                       y_train=np.array(train_set.targets))
    print(f"Samples Num: {[len(i) for i in net_dataidx_map.values()]}")
    net_dataset_map = {i: torch.utils.data.Subset(train_set, net_dataidx_map[i]) for i in net_dataidx_map.keys()}
    if dataset == "cifar100" or dataset == "cifar10":
        loader_map = {
            i: torch.utils.data.DataLoader(net_dataset_map[i], batch_size=batch_size, shuffle=True, num_workers=2) for
            i in net_dataset_map.keys()}
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size * 2, shuffle=False, num_workers=2)
    elif dataset == "AG_NEWS" or dataset == "YelpReviewPolarity":
        loader_map = {
            i: torch.utils.data.DataLoader(net_dataset_map[i], batch_size=batch_size, shuffle=True, num_workers=2,
                                           pin_memory=True, sampler=None, drop_last=False,
                                           collate_fn=caption_collate_fn)
            for i in net_dataset_map.keys()}
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size * 2, shuffle=False, num_workers=2,
                                                  pin_memory=True, sampler=None, drop_last=False,
                                                  collate_fn=caption_collate_fn)

    return loader_map, test_loader


def data_partitioner(dataset, num_samples, num_nets, partition='homo', check_dir=None, alpha=0.5, y_train=None):
    check_dir = check_dir + f'client_{dataset}'

    if partition == "homo":
        check_dir = check_dir + "_iid.pkl"
        if os.path.isfile(check_dir):
            net_dataidx_map = pickle.load(open(check_dir, 'rb'))
        else:
            idxs = np.random.permutation(num_samples)
            batch_idxs = np.array_split(idxs, num_nets)
            net_dataidx_map = {i: batch_idxs[i] for i in range(num_nets)}
            pickle.dump(net_dataidx_map, open(check_dir, 'wb'))

    elif partition == "hetero":
        check_dir = check_dir + "_noniid.pkl"
        if os.path.isfile(check_dir):
            net_dataidx_map = pickle.load(open(check_dir, 'rb'))
        else:
            min_size = 0
            K = max(y_train) + 1  # todo 
            net_dataidx_map = {}
            print('Hetero partition')
            while min_size < (10 if dataset == "cifar100" else (3000 if dataset == "AG_NEWS" else 500)):
                idx_batch = [[] for _ in range(num_nets)]
                # for each class in the dataset
                for k in range(K):
                    idx_k = np.where(y_train == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(alpha, num_nets))
                    ## Balance
                    proportions = np.array(
                        [p * (len(idx_j) < num_samples / num_nets) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

            for j in range(num_nets):
                np.random.shuffle(idx_batch[j])
                net_dataidx_map[j] = idx_batch[j]

            pickle.dump(net_dataidx_map, open(check_dir, 'wb'))

    return net_dataidx_map


def get_dataloader(args):
    if args.dataset.lower() == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(args.data_root, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.Resize((32, 32)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(args.data_root, train=False, download=True,
                           transform=transforms.Compose([
                               transforms.Resize((32, 32)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)


    elif args.dataset.lower() == 'svhn':
        print("Loading SVHN data")
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(args.data_root, split='train', download=True,
                          transform=transforms.Compose([
                              transforms.Resize((32, 32)),
                              transforms.ToTensor(),
                              transforms.Normalize((0.43768206, 0.44376972, 0.47280434),
                                                   (0.19803014, 0.20101564, 0.19703615)),
                              # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                          ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(args.data_root, split='test', download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.43768206, 0.44376972, 0.47280434),
                                                   (0.19803014, 0.20101564, 0.19703615)),
                              # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                          ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
    elif args.dataset.lower() == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_root, train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                             ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_root, train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                             ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)

    return train_loader, test_loader
