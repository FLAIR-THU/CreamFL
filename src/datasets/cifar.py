from torchvision.datasets.cifar import CIFAR100, CIFAR10
import torch.utils.data as data
import numpy as np
from PIL import Image
import torch

import pickle
import os


class Cifar(data.Dataset):

    def __init__(self, name='Cifar100', train=True, transform=None, is_iid=False,
                 client=-1, root=os.environ['HOME'] + '/data/'):
        if name == 'Cifar10':
            root = root + '/cifar10'
            dataset = CIFAR10(root=root, train=train, transform=None, target_transform=None, download=True)
        elif name == 'Cifar100':
            root = root + '/cifar100'
            dataset = CIFAR100(root=root, train=train, transform=None, target_transform=None, download=True)
        else:
            assert False, 'dataset name wrong'
        self.images = dataset.data
        self.labels = np.array(dataset.targets)

        root = './'
        if client > -1 and train:
            indices = self.iid(root)[client] if not is_iid else self.non_iid(root)[client]
            indices = np.array(list(indices)).astype(int)
            print(indices)

            self.images = self.images[indices]
            self.labels = self.labels[indices]

        self.transform = transform

    def iid(self, root='/data/mmdata/cifar10', num_users=10):
        """
        Sample I.I.D. client data from MNIST dataset
        :param dataset:
        :param num_users:
        :return: dict of image index
        """
        pkl_path = root + 'client_iid.pkl'
        if os.path.exists(pkl_path):
            dict_users = pickle.load(open(pkl_path, 'rb'))
        else:
            num_items = int(len(self.labels) / num_users)
            dict_users, all_idxs = {}, [i for i in range(len(self.labels))]
            for i in range(num_users):
                dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                                     replace=False))
                all_idxs = list(set(all_idxs) - dict_users[i])
            pickle.dump(dict_users, open(pkl_path, 'wb'))
        return dict_users

    def non_iid(self, root='/data/mmdata/cifar10', num_users=10):
        pkl_path = root + 'client_noniid.pkl'
        if os.path.exists(pkl_path):
            dict_users = pickle.load(open(pkl_path, 'rb'))
        else:
            num_shards = 200
            num_imgs = int(len(self.labels) / num_shards)
            idx_shard = [i for i in range(num_shards)]
            dict_users = {i: np.array([]) for i in range(num_users)}
            idxs = np.arange(num_shards * num_imgs)

            # divide and assign 2 shards/client
            for i in range(num_users):
                rand_set = set(np.random.choice(idx_shard, int(num_shards / num_users), replace=False))
                idx_shard = list(set(idx_shard) - rand_set)
                for rand in rand_set:
                    dict_users[i] = np.concatenate(
                        (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

            pickle.dump(dict_users, open(pkl_path, 'wb'))
        return dict_users

    def __getitem__(self, index):
        output = Image.fromarray(self.images[index])

        if self.transform is not None:
            output = self.transform(output)

        return output, self.labels[index]

    def __len__(self):
        return len(self.labels)
