import gc
import random

import operator
import os
from copy import deepcopy
import sys

import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from src.datasets.load_FL_datasets import get_FL_trainloader, get_dataloader
from src.algorithms.ClientTrainer import ClientTrainer
from src.algorithms.MMClientTrainer import MMClientTrainer
from src.utils.color_lib import RGBmean, RGBstdv

from src.algorithms.eval_coco import COCOEvaluator
from src.algorithms.retrieval_trainer import TrainerEngine, rawTrainerEngine
from src.utils.config import parse_config
from src.utils.load_datasets import prepare_coco_dataloaders
from src.utils.logger import PythonLogger

try:
    from apex import amp
except ImportError:
    print('failed to import apex')

class Client:
    def __init__(self, args, client_config, fed_config):
        # all configs
        self.args = args
        self.client_config = client_config
        self.fed = fed_config

        # commonly accessed configs
        self.name = client_config.name

        # set in get_
        self.device = None
        self.client_trainer = None

 
    def get_data_loader(self, client_config):
        self.logger.log('start creating model and partition datasets')
        self.device = torch.device("cuda:%d" % args.device)

        os.makedirs(os.environ['HOME'] + f'/data/yClient', exist_ok=True)

        args = self.args
        data_type = client_config.data_type
        data_partition_file_name = client_config.data_partition
        data_partition_index = client_config.data_partition_index

        alpha = args.alpha
        batch_size = args.batch_size
        max_size = args.max_size
       
        if data_type == 'txt':
            dataset = 'AG_NEWS'
            # this loads all train loaders for all clients, from old code
            train_loaders, test_set = get_FL_trainloader(dataset, os.environ['HOME'] + "/data",
                                                                    args.num_txt_clients, "hetero", alpha, batch_size, max_size)
            dst = os.environ['HOME'] + f'/data/yClient/{dataset}-{self.name}'
            self.trainer = ClientTrainer(args, dataset, dst, None, None, None, self.logger,
                                    global_test_set=test_set, inter_distance=4, client_id=data_partition_index, wandb=self.wandb)
            self.trainer.train_loader = train_loaders[data_partition_index]
            pass
        else:
            raise ValueError(f'client_config.data_type={data_type} in not implemented by federation.client.get_data_loader()')