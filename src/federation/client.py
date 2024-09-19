import os
from copy import deepcopy
import sys

import munch

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from src.custom_datasets.load_FL_datasets import get_FL_trainloader
from src.algorithms.ClientTrainer import ClientTrainer
from src.algorithms.MMClientTrainer import MMClientTrainer
from src.utils.color_lib import RGBmean, RGBstdv

from src.utils.config import parse_config

import api

try:
    from apex import amp
except ImportError:
    print('failed to import apex')

class Client:
    def __init__(self, context):
        # all configs
        self.context = context
        self.logger = context.logger
        self.args = context.args
        self.wandb = context.wandb

        # validation for clients
        if context.args.client_name is None:
            raise ValueError("The client_name argument is required for clients")
        
        my_client = None
        for client in context.fed_config.clients:
            if client["name"] == context.args.client_name:
                my_client = munch.Munch(client)
        if my_client is None:
            raise ValueError(f"Client name {context.args.client_name} not found in configuration file {context.args.fed_config}")

        # setup client
        self.client_config = my_client
        self.name = context.args.client_name
        self.device = context.device

        if self.client_config.data_type == 'txt':
            self.context.has_img_model = False
            self.context.has_txt_model = True
        elif self.client_config.data_type == 'img':
            self.context.has_img_model = True
            self.context.has_txt_model = False
        elif self.client_config.data_type == 'mm':
            self.context.has_img_model = True
            self.context.has_txt_model = True
        else:
            raise ValueError(f'client_config.data_type={self.client_config.data_type} is not implemented by federation.client.Client')

        # will setup in setup_data_loader
        self.trainer = None

    def setup_data_loader(self):

        self.logger.log('setup global dataloader')

        global_dataloader, _ = api.get_global_dataloader(self.context)
        self.global_eval_dataloader = global_dataloader['train_subset_eval' + f'_{self.args.pub_data_num}']

        self.logger.log('start creating model and partition datasets')

        os.makedirs(os.environ['HOME'] + f'/data/yClient', exist_ok=True)

        args = self.context.args
        client_config = self.client_config

        data_type = client_config.data_type
        # data_partition_file_name = client_config.data_partition # todo
        data_partition_index = client_config.data_partition_index

        alpha = args.alpha
        batch_size = args.batch_size
        max_size = args.max_size
       
        if data_type == 'txt':
            dataset = 'AG_NEWS'
            # this loads all train loaders for all clients, from old code
            train_loaders, test_set = get_FL_trainloader(dataset, os.environ['HOME'] + "/data",
                                                                    args.num_txt_clients, "hetero", alpha, batch_size, max_size)
            dst = os.environ['HOME'] + f'/data/yClient/{dataset}-{self.name}' # unused?
            self.trainer = ClientTrainer(args, dataset, dst, None, None, None, self.logger,
                                    global_test_set=test_set, inter_distance=4, client_id=data_partition_index, wandb=self.wandb)
            self.trainer.train_loader = train_loaders[data_partition_index]
        elif data_type == 'img':
            dataset = 'cifar100'
            # this loads all train loaders for all clients, from old code
            train_loaders, test_set = get_FL_trainloader(dataset, os.environ['HOME'] + "/data/cifar100",
                                                                 args.num_img_clients, "hetero", alpha, batch_size, max_size)
            dataset = 'Cifar100'
            dst = os.environ['HOME'] + f'/data/yClient/{dataset}-{self.name}'  # unused?
            self.trainer = ClientTrainer(args, dataset, dst, RGBmean['Cifar100'], RGBstdv['Cifar100'], None, self.logger,
                                    global_test_set=test_set, inter_distance=4, client_id=data_partition_index, wandb=self.wandb)
            self.trainer.train_loader = train_loaders[data_partition_index]
        elif data_type == 'mm':
            config = parse_config("./src/f30k.yaml", strict_cast=False)
            config.model.cache_dir = config.model.cache_dir + '-' + config.train.server_dataset
            config.train.output_file = os.path.join(config.model.cache_dir, config.train.output_file)
            config.train.best_model_save_path = os.path.join(config.model.cache_dir, config.train.best_model_save_path)
            config.train.model_save_path = os.path.join(config.model.cache_dir, config.train.model_save_path)
            config.model.embed_dim = self.args.feature_dim
            config.model.not_bert = True
            
            self.trainer = MMClientTrainer(args, config, self.logger, client=data_partition_index, num_users=args.num_mm_clients, dset_name="flicker30k",
                                    device='cuda',
                                    vocab_path='./src/custom_datasets/vocabs/coco_vocab.pkl',
                                    mlp_local=self.args.mlp_local)
        else:
            raise ValueError(f'client_config.data_type={data_type} is not implemented by federation.client.get_data_loader()')
        
    def train(self, gf: api.GlobalFeature):
        trainer = self.trainer
        self.logger.log(f"Training Client {self.name}")
        trainer.run(gf.img, gf.txt, gf.distill_index, self.global_eval_dataloader)
        self.logger.log(f"client {self.name} Generate Local Representations")
        vec, di = trainer.generate_logits(self.global_eval_dataloader)
        if di != gf.distill_index:
            raise ValueError(f"distill_index mismatch: {di} != {gf.distill_index}")
        img = vec.get('img')
        txt = vec.get('txt')
        return img, txt

    def submit(self, local_repr: api.ClientState):
        self.logger.log(f"Submitting local representations to server.")
        api.submit_local_repr(self.context, local_repr)
 
    def run(self):
        current_path = os.path.dirname(os.path.dirname(__file__))
        with open(os.path.join(current_path, 'accuracy.txt'), 'w') as f:
            f.write('')
        while True:
            self.logger.log(f"Client {self.name} is starting a new round.")
            server_state = api.get_server_state(self.context, expected_state=api.RoundState.COLLECT)
            self.logger.log(f"Server state: {server_state.round_state}, round number: {server_state.round_number}, global feature hash: {server_state.feature_hash}")
            global_feature = api.get_global_feature(self.context, server_state)
            self.logger.log(f"Global feature retrieved")
            img, txt = self.train(global_feature)
            del global_feature
            api.add_local_repr(self.context, server_state, img, txt, -1) # todo: set local rounds
            del img, txt
        
if __name__ == "__main__":
    from src.federation.context import new_client_context
    context = new_client_context()
    client = Client(context)
    client.setup_data_loader()
    client.run()


    