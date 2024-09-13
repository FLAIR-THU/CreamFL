import gc
import random

import operator
import os
from copy import deepcopy
import sys

import torch
import torch.nn as nn
import datasets
from tqdm import tqdm
import wandb

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from src.custom_datasets.load_FL_datasets import get_FL_trainloader
from src.algorithms.ClientTrainer import ClientTrainer
from src.algorithms.MMClientTrainer import MMClientTrainer
from src.utils.color_lib import RGBmean, RGBstdv

from src.algorithms.eval_coco import COCOEvaluator
from src.algorithms.retrieval_trainer import TrainerEngine
from src.algorithms.vqa_meta import VQAMetaData
from src.algorithms.vqa_trainer import VQAEngine, vqa_validation
from src.utils.config import parse_config
from src.utils.load_datasets import prepare_coco_dataloaders, vqa2_dataloader
from src.utils.logger import PythonLogger
from src.utils.util import print_model_tree
# from src.networks.zmq_client import Node

import zmq
import time
import logging
import threading
import pickle


logging.basicConfig(
    level=logging.DEBUG,  # 确保捕获所有日志信息
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

import zmq
import threading
import pickle

class Logger:
    def log(self, message):
        print(message)

class Node:
    def __init__(self, node_id, router_port, peers, logger=Logger()):
        self.context = zmq.Context()

        # ROUTER socket to receive messages from other nodes
        self.router = self.context.socket(zmq.ROUTER)
        self.router.bind(f"tcp://{node_id}:{router_port}")

        # DEALER socket to send messages to other nodes
        self.dealer = self.context.socket(zmq.DEALER)

        # Connect to all peers' ROUTER sockets
        for peer_id, peer_port in peers:
            self.dealer.connect(f"tcp://{peer_id}:{peer_port}")

        self.node_id = node_id
        self.peers = peers
        self.logger = logger
        self.running = True

        # Start a background thread to handle incoming messages
        self.recv_thread = threading.Thread(target=self.recv_msg_loop, daemon=True)
        self.recv_thread.start()
        self.recv_dict = {}

    def send_msg(self, message):
        """Send a message to a specific peer."""
        print("start send msg")
        data = pickle.dumps({self.node_id: message})
        for _ in range(len(self.peers)):
            self.dealer.send_multipart([data])
            self.logger.log(f"Node {self.node_id} sent message to peers")

    def recv_msg_loop(self):
        """Loop to receive messages from other nodes."""
        while self.running:
            try:
                _, message = self.router.recv_multipart()
                data = pickle.loads(message)
                # self.logger.log()
                self.logger.log(f"Node {self.node_id} received message: {data}")
                # self.recv_dict[address.decode('utf-8')] = data
                for key, val in data.items():
                    self.recv_dict[key] = val

            except zmq.ZMQError as e:
                self.logger.log(f"ZMQ Error: {e}")
                break
    
    def get_from(self, from_node="", retries=1000, time_interal=1):
        # tag = from_node + "_" + key
        for _ in range(retries):
            val = self.recv_dict.get(from_node, None)

            if val is not None:
                del self.recv_dict[from_node]
                self.logger.log(f"Get val from node {from_node}")
                # logging.debug(f"Get val with tag {val}")
                return val
            else:
                time.sleep(time_interal)

        logging.info("Max retry has exceed and result is none.")
        return None

    def stop(self):
        """Stop the node's operations."""
        self.running = False
        self.recv_thread.join()
        self.router.close()
        self.dealer.close()
        self.context.term()


try:
    from apex import amp
except ImportError:
    print('failed to import apex')

# TODO: test
is_test = False

class MMFL_Client(object):
    def __init__(self, args, wandb:wandb, node_id, router_port, peers):
        self.args = args
        self.wandb = wandb

        self.device = None
        self.img_local_trainers = None
        self.txt_local_trainers = None
        self.mm_local_trainers = None
        self.engine = None
        self.vqa_engine = None
        self.best_score = 0
        self.cur_epoch = 0

        # img & txt local dataloaders
        self.img_train_loaders, self.txt_train_loaders = None, None

        # coco global dataloaders
        self.dataloaders_global = None
        self.vqa_dataloader = None
        self.vqa_meta = None
        # universal test dataloader
        self.test_loader = None
        self.vqa_test_loader = None

        self.config = None
        self.set_config()

        self.logger = PythonLogger(output_file=self.config.train.output_file)

        self.img_vec, self.txt_vec = None, None
        self.global_img_feature = None
        self.global_txt_feature = None
        self.distill_index = None
        self.client_node = Node(node_id=node_id, router_port=router_port, peers=peers)
        self.peers = peers

    def set_config(self, img='cifa100', txt='AG_NEWS'):
        self.config = parse_config("./src/coco.yaml", strict_cast=False)
        self.config.train.model_save_path = 'model_last_no_prob'
        self.config.train.best_model_save_path = 'model_best_no_prob'
        self.config.train.output_file = 'model_noprob'
        self.config.model.img_client = img
        self.config.model.txt_client = txt
        self.config.train.model_save_path = self.config.train.model_save_path + '.pth'
        self.config.train.best_model_save_path = self.config.train.best_model_save_path + '.pth'
        self.config.train.output_file = self.config.train.output_file + '.log'

        self.config.model.embed_dim = self.args.feature_dim  # set global model dim

        if self.args.not_bert:
            self.config.model.not_bert = True
            self.config.model.cnn_type = 'resnet50'
        else:
            self.config.model.not_bert = False
            self.config.model.cnn_type = 'resnet101'

    def load_dataset(self, args, is_vqa=False):
        dataset_root = os.environ['HOME'] + '/data/mmdata/MSCOCO/2014'
        vocab_path = './src/custom_datasets/vocabs/coco_vocab.pkl'
        self.dataloaders_global, self.vocab = prepare_coco_dataloaders(self.config.dataloader, dataset_root, args.pub_data_num, args.max_size, vocab_path)

        self.engine = TrainerEngine()
        self.engine.set_logger(self.logger)
        
        if is_vqa:
            self.vqa_engine = VQAEngine(args,self.engine, self.wandb)
            self.config.vqa_dropout = self.args.vqa_dropout

        self.config.optimizer.learning_rate = self.args.server_lr

        self._dataloaders = self.dataloaders_global.copy()
        self.evaluator = COCOEvaluator(eval_method='matmul',
                                       verbose=True,
                                       eval_device='cuda',
                                       n_crossfolds=5)
        if is_vqa:
            vqa_dataset = datasets.load_dataset("HuggingFaceM4/VQAv2", split="train")
            meta = VQAMetaData()
            meta.build_or_load_categories_top()
            self.vqa_meta = meta
            self.vqa_dataloader = vqa2_dataloader(vqa_dataset, train=True, filter_unknown=args.vqa_filter_unknown, meta=meta)
            test_dataset = datasets.load_dataset("HuggingFaceM4/VQAv2", split="validation")
            self.vqa_test_loader = vqa2_dataloader(test_dataset, filter_unknown=args.vqa_filter_unknown, meta=meta)
            self.vqa_engine.create(self.config, self.vocab.word2idx, self.evaluator, self.args.mlp_local, meta)
            #print_model_tree(self.vqa_engine.fusion_model)
            if args.pretrained_model.endswith('_vqa.pt'):
                print(f"Loading pretrained model as VQAEngine {args.pretrained_model}")
                checkpoint = torch.load(args.pretrained_model)
                self.vqa_engine.fusion_model.load_state_dict(checkpoint['vqa'])
                self.best_score = getattr(checkpoint, 'score', self.best_score)
        else:
            self.engine.create(self.config, self.vocab.word2idx, self.evaluator, self.args.mlp_local)
        if args.pretrained_model.endswith('_net.pt'):
            print(f"Loading pretrained model as TrainerEngine {args.pretrained_model}")
            checkpoint = torch.load(args.pretrained_model)
            self.engine.model.load_state_dict(checkpoint['net'])
            if not is_vqa:
                self.best_score = getattr(checkpoint, 'score', self.best_score)

        #print_model_tree(self.engine.model)

        self.train_eval_dataloader = self._dataloaders.pop(
            'train_subset_eval' + f'_{self.args.pub_data_num}') if self._dataloaders is not None else None

        self.engine.model_to_device()
        torch.backends.cudnn.enabled = True
        if self.config.train.get('use_fp16'):
            self.engine.logger.log('Train with half precision')
            if is_vqa:
                self.vqa_engine.to_half()
            else:
                self.engine.to_half()
            

    def create_model(self, args):
        self.logger.log('start creating model and partition datasets')
        self.device = torch.device("cuda:%d" % args.device)
        self.global_round = args.comm_rounds

        os.makedirs(os.environ['HOME'] + f'/data/yClient', exist_ok=True)
        
        alpha = args.alpha # was hard-coded to 0.1
        batch_size = args.batch_size # was hard-coded to 512
        max_size = args.max_size # introduced by xiegeo

        # Create Client Models
        self.img_local_trainers, self.txt_local_trainers, self.mm_local_trainers = [], [], []
        # img clients
        if args.num_img_clients > 0:
            dataset = 'cifar100'
            self.img_trainloaders, test_set = get_FL_trainloader(dataset, os.environ['HOME'] + "/data/cifar100",
                                                                 args.num_img_clients, "hetero", alpha, batch_size, max_size)
            dataset = 'Cifar100'
            dst = os.environ['HOME'] + f'/data/yClient/{dataset}'
            self.img_local_trainers = []
            for i in range(args.num_img_clients):
                self.img_local_trainers.append(
                    ClientTrainer(args, dataset, dst, RGBmean['Cifar100'], RGBstdv['Cifar100'], None, self.logger,
                                  global_test_set=test_set, inter_distance=4, client_id=i, wandb=self.wandb))
                self.img_local_trainers[i].train_loader = self.img_trainloaders[i]
                if is_test and i == 0:
                    break
            self.cur_trainers = self.img_local_trainers
            self.cur_type = "img"

        # txt clients
        if args.num_txt_clients > 0:
            dataset = 'AG_NEWS'
            self.txt_trainloaders, test_set = get_FL_trainloader(dataset, os.environ['HOME'] + "/data",
                                                                 args.num_txt_clients, "hetero", alpha, batch_size, max_size)
            client_id = 1
            dst = os.environ['HOME'] + f'/data/yClient/{dataset}-{client_id}'
            self.txt_local_trainers = []
            for i in range(args.num_txt_clients):
                self.txt_local_trainers.append(
                    ClientTrainer(args, dataset, dst, RGBmean['Cifar100'], RGBstdv['Cifar100'], None, self.logger,
                                  global_test_set=test_set, inter_distance=4, client_id=i, wandb=self.wandb))
                self.txt_local_trainers[i].train_loader = self.txt_trainloaders[i]
                if is_test and i == 0:
                    break

            self.cur_trainers = self.txt_local_trainers
            self.cur_type = "txt"
        # mm clients
        if args.num_mm_clients > 0:
            # mm img models
            config = parse_config("./src/f30k.yaml", strict_cast=False)
            config.model.cache_dir = config.model.cache_dir + '-' + config.train.server_dataset
            config.train.output_file = os.path.join(config.model.cache_dir, config.train.output_file)
            config.train.best_model_save_path = os.path.join(config.model.cache_dir, config.train.best_model_save_path)
            config.train.model_save_path = os.path.join(config.model.cache_dir, config.train.model_save_path)
            config.model.embed_dim = self.args.feature_dim
            config.model.not_bert = True
            self.mm_local_trainers = []
            for client_id in range(args.num_mm_clients):
                self.mm_local_trainers.append(
                    MMClientTrainer(args, config, self.logger, client=client_id, dset_name="flicker30k",
                                    device='cuda',
                                    vocab_path='./src/custom_datasets/vocabs/coco_vocab.pkl',
                                    mlp_local=self.args.mlp_local))
                if is_test and client_id == 0:
                    break
            print(f"MM Clients Samples Num: {[len(i.train_loader.dataset) for i in self.mm_local_trainers]}")
            self.cur_trainers = self.mm_local_trainers
            self.cur_type = "mm"

        self.total_local_trainers = self.img_local_trainers + self.txt_local_trainers + self.mm_local_trainers

        for i in range(len(self.total_local_trainers)):
            self.total_local_trainers[i].client_idx = i + 1

    def train(self, round_n):
        self.cur_epoch = round_n
        # distill_index = data['msg']['distill_index']
        
        global_items = self.client_node.get_from(from_node=self.peers[0][0], time_interal=20)
    
        global_img_feature = global_items["global_img_feature"]
        global_txt_feature = global_items["global_txt_feature"]
        distill_index = global_items["distill_index"]

        # local training and generated representations
        img_vec, img_num = [], []
        txt_vec, txt_num = [], []
        for idx, trainer in enumerate(self.cur_trainers):
            self.logger.log(f"Training Client {trainer.client_idx} in gourds {round_n}!")
            trainer.cur_epoch = round_n
            trainer.run(global_img_feature, global_txt_feature, distill_index,
                        self._dataloaders['train_subset' + f'_{self.args.pub_data_num}'])
            self.logger.log("Generate Local Representations")
            _vec, i = trainer.generate_logits(
                self.dataloaders_global[
                    'train_subset_eval' + f'_{self.args.pub_data_num}'])  # {'img': img_vec, 'txt': txt_vec}
            # if not is_test:
            if distill_index is None:
                distill_index = i
            elif distill_index is not None:
                assert i == distill_index
            if _vec['img'] is not None:
                img_vec.append(_vec['img'])
                img_num.append(len(trainer.train_loader.dataset))
                print(f'img_vec {_vec["img"].shape}')
            if _vec['txt'] is not None:
                txt_vec.append(_vec['txt'])
                txt_num.append(len(trainer.train_loader.dataset))
                print(f'txt_vec {_vec["txt"].shape}')

        # send local item to global
        self.logger.log("start send local msg")
        self.client_node.send_msg({"img_vec":img_vec, "txt_vec":txt_vec, "img_num":img_num, "txt_num":txt_num})

        del img_vec, txt_vec
        gc.collect()

class MMFL_Global(object):
    def __init__(self, args, wandb:wandb, node_id, router_port, peers):
        self.args = args
        self.wandb = wandb

        self.device = None
        self.img_local_trainers = None
        self.txt_local_trainers = None
        self.mm_local_trainers = None
        self.engine = None
        self.vqa_engine = None
        self.best_score = 0
        self.cur_epoch = 0

        # img & txt local dataloaders
        self.img_train_loaders, self.txt_train_loaders = None, None

        # coco global dataloaders
        self.dataloaders_global = None
        self.vqa_dataloader = None
        self.vqa_meta = None
        # universal test dataloader
        self.test_loader = None
        self.vqa_test_loader = None

        self.config = None
        self.set_config()

        self.logger = PythonLogger(output_file=self.config.train.output_file)

        self.img_vec, self.txt_vec = None, None
        self.global_img_feature = None
        self.global_txt_feature = None
        self.distill_index = None
        self.global_node = Node(node_id=node_id, router_port=router_port, peers=peers)
        self.peers = peers

    def set_config(self, img='cifa100', txt='AG_NEWS'):
        self.config = parse_config("./src/coco.yaml", strict_cast=False)
        self.config.train.model_save_path = 'model_last_no_prob'
        self.config.train.best_model_save_path = 'model_best_no_prob'
        self.config.train.output_file = 'model_noprob'
        self.config.model.img_client = img
        self.config.model.txt_client = txt
        self.config.train.model_save_path = self.config.train.model_save_path + '.pth'
        self.config.train.best_model_save_path = self.config.train.best_model_save_path + '.pth'
        self.config.train.output_file = self.config.train.output_file + '.log'

        self.config.model.embed_dim = self.args.feature_dim  # set global model dim

        if self.args.not_bert:
            self.config.model.not_bert = True
            self.config.model.cnn_type = 'resnet50'
        else:
            self.config.model.not_bert = False
            self.config.model.cnn_type = 'resnet101'

    def load_dataset(self, args, is_vqa=False):
        dataset_root = os.environ['HOME'] + '/data/mmdata/MSCOCO/2014'
        vocab_path = './src/custom_datasets/vocabs/coco_vocab.pkl'
        self.dataloaders_global, self.vocab = prepare_coco_dataloaders(self.config.dataloader, dataset_root, args.pub_data_num, args.max_size, vocab_path)

        self.engine = TrainerEngine()
        self.engine.set_logger(self.logger)
        
        if is_vqa:
            self.vqa_engine = VQAEngine(args,self.engine, self.wandb)
            self.config.vqa_dropout = self.args.vqa_dropout

        self.config.optimizer.learning_rate = self.args.server_lr

        self._dataloaders = self.dataloaders_global.copy()
        self.evaluator = COCOEvaluator(eval_method='matmul',
                                       verbose=True,
                                       eval_device='cuda',
                                       n_crossfolds=5)
        if is_vqa:
            train_f = "/root/xus/CreamFL-main/vq_av2-train.arrow"
            vqa_dataset = datasets.Dataset.from_file(train_f)
            
            test_f = "/root/xus/CreamFL-main/vq_av2-validation.arrow"
            test_dataset = datasets.Dataset.from_file(test_f)
            meta = VQAMetaData()
            meta.build_or_load_categories_top()
            self.vqa_meta = meta
            self.vqa_dataloader = vqa2_dataloader(vqa_dataset, train=True, filter_unknown=args.vqa_filter_unknown, meta=meta)

            self.vqa_test_loader = vqa2_dataloader(test_dataset, filter_unknown=args.vqa_filter_unknown, meta=meta)
            self.vqa_engine.create(self.config, self.vocab.word2idx, self.evaluator, self.args.mlp_local, meta)
            #print_model_tree(self.vqa_engine.fusion_model)
            if args.pretrained_model.endswith('_vqa.pt'):
                print(f"Loading pretrained model as VQAEngine {args.pretrained_model}")
                checkpoint = torch.load(args.pretrained_model)
                self.vqa_engine.fusion_model.load_state_dict(checkpoint['vqa'])
                self.best_score = getattr(checkpoint, 'score', self.best_score)
        else:
            self.engine.create(self.config, self.vocab.word2idx, self.evaluator, self.args.mlp_local)
        if args.pretrained_model.endswith('_net.pt'):
            print(f"Loading pretrained model as TrainerEngine {args.pretrained_model}")
            checkpoint = torch.load(args.pretrained_model)
            self.engine.model.load_state_dict(checkpoint['net'])
            if not is_vqa:
                self.best_score = getattr(checkpoint, 'score', self.best_score)

        #print_model_tree(self.engine.model)

        self.train_eval_dataloader = self._dataloaders.pop(
            'train_subset_eval' + f'_{self.args.pub_data_num}') if self._dataloaders is not None else None

        self.engine.model_to_device()
        torch.backends.cudnn.enabled = True
        if self.config.train.get('use_fp16'):
            self.engine.logger.log('Train with half precision')
            if is_vqa:
                self.vqa_engine.to_half()
            else:
                self.engine.to_half()
            

    def create_model(self, args):
        self.logger.log('start creating model and partition datasets')
        self.device = torch.device("cuda:%d" % args.device)
        self.global_round = args.comm_rounds

        os.makedirs(os.environ['HOME'] + f'/data/yClient', exist_ok=True)
        
        alpha = args.alpha # was hard-coded to 0.1
        batch_size = args.batch_size # was hard-coded to 512
        max_size = args.max_size # introduced by xiegeo

        # Create Client Models
        self.img_local_trainers, self.txt_local_trainers, self.mm_local_trainers = [], [], []
        # img clients
        if args.num_img_clients > 0:
            dataset = 'cifar100'
            self.img_trainloaders, test_set = get_FL_trainloader(dataset, os.environ['HOME'] + "/data/cifar100",
                                                                 args.num_img_clients, "hetero", alpha, batch_size, max_size)
            dataset = 'Cifar100'
            dst = os.environ['HOME'] + f'/data/yClient/{dataset}'
            self.img_local_trainers = []
            for i in range(args.num_img_clients):
                self.img_local_trainers.append(
                    ClientTrainer(args, dataset, dst, RGBmean['Cifar100'], RGBstdv['Cifar100'], None, self.logger,
                                  global_test_set=test_set, inter_distance=4, client_id=i, wandb=self.wandb))
                self.img_local_trainers[i].train_loader = self.img_trainloaders[i]
                if is_test and i == 0:
                    break
        # txt clients
        if args.num_txt_clients > 0:
            dataset = 'AG_NEWS'
            self.txt_trainloaders, test_set = get_FL_trainloader(dataset, os.environ['HOME'] + "/data",
                                                                 args.num_txt_clients, "hetero", alpha, batch_size, max_size)
            client_id = 1
            dst = os.environ['HOME'] + f'/data/yClient/{dataset}-{client_id}'
            self.txt_local_trainers = []
            for i in range(args.num_txt_clients):
                self.txt_local_trainers.append(
                    ClientTrainer(args, dataset, dst, RGBmean['Cifar100'], RGBstdv['Cifar100'], None, self.logger,
                                  global_test_set=test_set, inter_distance=4, client_id=i, wandb=self.wandb))
                self.txt_local_trainers[i].train_loader = self.txt_trainloaders[i]
                if is_test and i == 0:
                    break
        # mm clients
        if args.num_mm_clients > 0:
            # mm img models
            config = parse_config("./src/f30k.yaml", strict_cast=False)
            config.model.cache_dir = config.model.cache_dir + '-' + config.train.server_dataset
            config.train.output_file = os.path.join(config.model.cache_dir, config.train.output_file)
            config.train.best_model_save_path = os.path.join(config.model.cache_dir, config.train.best_model_save_path)
            config.train.model_save_path = os.path.join(config.model.cache_dir, config.train.model_save_path)
            config.model.embed_dim = self.args.feature_dim
            config.model.not_bert = True
            self.mm_local_trainers = []
            for client_id in range(args.num_mm_clients):
                self.mm_local_trainers.append(
                    MMClientTrainer(args, config, self.logger, client=client_id, dset_name="flicker30k",
                                    device='cuda',
                                    vocab_path='./src/custom_datasets/vocabs/coco_vocab.pkl',
                                    mlp_local=self.args.mlp_local))
                if is_test and client_id == 0:
                    break
            print(f"MM Clients Samples Num: {[len(i.train_loader.dataset) for i in self.mm_local_trainers]}")

        self.total_local_trainers = self.img_local_trainers + self.txt_local_trainers + self.mm_local_trainers

        for i in range(len(self.total_local_trainers)):
            self.total_local_trainers[i].client_idx = i + 1

    def train(self, round_n):
        # for round_n in range(self.global_round):
        self.cur_epoch = round_n

        self.cur_trainers = self.total_local_trainers
        
        self.logger.log(f"Round {round_n}!")

        if not is_test and not self.args.no_retrieval_training:
            # global training
            print("start retrieval training")
            self.engine.train(tr_loader=self._dataloaders['train_subset' + f'_{self.args.pub_data_num}'])

        # global representations
        # if len(self.cur_trainers) == 0:
        #     print("No clients to train, skipping global representations")
        if self.args.agg_method == "con_w" or self.args.contrast_local_intra or self.args.contrast_local_inter:
            img_feature, txt_feature = [], []
            distill_index = []
            for idx, (images, captions, captions_word, caption_lens, _, _, index) in tqdm(
                    enumerate(self.dataloaders_global['train_subset_eval' + f'_{self.args.pub_data_num}']),
                    desc="Global Representations",
                    total=len(self.dataloaders_global['train_subset_eval' + f'_{self.args.pub_data_num}'])):
                with torch.no_grad():
                    images = images.to(self.engine.device)  # [bs, 3, 224, 224]
                    captions = captions.to(self.engine.device)  # [bs, seq_len]
                    caption_lens = caption_lens.to(self.engine.device)

                    output = self.engine.model(images, captions, captions_word, caption_lens)
                    out_img = output['image_features']
                    out_txt = output['caption_features']

                    out_img = out_img.cpu().detach()
                    out_txt = out_txt.cpu().detach()

                    img_feature.append(out_img)
                    txt_feature.append(out_txt)
                    distill_index.extend(index)

            self.global_img_feature = torch.concat(img_feature, dim=0)
            self.global_txt_feature = torch.concat(txt_feature, dim=0)
            print(self.global_txt_feature.shape, self.global_img_feature.shape)
            self.distill_index = distill_index
            del img_feature, txt_feature
            gc.collect()
        else:
            print("No agg_method or contrast, skipping global representations")
        
        self.global_node.send_msg({"global_img_feature":self.global_img_feature, "global_txt_feature":self.global_txt_feature, "distill_index": self.distill_index})

        img_num = None
        img_vec = None
        txt_num = None
        txt_vec = None
        
        for cur_peer in self.peers:
            cur_node = cur_peer[0]
            cur_items = self.global_node.get_from(from_node=cur_node, time_interal=20)
            cur_img_num = cur_items["img_num"]
            cur_img_vec = cur_items["img_vec"]
            cur_txt_num = cur_items["txt_num"]
            cur_txt_vec = cur_items["txt_vec"]
            
            if img_num is None:
                img_num = cur_img_num
            else:
                img_num += cur_img_num
            
            if img_vec is None:
                img_vec = cur_img_vec
            else:
                img_vec += cur_img_vec
            
            if txt_num is None:
                txt_num = cur_txt_num
            else:
                txt_num += cur_txt_num
            
            if txt_vec is None:
                txt_vec = cur_txt_vec
            else:
                txt_vec += cur_txt_vec

        # global distillation
        if not self.args.disable_distill:
            print("**********start distill*************")
            self.distill(round_n, img_vec, txt_vec, img_num, txt_num, self.distill_index)

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        # assert round_n + 1 == self.cur_epoch, "inconstant round_n vs cur_epoch, added to check that code clean up does not change logic."
        
        # record after each epoch training
        metadata = self.engine.metadata.copy()
        metadata['cur_epoch'] = round_n + 1
        metadata['lr'] = get_lr(self.engine.optimizer)
        
        score = 0
        
        test_scores = self.engine.evaluate({'test': self._dataloaders['test']})
        self.engine.report_scores(step=round_n + 1,
                                scores=test_scores,
                                metadata=metadata,
                                prefix=self.engine.eval_prefix)
        rsum = test_scores['test']['n_fold']['i2t']['recall_1'] + test_scores['test']['n_fold']['t2i']['recall_1'] + \
            test_scores['test']['i2t']['recall_1'] + test_scores['test']['t2i']['recall_1']
        self.wandb.log({"Server rsum_r1": rsum}, step=self.cur_epoch)
        self.wandb.log({"Server rsum": test_scores['test']['rsum']}, step=self.cur_epoch)
        self.wandb.log({"Server n_fold_i2t_r1": test_scores['test']['n_fold']['i2t']['recall_1']}, step=self.cur_epoch)
        self.wandb.log({"Server n_fold_t2i_r1": test_scores['test']['n_fold']['t2i']['recall_1']}, step=self.cur_epoch)
        self.wandb.log({"Server i2t_r1": test_scores['test']['i2t']['recall_1']}, step=self.cur_epoch)
        self.wandb.log({"Server t2i_r1": test_scores['test']['t2i']['recall_1']}, step=self.cur_epoch)
        score = rsum
        
        if self.vqa_engine is not None:
            test_loader = None
            if round_n == 0:
                test_loader = self.vqa_test_loader # only test during training in the first round
            self.vqa_engine.train_vqa(self.cur_epoch, self.vqa_dataloader, vqa2_test_dataloader=test_loader)
            test_scores = vqa_validation(10000, self.vqa_engine.fusion_model, self.vqa_meta, self.vqa_test_loader)
            #test_scores = vqa_validation(100000, self.vqa_engine.fusion_model, self.vqa_meta, self.vqa_test_loader)
            self.wandb.log(test_scores, step=self.cur_epoch)
            score = test_scores['accuracy']
        
        # print(f"**********current vqa accuracy {score}*************")
        print(f"**********current score {score}*************")
        

        def save_model(type_name, score=score):
            prefix = f'{self.args.name}_{type_name}_{self.args.feature_dim}'
            if self.vqa_engine is not None:
                torch.save({'vqa': self.vqa_engine.fusion_model.state_dict(),
                            'score':score}, f'{prefix}_vqa.pt')
            else:
                torch.save({'net': self.engine.model.state_dict(),
                            'score':score},  f'{prefix}_net.pt')
            
        
        if self.best_score < score:
            best_score = score
            metadata['best_score'] = best_score
            metadata['best_epoch'] = round_n + 1
            self.best_metadata, self.best_scores = metadata, test_scores
            save_model("best")

        if round_n == self.args.comm_rounds - 1:
            save_model("last")

        self.engine.lr_scheduler.step()
        if self.vqa_engine is not None:
            self.vqa_engine.vqa_lr_scheduler.step()

        del img_vec, txt_vec
        gc.collect()

    def distill(self, round_n, img_vec, txt_vec, img_num, txt_num, distill_index):
        
        if len(img_vec) == 0 and len(txt_vec) == 0:
            print("No img_vec and txt_vec to distill (no clients)")
            return

        self.engine.model.train()

        if self.config.model.use_img_client or self.config.model.use_txt_client or self.config.model.use_mm_client:
            client_loss_cri = nn.MSELoss()

        def aggregation(i_vec=img_vec, t_vec=txt_vec):
            if self.args.agg_method == "con_w":
                if not i_vec:
                     self.logger.log("distill.aggregation i_vec is empty")
                else:
                    contrastive_w = []
                    for vec in i_vec:  # vec: [50000, n_feature], global_txt_feature: [50000, n_feature]
                        logits = torch.matmul(vec, self.global_txt_feature.T)  # [50000, 50000]
                        exp_logits = torch.exp(logits)
                        log_prob = logits - torch.log(torch.sum(exp_logits, dim=1, keepdim=True))
                        contrastive_w.append(torch.diagonal(log_prob).reshape(1, -1))
                    if not contrastive_w:
                        self.logger.log("distill.aggregation No tensors were added to contrastive_w for images")
                    else:
                        contrastive_w = torch.softmax(torch.cat(contrastive_w, dim=0), dim=0)
                        for i in range(len(i_vec)):
                            i_vec[i] = (i_vec[i] * contrastive_w[i].reshape(-1, 1)).unsqueeze(0)
                    i_vec = torch.sum(torch.cat(i_vec, dim=0), dim=0)  # aggregated image vectors

                if not t_vec:
                     self.logger.log("distill.aggregation t_vec is empty")
                else:
                    contrastive_w = []
                    for vec in t_vec:  # vec: [50000, n_feature], global_txt_feature: [50000, n_feature]
                        logits = torch.matmul(vec, self.global_img_feature.T)  # [50000, 50000]
                        exp_logits = torch.exp(logits)
                        log_prob = logits - torch.log(torch.sum(exp_logits, dim=1, keepdim=True))
                        contrastive_w.append(torch.diagonal(log_prob).reshape(1, -1))
                    if not contrastive_w:
                        self.logger.log("distill.aggregation No tensors were added to contrastive_w for texts")
                    else:
                        contrastive_w = torch.softmax(torch.cat(contrastive_w, dim=0), dim=0)
                        for i in range(len(t_vec)):
                            t_vec[i] = (t_vec[i] * contrastive_w[i].reshape(-1, 1)).unsqueeze(0)
                    t_vec = torch.sum(torch.cat(t_vec, dim=0), dim=0)  # aggregated text vectors
            else:
                raise NotImplementedError

            return i_vec, t_vec

        # aggregation
        img_vec, txt_vec = aggregation()

        self.img_vec = img_vec
        self.txt_vec = txt_vec

        distill_dict = {b: a for a, b in enumerate(distill_index)}  # index in coco to index to list 'distill_index'
        # distill
        self.logger.log("start distilling")
        for idx, (images, captions, captions_word, caption_lens, _, _, index) in tqdm(
                enumerate(self.dataloaders_global['train_subset' + f'_{self.args.pub_data_num}'])):
            images = images.to(self.engine.device)  # [bs, 3, 224, 224]
            captions = captions.to(self.engine.device)  # [bs, seq_len]
            caption_lens = caption_lens.to(self.engine.device)

            output = self.engine.model(images, captions, captions_word, caption_lens)
            loss = 0

            def code_sim(output, target, config):
                output = output.sum(axis=1) if len(output.shape) == 3 else output
                target = target.type_as(output)

                return client_loss_cri(output, target.type_as(output))

            if len(img_num)> 0:
            # if self.args.num_img_clients > 0 and len(img_num)> 0:
                out_img = output['image_features']
                d_idx = operator.itemgetter(*index)(distill_dict)  # idx of the current batch
                target_img = self.img_vec[d_idx, :].type_as(out_img)
                loss += self.args.kd_weight * code_sim(out_img, target_img, self.config)
            # if self.args.num_txt_clients > 0 and len(txt_num) > 0:
            if len(txt_num) > 0:
                out_txt = output['caption_features']
                d_idx = operator.itemgetter(*index)(distill_dict)  # idx of the current batch
                target_txt = self.txt_vec[d_idx, :].type_as(out_txt)
                loss += self.args.kd_weight * code_sim(out_txt, target_txt, self.config)
            if len(img_num) > 0 and len(txt_num) > 0:
            # if self.args.num_mm_clients > 0 and len(img_num) > 0 and len(txt_num) > 0:
                out_img = output['image_features']
                d_idx = operator.itemgetter(*index)(distill_dict)  # idx of the current batch
                target_img = self.img_vec[d_idx, :].type_as(out_img)
                out_txt = output['caption_features']
                target_txt = self.txt_vec[d_idx, :].type_as(out_txt)
                loss += self.args.kd_weight * code_sim(out_img, target_img, self.config)
                loss += self.args.kd_weight * code_sim(out_txt, target_txt, self.config)

            self.engine.optimizer.zero_grad()

            if self.config.train.get('use_fp16'):
                with amp.scale_loss(loss, self.engine.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if self.config.train.grad_clip > 0:
                nn.utils.clip_grad.clip_grad_norm_(self.engine.model.parameters(),
                                                   self.config.train.grad_clip)
            self.engine.optimizer.step()
