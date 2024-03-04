# moved from MMFL.py with client logic removed

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

class Global:
    def __init__(self, args, wandb=None):
        self.args = args
        self.wandb = wandb

        self.device = None
        self.engine = None

        # coco global dataloaders
        self.dataloaders_global = None
        # universal test dataloader
        self.test_loader = None

        self.config = None
        self.set_config()

        self.logger = PythonLogger(output_file=self.config.train.output_file)

        self.img_vec, self.txt_vec = None, None
        self.global_img_feature = None
        self.global_txt_feature = None
        self.distill_index = None

    def set_config(self):
        self.config = parse_config("./src/coco.yaml", strict_cast=False)
        self.config.train.model_save_path = 'model_last_no_prob'
        self.config.train.best_model_save_path = 'model_best_no_prob'
        self.config.train.output_file = 'model_noprob'
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

    def load_dataset(self, args):
        dataset_root = os.environ['HOME'] + '/data/mmdata/MSCOCO/2014'
        vocab_path = './src/datasets/vocabs/coco_vocab.pkl'
        self.dataloaders_global, self.vocab = prepare_coco_dataloaders(self.config.dataloader, dataset_root, args.pub_data_num, args.max_size, vocab_path)

        self.engine = TrainerEngine()
        self.engine.set_logger(self.logger)

        self.config.optimizer.learning_rate = self.args.server_lr

        self._dataloaders = self.dataloaders_global.copy()
        self.evaluator = COCOEvaluator(eval_method='matmul',
                                       verbose=True,
                                       eval_device='cuda',
                                       n_crossfolds=5)
        self.engine.create(self.config, self.vocab.word2idx, self.evaluator, self.args.mlp_local)

        self.train_eval_dataloader = self._dataloaders.pop(
            'train_subset_eval' + f'_{self.args.pub_data_num}') if self._dataloaders is not None else None

        self.engine.model_to_device()
        torch.backends.cudnn.enabled = True
        if self.config.train.get('use_fp16'):
            self.engine.logger.log('Train with half precision')
            self.engine.to_half()

    def train(self, round_n):
        self.cur_epoch = round_n

        if not is_test:
            self.logger.log(f"Global training round {round_n + 1}!")
            self.engine.train(
                tr_loader=self._dataloaders['train_subset' + f'_{self.args.pub_data_num}'])  # global train

        # global representations
        if self.args.agg_method == "con_w" or self.args.contrast_local_intra or self.args.contrast_local_inter:
            img_feature, txt_feature = [], []
            distill_index = []
            for idx, (images, captions, captions_word, caption_lens, _, _, index) in tqdm(
                    enumerate(self.dataloaders_global['train_subset_eval' + f'_{self.args.pub_data_num}']),
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

        # waiting and retrieving local representations
        img_vec, img_num = [], []
        txt_vec, txt_num = [], []
        for idx, trainer in enumerate(self.cur_trainers):
            self.logger.log(f"Training Client {trainer.client_idx}!")
            trainer.cur_epoch = round_n
            trainer.run(self.global_img_feature, self.global_txt_feature, self.distill_index,
                        self._dataloaders['train_subset' + f'_{self.args.pub_data_num}'])
            self.logger.log("Download Local Representations")
            _vec, i = trainer.generate_logits(
                self.dataloaders_global[
                    'train_subset_eval' + f'_{self.args.pub_data_num}'])  # {'img': img_vec, 'txt': txt_vec}
            # if not is_test:
            if self.distill_index is None:
                self.distill_index = i
            elif self.distill_index is not None:
                assert i == self.distill_index
            if _vec['img'] is not None:
                img_vec.append(_vec['img'])
                img_num.append(len(trainer.train_loader.dataset))
                print(f'img_vec {_vec["img"].shape}')
            if _vec['txt'] is not None:
                txt_vec.append(_vec['txt'])
                txt_num.append(len(trainer.train_loader.dataset))
                print(f'txt_vec {_vec["txt"].shape}')

        # global distillation
        if not self.args.disable_distill:
            self.distill(round_n, img_vec, txt_vec, img_num, txt_num, self.distill_index)

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        # record after each epoch training
        metadata = self.engine.metadata.copy()
        metadata['cur_epoch'] = round_n + 1
        metadata['lr'] = get_lr(self.engine.optimizer)

        test_scores = self.engine.evaluate({'test': self._dataloaders['test']})
        self.engine.report_scores(step=round_n + 1,
                                  scores=test_scores,
                                  metadata=metadata,
                                  prefix=self.engine.eval_prefix)
        rsum = test_scores['test']['n_fold']['i2t']['recall_1'] + test_scores['test']['n_fold']['t2i']['recall_1'] + \
               test_scores['test']['i2t']['recall_1'] + test_scores['test']['t2i']['recall_1']
        self.wandb.log({"Server rsum_r1": rsum}, step=self.cur_epoch)
        self.wandb.log({"Server n_fold_i2t_r1": test_scores['test']['n_fold']['i2t']['recall_1']}, step=self.cur_epoch)
        self.wandb.log({"Server n_fold_t2i_r1": test_scores['test']['n_fold']['t2i']['recall_1']}, step=self.cur_epoch)
        self.wandb.log({"Server i2t_r1": test_scores['test']['i2t']['recall_1']}, step=self.cur_epoch)
        self.wandb.log({"Server t2i_r1": test_scores['test']['t2i']['recall_1']}, step=self.cur_epoch)

        if self.best_score < rsum:
            best_score = rsum
            metadata['best_score'] = best_score
            metadata['best_epoch'] = round_n + 1
            self.best_metadata, self.best_scores = metadata, test_scores

            torch.save({'net': self.engine.model.state_dict()}, self.args.name + '-best_model.pt')

        if round_n == self.args.comm_rounds - 1:
            torch.save({'net': self.engine.model.state_dict()}, self.args.name + '-last_model.pt')

        self.engine.lr_scheduler.step()

        del img_vec, txt_vec
        gc.collect()

    def distill(self, round_n, img_vec, txt_vec, img_num, txt_num, distill_index):

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

            if self.args.num_img_clients > 0 and len(img_num)> 0:
                out_img = output['image_features']
                d_idx = operator.itemgetter(*index)(distill_dict)  # idx of the current batch
                target_img = self.img_vec[d_idx, :].type_as(out_img)
                loss += self.args.kd_weight * code_sim(out_img, target_img, self.config)
            if self.args.num_txt_clients > 0 and len(txt_num) > 0:
                out_txt = output['caption_features']
                d_idx = operator.itemgetter(*index)(distill_dict)  # idx of the current batch
                target_txt = self.txt_vec[d_idx, :].type_as(out_txt)
                loss += self.args.kd_weight * code_sim(out_txt, target_txt, self.config)
            if self.args.num_mm_clients > 0 and len(img_num) > 0 and len(txt_num) > 0:
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
