import copy
import operator

import torch.optim
import torch.utils.data

torch.backends.cudnn.enabled = True

import numpy as np
import os
import random
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

import torch.nn as nn

from src.algorithms.base import EngineBase
from tqdm import tqdm
import torch

try:
    from apex import amp
except ImportError:
    print('failed to import apex')

from src.utils.serialize_utils import flatten_dict


def seed_torch(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


##################################################
# step -1: Predefined function
##################################################
import torch.utils.data.sampler as sampler


class SubsetSampler(sampler.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def cur_step(cur_epoch, idx, N, fmt=None):
    _cur_step = cur_epoch + idx / N
    if fmt:
        return fmt.format(_cur_step)
    else:
        return _cur_step


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


gpuid = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# TODO: test
is_test = False


class MMClientTrainer(EngineBase):

    def run(self, global_img_feature, global_txt_feature, distill_index, global_train_loader, prefix=''):
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval().cuda()
        self.model.cuda()
        if self.local_epoch == 0:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                                        opt_level='O2')
        self.model.train()

        for i in range(self.local_epochs):
            self.local_epoch += 1
            if self.logger is not None:
                self.logger.log(f"Epoch {self.local_epoch}")
            self.train_epoch(global_img_feature, global_txt_feature, distill_index, global_train_loader, prefix='')

        if self.args.save_client:
            torch.save(self.model.state_dict(), f'./saved_clients/Flicker30K/Client{self.client}-model_{self.local_epoch}.pth')

        self.old_model.cpu()
        self.model.cpu()

        del self.old_model
        import gc
        gc.collect()

    def train_epoch(self, global_img_feature, global_txt_feature, distill_index, global_train_loader, prefix=''):

        for idx, (images, captions, captions_word, caption_lens, _, _, index) in enumerate(self.train_loader):
            images = images.to(self.device)
            captions = captions.to(self.device)
            caption_lens = caption_lens.to(self.device)

            output = self.model(images, captions, captions_word, caption_lens)
            # print('img', output['image_features'].shape)
            # print('txt', output['caption_features'].shape)

            loss, loss_dict = self.criterion(**output)

            self.optimizer.zero_grad()

            if self.config.train.get('use_fp16'):
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if self.config.train.grad_clip > 0:
                nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),
                                                   self.config.train.grad_clip)
            self.optimizer.step()

            if is_test:
                break

        loss_dict = {'{}{}'.format(prefix, key): val
                     for key, val in loss_dict.items()}
        loss_dict['step'] = cur_step(self.cur_epoch, idx, len(self.train_loader))

        criterion = nn.CrossEntropyLoss().cuda()
        if self.args.contrast_local_intra and self.args.contrast_local_inter:
            global_img_feature, global_txt_feature = global_img_feature.cuda(), global_txt_feature.cuda()
            distill_dict = {b: a for a, b in enumerate(distill_index)}  # index in coco to index to list 'distill_index'
            print("Start Intra & Inter Contrasting!")
            for idx, (images, captions, captions_word, caption_lens, _, _, index) in tqdm(enumerate(global_train_loader), total=len(global_train_loader)):
                self.optimizer.zero_grad()
                d_idx = operator.itemgetter(*index)(distill_dict)  # idx of current batch

                images = images.to(self.device)
                captions = captions.to(self.device)
                caption_lens = caption_lens.to(self.device)

                output = self.model(images, captions, captions_word, caption_lens)

                out_img = output['image_features'].sum(axis=1) if len(output['image_features'].shape) == 3 else output[
                    'image_features']
                out_txt = output['caption_features'].sum(axis=1) if len(output['caption_features'].shape) == 3 else \
                    output['caption_features']

                target_img_feature = global_img_feature[d_idx, :].type_as(out_img)
                target_txt_feature = global_txt_feature[d_idx, :].type_as(out_txt)

                # pos
                pos_i = torch.sum(out_img * target_img_feature, dim=-1)
                pos_i = pos_i.reshape(-1, 1)
                pos_t = torch.sum(out_txt * target_txt_feature, dim=-1)
                pos_t = pos_t.reshape(-1, 1)
                # neg
                with torch.no_grad():
                    output_o = self.old_model(images, captions, captions_word, caption_lens)
                    out_img_o = output_o['image_features'].sum(axis=1) if len(output_o['image_features'].shape) == 3 else output_o['image_features']
                    out_txt_o = output_o['caption_features'].sum(axis=1) if len(output_o['caption_features'].shape) == 3 else output_o['caption_features']
                neg_i = torch.sum(out_img * out_img_o, dim=-1)
                neg_t = torch.sum(out_txt * out_txt_o, dim=-1)
                logits_1 = torch.cat((pos_i, neg_i.reshape(-1, 1)), dim=1)
                logits_2 = torch.cat((pos_t, neg_t.reshape(-1, 1)), dim=1)
                logits = torch.cat((logits_1, logits_2), dim=0)

                logits /= 0.5  # temperature
                labels = torch.zeros(images.size(0) * 2).cuda().long()

                loss_intra = criterion(logits, labels)

                # inter contrast
                logits_1_inter = torch.div(torch.matmul(out_img, global_txt_feature.T), 0.5)
                logits_2_inter = torch.div(torch.matmul(out_txt, global_img_feature.T), 0.5)

                labels_inter = torch.tensor(d_idx).cuda()

                loss_1_inter = criterion(logits_1_inter, labels_inter)
                loss_2_inter = criterion(logits_2_inter, labels_inter)
                loss_inter = loss_1_inter + loss_2_inter

                if not self.args.loss_scale:
                    loss = (loss_intra + loss_inter) * self.args.interintra_weight
                else:
                    loss = (loss_intra + loss_inter / (loss_inter / loss_intra).detach()) * self.args.interintra_weight

                self.optimizer.zero_grad()

                if self.config.train.get('use_fp16'):
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if self.config.train.grad_clip > 0:
                    nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),
                                                       self.config.train.grad_clip)
                self.optimizer.step()

                if is_test:
                    break
        elif self.args.contrast_local_intra:
            global_img_feature, global_txt_feature = global_img_feature.cuda(), global_txt_feature.cuda()
            distill_dict = {b: a for a, b in enumerate(distill_index)}  # index in coco to index to list 'distill_index'
            print("Start Intra Contrasting!")
            for idx, (images, captions, captions_word, caption_lens, _, _, index) in tqdm(enumerate(global_train_loader), total=len(global_train_loader)):
                self.optimizer.zero_grad()
                d_idx = operator.itemgetter(*index)(distill_dict)  # batchidx

                images = images.to(self.device)
                captions = captions.to(self.device)
                caption_lens = caption_lens.to(self.device)

                output = self.model(images, captions, captions_word, caption_lens)

                out_img = output['image_features'].sum(axis=1) if len(output['image_features'].shape) == 3 else output[
                    'image_features']
                out_txt = output['caption_features'].sum(axis=1) if len(output['caption_features'].shape) == 3 else \
                    output['caption_features']

                target_img_feature = global_img_feature[d_idx, :].type_as(out_img)
                target_txt_feature = global_txt_feature[d_idx, :].type_as(out_txt)

                # pos
                pos_i = torch.sum(out_img * target_img_feature, dim=-1)
                pos_i = pos_i.reshape(-1, 1)
                pos_t = torch.sum(out_txt * target_txt_feature, dim=-1)
                pos_t = pos_t.reshape(-1, 1)
                # neg
                with torch.no_grad():
                    output_o = self.old_model(images, captions, captions_word, caption_lens)
                    out_img_o = output_o['image_features'].sum(axis=1) if len(output_o['image_features'].shape) == 3 else output_o['image_features']
                    out_txt_o = output_o['caption_features'].sum(axis=1) if len(output_o['caption_features'].shape) == 3 else output_o['caption_features']
                neg_i = torch.sum(out_img * out_img_o, dim=-1)
                neg_t = torch.sum(out_txt * out_txt_o, dim=-1)
                logits_1 = torch.cat((pos_i, neg_i.reshape(-1, 1)), dim=1)
                logits_2 = torch.cat((pos_t, neg_t.reshape(-1, 1)), dim=1)
                logits = torch.cat((logits_1, logits_2), dim=0)

                logits /= 0.5  # temperature
                labels = torch.zeros(images.size(0) * 2).cuda().long()

                loss = criterion(logits, labels)

                self.optimizer.zero_grad()

                if self.config.train.get('use_fp16'):
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if self.config.train.grad_clip > 0:
                    nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),
                                                       self.config.train.grad_clip)
                self.optimizer.step()

                if is_test:
                    break
        elif self.args.contrast_local_inter:
            global_img_feature, global_txt_feature = global_img_feature.cuda(), global_txt_feature.cuda()
            distill_dict = {b: a for a, b in enumerate(distill_index)}  # index in coco to index to list 'distill_index'
            print("Start Inter-modal Contrasting!")
            for idx, (images, captions, captions_word, caption_lens, _, _, index) in tqdm(enumerate(global_train_loader),
                                                                           total=len(global_train_loader)):
                self.optimizer.zero_grad()
                d_idx = operator.itemgetter(*index)(distill_dict)  # batchidx

                images = images.to(self.device)
                captions = captions.to(self.device)
                caption_lens = caption_lens.to(self.device)

                output = self.model(images, captions, captions_word, caption_lens)

                out_img = output['image_features'].sum(axis=1) if len(output['image_features'].shape) == 3 else output[
                    'image_features']
                out_txt = output['caption_features'].sum(axis=1) if len(output['caption_features'].shape) == 3 else \
                    output['caption_features']

                logits_1 = torch.div(torch.matmul(out_img, global_txt_feature.T), 0.5)
                logits_2 = torch.div(torch.matmul(out_txt, global_img_feature.T), 0.5)

                labels = torch.tensor(d_idx).cuda()

                loss_1 = criterion(logits_1, labels)
                loss_2 = criterion(logits_2, labels)
                loss = loss_1 + loss_2

                self.optimizer.zero_grad()

                if self.config.train.get('use_fp16'):
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if self.config.train.grad_clip > 0:
                    nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),
                                                       self.config.train.grad_clip)
                self.optimizer.step()

                if is_test:
                    break

    def generate_logits(self, dataloader):
        self.model.cuda()
        self.model.eval()
        with torch.no_grad():
            img_vec = []
            txt_vec = []
            distill_index = []
            for idx, (images, captions, captions_word, caption_lens, _, _, index) in tqdm(enumerate(dataloader),
                                                                           total=len(dataloader)):
                images = images.to(self.device)
                captions = captions.to(self.device)
                caption_lens = caption_lens.to(self.device)

                output = self.model(images, captions, captions_word, caption_lens)

                out_img = output['image_features'].sum(axis=1) if len(output['image_features'].shape) == 3 else output[
                    'image_features']
                out_txt = output['caption_features'].sum(axis=1) if len(output['caption_features'].shape) == 3 else \
                    output['caption_features']
                img_vec.extend(out_img)
                txt_vec.extend(out_txt)
                distill_index.extend(index)

                if is_test and idx == 1:
                    break

        img_vec = torch.cat(img_vec, dim=0).view(-1, self.args.feature_dim)
        txt_vec = torch.cat(txt_vec, dim=0).view(-1, self.args.feature_dim)

        img_vec = img_vec.cpu()
        txt_vec = txt_vec.cpu()
        self.model.cpu()

        return {'img': img_vec, 'txt': txt_vec}, distill_index

    def report_scores(self, step, scores, metadata, prefix=''):
        report_dict = {data_key: flatten_dict(_scores, sep='_')
                       for data_key, _scores in scores.items()}
        report_dict = flatten_dict(report_dict, sep='__')
        tracker_data = report_dict.copy()

        report_dict = {'{}{}'.format(prefix, key): val for key, val in report_dict.items()}
        report_dict['step'] = step
        tracker_data['metadata'] = metadata
        tracker_data['scores'] = scores
