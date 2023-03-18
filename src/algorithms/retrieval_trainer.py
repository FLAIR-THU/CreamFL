import datetime
import sys

import torch.nn as nn

from tqdm import tqdm

import hashlib
import json
import munch

import torch

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")

from src.algorithms.optimizers import get_optimizer
from src.algorithms.optimizers import get_lr_scheduler
from src.criterions import get_criterion

from src.networks.models import get_model
from src.utils.config import parse_config
from src.utils.serialize_utils import flatten_dict, torch_safe_load

try:
    from apex import amp
except ImportError:
    print('failed to import apex')


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class EngineBase(object):
    def __init__(self, device='cuda', partition_train_distill=-1.):
        self.device = device
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.lr_scheduler = None
        self.evaluator = None

        self.config = None
        self.logger = None

        self.metadata = {}

        self.partition_train_distill = partition_train_distill

    def create(self, config, word2idx, evaluator, mlp_local):
        self.config = config
        self.word2idx = word2idx
        self.model = get_model(word2idx,
                               config.model, mlp_local)
        self.set_criterion(get_criterion(config.criterion.name,
                                         config.criterion))
        params = [param for param in self.model.parameters()
                  if param.requires_grad]
        params += [param for param in self.criterion.parameters()
                   if param.requires_grad]
        self.set_optimizer(get_optimizer(config.optimizer.name,
                                         params,
                                         config.optimizer))
        self.set_lr_scheduler(get_lr_scheduler(config.lr_scheduler.name,
                                               self.optimizer,
                                               config.lr_scheduler))
        evaluator.set_model(self.model)
        evaluator.set_criterion(self.criterion)
        self.set_evaluator(evaluator)
        if self.logger is not None:
            self.logger.log('Engine is created.')
            self.logger.log(config)

            self.logger.update_tracker({'full_config': munch.unmunchify(config)}, keys=['full_config'])

        self.prefix = 'train__'
        self.eval_prefix = ''
        if self.logger is not None:
            self.logger.log('start train')

        self.img_code, self.txt_code, self.mm_txt_code, self.mm_img_code = None, None, None, None

    def model_to_device(self):
        self.model.to(self.device)
        if self.criterion:
            self.criterion.to(self.device)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_criterion(self, criterion):
        self.criterion = criterion

    def set_lr_scheduler(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator
        self.evaluator.set_logger(self.logger)

    def set_logger(self, logger):
        self.logger = logger

    def to_half(self):
        # Mixed precision
        # https://nvidia.github.io/apex/amp.html
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                                    opt_level='O2')

    @torch.no_grad()
    def evaluate(self, val_loaders, n_crossfolds=None, **kwargs):
        if self.evaluator is None:
            if self.logger is not None:
                self.logger.log('[Evaluate] Warning, no evaluator is defined. Skip evaluation')
            return

        self.model_to_device()
        self.model.eval()

        if not isinstance(val_loaders, dict):
            val_loaders = {'te': val_loaders}

        scores = {}
        for key, data_loader in val_loaders.items():
            if key == "train" or key == "train_subset" or key == "train_subset_eval" or "train" in key:
                continue
            if self.logger is not None:
                self.logger.log('Evaluating {}...'.format(key))
            _n_crossfolds = -1 if key == 'val' else n_crossfolds
            scores[key] = self.evaluator.evaluate(data_loader, n_crossfolds=_n_crossfolds,
                                                  key=key, **kwargs)
        return scores

    def save_models(self, save_to, metadata=None):
        state_dict = {
            'model': self.model.state_dict(),
            'criterion': self.criterion.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'config': munch.unmunchify(self.config),
            'word2idx': self.word2idx,
            'metadata': metadata,
        }
        torch.save(state_dict, save_to)
        if self.logger is not None:
            self.logger.log('state dict is saved to {}, metadata: {}'.format(save_to, json.dumps(metadata, indent=4)))

    def load_models(self, state_dict_path, load_keys=None):
        with open(state_dict_path, 'rb') as fin:
            model_hash = hashlib.sha1(fin.read()).hexdigest()
            self.metadata['pretrain_hash'] = model_hash

        state_dict = torch.load(state_dict_path, map_location='cpu')

        if 'model' not in state_dict:
            torch_safe_load(self.model, state_dict, strict=False)
            return

        if not load_keys:
            load_keys = ['model', 'criterion', 'optimizer', 'lr_scheduler']
        for key in load_keys:
            try:
                torch_safe_load(getattr(self, key), state_dict[key])
            except RuntimeError as e:
                if self.logger is not None:
                    self.logger.log('Unable to import state_dict, missing keys are found. {}'.format(e))
                    torch_safe_load(getattr(self, key), state_dict[key], strict=False)
        if self.logger is not None:
            self.logger.log('state dict is loaded from {} (hash: {}), load_key ({})'.format(state_dict_path,
                                                                                            model_hash,
                                                                                            load_keys))

    def load_state_dict(self, state_dict_path, load_keys=None):
        state_dict = torch.load(state_dict_path)
        config = parse_config(state_dict['config'])
        self.create(config, state_dict['word2idx'])
        self.load_models(state_dict_path, load_keys)


class TrainerEngine(EngineBase):

    def train(self, tr_loader, pub_data_ratio=1.):

        self.model.train()

        torch.cuda.empty_cache()
        if self.logger is not None:
            self.logger.log("Global Training!")
        for idx, (images, captions, captions_word, caption_lens, a_, b_, index) in tqdm(enumerate(tr_loader), total=len(tr_loader)):
            images = images.to(self.device)  # [bs, 3, 224, 224]
            captions = captions.to(self.device)  # [bs, seq_len]
            caption_lens = caption_lens.to(self.device)

            if idx == int(len(tr_loader) * pub_data_ratio):
                break

            output = self.model(images, captions, captions_word, caption_lens)
            loss, _ = self.criterion(**output)

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

    def report_scores(self, step, scores, metadata, prefix=''):
        report_dict = {data_key: flatten_dict(_scores, sep='_')
                       for data_key, _scores in scores.items()}
        report_dict = flatten_dict(report_dict, sep='__')
        tracker_data = report_dict.copy()

        report_dict = {'{}{}'.format(prefix, key): val for key, val in report_dict.items()}
        report_dict['step'] = step
        if 'lr' in metadata:
            report_dict['{}lr'.format(prefix)] = metadata['lr']

        report_dict[
            'summary'] = f"{report_dict['__test__n_fold_i2t_recall_1']}, {report_dict['__test__n_fold_i2t_recall_5']}, {report_dict['__test__n_fold_i2t_recall_10']}, {report_dict['__test__n_fold_t2i_recall_1']}, {report_dict['__test__n_fold_t2i_recall_5']}, {report_dict['__test__n_fold_t2i_recall_10']}, {report_dict['__test__i2t_recall_1']}, {report_dict['__test__i2t_recall_5']}, {report_dict['__test__i2t_recall_10']}, {report_dict['__test__t2i_recall_1']}, {report_dict['__test__t2i_recall_5']}, {report_dict['__test__t2i_recall_10']}"
        if self.logger is not None:
            self.logger.report(report_dict,
                               prefix='[Eval] Report @step: ',
                               pretty=True)

        tracker_data['metadata'] = metadata
        tracker_data['scores'] = scores
        if self.logger is not None:
            self.logger.update_tracker(tracker_data)


class rawTrainerEngine(EngineBase):

    def _train_epoch(self, dataloader, cur_epoch, prefix='', pub_data_ratio=1.):
        self.model.train()
        torch.cuda.empty_cache()

        for idx, (images, captions, captions_word, caption_lens, _, _, index) in tqdm(enumerate(dataloader), total=len(dataloader)):
            images = images.to(self.device)  # [bs, 3, 224, 224]
            captions = captions.to(self.device)  # [bs, seq_len]
            caption_lens = caption_lens.to(self.device)

            if idx == int(len(dataloader) * pub_data_ratio):
                break

            output = self.model(images, captions, captions_word, caption_lens)
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

        loss_dict = {'{}{}'.format(prefix, key): val
                     for key, val in loss_dict.items()}

        def cur_step(cur_epoch, idx, N, fmt=None):
            _cur_step = cur_epoch + idx / N
            if fmt:
                return fmt.format(_cur_step)
            else:
                return _cur_step

        loss_dict['step'] = cur_step(cur_epoch, idx, len(dataloader))
        if self.logger is not None:
            self.logger.report(loss_dict, prefix='[Train] Report @step: ')

    def train(self, tr_loader, n_epochs,
              val_loaders=None,
              val_epochs=1,
              model_save_to='last.pth',
              best_model_save_to='best.pth', pub_data_ratio=1.):
        self.img_code, self.txt_code, self.mm_txt_code, self.mm_img_code = None, None, None, None
        if val_loaders and 'val' not in val_loaders:
            raise KeyError('val_loaders should contain key "val", '
                           'but ({})'.format(val_loaders.keys()))

        dt = datetime.datetime.now()

        prefix = 'train__'
        eval_prefix = ''
        if self.logger is not None:
            self.logger.log('start train')

        self.model_to_device()
        if self.config.train.get('use_fp16'):
            if self.logger is not None:
                self.logger.log('Train with half precision')
            self.to_half()

        best_score = 0
        for cur_epoch in range(n_epochs):
            self._train_epoch(tr_loader, cur_epoch, prefix=prefix,
                              pub_data_ratio=pub_data_ratio)

            metadata = self.metadata.copy()
            metadata['cur_epoch'] = cur_epoch + 1
            metadata['lr'] = get_lr(self.optimizer)

            if val_loaders is not None and ((cur_epoch + 1) % val_epochs == 0 or cur_epoch == 0):
                scores = self.evaluate(val_loaders)
                metadata['scores'] = scores['val']

                if best_score < scores['val']['rsum']:
                    self.save_models(best_model_save_to, metadata)
                    best_score = scores['val']['rsum']
                    metadata['best_score'] = best_score
                    metadata['best_epoch'] = cur_epoch + 1

                self.report_scores(step=cur_epoch + 1,
                                   scores=scores,
                                   metadata=metadata,
                                   prefix=eval_prefix)

            if self.config.lr_scheduler.name == 'reduce_lr_on_plateau':
                self.lr_scheduler.step(scores['val']['rsum'])
            else:
                self.lr_scheduler.step()

            self.save_models(model_save_to, metadata)

            elasped = datetime.datetime.now() - dt
            expected_total = elasped / (cur_epoch + 1) * n_epochs
            expected_remain = expected_total - elasped
            if self.logger is not None:
                self.logger.log('expected remain {}'.format(expected_remain))
        if self.logger is not None:
            self.logger.log('finish train, takes {}'.format(datetime.datetime.now() - dt))

    def report_scores(self, step, scores, metadata, prefix=''):
        report_dict = {data_key: flatten_dict(_scores, sep='_')
                       for data_key, _scores in scores.items()}
        report_dict = flatten_dict(report_dict, sep='__')
        tracker_data = report_dict.copy()

        report_dict = {'{}{}'.format(prefix, key): val for key, val in report_dict.items()}
        report_dict['step'] = step
        if 'lr' in metadata:
            report_dict['{}lr'.format(prefix)] = metadata['lr']
        if self.logger is not None:
            self.logger.report(report_dict, prefix='[Eval] Report @step: ', pretty=True)

        tracker_data['metadata'] = metadata
        tracker_data['scores'] = scores
        if self.logger is not None:
            self.logger.update_tracker(tracker_data)
