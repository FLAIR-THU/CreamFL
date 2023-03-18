import copy
import operator

import torch.optim as optim
import torch.nn as nn
import torch.optim
import torch.utils.data

from apex import amp
from sklearn.metrics import pairwise_distances

from src import losses
from src.datasets.cifar import Cifar
from src.datasets.dataset_L import caption_collate_fn, Language
from src.networks.language_model import EncoderText
from src.networks.resnet_client import resnet18_client
from src.utils.Reader import ImageReader
from src.utils.Utils import to_one_hot

torch.backends.cudnn.enabled = True

import torchvision.transforms as transforms

from tqdm import tqdm

import numpy as np
import os
import random
import torch
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


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


def get_result_list(query_sorted_idx, gt_list, ignore_list, top_k):
    return_retrieval_list = []
    count = 0
    while len(return_retrieval_list) < top_k:
        query_idx = query_sorted_idx[count]
        if query_idx in ignore_list:
            pass
        else:
            if query_idx in gt_list:
                return_retrieval_list.append(1)
            else:
                return_retrieval_list.append(0)
        count += 1
    return return_retrieval_list


def recall_at_k(feature, query_id, retrieval_list, top_k):
    distance = pairwise_distances(feature, feature)
    result = 0
    for i in range(len(query_id)):
        query_distance = distance[query_id[i], :]
        gt_list = retrieval_list[i][0]
        ignore_list = retrieval_list[i][1]
        query_sorted_idx = np.argsort(query_distance)
        query_sorted_idx = query_sorted_idx.tolist()
        result_list = get_result_list(query_sorted_idx, gt_list, ignore_list, top_k)
        result += 1. if sum(result_list) > 0 else 0
    result = result / float(len(query_id))
    return result


gpuid = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    temp = target.view(1, -1).expand_as(pred)
    temp = temp.to(gpuid)
    correct = pred.eq(temp)

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# TODO: test
is_test = False


class ClientTrainer:
    def __init__(self, args, dataset, dst, RGBmean, RGBstdv, data_dict, logger, global_test_set, inter_distance=4, loss='softmax',
                 gpuid='cuda:0', num_epochs=30, init_lr=0.0001, decay=0.1, batch_size=512,
                 imgsize=256, num_workers=4, print_freq=10, save_step=10, scale=128, pool_type='max_avg', client_id=-1, wandb=None):
        seed_torch()
        self.args = args
        if dataset == 'Flickr30k':
            init_lr = 0.0002
        self.client_id = client_id
        self.dset_name = dataset

        self.dst = dst  # save dir
        self.gpuid = gpuid if torch.cuda.is_available() else 'cpu'

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.decay_time = [False, False]
        self.init_lr = init_lr
        self.decay_rate = decay
        self.num_epochs = num_epochs
        self.cur_epoch = -1

        self.data_dict = data_dict

        self.imgsize = imgsize
        self.RGBmean = RGBmean
        self.RGBstdv = RGBstdv

        self.record = []
        self.epoch = 0
        self.print_freq = print_freq
        self.save_step = save_step
        self.loss = loss
        self.losses = AverageMeter()
        self.top1, self.test_top1 = AverageMeter(), AverageMeter()
        self.top5, self.test_top5 = AverageMeter(), AverageMeter()

        # model parameter
        self.scale = scale
        self.pool_type = pool_type
        self.inter_distance = inter_distance
        if not self.setsys(): print('system error'); return

        self.logger = logger
        self.wandb = wandb

        self.loadData()
        self.setModel()

        self.old_model = None

        self.local_epochs = args.local_epochs
        self.local_epoch = 0

        self.global_test_set = global_test_set

    def run(self, global_img_feature, global_txt_feature, distill_index, global_train_loader):
        self.model.to(self.gpuid)
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()
        self.old_model.cuda()

        self.lr_scheduler(self.cur_epoch)

        for i in range(self.local_epochs):
            self.local_epoch += 1
            self.tra(global_img_feature, global_txt_feature, distill_index, global_train_loader)

        self.test()

        if self.args.save_client:
            torch.save(self.model.state_dict(), f'./saved_clients/{self.dset_name}/Client{self.client_id}-model_{self.local_epoch}.pth')

        self.model.cpu()
        self.old_model.cpu()

        del self.old_model
        import gc
        gc.collect()

    ##################################################
    # step 0: System check and predefine function
    ##################################################
    def setsys(self):
        if not torch.cuda.is_available(): print('No GPU detected'); return False
        if not os.path.exists(self.dst): os.makedirs(self.dst)
        return True

    ##################################################
    # step 1: Loading Data
    ##################################################
    def loadData(self):

        self.data_transforms = transforms.Compose([transforms.Resize(int(self.imgsize * 1.1)),
                                                   transforms.RandomRotation(10),
                                                   transforms.RandomCrop(self.imgsize),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(self.RGBmean, self.RGBstdv)])
        if self.dset_name == 'Cifar100':
            dsets_test = Cifar(name=self.dset_name, train=False, transform=self.data_transforms)
            self.dsets_test = torch.utils.data.DataLoader(dsets_test, batch_size=self.batch_size, shuffle=False,
                                                          num_workers=self.num_workers)
            self.classSize = 100
            self.class_label_coco = torch.Tensor(np.array(range(80)))
        elif self.dset_name == 'Cifar10':
            dsets_test = Cifar(name=self.dset_name, train=False, transform=self.data_transforms)
            self.dsets_test = torch.utils.data.DataLoader(dsets_test, batch_size=self.batch_size, shuffle=False,
                                                          num_workers=self.num_workers)
            self.classSize = 10
            self.class_label_coco = torch.Tensor(np.array(range(80)))
        elif self.dset_name == 'AG_NEWS':
            dsets_test = Language(name=self.dset_name, train=False)
            self.dsets_test = torch.utils.data.DataLoader(dsets_test, batch_size=self.batch_size, shuffle=False,
                                                          num_workers=self.num_workers, pin_memory=True, sampler=None,
                                                          drop_last=False, collate_fn=caption_collate_fn
                                                          )
            self.classSize = 4
        elif self.dset_name == "YelpReviewPolarity":
            dsets_test = Language(name=self.dset_name, train=False)
            self.dsets_test = torch.utils.data.DataLoader(dsets_test, batch_size=self.batch_size, shuffle=False,
                                                          num_workers=self.num_workers, pin_memory=True, sampler=None,
                                                          drop_last=False, collate_fn=caption_collate_fn
                                                          )
            self.classSize = 2
        else:
            self.dsets = ImageReader(self.data_dict, self.data_transforms)
            self.classSize = len(self.data_dict)
            assert False, 'Dataset Not Supported!'
        self.class_label = torch.Tensor(np.array(range(self.classSize)))
        print('output size: {}'.format(self.classSize))

        return

    ##################################################
    # step 2: Set Model
    ##################################################
    def setModel(self):
        if self.logger is not None:
            self.logger.log(f'Setting model {self.client_id}')
        if self.dset_name == 'Cifar100' or self.dset_name == 'Cifar10':
            self.model = resnet18_client(pretrained=True, num_class=self.classSize, pool_type=self.pool_type,
                                         is_train=True, scale=self.scale, mlp_local=self.args.mlp_local, embed_dim=self.args.feature_dim)
            self.criterion = losses.create(self.loss)
            params = self.model.parameters()
        elif self.dset_name == 'AG_NEWS' or self.dset_name == 'YelpReviewPolarity':
            self.model = EncoderText(embed_dim=self.args.feature_dim, num_class=self.classSize, scale=self.scale, mlp_local=self.args.mlp_local)
            self.criterion = losses.create(self.loss)
            params = self.model.parameters()
        self.center_criterion = nn.MSELoss()
        self.optimizer = optim.SGD(params, lr=self.init_lr,
                                   momentum=0.9, weight_decay=0.00005)
        return

    def lr_scheduler(self, epoch):
        if epoch >= 0.5 * self.num_epochs and not self.decay_time[0]:
            self.decay_time[0] = True
            lr = self.init_lr * self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        if epoch >= 0.8 * self.num_epochs and not self.decay_time[1]:
            self.decay_time[1] = True
            lr = self.init_lr * self.decay_rate * self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        return

    ##################################################
    # step 3: Learning
    ##################################################
    def tra(self, global_img_feature, global_txt_feature, distill_index, global_train_loader):
        def printnreset(name):
            self.logger.log('Epoch: [{0}] {1}\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                self.local_epoch, name, loss=self.losses, top1=self.top1, top5=self.top5))

            self.losses = AverageMeter()
            self.top1 = AverageMeter()
            self.top5 = AverageMeter()

        # Set model to training mode
        self.model.train()

        for i, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                center_labels_var = torch.autograd.Variable(self.class_label.to(torch.long)).to(self.gpuid)
                if self.dset_name == 'Cifar100' or self.dset_name == 'Cifar10':
                    inputs_bt, labels_bt = data  # <FloatTensor> <LongTensor>
                    # labels_bt = labels_bt.to(torch.long)
                    inputs_var = torch.autograd.Variable(inputs_bt).to(self.gpuid)
                    labels_var = torch.autograd.Variable(labels_bt).to(self.gpuid)

                    fvec, _, class_weight, _ = self.model(inputs_var)

                elif self.dset_name == 'AG_NEWS' or self.dset_name == 'YelpReviewPolarity':
                    inputs_bt, labels_bt, caplens = data
                    caplens = caplens.to(self.gpuid)

                    inputs_bt, labels_bt = map(lambda t: torch.cat(t) if type(t) != torch.Tensor else t,
                                               (inputs_bt, labels_bt))
                    inputs_bt, labels_var = map(lambda t: t.to(self.gpuid).contiguous(), (inputs_bt, labels_bt))

                    fvec, _, class_weight, _ = self.model(inputs_bt, caplens)

                # on_hot vector
                labels_var_one_hot = to_one_hot(labels_var, n_dims=self.classSize)
                # inter_class_distance
                fvec = fvec - self.inter_distance * labels_var_one_hot.to(self.gpuid)
                # intra_class_distance
                loss = self.criterion(fvec, labels_var)
                center_loss = self.criterion(torch.mm(class_weight, torch.t(class_weight)), center_labels_var)
                total_loss = 0.5 * center_loss + loss
                if self.dset_name == 'Cifar100' or self.dset_name == 'Cifar10':
                    prec1, prec5 = accuracy(fvec.data, labels_bt, topk=(1, 5))
                elif self.dset_name == 'AG_NEWS':
                    prec1, prec5 = accuracy(fvec.data, labels_bt, topk=(1, 4))
                elif self.dset_name == 'YelpReviewPolarity':
                    prec1, prec5 = accuracy(fvec.data, labels_bt, topk=(1, 2))
                self.top1.update(prec1[0], inputs_bt.size(0))
                self.top5.update(prec5[0], inputs_bt.size(0))

                self.losses.update(total_loss.item(), inputs_bt.size(0))
                total_loss.backward()
                self.optimizer.step()
            if is_test:
                break

        printnreset(self.dset_name)

        if self.args.contrast_local_intra and self.args.contrast_local_inter:
            global_img_feature, global_txt_feature = global_img_feature.cuda(), global_txt_feature.cuda()
            distill_dict = {b: a for a, b in enumerate(distill_index)}  # index in coco to index to list 'distill_index'
            self.old_model.phase = "extract_conv_feature"
            self.old_model.is_train = False
            self.model.phase = "extract_conv_feature"
            self.model.is_train = False
            self.logger.log("Start Intra & Inter Contrasting!")
            for idx, (images, captions, captions_word, caption_lens, _, _, index) in tqdm(enumerate(global_train_loader), total=len(global_train_loader)):
                self.optimizer.zero_grad()
                d_idx = operator.itemgetter(*index)(distill_dict)  # batchidx
                if self.dset_name == 'Cifar100' or self.dset_name == 'Cifar10':
                    images = images.to(self.gpuid)
                    im_feature = self.model(images)
                    target_feature = global_img_feature[d_idx, :].type_as(im_feature)
                    # neg
                    with torch.no_grad():
                        old_im_feature = self.old_model(images)

                    logits_inter = torch.div(torch.matmul(im_feature, global_txt_feature.T), 0.5)
                elif self.dset_name == 'AG_NEWS' or self.dset_name == 'YelpReviewPolarity':
                    captions = captions.to(self.gpuid)
                    caption_lens = caption_lens.to(self.gpuid)
                    im_feature = self.model(captions, caption_lens).squeeze()
                    target_feature = global_txt_feature[d_idx, :].type_as(im_feature)
                    # neg
                    with torch.no_grad():
                        old_im_feature = self.old_model(captions, caption_lens).squeeze()

                    logits_inter = torch.div(torch.matmul(im_feature, global_img_feature.T), 0.5)

                labels_inter = torch.tensor(d_idx).cuda()
                loss_inter = self.criterion(logits_inter, labels_inter)

                # pos
                pos = torch.sum(im_feature * target_feature, dim=-1)
                pos = pos.reshape(-1, 1)
                # neg
                # neg = cos(im_feature, old_im_feature)
                neg = torch.sum(im_feature * old_im_feature, dim=-1)
                logits = torch.cat((pos, neg.reshape(-1, 1)), dim=1)

                logits /= 0.5  # temperature
                labels = torch.zeros(images.size(0)).cuda().long()

                loss_moon = self.criterion(logits, labels)

                if not self.args.loss_scale:
                    loss = (loss_moon + loss_inter) * self.args.interintra_weight
                else:
                    loss = (loss_moon + loss_inter / (loss_inter / loss_moon).detach()) * self.args.interintra_weight
                loss.backward()
                self.optimizer.step()

                if is_test:
                    break

            self.old_model.phase = "None"
            self.old_model.is_train = True
            self.model.phase = "None"
            self.model.is_train = True

        elif self.args.contrast_local_intra:
            global_img_feature, global_txt_feature = global_img_feature.cuda(), global_txt_feature.cuda()
            distill_dict = {b: a for a, b in enumerate(distill_index)}  # index in coco to index to list 'distill_index'
            self.old_model.phase = "extract_conv_feature"
            self.old_model.is_train = False
            self.model.phase = "extract_conv_feature"
            self.model.is_train = False
            self.logger.log("Start Intra-modal Contrasting!")
            for idx, (images, captions, captions_word, caption_lens, _, _, index) in tqdm(enumerate(global_train_loader), total=len(global_train_loader)):
                self.optimizer.zero_grad()
                d_idx = operator.itemgetter(*index)(distill_dict)  # batchidx
                if self.dset_name == 'Cifar100' or self.dset_name == 'Cifar10':
                    images = images.to(self.gpuid)
                    im_feature = self.model(images)
                    target_feature = global_img_feature[d_idx, :].type_as(im_feature)
                    # neg
                    with torch.no_grad():
                        old_im_feature = self.old_model(images)
                elif self.dset_name == 'AG_NEWS' or self.dset_name == 'YelpReviewPolarity':
                    captions = captions.to(self.gpuid)
                    caption_lens = caption_lens.to(self.gpuid)
                    im_feature = self.model(captions, caption_lens).squeeze()
                    target_feature = global_txt_feature[d_idx, :].type_as(im_feature)
                    # neg
                    with torch.no_grad():
                        old_im_feature = self.old_model(captions, caption_lens).squeeze()
                # pos
                pos = torch.sum(im_feature * target_feature, dim=-1)
                pos = pos.reshape(-1, 1)
                # neg
                # neg = cos(im_feature, old_im_feature)
                neg = torch.sum(im_feature * old_im_feature, dim=-1)
                logits = torch.cat((pos, neg.reshape(-1, 1)), dim=1)

                logits /= 0.5  # temperature
                labels = torch.zeros(images.size(0)).cuda().long()

                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                if is_test:
                    break

            self.old_model.phase = "None"
            self.old_model.is_train = True
            self.model.phase = "None"
            self.model.is_train = True

        elif self.args.contrast_local_inter:
            global_img_feature, global_txt_feature = global_img_feature.cuda(), global_txt_feature.cuda()
            distill_dict = {b: a for a, b in enumerate(distill_index)}  # index in coco to index to list 'distill_index'
            self.model.phase = "extract_conv_feature"
            self.model.is_train = False
            # Contrast
            self.logger.log("Start Inter-modal Contrasting!")
            for idx, (images, captions, captions_word, caption_lens, _, _, index) in tqdm(enumerate(global_train_loader), total=len(global_train_loader)):
                self.optimizer.zero_grad()
                d_idx = operator.itemgetter(*index)(distill_dict)  # batchidx
                if self.dset_name == 'Cifar100' or self.dset_name == 'Cifar10':
                    images = images.to(self.gpuid)
                    im_feature = self.model(images)
                    logits = torch.div(torch.matmul(im_feature, global_txt_feature.T), 0.5)
                elif self.dset_name == 'AG_NEWS' or self.dset_name == 'YelpReviewPolarity':
                    captions = captions.to(self.gpuid)
                    caption_lens = caption_lens.to(self.gpuid)
                    im_feature = self.model(captions, caption_lens).squeeze()
                    logits = torch.div(torch.matmul(im_feature, global_img_feature.T), 0.5)

                labels = torch.tensor(d_idx).cuda()

                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                if is_test:
                    break

            self.model.phase = "None"
            self.model.is_train = True

    def test(self):
        def printnreset(name):
            self.logger.log('TTTEST:  Epoch: [{0}] {1}\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                self.local_epoch, name, top1=self.test_top1, top5=self.test_top5))

            self.losses = AverageMeter()
            self.test_top1 = AverageMeter()
            self.test_top5 = AverageMeter()

        self.model.eval()

        with torch.no_grad():
            for i, data in enumerate(self.global_test_set):
                if self.dset_name == 'Cifar100' or self.dset_name == 'Cifar10':
                    inputs_bt, labels_bt = data  # <FloatTensor> <LongTensor>
                    # labels_bt = labels_bt.to(torch.long)
                    inputs_var = torch.autograd.Variable(inputs_bt).to(self.gpuid)
                    fvec, _, class_weight, _ = self.model(inputs_var)
                elif self.dset_name == 'AG_NEWS' or self.dset_name == 'YelpReviewPolarity':
                    inputs_bt, labels_bt, caplens = data
                    inputs_bt = inputs_bt.to(self.gpuid)
                    caplens = caplens.to(self.gpuid)
                    fvec, _, class_weight, _ = self.model(inputs_bt, caplens)

                # # on_hot vector
                # labels_var_one_hot = to_one_hot(labels_var, n_dims=self.classSize)
                # # inter_class_distance
                # fvec = fvec - self.inter_distance * labels_var_one_hot.to(self.gpuid)
                if self.dset_name == 'Cifar100' or self.dset_name == 'Cifar10':
                    prec1, prec5 = accuracy(fvec.data, labels_bt, topk=(1, 5))
                elif self.dset_name == 'AG_NEWS':
                    prec1, prec5 = accuracy(fvec.data, labels_bt, topk=(1, 4))
                elif self.dset_name == 'YelpReviewPolarity':
                    prec1, prec5 = accuracy(fvec.data, labels_bt, topk=(1, 2))
                self.test_top1.update(prec1[0], inputs_bt.size(0))
                self.test_top5.update(prec5[0], inputs_bt.size(0))

        printnreset(self.dset_name)
        self.model.train()

    # def test(self):
    #     self.model.eval()
    #     features, labels = self.extract_conv_feature(self.dsets_test)
    #
    #     def get_info_by_label(label):
    #         query_id = []
    #         unique_list = list(set(list(label)))
    #         retrieval_list = []
    #         label_dict = {}
    #         for i in unique_list:
    #             if i in label_dict:
    #                 continue
    #             else:
    #                 label_dict.update({i: [idx for idx, x in enumerate(label) if x == i]})
    #         for key, value in enumerate(label):
    #             gt = label_dict[value]
    #             query_id.append(key)
    #             retrieval_list.append([gt, [key]])
    #         return query_id, retrieval_list
    #
    #     query_id, retrieval_list = get_info_by_label(labels.tolist())
    #     top_k = [1, 2, 4, 5, 8, 10, 16, 32]
    #     result = []
    #     for i in top_k:
    #         result.append(recall_at_k(features, query_id, retrieval_list, i))
    #     self.model.train()
    #     return result

    def extract_conv_feature(self, dset):
        self.model.phase = 'extract_conv_feature'
        self.model.is_train = False
        feature = []
        labels = []
        # iterate batch
        for i, data in enumerate(dset):
            with torch.no_grad():
                if self.dset_name == 'Cifar100' or self.dset_name == 'Cifar10':
                    inputs_bt, labels_bt = data  # <FloatTensor> <LongTensor>
                    inputs_var = torch.autograd.Variable(inputs_bt).to(self.gpuid)

                    im_feature = self.model(inputs_var)

                elif self.dset_name == 'AG_NEWS' or self.dset_name == 'YelpReviewPolarity':
                    input, target, caplens = data
                    caplens = caplens.to(self.gpuid)

                    input, labels_bt = map(lambda t: torch.cat(t) if type(t) != torch.Tensor else t,
                                           (input, target))
                    input = input.to(self.gpuid).contiguous()

                    im_feature = self.model(input, caplens).squeeze()

                labels_var = labels_bt.numpy().squeeze()
                labels.extend(labels_var)

                im_feature = im_feature.cpu().detach().numpy().reshape(-1)
                feature.extend(im_feature)
                # print(f'im_feature {im_feature.shape} labels {labels_var.shape}')
                # if is_test and i == 1:
                #     break

        feature = np.array(feature).reshape(-1, 512)
        labels = np.array(labels).reshape(-1)
        # print(f'feature {feature.shape} labels {labels.shape}')
        self.model.phase = 'None'
        self.model.is_train = True
        return feature, labels

    def generate_logits(self, dataloader):
        vec, idx = self.extract_pub_feature(dataloader)
        if self.dset_name == 'Cifar100' or self.dset_name == 'Cifar10':
            return {'img': vec, 'txt': None}, idx
        elif self.dset_name == 'AG_NEWS' or self.dset_name == 'YelpReviewPolarity':
            return {'img': None, 'txt': vec}, idx
        else:
            assert False

    def extract_pub_feature(self, dataloader):
        self.model.cuda()

        self.model.phase = 'extract_conv_feature'
        self.model.is_train = False
        feature = []
        distill_index = []
        # iterate batch
        for idx, (images, captions, captions_word, caption_lens, _, _, index) in tqdm(enumerate(dataloader),
                                                                       total=len(dataloader)):
            with torch.no_grad():
                if self.dset_name == 'Cifar100' or self.dset_name == 'Cifar10':
                    images = images.to(self.gpuid)
                    im_feature = self.model(images)

                elif self.dset_name == 'AG_NEWS' or self.dset_name == 'YelpReviewPolarity':
                    captions = captions.to(self.gpuid)
                    caption_lens = caption_lens.to(self.gpuid)
                    im_feature = self.model(captions, caption_lens).squeeze()

                im_feature = im_feature.cpu().detach()
                feature.append(im_feature)
                distill_index.extend(index)
                # print(f'im_feature {im_feature.shape} labels {labels_var.shape}')
                # if is_test and idx == 1:
                #     break

        feature = torch.cat(feature, dim=0)
        # print(f'feature {feature.shape} labels {labels.shape}')
        self.model.phase = 'None'
        self.model.is_train = True

        self.model.cpu()
        return feature, distill_index

    def to_half(self):
        # Mixed precision
        # https://nvidia.github.io/apex/amp.html
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                                    opt_level='O2')

    def __getattr__(self, k):
        if k.startwith("__"):
            raise AttributeError
