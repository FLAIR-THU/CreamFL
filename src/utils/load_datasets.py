import os
import random
import copy
import sys

import numpy as np
import pickle
from torch.utils.data import DataLoader

sys.path.append("./")
sys.path.append("../")
sys.path.append("../..")
sys.path.append("../../..")

from src.custom_datasets._dataloader import image_to_caption_collate_fn
from src.custom_datasets.coco import CocoCaptionsCap
from src.algorithms.vqa_meta import VQAMetaData, unknown_category_id


#  COCO
def prepare_coco_dataloaders(dataloader_config,
                             dataset_root,
                             subset_num, # was hard coded to 50000
                             client_subset_num, # was hard coded to 10000
                             vocab_path='./vocabs/coco_vocab.pkl',
                             num_workers=12, tsne=False, client=-1):
    """Prepare MS-COCO Caption train / val / test dataloaders
    Args:
        dataloader_config (dict): configuration file which should contain "batch_size"
        dataset_root (str): root of your MS-COCO dataset (see README.md for detailed dataset hierarchy)
        vocab_path (str, optional): path for vocab pickle file (default: ./vocabs/coco_vocab.pkl).
        num_workers (int, optional): num_workers for the dataloaders (default: 12)
    Returns:
        dataloaders (dict): keys = ["train", "val", "te"], values are the corresponding dataloaders.
        vocab (Vocabulary object): vocab object
    """
    batch_size = dataloader_config['batch_size']
    tr_cutout_prob = dataloader_config.get('random_erasing_prob', 0.0)
    tr_caption_drop_prob = dataloader_config.get('caption_drop_prob', 0.0)
    eval_batch_size = dataloader_config.get('eval_batch_size', batch_size)

    vocab = load_vocab(vocab_path)
    train_ids, train_extra_ids, val_ids, te_ids, image_root, train_ann, val_ann = _get_coco_file_paths(dataset_root)

    dataloaders = {}

    # dataloaders['train'] = _get_coco_loader(
    #     image_root, train_ann, train_ids, vocab,
    #     num_workers=num_workers, batch_size=batch_size,
    #     train=True,
    #     extra_annotation_path=val_ann,
    #     extra_ids=train_extra_ids,
    #     cutout_prob=tr_cutout_prob,
    #     caption_drop_prob=tr_caption_drop_prob,
    # )

    if tsne:
        pass
    elif client > -1:
        dataloaders['train_client'] = _get_coco_loader(
            image_root, train_ann, train_ids, vocab,
            num_workers=num_workers, batch_size=batch_size,
            train=True,
            extra_annotation_path=val_ann,
            extra_ids=train_extra_ids,
            cutout_prob=tr_cutout_prob,
            caption_drop_prob=tr_caption_drop_prob,
            subset=False,
            client=client,
            client_subset_num=client_subset_num
        )
    else:
        dataloaders[f'train_subset_{subset_num}'] = _get_coco_loader(
            image_root, train_ann, train_ids, vocab,
            num_workers=num_workers, batch_size=batch_size,
            train=True,
            extra_annotation_path=val_ann,
            extra_ids=train_extra_ids,
            cutout_prob=tr_cutout_prob,
            caption_drop_prob=tr_caption_drop_prob,
            subset=True,
            subset_num=subset_num
        )

        dataloaders[f'train_subset_eval_{subset_num}'] = _get_coco_loader(
            image_root, train_ann, train_ids, vocab,
            num_workers=num_workers, batch_size=batch_size * 2,
            train=False,
            extra_annotation_path=val_ann,
            extra_ids=train_extra_ids,
            cutout_prob=tr_cutout_prob,
            caption_drop_prob=tr_caption_drop_prob,
            subset=True,
            subset_num=subset_num
        )

    dataloaders['val'] = _get_coco_loader(
        image_root, val_ann, val_ids, vocab,
        num_workers=num_workers, batch_size=eval_batch_size,
        train=False,
        #subset=True, #AttributeError: 'Subset' object has no attribute 'n_images'
        #subset_num=subset_num
    )

    dataloaders['test'] = _get_coco_loader(
        image_root, val_ann, te_ids, vocab,
        num_workers=num_workers, batch_size=eval_batch_size if not tsne else 200,
        train=False,
        #subset=True, #AttributeError: 'Subset' object has no attribute 'n_images'
        #subset_num=subset_num
    )

    return dataloaders, vocab


def _get_coco_file_paths(dataset_root):
    """Select proper train / val classes and omit id files.
    """
    train_ids = np.load('./src/custom_datasets/annotations/coco_train_ids.npy')
    train_extra_ids = np.load('./src/custom_datasets/annotations/coco_restval_ids.npy')
    val_ids = np.load('./src/custom_datasets/annotations/coco_dev_ids.npy')[:5000]
    te_ids = np.load('./src/custom_datasets/annotations/coco_test_ids.npy')

    image_root = os.path.join(dataset_root, 'allimages')
    train_ann = os.path.join(dataset_root, 'annotations/captions_train2014.json')
    val_ann = os.path.join(dataset_root, 'annotations/captions_val2014.json')

    return train_ids, train_extra_ids, val_ids, te_ids, image_root, train_ann, val_ann


def _get_coco_loader(image_root,
                     annotation_path,
                     ids, vocab,
                     num_workers,
                     batch_size=64,
                     train=False,
                     extra_ids=None,
                     extra_annotation_path=None,
                     cutout_prob=0.0,
                     caption_drop_prob=0.0,
                     subset=False,
                     subset_num=50000,
                     client=-1,
                     client_subset_num=10000):
    _image_transform = imagenet_transform(
        random_resize_crop=train,
        random_erasing_prob=cutout_prob,
    )
    _caption_transform = caption_transform(vocab,
                                           caption_drop_prob)

    coco_dataset = CocoCaptionsCap(image_root, annotation_path,
                                   extra_annFile=extra_annotation_path,
                                   ids=ids,
                                   extra_ids=extra_ids,
                                   transform=_image_transform,
                                   target_transform=_caption_transform, client=client)

    if subset:
        full_size = 566435
        subset_num = min(subset_num, full_size)

        subset_fn = f'coco_subset_idx_{subset_num}'
        if not os.path.exists(subset_fn):
            full_idx = [i for i in range(full_size)]
            random.shuffle(full_idx)
            idx = full_idx[0: subset_num]
            idx.sort()
            if not os.path.exists(subset_fn):
                with open(subset_fn, 'wb') as f:
                    pickle.dump(idx, f)
        
        with open(subset_fn, 'rb') as f:
            idx = pickle.load(f)

        coco_dataset = torch.utils.data.Subset(coco_dataset, idx)

    elif client > -1:
        size_per_client = 10000 # 10000 is the old hard coded value
        size_per_client = min(subset_num,client_subset_num)
        range_start = 100000+client*size_per_client
        idx = [i for i in range(range_start, range_start + size_per_client)]
        coco_dataset = torch.utils.data.Subset(coco_dataset, idx)

    dataloader = DataLoader(coco_dataset,
                            batch_size=batch_size,
                            shuffle=train,
                            num_workers=num_workers,
                            collate_fn=image_to_caption_collate_fn,
                            pin_memory=True)
    if subset or client > -1:
        print(f'Loading COCO Caption: n_captions {len(coco_dataset)}...')
    else:
        print(f'Loading COCO Caption: n_images {coco_dataset.n_images} n_captions {len(coco_dataset)}...')
    return dataloader

def vqa2_dataloader(dataset,
                    num_workers=12,
                    batch_size=64,
                    cutout_prob=0.0,
                    train=False,
                    filter_unknown=False,
                    meta:VQAMetaData = None):
    transform = imagenet_transform(
        random_resize_crop=train,
        random_erasing_prob=cutout_prob,
        handle_gray=True,
    )
    if filter_unknown:
        def filter_fn(example):
            return meta.get_category_id(example['multiple_choice_answer']) != unknown_category_id
        dataset = dataset.filter(filter_fn)
    def collate_fn():
        def func(examples):
            batch = {}
            batch['image'] = torch.stack([transform(example['image']) for example in examples])
            batch['question'] = [example['question'] for example in examples]
            batch['question_type'] = [example['question_type'] for example in examples]
            batch['question_rest'] = [example['question'][len(example['question_type'])+1:] for example in examples]
            batch['multiple_choice_answer'] = [example['multiple_choice_answer'] for example in examples]
            batch['answers'] = [example['answers'] for example in examples]
            return batch
        return func
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=train,
                      num_workers=num_workers,
                      collate_fn=collate_fn(),
                      )
    

def load_vocab(vocab_path):
    if isinstance(vocab_path, str):
        vocab = Vocabulary()
        vocab.load_from_pickle(vocab_path)
    else:
        vocab = vocab_path
    return vocab


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.idx = 0
        self.word2idx = {}
        self.idx2word = {}

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def load_from_pickle(self, data_path):
        with open(data_path, 'rb') as fin:
            data = pickle.load(fin)
        self.idx = data['idx']
        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

from functools import partial

from nltk.tokenize import word_tokenize

import random
import math
import torch
from torchvision import transforms


def imagenet_normalize():
    """Standard ImageNet normalize transform
    """
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])


def imagenet_transform(resize_size=256,
                       crop_size=224,
                       random_resize_crop=False,
                       random_erasing_prob=0.0,
                       handle_gray=False,
                       custom_transforms=None):
    """Standard ImageNet transform with resize/crop/normalize.

    Args:
        resize_size (int, Default: 256): resize for validation
            (only used when random_resize_crop is False).
        crop_size (int, Default: 224): final crop size.
        random_resize_crop (bool, Default: False): if True, use random transform (for training),
            if False, use center crop (for validation).
        custom_transforms (list of transform, Default: None): additional transforms.
    """
    if custom_transforms is not None:
        if not isinstance(custom_transforms, list):
            raise TypeError(f'custom_transforms should be list, not {type(custom_transforms)}')
    transform = []
    if random_resize_crop:
        transform.append(transforms.RandomResizedCrop(crop_size))
        transform.append(transforms.RandomHorizontalFlip())
    else:
        transform.append(transforms.Resize(resize_size))
        transform.append(transforms.CenterCrop(crop_size))
    if handle_gray:
        transform.append(transforms.Lambda(
            lambda img: img.convert("RGB")),  # Convert grayscale to RGB
        )
    transform.append(transforms.ToTensor())
    transform.append(imagenet_normalize())

    if custom_transforms:
        transform.extend(custom_transforms)

    if random_erasing_prob > 0:
        print(f'adding cutout {random_erasing_prob}')
        transform.append(RandomErasing(random_erasing_prob,
                                       mode='const',
                                       max_count=1, num_splits=0, device='cpu'))

    transform = transforms.Compose(transform)
    return transform


def tokenize(sentence, vocab, caption_drop_prob):
    """nltk word_tokenize for caption transform.
    """
    tokens = word_tokenize(str(sentence).lower())
    tokenized_sentence = []
    tokenized_sentence.append(vocab('<start>'))
    tokenized = [vocab(token) for token in tokens]
    if caption_drop_prob > 0:
        unk = vocab('<unk>')
        tokenized = [vocab(token) if random.random() > caption_drop_prob else unk for token in tokens]
    else:
        tokenized = [vocab(token) for token in tokens]
    if caption_drop_prob:
        N = int(len(tokenized) * caption_drop_prob)
        for _ in range(N):
            tokenized.pop(random.randrange(len(tokenized)))
    tokenized_sentence.extend(tokenized)
    tokenized_sentence.append(vocab('<end>'))
    return torch.Tensor(tokenized_sentence)


def caption_transform(vocab, caption_drop_prob=0):
    """Transform for captions.
    "caption drop augmentation" randomly alters the given input tokens as <unk>
    """
    transform = []
    if caption_drop_prob < 0 or caption_drop_prob is None:
        print('warning: wrong caption drop prob', caption_drop_prob, 'set to zero')
        caption_drop_prob = 0
    elif caption_drop_prob > 0:
        print('adding caption drop prob', caption_drop_prob)
    transform.append(partial(tokenize, vocab=vocab, caption_drop_prob=caption_drop_prob))
    transform = transforms.Compose(transform)
    return transform


def _get_pixels(per_pixel, rand_color, patch_size, dtype=torch.float32, device='cuda'):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        return torch.empty((patch_size[0], 1, 1), dtype=dtype, device=device).normal_()
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)


class RandomErasing:
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf

        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """

    def __init__(
            self,
            probability=0.5, min_area=0.02, max_area=1 / 3, min_aspect=0.3, max_aspect=None,
            mode='const', min_count=1, max_count=None, num_splits=0, device='cuda'):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if mode == 'rand':
            self.rand_color = True  # per block random normal
        elif mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == 'const'
        self.device = device

    def _erase(self, img, chan, img_h, img_w, dtype):
        if random.random() > self.probability:
            return
        area = img_h * img_w
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, top:top + h, left:left + w] = _get_pixels(
                        self.per_pixel, self.rand_color, (chan, h, w),
                        dtype=dtype, device=self.device)
                    break

    def __call__(self, input):
        if len(input.size()) == 3:
            self._erase(input, *input.size(), input.dtype)
        else:
            batch_size, chan, img_h, img_w = input.size()
            # skip first slice of batch if num_splits is set (for clean portion of samples)
            batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
            for i in range(batch_start, batch_size):
                self._erase(input[i], chan, img_h, img_w, input.dtype)
        return input
