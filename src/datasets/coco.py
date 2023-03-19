"""MS-COCO image-to-caption retrieval dataset code

reference codes:
https://github.com/pytorch/vision/blob/v0.2.2_branch/torchvision/datasets/coco.py
https://github.com/yalesong/pvse/blob/master/data.py
"""

import os

import torch
from torchvision import datasets

from src.datasets.dataset_L import caption_transform
from src.datasets.vocab import Vocabulary

try:
    import ujson as json
except ImportError:
    import json

from PIL import Image
from pycocotools.coco import COCO

from torch.utils.data import Dataset
from glob import glob

import numpy as np


class CocoCaptionsCap(Dataset):
    """`MS Coco Captions <http://mscoco.org/dataset/#captions-challenge2015>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        ids (list, optional): list of target caption ids
        extra_annFile (string, optional): Path to extra json annotation file (for training)
        extra_ids (list, optional): list of extra target caption ids (for training)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        instance_annFile (str, optional): Path to instance annotation json (for PMRP computation)

    Example:
        .. code:: python
            import torchvision.datasets as dset
            import torchvision.transforms as transforms
            cap = dset.CocoCaptions(root='dir where images are',
                                    annFile='json annotation file',
                                    transform=transforms.ToTensor())
            print('Number of samples: ', len(cap))
            img, target = cap[3] # load 4th sample
            print("Image Size: ", img.size())
            print(target)
        Output: ::
            Number of samples: 82783
            Image Size: (3L, 427L, 640L)
            [u'A plane emitting smoke stream flying over a mountain.',
            u'A plane darts across a bright blue sky behind a mountain covered in snow',
            u'A plane leaves a contrail above the snowy mountain top.',
            u'A mountain that has a plane flying overheard in the distance.',
            u'A mountain view with a plume of smoke in the background']
    """
    def __init__(self, root, annFile, ids=None,
                 extra_annFile=None, extra_ids=None,
                 transform=None, target_transform=None,
                 instance_annFile=None, client=-1):
        self.root = os.path.expanduser(root)
        if extra_annFile:
            self.coco = COCO()
            with open(annFile, 'r') as fin1, open(extra_annFile, 'r') as fin2:
                dataset = json.load(fin1)
                extra_dataset = json.load(fin2)
                if not isinstance(dataset, dict) or not isinstance(extra_dataset, dict):
                    raise TypeError('invalid type {} {}'.format(type(dataset),
                                                                type(extra_dataset)))
                if set(dataset.keys()) != set(extra_dataset.keys()):
                    raise KeyError('key mismatch {} != {}'.format(list(dataset.keys()),
                                                                  list(extra_dataset.keys())))
                for key in ['images', 'annotations']:
                    dataset[key].extend(extra_dataset[key])
            self.coco.dataset = dataset
            self.coco.createIndex()
        else:
            self.coco = COCO(annFile)
        self.ids = list(self.coco.anns.keys()) if ids is None else list(ids)
        if extra_ids is not None:
            self.ids += list(extra_ids)
        self.ids = [int(id_) for id_ in self.ids]
        self.transform = transform
        self.target_transform = target_transform

        self.all_image_ids = set([self.coco.loadAnns(annotation_id)[0]['image_id'] for annotation_id in self.ids])

        iid_to_cls = {}
        if instance_annFile:
            for ins_file in glob(instance_annFile + '/instances_*'):
                with open(ins_file) as fin:
                    instance_ann = json.load(fin)
                for ann in instance_ann['annotations']:
                    image_id = int(ann['image_id'])
                    code = iid_to_cls.get(image_id, [0] * 90)
                    code[int(ann['category_id']) - 1] = 1
                    iid_to_cls[image_id] = code

                seen_classes = {}
                new_iid_to_cls = {}
                idx = 0
                for k, v in iid_to_cls.items():
                    v = ''.join([str(s) for s in v])
                    if v in seen_classes:
                        new_iid_to_cls[k] = seen_classes[v]
                    else:
                        new_iid_to_cls[k] = idx
                        seen_classes[v] = idx
                        idx += 1
                iid_to_cls = new_iid_to_cls

                if self.all_image_ids - set(iid_to_cls.keys()):
                    # print(f'Found mismatched! {self.all_image_ids - set(iid_to_cls.keys())}')
                    print(f'Found mismatched! {len(self.all_image_ids - set(iid_to_cls.keys()))}')

        self.iid_to_cls = iid_to_cls
        self.n_images = len(self.all_image_ids)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is a caption for the annotation.
        """
        coco = self.coco
        annotation_id = self.ids[index]
        annotation = coco.loadAnns(annotation_id)[0]
        image_id = annotation['image_id']
        caption = annotation['caption']  # language caption

        path = coco.loadImgs(image_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(caption)

        return img, target, caption, annotation_id, image_id, index

    def __len__(self):
        return len(self.ids)


class CocoImageRetrieval(datasets.coco.CocoDetection):
    def __init__(self, root=os.environ['HOME'] + "/data/mmdata/MSCOCO/2014/train2014",
                 annFile=os.environ['HOME'] + "/data/mmdata/MSCOCO/2014/annotations/instances_train2014.json",
                 transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        # print(self.cat2cat)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        # output = torch.zeros((3, 80), dtype=torch.long)
        output = torch.zeros(80, dtype=torch.long)
        for obj in target:
            # if obj['area'] < 32 * 32:
            #     output[0][self.cat2cat[obj['category_id']]] = 1
            # elif obj['area'] < 96 * 96:
            #     output[1][self.cat2cat[obj['category_id']]] = 1
            # else:
            #     output[2][self.cat2cat[obj['category_id']]] = 1
            output[self.cat2cat[obj['category_id']]] = 1
        target = output
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class CocoTextRetrieval(Dataset):
    def __init__(self, root=os.environ['HOME'] + "/data/mmdata/MSCOCO/2014/train2014",
                 annFile=os.environ['HOME'] + "/data/mmdata/MSCOCO/2014/annotations/captions_train2014.json",
                 insFile=os.environ['HOME'] + '/data/mmdata/MSCOCO/2014/annotations/instances_train2014.json',
                 transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.coco = COCO(annFile)
        self.coco_ins = COCO(insFile)

        self.ids = list(self.coco.anns.keys())
        self.target_transform = target_transform

        self.cat2cat = dict()
        for cat in self.coco_ins.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)

        if not transform:
            vocab_path = './src/datasets/vocabs/coco_vocab.pkl'
            if isinstance(vocab_path, str):
                vocab = Vocabulary()
                vocab.load_from_pickle(vocab_path)
            else:
                vocab = vocab_path
            transform = caption_transform(vocab, 0)

        self.transform = transform

    def __getitem__(self, index):
        annotation_id = self.ids[index]
        annotation = self.coco.loadAnns(annotation_id)[0]

        ann_ids = self.coco_ins.getAnnIds(imgIds=annotation['image_id'])
        target = self.coco_ins.loadAnns(ann_ids)

        output = np.zeros(80)
        for obj in target:
            output[self.cat2cat[obj['category_id']]] = 1
        target = output

        if self.transform is not None:
            annotation = self.transform(annotation['caption'])

        if self.target_transform is not None:
            target = self.target_transform(target)
        return annotation, target

    # annotation_id = self.ids[index]
    # annotation = coco.loadAnns(annotation_id)[0]
    # image_id = annotation['image_id']

    def __len__(self):
        return len(self.ids)
        # return 100
