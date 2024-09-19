"""library for multi-modal dataset loaders.

Acknowledgements:
`image_to_caption_collate_fn` is based on
https://github.com/yalesong/pvse/blob/master/data.py
"""
import os

import numpy as np

import torch
from torch.utils.data import DataLoader

try:
    from custom_datasets.coco import CocoCaptionsCap
    from custom_datasets.flickr30k import F30kCaptionsCap
    from custom_datasets.cub import CUBCaption, CUBSampler
    from custom_datasets.vocab import Vocabulary
    from custom_datasets._transforms import imagenet_transform, caption_transform
except:
    try:
        from coco import CocoCaptionsCap
        from flickr30k import F30kCaptionsCap
        from cub import CUBCaption, CUBSampler
        from vocab import Vocabulary
        from _transforms import imagenet_transform, caption_transform
    except:
        import sys
        sys.path.append("./")
        sys.path.append("../")
        sys.path.append("../../")
        sys.path.append("../../../")
        from src.custom_datasets.coco import CocoCaptionsCap
        from src.custom_datasets.flickr30k import F30kCaptionsCap
        from src.custom_datasets.cub import CUBCaption, CUBSampler
        from src.custom_datasets.vocab import Vocabulary
        from src.custom_datasets._transforms import imagenet_transform, caption_transform


def image_to_caption_collate_fn(data):
    """Build mini-batch tensors from a list of (image, sentence) tuples.
    Args:
      data: list of (image, sentence) tuple.
        - image: torch tensor of shape (3, 256, 256) or (?, 3, 256, 256).
        - sentence: torch tensor of shape (?); variable length.

    Returns:
      images: torch tensor of shape (batch_size, 3, 256, 256) or
              (batch_size, padded_length, 3, 256, 256).
      targets: torch tensor of shape (batch_size, padded_length).
      lengths: list; valid length for each padded sentence.
    """
    # Sort a data list by sentence length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, sentences, captions, ann_ids, image_ids, index = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merge sentences (convert tuple of 1D tensor to 2D tensor)
    cap_lengths = [len(cap) for cap in sentences]
    targets = torch.zeros(len(sentences), max(cap_lengths)).long()
    for i, cap in enumerate(sentences):
        end = cap_lengths[i]
        targets[i, :end] = cap[:end]

    cap_lengths = torch.Tensor(cap_lengths).long()
    # print('cap_lengths', type(cap_lengths))
    return images, targets, captions, cap_lengths, ann_ids, image_ids, index


def load_vocab(vocab_path):
    if isinstance(vocab_path, str):
        vocab = Vocabulary()
        vocab.load_from_pickle(vocab_path)
    else:
        vocab = vocab_path
    return vocab


def _get_cub_file_paths(dataset_name, dataset_root, caption_root):
    """Select proper train / val classes and omit id files.
    The split is based on CVPR'17 Zero-Shot Learning -- The Good, the Bad and the Ugly
    See more details in
    https://arxiv.org/abs/1703.04394
    https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly
    Args:
      dataset_name: name of dataset
        - cub_trainval{idx} (idx in [1, 2, 3]):
            3-fold validation splits to search hyperparameters.
            Each split contains 100 train classes / 50 validation classes.
        - cub:
            The final split used for the final benchmark.
            This split contains 150 train classes / 50 unseen test classes (not in trainval)
    """
    if dataset_name == 'cub_trainval1':
        train_classes = './custom_datasets/annotations/cub/trainclasses1.txt'
        val_classes = './custom_datasets/annotations/cub/valclasses1.txt'
        omit_ids = './custom_datasets/annotations/cub/seen_test_images.txt'
    elif dataset_name == 'cub_trainval2':
        train_classes = './custom_datasets/annotations/cub/trainclasses2.txt'
        val_classes = './custom_datasets/annotations/cub/valclasses2.txt'
        omit_ids = './custom_datasets/annotations/cub/seen_test_images.txt'
    elif dataset_name == 'cub_trainval3':
        train_classes = './custom_datasets/annotations/cub/trainclasses3.txt'
        val_classes = './custom_datasets/annotations/cub/valclasses3.txt'
        omit_ids = './custom_datasets/annotations/cub/seen_test_images.txt'
    elif dataset_name == 'cub':
        train_classes = './custom_datasets/annotations/cub/trainvalclasses.txt'
        val_classes = './custom_datasets/annotations/cub/testclasses.txt'
        omit_ids = './custom_datasets/annotations/cub/seen_test_images.txt'
    else:
        raise ValueError(f'Invalid dataset_name: {dataset_name}')

    image_root = os.path.join(dataset_root, 'images/')

    return train_classes, val_classes, omit_ids, image_root, caption_root


def _get_cub_loader(image_root, caption_root,
                    data_classes, vocab,
                    num_workers,
                    batch_size=64,
                    train=False,
                    omit_ids=None,
                    ids=None,
                    cutout_prob=0.0,
                    caption_drop_prob=0.0):
    cub_dataset = CUBCaption(image_root, caption_root,
                             data_classes,
                             imagenet_transform(random_erasing_prob=cutout_prob),
                             caption_transform(vocab, caption_drop_prob),
                             omit_ids=omit_ids,
                             ids=ids)
    if train:
        sampler = CUBSampler(cub_dataset, len(cub_dataset.target_classes))
        dataloader = DataLoader(cub_dataset, batch_sampler=sampler,
                                num_workers=num_workers,
                                collate_fn=image_to_caption_collate_fn,
                                pin_memory=True)
    else:
        dataloader = DataLoader(cub_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                collate_fn=image_to_caption_collate_fn,
                                pin_memory=True)
    print(f'Loading CUB Caption: n_images {cub_dataset.n_images} n_captions {len(cub_dataset.targets)}...')
    return dataloader


def prepare_cub_dataloaders(dataloader_config,
                            dataset_name,
                            dataset_root,
                            caption_root,
                            vocab_path='./vocabs/cub_vocab.pkl',
                            num_workers=12):
    """Prepare CUB Caption train / val / test dataloaders
    CUB Caption loader has a fixed batch size
    - train loader: # classes (trainval = 100, full = 150)
    - test loader: 64 (hard coded at L#203)
    Args:
        dataloader_config (dict): configuration file which should contain "batch_size"
        dataset_name (str): name of dataset
            - cub_trainval{idx} (idx in [1, 2, 3]):
                3-fold validation splits to search hyperparameters.
                Each split contains 100 train classes / 50 validation classes.
            - cub:
                The final split used for the final benchmark.
                This split contains 150 train classes / 50 unseen test classes (not in trainval)
        dataset_root (str): root of your CUB images (see README.md for detailed dataset hierarchy)
        caption_root (str): root of your CUB captions (see README.md for detailed dataset hierarchy)
        vocab_path (str, optional): path for vocab pickle file (default: ./vocabs/cub_vocab.pkl).
        num_workers (int, optional): num_workers for the dataloaders (default: 12)
    Returns:
        dataloaders (dict): keys = ["train", "val", "val_in"], values are the corresponding dataloaders.
        vocab (Vocabulary object): vocab object
    """
    vocab = load_vocab(vocab_path)
    train_classes, val_classes, omit_ids, image_root, caption_root = _get_cub_file_paths(
        dataset_name, dataset_root, caption_root)

    cutout_prob = dataloader_config.get('random_erasing_prob', 0.0)
    caption_drop_prob = dataloader_config.get('caption_drop_prob', 0.0)

    dataloaders = {}
    dataloaders['train'] = _get_cub_loader(
        image_root, caption_root,
        train_classes,
        vocab, num_workers,
        train=True,
        omit_ids=omit_ids,
        cutout_prob=cutout_prob,
        caption_drop_prob=caption_drop_prob,
    )

    dataloaders['val'] = _get_cub_loader(
        image_root, caption_root,
        val_classes,
        vocab, num_workers,
        train=False,
    )

    dataloaders['val_in'] = _get_cub_loader(
        image_root, caption_root,
        train_classes,
        vocab, num_workers,
        train=False,
        ids=omit_ids
    )

    return dataloaders, vocab


def _get_coco_loader(image_root,
                     annotation_path,
                     ids, vocab,
                     num_workers,
                     batch_size=64,
                     train=False,
                     extra_ids=None,
                     extra_annotation_path=None,
                     cutout_prob=0.0,
                     caption_drop_prob=0.0):
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
                                   target_transform=_caption_transform)

    dataloader = DataLoader(coco_dataset,
                            batch_size=batch_size,
                            shuffle=train,
                            num_workers=num_workers,
                            collate_fn=image_to_caption_collate_fn,
                            pin_memory=True)
    print(f'Loading COCO Caption: n_images {coco_dataset.n_images} n_captions {len(coco_dataset)}...')
    return dataloader


def _get_coco_file_paths(dataset_root):
    """Select proper train / val classes and omit id files.
    """
    train_ids = np.load('./custom_datasets/annotations/coco_train_ids.npy')
    train_extra_ids = np.load('./custom_datasets/annotations/coco_restval_ids.npy')
    val_ids = np.load('./custom_datasets/annotations/coco_dev_ids.npy')[:5000]
    te_ids = np.load('./custom_datasets/annotations/coco_test_ids.npy')

    image_root = os.path.join(dataset_root, 'allimages')
    train_ann = os.path.join(dataset_root, 'annotations/captions_train2014.json')
    val_ann = os.path.join(dataset_root, 'annotations/captions_val2014.json')

    return train_ids, train_extra_ids, val_ids, te_ids, image_root, train_ann, val_ann


def prepare_coco_dataloaders(dataloader_config,
                             dataset_root,
                             vocab_path='./vocabs/coco_vocab.pkl',
                             num_workers=12):
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

    dataloaders['train'] = _get_coco_loader(
        image_root, train_ann, train_ids, vocab,
        num_workers=num_workers, batch_size=batch_size,
        train=True,
        extra_annotation_path=val_ann,
        extra_ids=train_extra_ids,
        cutout_prob=tr_cutout_prob,
        caption_drop_prob=tr_caption_drop_prob,
    )

    dataloaders['train_eval'] = _get_coco_loader(
        image_root, train_ann, train_ids, vocab,
        num_workers=num_workers, batch_size=batch_size,
        train=False,
        extra_annotation_path=val_ann,
        extra_ids=train_extra_ids,
        cutout_prob=tr_cutout_prob,
        caption_drop_prob=tr_caption_drop_prob,
    )

    dataloaders['val'] = _get_coco_loader(
        image_root, val_ann, val_ids, vocab,
        num_workers=num_workers, batch_size=eval_batch_size,
        train=False,
    )

    dataloaders['te'] = _get_coco_loader(
        image_root, val_ann, te_ids, vocab,
        num_workers=num_workers, batch_size=eval_batch_size,
        train=False,
    )

    return dataloaders, vocab


def see_coco_len(dataset_root=os.environ['HOME'] + '/data/mmdata/MSCOCO/2014'):
    """Prepare MS-COCO Caption train / val / test dataloaders
    Args:
        dataloader_config (dict): configuration file which should contain "batch_size"
        dataset_root (str): root of your MS-COCO dataset (see README.md for detailed dataset hierarchy)
        vocab_path (str, optional): path for vocab pickle file (default: ./vocabs/coco_vocab.pkl).
        num_workers (int, optional): num_workers for the dataloaders (default: 6)
    Returns:
        dataloaders (dict): keys = ["train", "val", "te"], values are the corresponding dataloaders.
        vocab (Vocabulary object): vocab object
    """

    train_ids, train_extra_ids, val_ids, te_ids, image_root, train_ann, val_ann = _get_coco_file_paths(dataset_root)

    train = CocoCaptionsCap(image_root, train_ann,
                            extra_annFile=val_ann,
                            ids=train_ids,
                            extra_ids=train_extra_ids,
                            transform=None,
                            target_transform=None)

    test = CocoCaptionsCap(image_root, val_ann,
                           ids=te_ids,
                           transform=None,
                           target_transform=None)

    print(f'see_coco_len train {len(train)}')
    print(f'see_coco_len test {len(test)}')


def _get_F30k_loader(vocab,
                     num_workers,
                     max_size,
                     batch_size=64,
                     train=False,
                     split='train',
                     cutout_prob=0.0,
                     caption_drop_prob=0.0,
                     client=-1,
                     num_users=-1):
    _image_transform = imagenet_transform(
        random_resize_crop=train,
        random_erasing_prob=cutout_prob,
    )
    _caption_transform = caption_transform(vocab,
                                           caption_drop_prob)

    coco_dataset = F30kCaptionsCap(train=True if split == 'train' else False,
                                   transform=_image_transform,
                                   target_transform=_caption_transform, client=client, num_users=num_users, max_size=max_size)
    dataloader = DataLoader(coco_dataset,
                            batch_size=batch_size,
                            shuffle=train,
                            num_workers=num_workers,
                            collate_fn=image_to_caption_collate_fn,
                            pin_memory=False)
    print(f'Loading F30k Caption: split {split} n_images {coco_dataset.n_images} n_captions {len(coco_dataset)}...')
    return dataloader


def prepare_f30k_dataloaders(dataloader_config,
                             dataset_root,
                             max_size,
                             vocab_path='./vocabs/coco_vocab.pkl',
                             client=-1,
                             num_users=-1,
                             num_workers=12):
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

    dataloaders = {}
    dataloaders['train'] = _get_F30k_loader(
        vocab,
        num_workers,
        max_size,
        batch_size=batch_size,
        train=True,
        split='train',
        cutout_prob=tr_cutout_prob,
        caption_drop_prob=tr_caption_drop_prob,
        client=client,
        num_users=num_users,
    )

    # dataloaders['val'] = _get_F30k_loader(
    #     vocab,
    #     num_workers=num_workers,
    #     batch_size=eval_batch_size,
    #     train=False,
    #     split='val',
    #     client=client
    #
    # )

    dataloaders['te'] = _get_F30k_loader(
        vocab,
        num_workers,
        max_size,
        batch_size=eval_batch_size,
        train=False,
        split='test',
        client=client,
        num_users=num_users
    )

    return dataloaders, vocab


def see_f30k_len():
    train = F30kCaptionsCap(split='train')

    test = F30kCaptionsCap(split='test')

    print(f'see_f30k_len train {len(train)}')
    print(f'see_f30k_len test {len(test)}')


if __name__ == '__main__':
    see_coco_len()
    see_f30k_len()
