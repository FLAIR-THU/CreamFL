import json
import pandas as pd
import pyarrow as pa
import random
import os

from tqdm import tqdm
from glob import glob
from collections import defaultdict
import pickle


def path2rest(path, iid2captions, iid2split):
    name = path.split("/")[-1]

    captions = iid2captions[name]
    # print(f'caps {len(captions)}')
    # assert False

    split = iid2split[name]

    return [path, captions, name, split]


def make_arrow(root, dataset_root):
    with open(f"{root}/karpathy/dataset_flickr30k.json", "r") as fp:
        captions = json.load(fp)

    captions = captions["images"]

    iid2captions = defaultdict(list)
    iid2split = dict()

    for cap in tqdm(captions):
        filename = cap["filename"]
        iid2split[filename] = cap["split"]
        for c in cap["sentences"]:
            iid2captions[filename].append(c["raw"])

    paths = list(glob(f"{root}/flickr30k-images/*.jpg"))
    random.shuffle(paths)
    caption_paths = [path for path in paths if path.split("/")[-1] in iid2captions]

    if len(paths) == len(caption_paths):
        print("all images have caption annotations")
    else:
        print("not all images have caption annotations")
    print(
        len(paths), len(caption_paths), len(iid2captions),
    )

    bs = [path2rest(path, iid2captions, iid2split) for path in tqdm(caption_paths)]

    train, val, test = [], [], []

    for b in bs:
        if b[-1] == 'train':
            for sen in b[1]:
                train.append([b[0], sen])
        elif b[-1] == 'val':
            for sen in b[1]:
                val.append([b[0], sen])
        elif b[-1] == 'test':
            for sen in b[1]:
                test.append([b[0], sen])
        else:
            assert False, 'split wrong'

    pickle.dump({'train': train, 'test': test, 'val': val},
                open(os.path.join(dataset_root, 'dataset_k_split.pkl'), 'wb'))


make_arrow('/data/mmdata/Flick30k', '/data/mmdata/Flick30k')
