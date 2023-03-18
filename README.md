# Multimodal Federated Learning via Contrastive Representation Ensemble

This repo contains a PyTorch implementation of the paper [Multimodal Federated Learning via Contrastive Representation Ensemble](https://arxiv.org/abs/2302.08888) (ICLR 2023). 

**<font color='red'>Note: This repository will be updated in the next few days for improved readability, easier environment setup, and datasets management.</font> Please stay tuned!**

![](imgs/method.png)

## Setup

The required packages are listed in `requirements.txt` .

### Datasets

For datasets, please first download the MSCOCO, CIFAR-10, CIFAR-100, and AG_NEWS datasets, and then arrange their directories as follows (create an empty directory for `yClient`):

```
os.environ['HOME'] + 'data/'
├── AG_NEWS
├── cifar100
│   └── cifar-100-python
├── flickr30k
│   └── flickr30k-images
├── mmdata
│   ├── Flick30k
│   │   ├── arrow
│   │   ├── data
│   │   │   ├── coco_precomp
│   │   │   └── f30k_precomp
│   │   ├── flickr30k-images
│   │   └── karpathy
│   ├── log
│   │   └── server
│   ├── MSCOCO
│   │   └── 2014
│   │       └── image-based
│   │           ├── allimages
│   │           ├── annotations
│   │           ├── arrow
│   │           ├── karpathy
│   │           └── val2014
└── yClient
    ├── AG_NEWS-1
    └── Cifar100
```

## Usage

Shell command for reproducing CreamFL with BERT and ResNet101 as server models:

```shell
# CreamFL
python src/main.py --name CreamFL --server_lr 1e-5 --agg_method con_w --contrast_local_inter --contrast_local_intra --interintra_weight 0.5
```

## Acknowledgement

Thanks for the code from [PCME](https://github.com/naver-ai/pcme) and [MOON](https://github.com/QinbinLi/MOON) repos.
