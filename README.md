# Networking for CreamFL

## Tasks

* get code base running locally.
  * figure out how to run though the entire code quickly.
  * quick test run: `python src/main.py --name quick --contrast_local_inter --contrast_local_intra --interintra_weight 0.5 --max_size 64 --pub_data_num 2 --feature_dim 2 --num_img_clients 2 --num_txt_clients 2 --num_mm_clients 3 --client_num_per_round 2 --local_epochs 2 --comm_rounds 2 --not_bert`
    * `--contrast_local_inter --contrast_local_intra --interintra_weight 0.5` Cream options.
    * `--max_size` added by xiegeo, 0 or 10000 for old behavior, client training data count, per client.
    * `--pub_data_num` public training data size (default 50000), proportional to communication cost (memory for local simulation) cost.
    * `--feature_dim` number of public features (default 256), proportional to communication cost.
    * `--num_img_clients 2 --num_txt_clients 2 --num_mm_clients 3 --client_num_per_round 2` number of max client of each type, and total number of client per round.
    * `--local_epochs 2 --comm_rounds 2` local and global rounds.
    * `--not_bert` use a simpler model

* get code to run in a network
  * see the "How to run the network" section.

## Goals

* Learn:
  * Transformers
    * Transformer [Attention Is All You Need 2017 v7(2023)](https://arxiv.org/abs/1706.03762)
    * [An Introduction to Transformers 2023 v5(2024)](https://arxiv.org/abs/2304.10557)
  * Multimodal
    * [DeViSE: A Deep Visual-Semantic Embedding Model 2013](https://research.google.com/pubs/archive/41473.pdf)
    * PCME [Probabilistic Embeddings for Cross-Modal Retrieval 2021 v2](https://arxiv.org/abs/2101.05068) <https://github.com/naver-ai/pcme>
  * Federated Learning
    * Federated Averaging [Communication-Efficient Learning of Deep Networks from Decentralized Data 2016 v4(2023)](https://arxiv.org/abs/1602.05629)

* Implement networking
  * try FedML? (to much rewrite for fedML to do it properly, otherwise too hacky.)
  * try custom network? (do a quick demo version)

## How to run the network

### Configuration

* flags: the same as local version.
* fed_config: setup server and client options.

### Run

A network requires n + 2 processes. Where n is the number of clients,
plus a command server over http, and a global round computation provider.

#### Command server

```bash
python src/federation/server.py --name test
```

#### Global round computation provider

```bash
python src/federation/global.py --name test --contrast_local_inter --contrast_local_intra --interintra_weight 0.5 --max_size 64 --pub_data_num 2 --feature_dim 2 --not_bert
```

#### Clients

Replace txt0 with the client to start.

```bash
python src/federation/client.py --name test --client_name txt_0 --max_size 64 --pub_data_num 2 --feature_dim 2 --not_bert
```

### File sharing

The network has to share the learned features. This could be through a file server,
a CDN, or shared network storage, ex.  Directly accessing the same files is the
easies to implement and easily extends to shared network storage, so this is implemented
first for ease of local testing without lose of generality.

## Prove of Concept

see [report/poc.pdf](report/poc.pdf)

------------------------
Begin original readme

# Multimodal Federated Learning via Contrastive Representation Ensemble

This repo contains a PyTorch implementation of the paper [Multimodal Federated Learning via Contrastive Representation Ensemble](https://arxiv.org/abs/2302.08888) (ICLR 2023). 

**<font color='red'>Note: This repository will be updated in the next few days for improved readability, easier environment setup, and datasets management.</font> Please stay tuned!**

![](imgs/method.png)

## Setup

### Environment

The required packages of the environment we used to conduct experiments are listed in `requirements.txt`.

Please note that you should install `apex` by following the instructions from https://github.com/NVIDIA/apex#installation, instead of directly running `pip install apex`.

### Datasets

For datasets, please download the MSCOCO, Flicker30K, CIFAR-100, and AG_NEWS datasets, and arrange their directories as follows:

```
os.environ['HOME'] + 'data/'
├── AG_NEWS
├── cifar100
│   └── cifar-100-python
├── flickr30k
│   └── flickr30k-images
├── mmdata
│   ├── MSCOCO
│   │   └── 2014
│   │       ├── allimages
│   │       ├── annotations
│   │       ├── train2014
│   │       └── val2014
```

## Usage

To reproduce CreamFL with BERT and ResNet101 as server models, run the following shell command:

```shell
python src/main.py --name CreamFL --server_lr 1e-5 --agg_method con_w --contrast_local_inter --contrast_local_intra --interintra_weight 0.5
```

## Run CreamFL retrieval parallely
[1] Run global server
```shell
bash retri_center.sh
```
[2] Run txt client
```shell
bash client_txt_retri.sh
```
[3] Run img client
```shell
bash client_img_retri.sh
```
[4] Run mm client
```shell
bash client_mm_retri.sh
```

## Citation

If you find the paper provides some insights into multimodal FL or our code useful 🤗, please consider citing:

```
@article{yu2023multimodal,
  title={Multimodal Federated Learning via Contrastive Representation Ensemble},
  author={Yu, Qiying and Liu, Yang and Wang, Yimu and Xu, Ke and Liu, Jingjing},
  journal={arXiv preprint arXiv:2302.08888},
  year={2023}
}
```

## Acknowledgements

We would like to thank for the code from [PCME](https://github.com/naver-ai/pcme) and [MOON](https://github.com/QinbinLi/MOON) repositories.
