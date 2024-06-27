
### server setup
conda activate creamfl
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_CACHE="/shared/.cache/huggingface/datasets"

cd /shared/project/xiegeo-dev/CreamFL

git pull && python src/vqa.py --seed 0 --vqa_hidden_sizes 2048 1024 1024 --vqa_unfreeze_base_epoch 10 --vqa_weight_decay 0.00001 --vqa_epochs 30

#vqa_pretrained_eval:
#test scores {'test': {'mean_log_image_sigma': 0.0, 'mean_log_caption_sigma': 0.0, 'n_fold': {'i2t': {'recall_1': 50.339999999999996, 'recall_5': 80.30000000000001, 'recall_10': 90.12, 'rsum': 220.76, 'medr': 1.4, 'meanr': 5.4586}, 't2i': {'recall_1': 38.620000000000005, 'recall_5': 75.02, 'recall_10': 87.124, 'rsum': 200.764, 'medr': 2.0, 'meanr': 7.05468}}, 'i2t': {'recall_1': 26.32, 'recall_5': 54.12, 'recall_10': 67.64, 'rsum': 148.07999999999998, 'medr': 5.0, 'meanr': 22.8744}, 't2i': {'recall_1': 18.348, 'recall_5': 44.296, 'recall_10': 58.544, 'rsum': 121.18799999999999, 'medr': 7.0, 'meanr': 30.7024}, 'rsum': 269.268, 'medr': 12.0, 'meanr': 53.576800000000006}}
