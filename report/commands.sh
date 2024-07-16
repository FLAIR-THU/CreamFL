
### server setup
conda activate creamfl
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_CACHE="/shared/.cache/huggingface/datasets"

cd /shared/project/xiegeo-dev/CreamFL

git pull && python src/vqa_exp.py --seed 0 --vqa_hidden_sizes 1024 --vqa_unfreeze_base_epoch 25 --vqa_weight_decay 0.00001 --vqa_epochs 30 --batch_size 64


#vqa_pretrained_eval:
#test scores {'test': {
#   'mean_log_image_sigma': 0.0, 'mean_log_caption_sigma': 0.0, 'n_fold': {
#       'i2t': {
#           'recall_1': 50.339999999999996, 'recall_5': 80.30000000000001, 'recall_10': 90.12, 'rsum': 220.76, 'medr': 1.4, 'meanr': 5.4586
#       }, 't2i': {
#           'recall_1': 38.620000000000005, 'recall_5': 75.02, 'recall_10': 87.124, 'rsum': 200.764, 'medr': 2.0, 'meanr': 7.05468
#   }}, 'i2t': {
#       'recall_1': 26.32, 'recall_5': 54.12, 'recall_10': 67.64, 'rsum': 148.07999999999998, 'medr': 5.0, 'meanr': 22.8744
#   }, 't2i': {
#       'recall_1': 18.348, 'recall_5': 44.296, 'recall_10': 58.544, 'rsum': 121.18799999999999, 'medr': 7.0, 'meanr': 30.7024
#   }, 'rsum': 269.268, 'medr': 12.0, 'meanr': 53.576800000000006
#}}


git pull && python src/vqa.py --name vqa_0c_100k --seed 0 --feature_dim 1024 --pub_data_num 100000 --disable_distill --client_num_per_round 0 --num_img_clients 0 --num_txt_clients 0 --num_mm_clients 0

git pull && python src/main.py --name base_intra_full_clients --server_lr 1e-5 --agg_method con_w --contrast_local_inter --contrast_local_intra --interintra_weight 0.5 --seed 0 --feature_dim 1024
git pull && python src/main.py --name base_0_clients --server_lr 1e-5 --seed 0 --feature_dim 1024  --disable_distill --client_num_per_round 0 --num_img_clients 0 --num_txt_clients 0 --num_mm_clients 0
