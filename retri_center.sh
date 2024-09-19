export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_CACHE="/shared/.cache/huggingface/datasets"

nohup python src/retrivel.py --name retri_center --server_lr 1e-5 --seed 0 --feature_dim 256 --pub_data_num 50000 --agg_method con_w --contrast_local_inter --contrast_local_intra --interintra_weight 0.5 --local_epochs 5 --client_num_per_round 3 --num_img_clients 1 --num_txt_clients 1 --num_mm_clients 1 > retri_center.log 2>&1 &