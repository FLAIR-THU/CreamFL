export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_CACHE="/shared/.cache/huggingface/datasets"

nohup python src/retri_client_txt.py --name retri_client_txt --server_lr 1e-5 --seed 0 --feature_dim 256 --pub_data_num 50000 --agg_method con_w --contrast_local_inter --contrast_local_intra --interintra_weight 0.5 --local_epochs 5 --client_num_per_round 1 --num_img_clients 0 --num_txt_clients 1 --num_mm_clients 0 > retri_client_txt.log 2>&1 &