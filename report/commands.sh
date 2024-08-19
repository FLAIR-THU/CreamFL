
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

#test
git pull && python src/vqa.py --name test --server_lr 1e-5 --feature_dim 256 --pub_data_num 1000 --client_num_per_round 1 --num_img_clients 1 --num_txt_clients 1 --num_mm_clients 1 --local_epochs 1 --comm_rounds 5 --client_init_local_epochs 2


git pull && python src/vqa.py --name vqa_0c_100k --server_lr 1e-5 --seed 0 --feature_dim 1024 --pub_data_num 100000 --disable_distill --client_num_per_round 0 --num_img_clients 0 --num_txt_clients 0 --num_mm_clients 0

git pull && python src/main.py --name base_intra_full_clients --server_lr 1e-5 --agg_method con_w --contrast_local_inter --contrast_local_intra --interintra_weight 0.5 --seed 0 --feature_dim 1024
git pull && python src/main.py --name base_0_clients --server_lr 1e-5 --seed 0 --feature_dim 1024  --disable_distill --client_num_per_round 0 --num_img_clients 0 --num_txt_clients 0 --num_mm_clients 0
git pull && python src/main.py --name base_0c200k --server_lr 1e-5 --seed 0 --feature_dim 1024 --pub_data_num 200000  --disable_distill --client_num_per_round 0 --num_img_clients 0 --num_txt_clients 0 --num_mm_clients 0
git pull && python src/main.py --name base_0c800k --server_lr 1e-5 --pretrained_model base_0c400k_best_1024_net.pt --seed 0 --feature_dim 1024 --pub_data_num 800000  --disable_distill --client_num_per_round 0 --num_img_clients 0 --num_txt_clients 0 --num_mm_clients 0
git pull && python src/main.py --name full_200k --server_lr 1e-5 --pretrained_model base_0c800k_best_1024_net.pt --seed 0 --feature_dim 1024 --pub_data_num 200000 --agg_method con_w --contrast_local_inter --contrast_local_intra --interintra_weight 0.5
git pull && python src/main.py --name full_200k --server_lr 1e-5 --pretrained_model full_200k_best_1024_net.pt --seed 0 --feature_dim 1024 --pub_data_num 200000 --agg_method con_w --contrast_local_inter --contrast_local_intra --interintra_weight 0.5


git pull && python src/vqa.py --name vqa_0c1_pre400k --server_lr 1e-5 --pretrained_model vqa_0c1_pre400k_best_1024_vqa.pt --seed 0 --feature_dim 1024 --pub_data_num 1 --disable_distill --client_num_per_round 0 --num_img_clients 0 --num_txt_clients 0 --num_mm_clients 0
git pull && python src/vqa.py --name vqa_allc50k_nort --server_lr 1e-5 --pretrained_model vqa_0c1_pre400k_best_1024_vqa.pt --seed 0 --feature_dim 1024 --pub_data_num 50000 --agg_method con_w --contrast_local_inter --contrast_local_intra --interintra_weight 0.5 --no_retrieval_training
git pull && python src/vqa.py --name vqa_allc50k_nortv3 --server_lr 1e-5 --pretrained_model vqa_allc50k_nortv2_best_1024_vqa.pt --seed 0 --feature_dim 1024 --pub_data_num 100000 --agg_method con_w --contrast_local_inter --contrast_local_intra --interintra_weight 0.5 --no_retrieval_training --client_init_local_epochs 10 --client_num_per_round 5 --num_img_clients 2 --num_txt_clients 2 --num_mm_clients 3
git pull && python src/vqa.py --name vqa_2c50k_nort --server_lr 1e-5 --pretrained_model vqa_allc50k_nortv3_best_1024_vqa.pt --seed 0 --feature_dim 1024 --pub_data_num 50000 --agg_method con_w --contrast_local_inter --contrast_local_intra --interintra_weight 0.5 --no_retrieval_training --client_init_local_epochs 5 --client_num_per_round 2 --num_img_clients 2 --num_txt_clients 2 --num_mm_clients 3
git pull && python src/vqa.py --name vqa_2cimg50k_nort --server_lr 1e-5 --pretrained_model vqa_2cimg50k_nort_best_1024_vqa.pt --seed 0 --feature_dim 1024 --pub_data_num 50000 --agg_method con_w --contrast_local_inter --contrast_local_intra --interintra_weight 0.5 --no_retrieval_training --client_init_local_epochs 5 --client_num_per_round 3 --num_img_clients 0 --num_txt_clients 0 --num_mm_clients 3 --comm_rounds 100
git pull && python src/vqa.py --name vqaall_allc50k --server_lr 1e-5 --pretrained_model vqa_allc50k_nortv2_best_1024_vqa.pt --seed 0 --feature_dim 1024 --vqa_data_size_per_epoch -1 --pub_data_num 50000 --agg_method con_w --contrast_local_inter --contrast_local_intra --interintra_weight 0.5 --client_init_local_epochs 10

git pull && python src/vqa.py --name vqa100kd01_2c50k --server_lr 1e-5 --pretrained_model vqaall_allc50k_best_1024_vqa.pt --seed 0 --feature_dim 1024 --vqa_data_size_per_epoch 100000 --pub_data_num 50000 --agg_method con_w --contrast_local_inter --contrast_local_intra --interintra_weight 0.5 --client_init_local_epochs 10 --local_epochs 10 --client_num_per_round 2 --comm_rounds 100 --vqa_dropout 0.1
git pull && python src/vqa.py --name vqaall_d01_allc50k_nort --server_lr 1e-5 --pretrained_model vqa100kd01_2c50k_last_1024_vqa.pt --seed 0 --feature_dim 1024 --vqa_data_size_per_epoch -1 --pub_data_num 50000 --agg_method con_w --contrast_local_inter --contrast_local_intra --interintra_weight 0.5 --client_init_local_epochs 20 --no_retrieval_training  --comm_roucnds 100 --vqa_dropout 0.1
git pull && python src/vqa.py --name vqaall_d05_0c_nort --server_lr 1e-5 --pretrained_model vqa100kd01_2c50k_last_1024_vqa.pt --seed 0 --feature_dim 1024 --vqa_data_size_per_epoch -1 --pub_data_num 50000 --no_retrieval_training  --comm_rounds 100 --vqa_dropout 0.5 --pub_data_num 1 --disable_distill --client_num_per_round 0 --num_img_clients 0 --num_txt_clients 0 --num_mm_clients 0
git pull && python src/vqa.py --name vqa100k_0c50k --server_lr 1e-5 --pretrained_model vqa100kd01_2c50k_last_1024_vqa.pt --seed 0 --feature_dim 1024 --vqa_data_size_per_epoch 100000 --pub_data_num 50000  --comm_rounds 100 --vqa_dropout 0 --disable_distill --client_num_per_round 0 --num_img_clients 0 --num_txt_clients 0 --num_mm_clients 0



git pull && python src/vqa.py --name fd1k_fte0_nort_0c --pretrained_model 0c1f_best_1024_vqa.pt --server_lr 1e-5 --seed 0 --feature_dim 1024 --pub_data_num 1 --vqa_full_training_epoch 0 --no_retrieval_training --disable_distill --client_num_per_round 0 --num_img_clients 0 --num_txt_clients 0 --num_mm_clients 0

git pull && python src/vqa.py --name vqa_0c50k_nort_d05 --server_lr 1e-5 --pretrained_model vqa_0c1_pre400k_best_1024_vqa.pt --seed 0 --feature_dim 1024 --pub_data_num 50000 --disable_distill --client_num_per_round 0 --num_img_clients 0 --num_txt_clients 0 --num_mm_clients 0 --no_retrieval_training --vqa_dropout 0.5 --comm_rounds 100
git pull && python src/vqa.py --name vqa_0c50k --server_lr 1e-5 --pretrained_model vqa_0c1_pre400k_best_1024_vqa.pt --seed 0 --feature_dim 1024 --pub_data_num 50000 --disable_distill --client_num_per_round 0 --num_img_clients 0 --num_txt_clients 0 --num_mm_clients 0 --comm_rounds 100
git pull && python src/vqa.py --name vqa100k_2c10t50k --server_lr 1e-5 --pretrained_model vqa_0c50k_best_1024_vqa.pt --seed 0 --feature_dim 1024 --vqa_data_size_per_epoch 100000 --pub_data_num 50000 --agg_method con_w --contrast_local_inter --contrast_local_intra --interintra_weight 0.5  --client_num_per_round 2 --num_img_clients 0 --num_txt_clients 10 --num_mm_clients 0 --comm_rounds 100 --client_init_local_epochs 10 --local_epochs 10 

git pull && python src/vqa.py --name vqa_cw1k_0c50k --vqa_cat_weight count+1000 --server_lr 1e-5 --pretrained_model vqa_0c50k_best_1024_vqa.pt --seed 0 --feature_dim 1024 --pub_data_num 50000 --disable_distill --client_num_per_round 0 --num_img_clients 0 --num_txt_clients 0 --num_mm_clients 0 --comm_rounds 100

git pull && python src/vqa.py --name vqa_full50k_nort --server_lr 1e-5 --seed 0 --feature_dim 1024 --pub_data_num 50000 --agg_method con_w --contrast_local_inter --contrast_local_intra --interintra_weight 0.5 --no_retrieval_training


git pull && python src/vqa.py --name f_all_0c_nort --server_lr 1e-5 --pretrained_model vqa100kd01_2c50k_last_1024_vqa.pt --seed 0 --feature_dim 1024 --vqa_filter_unknown --vqa_data_size_per_epoch -1 --pub_data_num 1 --no_retrieval_training --comm_rounds 10 --vqa_dropout 0 --disable_distill --client_num_per_round 0 --num_img_clients 0 --num_txt_clients 0 --num_mm_clients 0


git pull && python src/vqa.py --name f_100k_0c50k --server_lr 1e-5 --pretrained_model vqa100k_0c50k_best_1024_vqa.pt --seed 0 --feature_dim 1024 --vqa_filter_unknown --vqa_data_size_per_epoch 100000 --pub_data_num 50000  --comm_rounds 100 --vqa_dropout 0 --disable_distill --client_num_per_round 0 --num_img_clients 0 --num_txt_clients 0 --num_mm_clients 0
git pull && python src/vqa.py --name f_100k_0c_nort --server_lr 1e-5 --pretrained_model vqa100k_0c50k_best_1024_vqa.pt --seed 0 --feature_dim 1024 --vqa_filter_unknown --vqa_data_size_per_epoch 100000 --pub_data_num 1 --no_retrieval_training  --comm_rounds 100 --vqa_dropout 0 --disable_distill --client_num_per_round 0 --num_img_clients 0 --num_txt_clients 0 --num_mm_clients 0
git pull && python src/vqa.py --name f_100k_0c_nort --server_lr 1e-5 --pretrained_model f_100k_0c_nort_best_1024_vqa.pt --seed 0 --feature_dim 1024 --vqa_filter_unknown --vqa_data_size_per_epoch 100000 --pub_data_num 1 --no_retrieval_training  --comm_rounds 100 --vqa_dropout 0 --disable_distill --client_num_per_round 0 --num_img_clients 0 --num_txt_clients 0 --num_mm_clients 0


!git pull && python src/vqa.py --name f0_all_allc50k_nort --server_lr 1e-5 --seed 0 --feature_dim 1024 --vqa_data_size_per_epoch -1 --pub_data_num 50000 --agg_method con_w --contrast_local_inter --contrast_local_intra --interintra_weight 0.5 --no_retrieval_training  --comm_rounds 30
git pull && python src/vqa.py --name f0_all_allc50k_nort_run2 --server_lr 1e-5 --pretrained_model f0_all_allc50k_nort_best_1024_vqa.pt --client_init_local_epochs 5 --seed 0 --feature_dim 1024 --vqa_filter_unknown --vqa_data_size_per_epoch -1 --pub_data_num 50000 --agg_method con_w --contrast_local_inter --contrast_local_intra --interintra_weight 0.5 --no_retrieval_training  --comm_rounds 30

git pull && python src/vqa.py --name f0_all_0c_nort --server_lr 1e-5 --seed 0 --feature_dim 1024 --vqa_filter_unknown --vqa_data_size_per_epoch -1 --pub_data_num 1 --no_retrieval_training  --comm_rounds 30 --vqa_dropout 0 --disable_distill --client_num_per_round 0 --num_img_clients 0 --num_txt_clients 0 --num_mm_clients 0
git pull && python src/vqa.py --name f1_100k_0c_nort --server_lr 1e-5 --pretrained_model f0_all_0c_nort_best_1024_vqa.pt --seed 0 --feature_dim 1024 --vqa_filter_unknown --vqa_data_size_per_epoch 100000 --pub_data_num 1 --no_retrieval_training  --comm_rounds 300 --vqa_dropout 0 --disable_distill --client_num_per_round 0 --num_img_clients 0 --num_txt_clients 0 --num_mm_clients 0


git pull && python src/vqa.py --name f0_100k_0c50k --server_lr 1e-5 --seed 0 --feature_dim 1024 --vqa_filter_unknown --vqa_data_size_per_epoch 100000 --pub_data_num 50000  --comm_rounds 300 --vqa_dropout 0 --disable_distill --client_num_per_round 0 --num_img_clients 0 --num_txt_clients 0 --num_mm_clients 0
git pull && python src/vqa.py --name f2_100k_0c_nort_from50k --server_lr 1e-5 --pretrained_model f0_100k_0c50k_best_1024_vqa.pt --seed 0 --feature_dim 1024 --vqa_filter_unknown --vqa_data_size_per_epoch 100000 --pub_data_num 1 --no_retrieval_training --comm_rounds 300 --vqa_dropout 0 --disable_distill --client_num_per_round 0 --num_img_clients 0 --num_txt_clients 0 --num_mm_clients 0


git pull && python src/vqa.py --name f3all_0c_nort_pre400k --server_lr 1e-5 --pretrained_model base_0c400k_best_1024_net.pt --seed 0 --feature_dim 1024 --pub_data_num 1 --vqa_filter_unknown --vqa_data_size_per_epoch -1 --disable_distill --client_num_per_round 0 --num_img_clients 0 --num_txt_clients 0 --num_mm_clients 0 --vqa_full_training_epoch 10 --no_retrieval_training --comm_rounds 30

git pull && python src/vqa.py --name f3all_0c_nort_pre_raw --server_lr 1e-5 --seed 0 --feature_dim 1024 --pub_data_num 1 --vqa_filter_unknown --vqa_data_size_per_epoch -1 --disable_distill --client_num_per_round 0 --num_img_clients 0 --num_txt_clients 0 --num_mm_clients 0 --vqa_full_training_epoch 10 --no_retrieval_training --comm_rounds 30
git pull && python src/vqa.py --name f3all_allc_nort_pre_raw --server_lr 1e-5 --seed 0 --feature_dim 1024 --vqa_filter_unknown --vqa_data_size_per_epoch -1 --pub_data_num 50000 --agg_method con_w --contrast_local_inter --contrast_local_intra --interintra_weight 0.5 --vqa_full_training_epoch 10 --no_retrieval_training --comm_rounds 30
