
### server setup
conda activate creamfl
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_CACHE="/shared/.cache/huggingface/datasets"

cd /shared/project/xiegeo-dev/CreamFL

git pull && python src/vqa.py --seed 0 --vqa_hidden_sizes 256