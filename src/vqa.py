import os
import heapq
from tqdm import tqdm

from datasets import load_dataset, load_from_disk
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from src.networks.fusion_model import freeze_model, LinearFusionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device {device}")

text_retrieval_cache = {}
@torch.no_grad()
def get_text_features(engine, text):
    global text_retrieval_cache
    if text in text_retrieval_cache:
        return text_retrieval_cache[text]
    engine.eval()
    text_retrieval_cache[text] = engine.text_forward([],text,0)
    return text_retrieval_cache[text]

def get_matching_text(features, loss_fn=F.cosine_similarity, top_k=5):
    global text_retrieval_cache
    min_heap = []  # Using a min heap to keep track of top matches
    for text, text_features in text_retrieval_cache.items():
        match_score = loss_fn(features, text_features)
        heapq.heappush(min_heap, (-match_score, text))
        if len(min_heap) > top_k:
            heapq.heappop(min_heap)
    top_matches = [(-score, text) for score, text in sorted(min_heap, reverse=True)]
    return top_matches

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224), # make all images the same size
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
        transforms.ToTensor(),  # Converts image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizes tensor
    ])

def collate_fn(examples):
    batch = {}
    batch['image'] = torch.stack([transform(example['image']) for example in examples])
    batch['question'] = [example['question'] for example in examples]
    batch['multiple_choice_answer'] = [example['multiple_choice_answer'] for example in examples]
    return batch

if __name__ == "__main__":
    import common
    args, wandb = common.prepare_args(
        description="VQA for CreamFL Federated Learning (local simulation)",
        is_vqa=True)

    base_path = args.vqa_pretrained_base_model
    print(f"loading pretrained img txt model from {base_path}. Full path {os.path.abspath(base_path)}")
    from src.algorithms.retrieval_trainer import TrainerEngine
    retrieval_engine = TrainerEngine()
    print(f"  load COCOEvaluator")
    from src.algorithms.eval_coco import COCOEvaluator
    evaluator = COCOEvaluator(eval_method='matmul',
                            verbose=True,
                            eval_device='cuda',
                            n_crossfolds=5)
    print(f"  load models2")
    retrieval_engine.load_models2("./sl2-best_model.pt", evaluator)
    retrieval_engine.model_to_device()
    retrieval_model = retrieval_engine.model
    freeze_model(retrieval_model)

        
    if args.vqa_pretrained_eval:
        print(f"loading coco test set")
        dataset_root = os.environ['HOME'] + '/data/mmdata/MSCOCO/2014'
        vocab_path = './src/custom_datasets/vocabs/coco_vocab.pkl'
        from src.utils.config import parse_config
        config = parse_config("./src/coco.yaml", strict_cast=False)
        from src.utils.load_datasets import prepare_coco_dataloaders
        dataloaders, vocab = prepare_coco_dataloaders(
            config.dataloader, dataset_root, args.pub_data_num, args.max_size, vocab_path
        )
        test_dataloader = dataloaders['test']
        print(f"evaluate coco test set")
        test_scores = retrieval_engine.evaluate({'test': test_dataloader})
        print(f"test scores {test_scores}")
        
    
    print(f"loading vqa2 dataset")
    vqa2_train = load_dataset("HuggingFaceM4/VQAv2", split="train")
    #vqa2_train = vqa2_dataset["train"]
    # print(f"vqa2 train dataset size {len(vqa2_train)}")
    # print(f"vqa2 train dataset columns {vqa2_train.column_names}")
    # print(f"vqa2 train dataset features {vqa2_train.features}")
    # print(f"vqa2 train dataset example {vqa2_train[0]}")
    vqa2_dataloader = DataLoader(vqa2_train, batch_size=32, shuffle=True, collate_fn=collate_fn)

    
    print(f'init vqa fusion model "{args.vqa_fusion_network}"')
    fusion_model = None
    if args.vqa_fusion_network == "linear":
        fusion_model = LinearFusionModel(retrieval_model)
    else:
        print(f'vqa_fusion_network "{args.vqa_fusion_network}" is not supported')
    
    for epoch in range(1,6):
        print(f"epoch {epoch}")
        for i, batch in tqdm(enumerate(vqa2_dataloader), total=len(vqa2_dataloader)):
            images = batch['image'].to(device)
            questions = batch['question']
            answers = batch['multiple_choice_answer']
            outputs = fusion_model.forward(images, [], questions, 0)
            targets = [get_text_features(retrieval_engine, answer) for answer in answers]
            loss = F.cosine_similarity(outputs, targets)
            fusion_model.zero_grad()
            loss.backward()
            fusion_model.step()
            print(f'Loss: {loss.item()}')
            
    
    
    