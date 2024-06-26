import os
import heapq
import pickle
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

from src.networks.fusion_model import freeze_model, LinearFusionModelCategorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device {device}")

text_retrieval_cache = {}
@torch.no_grad()
def get_text_features(engine, text):
    global text_retrieval_cache
    if text in text_retrieval_cache:
        return text_retrieval_cache[text]
    text_retrieval_cache[text] = engine.text_forward([],text,0)['embedding']
    return text_retrieval_cache[text]

@torch.no_grad()
def get_matching_text(features, top_k=5):
    global text_retrieval_cache
    min_heap = []  # Using a min heap to keep track of top matches
    for text, text_features in text_retrieval_cache.items():
        match_score = F.cosine_similarity(features, text_features).item()
        heapq.heappush(min_heap, (match_score, text))
        if len(min_heap) > top_k:
            heapq.heappop(min_heap) # remove the worst score
    top_matches = [(score, text) for score, text in sorted(min_heap, reverse=True)]
    return top_matches

unknown_category = "<unknown>"
unknown_category_id = 0

category_list = []
category_dict = {}
@torch.no_grad()
def get_category_id(cat, add_new=False):
    global category_list
    global category_dict
    if len(category_list) == 0:
        category_dict[unknown_category] = unknown_category_id
        category_list.append(unknown_category)
    if cat in category_dict:
        return category_dict[cat]
    if not add_new:
        return unknown_category_id
    category_dict[cat] = len(category_list)
    category_list.append(cat)
    return category_dict[cat]

@torch.no_grad()
def get_category_by_id(cat_id):
    global category_list
    return category_list[cat_id]

def set_category_from_dataset(dataset):
    #for item in tqdm(dataset.map(lambda example: {'multiple_choice_answer': example['multiple_choice_answer']})):
    #    get_category_id(item['multiple_choice_answer'])
    dataset = dataset.map(lambda example: {'multiple_choice_answer': example['multiple_choice_answer']})
    dataloader = DataLoader(dataset, batch_size=1024, num_workers=num_workers,
                            collate_fn=lambda examples: {'multiple_choice_answer': [example['multiple_choice_answer'] for example in examples]})
    set_category_from_dataloader(dataloader)
        
def set_category_from_dataloader(dataloader):
    for batch in tqdm(dataloader):
        for answer in batch['multiple_choice_answer']:
            get_category_id(answer, add_new=True)
            
def build_or_load_categories():
    global category_list
    global category_dict
    if len(category_list) != 0:
        raise Exception("categories already loaded")
    fn = "vqa2_categories_common.pkl"
    if os.path.exists(fn):
        with open(fn, "rb") as f:
            category_list = pickle.load(f)
            category_dict = {cat: i for i, cat in enumerate(category_list)}
        return
    # extract common categories from train and validation datasets
    set_category_from_dataset(load_dataset("HuggingFaceM4/VQAv2", split="train"))
    train_dict = category_dict
    category_list = []
    category_dict = {}
    set_category_from_dataset(load_dataset("HuggingFaceM4/VQAv2", split="validation"))
    validation_list = category_list
    category_list = []
    category_dict = {}
    print(f"train categories {len(train_dict)}, validation categories {len(validation_list)}")
    for cat in validation_list:
        if cat in train_dict:
            get_category_id(cat, add_new=True)
    print(f"common categories {len(category_list)}")
    with open(fn, "wb") as f:
        pickle.dump(category_list, f)

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
        
    num_workers = 4
    
    print(f"loading vqa2 dataset")
    vqa2_train = load_dataset("HuggingFaceM4/VQAv2", split="train")
    build_or_load_categories()
    print(f"category_list size:{len(category_list)}")
    vqa2_dataloader = DataLoader(vqa2_train, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)

    vqa2_test = load_dataset("HuggingFaceM4/VQAv2", split="validation[:1000]")
    vqa2_test_dataloader = DataLoader(vqa2_test, batch_size=1, collate_fn=collate_fn, num_workers=num_workers)
    
    print(f'init vqa fusion model "{args.vqa_fusion_network}"')
    fusion_model = None
    if args.vqa_fusion_network == "linear":
        fusion_model = LinearFusionModelCategorical(retrieval_model, len(category_list), hidden_sizes=args.vqa_hidden_sizes).to(device)
    else:
        print(f'vqa_fusion_network "{args.vqa_fusion_network}" is not supported')
        exit(1)
                
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=0.001)
    
    n = 0
    
    for epoch in range(1,6):
        print(f"epoch {epoch}")
        with tqdm(enumerate(vqa2_dataloader), total=len(vqa2_dataloader)) as progress_bar:
            for i, batch in progress_bar:            
                optimizer.zero_grad()
                images = batch['image'].to(device)
                questions = batch['question']
                answers = batch['multiple_choice_answer']
                outputs = fusion_model.forward(images, [], questions, 0)
                targets = torch.stack([get_text_features(retrieval_model, answer) for answer in answers], dim=0)
                # loss = 1 - F.cosine_similarity(outputs, targets).mean()
                loss = F.cross_entropy(outputs, torch.tensor([get_category_id(answer) for answer in answers]).to(device))
                loss.backward()
                optimizer.step()
                progress_bar.set_description(f"Epoch {epoch}, Iter {i}, Loss: {loss.item():.4f}")
                
                if (i+1+(epoch-1)*len(vqa2_dataloader)) % (128*2**n) == 0:
                    right = 0
                    unknown_outputs = 0
                    unknown_answers = 0
                    unknown_unknown = 0
                    total = 0
                    for j, testBatch in tqdm(enumerate(vqa2_test_dataloader)):
                        images = testBatch['image'].to(device)
                        questions = testBatch['question']
                        answers = testBatch['multiple_choice_answer']
                        total += len(answers)
                        outputs = fusion_model.forward(images, [], questions, 0)
                        for k, answer in enumerate(answers):
                            #top_matches = get_matching_text(outputs[k], top_k=5)
                            #if answer == top_matches[0][1]:
                            #    right += 1
                            top_matches = torch.argsort(outputs[k], descending=True)[:5]
                            top_match_names = [get_category_by_id(cat_id.item()) for cat_id in top_matches]
                            if top_match_names[0] == answer:
                                right += 1
                            if answer == unknown_category:
                                unknown_answers += 1
                            if top_match_names[0] == unknown_category:
                                unknown_outputs += 1
                            if answer == unknown_category and top_match_names[0] == unknown_category:
                                unknown_unknown += 1
                            print(f"expected {answer}, got {top_match_names}")
                        if j >= n:
                            break
                    n += 1
                    accuracy = right / total
                    print(f"test accuracy {right}/{total}={accuracy}, unknown_answers:{unknown_answers}, unknown_outputs:{unknown_outputs}, unknown_unknown:{unknown_unknown}, at epoch {epoch}, iter {i}/{len(vqa2_dataloader)}, loss {loss.item()}")
            
    
    
    