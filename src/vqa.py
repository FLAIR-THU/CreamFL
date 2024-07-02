import os
import random
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

from src.networks.fusion_model import LinearFusionModelCategorical, VQAFusionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device {device}")

use_f16 = False
if device == torch.device("cuda"):
    try:
        from apex import amp
        #print("enable f16 and using apex.amp for mixed precision training")
        #use_f16 = True
    except ImportError as e:
        print('failed to import apex:', e)

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
category_counts = []
@torch.no_grad()
def get_category_id(cat, add_new=False):
    global category_list
    global category_dict
    global category_counts
    add_count = add_new # add count only when we are building the list of categories
    if len(category_list) == 0:
        category_dict[unknown_category] = unknown_category_id
        category_list.append(unknown_category)
        category_counts.append(0)
    if cat in category_dict:
        cat_id = category_dict[cat]
        if add_count:
            category_counts[cat_id] += 1
        return cat_id
    if not add_new:
        return unknown_category_id
    category_dict[cat] = len(category_list)
    category_list.append(cat)
    category_counts.append(1)
    return category_dict[cat]

def reset_category_list():
    global category_list
    global category_dict
    global category_counts
    category_list = []
    category_dict = {}
    category_counts = []

@torch.no_grad()
def get_category_by_id(cat_id):
    global category_list
    return category_list[cat_id]

def set_category_from_dataset(dataset):
    #for item in tqdm(dataset.map(lambda example: {'multiple_choice_answer': example['multiple_choice_answer']})):
    #    get_category_id(item['multiple_choice_answer'])
    dataset = dataset.map(lambda example: {'multiple_choice_answer': example['multiple_choice_answer']})
    dataloader = DataLoader(dataset, batch_size=2048, num_workers=32,
                            collate_fn=lambda examples: {'multiple_choice_answer': [example['multiple_choice_answer'] for example in examples]})
    set_category_from_dataloader(dataloader)
        
def set_category_from_dataloader(dataloader):
    for batch in tqdm(dataloader):
        for answer in batch['multiple_choice_answer']:
            get_category_id(answer, add_new=True)
            
def build_or_load_categories_top(top = 3000):
    global category_list
    global category_dict
    global category_counts
    if len(category_list) != 0:
        raise Exception("categories already loaded")
    fn = f"vqa2_categories_{top}.pkl"
    if os.path.exists(fn):
        with open(fn, "rb") as f:
            data = pickle.load(f)
            category_list = data['category_list']
            category_counts = data['category_counts']
            category_dict = {cat: i for i, cat in enumerate(category_list)}
        return
    # extract common categories from train and validation datasets
    set_category_from_dataset(load_dataset("HuggingFaceM4/VQAv2", split="train"))
    cutoff = heapq.nlargest(top, category_counts)[-1]
    train_list = category_list
    train_counts = category_counts
    reset_category_list()
    for i, cat in enumerate(train_list):
        if train_counts[i] >= cutoff:
            id = get_category_id(cat, add_new=True)
            category_counts[id] = train_counts[i]
    assert len(category_list) == top + 1 # top categories + 1 unknown
    with open(fn, "wb") as f:
        data = {'category_list': category_list, 'category_counts': category_counts}
        pickle.dump(data, f)

    
def build_or_load_categories():
    global category_list
    global category_dict
    global category_counts
    if len(category_list) != 0:
        raise Exception("categories already loaded")
    fn = "vqa2_categories_common_count.pkl"
    if os.path.exists(fn):
        with open(fn, "rb") as f:
            data = pickle.load(f)
            category_list = data['category_list']
            category_counts = data['category_counts']
            category_dict = {cat: i for i, cat in enumerate(category_list)}
        return
    # extract common categories from train and validation datasets
    set_category_from_dataset(load_dataset("HuggingFaceM4/VQAv2", split="train"))
    train_list = category_list
    train_counts = category_counts
    reset_category_list()
    set_category_from_dataset(load_dataset("HuggingFaceM4/VQAv2", split="validation"))
    validation_dict = category_dict
    reset_category_list()
    print(f"train categories {len(train_list)}, validation categories {len(validation_dict)}")
    unknowns = 0
    for i, cat in enumerate(train_list):
        if cat in validation_dict:
            cat_id = get_category_id(cat, add_new=True)
            category_counts[cat_id] = train_counts[i]
        else:
            unknowns += train_counts[i]
    category_counts[0] = unknowns
    print(f"common categories {len(category_list)}")
    with open(fn, "wb") as f:
        data = {'category_list': category_list, 'category_counts': category_counts}
        pickle.dump(data, f)

transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224), # make all images the same size
        transforms.Lambda(lambda img: img.convert("RGB")),  # Convert grayscale to RGB
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizes the image
    ])

random_transform = transforms.Compose([
        transform,
        transforms.RandomErasing(p=0.5, value='random', scale=(0.05, 0.20), ratio=(0.3, 3.3)),
        transforms.RandomRotation(10),
        transforms.RandomErasing(p=1, scale=(0.05, 0.20), ratio=(0.3, 3.3)),
        transforms.RandomResizedCrop(224),
    ])

def split_image_into_4_parts(image):
    width, height = image.size
    mid_width, mid_height = width / 2, height / 2
    overlap_width = width * 0.1 / 2
    overlap_height = height * 0.1 / 2
    
    top_left = (0, 0, mid_width + overlap_width, mid_height + overlap_height)
    top_right = (mid_width - overlap_width, 0, width, mid_height + overlap_height)
    bottom_left = (0, mid_height - overlap_height, mid_width + overlap_width, height)
    bottom_right = (mid_width - overlap_width, mid_height - overlap_height, width, height)
    
    return [image.crop(box) for box in [top_left, top_right, bottom_left, bottom_right]]

def prepare_question(is_train,question):
    return question # disable question transformation
    # if is_train:
    #     words = question.split()
    #     duplicated = words + words
    #     for i in random.sample(range(len(duplicated)), random.randint(0, 1)):
    #         duplicated[i] = ""
    #     return " ".join(" ".join(duplicated).split())
    # return question + " " + question 

def collate_fn(is_train: bool):
    t = random_transform if is_train else transform
    
    def func(examples):
        batch = {}
        if 'image' in examples[0]:
            batch['image'] = torch.stack([t(example['image']) for example in examples])
            batch['sub_images'] = torch.stack([
                torch.stack([t(sub_image) 
                    for sub_image in split_image_into_4_parts(example['image'])])
                        for example in examples
            ])
        batch['question'] = [prepare_question(is_train,example['question']) for example in examples]
        batch['question_type'] = [example['question_type'] for example in examples]
        batch['question_rest'] = [example['question'][len(example['question_type'])+1:] for example in examples]
        batch['multiple_choice_answer'] = [example['multiple_choice_answer'] for example in examples]
        return batch
    return func

@torch.no_grad()
def process_retrieval_batch(batch):
    # Transform and move the batch of images to the device
    images = torch.stack([transform(image) for image in batch['image']]).to(device)
    questions = batch['question']
    
    # Forward pass with the batch of images and questions
    batch_output, _ = retrieval_model.forward(images, [], questions, 0)
    batch['image_features'] = batch_output['image_features']
    batch['caption_features'] = batch_output['caption_features']
        
    # Remove the 'image' column from the batch
    del batch['image']
    
    return batch

def validation(n, fusion_model, validation_dataloader):
    right = 0
    unknown_outputs = 0
    unknown_answers = 0
    unknown_unknown = 0
    total = 0
    for j, testBatch in tqdm(enumerate(validation_dataloader)):
        answers = testBatch['multiple_choice_answer']
        outputs, _ = fusion_model.forward(testBatch)
        for k, answer in enumerate(answers):
            answer_id = get_category_id(answer)
            #top_matches = get_matching_text(outputs[k], top_k=5)
            #if answer == top_matches[0][1]:
            #    right += 1
            _, top_matches = torch.topk(outputs[k], 5, largest=True, sorted=True)
            top_match_names = [get_category_by_id(cat_id.item()) for cat_id in top_matches]
            if top_match_names[0] == answer:
                right += 1
            if answer_id == unknown_category_id:
                unknown_answers += 1
                answer = unknown_category + answer # mark answers not in the training set
            if top_matches[0] == unknown_category_id:
                unknown_outputs += 1
            if answer_id == unknown_category_id and top_matches[0] == unknown_category_id:
                unknown_unknown += 1
            if total + k < 8:
                tqdm.write(f"j {j}, k {k}, expected {answer}, got {top_match_names}")
        total += len(answers)
        if total >= n:
            break
    accuracy = right / total
    tqdm.write(f"test accuracy {right}/{total}={accuracy}, unknown_answers:{unknown_answers}, unknown_outputs:{unknown_outputs}, unknown_unknown:{unknown_unknown}")

if __name__ == "__main__":
  #with torch.autocast(device_type=device.type):
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
    #if use_f16:
    #    retrieval_engine.to_half()
    retrieval_model = retrieval_engine.model
        
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
        
    num_workers = 16
    if use_f16:
        num_workers = 16 # f16 requires more workers to keep the GPU busy
    
    print(f"loading vqa2 categories and category weights")
    build_or_load_categories()
    print(f"  category_list size:{len(category_list)}")
    print(f"  category_list:{category_list[:10]}")
    print(f"  category_count:{category_counts[:10]}")
    
    print(f"loading vqa2 dataset")
    vqa2_train = load_dataset("HuggingFaceM4/VQAv2", split="train")
    # precalculate the forward pass on the base retrieval model
    # vqa2_train = vqa2_train.map(
    #     process_retrieval_batch,
    #     batched=True, batch_size=32,
    # )
    
    vqa2_dataloader = DataLoader(vqa2_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn(random_transform), num_workers=num_workers)

    vqa2_test = load_dataset("HuggingFaceM4/VQAv2", split="validation[:10000]")
    test_batch_size = 100
    if args.batch_size < test_batch_size:
        test_batch_size = 10
    vqa2_test_dataloader = DataLoader(vqa2_test, batch_size=test_batch_size, collate_fn=collate_fn(transform), num_workers=num_workers)
    
    print(f'init vqa fusion model "{args.vqa_fusion_network}"')
    fusion_model = None
    if args.vqa_fusion_network == "linear":
        fusion_model = LinearFusionModelCategorical(retrieval_model,2+4+2, len(category_list), args.vqa_hidden_sizes, args.vqa_input_type,dropout_rate=args.vqa_dropout).to(device)
    if args.vqa_fusion_network == "vqa1":
        fusion_model = VQAFusionModel(retrieval_model,4,3, len(category_list), args.vqa_hidden_sizes, dropout_rate=args.vqa_dropout).to(device)
    else:
        print(f'vqa_fusion_network "{args.vqa_fusion_network}" is not supported')
        exit(1)
    
    total_count = sum(category_counts)
    epsilon = 1000 # 1e-8  # Small value to prevent division by zero
    total_count = total_count + epsilon * len(category_counts)
    category_weights = [total_count / (class_count + epsilon) for class_count in category_counts]
    weights_tensor = torch.tensor(category_weights).to(device)
    loss_function = torch.nn.CrossEntropyLoss(weight=weights_tensor)     
               
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=args.vqa_lr, weight_decay=args.vqa_weight_decay)
    
    if use_f16:
        fusion_model, optimizer = amp.initialize(fusion_model, optimizer, opt_level="O2")
    
    n = 0
    
    loss_avg = 0
    
    use_embed_loss = False
    
    for epoch in range(1,args.vqa_epochs+1):
        print(f"epoch {epoch}")
        if epoch >= args.vqa_unfreeze_base_epoch and fusion_model.frozen_base_model:
            print(f"unfreeze base model")
            fusion_model.unfreeze_base_model()
            n = 0
        with tqdm(enumerate(vqa2_dataloader), total=len(vqa2_dataloader)) as progress_bar:
            for i, batch in progress_bar:            
                optimizer.zero_grad()
                outputs, last_features = fusion_model.forward(batch)
                answers = batch['multiple_choice_answer']
                targets = torch.tensor([get_category_id(answer) for answer in answers]).to(device)
                loss = loss_function(outputs, targets)
                
                if last_features.shape[1] == retrieval_model.embed_dim:
                    if not use_embed_loss:
                        progress_bar.write("using embedding loss") # only print once
                        use_embed_loss = True
                    target_last_features = torch.stack([get_text_features(retrieval_model, answer) for answer in answers], dim=0).to(device)
                    loss += 1 - F.cosine_similarity(last_features, target_last_features).mean()
                
                if use_f16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                loss_avg_rate = max(i, 99)
                loss_avg = (loss_avg * loss_avg_rate + loss.item()) / (loss_avg_rate + 1)
                optimizer.step()
                progress_bar.set_description(f"Epoch {epoch}, Iter {i}, l100: {loss_avg:.4f}")
                
                if epoch == 1 and (i+1+(epoch-1)*len(vqa2_dataloader)) % (128*2**n) == 0:
                    validation(1000, fusion_model, vqa2_test_dataloader)
                    n += 1
        validation(10000, fusion_model, vqa2_test_dataloader)
    
    
    