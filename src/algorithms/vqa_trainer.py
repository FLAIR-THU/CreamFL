

import heapq
import os
import pickle

import torch
import datasets
from tqdm import tqdm
from algorithms.optimizers import get_optimizer, get_lr_scheduler
from algorithms.retrieval_trainer import TrainerEngine
from networks.fusion_model import VQAFusionModel
from utils.util import print_model_tree

try:
    from apex import amp
    #print("enable f16 and using apex.amp for mixed precision training")
    #use_f16 = True
except ImportError as e:
    print('failed to import apex:', e)

unknown_category = "<unknown>"
unknown_category_id = 0

def build_or_load_categories(fn, dataset):
    """
    Load categories from a file if it exists, otherwise build them from a dataset.

    This function checks if a file with the name `fn` exists. If it does, it loads and returns the
    categories stored in that file. If the file does not exist, it builds the categories from the
    provided dataset. The categories are sorted by their counts (excluding the first category, which
    is reserved for unknowns) from most to lest. The sorted categories and their counts are
    then saved to the file `fn` for future use.
    
    The intention of sorting the categories is to make picking the top N categories easier.

    Parameters:
    - fn (str): The filename where the categories are stored or will be stored.
    - dataset: The dataset from which to build the categories if the file does not exist.

    Returns:
    - dict: A dictionary containing the sorted category list under the key 'category_list' and the
      corresponding counts under the key 'category_counts'.
    """
    if os.path.exists(fn):
        with open(fn, "rb") as f:
            return pickle.load(f)
    builder = VQAMetaData()
    builder.set_category_from_dataset(dataset)
    sorted_pairs = sorted(zip(builder.category_list[1:], builder.category_counts[1:]), key=lambda x: x[1], reverse=True)
    data = {'category_list': builder.category_list[:1]+[cat for cat, _ in sorted_pairs],
            'category_counts': builder.category_counts[:1]+[count for _, count in sorted_pairs]}
    with open(fn, "wb") as f:
        pickle.dump(data, f)
        return data

class VQAMetaData():
    def __init__(self):
        self.category_list = [] # list of categories names
        self.category_dict = {} # category name to index
        self.category_counts = [] # category counts for each category from the training set

        
    def build_or_load_categories_top(self, top = 3000):
        if len(self.category_list) != 0:
            raise Exception("categories already loaded")
        fn = f"vqa2_categories_train.pkl"
        data = build_or_load_categories(fn, datasets.load_dataset("HuggingFaceM4/VQAv2", split="train"))
        
        for i, cat in enumerate(data['category_list']):
            count = data['category_counts'][i]
            if i <= top:
                self.category_list.append(cat)
                self.category_counts.append(count)
                self.category_dict[cat] = i
            else:
                self.category_counts[unknown_category_id] += count
    
    def get_category_size(self):
        return len(self.category_list)

    def get_category_id(self, cat, add_new=False):
        add_count = add_new # add count only when we are building the list of categories
        if len(self.category_list) == 0:
            self.category_dict[unknown_category] = unknown_category_id
            self.category_list.append(unknown_category)
            self.category_counts.append(0)
        if cat in self.category_dict:
            cat_id = self.category_dict[cat]
            if add_count:
                self.category_counts[cat_id] += 1
            return cat_id
        if not add_new:
            return unknown_category_id
        new_id = len(self.category_list)
        self.category_dict[cat] = new_id
        self.category_list.append(cat)
        self.category_counts.append(1)
        return new_id
    
    def get_category_by_id(self, cat_id):
        return self.category_list[cat_id]

    def set_category_from_dataset(self,dataset):
        #for item in tqdm(dataset.map(lambda example: {'multiple_choice_answer': example['multiple_choice_answer']})):
        #    get_category_id(item['multiple_choice_answer'])
        dataset = dataset.map(lambda example: {'multiple_choice_answer': example['multiple_choice_answer']})
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2048, num_workers=32,
                                collate_fn=lambda examples: {'multiple_choice_answer': [example['multiple_choice_answer'] for example in examples]})
        self.set_category_from_dataloader(dataloader)
            
    def set_category_from_dataloader(self, dataloader):
        for batch in tqdm(dataloader):
            for answer in batch['multiple_choice_answer']:
                self.get_category_id(answer, add_new=True)
    
    def get_weights(self, args):
        if args.vqa_cat_weight == '1':
            return None # use default uniform weights for CrossEntropyLoss
        epsilon = 1e-8  # Small value to prevent division by zero
        if args.vqa_cat_weight == 'count+1000':
            epsilon = 1000
        total_count = sum(self.category_counts)
        total_count = total_count + epsilon * self.get_category_size()
        return [total_count / (class_count + epsilon) for class_count in self.category_counts]
        

class VQAEngine():
    def __init__(self, args, base_trainer_engine:TrainerEngine, wandb, device='cuda'):
        self.args = args
        self.device = device
        self.trainer_engine = base_trainer_engine
        self.wandb = wandb
        self.fusion_model = None
        self.vqa_optimizer = None
        self.vqa_criterion = None
        self.vqa_lr_scheduler = None
        self.vqa_meta = None
        
    def weights_tensor(self, meta:VQAMetaData):
        weights = meta.get_weights(self.args)
        if weights is None:
            return None
        return torch.tensor(weights).to(self.device)

    def create(self, config, word2idx, evaluator, mlp_local, meta:VQAMetaData):
        self.config = config
        self.vqa_meta = meta
        self.trainer_engine.create(config, word2idx, evaluator, mlp_local)
        self.fusion_model = VQAFusionModel(self.device,self.trainer_engine.model,1,1, meta.get_category_size(), config.vqa_hidden_sizes, dropout_rate=config.vqa_dropout).to(self.trainer_engine.device)
        self.vqa_criterion = torch.nn.CrossEntropyLoss(weight=self.weights_tensor(meta)).to(self.device)
        self.vqa_optimizer = get_optimizer(config.optimizer.name,
                                         self.fusion_model.parameters(),
                                         config.optimizer)
        self.vqa_lr_scheduler = get_lr_scheduler(config.lr_scheduler.name,
                                               self.vqa_optimizer,
                                               config.lr_scheduler)
    def to_half(self):
        # Mixed precision
        # https://nvidia.github.io/apex/amp.html
        self.fusion_model, self.vqa_optimizer = amp.initialize(self.fusion_model, [self.vqa_optimizer, self.trainer_engine.optimizer],
                                                    opt_level='O2')
        
    def train(self, tr_loader, pub_data_ratio=1.):
        self.trainer_engine.train(tr_loader, pub_data_ratio)
        
    def train_vqa(self, epoch, vqa_loader, vqa2_test_dataloader = None):
        self.fusion_model.train()
        full_training_epoch = self.args.vqa_full_training_epoch
        if epoch < full_training_epoch:
            print("Freezing base model")
            self.fusion_model.freeze_base_model()
        else:
            print("Not freezing base model")
        # print_model_tree(self.fusion_model)
        
        max_batches = len(vqa_loader)
        if self.args.vqa_data_size_per_epoch == 0:
            max_batches = self.args.pub_data_num / vqa_loader.batch_size
        elif self.args.vqa_data_size_per_epoch > 0:
            max_batches = self.args.vqa_data_size_per_epoch / vqa_loader.batch_size
            
        n = 0
        loss_avg = 0
        with tqdm(enumerate(vqa_loader), total=len(vqa_loader)) as progress_bar:
            for i, batch in progress_bar:     
                if i >= max_batches:
                    break
                       
                self.vqa_optimizer.zero_grad()
                outputs, last_features = self.fusion_model.forward(batch)
                answers = batch['multiple_choice_answer']
                targets = torch.tensor([self.vqa_meta.get_category_id(answer) for answer in answers]).to(self.device)
                loss = self.vqa_criterion(outputs, targets)
                
                if self.config.train.get('use_fp16'):
                    with amp.scale_loss(loss, self.vqa_optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                    
                if epoch >= full_training_epoch and self.config.train.grad_clip > 0:
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.fusion_model.parameters(),
                                                   self.config.train.grad_clip)
                
                loss_avg_rate = max(i, 99)
                loss_avg = (loss_avg * loss_avg_rate + loss.item()) / (loss_avg_rate + 1)
                self.vqa_optimizer.step()
                progress_bar.set_description(f"Epoch {epoch}, Iter {i}, l100: {loss_avg:.4f}")
                
                if vqa2_test_dataloader is not None and (i+1) % (128*2**n) == 0:
                    #vqa_validation(1000, self.fusion_model, self.vqa_meta, vqa2_test_dataloader, 2)
                    #vqa_validation(1000, self.fusion_model, self.vqa_meta, vqa2_test_dataloader, 500)
                    vqa_validation(1000, self.fusion_model, self.vqa_meta, vqa2_test_dataloader)
                    vqa_validation(10000, self.fusion_model, self.vqa_meta, vqa2_test_dataloader)
                    n += 1
                    self.fusion_model.train()
                    if epoch < full_training_epoch:
                        self.fusion_model.freeze_base_model()
        self.wandb.log({"train_vqa_loss_100": loss_avg}, step=epoch)  
        #self.wandb.log({"train_vqa_lr": self.vqa_optimizer.param_groups[0].get_lr()})
        self.fusion_model.unfreeze_base_model()
        #print_model_tree(self.fusion_model)


@torch.no_grad()
def vqa_validation(n, fusion_model, meta, validation_dataloader, max_cats = 3000):        
    fusion_model.eval()
    right = 0
    unknown_right = 0
    unknown_outputs = 0
    unknown_answers = 0
    unknown_unknown = 0
    total = 0
    for j, testBatch in tqdm(enumerate(validation_dataloader)):
        answers = testBatch['multiple_choice_answer']
        outputs, _ = fusion_model.forward(testBatch)
        for k, answer in enumerate(answers):
            answer_id = meta.get_category_id(answer)
            if answer_id > max_cats:
                answer_id = unknown_category_id
            output = outputs[k]
            if len(output) > max_cats:
                output = output[:max_cats+1]
            _, top_matches = torch.topk(output, min(5,max_cats+1), largest=True, sorted=True)
            top_match_names = [meta.get_category_by_id(cat_id.item()) for cat_id in top_matches]
            if top_match_names[0] == answer:
                right += 1
            if top_matches[0] == unknown_category_id:
                unknown_outputs += 1
                if top_match_names[1] == answer:
                    unknown_right += 1
            if answer_id == unknown_category_id:
                unknown_answers += 1
                answer = unknown_category + answer # mark answers not in the training set
                if top_matches[0] == unknown_category_id:
                    unknown_unknown += 1
            if total + k < 8:
                tqdm.write(f"j {j}, k {k}, expected {answer}, got {top_match_names}")
        total += len(answers)
        if total >= n:
            break
    accuracy = (right + unknown_right) / total
    tqdm.write(f"test {max_cats} accuracy {right+unknown_right}/{total}={accuracy},  unknown_answers:{unknown_answers}, unknown_outputs:{unknown_outputs}, right after unknown:{unknown_right}, unknown_unknown:{unknown_unknown}")
    
    return {
        "max_cats": max_cats,
        "right0": right,
        "right": right+unknown_right,
        "total": total,
        "accuracy": accuracy,
        "unknown_answers": unknown_answers,
        "unknown_outputs": unknown_outputs,
        "right1": unknown_right,
        "unknown_unknown": unknown_unknown
    }
