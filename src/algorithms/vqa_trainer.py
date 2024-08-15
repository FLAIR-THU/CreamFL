from collections import Counter
import random
import torch
from tqdm import tqdm
from algorithms.optimizers import get_optimizer, get_lr_scheduler
from algorithms.retrieval_trainer import TrainerEngine
from algorithms.vqa_meta import VQAMetaData, unknown_category_id, unknown_category
from networks.fusion_model import VQAFusionModel

try:
    from apex import amp
    #print("enable f16 and using apex.amp for mixed precision training")
    #use_f16 = True
except ImportError as e:
    print('failed to import apex:', e)

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
        with tqdm(enumerate(vqa_loader), total=max_batches) as progress_bar:
            for i, batch in progress_bar:     
                if i >= max_batches:
                    break
                       
                self.vqa_optimizer.zero_grad()
                outputs, last_features = self.fusion_model.forward(batch)
                #answers = batch['multiple_choice_answer']
                answers = batch['answers'] # for multiple answers, learn a random answer based on popularity
                if isinstance(answers[0], list):
                    picked_answers = []
                    for answer_list in answers:
                        known = []
                        for answer in answer_list:
                            name = answer['answer']
                            id = self.vqa_meta.get_category_id(name)
                            if id != unknown_category_id:
                                known.append(id) 
                        if len(known) == 0:
                            known = [unknown_category_id] 
                        picked_answers.append(random.choice(known))
                    targets = torch.tensor(picked_answers).to(self.device)
                else:
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
                
                loss_avg_rate = min(i, 99)
                loss_avg = (loss_avg * loss_avg_rate + loss.item()) / (loss_avg_rate + 1)
                self.vqa_optimizer.step()
                progress_bar.set_description(f"Epoch {epoch}, Iter {i}, l100: {loss_avg:.4f}")
                
                if vqa2_test_dataloader is not None and (i+1) % (128*2**n) == 0:
                    #vqa_validation(1000, self.fusion_model, self.vqa_meta, vqa2_test_dataloader, 2)
                    #vqa_validation(1000, self.fusion_model, self.vqa_meta, vqa2_test_dataloader, 500)
                    #vqa_validation(1000, self.fusion_model, self.vqa_meta, vqa2_test_dataloader)
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
        #answers = testBatch['multiple_choice_answer'] # single answer
        answers = testBatch['answers'] # multiple answers
        outputs, _ = fusion_model.forward(testBatch)
        for k, answer in enumerate(answers):
            output = outputs[k]
            _, top_matches = torch.topk(output, min(5,max_cats+1), largest=True, sorted=True)
            top_match_names = [meta.get_category_by_id(cat_id.item()) for cat_id in top_matches]
            if isinstance(answer, list): # use testBatch['answers']
                expected_answer = [f"'{item}'x{count}" for item, count in Counter([v['answer'] for v in answer]).most_common()]
                for name in top_match_names:
                    if name in [v['answer'] for v in answer] :
                        answer = name # answer is the logical expected answer so our guess can be matched
                        break
                if isinstance(answer, list):
                    answer = answer[0]['answer'] # no matches, pick the first answer
            else:
                expected_answer = answer
            answer_id = meta.get_category_id(answer)
            match_type = "wrong answer"
            if answer_id > max_cats:
                answer_id = unknown_category_id
            if len(output) > max_cats:
                output = output[:max_cats+1]
            if top_match_names[0] == answer:
                right += 1
                match_type = "first"
            if top_matches[0] == unknown_category_id:
                unknown_outputs += 1
                match_type = "unknown output"
                if top_match_names[1] == answer:
                    unknown_right += 1
                    match_type = "second"
            if answer_id == unknown_category_id:
                unknown_answers += 1
                answer = unknown_category + answer # mark answers not in the training set
                match_type = "unknown answer"
                if top_matches[0] == unknown_category_id:
                    unknown_unknown += 1
                    match_type = "unknown unknown"
            if total + k < 16:
                tqdm.write(f"j {j}, k {k}, expected {expected_answer}, got {top_match_names}, match type {match_type}")
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
