import os
import pickle

import torch
import datasets
from tqdm import tqdm

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
        