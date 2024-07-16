import copy
import torch 
import torch.nn as nn

def print_model_tree(model:nn.Module, indent=0):
    for name, module in model.named_children():
        # Print the module type and increase indentation
        print(' ' * indent + f'{name}: {module.__class__.__name__}')
        
        # If the module has children, recursively print them
        if list(module.children()):
            print_model_tree(module, indent + 2)
        else:
            # If it's a leaf module, print the parameter shapes
            for param_name, param in module.named_parameters():
                print(' ' * (indent + 2) + f'{param_name} shape:{param.shape} grad:{param.requires_grad}')


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def sum_weights(w):
    """
    Returns the average of the weights.
    """
    w_sum = copy.deepcopy(w[0])
    for key in w_sum.keys():
        for i in range(1, len(w)):
            w_sum[key] += w[i][key]
    return w_sum



