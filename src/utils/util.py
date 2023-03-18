import copy
import torch 


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