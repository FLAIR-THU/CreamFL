from . probemb import MCSoftContrastiveLoss
from torch.nn import BCEWithLogitsLoss

def get_criterion(criterion_name, config):
    if criterion_name == 'pcme':
        return MCSoftContrastiveLoss(config)
    if criterion_name == "vqa":
        return BCEWithLogitsLoss(config)
    else:
        raise ValueError(f'Invalid criterion name: {criterion_name}')
