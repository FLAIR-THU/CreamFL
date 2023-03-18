from criterions.probemb import MCSoftContrastiveLoss


def get_criterion(criterion_name, config):
    if criterion_name == 'pcme':
        return MCSoftContrastiveLoss(config)
    else:
        raise ValueError(f'Invalid criterion name: {criterion_name}')
