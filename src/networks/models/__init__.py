__all__ = ['get_model']

from src.networks.models.pcme import PCME


def get_model(word2idx, config, mlp_local):
    return PCME(word2idx, config, mlp_local)
