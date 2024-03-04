import os
import munch
import yaml
from yaml.error import YAMLError
import copy

def read_config_file(fn string):
    with open(fn, 'r') as fin:
        return yaml.safe_load(fin)

def apply_defaults(dic, defaults):
    out = copy.deepcopy(defaults)
    for k, v in dic.items():
        if isinstance(v, dict) and k in defaults:
            out[k] = apply_defaults(v, defaults[k])
        else:
            out.update({k: v})
    return out

def load_config(config_file, defaults = None):
    config = read_config_file(config_file)
    if defaults is not None:
        config = apply_defaults(config, defaults)
    return munch.Munch(config)