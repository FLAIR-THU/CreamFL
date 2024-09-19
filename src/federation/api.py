from typing import Dict, Optional
from enum import Enum
from datetime import datetime
import os
import time
import copy
import pickle
import torch
import hashlib
import requests

import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
from src.utils.load_datasets import prepare_coco_dataloaders

from context import Context

url_prefix = '/cream_api'

class RoundState(Enum):
    BUSY = 1 # The server is busy processing the current round and calculating the next global model.
    COLLECT = 2 # The server is collecting the client updates

class ClientState:
    def __init__(self, 
             name: str,
             img_model: bool = False,
             txt_model: bool = False,
             local_rounds: int = 0,
             img_model_hash: str = "",
             txt_model_hash: str = "",
             ):
        self.name = name
        self.img_model = img_model
        self.txt_model = txt_model
        self.local_rounds = local_rounds
        self.img_model_hash = img_model_hash
        self.txt_model_hash = txt_model_hash

    def to_dict(self):
        data = {
            'name': self.name,
            'local_rounds': self.local_rounds,
        }
        if self.img_model:
            data['img_model'] = self.img_model
            data['img_model_hash']= self.img_model_hash
        if self.txt_model:
            data['txt_model'] = self.txt_model
            data['txt_model_hash']= self.txt_model_hash
        return data
    
    @classmethod
    def from_dict(cls, data):
        required_keys = ['name', 'local_rounds']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Key '{key}' is missing from ClientState")
        return cls(
            name=data['name'],
            img_model=data.get('img_model', False),
            txt_model=data.get('txt_model', False),
            local_rounds=data['local_rounds'],
            img_model_hash=data.get('img_model_hash', ''),
            txt_model_hash=data.get('txt_model_hash', ''),
        )
    
class ServerState:
    def __init__(self, 
             round_number: int = 0, # defaults are for the starting state of a new server
             round_state: RoundState = RoundState.BUSY,
             round_started_at: datetime = datetime.now(), # the timestamp of the start of the round
             clients_reported: Optional[Dict[str, ClientState]] = None, # map of names to ClientState that have reported their updates
             feature_hash: str = ""):
        self.round_number = round_number
        self.round_state = round_state
        self.round_started_at = round_started_at
        self.clients_reported = clients_reported
        if self.clients_reported is None:
            self.clients_reported = {}
        self.feature_hash = feature_hash # the hash of the global model's features, the actual data could be distributed by an external service.
    
    def add_client(self, client):
        """
        Used by server to add a client to the list of clients that have reported their updates.
        """
        self.clients_reported[client.name] = client
    
    def advance_round(self):
        """
        Used by the server to advance the round to the next round. When the condition for finishing the round is met.
        """
        self.round_number += 1
        self.round_state = RoundState.BUSY
        self.round_started_at = datetime.now()

    def update_feature_hash(self, feature_hash):
        """
        Used by the server to update the feature hash of the global model and set the round state to COLLECT.
        """
        self.round_state = RoundState.COLLECT
        self.clients_reported = {}
        self.feature_hash = feature_hash

    def to_dict(self):
        return {
            'round_number': self.round_number,
            'round_state': self.round_state.value,
            'round_started_at': self.round_started_at.isoformat(),
            'clients_reported': {k: v.to_dict() for k, v in self.clients_reported.items()},
            'feature_hash': self.feature_hash,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            round_number=data['round_number'],
            round_state=RoundState(data['round_state']),
            round_started_at= datetime.fromisoformat(data['round_started_at']),
            clients_reported={k: ClientState.from_dict(v) for k, v in data['clients_reported'].items()},
            feature_hash=data['feature_hash'],
        )

def feature_hash(data):
    return hashlib.sha3_256(data).hexdigest()

def save(obj, path):
    os.makedirs(path, exist_ok=True)
    data = pickle.dumps(obj)
    hash = feature_hash(data)
    fn = f"{path}/{hash}.pkl"
    with open(fn, 'wb') as f:
            f.write(data)
    return fn, hash

def load(cls, path, hash):
    fn = f"{path}/{hash}.pkl"
    with open(fn, 'rb') as f:
        obj = pickle.load(f)
    if not isinstance(obj, cls):
        raise ValueError(f"Object loaded from {fn} is not an instance of {cls}")
    return obj


class GlobalFeature:
    def __init__(self, img: torch.Tensor, txt: torch.Tensor, distill_index: list):
        self.img: torch.Tensor = img
        self.txt: torch.Tensor = txt
        self.distill_index = distill_index
        self.hash = None # calculated when saved to or loaded from disk

    def save(self, path):
        fn, hash = save(self, path)
        self.hash = hash
        return fn, hash
    
    @classmethod
    def load(cls, path, hash):
        obj = load(cls, path, hash)
        obj.hash = hash
        return obj
    
def get_api_url(context:Context):
    return context.fed_config.server["api_url"]
    
def get_global_dataloader(context:Context):
    dataset_root = os.environ['HOME'] + '/data/mmdata/MSCOCO/2014'
    vocab_path = './src/custom_datasets/vocabs/coco_vocab.pkl'
    return prepare_coco_dataloaders(context.config.dataloader, dataset_root, context.args.pub_data_num, context.args.max_size, vocab_path)

def status_sleep(context, msg):
    context.logger.log(f"{msg}, sleeping for 10 seconds.")
    time.sleep(10)

def error_sleep(context, error):
    context.logger.log(f"Error: {error}, sleeping for 60 seconds.")
    time.sleep(60)

# start shared api section
def get_server_state(context:Context, expected_state: Optional[RoundState] = None):
    """
    Get the current state of the server. 
    If expected_state is not None, this function will wait until the server is in the expected state.
    """
    while True:
        try:
            url = get_api_url(context)
            context.logger.log(f"Getting server state from {url}")
            resp = requests.get(url)
            state = ServerState.from_dict(resp.json()['current_state'])
            if expected_state is not None and state.round_state != expected_state:
                status_sleep(context, f"Server state is not {expected_state}")
                time.sleep(10)
                continue
            return state
        except Exception as e:
            error_sleep(context, e)

# start client api section


def get_global_feature(context:Context, state:ServerState):
    """
    Get the global feature from a distributed storage service.

    Only mounted files are supported currently.
    """
    return GlobalFeature.load(context.fed_config.feature_store, state.feature_hash)

def add_local_repr(context:Context, expected_server_state:ServerState, img, txt, local_rounds:int):
    """
    Submit the local representations to the server.

    local_rounds is the number of local rounds that the client has trained the model for, this is only used for reporting.
    
    Returns True if the submission was successful and the client should restart a new round.
    Returns False if the submission was not successful but the client should restart a new round.
    """
    context.logger.log(f"Saving local representations.")
    client_state = ClientState(name=context.args.client_name, img_model=context.has_img_model, txt_model=context.has_txt_model, local_rounds=local_rounds)
    if context.has_img_model:
        _, client_state.img_model_hash = save(img, context.fed_config.feature_store)
    if context.has_txt_model:
        _, client_state.txt_model_hash = save(txt, context.fed_config.feature_store)

    url = get_api_url(context)+ f'/add_client?round_number={expected_server_state.round_number}&feature_hash={expected_server_state.feature_hash}'
    context.logger.log(f"Submitting local representations to server.")
    while True:
        try:
            resp = requests.put(url, json=client_state.to_dict())
            if resp.status_code == 200:
                context.logger.log(f"Local representations submitted to server.")
                return True
            context.logger.log(f"Can not add local representations to server. Status code: {resp.status_code} body: {resp.text}")
            return False
        except Exception as e:
            error_sleep(context, e)
# end client api section
            
# start server api section
def submit_global_feature(context:Context, state:ServerState, global_feature:GlobalFeature):
    context.logger.log(f"Saving global feature to file.")
    _, hash = global_feature.save(context.fed_config.feature_store)
    url = get_api_url(context) + f'/set_global_feature?round_number={state.round_number}&old_feature_hash={state.feature_hash}&new_feature_hash={hash}'
    context.logger.log(f"Submitting global feature to server.")
    while True:
        try:
            resp = requests.put(url)
            if resp.status_code == 200:
                context.logger.log(f"Global feature submitted to server.")
                return True
            context.logger.log(f"Can not submit global feature to server. Status code: {resp.status_code} body: {resp.text}")
            error_sleep(context, resp.status_code)
            return False
        except Exception as e:
            error_sleep(context, e)

def get_clients_repr(context:Context, clients_reported: Dict[str, ClientState]):
    img_vec = []
    txt_vec = []
    for name, client in clients_reported.items():
        if client.img_model:
            img = load(torch.Tensor,context.fed_config.feature_store, client.img_model_hash)
            img_vec.append(img)
        if client.txt_model:
            txt = load(torch.Tensor,context.fed_config.feature_store, client.txt_model_hash)
            txt_vec.append(txt)
    return img_vec, txt_vec
    
# end server api section