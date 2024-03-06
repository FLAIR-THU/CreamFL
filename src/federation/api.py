from typing import Dict, Optional
from enum import Enum
from datetime import datetime
import copy
import pickle
import torch
import hashlib

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
            round_number=data['local_rounds'],
            img_model_hash=data.get('img_model_hash', ''),
            txt_model_hash=data.get('txt_model_hash', ''),
        )
    
class ServerState:
    def __init__(self, 
             round_number: int = 0, # defaults are for the starting state of a new server
             round_state: RoundState = RoundState.BUSY,
             round_started_at: Optional[datetime] = None, # the timestamp of the start of the round
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
            'round_started_at': self.round_started_at,
            'clients_reported': {k: v.to_dict() for k, v in self.clients_reported.items()},
            'feature_hash': self.feature_hash,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            round_number=data['round_number'],
            round_state=RoundState(data['round_state']),
            round_started_at=data['round_started_at'],
            clients_reported={k: ClientState.from_dict(v) for k, v in data['clients_reported'].items()},
            feature_hash=data['feature_hash'],
        )

def feature_hash(data):
    return hashlib.sha3_256(data).hexdigest()

class GlobalFeature:
    def __init__(self, img: torch.Tensor, txt: torch.Tensor, distill_index: list):
        self.img = img
        self.txt = txt
        self.distill_index = distill_index

    def save(self, path):
        data = pickle.dumps(self)
        hash = feature_hash(data)
        fn = f"{path}/{hash}.pkl"
        with open(fn, 'wb') as f:
            f.write(data)
        return fn, hash
    
    @classmethod
    def load(cls, path, hash):
        fn = f"{path}/{hash}.pkl"
        with open(fn, 'rb') as f:
            return pickle.load(f)
        