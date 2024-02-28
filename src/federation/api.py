from enum import Enum
import hashlib

class RoundState(Enum):
    BUSY = 1 # The server is busy processing the current round
    COLLECT = 2 # The server is collecting the client updates

class ServerState:
    def __init__(self, 
             round_number=0, # defaults are for the starting state of a new server
             round_state=RoundState.BUSY,
             round_duration=0, # number of seconds left in the current round, only used if round_state == COLLECT
             clients_reported=0, # number of clients that have reported their updates
             feature_hash=""):
        self.round_number = round_number
        self.round_state = round_state
        self.round_duration = round_duration
        self.clients_reported = clients_reported
        self.feature_hash = feature_hash # the hash of the global model's features, the actual data could be distributed by an external service.
    
    def to_dict(self):
        return {
            'round_number': self.round_number,
            'round_state': self.round_state.value,
            'round_duration': self.round_duration,
            'clients_reported': self.clients_reported,
            'feature_hash': self.feature_hash,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            round_number=data['round_number'],
            round_state=RoundState(data['round_state']),
            round_duration=data['round_duration'],
            clients_reported=data['clients_reported'],
            feature_hash=data['feature_hash'],
        )

def feature_hash(data):
    return hashlib.sha3_256(data).digest()

class ClientState:
    def __init__(self, name,
                 img_model = False,
                 txt_model = False,
                 round_number = 0,
                 img_model_hash = "",
                 txt_model_hash = "",
                 ):
        self.name = name
        self.img_model = img_model
        self.txt_model = txt_model
        self.round_number = round_number
        self.img_model_hash = img_model_hash
        self.txt_model_hash = txt_model_hash

    def to_dict(self):
        data = {
            'name': self.name,
            'round_number': self.round_number,
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
        required_keys = ['name', 'round_number']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Key '{key}' is missing from ClientState")
        return cls(
            name=data['name'],
            img_model=data.get('img_model', False),
            txt_model=data.get('txt_model', False),
            round_number=data['round_number'],
            img_model_hash=data.get('img_model_hash', ''),
            txt_model_hash=data.get('txt_model_hash', ''),
        )