# build in
import json
from http import HTTPStatus
# external dependencies
from flask import Flask, request
# internal dependencies
import sys
from src.algorithms.retrieval_trainer import TrainerEngine
from src.utils.load_datasets import imagenet_transform
from src.algorithms.eval_coco import COCOEvaluator
from PIL import Image
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
from src.utils.logger import PythonLogger
import api, context, os
import torch
server = Flask(__name__)

server_context = None # set by main

current_state = api.ServerState()

url_prefix = api.url_prefix

engine = TrainerEngine()

@server.route(f'{url_prefix}', methods=['GET'])
def get():
    return json.dumps({"current_state":current_state.to_dict()})

@server.route(f'{url_prefix}/last_train_result', methods=['GET'])
def get_result():
    current_path = os.path.dirname(os.path.dirname(__file__))
    accuracy = ''
    loss = ''
    recall = ''
    with open(os.path.join(current_path, 'accuracy.txt'), 'r') as f:
        accuracy = f.readlines()
    with open(os.path.join(current_path, 'loss.txt'), 'r') as f:
        loss = f.readlines()
    with open(os.path.join(current_path, 'recall.txt'), 'r') as f:
        recall = f.readlines()
    return json.dumps({"accuracy": accuracy, "loss": loss, "recall": recall})

@server.route(f'{url_prefix}/inference', methods=['POST'])
def inference():
    result = None
    if request.method == 'POST':
        f = request.files['file']
        captions = request.form['captions']
        # f.save(f.filename)

        global engine
        engine.model.eval()
        images = (torch.tensor(convert_img(f)))
        images = images.unsqueeze(0)
        images = images.to(engine.device)
        sentences = []
        captions = captions.split('\n')
        # dataloader = DataLoader(images,
        #                             batch_size=1,
        #                             shuffle=False,
        #                             num_workers=1,
        #                             collate_fn=image_to_caption_collate_fn,
        #                             pin_memory=True)
        result = engine.model(images, sentences, captions, len(sentences))
        # result = engine.evaluate(images, sentences, captions, len(sentences))
    return json.dumps({"status": "ok", "result": result})

def convert_img(path, cutout_prob=0.0):
    _image_transform = imagenet_transform(
        random_resize_crop=False,
        random_erasing_prob=cutout_prob,
    )
    img = Image.open(path).convert('RGB')
    img = _image_transform(img)
    return img


@server.route(f'{url_prefix}/set_global_feature', methods=['PUT'])
def set_global():
     # ensure that the server is in a state update the global model.
    if current_state.round_state != api.RoundState.BUSY:
        return json.dumps({"status":"error", "message":"The server should not update the global model at this time."}), HTTPStatus.CONFLICT
    # ensure that we are submitting to the correct round.
    round_number = request.args.get('round_number', default=-1)
    if round_number != str(current_state.round_number):
        return json.dumps({"status":"error", "message":f"The round has passed. expected={current_state.round_number}, got={round_number}"}), HTTPStatus.CONFLICT
    # ensure that the global model is identical to the one the client used. This should always be true considering the round number is already checked.
    old_feature_hash = request.args.get('old_feature_hash', default="missing")
    if old_feature_hash != current_state.feature_hash:
        return json.dumps({"status":"error", "message":f"The global model has changed {old_feature_hash}!={current_state.feature_hash}"}), HTTPStatus.CONFLICT
    new_feature_hash = request.args.get('new_feature_hash', default="")
    if new_feature_hash == "":
        return json.dumps({"status":"error", "message":"The new feature hash is required"}), HTTPStatus.BAD_REQUEST
    current_state.update_feature_hash(new_feature_hash)
    return json.dumps({"status":"ok"})

@server.route(f'{url_prefix}/add_client', methods=['PUT'])
def add_client():
    # ensure that the server is in a state to accept clients models.
    if current_state.round_state == api.RoundState.BUSY:
        return json.dumps({"status":"error", "message":"The server is not accepting clients at this time."}), HTTPStatus.CONFLICT
    # ensure that we are submitting to the correct round.
    round_number = request.args.get('round_number', default=-1)
    if round_number != str(current_state.round_number):
        return json.dumps({"status":"error", "message":f"The round has passed. expected={current_state.round_number}, got={round_number}"}), HTTPStatus.CONFLICT
    # ensure that the global model is identical to the one the client used. This should always be true considering the round number is already checked.
    feature_hash = request.args.get('feature_hash', default="")
    if feature_hash != current_state.feature_hash:
        return json.dumps({"status":"error", "message":"The global model has changed"}), HTTPStatus.CONFLICT

    data = request.get_json()
    client = api.ClientState.from_dict(data)
    # should also verify client auth on an untrusted network.
    current_state.clients_reported[client.name] = client
    if len(current_state.clients_reported) == server_context.fed_config.server['max_clients']:
        current_state.advance_round()
        server_context.logger.log(f"Global round {current_state.round_number} has collected the max number of clients.")

    return json.dumps({"status":"ok"})

if __name__ == '__main__':
    evaluator = COCOEvaluator(eval_method='matmul',
                                       verbose=True,
                                       eval_device='cuda',
                                       n_crossfolds=5)
    engine.load_models2("./test-best_model.pt", evaluator)
    engine.model_to_device()

    server_context = context.new_server_context()
    server.run(port=2323)


