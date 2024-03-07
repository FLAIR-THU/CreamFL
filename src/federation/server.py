# build in
import json
from http import HTTPStatus
# external dependencies
from flask import Flask, request
# internal dependencies
import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
from src.utils.logger import PythonLogger
import api, context

server = Flask(__name__)

server_context = None # set by main

current_state = api.ServerState()

url_prefix = api.url_prefix

@server.route(f'{url_prefix}', methods=['GET'])
def get():
    return json.dumps({"current_state":current_state.to_dict()})

@server.route(f'{url_prefix}/set_global', methods=['PUT'])
def set_global():
     # ensure that the server is in a state update the global model.
    if current_state.round_state != api.RoundState.BUSY:
        return json.dumps({"status":"error", "message":"The server should not update the global model at this time."}), HTTPStatus.CONFLICT
    # ensure that we are submitting to the correct round.
    round_number = request.args.get('round_number', default=-1)
    if round_number != current_state.round_number:
        return json.dumps({"status":"error", "message":"The round has passed"}), HTTPStatus.CONFLICT
    # ensure that the global model is identical to the one the client used. This should always be true considering the round number is already checked.
    old_feature_hash = request.args.get('old_feature_hash', default="")
    if old_feature_hash != current_state.feature_hash:
        return json.dumps({"status":"error", "message":"The global model has changed"}), HTTPStatus.CONFLICT
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
    if round_number != current_state.round_number:
        return json.dumps({"status":"error", "message":"The round has passed"}), HTTPStatus.CONFLICT
    # ensure that the global model is identical to the one the client used. This should always be true considering the round number is already checked.
    feature_hash = request.args.get('feature_hash', default="")
    if feature_hash != current_state.feature_hash:
        return json.dumps({"status":"error", "message":"The global model has changed"}), HTTPStatus.CONFLICT

    data = request.get_json()
    client = api.ClientState.from_dict(data)
    # should also verify client auth on an untrusted network.
    if len(current_state.clients_reported) == server_context.fed_config.max_clients:
        current_state.round_state = api.RoundState.BUSY
        server_context.logger.log(f"Global round {current_state.round_number} has collected the max number of clients.")

    return json.dumps({"status":"ok"})

if __name__ == '__main__':
    server_context = context.new_server_context()
    server.run(debug=True, port=2323)


