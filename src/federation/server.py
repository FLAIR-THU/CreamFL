import json

from flask import Flask

import api

server = Flask(__name__)

prefix = '/cream-api/'

current_state = api.ServerState()

@server.route(f'{prefix}', methods=['GET'])
def get():
    return json.dumps({"current_state":current_state.to_dict()})

if __name__ == '__main__':
    server.run(debug=True, port=2323)


