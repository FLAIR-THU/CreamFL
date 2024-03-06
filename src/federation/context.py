import torch

from src.common import prepare_args
from src.utils.logger import PythonLogger
import config


class Context():
    """
    Represents the context for executing a script. For sharing configuration logic between federation scripts.

    Args:
        description (str): A description of the script.
        script (str): The name/type of script to be executed.
    """
    def __init__(self, description: str, script: str):
        # from main
        args, wandb = prepare_args(description, script)
        self.args = args
        self.wandb = wandb

        # from MMFL
        self.logger = PythonLogger() # TODO: output_file=self.config.train.output_file

        # automatically set device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:%d" % args.device)
        else:
            self.device = torch.device("cpu")


        # federation specific config file
        self.fed_config = config.parse_config(args.fed_config)



def new_server_context():
    return Context(description="CreamFL Federated Learning (http server)", script="server")

def new_client_context():
    return Context(description="CreamFL Federated Learning (client)", script="client")

def new_global_context():
    return Context(description="CreamFL Federated Learning (global compute)", script="global")