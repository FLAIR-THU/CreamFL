import os
import argparse
from utils.helper import Helper as helper
import algorithms
import random

import common

parser = argparse.ArgumentParser(description='CreamFL Federated Learning (local simulation)')
common.add_args(parser)
args = parser.parse_args()

if __name__ == "__main__":

    from algorithms.MMFL import MMFL

    wandb = common.init_wandb(args)

    Algo = MMFL(args, wandb)

    args.save_dirs = helper.get_save_dirs(args.exp_dir, args.name)
    args.log_dir = args.save_dirs['logs']
    helper.set_seed(args.seed)

    Algo.create_model(args) # create client models and datasets
    Algo.load_dataset(args) # global model and dataset

    for round_n in range(args.comm_rounds):
        Algo.train(round_n)

    Algo.logger.log("Best:")
    Algo.engine.report_scores(step=args.comm_rounds,
                              scores=Algo.best_scores,
                              metadata=Algo.best_metadata,
                              prefix=Algo.engine.eval_prefix)
