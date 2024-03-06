import os
import argparse
import algorithms
import random

import common

if __name__ == "__main__":

    from algorithms.MMFL import MMFL

    args, wandb = common.prepare_args(description="CreamFL Federated Learning (local simulation)")

    Algo = MMFL(args, wandb)

    Algo.create_model(args) # create client models and datasets
    Algo.load_dataset(args) # global model and dataset

    for round_n in range(args.comm_rounds):
        Algo.train(round_n)

    Algo.logger.log("Best:")
    Algo.engine.report_scores(step=args.comm_rounds,
                              scores=Algo.best_scores,
                              metadata=Algo.best_metadata,
                              prefix=Algo.engine.eval_prefix)
