import common

if __name__ == "__main__":

    from algorithms.MMFL import MMFL

    args, wandb = common.prepare_args(
        description="CreamFL Federated Learning for VQA task (local simulation)",
        script="vqa",
        is_vqa=True)

    Algo = MMFL(args, wandb)

    Algo.create_model(args) # create client models and datasets
    Algo.load_dataset(args, is_vqa=True) # global model and dataset

    for round_n in range(args.comm_rounds):
        Algo.train(round_n)

    #Algo.logger.log("Best:")
    #Algo.engine.report_scores(step=args.comm_rounds,
    #                          scores=Algo.best_scores,
    #                          metadata=Algo.best_metadata,
    #                          prefix=Algo.engine.eval_prefix)
