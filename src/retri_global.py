import common

if __name__ == "__main__":

    from algorithms.MMFL_dist import MMFL_Global

    args, wandb = common.prepare_args(
        description="CreamFL Federated Learning for retri task (global rep)",
        script="retri_global",
        is_vqa=False,
    )

    Algo = MMFL_Global(
        args,
        wandb,
        node_id="localhost",
        router_port=5001,
        peers=[["localhost", 5002], ["localhost", 5003], ["localhost", 5004]],
    )

    Algo.create_model(args)  # create client models and datasets
    Algo.load_dataset(args, is_vqa=False)  # global model and dataset

    for round_n in range(args.comm_rounds):
        Algo.train(round_n)
