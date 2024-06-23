import os

from datasets import load_dataset, load_from_disk

if __name__ == "__main__":
    import common
    args, wandb = common.prepare_args(
        description="VQA for CreamFL Federated Learning (local simulation)",
        is_vqa=True)

    base_path = args.vqa_pretrained_base_model
    print(f"loading pretrained img txt model from {base_path}. Full path {os.path.abspath(base_path)}")
    from src.algorithms.retrieval_trainer import TrainerEngine
    retrieval_engine = TrainerEngine()
    print(f"  load COCOEvaluator")
    from src.algorithms.eval_coco import COCOEvaluator
    evaluator = COCOEvaluator(eval_method='matmul',
                            verbose=True,
                            eval_device='cuda',
                            n_crossfolds=5)
    print(f"  load models2")
    retrieval_engine.load_models2("./sl2-best_model.pt", evaluator)
    retrieval_engine.model_to_device()
    
    if args.vqa_pretrained_base_model_eval:
        print(f"loading coco test set")
        dataset_root = os.environ['HOME'] + '/data/mmdata/MSCOCO/2014'
        vocab_path = './src/custom_datasets/vocabs/coco_vocab.pkl'
        from src.utils.config import parse_config
        config = parse_config("./src/coco.yaml", strict_cast=False)
        from src.utils.load_datasets import prepare_coco_dataloaders
        dataloaders, vocab = prepare_coco_dataloaders(
            config.dataloader, dataset_root, args.pub_data_num, args.max_size, vocab_path
        )
        test_dataloader = dataloaders['test']
        test_scores = retrieval_engine.evaluate({'test': test_dataloader})
        
    # todo load vqa2 dataset
    print(f"loading vqa2 data set")
    vqa2_dataset = load_dataset("HuggingFaceM4/VQAv2")
    vqa2_train = vqa2_dataset["train"]
    print(f"vqa2 train dataset size {len(vqa2_train)}")
    print(f"vqa2 train dataset columns {vqa2_train.column_names}")
    print(f"vqa2 train dataset features {vqa2_train.features}")
    print(f"vqa2 train dataset example {vqa2_train[0]}")
    num_classes = 4000
    
    print(f'init vqa fusion model "{args.vqa_fusion_network}"')
    fusion_model = None
    if args.vqa_fusion_network == "linear":
        from src.networks.fusion_model import LinearFusionModel
        fusion_model = LinearFusionModel(retrieval_engine, num_classes)
    else:
        print(f'vqa_fusion_network "{args.vqa_fusion_network}" is not supported')
    
    
    
    