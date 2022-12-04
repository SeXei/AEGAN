from model import pred_wrap
import torch
import torch.utils.data as data
from pytorch_lightning.callbacks import ModelCheckpoint  # checkpoint
from pytorch_lightning.callbacks import StochasticWeightAveraging
from customDataset import customDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from collections import defaultdict
from tqdm import tqdm
import os
from pandas import DataFrame
import argparse
parser = argparse.ArgumentParser()

def parse_args():
    parser.add_argument("--accelerator", type=str, default="gpu", required=False, help="Uesd device to train model")
    parser.add_argument("--testset", type=str, default="uni3175", required=False, help="Test dataset")
    parser.add_argument("--batchsize", type=int, defaull=100, required=False, help="Batch size")
    parser.add_argument("--lack", type=str, default=None, required=False, help="Ablation experiment")
    parser.add_argument("--model", type=str, required=True, help="Trained model")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    store_data = defaultdict(list)
    trainer = pl.Trainer(
        devices=1,
        accelerator=args.accelerator
    )
    source_map = {
        "uni3175": "../database/validateset/",
        "EF_family": "../database/BenchmarkDataset_Processed/EF_family",
        "EF_fold": "../database/BenchmarkDataset_Processed/EF_fold",

        "EF_superfamily": "../database/BenchmarkDataset_Processed/EF_superfamily",
        "HA_superfamily": "../database/BenchmarkDataset_Processed/HA_superfamily",
        "NN":"../database/BenchmarkDataset_Processed/NN",
        "PC": "../database/BenchmarkDataset_Processed/PC"
    }
    if source_map.get(args.test) is not None:
        setpath = source_map.get(args.test)
    else:
        setpath = args.test
    validate_dataset = customDataset(setpath, Lack=args.lack)
    validate_loader = DataLoader(dataset=validate_dataset, batch_size=100, num_workers=40,
                                 drop_last=False, prefetch_factor=10, persistent_workers=True)
    index = ['test_accuracy', 'test_f_measure', 'test_mcc', 'test_precision', 'test_recall']
    if os.path.isdir(args.model):
        store_data[args.test] = index
        for m in tqdm(os.listdir(args.model)):
            Model = pred_wrap.load_from_checkpoint(os.path.join(args.model, m),
                                                   loss_fn=torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0])))
            result = trainer.test(model=Model, dataloaders=validate_loader)[0]
            store_data[m] = [result[k] for k in index]
    elif args.model == 'trained_full':
        store_data[args.test] = index
        for m in tqdm(os.listdir("./train_result_logs/lightning_logs/version_0/checkpoints/")):
            Model = pred_wrap.load_from_checkpoint(os.path.join(args.model, m),
                                                   loss_fn=torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0])))
            result = trainer.test(model=Model, dataloaders=validate_loader)[0]
            store_data[m] = [result[k] for k in index]
    elif args.model in ['EP', 'EAT', 'EAM']:
        store_data[args.test] = index
        for m in tqdm(os.listdir(f"./AblationExperiment/version_{args.model}/checkpoints/")):
            Model = pred_wrap.load_from_checkpoint(os.path.join(args.model, m),
                                                   loss_fn=torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0])))
            result = trainer.test(model=Model, dataloaders=validate_loader)[0]
            store_data[m] = [result[k] for k in index]
    else:
        Model = pred_wrap.load_from_checkpoint(args.model,
                                               loss_fn=torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0])))
        result = trainer.test(model=Model, dataloaders=validate_loader)[0]
        store_data[args.model] = [result[k] for k in index]
    df = DataFrame(store_data)
    df.to_excel(args.test + '.xlsx')
