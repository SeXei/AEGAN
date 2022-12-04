from model import pred_wrap
import torch
import torch.nn as nn
import torch.utils.data as data
from pytorch_lightning.callbacks import ModelCheckpoint  # checkpoint
from pytorch_lightning.callbacks import StochasticWeightAveraging
from customDataset import customDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import argparse
parser = argparse.ArgumentParser()

def parse_args():
    parser.add_argument("--trainset", type=str, default="../database/trainset", required=False,
                        help="Train dataset path")
    parser.add_argument("--batchsize", type=int, default=200, required=False, help="Batch size")
    parser.add_argument('-lr', "--learingrate", type=float, default=1e-2, required=False, help="Learning rate")
    parser.add_argument('-ly', "--layer", type=int, default=24, required=False, help="Numbers of AEGAN")
    parser.add_argument('--epoch', type=int, default=1000, required=False, help="Traing epochs")
    parser.add_argument("-sm", "--savemodel", type=int, required=True, help="Path of saving trained model")
    parser.add_argument("--accelerator", type=str, default="gpu", required=False, help="Uesd device to train model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batchsize = args.batchsize
    train_dataset = customDataset(args.trainset)
    train_set_size = int(len(train_dataset) * 0.9)
    valid_set_size = len(train_dataset) - train_set_size
    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = data.random_split(train_dataset, [train_set_size, valid_set_size], generator=seed)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=batchsize,
                              shuffle=True,
                              num_workers=35,
                              drop_last=False,
                              prefetch_factor=50,
                              pin_memory=True,
                              persistent_workers=True
                              )
    valid_loader = DataLoader(dataset=valid_set,
                              batch_size=batchsize,
                              num_workers=35,
                              drop_last=False,
                              prefetch_factor=50,
                              pin_memory=True,
                              persistent_workers=True
                              )
    Model = pred_wrap(
        edgefea=35,
        nodefea=36,
        dropout=0.5,
        head=3,
        alpha=0.3,
        layer=args.layer,
        weight_decay=1e-4,
        lr_rate=args.learningrate,
        loss_fn=nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0]))
    )

    checkpoint_callbacks = ModelCheckpoint(
        save_top_k=10,
        monitor="valid_mcc",
        mode="max",
        filename="aegan-{epoch:02d}-{valid_mcc:.4f}-{valid_f_measure:.4f}-{valid_recall:.4f}"
    )
    trainer = pl.Trainer(
        devices=1,
        accelerator=args.accelerator,
        default_root_dir=args.savemodel,
        max_epochs=args.epoch,
        callbacks=[checkpoint_callbacks, StochasticWeightAveraging(swa_lrs=1e-2)],
        accumulate_grad_batches=10,
        precision=16
    )
    trainer.fit(model=Model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
