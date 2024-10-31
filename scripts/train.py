'''
Inputs:
model (hyperparams)
data (chunk slice)
split idx (optional default to None, acts as cross val flag)
featurizer

To do:
Load data
Featurize
Construct model
Train
Test
Save: model, hyperparams to tracking csv, predictions, metrics
'''
from time import perf_counter
import pandas as pd
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger
from chemprop import featurizers, nn
from chemprop.nn import metrics
from chemprop.models import multi
from argparse import ArgumentParser
from krxns.config import filepaths
from krxns.ml import load_data, split_data, featurize_data

parser = ArgumentParser(description="Construct model and fit on train data")
parser.add_argument("data_dir", help="Path dir containing chunks, relative to configs data path")
parser.add_argument("n_chunks", type=int, help="Number of data chunks to load")
parser.add_argument("max_epochs", type=int, help="Max number training epochs")
parser.add_argument("--split-idx", type=int, default=-1, help="CV split form 0 to k - 1. If not provided, train on all data")
parser.add_argument("--experiment-name", default=None, help="Organize logs into subdir")

def main(args):
    all_data = load_data(filepaths['data'] / args.data_dir, args.n_chunks)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer() # TODO: extend to others w/ CL arg selecting

    if args.split_idx >= 0:
        train_data, val_data = split_data(all_data, args.split_idx)
        train_data, scaler = featurize_data(train_data, featurizer, train_mode=True) # Should only shuffle for train dataloader (see chemprop docs)
        val_data, _ = featurize_data(val_data, featurizer, train_mode=False)
    else:
        train_data, scaler = featurize_data(all_data, featurizer, train_mode=True)

    # Construct model
    mcmp = nn.MulticomponentMessagePassing(blocks=[nn.BondMessagePassing()], n_components=2, shared=True)
    agg = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(input_dim=mcmp.output_dim, output_transform=output_transform)
    metric_list = [metrics.RMSE(), metrics.MAE()] # Only the first metric is used for training and early stopping
    mcmpnn = multi.MulticomponentMPNN(mcmp, agg, ffn, metrics=metric_list, batch_norm=True)

    # Fit
    save_dir = filepaths['spl_cv'] / f"hp_0" / f"split_{args.split_idx}"
    logger = CSVLogger(save_dir=save_dir, name=args.experiment_name, version=f"version_0") # TODO: will probably change
    trainer = Trainer(
        logger=logger,
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=args.max_epochs, # number of epochs to train for
    )
    tic = perf_counter()
    trainer.fit(model=mcmpnn, train_dataloaders=train_data, val_dataloaders=val_data)
    toc = perf_counter()
    print(f"Training took: {toc - tic:.2f} seconds")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)