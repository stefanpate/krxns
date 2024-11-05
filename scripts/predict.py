'''
First must troubleshoot the gpu / cpu mem loading problem
with chemprops load function
'''

import torch
from krxns.config import filepaths
from krxns.ml import load_data, split_data, featurize_data
from chemprop import featurizers
from chemprop.models import multi
from lightning import pytorch as pl
import numpy as np
import pandas as pd
from chemprop import nn
from chemprop.nn import metrics
from chemprop.models import multi

# Latest ckpt helper fcn
def latest_ckpt(ckpt_dir: Path):
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    ckpt_rank = [tuple([elt.split("=")[-1] for elt in ckpt.stem.split("-")]) for ckpt in ckpts]
    srt_ckpts = sorted(zip(ckpt_rank, ckpts), key=lambda x : x[0], reverse=True)
    return srt_ckpts[0][1]

# Load data stuff
all_data = load_data(filepaths['data'] / data_dir, n_chunks)
featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
train_data, test_data = split_data(all_data, split_idx)
train_dataloader = featurize_data(train_data, featurizer)
test_dataloader, _ = featurize_data(test_data, featurizer)

# Chemprop way
ckpt = latest_ckpt(ckpt_dir)
model = multi.MulticomponentMPNN.load_from_checkpoint(ckpt, map_location=torch.device('cpu'))

# Hacky way (does work but requires you to remember metric list, batch_norm, ...)
super_dict = torch.load(ckpt, map_location=torch.device('cpu'))
mcmp = nn.MulticomponentMessagePassing(blocks=[nn.BondMessagePassing()], n_components=2, shared=True)
agg = nn.MeanAggregation()
output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
ffn = nn.RegressionFFN(input_dim=mcmp.output_dim, output_transform=output_transform)
metric_list = [metrics.RMSE(), metrics.MAE()] # Only the first metric is used for training and early stopping
model = multi.MulticomponentMPNN(mcmp, agg, ffn, metrics=metric_list, batch_norm=True)
model.load_state_dict(super_dict['state_dict'])

# Inference mode on!!!
with torch.inference_mode():
    trainer = pl.Trainer(
        logger=False,
        enable_progress_bar=True,
        accelerator="auto",
        devices=1
    )
    test_preds = trainer.predict(model, test_dataloader)

test_preds = np.concatenate(test_preds, axis=0)