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

import pandas as pd
from lightning import pytorch as pl
from chemprop import data, featurizers, models, nn
from chemprop.nn import metrics
from chemprop.models import multi
from argparse import ArgumentParser

def load_data(n_chunks: int, data_dir: str) -> pd.DataFrame:
    data = []
    for i in range(n_chunks):
        data.append(pd.read_parquet(data_dir / f"chunk_{i}.parquet"))

    data = pd.concat(data, axis=0).reset_index()
    return data

def featurize_data() -> nn.Dataloader: 
    pass

def split_data():
    pass



parser = ArgumentParser(description="Construct model and fit on train data")
parser.add_argument("data_dir", help="Path ")
parser.add_argument("n_chunks", type=int, help="Number of data chunks to load")

def main(args):
    pass

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)