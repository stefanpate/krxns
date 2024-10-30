'''
Machine learning helpers
'''
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from typing import Any
from chemprop import data
from pathlib import Path

def load_data(data_dir: Path, n_chunks: int = None) -> pd.DataFrame:
    df = []
    srt_fps = sorted(
        [(int(fp.stem.split("_")[-1]), fp) for fp in data_dir.glob("chunk_*.parquet")],
        key=lambda x : x[0]
    )
    for i, filepath in srt_fps:
        if n_chunks is not None and i == n_chunks:
            break
        else:
            df.append(pd.read_parquet(filepath))

    df = pd.concat(df, axis=0).reset_index(drop=True)
    
    return df

# TODO: Get the right types for featurizer and scaler
def featurize_data(df: pd.DataFrame, featurizer:Any, train_mode:bool) -> tuple[DataLoader, Any]:
    '''
    Converts dataframe with SMILES data to dataloader.

    Args
    ----
    df
    featurizer
    train_mode:bool
        If true will shuffle batches and return a target scaler, else does not shuffle
        and returned scaler is None
    '''
    smiles_cols = ['starter_smiles', 'target_smiles']
    target_cols = ['spl']
    smiss = df.loc[:, smiles_cols].values
    ys = df.loc[:, target_cols].values
    datapoints = [[data.MoleculeDatapoint.from_smi(smis[0], y) for smis, y in zip(smiss, ys)]]
    datapoints += [[data.MoleculeDatapoint.from_smi(smis[i]) for smis in smiss] for i in range(1, len(smiles_cols))]
    datasets = [data.MoleculeDataset(datapoints[i], featurizer) for i in range(len(smiles_cols))]
    mc_dataset = data.MulticomponentDataset(datasets=datasets)

    if train_mode:
        scaler = mc_dataset.normalize_targets()
    else:
        scaler = None

    dataloader = data.build_dataloader(mc_dataset, shuffle=train_mode)
    
    return dataloader, scaler

def split_data(df: pd.DataFrame, split_idx: int) -> tuple[pd.DataFrame]:
    k = 5
    if split_idx < 0 or split_idx > k:
        raise ValueError(f"Provided split index {split_idx} not between 0 and {k}")
    skf = StratifiedKFold(n_splits=k, shuffle=False)
    splits = list(skf.split(df[['starter_id', 'target_id', 'starter_smiles', 'target_smiles']], df[['spl']]))
    train_idx, test_idx = splits[split_idx]
    return df.loc[train_idx, :], df.loc[test_idx, :]