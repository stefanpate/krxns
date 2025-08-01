import pandas as pd
from tqdm import tqdm
from pathlib import Path
import hydra
from omegaconf import DictConfig
from krxns.network import ReactionNetwork


@hydra.main(version_base=None, config_path="../configs", config_name="construct_known_reaction_network")
def main(cfg: DictConfig):
    rc_0_mapped = pd.read_parquet(
        Path(cfg.filepaths.raw_data) / cfg.rc_plus_0_mapped
    )

    mechinformed_mapped = pd.read_parquet(
        Path(cfg.filepaths.raw_data) / cfg.mechinformed_mapped
    )

    # Prefer the mechinformed atom mapping to the rc_plus_0
    overlap = rc_0_mapped.rxn_id.isin(mechinformed_mapped.rxn_id)
    mapped_rxns = pd.concat([mechinformed_mapped, rc_0_mapped[~overlap]], ignore_index=True)

    # Get smi2name, smi2idx mappings
    kcs = pd.read_parquet(
        Path(cfg.filepaths.raw_data) / cfg.known_compounds
    )
    smi2name = dict(zip(kcs['smiles'], kcs['name']))

    # Save default set of sources
    sources = kcs[kcs["name"].isin(cfg.sources.source_names)]
    sources.to_csv(
        Path(cfg.filepaths.interim_data) / "default_sources.csv",
        index=False
    )

    G = ReactionNetwork()
    for _, row in tqdm(mapped_rxns.iterrows(), total=len(mapped_rxns), desc="Adding reactions to network"):
        G.add_reaction(am_rxn=row['am_smarts'], smi2name=smi2name) # Append to network

    G.to_json(Path(cfg.filepaths.processed_data) / "known_reaction_network.json") # Save

if __name__ == '__main__':
    main()