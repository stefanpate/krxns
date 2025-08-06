import hydra
from omegaconf import DictConfig
import pandas as pd
from krxns.network import ReactionNetwork
from pathlib import Path
from time import perf_counter
import json
import logging

logger  = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="retrosynthesis")
def main(cfg: DictConfig):
    G = ReactionNetwork.from_json(Path(cfg.known_reaction_network))

    default_sources = pd.read_csv(Path(cfg.default_sources))['smiles'].tolist()

    with open(Path(cfg.expansion_extract) / cfg.sources, 'r') as f:
        sources = [line.strip() for line in f.readlines()]

    with open(Path(cfg.expansion_extract) / cfg.targets, 'r') as f:
        targets = [line.strip() for line in f.readlines()]

    am_rxns = pd.read_parquet(Path(cfg.expansion_extract) / cfg.am_rxns)

    # TODO: Add operators as field
    logger.info("Adding reactions to network...")
    tic = perf_counter()
    for _, row in am_rxns.iterrows():
        try:
            G.add_reaction(row['am_smarts'])
        except:
            logger.info(f"Failed to add reaction: {row['am_smarts']}")
            continue
    toc = perf_counter()
    logger.info(f"Added {len(am_rxns)} reactions in {toc - tic:.2f} seconds.")

    rxn_lookup = {k: d['am_smarts'] for _, _, k, d in G.edges(keys=True, data=True)}

    logger.info("Setting sources...")
    G.set_sources(smiles=default_sources)
    G.set_sources(smiles=sources)

    target_ids = [G.get_nodes_by_prop('smiles', t)[0] for t in targets]

    logger.info("Enumerating synthetic trees...")
    target_to_trees = {}
    for tid in target_ids:
        tic = perf_counter()
        trees = G.enumerate_synthetic_trees(
            target=tid,
            max_depth=cfg.max_depth,
            max_leaves=cfg.max_leaves,
            tot_rnmc_lb=cfg.tot_rnmc_lb
        )
        toc = perf_counter()
        logger.info(f"Tree enumeration for target {tid} took {toc - tic:.2f} seconds.")
        target_to_trees[tid] = [
            [
                {k: rxn_lookup[v] for k, v in gen.items()}
                for gen in tree.generations
            ]
            for tree in trees
        ]
    
    logger.info("Saving results...")
    with open(f"{cfg.expansion}_synthetic_trees.json", 'w') as f:
        json.dump(target_to_trees, f)

if __name__ == "__main__":
    main()