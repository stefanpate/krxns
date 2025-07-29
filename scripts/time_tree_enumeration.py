import hydra
from omegaconf import DictConfig
from krxns.network import ReactionNetwork
import logging
from pathlib import Path
import pandas as pd
from time import perf_counter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="time_tree_enumeration")
def main(cfg: DictConfig):
    
    # Load the known reaction network
    G = ReactionNetwork.from_json(Path(cfg.known_reaction_network))

    logger.info("Full reaction network loaded from JSON.")
    logger.info(f"Number of nodes: {G.number_of_nodes()}, Number of edges: {G.number_of_edges()}")

    # Load & set sources
    default_sources = pd.read_csv(Path(cfg.default_sources), sep=',')['smiles'].tolist()
    addtl_sources = cfg.additional_sources
    sources = default_sources + addtl_sources
    logger.info(f"Total sources loaded: {len(sources)}")
    G.set_sources(smiles=sources)

    for max_depth in cfg.max_depths:
        for max_leaves in cfg.max_leaves:
            logger.info(f"Enumerating with max_depth={max_depth} and max_leaves={max_leaves}")
            tick = perf_counter()
            syntrees = G.enumerate_synthetic_trees(
                target=cfg.target,
                max_depth=max_depth,
                max_leaves=max_leaves,
                tot_rnmc_lb=cfg.tot_rnmc_lb,
            )
            tock = perf_counter()
            elapsed_time = tock - tick
            logger.info(f"Enumeration took {elapsed_time:.2f} seconds for max_depth={max_depth} and max_leaves={max_leaves}. Found {len(syntrees)} synthetic trees.")

if __name__ == "__main__":
    main()