import hydra
from omegaconf import DictConfig
from pathlib import Path
import pandas as pd
from functools import partial
from ergochemics.standardize import standardize_smiles
import json
from krxns.network import construct_reaction_network, SuperMultiDiGraph
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from tqdm import tqdm

def initializer(_standardize_smiles: callable, _G: SuperMultiDiGraph) -> None:
    """
    Initialize the worker with the standardize_smiles function and the graph.
    """
    print("Initializing worker...", flush=True)
    global std_smi, G, shortest_paths
    std_smi = partial(_standardize_smiles, quiet=True, neutralization_mode="simple")
    G = _G
    shortest_paths = dict(G.shortest_path())

def process_pair(pair: tuple[str, str]) -> dict[str, any]:
    """
    Standardize smiles, and try to find paths betweeen them in 
    the known reaction network.
    """
    tsmi = std_smi(pair[1])
    sidx = pair[0]
    tidx = G.get_nodes_by_prop("smiles", tsmi)

    if not tidx:
        return {}
    
    tidx = tidx[0]  # Nodes should be unique in smiles, so we take the first one
    if sidx == tidx:
        return {}
    elif sidx not in shortest_paths or tidx not in shortest_paths[sidx]:
        return {}
    else:
        path = shortest_paths[sidx][tidx]
        return {
            "source": sidx,
            "target": tidx,
            "path": path,
            "path_length": len(path) - 1,
            "source_smiles": G.nodes[sidx].get("smiles"),
            "target_smiles": tsmi,
            "source_name": G.nodes[sidx].get("name"),
            "target_name": G.nodes[tidx].get("name"),
        }

_std_smi = partial(standardize_smiles, quiet=True, neutralization_mode="simple")  

@hydra.main(version_base=None, config_path="../configs", config_name="find_paths")
def main(cfg: DictConfig) -> None:
    """
    Find paths based on the configuration provided.
    """
    # Load data
    sources = pd.read_csv(
        Path(cfg.filepaths.interim_data) / f"{cfg.sources}.csv"
    )['id'].tolist()

    kcs = pd.read_csv(
        Path(cfg.filepaths.interim_data) / f"{cfg.kcs}.csv"
    )

    starters = pd.read_csv(
        Path(cfg.filepaths.raw_data) / f"{cfg.starters}.csv" # TODO: change to parquet in case large
    )
    starters["smiles"] = starters["smiles"].apply(_std_smi) # TODO: std s and t smiles in parallel

    targets = pd.read_parquet(
        Path(cfg.filepaths.raw_data) / f"{cfg.targets}.parquet"
    )

    with open(Path(cfg.filepaths.interim_data) / "mass_contributions.json", 'r') as f:
        mass_contributions = json.load(f)

    if cfg.starters_as_sources:
        addtl_sources = kcs.loc[kcs["smiles"].isin(starters["smiles"]), "id"].tolist()
        sources += addtl_sources

    # Construct known reaction network
    edges, nodes = construct_reaction_network(
        mass_contributions=mass_contributions,
        compounds=kcs,
        sources=sources,
        pnmc_lb=cfg.pnmc_lb,
        rnmc_lb=cfg.rnmc_lb,
    )
    G = SuperMultiDiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    print(f"Constructed reaction network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Process pairs
    pairs = list(product(addtl_sources, targets["smiles"].tolist()))
    chunksize = max(1, int(len(pairs) / cfg.processes))
    with ProcessPoolExecutor(max_workers=cfg.processes, initializer=initializer, initargs=(standardize_smiles, G)) as executor:
        print("Processing w/ context: ", executor._mp_context)
        paths = list(
            tqdm(
                executor.map(process_pair, pairs, chunksize=chunksize),
                total=len(pairs),
                desc="Procesing pairs"
            )
        )

    # # TODO: remove. jsut for debugging
    # initializer(standardize_smiles, G)
    # print("Processing reactions in main process")
    # paths = []
    # for pr in tqdm(pairs, desc="Processing pairs", total=len(pairs)): 
    #     paths.append(process_pair(pr))

    paths = [p for p in paths if p]  # Filter out empty results
    paths_df = pd.DataFrame(paths)
    paths_df.to_parquet(
        f"known_paths_{cfg.starters}_to_{cfg.targets}_pnmc_lb_{cfg.pnmc_lb}_rnmc_lb_{cfg.rnmc_lb}_sources_{cfg.sources}_s_as_sources_{cfg.starters_as_sources}.parquet",
    )

if __name__ == "__main__":
    main()