from concurrent.futures import ProcessPoolExecutor
from argparse import ArgumentParser
from functools import partial
import json
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Callable
from networkx import Graph
from krxns.utils import str2int
from krxns.config import filepaths
from krxns.cheminfo import calc_mfp_matrix, tanimoto_similarity
from krxns.net_construction import construct_reaction_network
from krxns.networks import SuperMultiDiGraph
import numpy as np

G = None # To be reaction network
M = None # To be Morgan fingerprint matrix

def greedy_tanimoto(intermediate: int, target: int, G: Graph, V: np.ndarray):
    options = []
    t_mfp = V[target['cpd_ids'], :]
    for s in G.successors(intermediate):
        s_mfp = V[s, :]
        sim_to_target = tanimoto_similarity(s_mfp, t_mfp)
        options.append((s, sim_to_target))

    srt_options = sorted(options, key=lambda x : x[1], reverse=True)

    return srt_options[0][0] if options else None

def init_process(main_G, main_V):
    '''
    Initializes process with graph G and node embedding matrix V
    '''
    global G, V
    G = main_G
    V = main_V

def search(pair: tuple, max_steps: int, forward: Callable[[int, int, Graph], int]):
    G = globals["G"]
    V = globals()["V"]
    starter, target = pair
    intermediate = starter
    path = str(intermediate)
    while intermediate != target and len(path) <= max_steps + 1:
        intermediate = forward(intermediate, target, G, V)
        
        if intermediate is None:
            return (starter, target, path)

        path += f",{str(intermediate)}"

    return (starter, target, path)

def main():
    args = parser.parse_args()

    # Load known reaction data
    with open(filepaths['data'] / f"{args.rxns}.json", 'r') as f:
        rxns = json.load(f)

    # Load op connected reactions
    with open(filepaths['connected_reactions'] / f"{args.op_cxns}.json", 'r') as f:
        op_cxns = str2int(json.load(f))

    # Load sim connected reactions
    with open(filepaths['connected_reactions'] / f"{args.sim_cxns}.json", 'r') as f:
        sim_cxns = str2int(json.load(f))

    with open(filepaths['connected_reactions'] / f"{args.side_cts}.json", 'r') as f:
        side_counts = str2int(json.load(f))

    # Load coreactant whitelist
    with open(filepaths['coreactants'] / f"{args.whitelist}.json", 'r') as f:
        whitelist = json.load(f)

    # Construct network
    G = SuperMultiDiGraph()
    edges, nodes = construct_reaction_network(
        operator_connections=op_cxns,
        similarity_connections=sim_cxns,
        side_counts=side_counts,
        reactions=rxns,
        connect_nontrivial=args.connect_nontrivial,
        coreactant_whitelist=whitelist,
        atom_lb=args.atom_lb,
        add_multi_mol_nodes=args.multi_nodes
    )
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    node_smiles = {i: G.nodes[i]['smiles'] for i in G.nodes}

    print("Getting shortest paths")
    paths = G.shortest_path() # Get shortest paths

    # Filter out self-paths
    tmp = {}
    n_paths = 0
    for i in paths:
        destinations = {j: elt for j, elt in paths[i].items() if i != j}
        if destinations:
            n_paths += len(destinations)
            tmp[i] = destinations

    paths = tmp
    
    st_generator = ((i, j) for i in paths for j in paths[i]) # Starter target pairs (tasks)
    
    # Select search function, worker initializer stuff
    if args.strategy == "greedy-tanimoto":
        fcn = partial(search, max_steps=args.max_steps, forward=greedy_tanimoto)
        V = calc_mfp_matrix(node_smiles) # Morgan fingerprints

    with ProcessPoolExecutor(initializer=init_process, initargs=(G, V)) as pool:
        res = pool.map(fcn, st_generator)

    save_dir = filepaths['results'] / "graph_traversal" / f"{args.rxns}"
    
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    # Save to parquet
    fp = save_dir / f"traversed_paths_{args.strategy}_max_steps_{args.max_steps}_{args.whitelist}_atom_lb_{int(args.atom_lb * 100)}p_multi_nodes_{args.multi_nodes}.parquet"
    col_names = ["starter", "target", "path"]
    starter, target, path = list(zip(*res))
    starter = pa.array(starter, type=pa.int16())
    target = pa.array(target, type=pa.int16())
    path = pa.array(path, type=pa.string())
    res = pa.table([starter, target, path], names=col_names)
    pq.write_table(res, fp)

parser = ArgumentParser(description="Traverse reaction network with selected strategy")
parser.add_argument("strategy", help="Which strategy to traverse reaction net with (greedy-tanimoto, )")
parser.add_argument("max_steps", type=int, help="Maximum number of steps to take from starter")
parser.add_argument("whitelist", help="Filename of coreactants whitelisted as currency")
parser.add_argument("atom_lb", type=float, help="Below this fraction of heavy atoms, reactants will be ignored, i.e., not connected to products")
parser.add_argument("--op-cxns", default="sprhea_240310_v3_mapped_operator", help="Filename operator connected adjacency matrices")
parser.add_argument("--rxns", default="sprhea_240310_v3_mapped", help="Reactions dataset")
parser.add_argument("--sim-cxns", default="sprhea_240310_v3_mapped_similarity", help="Filename similarity connected adjacency matrices")
parser.add_argument("--side-cts", default="sprhea_240310_v3_mapped_side_counts", help="Counts of unique, non-currency molecules on each side of reactions")
parser.add_argument("--connect-nontrivial", action="store_true", help="Connect nontrivial reactions w/ similarity connections")
parser.add_argument("--multi-nodes", action="store_true", help="Add multiple molecule nodes to network")

if __name__ == '__main__':
    main()