from argparse import ArgumentParser
from krxns.config import filepaths
from krxns.utils import str2int
from krxns.networks import SuperMultiDiGraph
from krxns.net_construction import construct_reaction_network, extract_compounds
from krxns.cheminfo import calc_mfp_matrix, tanimoto_similarity
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from collections import defaultdict
from tqdm.contrib.concurrent import process_map
from concurrent.futures import ProcessPoolExecutor
import networkx as nx

def init_process(main_G, main_V):
    '''
    Initializes process with graph G and node embedding matrix V
    '''
    global G, V
    G = main_G
    V = main_V

def yield_decisions(path_values: dict, successor_values: dict, topks: list):
    for k in topks:
        for i in successor_values:
            for j in successor_values[i]:
                for step in range(1, len(path_values[i][j]) - 1):
                    successors = successor_values[i][j][step - 1]
                    
                    if len(successors) > k:
                        chosen = path_values[i][j][step]

                        yield (chosen, successors, k)

def count_topk(decision: tuple):
    chosen, successors, k = decision
    ep = 1e-5
    less_than_chosen = [elt < (chosen + ep) for elt in sorted(successors, reverse=True)]
    chosen_rank = less_than_chosen.index(True)
    in_top_k = 1 if chosen_rank < k else 0
    return (k, in_top_k)

def get_tani_sims(task: tuple):
    V = globals()["V"]
    G = globals()["G"]
    starter, target, path = task
    target_mfp = V[path[-1], :]
    path_tanis = []
    successor_tanis = []
    for i, elt in enumerate(path):
        path_tanis.append(tanimoto_similarity(V[elt, :], target_mfp))

        if i < len(path) - 2:
            tmp = []
            for s in G.successors(elt):
                tmp.append(tanimoto_similarity(V[s, :], target_mfp))
            
            successor_tanis.append(tmp)

    return [(starter, target), path_tanis, successor_tanis]

def spearman(path_values: list):
    x = np.arange(len(path_values))
    return spearmanr(x, path_values).statistic

def calc_path_spearmans(path_values: dict):
    '''
    Calculate spearmans coefficient between path similarities and 
    reaction step
    '''          
    ps_generator = (path_values[i][j] for i in path_values for j in path_values[i])
    print("Calculating spearman rs")
    res = process_map(spearman, ps_generator, chunksize=10)
    
    col_names = ['starter_id', 'target_id', 'spearman_r']
    path_spearmans = []
    idx_generator = ((i, j) for i in path_values for j in path_values[i])
    for n, (i, j) in enumerate(idx_generator):
        path_spearmans.append([i, j, res[n]])

    path_spearmans = pd.DataFrame(data=path_spearmans, columns=col_names)

    return path_spearmans
        
def calc_topk_rates(path_values: dict, successor_values: dict, topks: list[int]):
    '''
    Calculate fraction of time shortest path proceeds to top k most 
    similar intermediate to target
    '''
    decision_generator = yield_decisions(path_values, successor_values, topks)
    print("Counting topk")
    res = process_map(count_topk, decision_generator, chunksize=10)

    in_top_k = {k: 0 for k in topks}
    counts = {k: 0 for k in topks}
    for k, flag in res:
        in_top_k[k] += flag
        counts[k] += 1

    fracs = {k: in_top_k[k] / counts[k] if counts[k] > 0 else 0 for k in topks}

    return fracs

def pathwise_value_fcn(args):
    '''
    Evalutates value function for every intermediate along
    a path to target and likewise for successors of intermediates.
    Returns
    -------
    path_values:
    successor_values:
    '''
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

    print("Calculating node embeddings")
    if args.value_fcn == 'tanimoto':
        f = get_tani_sims
        V = calc_mfp_matrix(node_smiles) # Morgan fingerprints

    st_generator = ((i, j, paths[i][j]) for i in list(paths.keys())[::args.downsample] for j in paths[i])

    print("Processing paths")
    with ProcessPoolExecutor(initializer=init_process, initargs=(G, V)) as pool:
        res = pool.map(f, st_generator)

    path_values = defaultdict(dict)
    successor_values = defaultdict(dict)
    for elt in res:
        st, ps, ss = elt
        i, j = st
        path_values[i][j] = ps
        successor_values[i][j] = ss

    topk_rates = calc_topk_rates(path_values=path_values, successor_values=successor_values, topks=args.topks)
    path_spearmans = calc_path_spearmans(path_values=path_values)

    save_dir = filepaths['results'] / "pathwise_value_fcn" / f"{args.rxns}"
    fn_tail = f"{args.value_fcn}_{args.whitelist}_atom_lb_{int(args.atom_lb * 100)}p_multi_nodes_{args.multi_nodes}_ds_{args.downsample}"
    
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    # Convert path sims to dataframe
    data = []
    cols = ['starter_id', 'target_id', 'path_values']
    for i in path_values:
        for j in path_values[i]:
            string_values = ",".join([str(elt) for elt in path_values[i][j]])
            data.append([i, j, string_values])
    
    path_values = pd.DataFrame(data=data, columns=cols)

    to_store = path_values.merge(right=path_spearmans, on=['starter_id', 'target_id'], how='left')
    to_store.to_parquet(save_dir / f"pathwise_{fn_tail}.parquet", index=False)
    
    with open(save_dir / f"topk_{fn_tail}.json", 'w') as f:
        json.dump(topk_rates, f)


parser = ArgumentParser(description="Analyzes value fcn along paths. Saves value fcn along paths, calculates spearman w/ step index, saves topk")

# Pathwise similarity calculations
parser.add_argument("value_fcn", help="Which value function to use, e.g., tanimoto")
parser.add_argument("whitelist", help="Filename of coreactants whitelisted as currency")
parser.add_argument("atom_lb", type=float, help="Below this fraction of heavy atoms, reactants will be ignored, i.e., not connected to products")
parser.add_argument("--op-cxns", default="sprhea_240310_v3_mapped_operator", help="Filename operator connected adjacency matrices")
parser.add_argument("--rxns", default="sprhea_240310_v3_mapped", help="Reactions dataset")
parser.add_argument("--sim-cxns", default="sprhea_240310_v3_mapped_similarity", help="Filename similarity connected adjacency matrices")
parser.add_argument("--side-cts", default="sprhea_240310_v3_mapped_side_counts", help="Counts of unique, non-currency molecules on each side of reactions")
parser.add_argument("--downsample", type=int, default=1, help="Downsample path sources (starters) by this factor")
parser.add_argument("--multi-nodes", action="store_true", help="Add multiple molecule nodes to network")
parser.add_argument("--topks", nargs="+", type=int, default=[1, 2, 5, 10], help="Top ks")
parser.add_argument("--connect-nontrivial", action="store_true", help="Connect nontrivial reactions w/ similarity connections")
parser.set_defaults(func=pathwise_value_fcn)

def main():
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()