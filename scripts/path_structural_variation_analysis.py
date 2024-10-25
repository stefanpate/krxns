from argparse import ArgumentParser
from krxns.config import filepaths
from krxns.utils import str2int
from krxns.networks import SuperMultiDiGraph
from krxns.net_construction import construct_reaction_network, extract_compounds
from krxns.cheminfo import MorganFingerPrinter
import json
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict
from tqdm.contrib.concurrent import process_map
import multiprocessing as mp
from rdkit import Chem
import networkx as nx
from functools import partial
from time import perf_counter

def yield_decisions(path_tanis: dict, successor_tanis: dict, topks: list):
    for k in topks:
        for i in successor_tanis:
            for j in successor_tanis[i]:
                for step in range(1, len(path_tanis[i][j]) - 1):
                    successors = successor_tanis[i][j][step - 1]
                    
                    if len(successors) > k:
                        chosen = path_tanis[i][j][step]

                        yield (chosen, successors, k)

def count_topk(decision: tuple):
    chosen, successors, k = decision
    ep = 1e-5
    less_than_chosen = [elt < (chosen + ep) for elt in sorted(successors, reverse=True)]
    chosen_rank = less_than_chosen.index(True)
    in_top_k = 1 if chosen_rank < k else 0
    return (k, in_top_k)                        

def yield_path(paths: dict[dict], G: nx.Graph, T: np.ndarray, M:np.ndarray, downsample: int):
    for i in list(paths.keys())[::downsample]:
        for j in paths[i]:
            path = paths[i][j]
            path_cpd_ids = [G.nodes[p]['cpd_ids'] for p in path]
            target = path_cpd_ids[-1]
            successors_cpd_ids = [[G.nodes[s]['cpd_ids'] for s in G.successors(p)] for p in path[:-2]]

            target_mfp = M[target, :]
            path_tani_mfp = []
            for c in path_cpd_ids:
                if len(target) == 1 and len(c) == 1:
                    path_tani_mfp.append(T[target[0], c[0]])
                else:
                    path_tani_mfp.append(M[c, :])

            successor_tani_mfp = []
            for successors in successors_cpd_ids:
                tmp = []
                for s in successors:
                    if len(target) == 1 and len(s) == 1:
                        tmp.append(T[target[0], s[0]])
                    else:
                        tmp.append(M[s, :])
                
                successor_tani_mfp.append(tmp)

            yield [(i, j), target_mfp, path_tani_mfp, successor_tani_mfp]

def get_tani_sims(path: list):
    try: 
        st, target, path_tani_mfp, successor_tani_mfp = path
        for i, elt in enumerate(path_tani_mfp):
            if type(elt) is np.ndarray:
                path_tani_mfp[i] = multi_mol_tanimoto(elt, target)
        
        for i, successors in enumerate(successor_tani_mfp):
            for j, elt in enumerate(successors):
                if type(elt) is np.ndarray:
                    successor_tani_mfp[i][j] = multi_mol_tanimoto(elt, target)

        return [st, path_tani_mfp, successor_tani_mfp]
    except Exception as e:
        print(f"Error in child: {e}")

def multi_mol_tanimoto(mix1: np.ndarray, mix2: np.ndarray):
    '''
    Calculate tanimoto similarity between "mixtures" of
    molecules in multi mol nodes
    '''
    elt_wise_union = lambda arr : (arr.sum(axis=0) > 0).astype(int)
    mfp1 = elt_wise_union(mix1)
    mfp2 = elt_wise_union(mix2)
    dp = np.dot(mfp1, mfp2)
    tani = dp / (mfp1.sum() + mfp2.sum() - dp)

    return tani

def calc_mfp_matrix(compounds: dict):
    '''
    Calculate Morgan fingerprint matrix (n_mols x mfp_len)
    '''
    mfp_len = 1024
    mfper = MorganFingerPrinter(length=1024)
    mfps = [np.zeros(shape=(mfp_len,)) for _ in range(max(compounds.keys()) + 1)]
    for k in compounds.keys():
        mol = Chem.MolFromSmiles(compounds[k]['smiles'])
        mfps[k] = mfper.fingerprint(mol)

    return np.vstack(mfps)

def pathwise_sim_to_target(args):
    '''
    Computes similarity of intermediates along paths to targets
    and similarity of intermediates' successors to targets
    Returns
    -------
    path_sims:
    successor_sims:
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

    cpds, smi2id = extract_compounds(rxns) # Get known compounds

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

    print("Calculating MFPs and Tanis")
    M = calc_mfp_matrix(cpds) # Morgan fingerprints
    norms = np.sum(M, axis=1).reshape(-1, 1)
    S = np.matmul(M, M.T)
    T = S / (norms + norms.T - S) # Tanimoto sim mat

    st_generator = yield_path(paths, G, T, M, downsample=args.downsample)
    
    if args.similarity == 'tanimoto':
        f = get_tani_sims

    print("Processing paths")
    res = process_map(f, st_generator, chunksize=10, total=int(n_paths / args.downsample))

    path_sims = defaultdict(dict)
    successor_sims = defaultdict(dict)
    for elt in res:
        st, ps, ss = elt
        i, j = st
        path_sims[i][j] = ps
        successor_sims[i][j] = ss

    save_dir = filepaths['results'] / "path_structural_variation" / f"{args.rxns}"
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    
    with open(
        save_dir / f"path_similarities_to_target_{args.similarity}_{args.whitelist}_atom_lb_{int(args.atom_lb * 100)}p_multi_nodes_{args.multi_nodes}_ds_{args.downsample}.json",
        'w') as f:
        json.dump(path_sims, f)

    with open(
        save_dir / f"successor_similarities_to_target_{args.similarity}_{args.whitelist}_atom_lb_{int(args.atom_lb * 100)}p_multi_nodes_{args.multi_nodes}_ds_{args.downsample}.json",
        'w') as f:
        json.dump(successor_sims, f)

def spearman(path_tanis: list):
    x = np.arange(len(path_tanis))
    return spearmanr(x, path_tanis).statistic

def calc_path_spearmans(args):
    '''
    Calculate spearmans coefficient between path similarities and 
    reaction step
    '''
    for file in args.files:
        full_path = filepaths['results'] / "path_structural_variation" / f"{file}.json"
        with open(full_path, 'r') as f:
            path_tanis = str2int(json.load(f))
            
        tani_generator = (path_tanis[i][j] for i in path_tanis for j in path_tanis[i])
        print("Calculating spearman rs")
        res = process_map(spearman, tani_generator, chunksize=10)
        
        path_spearmans = defaultdict(dict)
        idx_generator = ((i, j) for i in path_tanis for j in path_tanis[i])
        for n, (i, j) in enumerate(idx_generator):
            path_spearmans[i][j] = res[n]

        fn = f"path_spearmans{full_path.stem.removeprefix('path_similarities_to_target')}.json"
        save_path = full_path.parent / fn
        with open(save_path, 'w') as f:
            json.dump(path_spearmans, f)
        

def calc_topk_rates(args):
    '''
    Calculate fraction of time shortest path proceeds to top k most 
    similar intermediate to target
    '''
    for fp1, fp2 in zip(args.path_sims, args.successor_sims):
        full_path1 = filepaths['results'] / "path_structural_variation" / f"{fp1}.json"
        with open(full_path1, 'r') as f:
            path_tanis = str2int(json.load(f))

        full_path2 = filepaths['results'] / "path_structural_variation" / f"{fp2}.json"
        with open(full_path2, 'r') as f:
            successor_tanis = str2int(json.load(f))

        decision_generator = yield_decisions(path_tanis, successor_tanis, args.topks)
        print("Counting topk")
        res = process_map(count_topk, decision_generator, chunksize=10)

        in_top_k = {k: 0 for k in args.topks}
        counts = {k: 0 for k in args.topks}
        for k, flag in res:
            in_top_k[k] += flag
            counts[k] += 1

        fracs = {k: in_top_k[k] / counts[k] if counts[k] > 0 else 0 for k in args.topks}

        fn = f"topk{full_path1.stem.removeprefix('path_similarities_to_target')}.json"
        with open(full_path1.parent / fn, 'w') as f:
            json.dump(fracs, f)


parser = ArgumentParser(description="Connects substrates in a reaction to construct a reaction network")
subparsers = parser.add_subparsers(title="Commands", description="Available commands")

# Pathwise similarity calculations
parser_pst = subparsers.add_parser("pst", help="Computes pathwise similarity to target for shortest paths in a provided graph")
parser_pst.add_argument("similarity", help="Which similarity measure to use, e.g., tanimoto")
parser_pst.add_argument("whitelist", help="Filename of coreactants whitelisted as currency")
parser_pst.add_argument("atom_lb", type=float, help="Below this fraction of heavy atoms, reactants will be ignored, i.e., not connected to products")
parser_pst.add_argument("--op-cxns", default="sprhea_240310_v3_mapped_operator", help="Filename operator connected adjacency matrices")
parser_pst.add_argument("--rxns", default="sprhea_240310_v3_mapped", help="Reactions dataset")
parser_pst.add_argument("--sim-cxns", default="sprhea_240310_v3_mapped_similarity", help="Filename similarity connected adjacency matrices")
parser_pst.add_argument("--side-cts", default="sprhea_240310_v3_mapped_side_counts", help="Counts of unique, non-currency molecules on each side of reactions")
parser_pst.add_argument("--downsample", type=int, default=10, help="Downsample path sources (starters) by this factor")
parser_pst.add_argument("--connect-nontrivial", action="store_true", help="Connect nontrivial reactions w/ similarity connections")
parser_pst.add_argument("--multi-nodes", action="store_true", help="Add multiple molecule nodes to network")
parser_pst.set_defaults(func=pathwise_sim_to_target)

# Pathwise spearman
parser_pst = subparsers.add_parser("spearman", help="Computes pathwise spearman corelation coefficient")
parser_pst.add_argument("files", nargs="+", help="Paths to files relative to path structural variability dir")
parser_pst.set_defaults(func=calc_path_spearmans)

# Top k choices
parser_pst = subparsers.add_parser("topk", help="Counts decisions in topk similarity to target")
parser_pst.add_argument("--path-sims", nargs="+", help="Paths to path sim files relative to path structural variability dir")
parser_pst.add_argument("--successor-sims", nargs="+", help="Paths to successor sim files relative to path structural variability dir")
parser_pst.add_argument("--topks", nargs="+", type=int, default=[1, 2, 5, 10], help="Top ks")
parser_pst.set_defaults(func=calc_topk_rates)

def main():
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()