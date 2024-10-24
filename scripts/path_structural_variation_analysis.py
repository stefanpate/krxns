from argparse import ArgumentParser
from krxns.config import filepaths
from krxns.utils import str2int
from krxns.networks import SuperMultiDiGraph
from krxns.net_construction import construct_reaction_network, extract_compounds
import json
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict
from tqdm import tqdm

def calc_mfp_matrix():
    '''
    Calculate Morgan fingerprint matrix (n_mols x mfp_len)
    '''

def pathwise_sim_to_target():
    '''
    Computes similarity of intermediates along paths to targets
    and similarity of intermediates' successors to targets
    Returns
    -------
    path_sims:
    successor_sims:
    '''

def calc_path_spearmans(args):
    '''
    Calculate spearmans coefficient between path similarities and 
    reaction step
    '''

    # Load known reaction data
    with open(filepaths['data'] / args.rxns, 'r') as f:
        rxns = json.load(f)

    # Load op connected reactions
    with open(filepaths['connected_reactions'] / args.op_cxns, 'r') as f:
        op_cxns = str2int(json.load(f))

    # Load sim connected reactions
    with open(filepaths['connected_reactions'] / args.sim_cxns, 'r') as f:
        sim_cxns = str2int(json.load(f))

    with open(filepaths['connected_reactions'] / args.side_counts, 'r') as f:
        side_counts = str2int(json.load(f))

    # Load coreactant whitelist
    with open(filepaths['coreactants'] / args.whitelist, 'r') as f:
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

    paths = G.shortest_path() # Get shortest paths

    # Filter out self-paths
    tmp = {}
    for i in paths:
        destinations = {j: elt for j, elt in paths[i].items() if i != j}
        if destinations:
            tmp[i] = destinations
    paths = tmp

def calc_topk_rates(args):
    '''
    Calculate fraction of time shortest path proceeds to top k most 
    similar intermediate to target
    '''

parser = ArgumentParser(description="Connects substrates in a reaction to construct a reaction network")
subparsers = parser.add_subparsers(title="Commands", description="Available commands")

# TODO
parser_pst = subparsers.add_parser("pst", help="Computes pathwise similarity to target for shortest paths in a provided graph")
parser_pst.add_argument("similarity", help="Which similarity measure to use, e.g., tanimoto")
parser_pst.add_argument("whitelist", help="Filename of coreactants whitelisted as currency")
parser_pst.add_argument("atom_lb", type=float, help="Below this fraction of heavy atoms, reactants will be ignored, i.e., not connected to products")
parser_pst.add_argument("--op-cxns", default="sprhea_240310_v3_mapped_operator", help="Filename operator connected adjacency matrices")
parser_pst.add_argument("--rxns", default="sprhea_240310_v3_mapped", help="Reactions dataset")
parser_pst.add_argument("--sim-cxns", default="sprhea_240310_v3_mapped_similarity", help="Filename similarity connected adjacency matrices")
parser_pst.add_argument("--side-cts", default="sprhea_240310_v3_mapped_side_counts", help="Counts of unique, non-currency molecules on each side of reactions")
parser_pst.add_argument("--connect-nontrivial", action="store_true", help="Connect nontrivial reactions w/ similarity connections")
parser_pst.add_argument("--multi-nodes", action="store_true", help="Add multiple molecule nodes to network")
parser_pst.set_defaults(func=pathwise_sim_to_target)

def main():
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()