from argparse import ArgumentParser
from krxns.config import filepaths
from krxns.utils import str2int
from krxns.networks import SuperMultiDiGraph
import json
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict
from tqdm import tqdm

def get_shortest_paths():
    '''
    Get shortest paths
    '''
    pass

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

def calc_topk_rates(args):
    '''
    Calculate fraction of time shortest path proceeds to top k most 
    similar intermediate to target
    '''

parser = ArgumentParser(description="Connects substrates in a reaction to construct a reaction network")
subparsers = parser.add_subparsers(title="Commands", description="Available commands")

# Connect with operator
parser_pst = subparsers.add_parser("pst", help="Computes pathwise similarity to target for shortest paths in a provided graph")
parser_pst.add_argument("reactions", help="Filename of reaction dataset e.g., sprhea_240310_v3_mapped.json")
parser_pst.set_defaults(func=_operator)

# Connect with similarity scores
parser_rxn_embed = subparsers.add_parser("sim-connect", help="Connects reaction substrates using similarity scores")
parser_rxn_embed.add_argument("reactions", help="Filename of reaction dataset e.g., sprhea_240310_v3_mapped.json")
parser_rxn_embed.set_defaults(func=_similarity)

def main():
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()