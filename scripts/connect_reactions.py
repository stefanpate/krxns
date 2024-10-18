from argparse import ArgumentParser
from krxns.config import filepaths
from krxns.net_construction import connect_reaction_w_operator, SimilarityConnector, construct_op_atom_map_to_rct_idx
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

def _load_reactions(fn: str) -> dict:
    # Load known reaction data
    with open(filepaths['data'] / fn, 'r') as f:
        known_reactions = json.load(f)

    known_reactions = {int(k): v for k,v in known_reactions.items()}

    # Remove reverses
    rids = set()
    for k, v in known_reactions.items():
        rids.add(tuple(sorted([k, v['reverse']])))

    keepers = [elt[0] for elt in rids]
    known_reactions = {k: known_reactions[k] for k in keepers}

    return known_reactions

def _operator(args):
    known_reactions = _load_reactions(args.reactions)
    ops = pd.read_csv(
        filepath_or_buffer=filepaths['operators'] / "imt_ops.tsv",
        sep='\t'
    ).set_index("Name")
    op_atom_map_to_rct_idx = construct_op_atom_map_to_rct_idx(ops)

    results = defaultdict(lambda : defaultdict(dict))
    for rid, rxn in tqdm(known_reactions.items()):
        if not rxn['imt_rules']:
            continue

        for rule in rxn['imt_rules']:
            rsma = rxn["smarts"]
            op = ops.loc[rule, "SMARTS"]
            am2rci = op_atom_map_to_rct_idx[rule]
            rct_inlinks, pdt_inlinks = connect_reaction_w_operator(rsma, op, am2rci)
            if len(rct_inlinks) > 0 and len(pdt_inlinks) > 0:
                results[rid][rule] = {'rct_inlinks': rct_inlinks, 'pdt_inlinks':pdt_inlinks}

    with open(filepaths['connected_reactions'] / f"{Path(args.reactions).stem}_operator.json", 'w') as f:
        json.dump(results, f)

def _similarity(args):
    known_reactions = _load_reactions(args.reactions)

    # Load unpaired cofactors
    with open(filepaths['cofactors'] / 'unpaired_cofactors.json', 'r') as f:
        cofactors = json.load(f)

    # Load cc sim mats
    cc_sim_mats = {
        'mcs': np.load(filepaths['sim_mats'] / "mcs.npy"),
        'tanimoto': np.load(filepaths['sim_mats'] / "tanimoto.npy")
    }

    sc = SimilarityConnector(
        reactions=known_reactions,
        cc_sim_mats=cc_sim_mats,
        unpaired_cofactors=cofactors
    )

    results, side_counts = sc.connect_reactions()

    with open(filepaths['connected_reactions'] / f"{Path(args.reactions).stem}_similarity.json", 'w') as f:
        json.dump(results, f)

    with open(filepaths['connected_reactions'] / f"{Path(args.reactions).stem}_side_counts.json", 'w') as f:
        json.dump(side_counts, f)

parser = ArgumentParser(description="Connects substrates in a reaction to construct a reaction network")
subparsers = parser.add_subparsers(title="Commands", description="Available commands")

# Connect with operator
parser_rxn_embed = subparsers.add_parser("op-connect", help="Connects reaction substrates using operator and atom-mapping")
parser_rxn_embed.add_argument("reactions", help="Filename of reaction dataset e.g., sprhea_240310_v3_mapped.json")
parser_rxn_embed.set_defaults(func=_operator)

# Connect with similarity scores
parser_rxn_embed = subparsers.add_parser("sim-connect", help="Connects reaction substrates using similarity scores")
parser_rxn_embed.add_argument("reactions", help="Filename of reaction dataset e.g., sprhea_240310_v3_mapped.json")
parser_rxn_embed.set_defaults(func=_similarity)

def main():
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()