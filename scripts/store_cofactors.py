from argparse import ArgumentParser
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from krxns.config import filepaths
from krxns.net_construction import SimilarityConnector

def _store_unpaired_cofactors(args):
    unpaired_ref = pd.read_csv(
        filepath_or_buffer= filepaths['coreactants'] / "unpaired_cofactors_reference.tsv",
        sep='\t'
    )

    filtered_unpaired = unpaired_ref.loc[~unpaired_ref['Name'].isin(args.blacklist), :]
    cofactors = {row['Smiles'] : row['Name'] for _, row in filtered_unpaired.iterrows()}

    with open(filepaths['coreactants'] / "manually_added_unpaired.json", 'r') as f:
        manual = json.load(f)

    cofactors = {**cofactors, ** manual}

    with open(filepaths['coreactants'] / "unpaired_cofactors.json", 'w') as f:
        json.dump(cofactors, f)

def _store_pickaxe_whitelist(args):
    with open(filepaths['coreactants'] / f"unpaired_cofactors.json", 'r') as f:
        unpaired_cofactors = json.load(f)
    
    paired_ref = pd.read_csv(
        filepath_or_buffer=filepaths['coreactants'] / "paired_cofactors_reference.tsv",
        sep='\t'
    )

    coreactant_whitelist = defaultdict(list)

    for i, row in paired_ref.iterrows():
        coreactant_whitelist[row['Smiles 1']].append(row['Smiles 2']) 
        coreactant_whitelist[row['Smiles 2']].append(row['Smiles 1'])

    for k in unpaired_cofactors.keys():
        coreactant_whitelist[k] = None

    with open(filepaths['coreactants'] / f"pickaxe_whitelist.json", "w") as f:
        json.dump(coreactant_whitelist, f)

def _store_topk_whitelist(args):
    # Load unpaired cofactors
    with open(filepaths['coreactants'] / f"unpaired_cofactors.json", 'r') as f:
        unpaired_cofactors = json.load(f)

    # Load known reaction data
    with open(filepaths['data'] / f"{args.reactions}.json", 'r') as f:
        known_reactions = json.load(f)

    # Load cc sim mats
    cc_sim_mats = {
        'mcs': np.load(filepaths['sim_mats'] / "mcs.npy"),
        'tanimoto': np.load(filepaths['sim_mats'] / "tanimoto.npy")
    }

    sc = SimilarityConnector(
        reactions=known_reactions,
        cc_sim_mats=cc_sim_mats,
        unpaired_cofactors=unpaired_cofactors,
        k_paired_cofactors=args.topk
    )

    coreactant_whitelist = defaultdict(list)
    for l, r in sc.paired_cofactors:
        lsmi, rsmi = sc.compounds[l]['smiles'], sc.compounds[r]['smiles']
        coreactant_whitelist[lsmi].append(rsmi)
        coreactant_whitelist[rsmi].append(lsmi)

    for k in unpaired_cofactors.keys():
        coreactant_whitelist[k] = None

    with open(filepaths['coreactants'] / f"top_{args.topk}_whitelist.json", "w") as f:
        json.dump(coreactant_whitelist, f)

parser = ArgumentParser(description="Store cofactor files for downstream use constructing and traversing reaction networks.")
subparsers = parser.add_subparsers(title="Commands", description="Available commands")

# Unpaired cofactors
parser_unpaired = subparsers.add_parser("unpaired", help="Store unpaired cofactors from pickaxe and manual adds")
parser_unpaired.add_argument("--blacklist", nargs="+", default=["acetyl-CoA"], help="Unpaired cofactors from reference to leave off the list")
parser_unpaired.set_defaults(func=_store_unpaired_cofactors)

# Whitelist pickaxe coreactants
parser_whitelist = subparsers.add_parser("whitelist-pickaxe", help="Store pickaxe coreactants as whitelist")
parser_whitelist.set_defaults(func=_store_pickaxe_whitelist)

# Whitelist top k most common paired cofactors in addition to unpaired cofactors
parser_topk_whitelist = subparsers.add_parser("whitelist-topk", help="Store topk coreactants from reaction dataset as whitelist")
parser_topk_whitelist.add_argument("topk", type=int, help="Topk highest Jaccard index paired cofactors")
parser_topk_whitelist.add_argument("reactions", help="Reaction dataset filename e.g., sprhea_240310_v3_mapped")
parser_topk_whitelist.set_defaults(func=_store_topk_whitelist)

if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)