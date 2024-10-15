from argparse import ArgumentParser
from krxns.cheminfo import expand_unpaired_cofactors, expand_paired_cofactors
from krxns.config import filepaths
import pandas as pd
import json
from itertools import chain

def _store_unpaired_cofactors(args):
    unpaired_ref = pd.read_csv(
        filepath_or_buffer= filepaths['cofactors'] / "unpaired_cofactors_reference.tsv",
        sep='\t'
    )

    filtered_unpaired = unpaired_ref.loc[~unpaired_ref['Name'].isin(args.blacklist), :]
    cofactors = expand_unpaired_cofactors(filtered_unpaired, k=args.topk)

    with open(filepaths["cofactors"] / "manually_added_unpaired.json", 'r') as f:
        manual = json.load(f)

    cofactors = {**cofactors, ** manual}

    with open(filepaths["cofactors"] / "expanded_unpaired_cofactors.json", 'w') as f:
        json.dump(cofactors, f)

def _store_coreactant_whitelist(args):
    with open(filepaths['cofactors'] / f"expanded_unpaired_cofactors.json", 'r') as f:
        unpaired_cofactors = json.load(f)
    
    paired_ref = pd.read_csv(
        filepath_or_buffer=filepaths['cofactors'] / "paired_cofactors_reference.tsv",
        sep='\t'
    )

    coreactant_whitelist ={}

    for i, row in paired_ref.iterrows():
        coreactant_whitelist[row['Smiles 1']] = row['Smiles 2']
        coreactant_whitelist[row['Smiles 2']] = row['Smiles 1']

    for k in unpaired_cofactors.keys():
        coreactant_whitelist[k] = None

    with open(filepaths['cofactors'] / f"{args.save_to}.json", "w") as f:
        json.dump(coreactant_whitelist, f)

parser = ArgumentParser(description="Store cofactor files for downstream use constructing and traversing reaction networks.")
subparsers = parser.add_subparsers(title="Commands", description="Available commands")

# Unpaired cofactors
parser_unpaired = subparsers.add_parser("unpaired", help="Tautomer expand and store unpaired cofactors")
parser_unpaired.add_argument("topk", type=int, help="Number of top scoring tautomers to keep")
parser_unpaired.add_argument("--blacklist", nargs="+", default=["acetyl-CoA"], help="Unpaired cofactors from reference to leave off the list")
parser_unpaired.set_defaults(func=_store_unpaired_cofactors)

# Whitelist coreactants
parser_whitelist = subparsers.add_parser("whitelist", help="Tautomer expand and store coreactant whitelist")
# parser_whitelist.add_argument("topk", type=int, help="Number of top scoring tautomers to keep")
parser_whitelist.add_argument("save_to", help="Save to file name")
parser_whitelist.set_defaults(func=_store_coreactant_whitelist)

if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)