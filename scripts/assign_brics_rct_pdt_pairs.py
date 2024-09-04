import numpy as np
import pandas as pd
import json
from itertools import chain
from rdkit import Chem
from krxns.cheminfo import expand_unpaired_cofactors, brics_assign
from krxns.config import data_filepath, cofactors_filepath, brics_sim_mats_filepath
from tqdm import tqdm

def main():
    # Load known reaction data
    with open(data_filepath / "sprhea_240310_v3_mapped.json", 'r') as f:
        known_reactions = json.load(f)

    known_reactions = {int(k): v for k,v in known_reactions.items()}

    # Extract known compounds
    known_compounds = {}
    for elt in known_reactions.values():
        subs = chain(*[side.split(".") for side in elt['smarts'].split(">>")])
        for sub in subs:
            known_compounds[sub] = elt['smi2name'].get(sub, None)

    known_compounds = {i: {'smiles': k, 'name': v} for i, (k, v) in enumerate(known_compounds.items())}
    smi2id = {v['smiles']: k for k,v in known_compounds.items()}

    # Remove reverses
    rids = set()
    for k, v in known_reactions.items():
        rids.add(tuple(sorted([k, v['reverse']])))

    keepers = [elt[0] for elt in rids]

    known_reactions = {k: known_reactions[k] for k in keepers}

    # Load cofactors
    k = 10
    unpaired_fp = cofactors_filepath / "unpaired_cofactors_reference.tsv"
    name_blacklist = [
        'acetyl-CoA',
        'CoA'
    ]

    unpaired_ref = pd.read_csv(
        filepath_or_buffer=unpaired_fp,
        sep='\t'
    )

    filtered_unpaired = unpaired_ref.loc[~unpaired_ref['Name'].isin(name_blacklist), :]
    cofactors = expand_unpaired_cofactors(filtered_unpaired, k=k)

    for rid, rxn in tqdm(known_reactions.items()):
        lhs, rhs = [set(side.split(".")) for side in rxn['smarts'].split(">>")] # Set out stoichiometric degeneracy
        lhs = [elt for elt in lhs if elt not in cofactors]
        rhs = [elt for elt in rhs if elt not in cofactors]

        if not lhs or not rhs:
            continue

        if len(lhs) > len(rhs):
            tmp = lhs
            lhs = rhs
            rhs = tmp

        lmols = [Chem.MolFromSmiles(elt) for elt in lhs]
        rmols = [Chem.MolFromSmiles(elt) for elt in rhs]

        left_ids, right_ids = np.array([smi2id[smi] for smi in lhs]), np.array([smi2id[smi] for smi in rhs])
        left_names, right_names = [known_compounds[id]['name'] for id in left_ids], [known_compounds[id]['name'] for id in right_ids]
        
        bm = brics_assign(lmols, rmols)
        df = pd.DataFrame(data=bm, columns=right_names, index=left_names)
        df.to_csv(brics_sim_mats_filepath / f"{rid}.csv", sep='\t')
        
if __name__ == '__main__':
    main()