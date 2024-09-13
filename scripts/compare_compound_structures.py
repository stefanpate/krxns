import json
from itertools import chain
import multiprocessing as mp
from krxns.cheminfo import MorganFingerPrinter, tanimoto_similarity, mcs
from krxns.config import filepaths
from rdkit import Chem
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    dt = np.float16
    rxns_fn ="sprhea_240310_v3_mapped.json"
    chunksize = 50

    # Load known reaction data
    with open(filepaths['data'] / rxns_fn, 'r') as f:
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
    known_compounds = {i: known_compounds[i] for i in range(5)} # DOWNSAMPLE
    n = len(known_compounds)

    # Construct sim mats based on molecular structure
    mfper = MorganFingerPrinter()
    tani_sim_mat = np.zeros(shape=(len(known_compounds), len(known_compounds)))

    def tani_wrapper(mols):
        mfps = [mfper.fingerprint(elt) for elt in mols]
        return tanimoto_similarity(*mfps, dtype=dt)
    
    def mcs_wrapper(mols):
        return mcs(mols, norm='max', dtype=dt)
       
    comparators = {
        'tanimoto': tani_wrapper,
        'mcs': mcs_wrapper

    }

    sim_mats = {
        'tanimoto': np.eye(len(known_compounds), dtype=dt),
        'mcs': np.eye(len(known_compounds), dtype=dt)
    }

    def pool_fcn(mols):
        return {key: comp(mols) for key, comp in comparators.items()}
    
    def sim_mat_idx_generator(known_compounds):
        for i in range(len(known_compounds) - 1):
            for j in range(i+1, len(known_compounds)):
                yield (i, j)
    
    def mol_pair_generator(known_compounds):
        for i, j in sim_mat_idx_generator(known_compounds):
            yield tuple([Chem.MolFromSmiles(known_compounds[i]['smiles']),
                            Chem.MolFromSmiles(known_compounds[j]['smiles'])])

    with mp.Pool() as pool:
        res = list(tqdm(pool.imap(pool_fcn, mol_pair_generator(known_compounds), chunksize=chunksize), total=int((n**2 - n) / 2)))


    i, j = [np.array(elt) for elt in zip(*sim_mat_idx_generator(known_compounds))]
    for k, sm in sim_mats.items():
        elts = np.array([r[k] for r in res])
        sm[i, j] = elts
        sm[j, i] = elts

    for k, sm in sim_mats.items():
        np.save(filepaths['artifacts_sim_mats'] / k, sm)