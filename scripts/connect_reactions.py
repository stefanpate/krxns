import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import hydra
from omegaconf import DictConfig
from rdkit import Chem

def mass_balance(am_rxn: str) -> tuple[dict[int, dict[int, float]], dict[int, dict[int, float]]]:
    '''
    Returns fraction of atoms in a reactant / product coming from a product / reactant, respectivley.

    Args
    ----
    am_rxn:str
        Atom-mapped reaction string in the form of "R1.R2.R3>>P1.P2.P3"

    Returns
    -------
    rct_inlinks:dict
        {rct_idx: {pdt_idx: fraction_atoms_from_pdt_idx}, }
    pdt_inlinks:dict
        {pdt_idx: {rct_idx: fraction_atoms_from_rct_idx}, }
    '''
    rcts, pdts = [
        [Chem.MolFromSmiles(mol) for mol in side.split('.')]
        for side in am_rxn.split('>>')
    ]
    n_atoms_rcts = [mol.GetNumAtoms() for mol in rcts]
    n_atoms_pdts = [mol.GetNumAtoms() for mol in pdts]
    
    # Collect atom map numbers to rct / pdt indices
    amn_to_rct_idx = {}
    amn_to_pdt_idx = {}
    _amns = []
    amns_ = []
    for rct_idx, rct in enumerate(rcts):
        for atom in rct.GetAtoms():
            amn = atom.GetAtomMapNum()
            
            if amn == 0:
                raise ValueError("Atom map numbers must be non-zero.")

            amn_to_rct_idx[amn] = rct_idx
            _amns.append(amn)
    
    for pdt_idx, pdt in enumerate(pdts):
        for atom in pdt.GetAtoms():
            amn = atom.GetAtomMapNum()

            if amn == 0:
                raise ValueError("Atom map numbers must be non-zero.")
            
            amn_to_pdt_idx[amn] = pdt_idx
            amns_.append(amn)

    # Check atom map nums are 1-to-1
    amns = set(_amns) & set(amns_)
    if len(amns) != len(_amns) or len(amns) != len(amns_):
        raise ValueError("Atom map numbers are not 1-to-1 between reactants and products.")

    # Count atoms received by molecule i from molecule j
    rct_inlinks = {i: {j: 0.0 for j in range(len(pdts))} for i in range(len(rcts))}
    pdt_inlinks = {i: {j: 0.0 for j in range(len(rcts))} for i in range(len(pdts))}
    for amn in amns:
        rct_idx = amn_to_rct_idx[amn]
        pdt_idx = amn_to_pdt_idx[amn]
        rct_inlinks[rct_idx][pdt_idx] += 1.0
        pdt_inlinks[pdt_idx][rct_idx] += 1.0

    # Normalize by number of atoms in reactants / products
    for rct_idx, pdt_dict in rct_inlinks.items():
        for pdt_idx, count in pdt_dict.items():
            rct_inlinks[rct_idx][pdt_idx] = count / n_atoms_rcts[rct_idx]
    
    for pdt_idx, rct_dict in pdt_inlinks.items():
        for rct_idx, count in rct_dict.items():
            pdt_inlinks[pdt_idx][rct_idx] = count / n_atoms_pdts[pdt_idx]

    return rct_inlinks, pdt_inlinks
    
@hydra.main(version_base=None, config_path="../configs", config_name="connect_reactions")
def main(cfg: DictConfig):
    rc_0_mapped = pd.read_parquet(
        Path(cfg.filepaths.raw_data) / cfg.rc_plus_0_mapped
    )

    mechinformed_mapped = pd.read_parquet(
        Path(cfg.filepaths.raw_data) / cfg.mechinformed_mapped
    )

    # Prefer the mechinformed atom mapping to the rc_plus_0
    overlap = rc_0_mapped.rxn_id.isin(mechinformed_mapped.rxn_id)
    mapped_rxns = pd.concat([mechinformed_mapped, rc_0_mapped[~overlap]], ignore_index=True)

    mass_links = {}
    for _, row in tqdm(mapped_rxns.iterrows(), total=len(mapped_rxns)):
        am_rxn = row['am_smarts']
        rct_inlinks, pdt_inlinks = mass_balance(am_rxn)
        mass_links[row['rxn_id']] = {
            'rct_inlinks': rct_inlinks,
            'pdt_inlinks': pdt_inlinks
        }

    with open(Path(cfg.filepaths.interim_data) / "mass_links.json", 'w') as f:
        json.dump(mass_links, f)        

if __name__ == '__main__':
    main()