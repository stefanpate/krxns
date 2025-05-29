import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import hydra
from omegaconf import DictConfig
from rdkit import Chem
import json

def compound_id_rxn(rxn: str, smi_to_cpid: dict[str, int]) -> tuple[list[int], list[int]]:
    '''
    Converts a reaction string to a tuple of lists of compound ids for reactants and products.

    Args
    ----
    rxn:str
        Reaction string in the form of "R1.R2.R3>>P1.P2.P3"

    smi_to_cpid:dict
        Dictionary mapping smiles to compound ids.

    Returns
    -------
    tuple[list[int], list[int]]
        Tuple containing two lists: reactant compound ids and product compound ids.
    '''
    rcts, pdts = rxn.split('>>')
    rct_ids = [smi_to_cpid[smi] for smi in rcts.split('.')]
    pdt_ids = [smi_to_cpid[smi] for smi in pdts.split('.')]
    
    return rct_ids, pdt_ids

def mass_balance(am_rxn: str, cpdid_rxn: tuple[list[int], list[int]]) -> dict[int, dict[int, float]]:
    '''
    Returns fraction of atoms in a reactant / product coming from a product / reactant, respectivley.

    Args
    ----
    am_rxn:str
        Atom-mapped reaction string in the form of "R1.R2.R3>>P1.P2.P3"
    cpdid_rxn:tuple[list[int], list[int]]
        Tuple containing two lists: reactant compound ids and product compound ids.
        Must be in same order as atom mapped reaction.
    
    Returns
    -------
    pdt_inlinks:dict
        {pdt_id: {rct_id: fraction_atoms_from_rct_id}, }
    '''
    rcts, pdts = [
        [Chem.MolFromSmiles(mol) for mol in side.split('.')]
        for side in am_rxn.split('>>')
    ]
    
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
    pdt_inlinks = {i_id: {j_id: 0.0 for j_id in cpdid_rxn[0]} for i_id in cpdid_rxn[1]}
    for amn in amns:
        rct_id = cpdid_rxn[0][amn_to_rct_idx[amn]]
        pdt_id = cpdid_rxn[1][amn_to_pdt_idx[amn]]
        pdt_inlinks[pdt_id][rct_id] += 1.0

    # Normalize by number of atoms in reactants / products
    for pdt_id, rct_dict in pdt_inlinks.items():
        tot_atoms = sum(rct_dict.values())
        for rct_id, count in rct_dict.items():
            pdt_inlinks[pdt_id][rct_id] = count / tot_atoms

    return pdt_inlinks
    
@hydra.main(version_base=None, config_path="../configs", config_name="process_reactions")
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

    # Get smi2name out of sprhea json
    with open(Path(cfg.filepaths.raw_data) / cfg.sprhea, 'r') as f:
        sprhea = json.load(f)

    smi2name = {smi: name for entry in sprhea.values() for smi, name in entry['smi2name'].items()}
    del sprhea

    # TODO: Standardize (incl tautomer form) all smiles in these reactions, preseriving the atom maping ans so on
    # TODO: Ultimately, handle cpd smiles canonicalization in pre-processing

    # Extract compounds from reactions
    compounds = set()
    for _, row in mapped_rxns.iterrows():
        for side in row['smarts'].split('>>'):
            for smi in side.split('.'):
                compounds.add(smi)

    compounds = sorted(compounds)
    compounds_df = pd.DataFrame({
        'id': range(len(compounds)),
        'smiles': compounds,
        'name': [smi2name.get(smi, '') for smi in compounds],
    })

    compounds_df.to_csv(
        Path(cfg.filepaths.interim_data) / "compounds.csv",
        index=False
    )

    # Add compound id reactions to enable canonical compound index in ultimate network
    smi_to_cpid = dict(zip(compounds_df['smiles'], compounds_df['id']))
    mapped_rxns["cpdid_rxn"] = mapped_rxns["smarts"].apply(
        lambda rxn: compound_id_rxn(rxn, smi_to_cpid)
    )

    # Get mass weighted inlink dicts 
    mass_links = {}
    for _, row in tqdm(mapped_rxns.iterrows(), total=len(mapped_rxns)):
        am_rxn = row['am_smarts']
        cpdid_rxn = row['cpdid_rxn']
        pdt_inlinks = mass_balance(am_rxn, cpdid_rxn)
        mass_links[row['rxn_id']] = pdt_inlinks

    with open(Path(cfg.filepaths.interim_data) / "mass_links.json", 'w') as f:
        json.dump(mass_links, f)        

if __name__ == '__main__':
    main()