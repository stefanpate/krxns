import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import hydra
from omegaconf import DictConfig
from rdkit import Chem
from collections import defaultdict
from ergochemics.standardize import standardize_smiles
from functools import lru_cache

@lru_cache(maxsize=10000)
def std_smi(smi: str) -> str:
    return standardize_smiles(
        smiles=smi,
        do_canon_taut=True,
        neutralization_method="simple",
        quiet=True,
        max_tautomers=100,
    )

def update_cpd_id_rxn(tmp_id_rxn: tuple[list[int], list[int]], tmp_id_to_cpd_id: dict[int, int]) -> tuple[list[int], list[int]]:
    '''
    Update compound ids in a reaction from temporary ids to compound ids.

    Args
    ----
    tmp_id_rxn:tuple[list[int], list[int]]
        Tuple containing two lists: reactant compound ids and product compound ids.
        Must be in the same order as atom mapped reaction.
    tmp_id_to_cpd_id:dict[int, int]
        Mapping from temporary compound ids to compound ids.
    
    Returns
    -------
    cpdid_rxn:tuple[list[int], list[int]]
        Tuple containing two lists: reactant compound ids and product compound ids,
        updated to use compound ids.
    '''
    return (
        [tmp_id_to_cpd_id[i] for i in tmp_id_rxn[0]],
        [tmp_id_to_cpd_id[i] for i in tmp_id_rxn[1]]
    )

def get_mass_contributions(am_rxn: str, cpdid_rxn: tuple[list[int], list[int]]) -> dict[str, dict[int, dict[int, float]]]:
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
    dict[str, dict[int, dict[int, float]]]
        With differently normalized mass contributions:
        {
            "rct_normed_mass_contrib": {
                pdt_id: {
                    rct_id: (atoms rct -> pdt) / tot_rct_atoms
                }
            },
            "pdt_normed_mass_contrib": {
                pdt_id: {
                    rct_id: (atoms rct -> pdt) / tot_pdt_atoms
                }
            }
        }
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
    atom_counts = {i_id: {j_id: 0.0 for j_id in cpdid_rxn[0]} for i_id in cpdid_rxn[1]}
    for amn in amns:
        rct_id = cpdid_rxn[0][amn_to_rct_idx[amn]]
        pdt_id = cpdid_rxn[1][amn_to_pdt_idx[amn]]
        atom_counts[pdt_id][rct_id] += 1.0

    # Collect rct n atoms to normalize mass contributions
    # in one returned dict
    rct_id_to_n_atoms = {}
    for rct, rct_id in zip(rcts, cpdid_rxn[0]):
        rct_id_to_n_atoms[rct_id] = rct.GetNumAtoms()
    
    # Normalize by number of atoms in reactant / product
    rct_normed_mass_contrib = {}
    pdt_normed_mass_contrib = {}
    for pdt_id, rct_dict in atom_counts.items():
        rct_normed_mass_contrib[pdt_id] = {}
        pdt_normed_mass_contrib[pdt_id] = {}
        tot_atoms = sum(rct_dict.values()) # Total atoms in product
        for rct_id, count in rct_dict.items():
            rct_normed_mass_contrib[pdt_id][rct_id] = count / rct_id_to_n_atoms[rct_id]
            pdt_normed_mass_contrib[pdt_id][rct_id] = count / tot_atoms

    return {"rct_normed_mass_contrib": rct_normed_mass_contrib, "pdt_normed_mass_contrib": pdt_normed_mass_contrib}
    
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

    # Extract compounds from reactions
    compounds = defaultdict(set)
    smi_to_tmp_id = {}
    mapped_rxns["cpd_id_rxn"] = None
    for idx, row in tqdm(mapped_rxns.iterrows(), total=len(mapped_rxns), desc="Extracting and standardizing compounds"):
        cpd_id_rxn = [[], []]
        for i, side in enumerate(row['smarts'].split('>>')):
            for smi in side.split('.'):
                smi = std_smi(smi)
                
                if smi not in smi_to_tmp_id:
                    smi_to_tmp_id[smi] = len(smi_to_tmp_id)

                cpd_id_rxn[i].append(smi_to_tmp_id[smi])
                compounds[smi].add(row['rxn_id'])
        
        mapped_rxns.at[idx, 'cpd_id_rxn'] = tuple(cpd_id_rxn)

    compounds = {k: len(v) for k, v in compounds.items()}
    compounds = sorted(compounds.items())
    smiles, rxn_counts = zip(*compounds)
    compounds_df = pd.DataFrame(
        {
            'id': range(len(compounds)),
            'smiles': smiles,
            'name': [smi2name.get(smi, '') for smi in smiles],
            'rxn_count': rxn_counts
        }
    )
    compounds_df["n_atoms"] = compounds_df['smiles'].apply(
        lambda smi: Chem.MolFromSmiles(smi).GetNumAtoms()
    )

    compounds_df.to_csv(
        Path(cfg.filepaths.interim_data) / "compounds.csv",
        index=False
    )

    # Save default set of sources
    sources = compounds_df[compounds_df["name"].isin(cfg.sources.source_names)]
    sources.to_csv(
        Path(cfg.filepaths.interim_data) / "default_sources.csv",
        index=False
    )

    # Update ids in cpd id reaction to lexicographically sorted smiles ones
    smi_to_cpid = dict(zip(compounds_df['smiles'], compounds_df['id']))
    tmp_id_to_cpd_id = {v: smi_to_cpid[k] for k, v in smi_to_tmp_id.items()}
    mapped_rxns["cpd_id_rxn"] = mapped_rxns["cpd_id_rxn"].apply(
        lambda x: update_cpd_id_rxn(x, tmp_id_to_cpd_id)
    )

    # Get mass weighted inlink dicts 
    mass_contributions = {}
    for _, row in tqdm(mapped_rxns.iterrows(), total=len(mapped_rxns), desc="Calculating mass contributions"):
        am_rxn = row['am_smarts']
        cpdid_rxn = row['cpd_id_rxn']
        atom_counts = get_mass_contributions(am_rxn, cpdid_rxn)
        atom_counts["am_smarts"] = am_rxn # Also sneak am smarts in there for convenience
        mass_contributions[row['rxn_id']] = atom_counts

    with open(Path(cfg.filepaths.interim_data) / "mass_contributions.json", 'w') as f:
        json.dump(mass_contributions, f)        

if __name__ == '__main__':
    main()