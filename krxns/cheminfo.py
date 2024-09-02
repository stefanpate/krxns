import rdkit
from rdkit import Chem
from rdkit.Chem import rdFMCS, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from itertools import product
import pandas
import numpy as np
from typing import Iterable

def tautomer_expand(molecule: rdkit.Chem.rdchem.Mol | str, k: int) -> list[rdkit.Chem.rdchem.Mol] | list[str]:
    '''
    Returns top k tautomers of given molecule in the
    format provided.

    Args
    ----
    molecule:Mol | str
        molecule as Mol obj or SMILES string
    k:int
        Number of tautomers to return

    Returns
    --------
    tautomers:List[Mol] | List[str]
    '''
    # TODO: Figure out how to do this w/o enumeratring 
    # twice (once thru canonicalize and once with enumerate)
    # Need to figure out how ties are broken... something to do 
    # with the canonical smiles...#
    if type(molecule) is str:
        molecule = Chem.MolFromSmiles(molecule)
        mode = 'str'
    elif type(molecule) is Chem.Mol:
        mode = 'mol'
    
    enumerator = rdMolStandardize.TautomerEnumerator()
    canon_mol = enumerator.Canonicalize(molecule)
    canon_smi = Chem.MolToSmiles(canon_mol)
    tauts = enumerator.Enumerate(molecule)
    smis = []
    mols = []
    for mol in tauts:
        smi = Chem.MolToSmiles(mol)
        if smi != canon_smi:
            smis.append(smi)
            mols.append(mol)

    if not smis and not mols:
        smis = [canon_smi]
        mols = [canon_mol]
    else:
        srt = sorted(list(zip(smis, mols)), key=lambda x : enumerator.ScoreTautomer(x[1]), reverse=True)
        smis, mols = [list(elt) for elt in zip(*srt)]
        smis = [canon_smi] + smis
        mols = [canon_mol] + mols

    if mode == 'str':
        return smis[:k]
    elif mode == 'mol':
        return mols[:k]
    
def expand_paired_cofactors(df: pandas.DataFrame, k: int) -> dict:
    '''
    Return k top canonical tautomers, expanded from paired cofactor
    reference file
    '''

    smi2name = {}
    for _, row in df.iterrows():
        smi_exp_1 = tautomer_expand(row["Smiles 1"], k)
        smi_exp_2 = tautomer_expand(row["Smiles 2"], k)
        for combo in product(smi_exp_1, smi_exp_2):
            smi2name[combo] = (row["Name 1"], row["Name 2"])

    return smi2name

def expand_unpaired_cofactors(df: pandas.DataFrame, k: int) -> dict:
    '''
    Return k top canonical tautomers, expanded from unpaired cofactor
    reference file
    '''
    
    smi2name = {}
    for _, row in df.iterrows():
        smi_exp = tautomer_expand(row["Smiles"], k)
        for smi in smi_exp:
            smi2name[smi] = row["Name"]

    return smi2name

class MorganFingerPrinter:
    def __init__(self, radius: int = 2, length: int = 2048) -> None:
        self._generator = rdFingerprintGenerator.GetMorganGenerator(radius= radius, fpSize=length)

    def fingerprint(self, mol: rdkit.Chem.rdchem.Mol) -> np.ndarray:
        return np.array(self._generator.GetFingerprint(mol))
    
def tanimoto_similarity(bit_vec_1: np.ndarray, bit_vec_2: np.ndarray, dtype=np.float32):
    dot = np.dot(bit_vec_1, bit_vec_2)
    return dtype(dot / (bit_vec_1.sum() + bit_vec_2.sum() - dot))

def mcs(mols: Iterable[rdkit.Chem.rdchem.Mol], norm: str = 'max', dtype=np.float32):
    '''
    Returns MCS similarity score between molecules by findings
    MCS & dividing # MCS atoms by # atoms in the largest molecule
    if norm == 'max' or smallest molecule if norm == 'min'
    '''
    res = rdFMCS.FindMCS(
        mols,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        matchChiralTag=False,
        ringMatchesRingOnly=True,
        completeRingsOnly=False,
        matchValences=True,
        timeout=10
    )

    # Compute mcs similarity score
    if res.canceled:
        full = 0
    elif norm == 'min':
        full = res.numAtoms / min(m.GetNumAtoms() for m in mols)
    elif norm == 'max':
        full = res.numAtoms / max(m.GetNumAtoms() for m in mols)

    return dtype(full)