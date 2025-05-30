import rdkit
from rdkit import Chem
from rdkit.Chem import rdFMCS, rdFingerprintGenerator, BRICS, Draw, Mol, AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdChemReactions import ChemicalReaction
from itertools import product, chain
from collections import Counter
import re
import pandas
import numpy as np
from typing import Iterable

def draw_molecule(mol: str | Mol, size: tuple = (200, 200), use_svg: bool = True):
    '''
    Draw molecule.

    Args
    ----
    mol:str | Mol
        Molecule
    size:tuple
        (width, height)
    use_svg:bool
    '''
    if type(mol) is str:
        mol = Chem.MolFromSmiles(mol)

    if use_svg:
        drawer = Draw.MolDraw2DSVG(*size)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        img = drawer.GetDrawingText()
    else:
        img = Draw.MolToImage(mol, size=size)

    return img

def draw_reaction(rxn: str | ChemicalReaction, sub_img_size: tuple = (200, 200), use_svg: bool = True, use_smiles: bool = True):
    '''
    Draw reaction.

    Args
    ----
    rxn:str | ChemicalReaction
    sub_img_size:tuple
        Substrate img size
    use_svg:bool
    use_smiles:bool
    '''
    if type(rxn) is str:
        rxn = Chem.rdChemReactions.ReactionFromSmarts(rxn, useSmiles=use_smiles)

    return Draw.ReactionToImage(rxn, useSVG=use_svg, subImgSize=sub_img_size)

def tautomer_expand(molecule: Mol | str, k: int) -> list[Mol] | list[str]:
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
        self.length = length
        self.radius = radius
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

def brics_assign(lhs: list[rdkit.Chem.rdchem.Mol], rhs: list[rdkit.Chem.rdchem.Mol]) -> np.ndarray:
    '''
    Determines which reactant-product pairs from a reaction share a 
    molecular fragment unique to them and therefore must have transfered
    this mass

    Args
    -----
    lhs, rhs: list[rdkit.Chem.rdchem.Mol]
        Reactants and products, respectively

    Returns
    --------
    link_mat: ndarray
        Binary matrix indicating unique fragment links
    '''
    def format_fragment(smi: str):
        patt = r'\d+#\d+'
        smarts = Chem.MolToSmarts(Chem.MolFromSmiles(smi))
        smarts = re.sub(patt, '*', smarts)
        return Chem.MolFromSmarts(smarts)

    def decompose_and_format(mol: rdkit.Chem.rdchem.Mol):
        frags = BRICS.BRICSDecompose(mol)
        return [format_fragment(elt) for elt in frags]
    
    def find_fragment_intersection(
            frags1: list[rdkit.Chem.rdchem.Mol],
            frags2: list[rdkit.Chem.rdchem.Mol]
        ) -> list[tuple[rdkit.Chem.rdchem.Mol]]:
        '''
        Fragments declared "intersecting" if they have the same number of atoms
        and one SubstructMatches the other
        '''
        intersection = []
        for f1, f2 in product(frags1, frags2):
            if f1.GetNumAtoms() != f2.GetNumAtoms():
                continue
            
            try:
                if f1.HasSubstructMatch(f2) or f2.HasSubstructMatch(f1):
                    intersection.append((f1, f2))
            except:
                continue

        return intersection

    left_idxs = [i for i in range(len(lhs))]
    right_idxs = [i for i in range(len(rhs))]

    left_frags = [decompose_and_format(elt) for elt in lhs]
    right_frags = [decompose_and_format(elt) for elt in rhs]

    link_mat = np.zeros(shape=(len(lhs), len(rhs)))
    for i, j in product(left_idxs, right_idxs):
        lf = left_frags[i]
        rf = right_frags[j]
        lf_complement = list(chain(*[left_frags[idx] for idx in left_idxs if idx != i]))
        rf_complement = list(chain(*[right_frags[idx] for idx in right_idxs if idx != j]))

        this_intersection = find_fragment_intersection(lf, rf) # All matching frag pairs for this rct-pdt pair
        if not this_intersection:
            continue
        
        # Check that the match is unique, i.e., this rct and pdt could only have gotten frag
        # from the other, and not from any other pdts or rcts, respectively
        for pair in this_intersection:
            left_right_complement_intersection = find_fragment_intersection([pair[0]], rf_complement)
            right_left_complement_intersection = find_fragment_intersection([pair[1]], lf_complement)

        if not left_right_complement_intersection and not right_left_complement_intersection:
            link_mat[i, j] = 1

    return link_mat

def post_standardize(mol, do_canon_taut):
    '''
    Standardize molecules after generating them with an operator in rdkit
        - Skip neutralization because assume operators only affect heavy atoms, not hydrogens and therefore
        protonation states
        - Skip find parent because assume I am not producing salts / fragments (TODO: pressure test this
        assumption)
    -
    '''
    do_neutralize = False
    do_find_parent = False
    
    return Chem.MolToSmiles(standardize_mol(mol, do_canon_taut=do_canon_taut, do_neutralize=do_neutralize, do_find_parent=do_find_parent))

def standardize_mol(mol, **kwargs):
    kwargs = _handle_kwargs(**kwargs)

    if kwargs['do_remove_stereo']:
        Chem.rdmolops.RemoveStereochemistry(mol)

    # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
    # Also checks valency, that mol is kekulizable
    mol = rdMolStandardize.Cleanup(mol)

    # if many fragments, get the "parent" (the actual mol we are interested in)
    if kwargs['do_find_parent']:
        mol = rdMolStandardize.FragmentParent(mol)

    if kwargs['do_neutralize']:
        mol = neutralize_charges(mol) # Remove charges on atoms matching common patterns

    # Enumerate tautomers and choose canonical one
    if kwargs['do_canon_taut']:
        te = rdMolStandardize.TautomerEnumerator()
        te.SetMaxTautomers(kwargs['max_tautomers'])
        mol = te.Canonicalize(mol)
    
    return mol

def standardize_smiles(smiles, **kwargs):
    kwargs = _handle_kwargs(**kwargs)
    mol = Chem.MolFromSmiles(smiles)
    mol = standardize_mol(
        mol,
        **kwargs
    )
    return Chem.MolToSmiles(mol)

def _handle_kwargs(**kwargs):
    default_kwargs = {
        'do_canon_taut':False,
        'do_neutralize':True,
        'do_find_parent':True,
        'do_remove_stereo':True,
        'max_tautomers':50,
    }
    filtered_kwargs = {k : v for k, v in kwargs.items() if k in default_kwargs}
    default_kwargs.update(filtered_kwargs)
    return default_kwargs

def neutralize_charges(mol: Chem.rdchem.Mol) -> Chem.rdchem.Mol:
    """Neutralize all charges in an rdkit mol.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule to neutralize.

    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol
        Neutralized molecule.
    """
    patts = (
        ("[n+;H]", "n"), # Imidazoles
        ("[N+;!H0]", "N"), # Amines
        ("[$([O-]);!$([O-][#7])]", "O"), # Carboxylic acids and alcohols
        ("[S-;X1]", "S"), # Thiols
        ("[$([N-;X2]S(=O)=O)]", "N"), # Sulfonamides
        ("[$([N-;X2][C,N]=C)]", "N"), # Enamines
        ("[n-]", "[nH]"), # Tetrazoles
        ("[$([S-]=O)]", "S"), # Sulfoxides
        ("[$([N-]C=O)]", "N"), # Amides
    )

    reactions = [
        (AllChem.MolFromSmarts(x), AllChem.MolFromSmiles(y, False)) for x,y in patts
    ]

    for (reactant, product) in reactions:
        while mol.HasSubstructMatch(reactant):
            rms = AllChem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]
    return mol

def count_elements(molecule: str | Chem.Mol):
    # Check if molecule is a SMILES string or a Mol object
    if isinstance(molecule, str):
        mol = Chem.MolFromSmiles(molecule)
        if mol is None:
            raise ValueError("Invalid SMILES string.")
    elif isinstance(molecule, Chem.Mol):
        mol = molecule
    else:
        raise TypeError("Input must be a SMILES string or an RDKit Mol object.")

    # Initialize a counter for elements
    element_counter = Counter()

    # Count each element in the molecule
    for atom in mol.GetAtoms():
        element = atom.GetSymbol()
        element_counter[element] += 1

    # Return the counts as a dictionary
    return dict(element_counter)

def calc_mfp_matrix(compounds: dict[int, str], dtype: np.dtype = np.int8):
    '''
    Calculate Morgan fingerprint matrix (n_mols x mfp_len)

    Args
    ----
    compounds: dict[int, str]
        Node ID to SMILES

    Returns
    --------
    nxd Morgan fingerprint embedding matrix
    '''
    mfper = MorganFingerPrinter()
    mfps = [np.zeros(shape=(mfper.length,)) for _ in range(max(compounds.keys()) + 1)]
    for k in compounds.keys():
        mol = Chem.MolFromSmiles(compounds[k])
        mfps[k] = mfper.fingerprint(mol)

    return np.vstack(mfps).astype(dtype)

if __name__ == '__main__':
    smiles = "CCO"  # Ethanol
    result = count_elements(smiles)
    print(result)  # Output: {'C': 2, 'O': 1, 'H': 6}

    mfper = MorganFingerPrinter()
    mfp = mfper.fingerprint(Chem.MolFromSmiles(smiles)).reshape(1, -1)
    print(globals()['mfp'])

