from rdkit import Chem
from rdkit.Chem import AllChem, Mol
from krxns.cheminfo import post_standardize
from itertools import permutations, chain, product
from collections import defaultdict
from typing import Iterable
import numpy as np
from scipy.stats import entropy

def connect_reaction_w_operator(reaction: str, operator: str, atom_map_to_rct_idx: dict) -> dict:
    '''
    Returns fraction of atoms in a reactant / product coming from a product / reactant, respectivley.
    Expects the order of the reactants and products to both be aligned with the operator.

    Args
    ----
    reaction: str
        Reaction SMARTS
    operator: str
        Operator SMARTS
    atom_map_to_rct_idx: dict
        Maps atom map numbers from LHS of operator to reactant idx on LHS of operator

    Returns
    ------
    rct_inlinks :dict
        {rct_idx: {pdt_idx: fraction_atoms_from_pdt_idx}, }
    pdt_inlinks: dict
        {pdt_idx: {rct_idx: fraction_atoms_from_rct_idx}, }
    
    '''
    operator = AllChem.ReactionFromSmarts(operator)
    
    rcts, pdts = [[smi for smi in side.split(".")] for side in reaction.split(">>")]
    rcts = [Chem.MolFromSmiles(smi) for smi in rcts]

    # Label rct atoms with rct idx
    for i, r in enumerate(rcts):
        for a in r.GetAtoms():
            a.SetIntProp('rct_idx', i)

    outputs = operator.RunReactants(rcts, maxProducts=10_000) # Apply operator
    aligned_output = align_outputs_w_products(outputs, pdts)
    if not aligned_output:
        return None
    
    # Tally up which atoms from where
    pdt_inlinks = {pdt_idx: {rct_idx: 0 for rct_idx in range(len(rcts))} for pdt_idx in range(len(pdts))} # {pdt_idx: {rct_idx: n_atoms_from}, }
    rct_inlinks = {rct_idx: {pdt_idx: 0 for pdt_idx in range(len(pdts))} for rct_idx in range(len(rcts))} # {rct_idx: {pdt_idx: n_atoms_from}, }
    for pdt_idx, o in enumerate(aligned_output):
        for a in o.GetAtoms():
            prop_dict = a.GetPropsAsDict()

            if 'rct_idx' in prop_dict:
                rct_idx = prop_dict['rct_idx']
            else:
                rct_idx =  atom_map_to_rct_idx[prop_dict['old_mapno']]

            pdt_inlinks[pdt_idx][rct_idx] += 1
            rct_inlinks[rct_idx][pdt_idx] += 1

    # Normalize by total number of atoms
    for pdt_idx, r_cts in pdt_inlinks.items():
        for rct_idx, ct in r_cts.items():
            pdt_inlinks[pdt_idx][rct_idx] = ct / aligned_output[pdt_idx].GetNumAtoms()

    for rct_idx, p_cts in rct_inlinks.items():
        for pdt_idx, ct in p_cts.items():
            rct_inlinks[rct_idx][pdt_idx] = ct / rcts[rct_idx].GetNumAtoms()

    return rct_inlinks, pdt_inlinks

class SimilarityConnector:
    def __init__(
            self, reactions: dict,
            cc_sim_mats: dict[str, np.ndarray],
            cofactors: dict[str, str],
            k_paired_cofactors: int = 21,
            n_rxns_lb: int = 5,
            include_paired_cofactors: Iterable[tuple] = [('ATP', 'AMP')]
        ) -> None:
        '''
        Args
        -----
        self, reactions: dict
        cc_sim_mats: dict[str, np.ndarray]
            Compound-compound similarity matrices. Indices assumed
            to be order of appearance in reactions TODO: Change this assumption. Too tenuous
        cofactors: dict[str, str]
            SMILES to name for unpaired cofactors
        k_paired_cofactors: int
        n_rxns_lb: int
        include_paired_cofactors: Iterable[tuple]
        '''
        self.reactions = reactions
        self.cofactors = cofactors # TODO: extract ids
        self.cc_sim_mats = cc_sim_mats
        self.smi2id, self.compounds = self._make_smi2id(reactions)
        self.cc_sim_mats['jaccard'] = self._construct_rxn_co_occurence_jaccard()
        self.paired_cofactors = self._make_paired_cofactors(k_paired_cofactors, n_rxns_lb, include_paired_cofactors)

    def _make_smi2id(self, reactions):
        compounds = {}
        for elt in reactions.values():
            subs = chain(*[side.split(".") for side in elt['smarts'].split(">>")])
            for sub in subs:
                name = elt['smi2name'][sub]
                compounds[sub] =  name if name else ''

        compounds = {i: {'smiles': k, 'name': v} for i, (k, v) in enumerate(compounds.items())}
        smi2id = {v['smiles']: k for k, v in compounds.items()}
        return smi2id, compounds
    
    def _construct_rxn_co_occurence_jaccard(self):
        cpd_corr = np.zeros(shape=(len(self.compounds), len(self.compounds)))
        for rxn in self.reactions.values():
            lhs, rhs = [set(side.split(".")) for side in rxn['smarts'].split(">>")] # Set out stoichiometric degeneracy
            lhs = [elt for elt in lhs if elt not in self.cofactors]
            rhs = [elt for elt in rhs if elt not in self.cofactors]

            for pair in product(lhs, rhs):
                i, j = [self.smi2id[elt] for elt in pair]
                cpd_corr[i, j] += 1
                cpd_corr[j, i] += 1
            
        row_sum = cpd_corr.sum(axis=1).reshape(-1, 1)
        col_sum = cpd_corr.sum(axis=0).reshape(1, -1)
        return cpd_corr / (row_sum + col_sum - cpd_corr) # Jaccard co-occurence-in-rxn index. Symmetric

    def _make_paired_cofactors(self, k_paired_cofactors, n_rxns_lb, include_paired_cofactors):
        id2jaccard = {}
        rxns_per_cpd = defaultdict(float)
        for rxn in self.reactions.values():
            lhs, rhs = [set(side.split(".")) for side in rxn['smarts'].split(">>")] # Set out stoichiometric degeneracy
            lhs = [elt for elt in lhs if elt not in self.cofactors]
            rhs = [elt for elt in rhs if elt not in self.cofactors]

            for elt in chain(lhs, rhs):
                rxns_per_cpd[self.smi2id[elt]] += 1

            for pair in product(lhs, rhs):
                ids = tuple(sorted([self.smi2id[elt] for elt in pair])) # This sort of ids critical
                id2jaccard[ids] = self.cc_sim_mats['jaccard'][ids[0], ids[1]]

        srt_cofactor_pairs = sorted(id2jaccard, key=lambda x : id2jaccard[x], reverse=True)
        srt_cofactor_pairs = [pair for pair in srt_cofactor_pairs if all([rxns_per_cpd[id] > n_rxns_lb for id in pair])] # Filter out rare cpds
        
        # Check for paired cofactors included by name
        srt_cofactor_names = [tuple(sorted([self.compounds[id]['name'] for id in pair])) for pair in srt_cofactor_pairs]
        add_idxs = []
        for elt in include_paired_cofactors:
            try:
                idx = srt_cofactor_names.index(elt)
                add_idxs.append(idx)
            except ValueError:
                continue

        all_idxs = set([i for i in range(k_paired_cofactors)]) | set(add_idxs)

        return [srt_cofactor_pairs[idx] for idx in all_idxs]
    
    def connect_reaction(self, rid: int):
        if rid not in self.reactions:
            raise ValueError(f"{rid} not found in reactions")
        
        rcts, pdts = [[smi for smi in side.split(".")] for side in self.reactions[rid]['smarts'].split(">>")] # SMILES

        # List ids for self.compounds. Mark unpaired cofactors wth None
        rct_ids = set([self.smi2id[smi] for smi in rcts if smi not in self.cofactors])
        pdt_ids = set([self.smi2id[smi] for smi in pdts if smi not in self.cofactors])

        # Initialize inlinks w/ all but unpaired cofactors
        pdt_inlinks = {pdt_id: {rct_id: 0 for rct_id in rct_ids} for pdt_id in pdt_ids}
        rct_inlinks = {rct_id: {pdt_id: 0 for pdt_id in pdt_ids} for rct_id in rct_ids}

        # Find paired cofactors, connect in inlinks, remove from rct/pdt ids
        # Remove best cofactor pair only
        to_remove = tuple()
        best_jaccard = 0
        for pair in product(rct_ids, pdt_ids):
            srt_pair = tuple(sorted(pair))
            jaccard = self.cc_sim_mats['jaccard'][srt_pair[0], srt_pair[1]]

            if srt_pair in self.paired_cofactors and jaccard > best_jaccard:
                to_remove = pair # Note NOT srt pair
                best_jaccard = jaccard

        if to_remove:
            rct_id, pdt_id = to_remove
            
            pdt_inlinks[pdt_id][rct_id] = 1
            rct_inlinks[rct_id][pdt_id] = 1

            # Remove creating sim block
            rct_ids.remove(rct_id)
            pdt_ids.remove(pdt_id)

        if len(rct_ids) == 0 and len(pdt_ids) == 0: # Done after paired cofactors
            return rct_inlinks, pdt_inlinks
        elif len(rct_ids) == 0 and not len(pdt_ids) == 0: # Remaining pdts must connect lone rct paired cof
            for rem_id in pdt_ids:
                pdt_inlinks[rem_id][rct_id] = 1
            return rct_inlinks, pdt_inlinks
        elif not len(rct_ids) == 0 and len(pdt_ids) == 0: # Remaining rcts must connect lone pdt paired cof
            for rem_id in rct_ids:
                rct_inlinks[rem_id][pdt_id] = 1
            return rct_inlinks, pdt_inlinks
        
        # Construct reaction similarity matrix
        i, j = zip(*product(rct_ids, pdt_ids))
        sim_block = []
        for v in self.cc_sim_mats.values():
            sim_block.append(v[i, j].reshape(len(rct_ids), len(pdt_ids)))
        sim_block = np.stack(sim_block, axis=-1) # (n_rcts x n_pdts x n_metrics)
    
        # Get inlinks
        rct_ids, pdt_ids = list(rct_ids), list(pdt_ids)
        for i, rct_id in enumerate(rct_ids):
            sim = sim_block[i, :, :].copy().reshape(-1, len(self.cc_sim_mats))
            partner = self._get_partner(sim, pdt_ids)
            rct_inlinks[rct_id][partner] = 1 

        for j, pdt_id in enumerate(pdt_ids):
            sim = sim_block[:, j, :].copy().reshape(-1, len(self.cc_sim_mats))
            partner = self._get_partner(sim, rct_ids)
            pdt_inlinks[pdt_id][partner] = 1
        
        return rct_inlinks, pdt_inlinks

    def _get_partner(self, sim:np.ndarray, candidates:list):
        '''
        Selects partner for a substrate (reactant 
        for a product or product for a reactant) using 
        the metric that gives the lowest entropy distribution

        Args
        ----
        sim
            (n_candidates x n_metrics) matrix
        candidates
            List of candidate ids
        '''
        # Filtering out probability-mass-less metrics
        mass = sim.sum(axis=0).reshape(1, -1)
        _, j = np.nonzero(mass)
        sim = sim[:, j] / mass[:, j]
        H = entropy(sim, axis=0).reshape(-1,)
        midx = np.argmin(H)
        return candidates[np.argmax(sim[:, midx])]


    
def align_outputs_w_products(outputs: tuple[tuple[Mol]], products: list[str]):
    output_idxs = [i for i in range(len(products))]

    # Try WITHOUT tautomer canonicalization
    for output in outputs:
        try:
            output_smi = [post_standardize(mol, do_canon_taut=False) for mol in output] # Standardize SMILES
        except:
            continue

        # Compare predicted to actual products. If mapped, return correct order of output
        for perm in permutations(output_idxs):
            smi_perm = [output_smi[elt] for elt in perm]
            if smi_perm == products: 
                return [output[elt] for elt in perm]
        
    # Try WITH tautomer canonicalization
    try:
        products = [post_standardize(Chem.MolFromSmiles(smi), do_canon_taut=True) for smi in products]
    except:
        return []
    
    for output in outputs:
        try:
            output_smi = [post_standardize(mol, do_canon_taut=True) for mol in output] # Standardize SMILES
        except:
            continue

        # Compare predicted to actual products. If mapped, return correct order of output
        for perm in permutations(output_idxs):
            smi_perm = [output_smi[elt] for elt in perm]
            if smi_perm == products: 
                return [output[elt] for elt in perm]
            
    return []