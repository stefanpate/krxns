from rdkit import Chem
from rdkit.Chem import AllChem, Mol
from krxns.cheminfo import post_standardize
from itertools import permutations, chain, product
from collections import defaultdict, Counter
from typing import Iterable
import numpy as np
from scipy.stats import entropy
import pandas as pd
from tqdm import tqdm

def construct_reaction_network(
        operator_connections: dict[int, dict],
        reactions: dict[int, dict],
        unpaired_cofactors: Iterable[str],
        similarity_connections:dict[int, dict] = {},
        side_counts:dict[int, list] = {},
        connect_nontrivial: bool = False,
        atom_lb: float = 0.0,
        coreactant_whitelist: Iterable = None

    ):
    '''
    Args
    -------

    Returns
    ---------
    edge_list:list[tuple]
        Entries are (from:int, to:int, properties:dict)
    node_list:list[tuple]
        Entries are (id:int, properties:dict)
    '''
    if atom_lb != 0.0 and connect_nontrivial:
        raise ValueError("Cannot enforce atom fraction lower bound if also including non-trivial similarity-based connections, which to not capture conservation of mass.")

    fix_op_w_sim = [10540, 15237] # Known problems w/ operator-reaction mapping
    include_edge_props = ('smarts', 'rhea_ids', 'imt_rules')
    direction_from_side = lambda side : 0 if side == 'rct_inlinks' else 1 if side == 'pdt_inlinks' else print("Error side key not found")

    reactions = fold_reactions(reactions)
    compounds, smi2id = extract_compounds(reactions)
    
    edge_list = []

    # Add from operator_connections
    for rid, rules in operator_connections.items():

        if rid in fix_op_w_sim:
            continue
        
        smiles = [elt.split('.') for elt in reactions[rid]['smarts'].split('>>')]

        # Remove unpaired cofactors
        filtered_rules = defaultdict(lambda : defaultdict(dict))
        for rule, sides in rules.items():
            for side, adj_mat in sides.items():
                direction = direction_from_side(side)
                adj_mat = remove_unpaired_cofactors(adj_mat, direction, smiles, unpaired_cofactors)
                filtered_rules[rule][side] = adj_mat

        sel_adj_mats = handle_multiple_rules(filtered_rules) # Resolve cases with multiple rules

        for side, adj_mat in sel_adj_mats.items():
            direction = direction_from_side(side)
            adj_mat = translate_operator_adj_mat(adj_mat, direction, smiles, smi2id)
            edge_props = { **{'rid': rid}, **{prop: reactions[rid][prop] for prop in include_edge_props} }

            if direction == 0:
                edge_props['smarts'] = ">>".join(edge_props['smarts'].split(">>")[::-1])

            edge_list += nested_adj_mat_to_edge_list(adj_mat, edge_props, smi2id, atom_lb, coreactant_whitelist)

    
    if similarity_connections and side_counts:
        if connect_nontrivial:
            sim_rids = list(similarity_connections.keys() - operator_connections.keys())
        else:
            sim_rids = {rid for rid in similarity_connections if 1 in side_counts[rid] and not 0 in side_counts[rid]}
            sim_rids = list(sim_rids - operator_connections.keys())
        
        sim_rids += fix_op_w_sim # Patch

        # Add from similarity connections
        for rid in sim_rids:
            for side, adj_mat in similarity_connections[rid].items():
                direction = direction_from_side(side)
                edge_props = { **{'rid': rid}, **{prop: reactions[rid][prop] for prop in include_edge_props} }
                if direction == 0:
                    edge_props['smarts'] = ">>".join(edge_props['smarts'].split(">>")[::-1])

                edge_list += nested_adj_mat_to_edge_list([adj_mat], edge_props, smi2id, atom_lb, coreactant_whitelist)

    # Assemble node list
    node_ids = set()
    for elt in edge_list:
        for i in range(2):
            node_ids.add(elt[i])

    node_list = []
    for id in node_ids:
        node_props = compounds[id]
        node_list.append((id, node_props))

    return edge_list, node_list
        

def remove_unpaired_cofactors(adj_mat: dict[int, dict[int, float]], direction: int, smiles: list[str], unpaired_cofactors: Iterable[str]):
    tmp = defaultdict(lambda : defaultdict(float))
    for i, inner in adj_mat.items():
        ismi = smiles[direction ^ 0][i]
                
        if ismi in unpaired_cofactors:
            continue
    
        for j, weight in inner.items():
            jsmi = smiles[direction ^ 1][j]
            
            if jsmi in unpaired_cofactors:
                continue

            tmp[i][j] = weight

    return tmp

def handle_multiple_rules(rules: dict[str, dict]):
    first_rule = next(iter(rules))

    if len(rules) == 1:
        return rules[first_rule]
    
    exactly_same = True
    directionally_same = True

    rct_inlinks_check = defaultdict(set)
    pdt_inlinks_check = defaultdict(set)
    for sides in rules.values():
        for side, outer in sides.items():
            for i, inner in outer.items():
                for j, weight in inner.items():
                    if side == 'rct_inlinks':
                        rct_inlinks_check[(i, j)].add(weight)
                    elif side == 'pdt_inlinks':
                        pdt_inlinks_check[(i, j)].add(weight)

    for v in rct_inlinks_check.values():
        if len(v) > 1:
            exactly_same = False
            break

    if exactly_same:
        for v in pdt_inlinks_check.values():
            if len(v) > 1:
                exactly_same = False
                break

    if exactly_same:
        return rules[first_rule]
    else:
        rct_inlinks_check = defaultdict(set)
        pdt_inlinks_check = defaultdict(set)
        for sides in rules.values():
            for side, outer in sides.items():
                for i, inner in outer.items():
                    distro = list(inner.values())
                    if side == 'rct_inlinks':
                        rct_inlinks_check[i].add(np.argmax(distro))
                    elif side == 'pdt_inlinks':
                        pdt_inlinks_check[i].add(np.argmax(distro))

        for v in rct_inlinks_check.values():
            if len(v) > 1:
                directionally_same = False
                break

        if directionally_same:
            for v in pdt_inlinks_check.values():
                if len(v) > 1:
                    directionally_same = False
                    break

        if directionally_same:
            return rules[first_rule]
        
        return {}

def translate_operator_adj_mat(adj_mat: dict[int: dict[int, float]], direction: int, smiles: str, smi2id: dict[str, int]) -> list[dict[int: dict[int, float]]]:
    i2id = {i: smi2id[smiles[direction ^ 0][i]] for i in adj_mat}
    j2id = {j: smi2id[smiles[direction ^ 1][j]] for j in next(iter(adj_mat.values()))}


    id2i = defaultdict(list)
    id2j = defaultdict(list)

    for i, id in i2id.items():
        id2i[id].append(i)

    for j, id in j2id.items():
        id2j[id].append(j)

    has_repeats = lambda x : any([len(v) > 1 for v in x.values()])
    if not has_repeats(id2i) and not has_repeats(id2j):
        return [{i2id[i]: {j2id[j]: weight for j, weight in inner.items()} for i, inner in adj_mat.items()}]
    else:
        expanded_adj_mat = []
        for i_combo in product(*id2i.values()):
            for j_combo in product(*id2j.values()):
                am = {i2id[i]: {j2id[j]: adj_mat[i][j] for j in j_combo} for i in i_combo}
                expanded_adj_mat.append(am)
        
        return expanded_adj_mat

def nested_adj_mat_to_edge_list(
        adj_mat: list[dict[int: dict[int, float]]],
        edge_props:dict,
        smi2id:dict,
        atom_lb: float,
        coreactant_whitelist: Iterable[str]
    ) -> list[tuple]:
    rcts, pdts = [side.split(".") for side in edge_props["smarts"].split(">>")]
    pull_other_subs = lambda x, exclude : dict(Counter([elt for elt in x if smi2id[elt] != exclude]))

    def fails_whitelist_check(whitelist: dict, requires: dict, other_products: dict):
        for k in requires.keys():
            if k not in whitelist:
                return True
            elif whitelist[k] is None:
                pass
            elif whitelist[k] not in other_products:
                return True
        return False


    iids = adj_mat[0].keys()
    
    inlinks = {i: {'from': -1, 'weight': -1} for i in iids}
    for elt in adj_mat:
        for i, inner in elt.items():
            neighbor, weight = sorted(inner.items(), key=lambda x : x[1], reverse=True)[0] # Max
            if weight > inlinks[i]['weight']:
                inlinks[i]['from'] = neighbor
                inlinks[i]['weight'] = weight

    # Note: returns as (from, to, props) because this is what networkx add_edges_from() expects
    edge_list = []
    for i in inlinks:
        if inlinks[i]['from'] == -1:
            continue
        
        j = inlinks[i]['from']
        requires = pull_other_subs(rcts, j)
        other_products = pull_other_subs(pdts, i)
        atom_frac = inlinks[i]['weight']

        if atom_frac < atom_lb:
            continue

        if coreactant_whitelist and fails_whitelist_check(coreactant_whitelist, requires, other_products):
            continue

        props = { **edge_props, **{'weight':atom_frac, "requires": requires, "other_products": other_products} }
        edge_list.append((j, i, props))

    return edge_list

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
        return {}, {}
    
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
            self, reactions: dict[int, dict],
            cc_sim_mats: dict[str, np.ndarray],
            unpaired_cofactors: dict[str, str],
            k_paired_cofactors: int = 21,
            n_rxns_lb: int = 5,
            include_paired_cofactors: Iterable[tuple] = [('ATP', 'AMP')]
        ) -> None:
        '''
        Args
        -----
        self, reactions: dict[int, dict]
        cc_sim_mats: dict[str, np.ndarray]
            Compound-compound similarity matrices. Indices assumed
            to be order of appearance in reactions TODO: Change this assumption. Too tenuous
        unpaired_cofactors: dict[str, str]
            SMILES to name for unpaired cofactors
        k_paired_cofactors: int
        n_rxns_lb: int
        include_paired_cofactors: Iterable[tuple]
        '''
        self.reactions = reactions
        self.unpaired_cofactors = unpaired_cofactors
        self.cc_sim_mats = cc_sim_mats
        self.compounds, self.smi2id  = extract_compounds(reactions)
        self.cc_sim_mats['jaccard'] = self._construct_rxn_co_occurence_jaccard()
        self.paired_cofactors = self._make_paired_cofactors(k_paired_cofactors, n_rxns_lb, include_paired_cofactors)
    
    def _construct_rxn_co_occurence_jaccard(self):
        cpd_corr = np.zeros(shape=(len(self.compounds), len(self.compounds)))
        for rxn in self.reactions.values():
            lhs, rhs = [set(side.split(".")) for side in rxn['smarts'].split(">>")] # Set out stoichiometric degeneracy
            lhs = [elt for elt in lhs if elt not in self.unpaired_cofactors]
            rhs = [elt for elt in rhs if elt not in self.unpaired_cofactors]

            for pair in product(lhs, rhs):
                i, j = [self.smi2id[elt] for elt in pair]
                cpd_corr[i, j] += 1
                cpd_corr[j, i] += 1
            
        row_sum = cpd_corr.sum(axis=1).reshape(-1, 1)
        col_sum = cpd_corr.sum(axis=0).reshape(1, -1)
        cpd_corr = np.where((row_sum + col_sum - cpd_corr) != 0, cpd_corr / (row_sum + col_sum - cpd_corr), 0) # Jaccard co-occurence-in-rxn index. Symmetric

        return cpd_corr

    def _make_paired_cofactors(self, k_paired_cofactors, n_rxns_lb, include_paired_cofactors):
        id2jaccard = {}
        rxns_per_cpd = defaultdict(float)
        for rxn in self.reactions.values():
            lhs, rhs = [set(side.split(".")) for side in rxn['smarts'].split(">>")] # Set out stoichiometric degeneracy
            lhs = [elt for elt in lhs if elt not in self.unpaired_cofactors]
            rhs = [elt for elt in rhs if elt not in self.unpaired_cofactors]

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
        rct_ids = set([self.smi2id[smi] for smi in rcts if smi not in self.unpaired_cofactors])
        pdt_ids = set([self.smi2id[smi] for smi in pdts if smi not in self.unpaired_cofactors])

        # No substrates on either side after removing unpaired cofactors
        if len(rct_ids) == 0 or len(pdt_ids) == 0:
            return {}, {}, [len(rct_ids), len(pdt_ids)]

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
            
            # Connect
            pdt_inlinks[pdt_id][rct_id] = 1
            rct_inlinks[rct_id][pdt_id] = 1

            # Remove creating sim block
            rct_ids.remove(rct_id)
            pdt_ids.remove(pdt_id)

        nr_np = [len(rct_ids), len(pdt_ids)]

        if nr_np[0] == 0 and nr_np[1] == 0: # Done after paired cofactors
            return rct_inlinks, pdt_inlinks, sorted(nr_np)
        elif nr_np[0] == 0 and not nr_np[1] == 0: # Remaining pdts must connect lone rct paired cof
            for rem_id in pdt_ids:
                pdt_inlinks[rem_id][rct_id] = 1
            return rct_inlinks, pdt_inlinks, sorted(nr_np)
        elif not nr_np[0] == 0 and nr_np[1] == 0: # Remaining rcts must connect lone pdt paired cof
            for rem_id in rct_ids:
                rct_inlinks[rem_id][pdt_id] = 1
            return rct_inlinks, pdt_inlinks, sorted(nr_np)
        
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
        
        return rct_inlinks, pdt_inlinks, sorted(nr_np)
    
    def connect_reactions(self):
        results = {}
        side_counts = {}
        for rid in tqdm(self.reactions.keys()):
            results[rid] = {}
            rct_inlinks, pdt_inlinks, nr_np = self.connect_reaction(rid)
            results[rid]['rct_inlinks'] = rct_inlinks
            results[rid]['pdt_inlinks'] = pdt_inlinks
            side_counts[rid] = nr_np

        return results, side_counts

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

def fold_reactions(reactions: dict[str, dict]):
    reactions = {int(k): v for k,v in reactions.items()} # Cast keys to int

    # Remove reverses
    rids = set()
    for k, v in reactions.items():
        rids.add(tuple(sorted([k, v['reverse']])))

    keepers = [elt[0] for elt in rids]
    reactions = {k: reactions[k] for k in keepers}

    return reactions

def extract_compounds(reactions: dict):
        '''
        Extracts compounds from reaction dict.

        Args
        ----
        reactions: dict
            Reaciton id -> {'smarts': , }

        Returns
        -------
        compounds:dict
            ID, i.e., row / col idx in compound x compound similarity matrix -> smiles and names
        smi2id:dict
            Smiles -> cxc similarity matrix
        '''
        tmp = {}
        for elt in reactions.values():
            smis = chain(*[side.split(".") for side in elt['smarts'].split(">>")])
            for smi in smis:
                name = elt['smi2name'][smi]
                tmp[smi] =  name if name else ''

        compounds = {}
        smi2id = {}
        for i, smi in enumerate(sorted(tmp.keys())): # Lexicographical sort of keys will be order of smi2id
            compounds[i] = {'smiles': smi, 'name': tmp[smi]}
            smi2id[smi] = i
        
        return compounds, smi2id


def construct_op_atom_map_to_rct_idx(ops: pd.DataFrame) -> dict:
    '''
    Returns dict mapping atom map number ot reactant idx on LHS
    of provided reaction operators
    '''
    op_atom_map_to_rct_idx = {}
    for op_name, row in ops.iterrows():

        atom_map_to_rct_idx = {}
        rxn = AllChem.ReactionFromSmarts(row["SMARTS"])
        for ri in range(rxn.GetNumReactantTemplates()):
            rt = rxn.GetReactantTemplate(ri)
            for atom in rt.GetAtoms():
                if atom.GetAtomMapNum():
                    atom_map_to_rct_idx[atom.GetAtomMapNum()] = ri

        op_atom_map_to_rct_idx[op_name] = atom_map_to_rct_idx
    
    return op_atom_map_to_rct_idx
        
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


if __name__ == '__main__':
    from krxns.config import filepaths
    from krxns.utils import str2int
    import json

    # Load unpaired cofactors
    with open(filepaths['cofactors'] / '241008_unpaired_cofactors.json', 'r') as f:
        unpaired_cofactors = json.load(f)

    # Load cc sim mats
    cc_sim_mats = {
        'mcs': np.load(filepaths['sim_mats'] / "mcs.npy"),
        'tanimoto': np.load(filepaths['sim_mats'] / "tanimoto.npy")
    }

    # Load known reaction data
    with open(filepaths['data'] / 'sprhea_240310_v3_mapped.json', 'r') as f:
        krs = json.load(f)

    # Load op connected reactions
    with open(filepaths['connected_reactions'] / 'sprhea_240310_v3_mapped_operator.json', 'r') as f:
        op_cxns = str2int(json.load(f))

    # Load sim connected reactions
    with open(filepaths['connected_reactions'] / 'sprhea_240310_v3_mapped_similarity.json', 'r') as f:
        sim_cxn = str2int(json.load(f))

    with open(filepaths['connected_reactions'] / 'sprhea_240310_v3_mapped_side_counts.json', 'r') as f:
        side_counts = str2int(json.load(f))

    construct_reaction_network(
        operator_connections=op_cxns,
        similarity_connections=sim_cxn,
        side_counts=side_counts,
        reactions=krs,
        unpaired_cofactors=unpaired_cofactors,
        connect_nontrivial=False,
        atom_lb=0.6,
        coreactant_whitelist=["O"]
    )