from rdkit import Chem
from rdkit.Chem import AllChem, Mol
from krxns.cheminfo import post_standardize
from itertools import permutations, chain, product, combinations
from collections import defaultdict, Counter
from typing import Iterable
import numpy as np
from scipy.stats import entropy
import pandas as pd
from tqdm import tqdm

def construct_reaction_network(
        operator_connections: dict[int, dict],
        reactions: dict[int, dict],
        atom_lb: float = 0.0,
        coreactant_whitelist: Iterable = None,
        add_multi_mol_nodes: bool = False

    ):
    '''
    Args
    -------

    Returns
    ---------
    edges:list[tuple]
        Entries are (from:int, to:int, properties:dict)
    nodes:list[tuple]
        Entries are (id:int, properties:dict)
    '''
    include_edge_props = ('smarts', 'rhea_ids', 'imt_rules')
    direction_from_side = lambda side : 0 if side == 'rct_inlinks' else 1 if side == 'pdt_inlinks' else print("Error side key not found")

    reactions = fold_reactions(reactions)
    compounds, smi2id = extract_compounds(reactions)
    id2smi = {v: k for k, v in smi2id.items()}
    unpaired_whitelist = [k for k in coreactant_whitelist if coreactant_whitelist[k] is None]
    
    tmp_edges = []

    # Add from operator_connections
    for rid, sides in operator_connections.items():
        smiles = [elt.split('.') for elt in reactions[rid]['smarts'].split('>>')]

        for side, inlinks in sides.items():
            direction = direction_from_side(side) # This is a binary switch to tranpose the adjacency matrix
            inlinks = remove_unpaired_whitelist(inlinks, direction, smiles, unpaired_whitelist) # Remove unpaired coreactants
            adj_mat = translate_operator_adj_mat(inlinks, direction, smiles, smi2id) # Convert nestes inlinks dict to adjacency matrix
            edge_props = { **{'rid': rid}, **{prop: reactions[rid][prop] for prop in include_edge_props} }

            if direction == 0:
                edge_props['smarts'] = ">>".join(edge_props['smarts'].split(">>")[::-1])

            tmp_edges += nested_adj_mat_to_edge_list(adj_mat, edge_props, smi2id, id2smi, atom_lb, coreactant_whitelist, add_multi_mol_nodes)

    edges = []
    next_node_id = 0
    mols_to_node_id = {}
    nodes = []
    for elt in tmp_edges:
        new_elt = []
        mols_st = elt[:2] # Starter target molecules tuples
        for mols in mols_st:
            if mols in mols_to_node_id:
                this_node_id = mols_to_node_id[mols]
            else:
                mols_to_node_id[mols] = next_node_id
                next_node_id += 1
                this_node_id = mols_to_node_id[mols]
                smiles, names = zip(*[(compounds[m]['smiles'], compounds[m]['name']) for m in mols])
                smiles = ".".join(smiles)
                nodes.append((this_node_id, {'smiles': smiles, 'names': names, 'cpd_ids': mols}))
            
            new_elt.append(this_node_id)

        new_elt.append(elt[-1])
        new_elt = tuple(new_elt)
        edges.append(new_elt)

    return edges, nodes
        
def remove_unpaired_whitelist(adj_mat: dict[int, dict[int, float]], direction: int, smiles: list[str], unpaired_whitelist: Iterable[str]):
    tmp = defaultdict(lambda : defaultdict(float))
    for i, inner in adj_mat.items():
        ismi = smiles[direction ^ 0][i]
                
        if ismi in unpaired_whitelist:
            continue
    
        for j, atom_frac in inner.items():
            jsmi = smiles[direction ^ 1][j]
            
            if jsmi in unpaired_whitelist:
                continue

            tmp[i][j] = atom_frac

    return tmp

# def handle_multiple_rules(rules: dict[str, dict]):
#     first_rule = next(iter(rules))

#     if len(rules) == 1:
#         return rules[first_rule]
    
#     exactly_same = True
#     directionally_same = True

#     rct_inlinks_check = defaultdict(set)
#     pdt_inlinks_check = defaultdict(set)
#     for sides in rules.values():
#         for side, outer in sides.items():
#             for i, inner in outer.items():
#                 for j, atom_frac in inner.items():
#                     if side == 'rct_inlinks':
#                         rct_inlinks_check[(i, j)].add(atom_frac)
#                     elif side == 'pdt_inlinks':
#                         pdt_inlinks_check[(i, j)].add(atom_frac)

#     for v in rct_inlinks_check.values():
#         if len(v) > 1:
#             exactly_same = False
#             break

#     if exactly_same:
#         for v in pdt_inlinks_check.values():
#             if len(v) > 1:
#                 exactly_same = False
#                 break

#     if exactly_same:
#         return rules[first_rule]
#     else:
#         rct_inlinks_check = defaultdict(set)
#         pdt_inlinks_check = defaultdict(set)
#         for sides in rules.values():
#             for side, outer in sides.items():
#                 for i, inner in outer.items():
#                     distro = list(inner.values())
#                     if side == 'rct_inlinks':
#                         rct_inlinks_check[i].add(np.argmax(distro))
#                     elif side == 'pdt_inlinks':
#                         pdt_inlinks_check[i].add(np.argmax(distro))

#         for v in rct_inlinks_check.values():
#             if len(v) > 1:
#                 directionally_same = False
#                 break

#         if directionally_same:
#             for v in pdt_inlinks_check.values():
#                 if len(v) > 1:
#                     directionally_same = False
#                     break

#         if directionally_same:
#             return rules[first_rule]
        
#         return {}

def translate_operator_adj_mat(adj_mat: dict[int: dict[int, float]], direction: int, smiles: str, smi2id: dict[str, int]) -> list[dict[int: dict[int, float]]]:
    '''
    Convert operator adjacency matrices, which have molecules labeled by order
    of appearance in the reaction, to those with molecules labeled by their unique
    ID per lexicographical sort of their SMILES. If there are multiples of the same
    unique molecule, multiple adjacency matrices are returned.
    '''
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
        return [{i2id[i]: {j2id[j]: atom_frac for j, atom_frac in inner.items()} for i, inner in adj_mat.items()}]
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
        id2smi:dict,
        atom_lb: float,
        coreactant_whitelist: Iterable[str],
        add_multi_mol_nodes: bool
    ) -> list[tuple]:
    '''
    Converts (mutliple) m_pdts x n_rcts adjacency matrices into list of edges between
    the molecules in those reactions, roughly respecting conservation of mass.

    Args
    -----
    adj_mat: list[dict[int: dict[int, float]]]
    edge_props:dict
    smi2id:dict
    id2smi:dict
    atom_lb:float
    coreactant_whitelist:Iterable[str]
    add_multi_mol_nodes:bool
    '''
    get_cos = lambda candidates, exclude : dict(Counter([elt for elt in candidates if smi2id[elt] not in exclude]))
    rcts, pdts = [side.split(".") for side in edge_props["smarts"].split(">>")]

    def fails_whitelist_check(whitelist: dict, coreactants: dict, coproducts: dict):
        '''
        Returns True if required co-reactants are not in the whitelist or if they are
        in the whitelist but co-products do not feature their canonical pairs.
        Applied as final check of edge before adding to edgelist.
        TODO: Still necessary? For similarity connections?
        '''
        for k in coreactants.keys():
            if k not in whitelist:
                return True
            elif whitelist[k] is None: # Unpaired coreactant
                pass
            elif all([cp not in coproducts for cp in whitelist[k]]):
                return True
        return False
    
    def skip_paired_whitelist(successor: int, predecessors: Iterable[int], whitelist: dict[str, str], id2smi: dict[int, str]):
        '''
        Returns True if the successor is a paired whitelisted coreactant and its pair is among the predecessors
        '''
        successor_smi = id2smi[successor]
        predecessor_smis = [id2smi[p] for p in predecessors]
        if successor_smi not in whitelist:
            return False
        elif whitelist[successor_smi] in predecessor_smis:
            return True
        else:
            return False

    iids = adj_mat[0].keys()
    
    # Extract inlinks from sufficient (mixtures of) molecules
    inlinks = {(i,): {'from':tuple(), 'atom_frac':-1} for i in iids}
    for elt in adj_mat:

        for i, inner in elt.items():

            # Skip over paired whitelist coreactants
            if skip_paired_whitelist(successor=i, predecessors=elt.keys(),
                    whitelist=coreactant_whitelist, id2smi=id2smi):
                continue

            if add_multi_mol_nodes:
                neighbor_fracs = [(k, v) for k, v in inner.items() if v > atom_lb]
                
                if neighbor_fracs:
                    neighbor, atom_frac = zip(*neighbor_fracs)
                    atom_frac = sum(atom_frac)
                else:
                    neighbor = tuple()
                    atom_frac = -1

            else:
                neighbor, atom_frac = sorted(inner.items(), key=lambda x : x[1], reverse=True)[0] # Max
                neighbor = (neighbor,)
                
            # If multiples of same unique mol, i, across multiple
            # adj mats, take the max
            if atom_frac > atom_lb and atom_frac > inlinks[(i,)]['atom_frac']:
                inlinks[(i,)]['from'] = tuple(sorted(neighbor))
                inlinks[(i,)]['atom_frac'] = atom_frac

    if add_multi_mol_nodes:
        combo_lens = [2, 3] # Just two and 3 mol combos for now
        non_currency_outputs = [elt for elt in inlinks.keys() if id2smi[elt[0]] not in coreactant_whitelist]
        multi_inlinks = {}
        for k in combo_lens:
            for combo in combinations(non_currency_outputs, k):
                rows = []
                for c in combo:
                    sufficient_candidates = inlinks[c]['from']
                    sufficient_candidates = set(filter(lambda x : id2smi[x] not in coreactant_whitelist, sufficient_candidates))
                    rows.append(sufficient_candidates)
                    
                sufficient = set.union(*rows)
                
                if len(sufficient) > 0:
                    sufficient = tuple(sorted(sufficient)) # Keep cpd ids sorted
                    multi_source = {'from': sufficient, 'atom_frac': 1.0} # TODO: Get actual weighted average of atom_frac, for now this is fine
                    multi_inlinks[tuple(sorted(chain(*combo)))] = multi_source # Keep cpd ids sorted

        inlinks = {**inlinks, **multi_inlinks}

    # Add edges to network edge list
    triple_set = set()
    edges = []
    for i in inlinks:
        if len(inlinks[i]['from']) == 0 or inlinks[i]['atom_frac'] == -1:
            continue
        
        j = inlinks[i]['from']
        coreactants = get_cos(candidates=rcts, exclude=j)
        coproducts = get_cos(candidates=pdts, exclude=i)
        atom_frac = inlinks[i]['atom_frac']

        if coreactant_whitelist and fails_whitelist_check(coreactant_whitelist, coreactants, coproducts):
            continue

        # Ensure no repeats
        # Note: gives (from, to, props) in order networkx expects
        if (j, i, atom_frac) not in triple_set:
            triple_set.add((j, i, atom_frac))
            props = { **edge_props, **{'atom_frac':atom_frac, "coreactants": coreactants, "coproducts": coproducts} }
            edges.append((j, i, props))

    return edges

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
        Extracts compounds from reaction dict, assigning them
        indices in lexicographical order of SMILES.

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

if __name__ == '__main__':
    from krxns.config import filepaths
    from krxns.utils import str2int
    import json

    # Load unpaired coreactants
    with open(filepaths['coreactants'] / 'unpaired_whitelist.json', 'r') as f:
        unpaired_whitelist = json.load(f)

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

    # Load coreactant whitelist
    with open(filepaths['coreactants'] / 'pickaxe_whitelist.json', 'r') as f:
        coreactant_whitelist = json.load(f)

    # Get known compounds
    kcs, smi2id = extract_compounds(krs)

    edges, nodes = construct_reaction_network(
        operator_connections=op_cxns,
        similarity_connections=sim_cxn,
        side_counts=side_counts,
        reactions=krs,
        unpaired_whitelist=unpaired_whitelist,
        connect_nontrivial=False,
        atom_lb=0.0,
        coreactant_whitelist=coreactant_whitelist,
        add_multi_mol_nodes=True
    )