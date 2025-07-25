import networkx as nx
from networkx.exception import NetworkXNoPath
from typing import Any
from copy import deepcopy
import pandas as pd
from dataclasses import dataclass, field
from collections import deque
from itertools import product
from rdkit import Chem
from typing import Iterable
# from ergochemics.noseque import hash_compound, hash_reaction # TODO

class ReactionNetwork(nx.MultiDiGraph):
    def __init__(self, incoming_graph_data=None, multigraph_input=None, **attr):
        super().__init__(incoming_graph_data, multigraph_input, **attr)
        self.ij2k = {}

    def add_node(self, node, **attr):
        # TODO: switch to hashing
        if type(node) is not int:
            raise TypeError("Node index must be an integer.")
        return super().add_node(node, **attr)
    
    def add_edge(self, u_for_edge, v_for_edge, key=None, **attr):
        # TODO: switch to hashing
        if type(u_for_edge) is not int or type(v_for_edge) is not int:
            raise TypeError("Node indices must be integers.")
        key = super().add_edge(u_for_edge, v_for_edge, key=key, **attr)
        ij = (u_for_edge, v_for_edge)
        if ij in self.ij2k:
            self.ij2k[ij].append(key)
        else:
            self.ij2k[ij] = [key]

        return key

    def get_nodes_by_prop(self, prop: str, value: Any) -> list[int]:
        return [x for x, y in self.nodes(data=True) if y[prop] == value]
    
    def get_edges_between(self, source:int, target:int, k:int = None):
        if k:
            return self.edges[source, target, k]
        else:
            return [self.edges[source, target, k] for k in self.ij2k[(source, target)]]
        
    def shortest_path(self, source:int = None, target:int = None, rm_req_target: bool = True, quiet: bool = False) -> dict | list:
        if source is None and target is None:
            return nx.shortest_path(self)
        elif (source is None) ^ (target is None):
            raise ValueError("Provide both source and target or neither")
        elif rm_req_target:
            target_smiles = self.nodes[target]['smiles']
            to_remove = [(i, j) for i, j, props in self.edges(data=True) if target_smiles in props['coreactants']]
            pruned = deepcopy(self)
            pruned.remove_edges_from(to_remove)
        else:
            pruned = self
        
        try:
            node_path = nx.shortest_path(pruned, source, target)
        except NetworkXNoPath as e:
            if not quiet:
                print(e)
            return [], [] # No path found
        
        edge_path = []
        for i in range(len(node_path) - 1):
            edge_path.append(pruned.get_edges_between(node_path[i], node_path[i+1]))

        return node_path, edge_path
    
    def add_reaction(self, am_rxn: str, rid: str | int, smi2name: dict[str, str]) -> None:
        '''
        Adds a reaction to the reaction network.

        Args
        ----
        am_rxn: str
            Atom-mapped reaction string in the form of "R1.R2.R3>>P1.P2.P3".
        rid: str | int
            Reaction ID, can be a string or an integer.
        smi2name: dict[str, str]
            Mapping from SMILES to compound names.
        '''
        node_indices = {data['smiles']: idx for idx, data in self.nodes(data=True)} # Collect SMILES: idx mapping for all existing nodes
        mass_contributions = get_mass_contributions(am_rxn)
        for pdt_smi, rcts in mass_contributions['pdt_normed_mass_contrib'].items():
            pdt_id = node_indices.get(pdt_smi, len(self.nodes))
            rcts = {k: v for k, v in rcts.items()}

            grouped_predecessors = []
            for rct_smi, pnmc in rcts.items():
                rct_id = node_indices.get(rct_smi, len(self.nodes))
                grouped_predecessors.append(rct_id)
                rnmc = mass_contributions['rct_normed_mass_contrib'][pdt_smi][rct_smi]
                
                if rct_id not in self.nodes:
                    rct_attrs = {'smiles': rct_smi, 'name': smi2name.get(rct_smi, "Unknown")}
                    rct_attrs['source'] = False
                    rct_attrs['grouped_predecessors'] = {}
                    rct_attrs['tot_rnmc'] = {}
                    self.add_node(rct_id, **rct_attrs)
                
                if pdt_id in self.nodes:
                    self.nodes[pdt_id]['grouped_predecessors'][rid] = grouped_predecessors
                    self.nodes[pdt_id]['tot_rnmc'][rid] = mass_contributions['tot_rct_normed_mass_contrib'][pdt_smi]
                else:
                    pdt_attrs = {'smiles': pdt_smi, 'name': smi2name.get(pdt_smi, "Unknown")}
                    pdt_attrs['grouped_predecessors'] = {rid: grouped_predecessors}
                    pdt_attrs['tot_rnmc'] = {rid: mass_contributions['tot_rct_normed_mass_contrib'][pdt_smi]}
                    pdt_attrs['source'] = False
                    self.add_node(pdt_id, **pdt_attrs)
                self.add_edge(
                    rct_id,
                    pdt_id,
                    **{
                        'reaction_id': rid,
                        'pdt_normed_mass_frac': pnmc,
                        'rct_normed_mass_frac': rnmc,
                        'am_smarts': am_rxn,
                    }
                )
    
    def set_sources(self, smiles: Iterable[str] = None, indices: Iterable[int] = None) -> None:
        '''
        Sets the source compounds in the reaction network.

        Args
        ----
        smiles: Iterable[str], optional
            An iterable of SMILES strings representing the source compounds.
        indices: Iterable[int], optional
            An iterable of node indices representing the source compounds.
        
        Raises
        ------
        ValueError
            If neither `smiles` nor `indices` are provided.
        '''
        if smiles is None and indices is None:
            raise ValueError("Provide either smiles or indices to set sources.")
        
        if smiles is not None:
            indices = [self.get_nodes_by_prop('smiles', smi) for smi in smiles]
        
        for idx in indices:
            if idx in self.nodes:
                self.nodes[idx]['source'] = True
            else:
                raise ValueError(f"Node index {idx} not found in the network.")

def get_mass_contributions(am_rxn: str) -> dict[str, dict[int, dict[int, float]]]:
    '''
    Returns fraction of atoms in a reactant / product coming from a product / reactant, respectively
    plus a summed rct normed mass contribution for each product.

    Args
    ----
    am_rxn:str
        Atom-mapped reaction string in the form of "R1.R2.R3>>P1.P2.P3"
    
    Returns
    -------
    dict[str, dict[int, dict[int, float]]]
        With differently normalized mass contributions:
        {
            "rct_normed_mass_contrib": {
                pdt_smi: {
                    rct_smi: (atoms rct -> pdt) / tot_rct_atoms
                }
            },
            "pdt_normed_mass_contrib": {
                pdt_smi: {
                    rct_smi: (atoms rct -> pdt) / tot_pdt_atoms
                }
            }
            "tot_rct_normed_mass_contrib": {
                pdt_smi: \sum(rnmc) / \sum(rct atoms)
            }
        }

    Notes
    -----
    Stoichiometric multiples of a unique molecule are aggregated into one account.
    e.g., if A_1 + A_2 >> C + D and A_1 contributes 2 atoms to C and A_2 contributes 3 atoms to C,
    it will be counted as A contributes 5 atoms to C.
    '''
    rcts_smiles, pdts_smiles = de_am(am_rxn)

    rcts, pdts = [
        [Chem.MolFromSmiles(elt) for elt in side.split('.')]
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
    
    # Here and below you will count atoms for stoichiometric multiples into the same 
    # key values, the smiles of the reactant or product
    # Count atoms received by molecule i from molecule j
    atom_counts = {i_smi: {j_smi: 0 for j_smi in rcts_smiles} for i_smi in pdts_smiles}
    for amn in amns:
        rct_smi = rcts_smiles[amn_to_rct_idx[amn]]
        pdt_smi = pdts_smiles[amn_to_pdt_idx[amn]]
        atom_counts[pdt_smi][rct_smi] += 1

    # Collect rct n atoms to normalize mass contributions
    # in one returned dict
    rct_smi_to_n_atoms = {}
    for rct, rct_smi in zip(rcts, rcts_smiles):
        rct_smi_to_n_atoms[rct_smi] = rct.GetNumAtoms()
    
    # Normalize by number of atoms in reactant / product
    rct_normed_mass_contrib = {}
    pdt_normed_mass_contrib = {}
    tot_rct_normed_mass_contrib = {}
    for pdt_smi, rct_dict in atom_counts.items():
        rct_normed_mass_contrib[pdt_smi] = {}
        pdt_normed_mass_contrib[pdt_smi] = {}
        tot_rct_normed_mass_contrib[pdt_smi] = 0
        tot_atoms = sum(rct_dict.values()) # Total atoms in product
        for rct_smi, count in rct_dict.items():
            rct_normed_mass_contrib[pdt_smi][rct_smi] = count / rct_smi_to_n_atoms[rct_smi]
            pdt_normed_mass_contrib[pdt_smi][rct_smi] = count / tot_atoms
            tot_rct_normed_mass_contrib[pdt_smi] += count
        
        tot_rct_normed_mass_contrib[pdt_smi] /= sum(rct_smi_to_n_atoms.values()) # Normalize by total number of atoms in reactants

    return {
        "rct_normed_mass_contrib": rct_normed_mass_contrib,
        "pdt_normed_mass_contrib": pdt_normed_mass_contrib,
        "tot_rct_normed_mass_contrib": tot_rct_normed_mass_contrib,
    }

def de_am(am_rxn: str) -> tuple[str, str]:
    '''
    Converts an atom-mapped reaction string to a de atom mapped SMILES
    of reactants and pdts.

    Args
    ----
    am_rxn: str
        Atom-mapped reaction string in the form of "R1.R2.R3>>P1.P2.P3"
    
    Returns
    -------
    rcts: list[str]
        List of reactant SMILES strings.
    pdts: list[str]
        List of product SMILES strings.
    '''
    am_rcts, am_pdts = [[Chem.MolFromSmiles(elt) for elt in side.split('.')] for side in am_rxn.split('>>')]
    for mol in am_rcts + am_pdts:
        if mol is None:
            raise ValueError(f"Invalid SMILES in reaction: {am_rxn}")
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)

    rcts = [Chem.MolToSmiles(mol) for mol in am_rcts]
    pdts = [Chem.MolToSmiles(mol) for mol in am_pdts]
    return rcts, pdts
    

@dataclass
class SyntheticTree:
    '''
    Represents a retrosynthetic tree

    Attributes
    ----------
    root: int
        Node index of the root of the tree, typically the target compound.
    generations: list[dict[int, str]]
        A list of dictionaries representing each generation in the tree.
        Each dictionary maps compound node indices to the reaction ID that produced them. Value
        is None for compounds that have not been produced by any reaction.
    leaves: list[tuple[int, int]]
        A list of tuples where each tuple contains a node index and its generation index.
        Represents the current leaves of the tree, i.e., compounds that have not been further reacted
        or transformed. The generation index is required to disambiguate leaves of the same compound
        that are at different generations of the tree.
    n_gens: int
        Returns the number of generations in the synthetic tree.
    n_leaves: int
        Returns the number of leaves in the synthetic tree, i.e., compounds that have not been further reacted
        or transformed.
    
    Methods
    -------
    copy() -> SyntheticTree
        Returns a deep copy of the SyntheticTree instance.
    grow(leaf: tuple[str, int], rxn_id: str, rcts: list[str])
        Grows the synthetic tree by adding a new reaction at the specified leaf.
        The leaf is a tuple containing the compound ID and its generation index.
        The reaction ID and the list of reactants are provided to update the tree.
    '''
    root: int
    generations: list[dict[int, str]] = field(default_factory=list)
    _leaves: list[tuple[int, int]] = field(default_factory=list)

    def copy(self):
        return SyntheticTree(
            root=self.root,
            generations=deepcopy(self.generations),
            _leaves=deepcopy(self._leaves)
        )
    
    def __post_init__(self):
        if len(self.generations) == 0:
            self.generations = [{self.root: None}]
            self._leaves.append((self.root, 0))

    
    def grow(self, leaf: tuple[str, int], rxn_id: str, rcts: list[str]):
        if leaf not in self._leaves:
            raise ValueError(f"Leaf {leaf} not in tree")
        
        self.generations[leaf[1]][leaf[0]] = rxn_id # Tracks which reaction produced this leaf
        self._leaves.remove(leaf) # Remove leaf from leaves list now that it has been produced
        
        if leaf[1] == self.n_gens:
            self.generations.append({}) # Ensures new generation created if needed

        for rct in rcts:
            self.generations[leaf[1] + 1][rct] = None
            self._leaves.append((rct, leaf[1] + 1))

    @property
    def leaves(self) -> list[tuple[str, int]]:
        return deepcopy(self._leaves)  
        
    @property
    def n_gens(self) -> int:
        return len(self.generations) - 1
    
    @property
    def n_leaves(self) -> int:
        return len(self._leaves)

def enumerate_synthetic_trees(target: int, sources: set[int], G: ReactionNetwork, max_depth: int, max_leaves: int, rnmc_lb: float = 0.1) -> list[SyntheticTree]:
    '''
    Enumerates synthetic trees for a given target compound in a reaction network.

    Args
    ----
    target: int
        Node index of the target compound for which to enumerate synthetic trees.
    sources: set[int]
        Node indices of the source compounds that can be used in the synthesis.
    G: nx.Graph
        The reaction network graph. Must contain nodes with 'tot_rnmc' and 'grouped_predecessors' attributes.
    max_depth: int
        Maximum depth of the synthetic tree.
    max_leaves: int
        Maximum number of leaves in the synthetic tree.

    Returns
    -------
    list[SyntheticTree]
        A list of synthetic trees enumerated from the reaction network.
    '''
    synthetic_trees = []
    tree = SyntheticTree(root=target)
    stack = deque()
    stack.append(tree)
    while stack:
        tree = stack.pop()
        
        if tree.n_gens > max_depth or tree.n_leaves > max_leaves: # Exclusion criteria
            continue
        
        if all([leaf[0] in sources for leaf in tree.leaves]): # Inclusion criteria. Compare just compound node index to sources
            synthetic_trees.append(tree)
            continue

        # Each expansion step must make a choice of reaction for each leaf
        # First collect reaction choices for each leaf
        leaf_choices = {}
        for leaf in tree.leaves:
            if leaf[0] in sources: # No need to grow tree from sources
                continue

            leaf_choices[leaf] = []

            for rxn, tot_rnmc in G.nodes[leaf[0]]['tot_rnmc'].items():
                if tot_rnmc < rnmc_lb: # Exclude reaction on atom economic grounds
                    continue

                leaf_choices[leaf].append(rxn)
        
        # Make a choice for each leaf and stack new trees
        choices = product(*leaf_choices.values())
        for choice in choices:
            new_tree = tree.copy()
            for leaf, rxn in zip(leaf_choices.keys(), choice):
                rcts = list(G.nodes[leaf[0]]['grouped_predecessors'][rxn])
                new_tree.grow(leaf, rxn, rcts)
            stack.append(new_tree)

    return synthetic_trees
       
if __name__ == '__main__':
    nodes = [
        (0, {'grouped_predecessors': {'R1': [1,], 'R2': [2,], 'R3': [3, 4]}, 'tot_rnmc': {'R1': 1.0, 'R2': 1.0, 'R3': 1.0}}),
        (1, {'grouped_predecessors': {'R4': [5,], 'R5': [6, 7]}, 'tot_rnmc': {'R4': 1.0, 'R5': 1.0}}),
        (2, {'grouped_predecessors': {}, 'tot_rnmc': {}}),
        (3, {'grouped_predecessors': {'R6': [8, 9]}, 'tot_rnmc': {'R6': 0.8}}),
        (4, {'grouped_predecessors': {'R7': [10,]}, 'tot_rnmc': {'R7': 1.0}}),
        (5, {'grouped_predecessors': {}, 'tot_rnmc': {}}),
        (6, {'grouped_predecessors': {}, 'tot_rnmc': {}}),
        (7, {'grouped_predecessors': {'R8': [11,]}, 'tot_rnmc': {'R8': 0.9}}),
        (8, {'grouped_predecessors': {'R9': [12,]}, 'tot_rnmc': {'R9': 1.0}}),
        (9, {'grouped_predecessors': {'R10': [13,]}, 'tot_rnmc': {'R10': 0.95}}),
        (10, {'grouped_predecessors': {}, 'tot_rnmc': {}}),
        (11, {'grouped_predecessors': {}, 'tot_rnmc': {}}),
        (12, {'grouped_predecessors': {'R11': [14,]}, 'tot_rnmc': {'R11': 0.85}}),
        (13, {'grouped_predecessors': {}, 'tot_rnmc': {}}),
        (14, {'grouped_predecessors': {}, 'tot_rnmc': {}}),
    ]

    edges = [
        (1, 0, {'rid': 'R1'}),
        (2, 0, {'rid': 'R2'}),
        (3, 0, {'rid': 'R3'}),
        (4, 0, {'rid': 'R3'}),
        (5, 1, {'rid': 'R4'}),
        (6, 1, {'rid': 'R5'}),
        (7, 1, {'rid': 'R5'}),
        (8, 3, {'rid': 'R6'}),
        (9, 3, {'rid': 'R6'}),
        (10, 4, {'rid': 'R7'}),
        (11, 7, {'rid': 'R8'}),
        (12, 8, {'rid': 'R9'}),
        (13, 9, {'rid': 'R10'}),
        (14, 12, {'rid': 'R11'}),

    ]

    target = 0
    sources = {6, 10, 11, 13, 14}
    max_leaves = 5
    max_depth = 5
    
    G = nx.MultiDiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    synthetic_trees = enumerate_synthetic_trees(
        target=target,
        sources=sources,
        G=G,
        max_depth=max_depth,
        max_leaves=max_leaves,
        rnmc_lb=0.1
    )
    print()


    import json
    from pathlib import Path
    root_dir = Path(__file__).parent.parent.parent
    kcs = pd.read_csv(root_dir / "data/interim/compounds.csv")
    sources = pd.read_csv(root_dir / "data/interim/default_sources.csv")
    sources = sources['id'].tolist()
    with open(root_dir / "data/interim/mass_contributions.json", 'r') as f:
        mass_contributions = json.load(f)

    mass_contributions = {k: v for k, v in mass_contributions.items() if k == '1148'}

    edges, nodes = construct_reaction_network(
        mass_contributions=mass_contributions,
        compounds=kcs,
        sources=sources,
        pnmc_lb=0.33,
        rnmc_lb=0.33,
        tot_mass_lb=0.7
    )

    edges, nodes = construct_reaction_network(
        mass_contributions=mass_contributions,
        compounds=kcs,
        sources=[]
    )