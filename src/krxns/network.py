import networkx as nx
from networkx.exception import NetworkXNoPath
from typing import Any
from copy import deepcopy, copy
from dataclasses import dataclass, field
from collections import deque
from itertools import product
from rdkit import Chem
from typing import Iterable
import pathlib
import json
from ergochemics.standardize import hash_compound, hash_reaction
import logging
logger = logging.getLogger(__name__)

@dataclass
class SyntheticTree:
    '''
    Represents a retrosynthetic tree

    Attributes
    ----------
    root: str
        Node index of the root of the tree, typically the target compound.
    generations: list[dict[str, str]]
        A list of dictionaries representing each generation in the tree.
        Each dictionary maps compound node ids to the reaction id that produced them. Value
        is None for compounds that have not been produced by any reaction.
    leaves: list[tuple[str, int]]
        A list of tuples where each tuple contains a node id and its generation index.
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
    root: str
    generations: list[dict[str, str]] = field(default_factory=list)
    leaves: list[tuple[str, int]] = field(default_factory=list)

    def copy(self):
        return SyntheticTree(
            root=self.root,
            generations=deepcopy(self.generations),
            leaves=copy(self.leaves)
        )
    
    def __post_init__(self):
        if len(self.generations) == 0:
            self.generations = [{self.root: None}]
            self.leaves.append((self.root, 0))

    def grow(self, leaf: tuple[str, int], rxn_id: str, rcts: list[str]):
        if leaf not in self.leaves:
            raise ValueError(f"Leaf {leaf} not in tree")
        
        self.generations[leaf[1]][leaf[0]] = rxn_id # Tracks which reaction produced this leaf
        self.leaves.remove(leaf) # Remove leaf from leaves list now that it has been produced
        
        if leaf[1] == self.n_gens:
            self.generations.append({}) # Ensures new generation created if needed

        for rct in rcts:
            self.generations[leaf[1] + 1][rct] = None
            self.leaves.append((rct, leaf[1] + 1))
 
    @property
    def n_gens(self) -> int:
        return len(self.generations) - 1
    
    @property
    def n_leaves(self) -> int:
        return len(self.leaves)

class ReactionNetwork(nx.MultiDiGraph):
    def __init__(self, incoming_graph_data=None, multigraph_input=None, **attr):
        super().__init__(incoming_graph_data, multigraph_input, **attr)
        self.logger = logging.getLogger(__name__)

    @classmethod
    def from_json(cls, fp: pathlib.Path | str) -> "ReactionNetwork":
        
        with open(fp, 'r') as f:
            data = json.load(f)

        G = nx.node_link_graph(data, edges="edges")
        return cls(incoming_graph_data=G)
    
    def to_json(self, fp: pathlib.Path | str) -> None:
        data = nx.node_link_data(self, edges="edges")
        
        with open(fp, "w") as f:
            json.dump(data, f)

    def get_nodes_by_prop(self, prop: str, value: Any) -> list[int]:
        return [x for x, y in self.nodes(data=True) if y[prop] == value]
        
    def shortest_path(self, source: str = None, target: str = None, rm_req_target: bool = True, quiet: bool = False) -> dict | list:
        if source is None and target is None:
            return nx.shortest_path(self)
        elif (source is None) ^ (target is None):
            raise ValueError("Provide both source and target or neither")
        elif rm_req_target: # TODO: remember what this was for
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
    
    def add_reaction(self, am_rxn: str, rid: str | None = None, smi2name: dict[str, str] = {}) -> None:
        '''
        Adds a reaction to the reaction network.

        Args
        ----
        am_rxn: str
            Atom-mapped reaction string in the form of "R1.R2.R3>>P1.P2.P3".
        rid: str | None
            Reaction ID. If None, uses a hash of the reaction string.
        smi2name: dict[str, str] (Optional)
            Mapping from SMILES to compound names.
        '''
        mass_contributions, de_am_rxn = get_mass_contributions(am_rxn)
        rid = rid or hash_reaction(de_am_rxn)
        for pdt_smi, rcts in mass_contributions['pdt_normed_mass_contrib'].items():
            pdt_id = hash_compound(pdt_smi)

            # Create new node with all but the grouped predecessors
            if pdt_id not in self.nodes:
                self.add_node(
                    pdt_id,
                    smiles = pdt_smi,
                    name = smi2name.get(pdt_smi, "Unknown"),
                    source = False,
                    grouped_predecessors = {},
                    tot_rnmc = {rid: mass_contributions['tot_rct_normed_mass_contrib'][pdt_smi]},
                )
            else: # Fill in tot_rnmc for exisiting node, new rxn
                self.nodes[pdt_id]['tot_rnmc'][rid] = mass_contributions['tot_rct_normed_mass_contrib'][pdt_smi]
           
            grouped_predecessors = []
            for rct_smi, pnmc in rcts.items():
                rct_id = hash_compound(rct_smi)

                grouped_predecessors.append(rct_id)
                rnmc = mass_contributions['rct_normed_mass_contrib'][pdt_smi][rct_smi]
                
                if rct_id not in self.nodes:
                    rct_attrs = {'smiles': rct_smi, 'name': smi2name.get(rct_smi, "Unknown")}
                    rct_attrs['source'] = False
                    rct_attrs['grouped_predecessors'] = {}
                    rct_attrs['tot_rnmc'] = {}
                    self.add_node(rct_id, **rct_attrs)
                
                self.add_edge( # TODO: add way to check if edge already exists
                    rct_id,
                    pdt_id,
                    key=rid,
                    **{
                        'pnmc': pnmc,
                        'rnmc': rnmc,
                        'am_smarts': am_rxn,
                    }
                )

            # Finally add grouped predecessors
            self.nodes[pdt_id]['grouped_predecessors'][rid] = grouped_predecessors
    
    def set_sources(self, smiles: Iterable[str] = None, ids: Iterable[int] = None, quiet: bool = False) -> None:
        '''
        Sets the source compounds in the reaction network.

        Args
        ----
        smiles: Iterable[str], optional
            An iterable of SMILES strings representing the source compounds.
        ids: Iterable[int], optional
            An iterable of node indices representing the source compounds.
        quiet: bool, optional
            If True, suppresses output messages.
        
        Raises
        ------
        ValueError
            If neither `smiles` nor `ids` are provided.
        '''
        if smiles is None and ids is None:
            raise ValueError("Provide either smiles or node ids to set sources.")
        
        if smiles is not None:
            ids = [_id for smi in smiles for _id in self.get_nodes_by_prop('smiles', smi)]
        
        ct = 0
        for _id in ids:
            if _id in self.nodes:
                self.nodes[_id]['source'] = True
                ct += 1
            else:
                raise ValueError(f"Node id {_id} not found in the network.")
            
        if not quiet:
            self.logger.info(f"Set {ct} source compounds in the reaction network.")
   
    def prune(self, pnmc_lb: float, rnmc_lb: float, source_augmented_pnmc_lb: float) -> None:
        '''
        Prunes the reaction network based on the provided thresholds.

        Args
        ----
        pnmc_lb: float
            Lower bound for product normalized mass contribution.
        rnmc_lb: float
            Lower bound for total reaction normalized mass contribution.
        source_augmented_pnmc_lb: float
            Lower bound for augmented product normalized mass contribution from sources.
        '''
        to_remove = []
        
        for node, data in self.nodes(data=True):
            if 'grouped_predecessors' not in data or 'tot_rnmc' not in data:
                continue
            
            for rxn_id, preds in data['grouped_predecessors'].items():
                rxn_pnmcs = {}
                rxn_sources = set()
                for pred in preds:
                    edge_data = self.get_edge_data(pred, node, key=rxn_id)
                    rxn_pnmcs[pred] = edge_data['pnmc']
                    if self.nodes[pred]['source']:
                        rxn_sources.add(pred)

                for pred in preds:
                    # Mark for deletion and move on if fails either indepenedent criteria
                    if edge_data['rnmc'] < rnmc_lb or edge_data['pnmc'] < pnmc_lb:
                        to_remove.append((pred, node, rxn_id))
                        continue
                    
                    source_mass = sum(rxn_pnmcs[s] for s in rxn_sources if s != pred) # Source contribution, excl pred if it is a source, avoid double counting
                    # Mark for deletion if fails augmented mass criteria
                    if rxn_pnmcs[pred] + source_mass < source_augmented_pnmc_lb:
                        to_remove.append((pred, node, rxn_id))

        # Remove from grouped predecessors attr
        # TODO: consider alt soln e.g., caching and raw atom counts to avoid this
        for i, j, k in to_remove:
            self.nodes[j]['grouped_predecessors'][k].remove(i) # Remove from grouped predecessors

            if len(self.nodes[j]['grouped_predecessors'][k]) == 0: # Remove key if no predecessors left
                self.nodes[j]['grouped_predecessors'].pop(k)
        
        self.remove_edges_from(to_remove) # Prune edges
        self.remove_nodes_from(list(nx.isolates(self))) # Prune disconnected nodes
    
    def enumerate_synthetic_trees(self, target: str, max_depth: int, max_leaves: int, tot_rnmc_lb: float = 0.1, quiet: bool = False) -> list[SyntheticTree]:
        '''
        Enumerates synthetic trees for a given target compound in a reaction network.

        Args
        ----
        target: str
            Node id of the target compound for which to enumerate synthetic trees.
        max_depth: int
            Maximum depth of the synthetic tree.
        max_leaves: int
            Maximum number of leaves in the synthetic tree.
        tot_rnmc_lb: float (optional)
            Lower bound on total fraction of mass from reactants transferred to a
            product (i.e., yield, atom economy).
        quiet: bool (optional)
            If True, suppresses output messages.

        Returns
        -------
        list[SyntheticTree]
            A list of synthetic trees enumerated from the reaction network.
        '''
        synthetic_trees = []
        tree = SyntheticTree(root=target)
        stack = deque()
        stack.append(tree)
        ct = 1
        while stack:
            tree = stack.pop()
            ct+=1
            
            if tree.n_gens > max_depth or tree.n_leaves > max_leaves: # Exclusion criteria
                continue
            
            if all([self.nodes[leaf[0]]['source'] for leaf in tree.leaves]): # Inclusion criteria. All leaf nodes designated as sources
                synthetic_trees.append(tree)
                continue

            # Each expansion step must make a choice of reaction for each leaf
            # First collect reaction choices for each leaf
            leaf_choices = {}
            for leaf in tree.leaves:
                if self.nodes[leaf[0]]['source']: # No need to grow tree from a source
                    continue

                leaf_choices[leaf] = []

                for rxn in self.nodes[leaf[0]]['grouped_predecessors'].keys():
                    if self.nodes[leaf[0]]['tot_rnmc'][rxn] < tot_rnmc_lb: # Exclude reaction on atom economic grounds
                        continue

                    leaf_choices[leaf].append(rxn)
            
            # Make a choice for each leaf and stack new trees
            choices = product(*leaf_choices.values())
            for choice in choices:
                new_tree = tree.copy()
                for leaf, rxn in zip(leaf_choices.keys(), choice):
                    rcts = list(self.nodes[leaf[0]]['grouped_predecessors'][rxn])
                    new_tree.grow(leaf, rxn, rcts)
                stack.append(new_tree)

        if not quiet:
            self.logger.info(f"Considered {ct} trees, found {len(synthetic_trees)} synthetic trees.")
        
        return synthetic_trees
                
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
    mass_contributrions: dict[str, dict[int, dict[int, float]]]
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
                pdt_smi: sum(rnmc) / sum(rct atoms)
            }
        }
    de_am_rxn: str
        Atom-mapped reaction string with atom map numbers removed, in the form of "R1.R2.R3>>P1.P2.P3"

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

    mass_contributions = {
        "rct_normed_mass_contrib": rct_normed_mass_contrib,
        "pdt_normed_mass_contrib": pdt_normed_mass_contrib,
        "tot_rct_normed_mass_contrib": tot_rct_normed_mass_contrib,
    }
    de_am_rxn = '.'.join(rcts_smiles) + '>>' + '.'.join(pdts_smiles)

    return mass_contributions, de_am_rxn


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
       
if __name__ == '__main__':
    import pandas as pd
  
    G = ReactionNetwork.from_json('/home/stef/krxns/data/processed/known_reaction_network.json')
    print("Full reaction network loaded from JSON.")
    print(f"Number of nodes: {G.number_of_nodes()}, Number of edges: {G.number_of_edges()}")
    rnmc_lb = 0.25
    pnmc_lb = 0.25
    aug_mc_lb = 0.8
    G.prune(pnmc_lb=pnmc_lb, rnmc_lb=rnmc_lb, source_augmented_pnmc_lb=aug_mc_lb)
    print(f"Pruned reaction network with pnmc_lb={pnmc_lb}, rnmc_lb={rnmc_lb}, source_augmented_pnmc_lb={aug_mc_lb}.")
    print(f"Number of nodes after pruning: {G.number_of_nodes()}, Number of edges: {G.number_of_edges()}")

    G = ReactionNetwork.from_json('/home/stef/krxns/data/processed/known_reaction_network.json')
    print("Full reaction network loaded from JSON.")
    print(f"Number of nodes: {G.number_of_nodes()}, Number of edges: {G.number_of_edges()}")
    sources = pd.read_csv('/home/stef/krxns/data/interim/default_sources.csv')['smiles'].tolist()
    G.set_sources(smiles=sources)
    rnmc_lb = 0.25
    pnmc_lb = 0.25
    aug_mc_lb = 0.8
    G.prune(pnmc_lb=pnmc_lb, rnmc_lb=rnmc_lb, source_augmented_pnmc_lb=aug_mc_lb)
    print(f"Pruned reaction network with pnmc_lb={pnmc_lb}, rnmc_lb={rnmc_lb}, source_augmented_pnmc_lb={aug_mc_lb}, and default sources.")
    print(f"Number of nodes after pruning: {G.number_of_nodes()}, Number of edges: {G.number_of_edges()}")

    G = ReactionNetwork.from_json('/home/stef/krxns/data/processed/known_reaction_network.json')
    default_sources = pd.read_csv('/home/stef/krxns/data/interim/default_sources.csv')['smiles'].tolist()
    rnmc_lb = 0.25
    pnmc_lb = 0.25
    aug_mc_lb = 0.8
    addtl_sources = {
    'lactate': 'CC(O)C(=O)O',
    'threonine': 'CC(O)C(N)C(=O)O'
    }
    sources = default_sources + list(addtl_sources.values())
    G.set_sources(smiles=sources)

    targets = {'2-ethyl-2-hydroxy-3-oxobutanoate': 'CCC(O)(C(C)=O)C(=O)O'}
    target = G.get_nodes_by_prop('smiles', targets['2-ethyl-2-hydroxy-3-oxobutanoate'])[0]
    trees = G.enumerate_synthetic_trees(
        target=target,
        max_depth=2,
        max_leaves=3,
        tot_rnmc_lb=0.1
    )
    print(trees)