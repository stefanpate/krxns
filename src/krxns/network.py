import networkx as nx
from networkx.exception import NetworkXNoPath
from typing import Any
from copy import deepcopy
from typing import Iterable
import pandas as pd
from dataclasses import dataclass, field
from collections import deque
from itertools import product

class ReactionNetwork(nx.MultiDiGraph):
    def __init__(self, incoming_graph_data=None, multigraph_input=None, **attr):
        super().__init__(incoming_graph_data, multigraph_input, **attr)
    
    def add_edges_from(self, ebunch_to_add, **attr):
        multi_keys =  super().add_edges_from(ebunch_to_add, **attr)
        ij2k = {}
        for edge, k in zip(ebunch_to_add, multi_keys):
            ij = edge[:2]
            if ij in ij2k:
                ij2k[ij].append(k)
            else:
                ij2k[ij] = [k]

        self.ij2k = ij2k

    def get_nodes_by_prop(self, prop:str, value:Any) -> list[int]:
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

def construct_reaction_network(
        mass_contributions: dict[str, str | dict[str, dict[str, float]]],
        compounds: pd.DataFrame,
        sources: Iterable[int] = [],
        rnmc_lb: float = 0,
        pnmc_lb: float = 0,
        tot_mass_lb: float = 1.0,
    ):
    '''
    Args
    ----
    mass_contributions:dict[str, str or dict[str, dict[str, float]]]
        With differently normalized mass contributions:
        {
            "am_smarts": reaction,
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
    compounds:pd.DataFrame
        DataFrame containing compound information with 'id', 'smiles' and 'name' columns.
    sources:Iterable[int]
        List of source compound IDs to consider for mass balance. If empty, all compounds are considered.
    rnmc_lb:float
        Lower bound for reactant normalized mass contribution from reactant.
    pnmc_lb:float
        Lower bound for product normalized mass contribution from reactant.
    tot_mass_lb:float
        Lower bound for total mass contribution from reactant and sources.
    
    Returns
    -------
    edges:list[tuple]
        Entries are (from:int, to:int, properties:dict)
    nodes:list[tuple]
        Entries are (id:int, properties:dict)
    '''
    edges = []
    nodes = {}
    ep = 1e-3
    for rid, entry in mass_contributions.items():
        rid = int(rid)
        am_smarts = entry.get('am_smarts', None)
        rct_normed_mass_contrib = entry.get('rct_normed_mass_contrib', {})
        pdt_normed_mass_contrib = entry.get('pdt_normed_mass_contrib', {})
        for pdt_id, rcts in pdt_normed_mass_contrib.items():
            pdt_id = int(pdt_id)
            rcts = {int(k): v for k, v in rcts.items()}
            this_sources = set(u for u in rcts if u in sources)
            
            for rct_id, pnmc in rcts.items():
                source_mass = sum(rcts[s] for s in this_sources - {rct_id}) # Mass contribution from designated sources
                rnmc = rct_normed_mass_contrib[str(pdt_id)][str(rct_id)]

                # Reactant must contribute more than lower bounds on both pdt- and rct- 
                # normed mass contributions and together
                # w/ the designated sources must contribue all the mass (minus fudge factor)
                if pnmc >= pnmc_lb and rnmc >= rnmc_lb and (pnmc + source_mass) >= tot_mass_lb - ep:
                    edges.append(
                        (
                            rct_id,
                            pdt_id,
                            {
                                'reaction_id': rid,
                                'pdt_normed_mass_frac': pnmc,
                                'rct_normed_mass_frac': rnmc,
                                'am_smarts': am_smarts,
                                'coreactants': tuple(this_sources),
                                'coproducts': tuple(set(int(k) for k in pdt_normed_mass_contrib.keys()) - {pdt_id}),
                            }
                        )
                    )

                    nodes[rct_id] = (rct_id, compounds.loc[compounds.id == rct_id, ['smiles', 'name']].to_dict('records')[0])
                    nodes[pdt_id] = (pdt_id, compounds.loc[compounds.id == pdt_id, ['smiles', 'name']].to_dict('records')[0])

    return edges, list(nodes.values())

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