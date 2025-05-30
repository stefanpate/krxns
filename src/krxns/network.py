import networkx as nx
from networkx.exception import NetworkXNoPath
from typing import Any
from copy import deepcopy
from typing import Iterable
import pandas as pd

class SuperMultiDiGraph(nx.MultiDiGraph):
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
        mass_contributions: dict[str, str or dict[str, dict[str, float]]],
        compounds: pd.DataFrame,
        sources: Iterable[int] = [],
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
    
    Returns
    -------
    edges:list[tuple]
        Entries are (from:int, to:int, properties:dict)
    nodes:list[tuple]
        Entries are (id:int, properties:dict)
    '''
    edges = []
    nodes = {}
    ep = 1e-2
    for rid, entry in mass_contributions.items():
        rid = int(rid)
        am_smarts = entry.get('am_smarts', None)
        rct_normed_mass_contrib = entry.get('rct_normed_mass_contrib', {})
        pdt_normed_mass_contrib = entry.get('pdt_normed_mass_contrib', {})
        for pdt_id, rcts in pdt_normed_mass_contrib.items():
            pdt_id = int(pdt_id)
            rcts = {int(k): v for k, v in rcts.items()}
            this_sources = set(u for u in rcts if u in sources)
            
            # Reaction must not require more rcts than |sources| + 1
            if len(this_sources) < len(rcts) -1:
                break
            
            for rct_id, mass_frac in rcts.items():
                source_mass = sum(rcts[s] for s in this_sources - {rct_id})

                # Reactant must contribute some mass to product on its own and
                # together with sources must contribue all the mass (minus fudge factor)
                if mass_frac > 0 and (mass_frac + source_mass) >= 1.0 - ep:
                    edges.append(
                        (
                            rct_id,
                            pdt_id,
                            {
                                'reaction_id': rid,
                                'pdt_normed_mass_frac': mass_frac,
                                'rct_normed_mass_frac': rct_normed_mass_contrib[str(pdt_id)][str(rct_id)],
                                'am_smarts': am_smarts,
                                'coreactants': this_sources,
                                'coproducts': set(int(k) for k in pdt_normed_mass_contrib.keys()) - {pdt_id},
                            }
                        )
                    )

                    nodes[rct_id] = (rct_id, compounds.loc[compounds.id == rct_id, ['smiles', 'name']].to_dict('records')[0])
                    nodes[pdt_id] = (pdt_id, compounds.loc[compounds.id == pdt_id, ['smiles', 'name']].to_dict('records')[0])

    return edges, list(nodes.values())
       
if __name__ == '__main__':
    import json
    from pathlib import Path
    root_dir = Path(__file__).parent.parent.parent
    kcs = pd.read_csv(root_dir / "data/interim/compounds.csv")
    with open(root_dir / "data/interim/mass_links.json", 'r') as f:
        mass_contributions = json.load(f)

    edges, nodes = construct_reaction_network(
        mass_contributions=mass_contributions,
        compounds=kcs,
        sources=[]
    )