import networkx as nx
from networkx.exception import NetworkXNoPath
from typing import Any
from copy import deepcopy

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
        
    def shortest_path(self, source, target) -> list:
        target_smiles = self.nodes[target]['smiles']
        to_remove = [(i, j) for i, j, props in self.edges(data=True) if target_smiles in props['requires']]
        pruned = deepcopy(self)
        pruned.remove_edges_from(to_remove)
        try:
            node_path = nx.shortest_path(pruned, source, target)
        except NetworkXNoPath as e:
            print(e)
            return [], [] # No path found
        
        edge_path = []
        for i in range(len(node_path) - 1):
            edge_path.append(pruned.get_edges_between(node_path[i], node_path[i+1]))

        return node_path, edge_path