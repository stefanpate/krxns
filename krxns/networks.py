import networkx as nx
from typing import Any

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
    
    def get_edges_between(self, from_id:int, to_id:int, k:int = None):
        if k:
            return self.edges[from_id, to_id, k]
        else:
            return [self.edges[from_id, to_id, k] for k in self.ij2k[(from_id, to_id)]]
