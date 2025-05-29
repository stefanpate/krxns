from typing import Iterable
import pandas as pd

def construct_reaction_network(
        mass_inlinks: dict[str, dict[str, dict[str, float]]],
        compounds: pd.DataFrame,
        sources: Iterable[str] = [],
    ):
    '''
    Args
    ----
    mass_inlinks:dict
        {reaction_id: {product_id: {reactant_id: fraction_atoms_from_reactant}, ...}, ...}
        Fraction of atoms in a product coming from a reactant.
    compounds:pd.DataFrame
        DataFrame containing compound information with 'id', 'smiles' and 'name' columns.
    
    Returns
    -------
    edges:list[tuple]
        Entries are (from:int, to:int, properties:dict)
    nodes:list[tuple]
        Entries are (id:int, properties:dict)
    '''
    edges = []
    nodes = {}
    ep = 5e-3
    for rid, pdt_inlinks in mass_inlinks.items():
        for pdt_id, rcts in pdt_inlinks.items():
            this_sources = set(u for u in rcts if u in sources)
            for rct_id, mass_frac in rcts.items():
                source_mass = sum(rcts[s] for s in this_sources - {rct_id})

                if (mass_frac + source_mass) >= 1.0 - ep:
                    edges.append(
                        (
                            rct_id,
                            pdt_id,
                            {
                                'reaction_id': rid,
                                'mass_frac': mass_frac,
                                'coreactants': this_sources,
                                'coproducts': set(pdt_inlinks.keys()) - {pdt_id},
                            }
                        )
                    )

                    nodes[rct_id] = (rct_id, compounds.loc[compounds.id == rct_id, ['smiles', 'name']].to_dict())
                    nodes[pdt_id] = (pdt_id, compounds.loc[compounds.id == pdt_id, ['smiles', 'name']].to_dict())

    return edges, nodes
       
if __name__ == '__main__':
    import json
    from pathlib import Path
    root_dir = Path(__file__).parent.parent
    kcs = pd.read_csv(root_dir / "data/interim/compounds.csv")
    with open(root_dir / "data/interim/mass_links.json", 'r') as f:
        mass_inlinks = json.load(f)

    edges, nodes = construct_reaction_network(
        mass_inlinks=mass_inlinks,
        compounds=kcs,
        sources=[]
    )