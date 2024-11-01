from argparse import ArgumentParser
import json
import pandas as pd
from krxns.utils import str2int
from krxns.config import filepaths
from krxns.net_construction import construct_reaction_network
from krxns.networks import SuperMultiDiGraph
from sklearn.model_selection import StratifiedKFold

parser = ArgumentParser(description="Store shortest path length dataset in chunks")
parser.add_argument("whitelist", help="Filename of coreactants whitelisted as currency")
parser.add_argument("atom_lb", type=float, help="Below this fraction of heavy atoms, reactants will be ignored, i.e., not connected to products")
parser.add_argument("--n-splits", type=int, default=1000, help="How many ways to divide the dataset")
parser.add_argument("--op-cxns", default="sprhea_240310_v3_mapped_operator", help="Filename operator connected adjacency matrices")
parser.add_argument("--rxns", default="sprhea_240310_v3_mapped", help="Reactions dataset")
parser.add_argument("--sim-cxns", default="sprhea_240310_v3_mapped_similarity", help="Filename similarity connected adjacency matrices")
parser.add_argument("--side-cts", default="sprhea_240310_v3_mapped_side_counts", help="Counts of unique, non-currency molecules on each side of reactions")
parser.add_argument("--connect-nontrivial", action="store_true", help="Connect nontrivial reactions w/ similarity connections")
parser.add_argument("--multi-nodes", action="store_true", help="Add multiple molecule nodes to network")

def main(args):
    # Load known reaction data
    with open(filepaths['data'] / f"{args.rxns}.json", 'r') as f:
        rxns = json.load(f)

    # Load op connected reactions
    with open(filepaths['connected_reactions'] / f"{args.op_cxns}.json", 'r') as f:
        op_cxns = str2int(json.load(f))

    # Load sim connected reactions
    with open(filepaths['connected_reactions'] / f"{args.sim_cxns}.json", 'r') as f:
        sim_cxns = str2int(json.load(f))

    with open(filepaths['connected_reactions'] / f"{args.side_cts}.json", 'r') as f:
        side_counts = str2int(json.load(f))

    # Load coreactant whitelist
    with open(filepaths['coreactants'] / f"{args.whitelist}.json", 'r') as f:
        whitelist = json.load(f)

    # Construct network
    G = SuperMultiDiGraph()
    edges, nodes = construct_reaction_network(
        operator_connections=op_cxns,
        similarity_connections=sim_cxns,
        side_counts=side_counts,
        reactions=rxns,
        connect_nontrivial=args.connect_nontrivial,
        coreactant_whitelist=whitelist,
        atom_lb=args.atom_lb,
        add_multi_mol_nodes=args.multi_nodes
    )
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)


    print("Getting shortest paths")
    paths = G.shortest_path() # Get shortest paths

    col_names = ['starter_id', 'target_id', 'starter_smiles', 'target_smiles', 'spl']
    data = []
    for i in paths:
        for j in paths[i]:

            if i == j: # Filter out self paths
                continue

            spl = len(paths[i][j]) - 1
            data.append([i, j, G.nodes[i]['smiles'], G.nodes[j]['smiles'], spl])

    df = pd.DataFrame(data=data, columns=col_names)

    print("Saving")
    save_dir = filepaths['data'] / "shortest_path_length" / f"{args.rxns}_{args.whitelist}_atom_lb_{int(args.atom_lb * 100)}p_multi_nodes_{args.multi_nodes}"
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=1234)
    for i, (_, test_idx) in enumerate(skf.split(df[['starter_id', 'target_id', 'starter_smiles', 'target_smiles']], df[['spl']])):
        chunk = df.loc[test_idx, :]
        chunk.to_parquet(save_dir / f"chunk_{i}.parquet", index=False)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)