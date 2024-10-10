
from argparse import ArgumentParser
from krxns.config import filepaths
from krxns.net_construction import construct_reaction_network
from krxns.networks import SuperMultiDiGraph
from krxns.utils import str2int
import json
import gzip
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
from itertools import permutations, chain
import os
from time import perf_counter

def get_paths(chunk, G, save_to):
    chunk_number, chunk = chunk
    res = {}
    for pair in chunk:
        source, target = pair
        node_path, _ = G.shortest_path(source, target, quiet=True)
        if node_path:
            res[">>".join([str(source), str(target)])] = node_path

    res = json.dumps(res).encode('utf-8')

    with gzip.open(save_to / f"chunk_{chunk_number}.json.gz", 'wb') as f:
        f.write(res)

    

def _shortest_paths(args):
    tmp = {k: v for k,v in vars(args).items() if k != 'func'}
    fps = {
        'operator_connections': 'connected_reactions',
        "reactions": 'data',
        "unpaired_cofactors": "cofactors",
        "similarity_connections": 'connected_reactions',
        "side_counts": 'connected_reactions',
        "coreactant_whitelist": "cofactors",
    }

    kwargs = {}
    for k, v in tmp.items():
        if k in fps and v is not None:
            with open(filepaths[fps[k]] / v, 'r') as f:
                arg = json.load(f)

            if k != "coreactant_whitelist":
                arg = str2int(arg)

            kwargs[k] = arg
        
        else:
            kwargs[k] = v

    # Construct network
    G = SuperMultiDiGraph()
    edges, nodes = construct_reaction_network(**kwargs)
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Make superdir
    save_to = filepaths['paths'] / "test"
    if not save_to.exists():
        save_to.mkdir()
    
    # Process
    n_processes = os.cpu_count()
    ds = n_processes * 32
    print(f"Detected {n_processes} processes")
    pairs = list(permutations(G.nodes, 2))[:ds]
    chunksize = len(pairs) // n_processes
    print(f"Chunksize: {chunksize}")
    f = partial(get_paths, G=G, save_to=save_to)
    args = [(n, pairs[i : i + chunksize]) for n, i in enumerate(range(0, len(pairs), chunksize)) ]
    # with mp.Pool(processes=n_processes) as pool:
    #     _ = list(
    #         tqdm(
    #             pool.imap(
    #                 f,
    #                 args
    #             ),
    #             total=len(args)
    #         )
    #     )


parser = ArgumentParser(description="Sample paths from a constructed reaction network")
subparsers = parser.add_subparsers(title="Commands", description="Available commands")

# Connect with operator
parser_shortest = subparsers.add_parser("shortest", help="Get shortest paths between molecules")
parser_shortest.add_argument("operator_connections", help="File with opeartor-connected molecules")
parser_shortest.add_argument("reactions", help="File with reaction info, e.g., SMARTS")
parser_shortest.add_argument("unpaired_cofactors", help="File with unpaired cofactor SMILES as keys")
parser_shortest.add_argument("--similarity_connections", '-s', default={}, help="File with similarity-connected molecules")
parser_shortest.add_argument("--side_counts", '-c', default={}, help="File with unique molecule counts on reaction sides")
parser_shortest.add_argument("--connect_nontrivial", '-n', action="store_true", help="Connect reactions with more than single non-paired-cofactor substrate")
parser_shortest.add_argument("--atom_lb", '-l', type=float, default=0.0, help="Lower bound on atom fraction inlink weight")
parser_shortest.add_argument("--coreactant_whitelist", '-w', help="Acceptable coreactants in the 'requires' edge property")
parser_shortest.set_defaults(func=_shortest_paths)

if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)