import pickle
import networkx as nx
import numpy as np
from utils.ba_ts_generator import barabasi_albert_graph_ts
from utils.bipartite_contraction_generator import generate_bipartite_contraction_ts
from utils.comm_decay_generator import generate_3_comm_decay_ts, generate_comm_total_decay_ts
from functools import partial
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map


def generate_graphs(args):
    graph_type = args.name
    if graph_type == "3_comm_decay":
        # fn = n_community_decay_ts
        fn = generate_3_comm_decay_ts
    if graph_type == "3_comm_total_decay":
        fn = generate_comm_total_decay_ts
    # These two are for debugging.
    elif graph_type == "3_comm_const":
        fn = n_community_const_ts
    elif graph_type == "3_comm_interpolation":
        ts = generate_3_comm_decay_ts(c_sizes=args.c_sizes, T=args.T, p_int=args.p_int, p_ext=args.p_ext)
        ts_list = [ts for _ in range(args.N)]
    elif graph_type == "ba":
        fn = barabasi_albert_graph_ts
    elif graph_type == "from_file":
        path = args.dataset_file_path
        with open(path, 'rb') as file:
            graphs = pickle.load(file)
        for g in graphs:
            print(nx.to_numpy_array(g).shape)
        return [graphs, graphs]

    elif graph_type == "bipartite_contraction":
        fn = generate_bipartite_contraction_ts

    if graph_type != "3_comm_interpolation":
        fn = partial(fn, **args)
        fn = partial(wrapper, fn=fn)
        ts_list = process_map(fn, range(args.N), max_workers=args.n_workers)
        # with Pool(args.n_workers) as p:
        #     ts_list = p.map(fn, [_ for _ in range(args.N)])
        # test_list = p.map(fn, [_ for _ in range(args.N)])
    args.T = len(ts_list[0])

    return ts_list


def compute_adj_delta(ts, abs=True):
    """

    Args:
        ts_list: network time series in networkx format
    Returns:
        list of delta matrices for the network time series
    """

    adj_ts = [np.tril(nx.to_numpy_array(g)) for g in ts]
    delta_ts = []
    for t in range(1, len(ts)):
        delta = adj_ts[t] - adj_ts[t - 1]
        if abs:
            delta = np.abs(delta)
        delta_ts.append(delta)
    return delta_ts


def wrapper(x, fn):
    return fn(seed=x)


def n_community_decay_ts(c_sizes, T, p_int=0.7, p_ext=0.01, decay_prop=0.2, **kwargs):
    G = nx.random_partition_graph(c_sizes, p_int, p_ext)
    communities = [G.subgraph(c) for c in G.graph["partition"]]
    decay_edges = list(communities[-1].edges)
    decay_nodes = list(communities[-1].nodes)
    non_decay_nodes = [node for comm in communities[:-1] for node in comm]
    # G = add_external_connections(G, communities, p_ext)

    ts = [G]
    for t in range(1, T):
        G_t = nx.Graph(ts[t - 1])
        # Change a proportion of 'in' edges to 'out' edges in the decay community
        n = int(decay_prop * len(decay_edges))
        for i in range(n):
            ix = np.random.choice(len(decay_edges))
            edge = decay_edges.pop(ix)
            G_t.remove_edge(*edge)
            # Add back the same number of random 'out' edges
            edge_exists = True
            while edge_exists:
                n1 = np.random.choice(decay_nodes)
                n2 = np.random.choice(non_decay_nodes)
                edge_exists = G_t.has_edge(n1, n2)
            G_t.add_edge(n1, n2)
        ts.append(G_t)
    return ts


def n_community_const_ts(c_sizes, T, p_int=0.7, p_ext=0.01, **kwargs):
    # Used for model verification
    ts = []
    G = nx.random_partition_graph(c_sizes, p_int, p_ext)
    for t in range(T):
        ts.append(G)
    return ts


if __name__ == "__main__":
    ts = n_community_decay_ts([5, 5, 5], 5, 0.9, 0.01)
    deltas = compute_adj_delta(ts)
    import matplotlib.pyplot as plt

    nx.draw(ts[0], with_labels=True)
    plt.show()
    nx.draw(ts[1], with_labels=True)
    plt.show()
    print(deltas[0])
    print("fin")
