import argparse
import os
from pathlib import Path
import pickle

import networkx as nx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-input_dataset",
        "-i",
        type=Path,
        help="path to the text file holding the dataset to convert. Expected each line to be nodeA, nodeB, *features, timestamp",
    )
    parser.add_argument("-output_file", "-o", type=Path, help="destination file")
    parser.add_argument("-start_time", "-s", type=int, default=1, help="first timestamp considered. Any edge declared before then results in undifined behavior")

    args = parser.parse_args()

    file = args.input_dataset

    edge_list = {}
    nb_nodes = 0

    with open(file, "r") as file:
        for (i, line) in enumerate(file):
            elems = list(map(int, line.split(" ")))

            nb_nodes = max([nb_nodes, elems[0], elems[1]])
            if elems[-1] in edge_list:
                 edge_list[elems[-1]].append((elems[0], elems[1], elems[2:-1]))
            else:
                edge_list[elems[-1]] = [(elems[0], elems[1], elems[2:-1])]

    print("{} snaphots in data.".format(len(edge_list)))
    nb_nodes += 1
    max_timestamp = max(edge_list.keys()) + 1

    graph_list = [nx.Graph() for _ in range(args.start_time, max_timestamp)]

    for t, edges in edge_list.items():
        for n in range(nb_nodes):
             graph_list[t-args.start_time].add_node(n)
        for (n1, n2, attr) in edges:
            graph_list[t-args.start_time].add_edge(n1, n2, feat=attr)

    dataset_file = args.output_file

    os.makedirs(dataset_file.parent, exist_ok=True)
    with open(dataset_file, "wb") as file:
        pickle.dump(graph_list, file)
