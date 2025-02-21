import argparse

import pyrootutils
import torch
from torch_geometric_temporal import DynamicGraphTemporalSignal
from torch_geometric_temporal.dataset import (
    EnglandCovidDatasetLoader,
    TwitterTennisDatasetLoader,
)

from brainiac_temporal.data.datasets import *
from brainiac_temporal.data.utils import remove_isolated_nodes

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--dataset_name",
    type=str,
    default="insecta",
)

root = pyrootutils.setup_root(
    search_from="sota_paper_implementation",
    indicator="convertors",
    pythonpath=True,
    cwd=True,
)

args = parser.parse_args()
dataset_name = args.dataset_name

import os

from convertors.damnets2tgt import convert_tgt_to_damnets

path = lambda name: os.path.join(root, "damnets-tagGen-dymonds", "data", f"{name}.pkl")


# Do your dataset here.
if dataset_name == "insecta":
    reference_dataset = fetch_insecta_dataset(colony=6)
elif dataset_name == "imdb":
    reference_dataset = load_imdb_dynamic_tgt()
elif dataset_name in ["wikismall", "bitcoin", "reddit"]:
    reference_dataset = load_tigger_datasets_tgt(dataset_name)
elif dataset_name == "encovid":
    reference_dataset = EnglandCovidDatasetLoader().get_dataset()
elif dataset_name == "twittertennis":
    reference_dataset = TwitterTennisDatasetLoader().get_dataset()
else:
    raise ValueError(f"Unknown dataset {dataset_name}")

dataset = remove_isolated_nodes(reference_dataset)

convert_tgt_to_damnets(dataset, path(dataset_name))
