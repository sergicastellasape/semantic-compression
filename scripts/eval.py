# Generic
import argparse
import yaml
import time
import gc
import datetime
import os
import sys

# Data & Math
import math
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

# Transformers
from transformers import BertModel, BertTokenizer

# Custom imports
sys.path.append(".")  # Add parent directory to sys.path to import from root dir
from model.model import MultiTaskNet, End2EndModel
from model.classifiers import (
    AttentionClassifier,
    SeqPairAttentionClassifier,
    NaivePoolingClassifier,
    SeqPairFancyClassifier,
)
from model.generators import IdentityGenerator, EmbeddingGenerator
from model.bracketing import (
    IdentityChunker,
    NNSimilarityChunker,
    AgglomerativeClusteringChunker,
    cos,
)
from model.transformer import Transformer
from model.data_utils import get_batch_SST2_from_indices, get_batch_QQP_from_indices
from model.utils import (
    eval_model_on_DF,
    make_connectivity_matrix,
    add_space_to_special_characters,
    filter_indices,
    expand_indices,
    time_since,
    txt2list,
    abs_max_pooling,
    hotfix_pack_padded_sequence
)

def str2bool(v):
    """
    To pass True or False boolean arguments
    in argparse. Code from stackoverflow.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Model Options")
parser.add_argument(
    "--run-identifier",
    "-id",
    dest="run_id",
    type=str,
    required=True,
    help="Add an identifier that will be used to store the run in tensorboard.",
)

parser.add_argument(
    "--similarity-threshold",
    "-thr",
    dest="sim_threshold",
    type=float,
    required=True,
    help="Similarity threshold used for chunking in the embedding space.",
)

parser.add_argument(
    "--chunker",
    dest="chunker",
    type=str,
    required=True,
    choices=["NNSimilarity", "agglomerative"],
    help="Specify the bracketing part of the net",
)

parser.add_argument(
    "--eval-compression",
    "-ec",
    required=True,
    type=str2bool,
    dest="eval_comp",
    help="set if compression happens during evaluation, True or False",
)


args = parser.parse_args()
assert os.path.exists(
    f"./assets/checkpoints/{args.run_id}.pt"
), "Checkpoint for run_id doesn't exist!"

# load config file from datasets
with open("./config/datasets.yml", "r") as file:
    config = yaml.load(file, Loader=yaml.Loader)
with open("./config/model.yml", "r") as file:
    model_config = yaml.load(file, Loader=yaml.Loader)

#############################################################################
############################### LOAD DATASETS ###############################
print("Loading datasets...")
dataframes = {}
for dataset in config["datasets"]:
    dataframes[dataset] = {}
    for kind in ["train", "test", "dev"]:
        dataframes[dataset][kind] = pd.read_csv(
            config[dataset]["path"][kind], sep="\t")


#############################################################################
############################### LOAD MODELS #################################
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device being used: {device}")

transformer_net = Transformer(
    model_class=BertModel,
    tokenizer_class=BertTokenizer,
    pre_trained_weights="bert-base-uncased",
    output_layer=-2,
    device=device,
)

if args.chunker == "NNSimilarity":
    print("Using NNsimilarity chunker")
    bracketing_net = NNSimilarityChunker(
        sim_function=cos,
        threshold=args.sim_threshold,
        exclude_special_tokens=False,
        combinatorics="sequential",
        chunk_size_limit=4,
        device=device,
    )
elif args.chunker == "agglomerative":
    print("Using AGGLOMERATIVE chunker")
    bracketing_net = AgglomerativeClusteringChunker(
        threshold=args.sim_threshold, device=device
    )
else:
    raise ValueError("You must pass a valid chunker as an argument!")

generator_net = EmbeddingGenerator(pool_function=abs_max_pooling,
                                   device=device)

seq_classifier = AttentionClassifier(embedding_dim=768,
                                     sentset_size=2,
                                     dropout=0.3,
                                     n_sentiments=4,
                                     pool_mode="concat",
                                     device=device).to(device)

seq_pair_classifier = SeqPairFancyClassifier(embedding_dim=768,
                                             num_classes=2,
                                             dropout=0.3,
                                             n_attention_vecs=2,
                                             device=device).to(device)

naive_classifier = NaivePoolingClassifier(embedding_dim=768,
                                          num_classes=2,
                                          dropout=0.0,
                                          pool_mode="max_pooling",
                                          device=device).to(device)

multitask_net = MultiTaskNet(seq_classifier, seq_pair_classifier, device=device).to(device)

model = End2EndModel(transformer=transformer_net,
                     bracketer=bracketing_net,
                     generator=generator_net,
                     multitasknet=multitask_net,
                     device=device).to(device)


##########################################################################
########################## DEFINE CONSTANTS ##############################
torch.manual_seed(10)
run_identifier = args.run_id

checkpoints_path = os.path.join(
    "./assets/checkpoints/", f"{run_identifier}.pt")
print('checkpoints path', checkpoints_path)
model.load_state_dict(torch.load(checkpoints_path, map_location=device))


# LOAD CONFIG DICTS AND CREATE NEW ONES FROM THOSE
counter = {dataset: config[dataset]["counter"]
           for dataset in config["datasets"]}
batch_size = {dataset: config[dataset]["batch_size"]
              for dataset in config["datasets"]}
n_batches = {
    dataset: math.floor(
        len(dataframes[dataset]["train"]) / batch_size[dataset])
    for dataset in config["datasets"]
}
get_batch_function = {dataset: config[dataset]["get_batch_fn"]
                      for dataset in config["datasets"]}
dev_dataframes_dict = {dataset: dataframes[dataset]["dev"]
                       for dataset in config["datasets"]}
test_dataframes_dict = {dataset: dataframes[dataset]["test"]
                        for dataset in config["datasets"]}
batch_indices = {}

global_counter, batch_loss, max_acc = 0, 0, 0

# load best checkpoint
model.eval()
metrics_dict, compression_dict = eval_model_on_DF(
    model,
    test_dataframes_dict,
    get_batch_function,
    batch_size=16,
    global_counter=global_counter,
    compression=args.eval_comp,
    return_comp_rate=True,
    device=device,
)
print("Full test set losses: ", metrics_dict)
print("Compression on test sets:", compression_dict)
