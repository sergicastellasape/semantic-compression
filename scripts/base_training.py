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
    str2bool,
    str2list,
    filter_indices,
    expand_indices,
    time_since,
    txt2list,
    abs_max_pooling,
    hotfix_pack_padded_sequence
)

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
    "--learning-rate",
    "-lr",
    dest="lr",
    type=float,
    required=False,
    default=0.0001,
    help="Learning rate for Adam optimizer",
)
parser.add_argument(
    "--tensorboard-dir",
    "-tbdir",
    dest="log_dir",
    type=str,
    required=False,
    default="./tensorboard",
    help="rood directory where tensorboard logs are stored. ./tensorboard by default",
)
parser.add_argument(
    "--tensorboard-comment",
    "-tbcomment",
    dest="tensorboard_comment",
    type=str,
    required=False,
    default=None,
)
parser.add_argument(
    "--eval-periodicity",
    "-evalperiod",
    type=int,
    required=False,
    default=50,
    dest="evalperiod",
    help="How often in iterations the model is evaluated",
)
parser.add_argument(
    "--load-checkpoint",
    "-load",
    dest="load_checkpoint",
    required=False,
    action="store_true",
)
parser.add_argument(
    "--wall-time",
    "-wt",
    dest="walltime",
    required=False,
    type=int,
    default=3600,
    help="Walltime for training",
)
parser.add_argument(
    "--train-compression",
    "-tc",
    required=True,
    type=str2bool,
    dest="train_comp",
    help="set if compression happens during training, True or False",
)
parser.add_argument(
    "--eval-compression",
    "-ec",
    required=True,
    type=str2bool,
    dest="eval_comp",
    help="set if compression happens during evaluation, True or False",
)

parser.add_argument(
    "--full-test-eval",
    "-fev",
    required=False,
    default='False',
    type=str2bool,
    dest="full_test_eval",
    help="Set if an evaluation on the full test set is made at the end.",
)

parser.add_argument(
    "--datasets",
    "-dts",
    required=False,
    default='[SST2, QQP]',
    type=str2list,
    dest="datasets",
    help="Set the datasets to train on.",
)


args = parser.parse_args()
if args.load_checkpoint:
    assert os.path.exists(
        f"./assets/checkpoints/{args.run_id}.pt"
    ), "Checkpoint for run_id doesn't exist!"

# load config file from datasets
with open("./config/datasets.yml", "r") as file:
    config = yaml.load(file, Loader=yaml.Loader)
with open("./config/model.yml", "r") as file:
    model_config = yaml.load(file, Loader=yaml.Loader)

# modify the datasets according to the arg passed
config["datasets"] = args.datasets

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
        chunk_size_limit=60,
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
                                     task='SST2',
                                     dropout=0.3,
                                     n_sentiments=4,
                                     pool_mode="concat",
                                     device=device).to(device)

seq_pair_classifier = SeqPairFancyClassifier(embedding_dim=768,
                                             num_classes=2,
                                             task='QQP',
                                             dropout=0.3,
                                             n_attention_vecs=2,
                                             device=device).to(device)

seq_pair_classifier2 = SeqPairFancyClassifier(embedding_dim=768,
                                              num_classes=3,
                                              task='MNLI',
                                              dropout=0.3,
                                              n_attention_vecs=2,
                                              device=device).to(device)

pooling_classifier = NaivePoolingClassifier(embedding_dim=768,
                                            num_classes=2,
                                            task='QQP',
                                            dropout=0.03,
                                            device=device).to(device)

multitask_net = MultiTaskNet(pooling_classifier,
                             device=device).to(device)

model = End2EndModel(transformer=transformer_net,
                     bracketer=bracketing_net,
                     generator=generator_net,
                     multitasknet=multitask_net,
                     device=device).to(device)

for dataset in config['datasets']:
    assert dataset in model.multitasknet.parallel_net_dict.keys(), \
        f"You forgot to add a model for task {dataset}"

##########################################################################
########################## DEFINE CONSTANTS ##############################
torch.manual_seed(10)
LOG_DIR = args.log_dir
run_identifier = args.run_id
eval_periodicity = args.evalperiod

# Tensorboard init
writer = SummaryWriter(
    log_dir=os.path.join(LOG_DIR, run_identifier), comment=args.tensorboard_comment
)
checkpoints_path = os.path.join(
    "./assets/checkpoints/", f"{run_identifier}.pt")
if args.load_checkpoint:
    print(f'Loading checkpoint from {checkpoints_path}')
    model.load_state_dict(torch.load(checkpoints_path))

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
get_batch_function = {
    dataset: config[dataset]["get_batch_fn"] for dataset in config["datasets"]
}
dev_dataframes_dict = {
    dataset: dataframes[dataset]["dev"] for dataset in config["datasets"]
}
test_dataframes_dict = {
    dataset: dataframes[dataset]["test"] for dataset in config["datasets"]
}
batch_indices = {}

global_counter, batch_loss, max_acc = 0, 0, 0


##########################################################################
########################### ACUTAL TRAINING ##############################
initial_time = time.time()
optimizer = torch.optim.Adam(
    multitask_net.parameters(),
    lr=args.lr,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0.0001,
    amsgrad=False,
)

finished_training = False
t = time.time()
while not finished_training:
    # Reset counter for dataset if it's been finished
    for dataset in config["datasets"]:
        if counter[dataset] >= n_batches[dataset] or global_counter == 0:
            print(f"NEW EPOCH STARTED FOR DATASET {dataset}!")
            counter[dataset] = 0
            batch_indices[dataset] = None
            torch.cuda.empty_cache()
            # Re-shuffle the training batches data
            batch_indices[dataset] = torch.randperm(
                n_batches[dataset] * batch_size[dataset], device=torch.device("cpu")
            ).reshape(-1, batch_size[dataset])

    # Generate new batch
    batch_sequences, batch_targets, batch_slices, j = [], {}, {}, 0
    for dataset in config["datasets"]:
        idx = counter[dataset]
        dataset_batch = get_batch_function[dataset](
            dataframes[dataset]["train"], batch_indices[dataset][idx, :]
        )
        # List of tensors, one for each task
        try:
            batch_targets[dataset] = torch.tensor([data[1] for data in dataset_batch],
                                                  dtype=torch.int64, device=device)
        except Exception as error:
            L = [data[1] for data in dataset_batch]
            raise Exception(
                f"This thing failed when the target tensor was in dataset \
                              {dataset}: {L}, indices: {batch_indices[dataset][idx, :]}"
            )

        # Big list combining the input sequences/ tuple of sequences because the batch needs
        # to be at the same "depth" level
        batch_sequences.extend([data[0] for data in dataset_batch])
        batch_slices[dataset] = slice(j, j + len(dataset_batch))
        j += len(dataset_batch)
        counter[dataset] += 1

    # Forward pass
    model.train()
    batch_predictions = model.forward(batch_sequences,
                                      batch_slices=batch_slices,
                                      compression=args.train_comp,
                                      return_comp_rate=False)
    L = model.loss(batch_predictions, batch_targets, weights=None)
    metrics = model.metrics(batch_predictions, batch_targets)
    # print('training metrics:', metrics)
    # Log to tensorboard
    writer.add_scalar(f"loss/train/{run_identifier}", L.item(), global_counter)
    writer.add_scalars(f"metrics/train/{run_identifier}", metrics, global_counter)
    # Update net
    optimizer.zero_grad()
    L.backward()
    optimizer.step()
    batch_loss += L.item()
    if (global_counter % eval_periodicity == 0) and (global_counter != 0):
        print(
            f"################### GLOBAL COUNTER {global_counter} ###################"
        )
        print(f"Iterations per second: {eval_periodicity/(time.time()-t)}")
        # evaluate on dev sets
        model.eval()
        metrics_dict, compression_dict = eval_model_on_DF(
            model,
            dev_dataframes_dict,
            get_batch_function,
            batch_size=16,
            global_counter=global_counter,
            compression=args.eval_comp,
            return_comp_rate=True,
            device=device,
        )
        avg_acc = sum(metrics_dict.values()) / len(metrics_dict)
        if avg_acc > max_acc:
            torch.cuda.empty_cache()
            torch.save(model.state_dict(), checkpoints_path)
            max_acc = avg_acc
            print("NEW CHECKPOINT SAVED!")

        print("Eval metrics:", metrics_dict)
        print(f"Global Loss: {batch_loss/eval_periodicity}")
        print(f"Compression Rates:", compression_dict)
        batch_loss, t = 0, time.time()
        # Log to tensorboard
        writer.add_scalars(f"metrics/dev/{run_identifier}", metrics_dict, global_counter)
    global_counter += 1
    finished_training = True if (
        time.time() - initial_time) > args.walltime else False

if args.full_test_eval:
    print("########## FINAL EVAL ON FULL TEST SET #############")
    # load best checkpoint
    model.load_state_dict(torch.load(checkpoints_path))
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
    writer.add_scalars(f"metrics/test/{run_identifier}", metrics_dict, 0)