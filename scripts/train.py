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
from model.model import End2EndModel

# Neural Network Builder functions
from model.builder.transformer import make_transformer
from model.builder.bracketing import make_bracketer
from model.builder.multitask import make_multitask_net
from model.builder.generators import make_generator

from model.utils import (
    eval_model_on_DF,
    str2bool,
    str2list,
    txt2list,
    abs_max_pooling,
    mean_pooling
)

# Import comand line arguments
from args_train import args

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

# Assign threshold if it was given in args
sim_threshold = args.sim_threshold if args.sim_threshold != 666 else None
span = args.span if args.span != 0 else None

transformer_net = make_transformer(output_layer=-2,
                                   device=device)
bracketing_net = make_bracketer(name=args.chunker,
                                device=device,
                                sim_threshold=sim_threshold,
                                span=span)
generator_net = make_generator(pooling=args.pooling,
                               device=device)
multitask_net = make_multitask_net(datasets=args.datasets,
                                   config=config,
                                   device=device)

model = End2EndModel(transformer=transformer_net,
                     bracketer=bracketing_net,
                     generator=generator_net,
                     multitasknet=multitask_net,
                     device=device).to(device)


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

# Create checkpoints dir if it doesn't exist yet
if not os.path.exists('./assets/checkpoints'):
    os.makedirs('./assets/checkpoints')

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
    if (global_counter % eval_periodicity == 0):  #and (global_counter != 0):
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
