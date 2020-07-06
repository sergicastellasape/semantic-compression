# Generic
import argparse
import yaml
import time
import gc
import datetime
import logging
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
from model.utils import write_google_sheet

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
from args import args, LOGGING_PATH
log_level = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING
}

if not os.path.exists(LOGGING_PATH):
    os.makedirs(LOGGING_PATH)

logging.basicConfig(filename=os.path.join(LOGGING_PATH, f'{args.run_id}.txt'),
                    filemode='a',
                    format='%(asctime)s | %(levelname)s : %(message)s',
                    datefmt='%H:%M:%S',
                    level=log_level[args.log_level])

logging.getLogger("transformers").setLevel(logging.WARNING)

# Load config file from datasets
with open(f"./config/{args.datasets_config}.yml", "r") as f:
    datasets_config = yaml.load(f, Loader=yaml.Loader)
with open(f"./config/{args.model_config}.yml", "r") as f:
    model_config = yaml.load(f, Loader=yaml.Loader)
with open(f"./config/{args.optimizer_config}.yml", "r") as f:
    optimizer_config = yaml.load(f, Loader=yaml.Loader)

# Modify the datasets according to the arg passed, if argument is empty
if args.datasets is None:
    args.datasets = datasets_config["datasets"]

# Raise error if compression is activated but no chunker or pooling is provided
if args.train_comp or args.eval_comp:
    assert args.pooling is not None, "You must pass a --pooling arg if compression is activated!"
    assert args.chunker is not None, "You must pass a --chunker arg if compression is activated!"

# Assign agg_layer as the output of the transformer if no specific agg_layer is
# passed as an argument
if args.agg_layer is None:
    args.agg_layer = args.trf_out_layer

if args.write_google_sheet:
    assert int(args.run_id[-1]), "Run identifier must end with version number from 0 to 9!"

################################################################################
################################ LOAD DATASETS #################################
logging.info(f"Loading datasets: {args.datasets}")
dataframes = {}
for dataset in args.datasets:
    dataframes[dataset] = {}
    for kind in ["train", "test", "dev"]:
        dataframes[dataset][kind] = pd.read_csv(
            datasets_config[dataset]["path"][kind], sep="\t")

################################################################################
################################# LOAD MODELS ##################################
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.info(f"DEVICE: {device}")

transformer_net = make_transformer(args, device=device)
bracketing_net = make_bracketer(args, device=device)
generator_net = make_generator(args, device=device)
multitask_net = make_multitask_net(args,
                                   datasets_config,
                                   model_config,
                                   device=device)
model = End2EndModel(transformer=transformer_net,
                     bracketer=bracketing_net,
                     generator=generator_net,
                     multitasknet=multitask_net,
                     device=device).to(device)

logging.info(f"Compression in Training: {args.train_comp}")
logging.info(f"Compression in Evaluation: {args.eval_comp}")
logging.info(f"Transformer Layer used: {args.trf_out_layer}")

################################################################################
############################## DEFINE CONSTANTS ################################
#torch.manual_seed(0)

# Tensorboard initialization
writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.run_id))

# Load tensorboard's global counter, if it exists
counter_path = os.path.join(args.log_dir, args.run_id, 'global_counter.pt')
if os.path.exists(counter_path):
    logging.info("Resuming global counter from Tensorboard events directory...")
    global_counter = int(torch.load(counter_path))
else:
    global_counter = 0

# Load checkpoint from modules in --load-modules argument
model.load_modules(args.checkpoint_id,
                   modules=args.modules_to_load,
                   parent_path='./assets/checkpoints')

# Load configuration dicts and create new ones for couner, batch, functions, etc.
counter = {dataset: datasets_config[dataset]["counter"]
           for dataset in args.datasets}
batch_size = {dataset: datasets_config[dataset]["batch_size"]
              for dataset in args.datasets}
n_batches = {
    dataset: math.floor(
        len(dataframes[dataset]["train"]) / batch_size[dataset])
    for dataset in args.datasets
}
get_batch_function = {dataset: datasets_config[dataset]["get_batch_fn"]
                      for dataset in args.datasets}
dev_dataframes_dict = {dataset: dataframes[dataset]["dev"]
                       for dataset in args.datasets}
test_dataframes_dict = {dataset: dataframes[dataset]["test"]
                        for dataset in args.datasets}
batch_indices = {}

################################################################################
############################### ACUTAL TRAINING ################################
initial_time = time.time()

logging.info(f"Training modules: {args.modules_to_train}")
params = []
for module_name in args.modules_to_train:
    params += list(getattr(model, module_name).parameters())

optimizer = torch.optim.Adam(
    params,
    lr=optimizer_config['learning_rate'],
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0.0001,
    amsgrad=False,
)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=optimizer_config['milestones'],
                                                 gamma=optimizer_config['gamma'])

finished_training, first_iteration = False, True
batch_loss, max_acc = 0, 0
t = time.time()
while not finished_training:
    # Reset counter for dataset if it's been finished
    for dataset in args.datasets:
        if counter[dataset] >= n_batches[dataset] or first_iteration:
            logging.info(f"NEW EPOCH STARTED FOR DATASET {dataset}!")
            counter[dataset] = 0
            batch_indices[dataset] = None
            torch.cuda.empty_cache()
            # Re-shuffle the training batches data
            batch_indices[dataset] = torch.randperm(
                n_batches[dataset] * batch_size[dataset], device=torch.device("cpu")
            ).reshape(-1, batch_size[dataset])

    # Generate new batch
    batch_sequences, batch_targets, batch_slices, j = [], {}, {}, 0
    for dataset in args.datasets:
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

        # Big list combining the input sequences/ tuple of sequences because
        # the batch needs to be at the same "depth" level
        batch_sequences.extend([data[0] for data in dataset_batch])
        batch_slices[dataset] = slice(j, j + len(dataset_batch))
        j += len(dataset_batch)
        counter[dataset] += 1

    # Forward pass
    model.train()
    model.transformer.eval()

    batch_predictions = model.forward(batch_sequences,
                                      batch_slices=batch_slices,
                                      compression=args.train_comp,
                                      return_comp_rate=False,
                                      max_length=model_config['max_length'])
    L = model.loss(batch_predictions, batch_targets, weights=None)
    metrics = model.metrics(batch_predictions, batch_targets)

    # Log to tensorboard
    writer.add_scalar(f"loss/train/{args.run_id}", L.item(), global_step=global_counter)
    writer.add_scalars(f"metrics/train/{args.run_id}", metrics, global_step=global_counter)

    # Update net
    optimizer.zero_grad()
    L.backward()
    optimizer.step()
    scheduler.step()
    batch_loss += L.item()

    if (global_counter % args.evalperiod == 0):  # and (global_counter != 0):
        logging.info(
            f"################### GLOBAL COUNTER {global_counter} ###################"
        )
        logging.info(f"Iterations per second: {args.evalperiod / (time.time()-t)}")
        # Evaluate on dev sets
        model.eval()
        metrics_dict, compression_dict = eval_model_on_DF(
            model,
            dev_dataframes_dict,
            get_batch_function,
            batch_size=16,
            compression=args.eval_comp,
            return_comp_rate=True,
            max_length=model_config['max_length'],
            device=device,
        )
        avg_acc = sum(metrics_dict.values()) / len(metrics_dict)
        if avg_acc > max_acc:
            torch.cuda.empty_cache()
            model.save_modules(args.run_id,
                               modules=args.modules_to_save,
                               parent_path='./assets/checkpoints/')
            max_acc = avg_acc
            logging.info("NEW CHECKPOINT SAVED!")

        logging.info(f"Eval metrics: {metrics_dict}")
        logging.info(f"Global Loss: {batch_loss / args.evalperiod}")
        logging.info(f"Compression Rates: {compression_dict}")
        if optimizer.__dict__['param_groups'][0]['lr']:
            logging.info(f"Current Learning Rate: {optimizer.__dict__['param_groups'][0]['lr']}")
        batch_loss, t = 0, time.time()
        # Log to tensorboard
        writer.add_scalars(f"metrics/dev/{args.run_id}",
                           metrics_dict, global_step=global_counter)

    first_iteration = False
    # Update the saved global_counter in tensorboard directory
    torch.save(global_counter, counter_path)

    # Check if training should be finished accroding to --wall-time or --wall-steps
    finished_training = True if (time.time() - initial_time) > args.walltime or \
        global_counter >= args.wallsteps else False

    global_counter += 1

if args.full_test_eval:
    logging.info("########## FINAL EVAL ON FULL TEST SET #############")
    # Load 'best' performing checkpoint according to dev set
    model.load_modules(args.run_id,
                       modules=args.modules_to_save,
                       parent_path='./assets/checkpoints/')
    model.eval()
    metrics_dict, compression_dict = eval_model_on_DF(
        model,
        test_dataframes_dict,
        get_batch_function,
        batch_size=16,
        compression=args.eval_comp,
        return_comp_rate=True,
        device=device,
    )
    logging.info(f"Full test set compressions: {compression_dict}")
    logging.info(f"Full test set losses: {metrics_dict}")
    writer.add_scalars(f"metrics/test/{args.run_id}", metrics_dict, 0)

    # Write results on Google Sheets if provided. This should generally *not be
    # used*, unless you have set up credentials in ./config and modified the
    # function write_google_sheet to your needs
    if args.write_google_sheet:
        row = args.trf_out_layer + 2
        run_v = int(args.run_id[-1])
        write_google_sheet(metrics_dict,
                           row=row,
                           name=datasets_config['google_sheet']['name'],
                           sheet_name=f'run{run_v}')
