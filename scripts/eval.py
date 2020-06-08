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
assert args.checkpoint_id is not None, "You must provide checkpoint-id!"

if not os.path.exists(LOGGING_PATH):
    os.makedirs(LOGGING_PATH)

logging.basicConfig(filename=os.path.join(LOGGING_PATH, f'{args.run_id}.txt'),
                    filemode='a',
                    format='%(asctime)s | %(levelname)s : %(message)s',
                    datefmt='%H:%M:%S',
                    level=log_level[args.log_level])

logging.getLogger("transformers").setLevel(logging.WARNING)

# load config file from datasets
with open("./config/datasets.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.Loader)
with open("./config/model.yml", "r") as f:
    model_config = yaml.load(f, Loader=yaml.Loader)

# modify the datasets according to the arg passed
if args.datasets is not None:
    config["datasets"] = args.datasets
else:
    args.datasets = config["datasets"]

if args.eval_comp:
    assert args.pooling is not None
    assert args.chunker is not None

if args.write_google_sheet:
    assert int(args.run_id[-1]), "Run identifier must end with version number from 0 to 9!"

################################################################################
################################ LOAD DATASETS #################################
logging.info(f"Loading datasets: {config['datasets']}")
dataframes = {}
for dataset in config["datasets"]:
    dataframes[dataset] = {}
    for kind in ["train", "test", "dev"]:
        dataframes[dataset][kind] = pd.read_csv(
            config[dataset]["path"][kind], sep="\t")

################################################################################
################################# LOAD MODELS ##################################
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.info(f"DEVICE: {device}")

transformer_net = make_transformer(args, device=device)
bracketing_net = make_bracketer(args, device=device)
generator_net = make_generator(args, device=device)
multitask_net = make_multitask_net(args, config, device=device)
model = End2EndModel(transformer=transformer_net,
                     bracketer=bracketing_net,
                     generator=generator_net,
                     multitasknet=multitask_net,
                     device=device).to(device)

logging.info(f"Compression in Training: {args.train_comp}")
logging.info(f"Compression in Evaluation: {args.eval_comp}")

################################################################################
############################## DEFINE CONSTANTS ################################
#torch.manual_seed(0)

# LOAD CONFIG DICTS AND CREATE NEW ONES FROM THOSE
get_batch_function = {
    dataset: config[dataset]["get_batch_fn"] for dataset in config["datasets"]
}
test_dataframes_dict = {
    dataset: dataframes[dataset]["test"] for dataset in config["datasets"]
}

# Load checkpoint from modules in --load-modules argument
model.load_modules(args.checkpoint_id,
                   modules=args.modules_to_load,
                   parent_path='./assets/checkpoints')

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
logging.info(f"Full test set losses: {metrics_dict}")

if args.write_google_sheet:
    row = args.trf_out_layer + 2
    run_v = int(args.run_id[-1])
    write_google_sheet(metrics_dict,
                       row=row,
                       name=config['google_sheet']['name'],
                       sheet_name=f'run{run_v}')
