import sys
sys.path.append('.') # Add parent directory to sys path to enable to imports from the root dir
import os
import datetime
import math
import gc
import time
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
from transformers import BertModel, BertTokenizer

# Custom imports
from model.utils import *
from model.data_utils import get_batch_SST2_from_indices, get_batch_QQP_from_indices
from model.transformer import Transformer
from model.bracketing import IdentityChunker, NNSimilarityChunker, cos
from model.generators import IdentityGenerator, EmbeddingGenerator
from model.classifiers import AttentionClassifier, SeqPairAttentionClassifier, NaivePoolingClassifier, SeqPairFancyClassifier
from model.model import MultiTaskNet, End2EndModel


parser = argparse.ArgumentParser(description='Model Options')
parser.add_argument('--run-identifier', '-id', 
                    dest='run_id', type=str, required=True,
                    help='Add an identifier that will be used to store the run in tensorboard.')
parser.add_argument('--similarity-threshold', '-thr', 
                    dest='sim_threshold', type=float, required=True,
                    help='Similarity threshold used for chunking in the embedding space.')
parser.add_argument('--learning-rate', '-lr',
                    dest='lr', type=float, required=False, default=0.0001,
                    help="Learning rate for Adam optimizer")
parser.add_argument('--tensorboard-dir', '-tbdir',
                    dest='log_dir', type=str, required=False, default='./tensorboard',
                    help='rood directory where tensorboard logs are stored. ./tensorboard by default')
parser.add_argument('--tensorboard-comment', '-tbcomment',
                    dest='tensorboard_comment', type=str, required=False, default=None)
parser.add_argument('--eval-periodicity', '-evalperiod',
                    dest='eval_periodicity', type=int, required=False, default=20)
parser.add_argument('--load-checkpoint', '-load',
                    dest='load_checkpoint', required=False, action='store_true')
parser.add_argument('--wall-time', '-wt',
                    dest='walltime', required=False, type=int, default=3600, 
                    help='Walltime for training')

args = parser.parse_args()
#############################################################################
############################### LOAD DATASETS ############################### 
print('Loading datasets...')
DATA_SST_TRAIN = pd.read_csv('./assets/datasets/SST2/train.tsv', sep='\t')
DATA_SST_TEST = pd.read_csv('./assets/datasets/SST2/test.tsv', sep='\t')
DATA_SST_DEV = pd.read_csv('./assets/datasets/SST2/dev.tsv', sep='\t')

columns = ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']
types_dict = {'id': int, 'qid1': int, 'qid2': int , 
              'question1': str, 'question2': str, 'is_duplicate': int}
DATA_QQP_TRAIN = pd.read_csv('./assets/datasets/QQP/train.tsv', sep='\t', dtype=types_dict)
DATA_QQP_TEST = pd.read_csv('./assets/datasets/QQP/test.tsv', sep='\t', dtype=types_dict)
DATA_QQP_DEV = pd.read_csv('./assets/datasets/QQP/dev.tsv', sep='\t', dtype=types_dict)


###############################################################################
############################### LOAD MODELS ###################################
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Device being used: {device}")


transformer_net = Transformer(model_class=BertModel,
                              tokenizer_class=BertTokenizer,
                              pre_trained_weights='bert-base-uncased',
                              output_layer=-2,
                              device=device)

bracketing_net = NNSimilarityChunker(sim_function=cos,
                                     threshold=args.sim_threshold,
                                     exclude_special_tokens=False,
                                     combinatorics='sequential',
                                     chunk_size_limit=4,
                                     device=device)

generator_net = EmbeddingGenerator(pool_function=abs_max_pooling, 
                                   device=device)

seq_classifier = AttentionClassifier(embedding_dim=768,
                                     sentset_size=2,
                                     dropout=0.3,
                                     n_sentiments=4,
                                     pool_mode='concat',
                                     device=device).to(device)

seq_pair_classifier = SeqPairFancyClassifier(embedding_dim=768,
                                             num_classes=2,
                                             dropout=0.3,
                                             n_attention_vecs=2,
                                             device=device)#.to(device)

multitask_net = MultiTaskNet(seq_classifier,
                             seq_pair_classifier,
                             device=device).to(device)

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
eval_periodicity = args.eval_periodicity
wall_time = 1000 # an hour training as a wall-time

# Tensorboard init
writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, run_identifier), comment=args.tensorboard_comment)
checkpoints_path = os.path.join('./assets/checkpoints/', run_identifier)
if args.load_checkpoint:
    model.load_state_dict(torch.load(checkpoints_path))

# Dicts
counter = {'SST2': 0, 'QQP': 0}
batch_size = {'SST2': 16, 'QQP': 16}
n_batches = {'SST2': math.floor(len(DATA_SST_TRAIN)/16), 'QQP': math.floor(len(DATA_QQP_TRAIN)/16)}
get_batch_function = {'SST2': get_batch_SST2_from_indices, 'QQP': get_batch_QQP_from_indices}
dataframe = {'SST2': DATA_SST_TRAIN, 'QQP': DATA_QQP_TRAIN}
dev_dataframes_dict = {'SST2': DATA_SST_DEV, 'QQP': DATA_QQP_DEV}
datasets = ['SST2','QQP'] # here the datasets in training
batch_indices = {}


global_counter, losseval, max_acc = 0, 0, 0



##########################################################################
########################### ACUTAL TRAINING ##############################
initial_time = time.time()
optimizer = torch.optim.Adam(multitask_net.parameters(), 
                             lr=args.lr,
                             betas=(0.9, 0.999),
                             eps=1e-08, 
                             weight_decay=0.0001, 
                             amsgrad=False)

finished_training = False
t = time.time()
while not finished_training:
    for dataset in datasets:
        if counter[dataset] >= n_batches[dataset] or global_counter == 0:
            print(f"NEW EPOCH STARTED FOR DATASET {dataset}!")
            counter[dataset] = 0
            batch_indices[dataset] = None
            torch.cuda.empty_cache()
            # Re-shuffle the training batches data
            batch_indices[dataset] = torch.randperm(n_batches[dataset]*batch_size[dataset],
                                                    device=torch.device('cpu')).reshape(-1, batch_size[dataset])
    
    batch_sequences, batch_targets, batch_splits = [], [], [0]
    for dataset in datasets:
        idx = counter[dataset]
        dataset_batch = get_batch_function[dataset](dataframe[dataset], 
                                                    batch_indices[dataset][idx, :])
        # List of tensors, one for each task
        try:
            batch_targets.append(torch.tensor([data[1] for data in dataset_batch], 
                                            dtype=torch.int64, 
                                            device=device))
        except:
            L = [data[1] for data in dataset_batch]
            
            raise ValueError(f'This thing failed when the target tensor was in dataset \
                              {dataset}: {L}, indices: {batch_indices[dataset][idx, :]}')
        
        # Big list combining the input sequences/ tuple of sequences because the batch needs
        # to be at the same "depth" level
        batch_sequences.extend([data[0] for data in dataset_batch])
        batch_splits.append(batch_splits[-1] + len(dataset_batch))
        counter[dataset] += 1

    model.train()
    batch_predictions = model.forward(batch_sequences, batch_splits=batch_splits)
    L = model.loss(batch_predictions, batch_targets, weights=None)
    metrics = model.metrics(batch_predictions, batch_targets)
    # Update net
    optimizer.zero_grad()
    L.backward()
    optimizer.step()
    losseval += L.item()

    if global_counter % eval_periodicity == 0:
        print(f'################### GLOBAL COUNTER {global_counter} ###################')
        print(f'Iterations per second: {eval_periodicity/(time.time()-t)}')
        ###################### evaluate on dev sets
        model.eval()
        metrics_dict = eval_model_on_DF(model, 
                                        dev_dataframes_dict, 
                                        get_batch_function, 
                                        batch_size=16, 
                                        global_counter=global_counter, 
                                        device=device)
        avg_acc = sum(metrics_dict.values())/len(metrics_dict)
        if avg_acc > max_acc:
            torch.save(model.state_dict(), checkpoints_path)
            max_acc = avg_acc
        print('eval metrics are:', metrics_dict)
        ######################
        t = time.time()
        print(f'Global Loss: {losseval/eval_periodicity}')
        losseval = 0
        
    # Log to tensorboard
    writer.add_scalar(f'{run_identifier}/loss/train', L.item(), global_counter)
    writer.add_scalars(f'{run_identifier}/metrics/train', {datasets[i]: metrics[i] for i in range(len(datasets))}, global_counter)
    writer.add_scalars(f'{run_identifier}/metrics/dev', metrics_dict, global_counter)
    global_counter += 1

    finished_training = True if (time.time() - initial_time) > args.walltime else False
    
