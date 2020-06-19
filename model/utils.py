"""
Collection of useful little functions.
"""
import argparse
from collections import defaultdict
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import re
import time
from tqdm import tqdm

import math
import numpy as np
import scipy
import torch
from torch.nn.utils.rnn import PackedSequence


def eval_model_on_DF(
    model,
    dataframes_dict,
    get_batch_function_dict,
    batch_size=16,
    global_counter=0,
    compression=None,
    return_comp_rate=False,
    max_length=256,
    device=torch.device("cpu"),
):

    assert compression is not None
    metrics_dict, compression_dict = {}, {}
    with torch.no_grad():
        for dataset, df in dataframes_dict.items():
            # Divide the dataset in batches of batch_size, adding the last
            # batch if necessary
            n_batches = math.floor(len(df) / batch_size)
            batches = [batch_size] * n_batches + [len(df) % batch_size]
            if 0 in batches: batches.remove(0)  # remove last 0 if it exists
            # make sure all slices are empty for datasets that are not the current one
            batch_slices = defaultdict(lambda: slice(0, 0))
            dev_acc, cummulative_comp, idx_left = 0, 0, 0
            for step_size in tqdm(batches, desc=f'Progress on {dataset}'):
                batch_targets, batch_sequences = {}, []
                #indices = list(range(i * batch_size, (i + 1) * batch_size))
                indices = list(range(idx_left, idx_left + step_size))
                batch_slices[dataset] = slice(0, len(indices))
                dataset_batch = get_batch_function_dict[dataset](df, indices)
                # construct targets
                batch_targets[dataset] = torch.tensor([data[1] for data in dataset_batch],
                                                      dtype=torch.int64, device=device)
                # construct sequences
                batch_sequences.extend([data[0] for data in dataset_batch])
                batch_predictions, compression_rate = model.forward(
                    batch_sequences,
                    batch_slices=batch_slices,
                    compression=compression,
                    return_comp_rate=return_comp_rate,
                    max_length=max_length,
                )
                L = model.loss(batch_predictions, batch_targets, weights=None)
                m = model.metrics(batch_predictions, batch_targets)
                dev_acc += m[dataset]
                cummulative_comp += compression_rate
                idx_left += step_size
            # Average over all the batches (accounting for the fact that the last
            # batch is possibly of a different size)
            acc = dev_acc * batch_size / sum(batches)
            comp = cummulative_comp * batch_size / sum(batches)
            metrics_dict[dataset] = acc
            compression_dict[dataset] = comp

    if return_comp_rate:
        return metrics_dict, compression_dict
    else:
        return metrics_dict

def write_google_sheet(results_dict, row=2, name='results_layers', sheet_name='run1'):
    # Google API stuff
    scope = ["https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name('config/gcp-credentials.json', scope)
    client = gspread.authorize(creds)
    sheet = client.open(name).worksheet(sheet_name)
    sheet.update(f'B{row}:D{row}', [list(results_dict.values())])


def make_connectivity_matrix(length, span=1):
    span = min(length - 1, span)
    col, row = [], []
    for d in range(span):
        y = list(range(0, length - (d + 1)))
        x = list(range(d + 1, length))
        col.extend(x)
        col.extend(y)
        row.extend(y)
        row.extend(x)
    # Precomput the number of 1s in the sparse mtrx
    N_ones = 2 * (span * length - sum(range(span + 1)))
    data = np.ones(N_ones, dtype=int)
    connectivity_matrix = scipy.sparse.coo_matrix(
        (data, (row, col)), shape=(length, length)
    ).toarray()
    return connectivity_matrix


def compression2euclideandist(compression):
    distance_threshold = None
    return distance_threshold


def add_space_to_special_characters(string, characters=[]):
    for char in characters:
        string = string.replace(char, f" {char} ")
    return string


def filter_indices(indices_batch):
    """
    Removes the lists of lengths 1 in a list of lists.
    """
    for b, indices in enumerate(indices_batch):
        for i, idx in enumerate(indices):
            if len(idx) == 1:
                indices_batch[b].pop(i)

    return indices_batch


def expand_indices(indices_batch, target_lengths):
    """
    Adds a leading indices in tuples (i.e. (0,)) to the list of indices if it's
    missing and the same for the last indices
    """

    assert len(indices_batch) == len(target_lengths)

    for b, target_length in enumerate(target_lengths):
        if target_length > len(indices_batch[b]):

            while indices_batch[b][0][0] > 0:
                first_idx = indices_batch[b][0][0]
                indices_batch[b].insert(0, (first_idx - 1,))

            while indices_batch[b][-1][-1] < target_length - 1:
                last_idx = indices_batch[b][-1][-1]
                indices_batch[b].append((last_idx + 1,))

    return indices_batch


def time_since(t, message):
    print(message, time.time() - t)


def txt2list(txt_path=None):
    """
    Load txt file and split the sentences into a list of strings
    """
    if txt_path is None:
        raise ValueError(
            "txt_path must be specified as a named argument! \
        E.g. txt_path='../dataset/yourfile.txt'"
        )

    # Read input sequences from .txt file and put them in a list
    with open(txt_path) as f:
        text = f.read()
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)
    try:
        sentences.remove("")  # remove possible empty strings
    except Exception as error:
        raise Exception('something went wrong dude.')

    return sentences


def abs_max_pooling(T, dim=-1, keepdim=False, **kwargs):
    # Reduce with absolute max pooling over the specified dimension
    abs_max, _ = torch.max(T.abs(), dim=dim, keepdim=True)
    bool_mask = T.abs() >= abs_max
    return (T * bool_mask).sum(dim=dim, keepdim=keepdim)


def mean_pooling(T, dim=-1, keepdim=False, **kwargs):
    return T.mean(dim=dim, keepdim=keepdim)

def freq_pooling(T, dim=-1, keepdim=False, token_ids=None, **kawrgs):
    a = 1e-4
    log_p = log_zipf_law(token_ids.unsqueeze(-1))
    weights = a / (torch.exp(log_p) + a)  # Strategy from Arora et al 2018.
    weights /= weights.norm(p=1, dim=dim)  # normalize so they add up to 1
    return (T * weights).sum(dim=dim, keepdim=keepdim)

def rnd_pooling(T, dim=-1, keepdim=False, **kwargs):
    idx = torch.randint(T.size(dim), (1,), device=T.device)
    if keepdim:
        return T.index_select(dim, idx)
    else:
        return T.index_select(dim, idx).squeeze(dim)

def log_zipf_law(inp, alpha=1., log_ct=-1.0, rank_first=1996):
    # Log_ct is the term reflecting the probability of the element of rank = 1
    # log p("the") = log 0.02 = -2.7, for reference
    ranks = (inp - rank_first) * (inp > rank_first) + 1.
    log_rank = torch.log(ranks.float())
    return log_ct - alpha * log_rank


def hotfix_pack_padded_sequence(inp,
                                lengths,
                                batch_first=False,
                                enforce_sorted=True):
    lengths = torch.as_tensor(lengths, dtype=torch.int64)
    lengths = lengths.cpu()
    if enforce_sorted:
        sorted_indices = None
    else:
        lengths, sorted_indices = torch.sort(lengths, descending=True)
        sorted_indices = sorted_indices.to(inp.device)
        batch_dim = 0 if batch_first else 1
        inp = inp.index_select(batch_dim, sorted_indices)

    data, batch_sizes = torch._C._VariableFunctions._pack_padded_sequence(
        inp, lengths, batch_first
    )
    return PackedSequence(data, batch_sizes, sorted_indices)


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


def str2list(string):
    clean = string.strip('[') \
                  .strip(' ') \
                  .strip(']') \
                  .strip("'") \
                  .strip('"')
    if clean == '':  # 'weird' behaviour of .split with empty strings
        return []
    else:
        return clean.split(',')
