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
    compression=None,
    return_comp_rate=False,
    max_length=256,
    device=torch.device("cpu"),
):
    """Evaluates a model on a dataframe.
    Args:
        model: full model to evaluate (nn.Module)
        dataframes_dic: dictionary of dataframes, keys are the dataset names
        get_batch_function_dict: dictionary of functions to load batches
        batch_size: size of the batch to run samples. Higher batch sizes
            improve speed, but a too high size could cause a memory error.
        compression: compress sequence in forward pass.
        return_comp_rate: flag to return a dictionary of average compression
            rates for each dataset.
        max_length: maximum length to truncate sequences (it can prevent
            memory errors if a sequence in a dataset is very long).
        device: torch.device in use, cpu or cuda.
    Returns:
        metrics_dict: dictionary of metrics with dataset names as keys.
        compression_dict: if return_comp_rate=True, it returns compression
            rates as a dictionary with dataset names as keys.
    """

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
    """Writes a dictionary of results in a Googls Spreadsheet. This function is
    mostly for a one-time use, so it's not very reusable.
    """
    # Google API stuff
    scope = ["https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name('config/gcp-credentials.json', scope)
    client = gspread.authorize(creds)
    sheet = client.open(name).worksheet(sheet_name)
    sheet.update(f'B{row}:D{row}', [list(results_dict.values())])

def make_connectivity_matrix(length, span=1):
    """Returns a (length x length) connectixity matrix of 1s in a 'thick'
    diagonal and 0s on the rest, which can be fed into the agglomerative
    clustering function, such that only datapoints that are connected can be
    merged into the same cluster.
    Sleek line of code originally written by Boris Reuderink.
    Args:
        length: size of the matrix
        span: number of elements above and below the diagonal that will be 1s.
            if span=1, only one row above and below the diagonal will be 1s,
            with span >= 1. If span=2 two layers above and below the diagonal
            will be 1s and so on. The 'thickness' of 1s in the diagonal is
            2 * span + 1.
    Returns:
        connectivity_matrix: np.array of length x length with 1s in a 'thick'
            diagonal and 0s in the rest.

    """
    return np.fromfunction(lambda i, j: abs(i - j) <= span, (length, length))


def compression2euclideandist(compression):
    raise NotImplementedError()


def add_space_to_special_characters(string, characters=[]):
    """Little function that adds a space to the left and right of a list of
    'special characters' provided. This is only used for the deprecated
    spacey_based_bracketing function.
    Args:
        string: original string to apply the transformation to.
        characters: list of characters you want to add spaces to.
    Returns:
        string: modified string with spaces around the characters.
    """
    for char in characters:
        string = string.replace(char, f" {char} ")
    return string


def filter_indices(indices_batch):
    """Removes the lists of lengths 1 in a list of lists. Not used anymore.
    Args:
        indices_batch: list of lists
    Returns
        indices_batch: list of lists where inner lists of length=1 have been
            removed.
    """
    for b, indices in enumerate(indices_batch):
        for i, idx in enumerate(indices):
            if len(idx) == 1:
                indices_batch[b].pop(i)
    return indices_batch


def expand_indices(indices_batch, target_lengths):
    """Adds a leading indices in tuples (i.e. (0,)) to the list of indices if
    it's missing and the same for the last indices. Only used for the unused
    spacy_based_bracketing.
    Args:
        indices_batch: list of lists of indices
        target_lengths: desired length of inner lists
    Returns:
        indices_batch: modified indices_batch where inner lists are all the same
            target lengths.
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
    """Prints a message and the time it has passed since time t next to it"""
    print(message, time.time() - t)


def txt2list(txt_path=None):
    """Load txt file and split the sentences into a list of strings, splitted
    by a regex expression.
    Args:
        txt_path: path to the .txt file to convert to list of strings.
    Returns:
        sentences: list of sentences from the .txt file.
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
    """Performs the absolute max pooling operation along a dimension, similarly
    to how torch.mean() or torch.max() would.
    Args:
        T: tensor to apply the operation to.
        dim: dimension along the operation is performed.
        keepdim: Flag which if `True`, keeps the dimensions of the original tensor
            by returning an 'unsqueezed' tensor at dimension `dim`.
    Returns:
        torch.tensor: tensor formed by the maximum absolute values along
            dimension `dim`.
    """
    # Reduce with absolute max pooling over the specified dimension
    abs_max, _ = torch.max(T.abs(), dim=dim, keepdim=True)
    bool_mask = T.abs() >= abs_max
    return (T * bool_mask).sum(dim=dim, keepdim=keepdim)


def mean_pooling(T, dim=-1, keepdim=False, **kwargs):
    """Performs the mean pooling operation, exactly as torch.mean() does. This
    function exists for consistency between different pooling operations.
    Args:
        T: tensor to apply the operation to.
        dim: dimension along the operation is performed.
        keepdim: Flag which if `True`, keeps the dimensions of the original tensor
            by returning an 'unsqueezed' tensor at dimension `dim`.
    Returns:
        torch.tensor: tensor formed by the mean values along dimension `dim`.
    """
    return T.mean(dim=dim, keepdim=keepdim)

def freq_pooling(T, dim=-1, keepdim=False, token_ids=None, **kawrgs):
    """Performs a pooling operation based on frequencies of elements along a
    dimension following a weighting according to `a / (p(i) + a)`, where p(i) is
    the probability of element `i` (or its inverse frequency) and `a` is a constant.
     It is inspired by Arora et. al 2017
    https://openreview.net/pdf?id=SyK00v5xx It uses the Zipf law to estimate
    element frequencies based on their rank.
    Args:
        T: tensor to apply the operation to.
        dim: dimension along the operation is performed.
        keepdim: Flag which if `True`, keeps the dimensions of the original tensor
            by returning an 'unsqueezed' tensor at dimension `dim`.
    Returns:
        torch.tensor: pooled tensor along dimension `dim` with a noramlized
            weighted sum.
    """
    a = 1e-4
    log_p = log_zipf_law(token_ids.unsqueeze(-1))
    weights = a / (torch.exp(log_p) + a)  # Strategy from Arora et al 2018.
    weights /= weights.norm(p=1, dim=dim)  # normalize so they add up to 1
    return (T * weights).sum(dim=dim, keepdim=keepdim)

def rnd_pooling(T, dim=-1, keepdim=False, **kwargs):
    """Performs a pooling operation along dimension `dim` by selecting a random
    index from a uniform distribution along the pooling dimension.
    Args:
        T: tensor to apply the operation to.
        dim: dimension along the operation is performed.
        keepdim: Flag which if `True`, keeps the dimensions of the original tensor
            by returning an 'unsqueezed' tensor at dimension `dim`.
    Returns:
        torch.tensor: randomly pooled tensor along dimension `dim`.
    """
    idx = torch.randint(T.size(dim), (1,), device=T.device)
    if keepdim:
        return T.index_select(dim, idx)
    else:
        return T.index_select(dim, idx).squeeze(dim)

def log_zipf_law(inp, alpha=1., log_ct=-1.0, rank_first=1996):
    """Calculates the log Zipf law of a tensor of ranks, returning the estimated
    log probability for each one of them. The Zipf law is an empirical
    'heavy tail' distribution that determines the frequencies of items given the
    ranking of their frequencies. Frequencies of words in languages can
    generally be approximated by this law: https://www.aclweb.org/anthology/W98-1218.pdf
    Args:
        inp: torch.tensor of any size with ranks (i.e. natural numbers)
        apha: parameter for the Zipf law, it's normally close to 1 in English.
        log_ct: parameter for the Zipf law determining the log probability of
            the most frequent word. We set it to -1.
        rank_first: rank of the most probable element (i.e. when you should
        start counting, if the rank numbers don't start at 1). It is set to
        1996 because the Bert Tokenizer starts with the english vocabulary at
        that id.
    Returns:
        torch.tensor: torch.tensor of the same size as the input with log
            probabities for each element.
    """
    # Log_ct is the term reflecting the probability of the element of rank = 1
    # log p("the") = log 0.02 = -2.7, for reference
    ranks = (inp - rank_first) * (inp > rank_first) + 1.
    log_rank = torch.log(ranks.float())
    return log_ct - alpha * log_rank


def hotfix_pack_padded_sequence(inp,
                                lengths,
                                batch_first=False,
                                enforce_sorted=True):
    """Quick fix for the pytorch pack_padded_sequence function, which had a bug
    related to failing to use cuda properly. The behavior is supposed to emulate
    that of torch.nn.utils.rnn.pack_padded_sequence(); find the documentation in
    https://pytorch.org/docs/master/generated/torch.nn.utils.rnn.pack_padded_sequence.html

    Code authored by @ptrblck and shared in the open forum https://discuss.pytorch.org
    """
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
    """Converts a string into a boolean variable. Used to pass boolean arguments
    in the command line arguments instead of using flags. Inspired from a
    StackOverflow respone from
    Args:
        v: string
    Returns:
        bool: boolean value from parsing the string.
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
