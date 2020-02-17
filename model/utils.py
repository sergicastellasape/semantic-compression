"""
Collection of useful little functions.
"""
import re
import time
import math
import torch
from torch.nn.utils.rnn import PackedSequence


def eval_model_on_DF(model, dataframes_dict, get_batch_function_dict, batch_size=16, global_counter=0, compression=None, device=torch.device('cpu')):    
    assert compression is not None
    k=0
    metrics_dict = {}
    for dataset, df in dataframes_dict.items():
        n_batches = math.floor(len(df)/batch_size)
        batch_splits = [-1]*(len(dataframes_dict)+1)
        batch_splits[k] = 0 # [-1, -1, 0, -1, -1]
        batch_splits[k+1] = len(df)
        dev_acc = 0
        for i in range(n_batches):
            batch_targets, batch_sequences = [], []
            indices = list(range(i*batch_size, (i+1)*batch_size))
            dataset_batch = get_batch_function_dict[dataset](df, indices)
            # construct targets
            batch_targets.append(torch.tensor([data[1] for data in dataset_batch], 
                                              dtype=torch.int64, 
                                              device=device))
            # construct sequences
            batch_sequences.extend([data[0] for data in dataset_batch])
            batch_predictions = model.forward(batch_sequences, batch_splits=batch_splits, compression=compression)
            L = model.loss(batch_predictions, batch_targets, weights=None)
            m = model.metrics(batch_predictions, batch_targets)
            dev_acc += m[0]
        k += 1
        acc = dev_acc/n_batches
        metrics_dict[dataset] = acc
        
    return metrics_dict


def add_space_to_special_characters(string, characters=[]):
    for char in characters:
        string = string.replace(char, f' {char} ')
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
    Adds a leading indices in tuples (i.e. (0,)) to the list of indices if it's missing
    and the same for the last indices
    """

    assert len(indices_batch) == len(target_lengths)

    for b, target_length in enumerate(target_lengths):
        if target_length > len(indices_batch[b]):

            while indices_batch[b][0][0] > 0:
                first_idx = indices_batch[b][0][0]
                indices_batch[b].insert(0, (first_idx-1,))

            while indices_batch[b][-1][-1] < target_length - 1:
                last_idx = indices_batch[b][-1][-1]
                indices_batch[b].append((last_idx+1,))
    
    return indices_batch


# Print time since t
def time_since(t, message):
    print(message, time.time() - t)



def txt2list(txt_path=None):
    """
    Load txt file and split the sentences into a list of strings
    """
    if txt_path is None:
        raise ValueError("txt_path must be specified as a named argument! \
        E.g. txt_path='../dataset/yourfile.txt'")

    # Read input sequences from .txt file and put them in a list
    with open(txt_path) as f:
        text = f.read()
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    try:
        sentences.remove('') # remove possible empty strings
    except:
        None
    
    return sentences


def abs_max_pooling(T, dim=1):
    # input is (batch, seq_length, emb_dimension)
    _, abs_max_i = torch.max(T.abs(), dim=dim) #max over abs in sequence dimension
    # convert indices into one_hot vectors
    one_hot = torch.nn.functional.one_hot(abs_max_i, num_classes=T.size()[dim]).transpose(dim, -1).type(torch.float)
    # multily original with one hot to apply mask and then sum over the dimension
    max_abs_tensor = torch.mul(T, one_hot).sum(dim=dim)
    return max_abs_tensor

#torch.set_default_tensor_type(torch.cuda.FloatTensor)

def hotfix_pack_padded_sequence(inp, lengths, batch_first=False, enforce_sorted=True):
    lengths = torch.as_tensor(lengths, dtype=torch.int64)
    lengths = lengths.cpu()
    if enforce_sorted:
        sorted_indices = None
    else:
        lengths, sorted_indices = torch.sort(lengths, descending=True)
        sorted_indices = sorted_indices.to(inp.device)
        batch_dim = 0 if batch_first else 1
        inp = inp.index_select(batch_dim, sorted_indices)

    data, batch_sizes = \
        torch._C._VariableFunctions._pack_padded_sequence(inp, lengths, batch_first)
    return PackedSequence(data, batch_sizes, sorted_indices)
