"""
Collection of useful little functions.
"""
import re
import time
import torch
from torch.nn.utils.rnn import PackedSequence


# THIS FOR IMDB DATASET
def sentiment2tensorIMDB(sent_list):
    logits = []
    for sent in sent_list:
        if sent == 'positive':
            logits.append(1)
        elif sent == 'negative':
            logits.append(0)
        else:
            raise ValueError("A sentiment wasn't positive or negative!")
    return torch.tensor(logits)


# For SST2 the sentiment column is already numbers so no 
# need to do fancy stuff, just to tensor
def sentiment2tensorSST(sent_list):
    return torch.tensor(sent_list)


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
