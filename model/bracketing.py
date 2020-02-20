import itertools
from typing import List
from collections import Counter
import re
from sklearn import cluster
import torch
import torch.nn as nn
from model.utils import make_connectivity_matrix

cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-5)  # default similarity func.


class NNSimilarityChunker(nn.Module):
    """
    This class implements a model that performs "chunking" based on a similarity threshold,
    as defined by a similarity function (cosine similarity by default).
    """
    def __init__(self,
                 sim_function=cos,
                 threshold=0.9,
                 exclude_special_tokens=False,
                 combinatorics='sequential',
                 device=torch.device('cpu'),
                 chunk_size_limit=8):
        super().__init__()

        self.device = device
        self.sim_function = sim_function
        self.exclude_special_tokens = exclude_special_tokens
        self.threshold = threshold
        assert combinatorics in ['sequential', 'all']
        self.combinatorics = combinatorics
        self.limit = chunk_size_limit

    def forward(self, batch_sequence_tensors: torch.Tensor, masks_dict=None) -> List[List[List]]:
        assert masks_dict is not None
        indices_to_compact = self.indices_to_compact_by_similarity_threshold(batch_sequence_tensors,
                                                                             masks_dict=masks_dict)
        return indices_to_compact

    def indices_to_compact_by_similarity_threshold(self, batch_sequence_tensors, masks_dict=None) -> List[List]:
        assert masks_dict is not None
        batch_size, seq_length, _ = batch_sequence_tensors.size()  # make sure the input is proper size!!
        indices = list(range(seq_length))
        regular_tokens_mask = masks_dict['regular_tokens_mask']

        # Combinations of indices that are group candidates: only sequential (indices[s:e])
        if self.combinatorics == 'sequential':
            if self.exclude_special_tokens: #don't use this, im not 100% sure it's correct
                idx_combinations = [indices[s:e+1] for s, e in itertools.combinations(range(1, len(indices)), 2)]
            else:
                idx_combinations = [indices[s:e] for s, e in itertools.combinations(range(len(indices)+1), 2)]

        # All permutations (permutations explode quickly)
        elif self.combinatorics == 'all':
            idx_combinations = []
            for L in range(2, seq_length+1):
                combinations = list(itertools.combinations(indices, r=L))
                idx_combinations.extend(combinations)

        # Remove too large groups of indices
        i = 0
        for indices in idx_combinations.copy():
            if len(indices) > self.limit:
                idx_combinations.pop(i)
                i -= 1
            i += 1

        # Initialize empty list of lists of length batch_size
        batch_all_indices_to_compact = [[] for _ in range(batch_size)]

        for indices in idx_combinations:
            batch_group_candidates = batch_sequence_tensors[:, indices, :]
            batch_centers = torch.mean(batch_group_candidates, dim=1).unsqueeze(1)
            # Calculate all embeddings similarities w.r.t. the center of the group
            similarities = self.sim_function(batch_centers, batch_group_candidates)
            worst_sim, _ = torch.min(similarities, dim=-1)
            # Construct torch.bool tensor of size (batch,) as a 'mask'
            batch_include_group_mask = worst_sim >= self.threshold

            if len(indices) > 1:
                # Are all tokens in the group regular tokens?
                batch_all_regular_tokens = regular_tokens_mask[:, indices].prod(dim=1) == 1
            else:
                # If there is only 1 index we want to include it
                batch_all_regular_tokens = torch.ones_like(batch_include_group_mask,
                                                           dtype=torch.bool,
                                                           device=self.device)

            # generate logical tensor with only True if both threshold and regular
            # tokens criteria
            batch_include = batch_include_group_mask * batch_all_regular_tokens
            # Iterate over mask and fill the List of indices to compact
            for b, include_group in enumerate(batch_include):
                if include_group: batch_all_indices_to_compact[b].append(indices)

        batch_indices_to_compact = batch_remove_subsets(batch_all_indices_to_compact)

        return batch_indices_to_compact


class AgglomerativeClusteringChunker(nn.Module):
    """
    Wrapper that implements sklearn.cluster.MeanShift for a batch of tensors
    """
    def __init__(self,
                 threshold=0.9,
                 device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.dist_threshold = 1-threshold
        self.id = nn.Identity()

    def forward(self, input, **kwargs):
        _, length, _ = input.size()
        connectivity_matrix = make_connectivity_matrix(length)
        cl = cluster.AgglomerativeClustering(n_clusters=None,
                                             affinity='cosine',
                                             memory=None,
                                             connectivity=connectivity_matrix,
                                             compute_full_tree=True,
                                             linkage='average',
                                             distance_threshold=self.dist_threshold)
        indices_to_compact = []
        for embeddings in input.detach().numpy(): # loop over each element in batch
            N = cl.fit_predict(embeddings)
            C = Counter(N)
            ordered_idx, i = [], 0
            for k, v in C.items():
                ordered_idx.append(tuple(range(i, i+v)))
                i += v
            indices_to_compact.append(ordered_idx)

        return indices_to_compact




class IdentityChunker(nn.Module):
    """
    IdentityChunker docstring
    """
    def __init__(self, *args, **kargs):
        super(IdentityChunker, self).__init__()

    def forward(self, x, *args, masks_dict=None, **kwargs):
        assert masks_dict is not None
        return [[None]]*x.size(0)  #return empty list of lists of batch size


def batch_remove_subsets(batch_L):
    batch_filtered = []
    for L in batch_L:
        filtered = filter(lambda f: not any(set(f) < set(g) for g in L), L)
        batch_filtered.append(list(filtered))
    return batch_filtered




############################## DEPRECATED ####################################
# This class was used in other experiments but it won't work here
"""class Chunker():
    def __init__(self,
                 layer=-1,
                 sim_function=cos,
                 threshold=0.9,
                 exclude_special_tokens=True,
                 combinatorics='sequential'):

        self.layer = layer
        self.sim_function = sim_function
        self.threshold = threshold
        self.exclude_special_tokens = exclude_special_tokens

        if combinatorics not in ['sequential', 'all']:
            raise ValueError("You must specify the combinatorics as 'sequencial' or 'all'!!")
        self.combinatorics = combinatorics


    def compact(self, sentence, threshold=None, layer=None):
        if threshold is not None: self.threshold = threshold
        if layer is not None: self.layer = layer

        assert type(sentence) == TransformerSentence #, "Input must be a TransformerSentence Object!"
        if not sentence.summary:
            sentence.write_summary()

        indices_to_compact = self.indices_to_compact_by_similarity_threshold(sentence)
        new_embeddings = self.compact_embeddings_avg(sentence, indices_to_compact)
        new_tokens = self.new_tokens(sentence, indices_to_compact)

        return new_embeddings, new_tokens


    def indices_to_compact_by_similarity_threshold(self, sentence) -> List[List]:

        sequence_embeddings = sentence.summary['states'][self.layer, :, :]

        # combinatorics= 'sequential', 'all'
        seq_length, embedding_size = sequence_embeddings.size() #make sure the input is proper size!!
        indices = list(range(seq_length))

        # Combinations of indices that are group candidates
        if self.combinatorics == 'sequential':
            if self.exclude_special_tokens:
                idx_combinations = [indices[s:e] for s, e in itertools.combinations(range(1, len(indices)), 2)]
            else:
                idx_combinations = [indices[s:e] for s, e in itertools.combinations(range(len(indices)+1), 2)]

        elif self.combinatorics == 'all':
            idx_combinations = []
            for L in range(2, seq_length+1):
                combinations = list(itertools.combinations(indices, r=L))
                idx_combinations.extend(combinations)


        all_indices_to_compact = []
        for indices in idx_combinations:
            group_candidate = sequence_embeddings[indices, :]
            group_size = len(indices)
            center = torch.mean(group_candidate, dim=0)
            center = center.repeat(group_size, 1)
            # calculate all embeddings similarities w.r.t. the center of the group
            similarities = self.sim_function(center, group_candidate)
            worst_sim, _ = torch.min(similarities, dim=0)
            if worst_sim > self.threshold: all_indices_to_compact.append(indices)

        indices_to_compact = Chunker.remove_subsets(all_indices_to_compact)

        return indices_to_compact


    def compact_embeddings_avg(self, sentence, indices_to_compact) -> torch.Tensor:
        original_embeddings = sentence.summary['states'][self.layer, :, :]
        new_embeddings_list = []
        for indices in indices_to_compact:
            group = original_embeddings[indices, :]
            center = torch.mean(group, dim=0)
            new_embeddings_list.append(center)

        try:
            new_embeddings = torch.stack(new_embeddings_list, dim=0)
        except:
            print("No chunks were found!")
            new_embeddings = None

        return new_embeddings


    def new_tokens(self, sentence, indices_to_compact) -> List[str]:
        sentence_tokens = sentence.summary['input_tokens']
        new_tokens = []
        for chunk_indices in indices_to_compact:
            if len(chunk_indices) > 1:
                tokens = [sentence_tokens[i] for i in chunk_indices]
                # remove everything after _ in tokens to match the words
                tokens = [re.sub(r'_(.*)', '', token) for token in tokens]
                joint_token = '_'.join(tokens).replace('_##', '')
                new_tokens.append(joint_token)
            else:

                token = sentence_tokens[chunk_indices[0]]
                new_tokens.append(token)

        return new_tokens


    @staticmethod
    def remove_subsets(L):
        filtered = filter(lambda f: not any(set(f) < set(g) for g in L), L)
        return list(filtered)
    """



