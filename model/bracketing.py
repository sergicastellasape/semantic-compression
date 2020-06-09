import itertools
from typing import List
from collections import Counter
import re
from sklearn import cluster
import numpy as np
import torch
import torch.nn as nn
from model.utils import make_connectivity_matrix, log_zipf_law

cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-5)  # default similarity func.


class NNSimilarityChunker(nn.Module):
    """
    This class implements a model that performs "chunking" based on a similarity threshold,
    as defined by a similarity function (cosine similarity by default).
    """

    def __init__(
        self,
        sim_function=cos,
        threshold=0.9,
        exclude_special_tokens=False,
        combinatorics="sequential",
        device=torch.device("cpu"),
        chunk_size_limit=8,
    ):
        super().__init__()

        self.device = device
        self.sim_function = sim_function
        self.exclude_special_tokens = exclude_special_tokens
        self.threshold = threshold
        assert combinatorics in ["sequential", "all"]
        self.combinatorics = combinatorics
        self.limit = chunk_size_limit

    def forward(
        self, batch_sequence_tensors: torch.Tensor, masks_dict=None
    ) -> List[List[List]]:
        assert masks_dict is not None
        indices_to_compact = self.indices_to_compact_by_similarity_threshold(
            batch_sequence_tensors, masks_dict=masks_dict
        )
        return indices_to_compact

    def indices_to_compact_by_similarity_threshold(
        self, batch_sequence_tensors, masks_dict=None
    ) -> List[List]:
        assert masks_dict is not None
        # make sure the input is proper size!!
        batch_size, seq_length, _ = batch_sequence_tensors.size()
        indices = list(range(seq_length))
        regular_tokens_mask = masks_dict["regular_tokens_mask"]

        # Combinations of indices that are group candidates: only sequential (indices[s:e])
        if self.combinatorics == "sequential":
            if (
                self.exclude_special_tokens
            ):  # don't use this, im not 100% sure it's correct
                idx_combinations = [
                    indices[s: e + 1]
                    for s, e in itertools.combinations(range(1, len(indices)), 2)
                ]
            else:
                idx_combinations = [
                    indices[s:e]
                    for s, e in itertools.combinations(range(len(indices) + 1), 2)
                ]

        # All permutations (permutations explode quickly)
        elif self.combinatorics == "all":
            idx_combinations = []
            for L in range(2, seq_length + 1):
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
                batch_all_regular_tokens = (
                    regular_tokens_mask[:, indices].prod(dim=1) == 1
                )
            else:
                # If there is only 1 index we want to include it
                batch_all_regular_tokens = torch.ones_like(
                    batch_include_group_mask, dtype=torch.bool, device=self.device
                )

            # generate logical tensor with only True if both threshold and regular
            # tokens criteria
            batch_include = batch_include_group_mask * batch_all_regular_tokens
            # Iterate over mask and fill the List of indices to compact
            for b, include_group in enumerate(batch_include):
                if include_group:
                    batch_all_indices_to_compact[b].append(indices)

        batch_indices_to_compact = batch_remove_subsets(batch_all_indices_to_compact)
        return batch_indices_to_compact


class AgglomerativeClusteringChunker(nn.Module):
    """
    Wrapper that implements sklearn.cluster.MeanShift for a batch of tensors.
    This turned into a nightmare after realizing that the agglomerative
    clustering from sklearn can't take a non-fully connected connectivity
    matrix. So the code is UGLY... but I needed to implement somehow the option to
    remove the special tokens and also prevent merging seq pairs...
    """
    def __init__(self, threshold=0.9, max_skip=4, device=torch.device("cpu")):
        super().__init__()
        self.device = device
        # agg clustering wants distance not similarity
        # self.dist_threshold = 1 - threshold
        self.dist_threshold = threshold
        self.span = max_skip
        self.id = nn.Identity()

    def forward(self, inp, masks_dict=None, mask_special_tokens=True, **kwargs):
        assert masks_dict is not None
        if not mask_special_tokens:
            keep_mask = (masks_dict['padding_mask'] == 1).detach().cpu().numpy()
        else:
            keep_mask = (masks_dict['regular_tokens_mask'] == 1).detach().cpu().numpy()
        # Mask with 1s on tokens that are not regular only if
        # mask_special_tokens=True, otherwise this will be 0s
        special_tokens = masks_dict['padding_mask'].detach().cpu().numpy() - keep_mask
        full_seq_pair_mask = masks_dict['seq_pair_mask'].detach().cpu().numpy()
        indices_to_compact = []
        # loop over each element in batch, I know. it's a shame this is sequential
        for b, embeddings in enumerate(inp.detach().cpu().numpy()):
            filtered_embedding = embeddings[keep_mask[b, :], :]
            # remove pad and special tokens if necessary
            seq_pair_mask = full_seq_pair_mask[b, keep_mask[b, :]]
            # keep track of the indices that need to be added later
            idxs_filtered_out = special_tokens[b, :].nonzero()[0]
            # iterate for the different values in seq_pair_mask, which will be
            # either [0] or [0, 1], so loop over each seq in the seq pair (if any)
            L, max_L = [], 0
            for i in list(Counter(seq_pair_mask).keys()):
                mask = seq_pair_mask == i
                length = sum(mask)
                connectivity_matrix = make_connectivity_matrix(length,
                                                               span=self.span)
                cl = cluster.AgglomerativeClustering(n_clusters=None,
                                                     affinity="euclidean",
                                                     memory=None,
                                                     connectivity=connectivity_matrix,
                                                     compute_full_tree=True,
                                                     linkage="ward",
                                                     distance_threshold=self.dist_threshold)
                # This outputs an unordered cluster labelling:
                # L_ = [4, 4, 1, 4, 1, 1, 0, 3, 2, 2, 3]
                if filtered_embedding[mask].shape[0] > 1:
                    L_ = cl.fit_predict(filtered_embedding[mask]).tolist()
                else:  # agg clustering raises an error if there's only 1 point
                    L_ = [0]
                L_ = [i + max_L for i in L_]  # make labels in range(max_L+1, max(L_)+max_L+1)
                # ugly extension for each sentence in the 'pair'.
                # good thing this is extendable to more than pairs
                L.extend(L_)
                max_L = max(L) + 1
            # Add isolated label for the special tokens in case
            # those were prohibited to be clustered
            # L = [5, 4, 4, 1, 4, 1, 1, 6, 0, 3, 2, 2, 3, 7]
            for pos in idxs_filtered_out:
                val = max(L) + 1  # new label for index
                L.insert(pos, val)
            # Here we order and group the indices of the
            # vectors in tuple such that:
            # L -> [(0, 1, 3), (2, 4, 5), (6), (7, 10), (8, 9)]
            ordered_idxs = [()] * (max(L) + 1)
            seen_clusters, cluster_dict, cluster_counter = [], {}, 0
            for idx, cluster_ in enumerate(L):
                if cluster_ not in seen_clusters:
                    cluster_dict[str(cluster_)] = cluster_counter
                    cluster_counter += 1
                seen_clusters.append(cluster_)
                pos = cluster_dict[str(cluster_)]
                ordered_idxs[pos] += (idx,)
            indices_to_compact.append(ordered_idxs)
        return indices_to_compact


class HardSpanChunker(nn.Module):
    def __init__(self, span=None, device=torch.device('cpu')):
        super().__init__()
        assert span is not None
        self.device = device
        self.id = nn.Identity()
        self.span = span

    def forward(self, inp, masks_dict=None, mask_special_tokens=True, **kwargs):
        assert masks_dict is not None
        batch_size = inp.size(0)
        if not mask_special_tokens:
            keep_mask = (masks_dict['padding_mask'] == 1).detach().cpu().numpy()
        else:
            keep_mask = (masks_dict['regular_tokens_mask'] == 1).detach().cpu().numpy()
        indices_to_compact = []
        for b in range(batch_size):
            # m = [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
            m = keep_mask[b, :]
            start_stop = [0]
            s = [start_stop.append(i) for i in range(1, len(m)) if m[i] != m[i - 1]]
            # start_stop = [0, 1, 4, 5, 12] ; where the switches happen
            if m[-1] == 1:  # [......1]
                start_stop.append(len(m))  # add last element if it's beefy
            else:           # [......0]
                # add 1 step for final <sep> token, useful for qualitative analysis
                start_stop.append(max(start_stop) + 1)

            ordered_idxs = []
            for i in range(len(start_stop) - 1):
                start = start_stop[i]
                stop = start_stop[i + 1]
                L_ = list(range(start, stop + 1, self.span))
                if max(L_) < stop:
                    L_.append(stop)
                # L_ = [0, 1]; [1, 4]; [4, 5]; etc. the 'ranges' to merge
                L = [tuple(range(L_[i], L_[i + 1])) for i in range(len(L_) - 1)]
                ordered_idxs.extend(L)
            # ordered_idxs = [(0,), (1, 2, 3), (4,), (5, 6, 7), (8, 9, 10), (11,)]
            indices_to_compact.append(ordered_idxs)
        return indices_to_compact


class FixedOutChunker(nn.Module):
    def __init__(self, out_num=None, device=torch.device('cpu')):
        super().__init__()
        assert out_num is not None
        self.device = device
        self.id = nn.Identity()
        self.out_num = out_num

    def forward(self, inp, masks_dict=None, mask_special_tokens=True, **kwargs):
        assert masks_dict is not None
        batch_size = inp.size(0)
        if not mask_special_tokens:
            keep_mask = (masks_dict['padding_mask'] == 1).detach().cpu().numpy()
        else:
            keep_mask = (masks_dict['regular_tokens_mask'] == 1).detach().cpu().numpy()
        indices_to_compact = []
        for b in range(batch_size):
            # m = [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
            m = keep_mask[b, :]
            start_stop = [0]
            s = [start_stop.append(i) for i in range(1, len(m)) if m[i] != m[i - 1]]
            # start_stop = [0, 1, 4, 5, 12] ; where the switches happen
            if m[-1] == 1:  # [......1]
                start_stop.append(len(m))  # add last element if it's beefy
            else:           # [......0]
                # add 1 step for final <sep> token, useful for qualitative analysis
                start_stop.append(max(start_stop) + 1)

            ordered_idxs = []
            for i in range(len(start_stop) - 1):
                start = start_stop[i]
                stop = start_stop[i + 1]
                # Same as for hard span chunker but specifying the number of vectors
                L_ = np.linspace(start, stop, num=self.out_num + 1, endpoint=True, dtype=int)
                L_ = list(np.unique(L_))
                if max(L_) < stop:
                    L_.append(stop)
                # L_ = [0, 1]; [1, 4]; [4, 5]; etc. the 'ranges' to merge
                L = [tuple(range(L_[i], L_[i + 1])) for i in range(len(L_) - 1)]
                ordered_idxs.extend(L)

            # ordered_idxs = [(0,), (1, 2, 3), (4,), (5, 6, 7), (8, 9, 10), (11,)]
            indices_to_compact.append(ordered_idxs)
        return indices_to_compact


class FreqChunker(nn.Module):
    def __init__(self,
                 alpha=1.0,
                 log_threshold=None,
                 device=torch.device('cpu'),
                 **kwargs):
        super().__init__()
        assert log_threshold is not None, "log_threshold is a required argument"
        assert log_threshold < 0, "Log threshold must be negative! i.e. -5, -20"
        self.device = device
        self.alpha = alpha
        self.log_threshold = log_threshold

    def forward(self,
                inp,
                masks_dict=None,
                mask_special_tokens=True,
                token_ids=None,
                **kwargs):

        assert masks_dict is not None, "masks_dict is a required argument"

        batch_size = inp.size(0)

        if not mask_special_tokens:
            keep_mask = (masks_dict['padding_mask'] == 1).detach()
        else:
            keep_mask = (masks_dict['regular_tokens_mask'] == 1).detach()

        # 'the' is the first wordpiece token at position 1996
        token_log_likelihoods = log_zipf_law(token_ids, rank_first=1996)

        indices_to_compact = []
        for b in range(batch_size):
            # m = [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
            m = keep_mask[b, :]
            sums = torch.tensor([token_log_likelihoods[b, :i].detach().sum(dim=-1) for i in range(0, len(m))])
            idx_left, idx_right, finished, idxs_b = 0, 0, False, []
            while idx_right < len(m):
                try:
                    if m[idx_left] == 0:
                        idx_right = idx_left + 1
                    elif m[idx_left] == 1:
                        bo = (sums - sums[idx_left]) < self.log_threshold
                        idx_right = list(bo + ~m)[idx_left:].index(True) + idx_left
                except ValueError:
                    idx_right = len(m)
                idxs_b.append(list(range(idx_left, idx_right)))
                idx_left = idx_right

            indices_to_compact.append(idxs_b)

        return indices_to_compact


class IdentityChunker(nn.Module):
    """
    IdentityChunker docstring
    """
    def __init__(self, *args, **kargs):
        super(IdentityChunker, self).__init__()

    def forward(self, x, *args, masks_dict=None, **kwargs):
        assert masks_dict is not None
        return [[None]] * x.size(0)  # return empty list of lists of batch size


def batch_remove_subsets(batch_L):
    batch_filtered = []
    for L in batch_L:
        filtered = filter(lambda f: not any(set(f) < set(g) for g in L), L)
        batch_filtered.append(list(filtered))
    return batch_filtered
