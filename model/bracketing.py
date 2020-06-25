import itertools
from typing import List
from collections import Counter
import re
from sklearn import cluster
import random
import numpy as np
import torch
import torch.nn as nn
from model.utils import make_connectivity_matrix, log_zipf_law

cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-5)  # default similarity func.


class NNSimilarityChunker(nn.Module):
    """This class implements a model that performs "chunking" based on a
    similarity threshold, as defined by a similarity function (cosine similarity
    by default). It was deprecated because it is very slow, chunks can overlap
    and it's overall a bad implementation.
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
    """Wrapper that implements sklearn.cluster.AgglomerativeClustering for a batch
    of tensors.
    Args:
        threshold: distance threshold above which the clustering stops. If
            normalize is True, typicallly ranges from 0.9 to 1.7
        max_skip: maximum skipping of the sequentiality allowed by clustering,
            for instance, if max_skip=1 only contiguous points can be assigned
            the same cluster, whereas for max_skip=3 we allow a 'jump' of over
            2 datapoints.
        normalize: boolean which if True, does an L2 normalization on the
            embeddings before usin euclidean distance, such that it behaves
            closer to cosine similarity and the distance threshold is more
            predictable. Cosine similarity is not used as a distance metric
            because it doesn't accept ward linkage which stabilizes the
            clustering (with mean linkage, big clusters tend to swallow the
            rest, resulting in very 'unstable' clustering)
        device: `torch.device` to use, for cpu or cuda.
    """
    def __init__(self, threshold=0.9, max_skip=4, normalize=True, device=torch.device("cpu")):
        super().__init__()
        self.device = device
        # agg clustering wants distance not similarity
        # self.dist_threshold = 1 - threshold
        self.dist_threshold = threshold
        self.span = max_skip
        self.normalize = normalize
        self.id = nn.Identity()

    def forward(self, inp, masks_dict=None, mask_special_tokens=True, **kwargs):
        """Given a batch of sequences, performs agglomerative clustering on them,
        and returns a list of lists of tuples as indices for the embeddings that
        should be compressed. It only clusters regular tokens, and if
        mask_special_tokens is true, they remain in their own cluster. It is
        implemented sequentially for each element in the batch because sklearn
        implementation doesn't allow for parallelization and it would involve
        technical overhead, but there is no reason why it could not be
        parallelized.
        Args:
            inp: `torch.tensor` of size (batch, max_length, embedding_dim)
            masks_dict: dictionary of masks (torch.tensor of uint8) for the
                output with keys `padding_mask`, `seq_pair_mask` and
                `regular_tokens_mask`.
            mask_special_tokens: boolean which if True, leaves out of clustering
                special tokens (<cls>, <sep>) in addition to padding ones.
            **kwargs: just as a safeguard if a generic network call to trigger
                this method, which uses other keyword arguments.
        Returns:
            indices_to_compact: list of lists of tuples with the indices that
                the generator needs to compress.
        """
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
        if self.normalize:
            norm_inp = (inp / inp.norm(p=2, dim=-1, keepdim=True)).detach().cpu()
        else:
            norm_inp = inp.detach().cpu()
        # loop over each element in batch, I know. it's a shame this is sequential
        for b, embeddings in enumerate(norm_inp.numpy()):
            filtered_embedding = embeddings[keep_mask[b, :], :]
            # remove pad and special tokens if necessary
            seq_pair_mask = full_seq_pair_mask[b, keep_mask[b, :]]
            # keep track of the indices that need to be added later
            idxs_filtered_out = special_tokens[b, :].nonzero()[0]
            # iterate for the different values in seq_pair_mask, which will be
            # either [0] or [0, 1], so loop over each seq in the seq pair (if any)
            L, max_L = [], 0
            for i in list(Counter(seq_pair_mask).keys()):  # loop if two seq.
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
    """Class for the chunker that implements 'hard chunking', that is, segmenting
    a sequence in chunks whose size is a fixed 'window' or 'span'.
    Args:
        span: size of the chunks, for instance, if span=3, datapoints will be
            chunked as (1, 2, 3), (4, 5, 6), (7, 8, 9) etc.
        device: `torch.device` to use, for cpu or cuda.
    """
    def __init__(self, span=None, device=torch.device('cpu')):
        super().__init__()
        assert span is not None
        self.device = device
        self.id = nn.Identity()
        self.span = span

    def forward(self, inp, masks_dict=None, mask_special_tokens=True, **kwargs):
        """Performs the forward pass of hard chunking on a batch of tensors.
        Args:
            inp: `torch.tensor` of size (batch, max_length, embedding_dim)
            masks_dict: dictionary of masks (torch.tensor of uint8) for the
                output with keys `padding_mask`, `seq_pair_mask` and
                `regular_tokens_mask`.
            mask_special_tokens: boolean which if True, leaves out of clustering
                special tokens (<cls>, <sep>) in addition to padding ones.
            **kwargs: just as a safeguard if a generic network call to trigger
                this method, which uses other keyword arguments.
        Returns:
            indices_to_compact: list of lists of tuples with the indices that
                the generator needs to compress.
        """
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


class RndSpanChunker(nn.Module):
    """Variation of the hardspan chunker that randomizes the size of the chunks.
    Given a base span, the size of each chunk will be drawn from a uniform
    distribution of integers of U[1, 2 * span), such that the expected
    compression will be the same, but the chunks will not be of uniform size.
    Args:
        span: size of the chunks, for instance, if span=3, datapoints will be
            chunked as (1, 2, 3), (4, 5, 6), (7, 8, 9) etc.
        device: `torch.device` to use, cpu or cuda.
    """
    def __init__(self, span=None, device=torch.device('cpu')):
        super().__init__()
        assert span is not None
        self.device = device
        self.id = nn.Identity()
        self.span = span

    def forward(self, inp, masks_dict=None, mask_special_tokens=True, **kwargs):
        """Performs a forward pass of random chunking of a batch of tensors.
        Args:
            inp: `torch.tensor` of size (batch, max_length, embedding_dim)
            masks_dict: dictionary of masks (torch.tensor of uint8) for the
                output with keys `padding_mask`, `seq_pair_mask` and
                `regular_tokens_mask`.
            mask_special_tokens: boolean which if True, leaves out of clustering
                special tokens (<cls>, <sep>) in addition to padding ones.
            **kwargs: just as a safeguard if a generic network call to trigger
                this method, which uses other keyword arguments.
        Returns:
            indices_to_compact: list of lists of tuples with the indices that
                the generator needs to compress.
        """
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
                start, stop = start_stop[i], start_stop[i + 1]
                L_, next_ = [start], 0
                while next_ < stop:
                    next_ = start + random.randrange(1, 2 * self.span)
                    L_.append(next_) if next_ < stop else L_.append(stop)
                    start = next_
                # L_ = [0, 1]; [1, 4]; [4, 5]; etc. the 'ranges' to merge
                L = [tuple(range(L_[i], L_[i + 1])) for i in range(len(L_) - 1)]
                ordered_idxs.extend(L)
            # ordered_idxs = [(0,), (1, 2, 3), (4,), (5, 6, 7), (8, 9, 10), (11,)]
            indices_to_compact.append(ordered_idxs)
        return indices_to_compact


class FixedOutChunker(nn.Module):
    """Class for the chunker that implements 'fixed chunking', that is,
    segmenting in a predefined number of chunks, regardless of its length.
    Args:
        out_num: number of chunks to segment each sequence in.
        device: `torch.device` to use, for cpu or cuda.
    """
    def __init__(self, out_num=None, device=torch.device('cpu')):
        super().__init__()
        assert out_num is not None
        self.device = device
        self.id = nn.Identity()
        self.out_num = out_num

    def forward(self, inp, masks_dict=None, mask_special_tokens=True, **kwargs):
        """Performs a forward pass of random chunking of a batch of tensors.
        Args:
            inp: `torch.tensor` of size (batch, max_length, embedding_dim)
            masks_dict: dictionary of masks (torch.tensor of uint8) for the
                output with keys `padding_mask`, `seq_pair_mask` and
                `regular_tokens_mask`.
            mask_special_tokens: boolean which if True, leaves out of clustering
                special tokens (<cls>, <sep>) in addition to padding ones.
            **kwargs: just as a safeguard if a generic network call to trigger
                this method, which uses other keyword arguments.
        Returns:
            indices_to_compact: list of lists of tuples with the indices that
                the generator needs to compress.
        """
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
    """Class for the chunker that implements 'frequency based chunking'. This
    means that given an information budget: chunks are filled with embeddings
    from tokens and add their self-information (log probability), such as the
    less probable they are, the more informative they are thus the more
    information budget they use.
    Args:
        alpha: parameter to ass when the token frequencies are approximated
            using the Zipf law.
        log_threshold: log probability threshold. It is a negative value, which
            ranges typically from -4 to -60, where the more negative the more
            compression.
        device: `torch.device` to use, for cpu or cuda.
        **kwargs: just as a safeguard if a generic network call to trigger
                this class, which uses other keyword arguments.
    """
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
        """Performs a forward pass of random chunking of a batch of tensors.
        Args:
            inp: `torch.tensor` of size (batch, max_length, embedding_dim)
            masks_dict: dictionary of masks (torch.tensor of uint8) for the
                output with keys `padding_mask`, `seq_pair_mask` and
                `regular_tokens_mask`.
            mask_special_tokens: boolean which if True, leaves out of clustering
                special tokens (<cls>, <sep>) in addition to padding ones.
            token_ids: `torch.tensor` of size (batch, max_length) containing
                the token_ids that will be used as their rank to approximate
                their frequencies using the Zipf law.
            **kwargs: just as a safeguard if a generic network call to trigger
                this method, which uses other keyword arguments.
        Returns:
            indices_to_compact: list of lists of tuples with the indices that
                the generator needs to compress.
        """
        assert masks_dict is not None, "masks_dict is a required argument"

        batch_size = inp.size(0)

        if not mask_special_tokens:
            keep_mask = (masks_dict['padding_mask'] == 1).detach().cpu()
        else:
            keep_mask = (masks_dict['regular_tokens_mask'] == 1).detach().cpu()

        # 'the' is the first wordpiece token at position 1996
        token_log_likelihoods = log_zipf_law(token_ids.cpu(), rank_first=1996) * keep_mask
        sums = token_log_likelihoods.cumsum(dim=-1)

        indices_to_compact = []
        for b in range(batch_size):
            # m = [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
            m = keep_mask[b, :]
            idx_left, idx_right, finished, idxs_b = 0, 0, False, []
            while idx_right < len(m):
                try:
                    if m[idx_left] == 0:
                        idx_right = idx_left + 1
                    elif m[idx_left] == 1:
                        bo = (sums[b, :] - sums[b, idx_left]) < self.log_threshold
                        idx_right = (bo + ~m).tolist()[idx_left:].index(True) + idx_left

                except ValueError:
                    idx_right = len(m)
                idxs_b.append(tuple(range(idx_left, idx_right)))
                idx_left = idx_right

            indices_to_compact.append(idxs_b)

        return indices_to_compact


class IdentityChunker(nn.Module):
    """Placeholder chunker when it needs to be placed for consistency in different
    configurations of the model, but its output is useless.
    """
    def __init__(self, *args, **kargs):
        super(IdentityChunker, self).__init__()

    def forward(self, x, *args, masks_dict=None, **kwargs):
        assert masks_dict is not None
        return [[None]] * x.size(0)  # return empty list of lists of batch size


def batch_remove_subsets(batch_L):
    """Utility function for the deprecated NNSimilarity chunker that removes
    chunks that are fully contained by others.
    """
    batch_filtered = []
    for L in batch_L:
        filtered = filter(lambda f: not any(set(f) < set(g) for g in L), L)
        batch_filtered.append(list(filtered))
    return batch_filtered
