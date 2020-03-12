"""
The idea is to have a class that inicializes the representation generation mechanism.
Then, a forward pass gives a tensor (or batch of tensors) and the corresponding list of indices
and performs the generation.
"""
import torch
import torch.nn as nn
from model.utils import abs_max_pooling


class EmbeddingGenerator:
    def __init__(self, pool_function=abs_max_pooling, device=torch.device("cpu")):
        try:
            t = torch.rand((16, 50, 200))
            _ = pool_function(t)
            self.pool_function = pool_function
            self.device = device
        except Exception as error:
            raise ValueError("The pool_function seems to not work!")

    def forward(self, input_tensors, indices, masks_dict=None):
        assert masks_dict is not None
        compact_representation, compact_masks_dict, compression_rate = self.generate(
            input_tensors, indices, masks_dict=masks_dict
        )
        return compact_representation, compact_masks_dict, compression_rate.item()

    def generate(
        self, tensors_batch, indices_batch, masks_dict=None, return_comp_rate=False
    ):
        # tensors_batch.shape() = batch, seq_length, embedding_size
        # indices batch: list of lists of tuples
        # [[(0,), (1,), (2, 3, 4), (5,), (6,)], [(etc.)]]

        compact_tensors_batch = self.initialize_padding_tensor_like(
            tensors_batch)

        # as all are zeros, this starts as an all false boolean mask
        batch_size, max_len, _ = tensors_batch.size()

        mask_padding = torch.zeros(
            (batch_size, max_len), dtype=torch.int8, device=self.device
        )  # initialize w/ -1 to detect later what didnt change
        mask_regular_tokens = torch.zeros(
            (batch_size, max_len), dtype=torch.int8, device=self.device
        )
        mask_seq_pair = torch.zeros(
            (batch_size, max_len), dtype=torch.int8, device=self.device
        )

        for b, chunk_indices in enumerate(indices_batch):
            for i, idx_tuple in enumerate(chunk_indices):
                # for each group calculate the compacted with "pool_function" (which will eventually
                # be something more complex, not just max pooling)
                joint = self.pool_function(
                    tensors_batch[b, idx_tuple, :].unsqueeze(0))
                compact_tensors_batch[b, i, :] = joint

                assert masks_dict["padding_mask"][b, idx_tuple].bool().any() == masks_dict["padding_mask"][b, idx_tuple].bool().all()
                assert masks_dict["seq_pair_mask"][b, idx_tuple].bool().any() == masks_dict["seq_pair_mask"][b, idx_tuple].bool().all()
                assert masks_dict["regular_tokens_mask"][b, idx_tuple].bool().any() == masks_dict["regular_tokens_mask"][b, idx_tuple].bool().all()

                # update compact masks. Given the guarantees from the bracketing to
                # not mix padding, regular and special tokens, the 'loose' criteria
                # can be assumed to be 'hard'
                # if any of the tokens is not padding (0s), add it as non-padding (1s)
                if masks_dict["padding_mask"][b, idx_tuple].sum() != 0:
                    mask_padding[b, i] = 1
                # if some tokens are "regular" add the index to the mask
                if masks_dict["regular_tokens_mask"][b, idx_tuple].sum() != 0:
                    mask_regular_tokens[b, i] = 1
                # if no token belongs to the first sequence, add it to the mask seq pair
                # we deal with the padding mask for 'seq_pair' later
                if masks_dict["seq_pair_mask"][b, idx_tuple].prod() != 0:
                    mask_seq_pair[b, i] = 1

        mask_seq_pair[mask_padding == 0] = -1.
        # To remove the dimensions in the sequence length where all the sequences are now padded because
        # of the compression
        all_padding_elements = mask_padding.sum(dim=0) == 0  # True where ALL elements were kept unchanged
        mask_remove_unnecessary_padding = ~all_padding_elements
        compact_dict = {
            "padding_mask": mask_padding[:, mask_remove_unnecessary_padding],
            "regular_tokens_mask": mask_regular_tokens[:, mask_remove_unnecessary_padding],
            "seq_pair_mask": mask_seq_pair[:, mask_remove_unnecessary_padding],
        }

        compression_rate = (
            torch.sum(compact_dict["regular_tokens_mask"]).float() /
            torch.sum(masks_dict["regular_tokens_mask"]).float()
        )

        return (
            compact_tensors_batch[:, mask_remove_unnecessary_padding, :],
            compact_dict,
            compression_rate,
        )

    def initialize_padding_tensor_like(self, tensor):
        # this function should be better, like initialize randomly from a distribution, because
        # the elements that are not overwritten by the originial tensors or pooling from them
        # will be padding
        init_tensor = torch.zeros_like(tensor, device=self.device)
        return init_tensor


class IdentityGenerator(nn.Module):
    def __init__(self, *args, **kwargs):
        super(IdentityGenerator, self).__init__()

    def forward(self, x, *args, masks_dict=None, **kwargs):
        assert masks_dict is not None
        return x, masks_dict
