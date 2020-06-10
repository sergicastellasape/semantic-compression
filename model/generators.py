"""
The idea is to have a class that inicializes the representation generation mechanism.
Then, a forward pass gives a tensor (or batch of tensors) and the corresponding list of indices
and performs the generation.
"""
import torch
import torch.nn as nn
from model.utils import abs_max_pooling
from model.customlayers import Attention


class EmbeddingGenerator(nn.Module):
    def __init__(self, pool_function=abs_max_pooling, device=torch.device("cpu")):
        super().__init__()
        try:
            t = torch.rand((16, 50, 768))
            _ = pool_function(t)
            self.pool_function = pool_function
            self.device = device
        except Exception as error:
            raise ValueError("The pool_function seems to not work!")

    def forward(self, input_tensors, indices, masks_dict=None, token_ids=None):
        assert masks_dict is not None
        compact_representation, compact_masks_dict, compression_rate = self.generate(
            input_tensors, indices, masks_dict=masks_dict, token_ids=token_ids
        )
        return compact_representation, compact_masks_dict, compression_rate.item()

    def generate(
        self, tensors_batch, indices_batch, masks_dict=None, token_ids=None
    ):
        # tensors_batch.shape() = batch, seq_length, embedding_size
        # indices batch: list of lists of tuples
        # [[(0,), (1,), (2, 3, 4), (5,), (6,)], [(etc.)]]

        compact_tensors_batch = torch.zeros_like(tensors_batch, device=self.device)

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
                # Apply pooling function for the group of tensors
                joint = self.pool_function(
                    tensors_batch[b, idx_tuple, :], dim=0,
                    token_ids=token_ids[b, idx_tuple])  # idx_tuple dimension
                compact_tensors_batch[b, i, :] = joint.requires_grad_(True)

                # Make sure we're not mixing up regular tokens, special tokens
                # and padding tokens.
                assert masks_dict["padding_mask"][b, idx_tuple].bool().any() ==\
                    masks_dict["padding_mask"][b, idx_tuple].bool().all()
                assert masks_dict["seq_pair_mask"][b, idx_tuple].bool().any() ==\
                    masks_dict["seq_pair_mask"][b, idx_tuple].bool().all()
                assert masks_dict["regular_tokens_mask"][b, idx_tuple].bool().any() ==\
                    masks_dict["regular_tokens_mask"][b, idx_tuple].bool().all()

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
        # To remove the dimensions in the sequence length where all the
        # sequences are now padded because of the compression
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


class ParamEmbeddingGenerator(nn.Module):
    def __init__(self, embedding_dim=768, gen_net=None, device=torch.device("cpu"), **kwargs):
        super().__init__()
        assert gen_net is not None

        self.device = device
        self.gen_net = gen_net(embedding_dim=embedding_dim, device=device).to(device)

    def forward(self, tensors_batch, indices_batch, masks_dict=None):
        # tensors_batch.shape() = batch, seq_length, embedding_size
        # indices batch: list of lists of tuples
        # [[(0,), (1,), (2, 3, 4), (5,), (6,)], [(etc.)]]

        compact_tensors_batch = torch.zeros_like(tensors_batch, device=self.device)

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
                # Forward pass on Generator Net for group of tensors
                joint = self.gen_net(tensors_batch[b, idx_tuple, :].unsqueeze(0))
                compact_tensors_batch[b, i, :] = joint

                assert masks_dict["padding_mask"][b, idx_tuple].bool().any() ==\
                    masks_dict["padding_mask"][b, idx_tuple].bool().all()
                assert masks_dict["seq_pair_mask"][b, idx_tuple].bool().any() ==\
                    masks_dict["seq_pair_mask"][b, idx_tuple].bool().all()
                assert masks_dict["regular_tokens_mask"][b, idx_tuple].bool().any() ==\
                    masks_dict["regular_tokens_mask"][b, idx_tuple].bool().all()

                # update compact masks. Given the guarantees from the bracketing to
                # not mix padding, regular and special tokens, the 'loose' criteria
                # can be assumed to be 'hard' if any of the tokens is not padding
                # (0s), add it as non-padding (1s)
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
        # To remove the dimensions in the sequence length where all the
        # sequences are now padded because of the compression
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
            compression_rate.item()
        )


class ConvAtt(nn.Module):
    def __init__(self, embedding_dim, device=torch.device('cpu'), **kwargs):
        super().__init__()

        self.device = device

        self.conv1D = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=5, padding=2)  # padding=(kernel-1)/2
        self.skip_weight = nn.Parameter(torch.tensor(0.1, requires_grad=True))
        self.attend = Attention(embedding_dim, attention_type="general", device=device)

        # Initialize the attention vec for the generation of compressed rep
        init_normal = torch.empty(1, 1, embedding_dim).normal_(
            mean=0, std=0.3
        )
        self.attention_vecs = nn.Parameter(
            init_normal.detach().requires_grad_(True).to(device)
        )

    def forward(self, inp, mask=None):
        if mask is None:
            mask = torch.ones_like(inp, dtype=torch.int8)

        # concatenate along sequence length dimension
        query = self.attention_vecs.repeat(
            inp.size(0), 1, 1
        )  # expand for batch size

        # inp.size() = (b, len, embedding_dim)
        conv_out = self.conv1D(inp.transpose(1, 2)).transpose(1, 2) * mask
        mix = (1 - self.skip_weight) * inp + self.skip_weight * conv_out
        output, _ = self.attend(query, mix, context_mask=mask)
        return output


class IdentityGenerator(nn.Module):
    def __init__(self, device=torch.device('cpu'), *args, **kwargs):
        super(IdentityGenerator, self).__init__()
        self.device = device

    def forward(self, x, *args, masks_dict=None, **kwargs):
        assert masks_dict is not None
        return x, masks_dict
