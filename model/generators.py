"""
The idea is to have a class that inicializes the representation generation mechanism.
Then, a forward pass gives a tensor (or batch of tensors) and the corresponding list of indices
and performs the generation.
"""
import torch
import torch.nn as nn
from model.utils import abs_max_pooling
from model.customlayers import Attention
from model.utils import hotfix_pack_padded_sequence


class EmbeddingGenerator(nn.Module):
    """Class to create all non-parametrized generators.
    Args:
        pool_function: non-parametrized pooling function, which accepts a
            `torch.tensor` input, a dim keyword argument, and possible extra
            keyword arguments, which may vary for different pooling functions.
        device: `torch.device` to use, for cpu or cuda.
    Returns:
        Generator module.
    """
    def __init__(self, pool_function=abs_max_pooling, device=torch.device("cpu")):
        super().__init__()
        try:
            T = torch.rand((50, 768))
            _ = pool_function(T, dim=0, token_ids=torch.randint(30000, (50,)))
            self.pool_function = pool_function
            self.device = device
        except Exception as error:
            raise ValueError("The pool_function seems to not work!")

    def forward(self, tensors_batch, indices, masks_dict=None, token_ids=None):
        """Given a set of input tensors and indices to compact (from a chunking
        module) returns new generated (i.e. compressed) embeddings. The core is
        the generate() method, which could be itself the forward method acutally,
        but we separated the implementation, in case future iterations implement
        other procedures in the forward() pass.
        Args:
            tensors_batch: `torch.tensor` of size (batch, max_length, embedding_dim)
            indices: list (batch length) of list (sequence length) of tuples
            (indices to compact), coming from the chunking module.
            masks_dict: dictionary of masks (torch.tensor of uint8) with keys
                `padding_mask`, `seq_pair_mask` and `regular_tokens_mask`,
                coming from the Transformer module.
            token_ids: `torch.tensor` with token_ids of size (batch, max_length).
        Returns:
            compact_representations: compressed generated `torch.tensor`
            compact_masks_dict: dictionary like masks_dict, but with masks for
                the new compressed embeddings.
            compression_rate: float of the average compression rate
                (i.e. compressed number of regular embeddings over original
                number of regular embeddings, not <cls>, <sep> or <pad>).
        """
        assert masks_dict is not None
        compact_representation, compact_masks_dict, compression_rate = self.generate(
            tensors_batch, indices, masks_dict=masks_dict, token_ids=token_ids
        )
        return compact_representation, compact_masks_dict, compression_rate.item()

    def generate(
        self, tensors_batch, indices_batch, masks_dict=None, token_ids=None
    ):
        """Core of the Generator Module. Given a set of input tensors and
        indices to compact (from a chunking module) returns new generated
        (i.e. compressed) embeddings.
        Args:
            tensors_batch: `torch.tensor` of size (batch, max_length, embedding_dim)
            indices_batch: list (batch length) of list (sequence length) of tuples
            (indices to compact), coming from the chunking module.
            masks_dict: dictionary of masks (torch.tensor of uint8) with keys
                `padding_mask`, `seq_pair_mask` and `regular_tokens_mask`,
                coming from the Transformer module.
            token_ids: `torch.tensor` with token_ids of size (batch, max_length).
        Returns:
            compact_representations: compressed generated `torch.tensor`
            compact_masks_dict: dictionary like masks_dict, but with masks for
                the new compressed embeddings.
            compression_rate: float of the average compression rate
                (i.e. compressed number of regular embeddings over original
                number of regular embeddings, not <cls>, <sep> or <pad>).
        """
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
                # Apply pooling function for the group of tensors in idx_tuple
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
    """Class to create all *parametrized* generators. It is equivalent and for
    the most part equivalent to EmbeddingGenerator, but instead of a pooling function
    a pooling network is passed.
    Args:
        embedding_dim: dimension of the embedding to initialize `gen_net`
        gen_net: parametrized neural net that models many-to-one relationship.
            It accepts a `torch.tensor` input, and the dimension being compressed
            is always the dim=1 (i.e. the second one) and the tensor has size
            (batch, len, embedding_dim). Batch will be typically 1 here, as we
            loop for each element in the batch sequentially.
        device: `torch.device` to use, for cpu or cuda.
    Returns:
        Generator module.
    """
    def __init__(self, embedding_dim=768, gen_net=None, device=torch.device("cpu"), **kwargs):
        super().__init__()
        assert gen_net is not None, "You must provide a valid Generator Network gen_net!"

        self.device = device
        self.gen_net = gen_net(embedding_dim=embedding_dim, device=device).to(device)

    def forward(self, tensors_batch, indices_batch, masks_dict=None, **kwargs):
        """Given a set of input tensors and indices to compact (from a chunking
        module) returns new generated (i.e. compressed) embeddings.
        Args:
            tensors_batch: `torch.tensor` of size (batch, max_length, embedding_dim)
            indices: list (batch length) of list (sequence length) of tuples
            (indices to compact), coming from the chunking module.
            masks_dict: dictionary of masks (torch.tensor of uint8) with keys
                `padding_mask`, `seq_pair_mask` and `regular_tokens_mask`,
                coming from the Transformer module.
            token_ids: `torch.tensor` with token_ids of size (batch, max_length).
        Returns:
            compact_representations: compressed generated `torch.tensor`
            compact_masks_dict: dictionary like masks_dict, but with masks for
                the new compressed embeddings.
            compression_rate: float of the average compression rate
                (i.e. compressed number of regular embeddings over original
                number of regular embeddings, not <cls>, <sep> or <pad>).
        """
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
    """Parametrized generator function based on a 1D convolution + Attention
    (1 head) and a final Linear layer.
    Args:
        embedding_dim: size of the embedding dimension. Typically 768 for Bert Base.
        device: `torch.device` to use, for cpu or cuda.
    Returns:
        Module that does a transformation on the second dimension. (see forward())
    """
    def __init__(self, embedding_dim, device=torch.device('cpu'), **kwargs):
        super().__init__()

        self.device = device

        self.conv1D = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=5, padding=2)  # padding=(kernel-1)/2
        self.skip_weight = nn.Parameter(torch.tensor(0.1, requires_grad=True))
        self.attend = Attention(embedding_dim, attention_type="general", device=device)
        self.linear = nn.Linear(embedding_dim, embedding_dim)

        # Initialize the attention vec for the generation of compressed rep
        init_normal = torch.empty(1, 1, embedding_dim).normal_(
            mean=0, std=0.3
        )
        self.attention_vecs = nn.Parameter(
            init_normal.detach().requires_grad_(True).to(device)
        )

    def forward(self, inp, mask=None, **kwargs):
        """Perform forward pass of the generator net.
        Args:
            inp: `torch.tensor` of size (batch, length, embedding_dim)
            mask: 1s and 0s mask of `torch.int8`, size (batch, length, 1).
                Generally a mask won't be necessary because sequences are processed
                sequentially so no padding is involved and everything is processed.
        Returns:
            output: generated `torch.tensor` of size (batch, 1, embedding_dim)
        """
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

        return self.linear(output)


class LSTM(nn.Module):
    """Parametrized generator function based on an LSTM, for which the second
    dimension (dim=1) is the dimension of recurrence.
    Args:
        embedding_dim: size of the embedding dimension. Typically 768 for Bert Base.
        device: `torch.device` to use, for cpu or cuda.
    Returns:
        Module that does a transformation on the second dimension. (see forward())
    """
    def __init__(self, embedding_dim, hidden_dim=768, num_layers=1, dropout=0.0, device=torch.device('cpu'), **kwargs):
        super().__init__()
        self.device = device
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.directions = 1
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            bidirectional=False,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bias=True,
        )

    def init_hidden(self, batch_size):
        # As of the documentation from nn.LSTM in pytorch, the input to the lstm cell is
        # the input and a tuple of (h, c) hidden state and memory state. We initialize that
        # tuple with the proper shape: num_layers*directions, batch_size, hidden_dim. Don't worry
        # that the batch here is second, this is dealt with internally if the lstm is created with
        # batch_first=True
        shape = (self.num_layers * self.directions, batch_size, self.hidden_dim)
        return (
            torch.zeros(shape, requires_grad=True, device=self.device),
            torch.zeros(shape, requires_grad=True, device=self.device),
        )

    def forward(self, inp, mask=None, **kwargs):
        """Perform forward pass of the generator net.
        Args:
            inp: `torch.tensor` of size (batch, length, embedding_dim)
            mask: 1s and 0s mask of `torch.int8`, size (batch, length, 1).
                Generally a mask won't be necessary because sequences are processed
                sequentially so no padding is involved and everything is processed.
        Returns:
            output: generated `torch.tensor` of size (batch, 1, embedding_dim)
        """
        if mask is None:
            mask = torch.ones((inp.size(0), inp.size(1)),
                              dtype=torch.int8,
                              device=inp.device,
                              requires_grad=False)
        # Calculate original lengths
        lengths = mask.sum(dim=1)
        packed_tensors = hotfix_pack_padded_sequence(
            inp, lengths, enforce_sorted=False, batch_first=True
        )
        # detach to make the computation graph for the backward pass only for 1 sequence
        hidden, cell = self.init_hidden(inp.size(0))  # feed with batch_size
        lstm_out, (hidden_out, cell_out) = self.lstm(packed_tensors,
                                                     (hidden, cell))

        # we unpack and use the last lstm output for classification
        unpacked_output = torch.nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True
        )[0]
        # we return the mean over non-masked tokens, although we could do
        # something fancier like getting the last hidden state.
        return unpacked_output.sum(dim=1) / mask.sum(dim=1)

class IdentityGenerator(nn.Module):
    """This is just an identity Generator if we need to keep a generator
    network for convenience but bypass it. For instance, it is used implicitly
    when no generator argument is passed and still we want it to create the
    end to end model; it is fully bypassed still then compression=False in the
    end to end model forward pass.
    """
    def __init__(self, device=torch.device('cpu'), *args, **kwargs):
        super(IdentityGenerator, self).__init__()
        self.device = device

    def forward(self, x, *args, masks_dict=None, **kwargs):
        assert masks_dict is not None
        return x, masks_dict
