"""
The idea is to have a class that inicializes the representation generation mechanism.
Then, a forward pass gives a tensor (or batch of tensors) and the corresponding list of indices
and performs the generation.
"""
import torch
import torch.nn as nn
from model.utils import abs_max_pooling


class EmbeddingGenerator():
    def __init__(self, pool_function=abs_max_pooling, device=torch.device('cpu')):
        try:
            t = torch.rand((16, 50, 200))
            _ = pool_function(t)
            self.pool_function = pool_function
            self.device = device
        except:
            raise ValueError("The pool_function seems to not work!")

    def forward(self, input_tensors, indices):
        compact_representation = self.generate(input_tensors, indices)
        return compact_representation

    def generate(self, tensors_batch, indices_batch):
        # tensors_batch.shape() = batch, seq_length, embedding_size
        # indices batch: list of lists of tuples
        # [[(0,), (1,), (2, 3, 4), (5,), (6,)]]

        compact_tensors_batch = self.initialize_padding_tensor_like(tensors_batch)
        # as all are zeros, this starts as an all false boolean mask
        batch_size, max_len, _ = tensors_batch.size()
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=self.device)

        for b, chunk_indices in enumerate(indices_batch):
            for i, idx_tuple in enumerate(chunk_indices):
                # for each group calculate the compacted with "pool_function" (which will eventually
                # be something more complex, not just max pooling)
                joint = self.pool_function(tensors_batch[b, idx_tuple, :].unsqueeze(0))
                compact_tensors_batch[b, i, :] = joint
                mask[b, i] = True

        return compact_tensors_batch, mask

    def initialize_padding_tensor_like(self, tensor):
        # this function should be better, like initialize randomly from a distribution, because
        # the elements that are not overwritten by the originial tensors or pooling from them
        # will be padding
        init_tensor = torch.zeros_like(tensor, device=self.device)
        return init_tensor


class IdentityGenerator(nn.Module):
    def __init__(self, *args, **kwargs):
        super(IdentityGenerator, self).__init__()

    def forward(self, x, *args, **kwargs):
        return x
