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

    def forward(self, input_tensors, indices, mask_dict=mask_dict):
        compact_representation, mask = self.generate(input_tensors, indices, mask_dict=mask_dict)
        compact_dict = {}
        return compact_representation, compact_dict

    def generate(self, tensors_batch, indices_batch, mask_dict=mask_dict):
        # tensors_batch.shape() = batch, seq_length, embedding_size
        # indices batch: list of lists of tuples
        # [[(0,), (1,), (2, 3, 4), (5,), (6,)], [(etc.)]]

        compact_tensors_batch = self.initialize_padding_tensor_like(tensors_batch)
        # as all are zeros, this starts as an all false boolean mask
        batch_size, max_len, _ = tensors_batch.size()
        mask_padding = torch.zeros((batch_size, max_len), dtype=torch.bool, device=self.device)

        for b, chunk_indices in enumerate(indices_batch):
            for i, idx_tuple in enumerate(chunk_indices):
                # for each group calculate the compacted with "pool_function" (which will eventually
                # be something more complex, not just max pooling)
                joint = self.pool_function(tensors_batch[b, idx_tuple, :].unsqueeze(0))
                compact_tensors_batch[b, i, :] = joint
                mask_padding[b, i] = True
                
                    
        # To remove the dimensions in the sequence length where all the sequences are now padded because
        # of the compression
        mask_not_all_padding = mask_padding.sum(dim=0) == True


        return compact_tensors_batch[:, mask_not_all_padding, :], mask_padding[:, mask_not_all_padding]

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
        compact_dict = {}
        return x, compact_dict
