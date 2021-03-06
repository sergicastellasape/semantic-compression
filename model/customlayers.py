import torch
import torch.nn as nn

# this implementation is from torchnlp from Michael Petrochuk, the copy paste is because
# the fulltorchnlp library has a dependency on torch 1.0.0, and I need to use a newer version
class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type="general", device=torch.device('cpu')):
        super(Attention, self).__init__()

        if attention_type not in ["dot", "general"]:
            raise ValueError("Invalid attention type selected.")
        self.device = device
        self.attention_type = attention_type
        if self.attention_type == "general":
            self.linear_in = nn.Linear(dimensions, dimensions, bias=True)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context, query_mask=None, context_mask=None, temperature=1.):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        context_len = context.size(1)

        # TODO: Include mask on PADDING_INDEX?
        # this is still bad.
        if query_mask is not None:
            query_mask = query_mask.float()
        else:
            query_mask = torch.ones_like(query, dtype=torch.float, device=self.device)
        if context_mask is not None:
            context_mask = context_mask.float()
        else:
            context_mask = torch.ones_like(context, dtype=torch.float, device=self.device)

        # Generate the attention mask where it only 'accepts' where both query and
        # context are 1s: # (batch_size, output_len, context_len)
        bit_att_mask = torch.bmm(query_mask[:, :, 0].unsqueeze(-1),
                                 context_mask[:, :, 0].unsqueeze(-1).transpose(1, 2))
        # Convert to 0s and -inf for softmax
        extended_attention_mask = (1. - bit_att_mask) * -10000.0

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        # (batch_size, output_len, dimensions) * (batch_size, context_len, dimensions) ->
        # (batch_size, output_len, context_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous()) + extended_attention_mask

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, context_len)
        attention_weights = self.softmax(attention_scores / temperature)
        attention_weights = attention_weights.view(batch_size, output_len, context_len)

        # (batch_size, output_len, context_len) * (batch_size, context_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights
