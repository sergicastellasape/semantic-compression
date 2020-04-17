"""
Decoder models: any model whose output is in the orginal embeddings space,
like a reconstructor decoder for next, previos or self reconstruction, to
the 'pointwise mutual information estimators'.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Predictor(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim=1024,
                 device=torch.device('cpu')):
        self.device = device

        self.fc1 = nn.Linear(embedding_dim, hidden_dim, bias=True)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, embedding_dim, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, inp, mask):
        # inp.size()= batch, seq_length, embedding_dim
        h1 = self.elu(self.fc1(inp * mask))
        out = self.tanh(self.fc2(h1 * mask))
        return out

    def loss(self, prediction, target, mask, loss_type='cosine'):
        if loss_type == 'cosine':
            sim = F.cosine_similarity(prediction, target, dim=-1)
            return torch.mean(sim, dim=)
        elif loss_type == 'L1':
            return F.l1_loss(prediction, target, reduce='mean')
