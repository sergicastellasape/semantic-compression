import torch
import torch.nn as nn
import torch.nn.functional as F

cos = F.cosine

class InfoCoder(nn.Module):
    def __init__(self,
                 embedding_dim,
                 infospace_dim,
                 device=torch.device('cpu')):
        self.device = device
        self.fc1 = nn.Linear(embedding_dim, infospace_dim)
        self.cosine_sim = nn.CosineSimilarity(dim=-1)
        self.loss_fn = nn.L1Loss()

    def forward(self, inp):
        # inp.size() = batch, len, embedding_dim
        return self.fc1(inp)

    def loss(self, info_embeddings, target_similarity):
        # info_embeddings.size() = batch, len, embedding_dim
        pairwise_sim = self.cosine_sim(
            info_embeddings.unsqueeze(-2),
            info_embeddings.unsqueeze(-3)
        )

