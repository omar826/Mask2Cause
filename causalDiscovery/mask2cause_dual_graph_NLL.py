"""Two global graphs, separate mean/variance paths, shared encoder weights.

The original NLL architecture and loader are unchanged. Each graph is shared
across all encoder layers and attention heads in its respective prediction path.
Only variable-local tokenization is shared as activations; cross-variable
representations are computed independently for the two output heads.
"""
import torch
from torch import nn
from torch.nn import functional as F

from causalDiscovery.mask2cause_NLL import CausalGraphTransformer as SharedGraphTransformer


class CausalGraphTransformer(SharedGraphTransformer):
    def __init__(self, args):
        super().__init__(args)
        # Retain adj_logits as the mean graph for compatibility and matched init.
        self.variance_adj_logits = nn.Parameter(self.adj_logits.detach().clone())

    def get_adjacency_matrices(self):
        mean = self.get_adjacency_matrix()
        variance = torch.sigmoid(self.variance_adj_logits + self.self_mask * self.diagonal_force)
        return mean, variance

    def forward(self, x):
        tokens = self.enc_embedding(x) + self.var_embedding
        mean_graph, variance_graph = self.get_adjacency_matrices()
        mean_tokens, variance_tokens = tokens, tokens
        for layer in self.encoder:
            mean_tokens = layer(mean_tokens, mean_graph)
            variance_tokens = layer(variance_tokens, variance_graph)
        mu = self.mu_projector(mean_tokens).squeeze(-1)
        var = F.softplus(self.sigma_projector(variance_tokens).squeeze(-1)) + 1e-6
        return mu, var
