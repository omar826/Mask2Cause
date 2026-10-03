
import math
import torch
from torch import nn
from torch.nn import functional as F

class DataEmbedding_Inverted(nn.Module):
    """
    Tokenizes the history window of each variable into a single dense vector by projecting the sequence length dimension.
    """
    def __init__(self, seq_len, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.value_embedding(x)
        return self.dropout(x)

class CausalMaskedAttention(nn.Module):
    """
    Self-attention mechanism that incorporates a learned structural adjacency mask into the attention scores.
    """
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, mix=False):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = True
        self.query_projection = nn.Linear(d_model, n_heads * d_keys)
        self.key_projection = nn.Linear(d_model, n_heads * d_keys)
        self.value_projection = nn.Linear(d_model, n_heads * d_values)
        self.out_projection = nn.Linear(n_heads * d_values, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, adjacency_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        Q = self.query_projection(queries).view(B, L, H, -1)
        K = self.key_projection(keys).view(B, S, H, -1)
        V = self.value_projection(values).view(B, S, H, -1)
        scores = torch.einsum("blhe,bshe->bhls", Q, K)
        scores = scores / math.sqrt(Q.shape[-1])
        mask_log = torch.log(adjacency_mask + 1e-6) 
        scores = scores + mask_log.unsqueeze(0).unsqueeze(0)
        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhls,bshd->blhd", attn_weights, V)
        out = out.reshape(B, L, -1)
        return self.out_projection(out), attn_weights

class EncoderLayer(nn.Module):
    """
    A single Transformer encoder layer consisting of causal masked attention followed by a point-wise feed-forward network.
    """
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attn = CausalMaskedAttention(d_model, n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x, adjacency_mask):
        new_x, _ = self.attn(x, x, x, adjacency_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)

class CausalGraphTransformer(nn.Module):
    """
    Transformer-based architecture that learns a latent causal graph to guide attention across multivariate time series.
    """
    def __init__(self, args):
        super().__init__()
        self.num_variates = args.enc_in
        self.seq_len = args.seq_len
        self.d_model = getattr(args, 'd_model', 64)
        self.n_heads = getattr(args, 'n_heads', 4)
        self.e_layers = getattr(args, 'e_layers', 2)
        self.diagonal_force = getattr(args, 'diagonal_force', 100.0)
        self.var_embedding = nn.Parameter(torch.randn(1, self.num_variates, self.d_model))
        self.enc_embedding = DataEmbedding_Inverted(self.seq_len, self.d_model, dropout=getattr(args, "dropout", 0.1))
        self.adj_logits = nn.Parameter(torch.ones(self.num_variates, self.num_variates) * getattr(args, "adj_init", 1.0))
        self.register_buffer('self_mask', torch.eye(self.num_variates))
        self.encoder = nn.ModuleList([
            EncoderLayer(self.d_model, self.n_heads, dropout=getattr(args, "dropout", 0.1)) for _ in range(self.e_layers)
        ])
        self.mu_projector = nn.Linear(self.d_model, 1)
        self.sigma_projector = nn.Linear(self.d_model, 1)

    def get_adjacency_matrix(self):
        """
        Computes the sigmoid-activated adjacency matrix with an enforced diagonal bias for self-dependency.
        """
        effective_logits = self.adj_logits + (self.self_mask * self.diagonal_force)
        return torch.sigmoid(effective_logits)

    def forward(self, x):
        """
        Passes input through embeddings and encoder layers to predict the mean and variance for the next time step.
        """
        x_enc = self.enc_embedding(x)
        x_enc = x_enc + self.var_embedding
        mask = self.get_adjacency_matrix()
        for layer in self.encoder:
            x_enc = layer(x_enc, mask)
        mu = self.mu_projector(x_enc).squeeze(-1)
        sigma_out = self.sigma_projector(x_enc).squeeze(-1)
        sigma_sq = F.softplus(sigma_out) + 1e-6 
        return mu, sigma_sq

from causalDiscovery.datasets import (load_heteroscedastic_data, load_raw_traffic_data,
    load_dream3_data, load_lorenz_data, load_var_data, load_rhino_arrays)

def train_and_validate(config, device, verbose=False):
    from causalDiscovery.training import train
    return train(config, device, model_name="nll", verbose=verbose)
