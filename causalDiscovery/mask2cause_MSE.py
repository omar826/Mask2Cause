
import math
import torch
from torch import nn
from torch.nn import functional as F

class DataEmbedding_Inverted(nn.Module):
    """
    Tokenizes the history window of each variable into a single dense vector.
    Input: [Batch, Seq_Len, Num_Vars]
    Output: [Batch, Num_Vars, D_Model]
    """
    def __init__(self, seq_len, d_model, dropout=0.1):
        super().__init__()
        # Project raw history window (L) -> Token Dimension (D)
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: [Batch, Seq_Len, N_Vars]
        # Permute to [Batch, N_Vars, Seq_Len] so we treat Seq_Len as features
        x = x.permute(0, 2, 1)
        
        # Linear projection: [Batch, N, L] -> [Batch, N, D]
        x = self.value_embedding(x)
        return self.dropout(x)

class CausalMaskedAttention(nn.Module):
    """
    Standard Self-Attention but with a Learned Structural Mask injection.
    Calculates: Softmax( (QK^T)/sqrt(d) + log(Mask) ) V
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
        # queries, keys, values: [Batch, N_Vars, D_Model]
        # adjacency_mask: [N_Vars, N_Vars] (Values 0 to 1)
        
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # 1. Project Q, K, V
        # Shape: [Batch, N_Vars, Heads, D_Head]
        Q = self.query_projection(queries).view(B, L, H, -1)
        K = self.key_projection(keys).view(B, S, H, -1)
        V = self.value_projection(values).view(B, S, H, -1)

        # 2. Calculate Raw Scores: (Q @ K.T)
        # Shape: [Batch, Heads, N_Vars, N_Vars]
        scores = torch.einsum("blhe,bshe->bhls", Q, K)
        scores = scores / math.sqrt(Q.shape[-1])

        # 3. Apply Causal Mask (The Novelty)
        # We broadcast the learned [N, N] mask across Batch and Heads
        # Mask logic: Add log(mask). If mask ~ 1, add 0. If mask ~ 0, add -inf.
        # Clamp to avoid log(0)
        mask_log = torch.log(adjacency_mask + 1e-6) 
        
        # Shape broadcast: [1, 1, N, N]
        scores = scores + mask_log.unsqueeze(0).unsqueeze(0)

        # 4. Softmax & Context
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Shape: [Batch, Heads, N, D_head]
        out = torch.einsum("bhls,bshd->blhd", attn_weights, V)
        
        # 5. Output Projection
        # Shape: [Batch, N, D_Model]
        out = out.reshape(B, L, -1)
        return self.out_projection(out), attn_weights

class EncoderLayer(nn.Module):
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
        # x: [Batch, N, D]
        
        # Attention Block
        new_x, _ = self.attn(x, x, x, adjacency_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        # Feed Forward Block (using Conv1d for pointwise MLP)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm2(x + y)

class CausalGraphTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_variates = args.enc_in
        self.seq_len = args.seq_len
        self.d_model = getattr(args, 'd_model', 64) # Default to 64 if not set
        self.n_heads = getattr(args, 'n_heads', 4)
        self.e_layers = getattr(args, 'e_layers', 2)
        self.diagonal_force = getattr(args, 'diagonal_force', 100.0)
        
        # 1. Variable Embeddings (Eid)
        # Unique vector for each variable index
        self.var_embedding = nn.Parameter(torch.randn(1, self.num_variates, self.d_model))
        
        # 2. Window Tokenizer
        self.enc_embedding = DataEmbedding_Inverted(self.seq_len, self.d_model, dropout=getattr(args, "dropout", 0.1))

        # 3. The Learnable Causal Graph (Adjacency Matrix)
        # Logits initialized to 2.0 (Sigmoid ~ 0.88) for a "dense start"
        self.adj_logits = nn.Parameter(torch.ones(self.num_variates, self.num_variates) * getattr(args, "adj_init", -1.0))
        
        # Buffer for Self-Loops (Variables must attend to themselves)
        self_mask = torch.eye(self.num_variates)
        self.register_buffer('self_mask', self_mask)

        # 4. Encoder Stack
        self.encoder = nn.ModuleList([
            EncoderLayer(self.d_model, self.n_heads, dropout=getattr(args, "dropout", 0.1)) for _ in range(self.e_layers)
        ])
        
        # 5. Prediction Head
        self.projector = nn.Linear(self.d_model, 1, bias=True)

    def get_adjacency_matrix(self):
        # Force diagonal to be High
        effective_logits = self.adj_logits + (self.self_mask * self.diagonal_force)
        return torch.sigmoid(effective_logits)

    def forward(self, x):
        # Input: [Batch, Seq_Len, N_Vars]
        
        # A. Tokenize: [Batch, N, D]
        x_enc = self.enc_embedding(x)
        
        # B. Add Variable Identity (Eid)
        x_enc = x_enc + self.var_embedding
        
        # C. Get Causal Mask
        mask = self.get_adjacency_matrix() # [N, N]
        
        # D. Transformer Layers
        for layer in self.encoder:
            x_enc = layer(x_enc, mask)
            
        # E. Prediction
        # x_enc is [Batch, N, D] -> Project to [Batch, N, 1]
        dec_out = self.projector(x_enc)
        
        # Squeeze last dim to get [Batch, N]
        return dec_out.squeeze(-1)

from causalDiscovery.datasets import (load_heteroscedastic_data, load_raw_traffic_data,
    load_dream3_data, load_lorenz_data, load_var_data, load_rhino_arrays)

def train_and_validate(config, device, verbose=False):
    from causalDiscovery.training import train
    return train(config, device, model_name="mse", verbose=verbose)
