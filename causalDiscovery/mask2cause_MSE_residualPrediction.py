import csv
from xml.parsers.expat import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_curve, auc, f1_score, roc_auc_score
from tqdm import tqdm
import re

import matplotlib.pyplot as plt
import seaborn as sns
# Try importing DirectML for Windows GPU acceleration
try:
    import torch_directml
    HAS_DIRECTML = True
except ImportError:
    HAS_DIRECTML = False


# 1. NEW ARCHITECTURE: Causal Graph Transformer


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
        self.enc_embedding = DataEmbedding_Inverted(self.seq_len, self.d_model)

        # 3. The Learnable Causal Graph (Adjacency Matrix)
        # Logits initialized to 2.0 (Sigmoid ~ 0.88) for a "dense start"
        self.adj_logits = nn.Parameter(torch.ones(self.num_variates, self.num_variates) * (-3))
        
        # Buffer for Self-Loops (Variables must attend to themselves)
        self_mask = torch.eye(self.num_variates)
        self.register_buffer('self_mask', self_mask)

        # 4. Encoder Stack
        self.encoder = nn.ModuleList([
            EncoderLayer(self.d_model, self.n_heads) for _ in range(self.e_layers)
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
        out = self.projector(x_enc).squeeze(-1) # [Batch, N]
        
        last_input = x[:, -1, :]
        
        return last_input + out



# 2. DATA LOADING  

def load_raw_traffic_data(data_dir, seq_len, batch_size):
    data_path = os.path.join(data_dir, 'gen_data.npy')
    graph_path = os.path.join(data_dir, 'graph.npy')
    
    print(f"Loading Raw Data from {data_dir}...")
    
    if not os.path.exists(data_path) or not os.path.exists(graph_path):
        raise FileNotFoundError(f"Missing .npy files in {data_dir}")
        
    raw_data = np.load(data_path).astype(np.float32)
    GC = np.load(graph_path)
    
    num_vars = GC.shape[0]
    X_causal = raw_data[:, :, :num_vars] 
    
    mean = np.mean(X_causal, axis=(0, 1))
    std = np.std(X_causal, axis=(0, 1)) + 1e-5
    X_norm = (X_causal - mean) / std

    samples_X, samples_Y = [], []
    num_samples, time_steps, _ = X_norm.shape
    windows_per_sample = time_steps - seq_len
    
    if windows_per_sample < 1:
        raise ValueError(f"Sequence length ({seq_len}) too large.")

    for s_idx in range(num_samples):
        sample_data = X_norm[s_idx]
        for t in range(windows_per_sample):
            x_window = sample_data[t : t+seq_len]
            y_target = sample_data[t+seq_len]
            samples_X.append(x_window)
            samples_Y.append(y_target)
            
    samples_X = torch.tensor(np.array(samples_X))
    samples_Y = torch.tensor(np.array(samples_Y))
    
    print(f"Generated {len(samples_X)} clean windows.")
    
    dataset = TensorDataset(samples_X, samples_Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader, GC

def load_dream3_data(pt_path, seq_len, batch_size):
    print(f"Loading DREAM3 data from {pt_path}...")
    
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Data file not found: {pt_path}")
        
    # --- 1. Load Time Series ---
    try:
        data_dict = torch.load(pt_path, weights_only=False) 
    except TypeError:
        data_dict = torch.load(pt_path)
        
    if 'TsData' not in data_dict:
        raise ValueError(f"Expected key 'TsData', found: {data_dict.keys()}")
    
    X = data_dict['TsData'] # Shape: [966, 100]
    
    # Normalize (Important: Normalize across the whole dataset)
    mean = torch.mean(X, dim=0)
    std = torch.std(X, dim=0) + 1e-5
    X = (X - mean) / std
    
    # --- 2. Load Ground Truth ---
    parent_dir = os.path.dirname(os.path.dirname(pt_path)) # data/dream3
    filename = os.path.basename(pt_path) # Size10Ecoli1.pt
    
    # Extract the identifier (e.g., "Ecoli1" from "Size10Ecoli1.pt")
    # Matches "Ecoli1", "Yeast3", etc.
    match = re.search(r'(Ecoli|Yeast)[0-9]+', filename, re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not parse dataset ID (Ecoli/Yeast) from {filename}")
    
    dataset_id = match.group(0) # e.g., "Ecoli1"
    size_str = "Size100" if "Size100" in filename else "Size10" # Detect size
    
    # Construct probable graph filename
    # Pattern: InSilicoSize10-Ecoli1.tsv
    graph_filename = f"InSilico{size_str}-{dataset_id}.tsv"
    graph_path = os.path.join(parent_dir, 'TrueGeneNetworks', graph_filename)
    
    # Case-insensitive fallback if file not found (Windows/Linux differences)
    if not os.path.exists(graph_path):
        # Try finding the file by walking the directory
        graph_dir = os.path.join(parent_dir, 'TrueGeneNetworks')
        if os.path.exists(graph_dir):
            for f in os.listdir(graph_dir):
                if dataset_id.lower() in f.lower() and size_str.lower() in f.lower() and f.endswith('.tsv'):
                    graph_path = os.path.join(graph_dir, f)
                    print(f"Found graph file via search: {graph_path}")
                    break
    
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Could not locate graph file at {graph_path}")

    # Parse TSV
    num_nodes = X.shape[1]
    GC_true = np.zeros((num_nodes, num_nodes))
    
    with open(graph_path, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            # Format: "G1" "G2" "+"
            # G1 -> G2 means Source -> Target
            if len(row) >= 2:
                src_idx = int(row[0][1:]) - 1 # "G1" -> 0
                tgt_idx = int(row[1][1:]) - 1 # "G2" -> 1
                
                # Set adjacency (Source, Target) = 1
                GC_true[tgt_idx, src_idx] = 1

    print(f"Graph loaded. Shape: {GC_true.shape}. Edges: {np.sum(GC_true)}")

    # --- 3. Create Trajectory-Aware Windows ---
    samples_X, samples_Y = [], []
    
    # DREAM-3 standard trajectory length is 21
    TRAJ_LEN = 21 
    total_len = X.shape[0]
    
    # Verification
    if total_len % TRAJ_LEN != 0:
        print(f"WARNING: Data length {total_len} is not divisible by {TRAJ_LEN}. Assuming continuous.")
        num_trajs = 1
        TRAJ_LEN = total_len
    else:
        num_trajs = total_len // TRAJ_LEN
        print(f"Detected {num_trajs} trajectories of length {TRAJ_LEN}.")

    count_skipped = 0
    
    for i in range(num_trajs):
        # Slice out the specific experiment (e.g., steps 0-21, 21-42)
        start_idx = i * TRAJ_LEN
        end_idx = (i + 1) * TRAJ_LEN
        traj = X[start_idx : end_idx]
        
        # Sliding window ONLY within this clean trajectory
        # If trajectory is 21 and seq_len is 10, we get 11 samples.
        # We CANNOT take a window that crosses end_idx.
        for t in range(TRAJ_LEN - seq_len):
            x_window = traj[t : t+seq_len]     # Inputs: t to t+L
            y_target = traj[t+seq_len]         # Target: t+L (Residual calculated in model)
            
            samples_X.append(x_window)
            samples_Y.append(y_target)
        
    samples_X = torch.stack(samples_X)
    samples_Y = torch.stack(samples_Y)
    
    print(f"Created {len(samples_X)} clean samples (Skipped {num_trajs} boundary jumps).")
    
    dataset = TensorDataset(samples_X, samples_Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader, GC_true

def load_lorenz_data(path, seq_len, batch_size):
    print(f"Loading Lorenz data from {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
        
    data = np.load(path)
    
    # 1. Handle Key Variations
    if 'X_np' in data.files:
        X = data['X_np']
    elif 'X' in data.files:
        X = data['X']
    elif 'data' in data.files:
        X = data['data']
    else:
        raise ValueError(f"Could not find data key. Available: {data.files}")
        
    if 'Gref' in data.files:
        GC = data['Gref']
    elif 'GC' in data.files:
        GC = data['GC']
    elif 'true_graph' in data.files:
        GC = data['true_graph']
    else:
        raise ValueError("Could not find ground truth graph.")
    
    # 2. Check & Fix Dimensions
    # We need shape (Time, Vars). If (Vars, Time), transpose it.
    num_vars = GC.shape[0]
    
    # Case A: Single Time Series (N, T) -> Transpose to (T, N)
    if X.ndim == 2:
        if X.shape[0] == num_vars and X.shape[1] != num_vars:
            print(f"Transposing input from {X.shape} to ({X.shape[1]}, {X.shape[0]})...")
            X = X.T
            
    # Case B: Multiple Samples (Samples, N, T) -> Transpose to (Samples, T, N)
    elif X.ndim == 3:
        if X.shape[1] == num_vars and X.shape[2] != num_vars:
            print(f"Transposing input from {X.shape} to ({X.shape[0]}, {X.shape[2]}, {X.shape[1]})...")
            X = X.transpose(0, 2, 1)

    X = X.astype(np.float32)

    # 3. Normalize
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-5
    X = (X - mean) / std

    # 4. Windowing
    samples_X, samples_Y = [], []
    
    # Logic for Single Long Time Series
    if X.ndim == 2:
        L_data = len(X)
        for i in range(L_data - seq_len - 1):
            s_begin = i
            s_end = s_begin + seq_len
            r_index = s_end 
            samples_X.append(X[s_begin:s_end])
            samples_Y.append(X[r_index])
            
    # Logic for Multiple Samples
    elif X.ndim == 3:
        num_samples, time_steps, _ = X.shape
        for s in range(num_samples):
            for i in range(time_steps - seq_len - 1):
                samples_X.append(X[s, i : i+seq_len])
                samples_Y.append(X[s, i+seq_len])
        
    samples_X = torch.tensor(np.array(samples_X))
    samples_Y = torch.tensor(np.array(samples_Y))
    
    print(f"Created {len(samples_X)} Lorenz samples. Shape: {samples_X.shape}")
    
    # Use ALL data (no validation split)
    dataset = TensorDataset(samples_X, samples_Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader, GC


def load_var_data(path, seq_len, batch_size):
    print(f"Loading VAR data from {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
        
    data = np.load(path)
    
    # 1. Handle Key Variations
    if 'X_np' in data.files:
        X = data['X_np']  # Found your key!
    elif 'X' in data.files:
        X = data['X']
    elif 'data' in data.files:
        X = data['data']
    else:
        raise ValueError(f"Could not find data key. Available: {data.files}")
        
    if 'Gref' in data.files:
        GC = data['Gref'] # Found your key!
    elif 'GC' in data.files:
        GC = data['GC']
    elif 'true_graph' in data.files:
        GC = data['true_graph']
    else:
        raise ValueError("Could not find ground truth graph.")
    
    # 2. Check & Fix Dimensions 
    # We need shape (Time, Vars) for the Transformer.
    # If data is (Vars, Time), we transpose it.
    num_vars = GC.shape[0]
    
    # Case A: Single Time Series (N, T) -> Transpose to (T, N)
    if X.ndim == 2:
        if X.shape[0] == num_vars and X.shape[1] != num_vars:
            print(f"Transposing input from {X.shape} to ({X.shape[1]}, {X.shape[0]})...")
            X = X.T
            
    # Case B: Multiple Samples (Samples, N, T) -> Transpose to (Samples, T, N)
    elif X.ndim == 3:
        if X.shape[1] == num_vars and X.shape[2] != num_vars:
            print(f"Transposing input from {X.shape} to ({X.shape[0]}, {X.shape[2]}, {X.shape[1]})...")
            X = X.transpose(0, 2, 1)

    X = X.astype(np.float32)

    # 3. Normalize
    mean = np.mean(X, axis=0) # Mean across time
    std = np.std(X, axis=0) + 1e-5
    X = (X - mean) / std

    # 4. Windowing
    samples_X, samples_Y = [], []
    
    # Logic for Single Long Time Series (Typical for VAR)
    if X.ndim == 2:
        L_data = len(X)
        for i in range(L_data - seq_len - 1):
            s_begin = i
            s_end = s_begin + seq_len
            r_index = s_end 
            samples_X.append(X[s_begin:s_end])
            samples_Y.append(X[r_index])
            
    # Logic for Multiple Samples (If your npz has multiple experiments)
    elif X.ndim == 3:
        num_samples, time_steps, _ = X.shape
        for s in range(num_samples):
            for i in range(time_steps - seq_len - 1):
                samples_X.append(X[s, i : i+seq_len])
                samples_Y.append(X[s, i+seq_len])
        
    samples_X = torch.tensor(np.array(samples_X))
    samples_Y = torch.tensor(np.array(samples_Y))
    
    print(f"Created {len(samples_X)} VAR samples. Shape: {samples_X.shape}")
    
    dataset = TensorDataset(samples_X, samples_Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader, GC


# 3. METRICS  

def find_threshold_for_density(mask, target_density):
    weights = mask.flatten()
    weights = np.sort(weights)[::-1] 
    k = int(len(weights) * target_density)
    k = max(1, min(k, len(weights)-1))
    return weights[k]

def calculate_metrics_at_target(ground_truth, learned_prob):
    gt_density = np.mean(ground_truth)
    target_thresh = find_threshold_for_density(learned_prob, gt_density)
    pred_adj = (learned_prob > target_thresh).astype(int)
    
    diff = np.abs(ground_truth - pred_adj)
    shd = np.sum(diff)
    f1 = f1_score(ground_truth.flatten(), pred_adj.flatten(), zero_division=0)
    
    print(f"Threshold: {target_thresh:.4f} | SHD: {shd} | F1: {f1:.4f}")
    return shd, target_thresh

def calculate_metrics_best(ground_truth, learned_prob):
    gt_flat = ground_truth.flatten()
    prob_flat = learned_prob.flatten()
    precision, recall, _ = precision_recall_curve(gt_flat, prob_flat)
    aupr = auc(recall, precision)
    
    thresholds = np.linspace(0.01, 0.99, 100)
    best_shd = float('inf')
    best_thresh = 0.5
    for t in thresholds:
        pred_adj = (learned_prob > t).astype(int)
        diff = np.abs(ground_truth - pred_adj)
        current_shd = np.sum(diff)
        if current_shd < best_shd:
            best_shd = current_shd
            best_thresh = t
    return aupr, best_shd, best_thresh


# 4. UPDATED TRAIN & VALIDATE

def train_and_validate(config, device, verbose=False):
    # 1. Data Prep
    if 'lorenz' in config['data_dir'].lower():
        train_loader, GC_true = load_lorenz_data(config['data_dir'], config['seq_len'], config['batch_size'])
    elif 'var' in config['data_dir'].lower():
        train_loader, GC_true = load_var_data(config['data_dir'], config['seq_len'], config['batch_size'])
    elif 'dream' in config['data_dir'].lower():
        train_loader, GC_true = load_dream3_data(config['data_dir'], config['seq_len'], config['batch_size'])
    else:
        train_loader, GC_true = load_raw_traffic_data(config['data_dir'], config['seq_len'], config['batch_size'])
    # full_dataset = full_loader.dataset
    # train_size = int(0.8 * len(full_dataset))
    # val_size = len(full_dataset) - train_size
    # train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    # val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)
    
    # 2. Model Init
    class Args: pass 
    args = Args()
    args.enc_in = GC_true.shape[0]
    args.seq_len = config['seq_len']
    # Transformer Hyperparams (You can tune these via config too)
    args.d_model = config.get('d_model', 64)
    args.n_heads = config.get('n_heads', 4)
    args.e_layers = config.get('e_layers', 2)
    args.diagonal_force = config.get('diagonal_force', 100.0)
    
    # *** SWAPPED TO NEW MODEL ***
    model = CausalGraphTransformer(args).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.MSELoss()
    
    # 3. Training Loop
    patience = config['epochs']
    patience_counter = 0
    best_val_mse = float('inf')
    best_loss = float('inf')
    max_epochs = 5000
    early_stop = config.get('early_stop', False)
    if not early_stop:
        max_epochs = config['epochs']
    
    iterator = range(max_epochs)

    history = {'train_loss': [], 'val_mse': [], 'auroc': [], 'aupr': []}
    if verbose: iterator = tqdm(iterator, desc="Training")

    for epoch in iterator:
        # Annealing L1 penalty
        if epoch <= config['l1_anneal_epochs'] and config['l1_anneal_epochs'] > 0:
            if epoch == config['l1_anneal_epochs']:
                patience_counter = 0
                best_val_mse = float('inf') 
                best_loss = float('inf')
                if verbose: print("\n[Info] L1 Annealing Complete. Resetting patience to find sparse solution...")
            curr_lambda = config['lambda_l1'] * (epoch / config['l1_anneal_epochs'])
        else:
            curr_lambda = config['lambda_l1']
            
        # Train Step
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            outputs = model(batch_x)
            pred_loss = criterion(outputs, batch_y)
            
            # --- UPDATED L1 LOGIC ---
            # Get the probability mask [N, N]
            mask_prob = model.get_adjacency_matrix()
            
            
            # Penalize only off-diagonals (Self-loops are free/forced)
            penalized_mask = mask_prob * (1.0 - model.self_mask)
            
            # Sum of probabilities in off-diagonal entries
            num_off = penalized_mask.numel() - torch.sum(model.self_mask).item()
            l1_loss = torch.sum(penalized_mask) / (num_off + 1e-5)
            
            loss = pred_loss + (curr_lambda * l1_loss)
            loss.backward()
            optimizer.step()
            
        # Validation Step
        # model.eval()
        # val_mse = 0
        # with torch.no_grad():
        #     for bx, by in val_loader:
        #         bx, by = bx.to(device), by.to(device)
        #         out = model(bx)
        #         val_mse += criterion(out, by).item()
        # val_mse /= len(val_loader)
        
        history['train_loss'].append(loss.item()) 
        # history['val_mse'].append(val_mse)
        curr_mask = model.get_adjacency_matrix().detach().cpu().numpy()
        mask0 = curr_mask.copy()
        if 'dream' in config['data_dir'].lower():
            np.fill_diagonal(mask0, 0)  # DREAM3 has no self-loops
        try:
            epoch_auroc = roc_auc_score(GC_true.flatten(), mask0.flatten())
        except ValueError:
            epoch_auroc = 0.5
        precision, recall, _ = precision_recall_curve(GC_true.flatten(), mask0.flatten())
        epoch_aupr = auc(recall, precision)
        history['auroc'].append(epoch_auroc)
        history['aupr'].append(epoch_aupr)

        mask_prob = model.get_adjacency_matrix()
        penalized_mask = mask_prob * (1.0 - model.self_mask)
        num_off = penalized_mask.numel() - torch.sum(model.self_mask).item()
        # l1_penalty = torch.sum(penalized_mask) / (num_off + 1e-5)

        # 2. Combine Val MSE + L1 Penalty
        # This is the "honest" metric: how well it predicts + how sparse it is
        # loss_new = val_mse + (curr_lambda * l1_penalty.item())
        
        # if loss_new < best_loss:
        #     best_loss = loss_new
        #     patience_counter = 0
        # else:
        #     patience_counter += 1
        #     if patience_counter >= patience:
        #         if verbose:
        #             print(f"Early stopping at epoch {epoch}. Best Val MSE: {best_val_mse:.6f}")
        #         break
        # if val_mse < best_val_mse:
        #     best_val_mse = val_mse

    # 4. Final Evaluation
    mask_prob = model.get_adjacency_matrix().detach().cpu().numpy()
    
    # No max across lags needed anymore, the mask is already N x N!
    final_adj = mask_prob.copy()
    if 'dream' in config['data_dir'].lower():
        np.fill_diagonal(final_adj, 0)  # DREAM3 has no self-loops
    
    aupr, best_shd, best_thresh = calculate_metrics_best(GC_true, final_adj)
    target_shd, target_thresh = calculate_metrics_at_target(GC_true, final_adj)
    
    try:
        auroc = roc_auc_score(GC_true.flatten(), final_adj.flatten())
    except ValueError:
        auroc = 0.5 
    if True:
        save_dir = 'results/matrices'
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.basename(config['data_dir'])

        data_set_name = base_name.replace('.npz', '')
        binary_path = os.path.join(save_dir, f"{data_set_name}_predicted_adj.npy")
        binary_matrix = (final_adj > best_thresh).astype(int)
        np.save(binary_path, binary_matrix)
        print(f"Saved predicted adjacency matrix to {binary_path}")
        
        unique_vals = np.unique(binary_matrix)
        print(f"DEBUG: Unique values in binary matrix: {unique_vals}")
        print(f"DEBUG: Best Threshold used: {best_thresh}")
    
    # print best mse, aupr, auroc in history and the respective epochs
    #best_mse_epoch = np.argmin(history['val_mse'])
    best_aupr_epoch = np.argmax(history['aupr'])
    best_auroc_epoch = np.argmax(history['auroc'])
    #best_mse = history['val_mse'][best_mse_epoch]
    best_aupr = history['aupr'][best_aupr_epoch]
    best_auroc = history['auroc'][best_auroc_epoch]
    #print(f"Best Val MSE: {best_mse:.6f} at epoch {best_mse_epoch}")
    print(f"Best AUROC: {best_auroc:.4f} at epoch {best_auroc_epoch}")
    print(f"Best AUPR: {best_aupr:.4f} at epoch {best_aupr_epoch}")
    print(f"final auroc: {auroc:.4f}, final aupr: {aupr:.4f}")


    return {
        #'val_mse': best_mse,
        'auroc': auroc,
        'aupr': aupr,
        'best_shd': best_shd,
        'target_shd': target_shd
    }
