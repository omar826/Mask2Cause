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

try:
    import torch_directml
    HAS_DIRECTML = True
except ImportError:
    HAS_DIRECTML = False

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
        self.enc_embedding = DataEmbedding_Inverted(self.seq_len, self.d_model)
        self.adj_logits = nn.Parameter(torch.ones(self.num_variates, self.num_variates) * 1.0)
        self.register_buffer('self_mask', torch.eye(self.num_variates))
        self.encoder = nn.ModuleList([
            EncoderLayer(self.d_model, self.n_heads) for _ in range(self.e_layers)
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

def load_heteroscedastic_data(path, seq_len, batch_size):
    """
    Loads and windows heteroscedastic time series data from a .npz file, applying standard normalization.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    data = np.load(path)
    if 'X_np' in data.files: X = data['X_np']
    elif 'X' in data.files: X = data['X']
    elif 'data' in data.files: X = data['data']
    else: raise ValueError(f"Could not find data key. Available: {data.files}")
    if 'Gref' in data.files: GC = data['Gref']
    elif 'GC' in data.files: GC = data['GC']
    elif 'true_graph' in data.files: GC = data['true_graph']
    else: raise ValueError("Could not find ground truth graph.")
    num_vars = GC.shape[0]
    if X.ndim == 2:
        if X.shape[0] == num_vars and X.shape[1] != num_vars: X = X.T
    elif X.ndim == 3:
        if X.shape[1] == num_vars and X.shape[2] != num_vars: X = X.transpose(0, 2, 1)
    X = X.astype(np.float32)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-5
    X = (X - mean) / std
    samples_X, samples_Y = [], []
    if X.ndim == 2:
        for i in range(len(X) - seq_len - 1):
            samples_X.append(X[i:i+seq_len])
            samples_Y.append(X[i+seq_len])
    elif X.ndim == 3:
        num_samples, time_steps, _ = X.shape
        for s in range(num_samples):
            for i in range(time_steps - seq_len - 1):
                samples_X.append(X[s, i : i+seq_len])
                samples_Y.append(X[s, i+seq_len])
    samples_X = torch.tensor(np.array(samples_X))
    samples_Y = torch.tensor(np.array(samples_Y))
    dataset = TensorDataset(samples_X, samples_Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, GC

def load_raw_traffic_data(data_dir, seq_len, batch_size):
    """
    Loads raw traffic data from multiple .npy files and performs sample-aware sliding window slicing.
    """
    data_path = os.path.join(data_dir, 'gen_data.npy')
    graph_path = os.path.join(data_dir, 'graph.npy')
    if not os.path.exists(data_path) or not os.path.exists(graph_path):
        raise FileNotFoundError(f"Missing .npy files in {data_dir}")
    raw_data = np.load(data_path).astype(np.float32) 
    GC = np.load(graph_path)
    num_vars_data = raw_data.shape[2]
    num_vars_graph = GC.shape[0]
    if num_vars_data < num_vars_graph:
        raise ValueError(f"Data has {num_vars_data} vars but graph has {num_vars_graph}!")
    mean = np.mean(raw_data, axis=(0, 1))
    std = np.std(raw_data, axis=(0, 1)) + 1e-5
    X_norm = (raw_data - mean) / std
    samples_X, samples_Y = [], []
    num_samples, time_steps, _ = X_norm.shape
    windows_per_sample = time_steps - seq_len
    if windows_per_sample < 1:
        raise ValueError(f"Sequence length ({seq_len}) too large for time dimension ({time_steps}).")
    for s_idx in range(num_samples):
        sample_data = X_norm[s_idx]
        for t in range(windows_per_sample):
            samples_X.append(sample_data[t : t+seq_len])
            samples_Y.append(sample_data[t+seq_len])
    samples_X = torch.tensor(np.array(samples_X))
    samples_Y = torch.tensor(np.array(samples_Y))
    dataset = TensorDataset(samples_X, samples_Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, GC

def load_dream3_data(pt_path, seq_len, batch_size):
    """
    Loads DREAM3 gene network data from .pt files and parses associated .tsv ground truth graphs.
    """
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Data file not found: {pt_path}")
    try: data_dict = torch.load(pt_path, weights_only=False) 
    except TypeError: data_dict = torch.load(pt_path)
    if 'TsData' not in data_dict:
        raise ValueError(f"Expected key 'TsData' in .pt file, found: {data_dict.keys()}")
    X = data_dict['TsData']
    mean = torch.mean(X, dim=0)
    std = torch.std(X, dim=0) + 1e-5
    X = (X - mean) / std
    parent_dir = os.path.dirname(os.path.dirname(pt_path))
    filename = os.path.basename(pt_path)
    match = re.search(r'(Ecoli|Yeast)[0-9]+', filename, re.IGNORECASE)
    if not match: raise ValueError(f"Could not parse dataset ID (Ecoli/Yeast) from {filename}")
    dataset_id = match.group(0)
    size_str = "Size100" if "Size100" in filename else "Size10"
    graph_filename = f"InSilico{size_str}-{dataset_id}.tsv"
    graph_path = os.path.join(parent_dir, 'TrueGeneNetworks', graph_filename)
    if not os.path.exists(graph_path):
        graph_dir = os.path.join(parent_dir, 'TrueGeneNetworks')
        if os.path.exists(graph_dir):
            for f in os.listdir(graph_dir):
                if dataset_id.lower() in f.lower() and size_str.lower() in f.lower() and f.endswith('.tsv'):
                    graph_path = os.path.join(graph_dir, f)
                    break
    if not os.path.exists(graph_path): raise FileNotFoundError(f"Could not locate graph file at {graph_path}")
    num_nodes = X.shape[1]
    GC_true = np.zeros((num_nodes, num_nodes))
    with open(graph_path, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                src_idx = int(row[0][1:]) - 1
                tgt_idx = int(row[1][1:]) - 1
                GC_true[tgt_idx, src_idx] = 1
    samples_X, samples_Y = [], []
    num_time_steps = X.shape[0]
    for t in range(num_time_steps - seq_len):
        samples_X.append(X[t : t+seq_len])
        samples_Y.append(X[t+seq_len])
    samples_X = torch.stack(samples_X)
    samples_Y = torch.stack(samples_Y)
    dataset = TensorDataset(samples_X, samples_Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, GC_true

def load_lorenz_data(path, seq_len, batch_size):
    """
    Loads and prepares the Lorenz system chaotic time series data.
    """
    if not os.path.exists(path): raise FileNotFoundError(f"Data file not found: {path}")
    data = np.load(path)
    if 'X_np' in data.files: X = data['X_np']
    elif 'X' in data.files: X = data['X']
    elif 'data' in data.files: X = data['data']
    else: raise ValueError(f"Could not find data key. Available: {data.files}")
    if 'Gref' in data.files: GC = data['Gref']
    elif 'GC' in data.files: GC = data['GC']
    elif 'true_graph' in data.files: GC = data['true_graph']
    else: raise ValueError("Could not find ground truth graph.")
    num_vars = GC.shape[0]
    if X.ndim == 2:
        if X.shape[0] == num_vars and X.shape[1] != num_vars: X = X.T
    elif X.ndim == 3:
        if X.shape[1] == num_vars and X.shape[2] != num_vars: X = X.transpose(0, 2, 1)
    X = X.astype(np.float32)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-5
    X = (X - mean) / std
    samples_X, samples_Y = [], []
    if X.ndim == 2:
        for i in range(len(X) - seq_len - 1):
            samples_X.append(X[i:i+seq_len])
            samples_Y.append(X[i+seq_len])
    elif X.ndim == 3:
        num_samples, time_steps, _ = X.shape
        for s in range(num_samples):
            for i in range(time_steps - seq_len - 1):
                samples_X.append(X[s, i : i+seq_len])
                samples_Y.append(X[s, i+seq_len])
    samples_X = torch.tensor(np.array(samples_X))
    samples_Y = torch.tensor(np.array(samples_Y))
    dataset = TensorDataset(samples_X, samples_Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, GC

def load_var_data(path, seq_len, batch_size):
    """
    Loads and prepares Vector Autoregression (VAR) generated data.
    """
    if not os.path.exists(path): raise FileNotFoundError(f"Data file not found: {path}")
    data = np.load(path)
    if 'X_np' in data.files: X = data['X_np']
    elif 'X' in data.files: X = data['X']
    elif 'data' in data.files: X = data['data']
    else: raise ValueError(f"Could not find data key. Available: {data.files}")
    if 'Gref' in data.files: GC = data['Gref']
    elif 'GC' in data.files: GC = data['GC']
    elif 'true_graph' in data.files: GC = data['true_graph']
    else: raise ValueError("Could not find ground truth graph.")
    num_vars = GC.shape[0]
    if X.ndim == 2:
        if X.shape[0] == num_vars and X.shape[1] != num_vars: X = X.T
    elif X.ndim == 3:
        if X.shape[1] == num_vars and X.shape[2] != num_vars: X = X.transpose(0, 2, 1)
    X = X.astype(np.float32)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-5
    X = (X - mean) / std
    samples_X, samples_Y = [], []
    if X.ndim == 2:
        for i in range(len(X) - seq_len - 1):
            samples_X.append(X[i:i+seq_len])
            samples_Y.append(X[i+seq_len])
    elif X.ndim == 3:
        num_samples, time_steps, _ = X.shape
        for s in range(num_samples):
            for i in range(time_steps - seq_len - 1):
                samples_X.append(X[s, i : i+seq_len])
                samples_Y.append(X[s, i+seq_len])
    samples_X = torch.tensor(np.array(samples_X))
    samples_Y = torch.tensor(np.array(samples_Y))
    dataset = TensorDataset(samples_X, samples_Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, GC

def find_threshold_for_density(mask, target_density):
    """
    Finds the specific weight threshold required to achieve a target graph density.
    """
    weights = mask.flatten()
    weights = np.sort(weights)[::-1] 
    k = int(len(weights) * target_density)
    k = max(1, min(k, len(weights)-1))
    return weights[k]

def calculate_metrics_at_target(ground_truth, learned_prob):
    """
    Calculates Structural Hamming Distance and threshold for a prediction at the true graph's density.
    """
    gt_density = np.mean(ground_truth)
    target_thresh = find_threshold_for_density(learned_prob, gt_density)
    pred_adj = (learned_prob > target_thresh).astype(int)
    diff = np.abs(ground_truth - pred_adj)
    shd = np.sum(diff)
    return shd, target_thresh

def calculate_metrics_best(ground_truth, learned_prob):
    """
    Calculates Area Under Precision-Recall Curve and the best SHD across a range of thresholds.
    """
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

def train_and_validate(config, device, verbose=False):
    """
    Executes the full training loop with L1 annealing and evaluates the learned causal graph against ground truth.
    """
    if 'lorenz' in config['data_dir'].lower():
        train_loader, GC_true = load_lorenz_data(config['data_dir'], config['seq_len'], config['batch_size'])
    elif 'var' in config['data_dir'].lower():
        train_loader, GC_true = load_var_data(config['data_dir'], config['seq_len'], config['batch_size'])
    elif 'dream' in config['data_dir'].lower():
        train_loader, GC_true = load_dream3_data(config['data_dir'], config['seq_len'], config['batch_size'])
    elif 'heteroscedastic' in config['data_dir'].lower():
        train_loader, GC_true = load_heteroscedastic_data(config['data_dir'], config['seq_len'], config['batch_size'])
    else:
        train_loader, GC_true = load_raw_traffic_data(config['data_dir'], config['seq_len'], config['batch_size'])
    class Args: pass 
    args = Args()
    sample_x, _ = train_loader.dataset[0]
    args.enc_in = sample_x.shape[-1]
    args.seq_len = config['seq_len']
    args.d_model = config.get('d_model', 64)
    args.n_heads = config.get('n_heads', 4)
    args.e_layers = config.get('e_layers', 2)
    args.diagonal_force = config.get('diagonal_force', 100.0)
    model = CausalGraphTransformer(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.GaussianNLLLoss()
    max_epochs = 5000
    early_stop = config.get('early_stop', False)
    if not early_stop: max_epochs = config['epochs']
    iterator = range(max_epochs)
    history = {'train_loss': [], 'auroc': [], 'aupr': []}
    if verbose: iterator = tqdm(iterator, desc="Training")
    for epoch in iterator:
        if epoch <= config['l1_anneal_epochs'] and config['l1_anneal_epochs'] > 0:
            curr_lambda = config['lambda_l1'] * (epoch / config['l1_anneal_epochs'])
        else:
            curr_lambda = config['lambda_l1']
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            mu, sigma_sq = model(batch_x)
            pred_loss = criterion(mu, batch_y, sigma_sq)
            mask_prob = model.get_adjacency_matrix()
            penalized_mask = mask_prob * (1.0 - model.self_mask)
            num_off = penalized_mask.numel() - torch.sum(model.self_mask).item()
            l1_loss = torch.sum(penalized_mask) / (num_off + 1e-5)
            loss = pred_loss + (curr_lambda * l1_loss)
            loss.backward()
            optimizer.step()
        history['train_loss'].append(loss.item()) 
        curr_mask = model.get_adjacency_matrix().detach().cpu().numpy()
        mask0 = curr_mask.copy()
        if 'dream' in config['data_dir'].lower(): np.fill_diagonal(mask0, 0)
        if any(x in config['data_dir'].lower() for x in ['traffic', 'medical', 'pm25']):
            if mask0.shape[0] > GC_true.shape[0]: mask0 = mask0[:GC_true.shape[0], :GC_true.shape[0]]
        try: epoch_auroc = roc_auc_score(GC_true.flatten(), mask0.flatten())
        except ValueError: epoch_auroc = 0.5
        precision, recall, _ = precision_recall_curve(GC_true.flatten(), mask0.flatten())
        epoch_aupr = auc(recall, precision)
        history['auroc'].append(epoch_auroc)
        history['aupr'].append(epoch_aupr)
    mask_prob = model.get_adjacency_matrix().detach().cpu().numpy()
    final_adj = mask_prob.copy()
    if 'dream' in config['data_dir'].lower(): np.fill_diagonal(final_adj, 0)
    if any(x in config['data_dir'].lower() for x in ['traffic', 'medical', 'pm25']):
        if final_adj.shape[0] > GC_true.shape[0]: final_adj = final_adj[:GC_true.shape[0], :GC_true.shape[0]]
    aupr, best_shd, best_thresh = calculate_metrics_best(GC_true, final_adj)
    target_shd, target_thresh = calculate_metrics_at_target(GC_true, final_adj)
    try: auroc = roc_auc_score(GC_true.flatten(), final_adj.flatten())
    except ValueError: auroc = 0.5 
    save_dir = 'results/matrices'
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.basename(config['data_dir'])

    data_set_name = base_name.replace('.npz', '')
    binary_path = os.path.join(save_dir, f"{data_set_name}_predicted_adj.npy")
    binary_matrix = (final_adj > best_thresh).astype(int)
    np.save(binary_path, binary_matrix)
    print(f"Final Results - AUROC: {auroc:.4f}| AUPR: {aupr:.4f} | SHD: {target_shd}")
    return {
        'auroc': auroc,
        'aupr': aupr,
        'best_shd': best_shd,
        'target_shd': target_shd
    }