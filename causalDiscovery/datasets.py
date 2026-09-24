"""Original benchmark loaders; RHINO loader ported from the supplied artifact.
DREAM3 retains the original flattened-series windowing.
"""
import os, re, csv
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

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
    size_str = re.search(r"Size(10|50|100)(?=Ecoli|Yeast)", filename).group(0)
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

def load_rhino_arrays(path, lag):
    """Return normalized trajectory-safe windows and target/source lag-union graph."""
    from pathlib import Path
    p = Path(path)
    if p.is_dir():
        rows = np.loadtxt(p / 'train.csv', delimiter=',')
        raw = np.stack([rows[rows[:, 0] == i, 1:] for i in np.unique(rows[:, 0])])
        graph = np.load(p / 'adj_matrix.npy', allow_pickle=False)
    else:
        with np.load(p, allow_pickle=False) as data:
            raw, graph = data['train'], data['adj_matrix']
    raw = raw.astype(np.float32)
    if raw.ndim != 3 or not np.isfinite(raw).all():
        raise ValueError('Expected finite [trajectory,time,variable] training data')
    n = raw.shape[-1]
    if graph.ndim != 3 or graph.shape[1:] != (n,n) or graph[0].any():
        raise ValueError('Expected lagged-only RHINO adjacency [lag,source,target]')
    if not 1 <= lag < raw.shape[1]:
        raise ValueError('Input lag must be positive and shorter than a trajectory')
    mean, std = raw.mean(axis=(0,1)), raw.std(axis=(0,1)) + 1e-5
    normalized = (raw-mean)/std
    windows = np.stack([series[t:t+lag+1] for series in normalized for t in range(len(series)-lag)])
    truth = (graph[1:].max(axis=0).T != 0).astype(int)
    return windows, truth, mean, std
