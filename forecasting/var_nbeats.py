from pathlib import Path
try:
    from . import artifact_runtime as runtime
except ImportError:
    import artifact_runtime as runtime
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from collections import defaultdict
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ---------------- SETTINGS ----------------
PRED_LEN = 1
EPOCHS = 20
NUM_VARS = 10
torch.manual_seed(42)

BASE_PATH = str(Path(__file__).resolve().parent / "data_for_forecasting/var_data")
DATA_DIR = os.path.join(BASE_PATH, "var_datasets")
M2C_DIR = os.path.join(BASE_PATH, "mask2cause_matrices")

# ---------------- N-BEATS MODEL ----------------
class NBeatsBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, theta_dim=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.theta = nn.Linear(hidden_dim, theta_dim)
        self.basis = nn.Linear(theta_dim, 1)

    def forward(self, x):
        return self.basis(self.theta(self.fc(x)))

class SimpleNBeats(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Stack of 3 blocks for signal decomposition
        self.blocks = nn.ModuleList([NBeatsBlock(input_dim) for _ in range(3)])

    def forward(self, x):
        out = 0
        for b in self.blocks:
            out += b(x)
        return out

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ---------------- HELPERS ----------------
def load_m2c_matrix(folder, prefix):
    """Load Mask2Cause adjacency matrix with thresholding."""
    if not os.path.exists(folder):
        return None
    for f in os.listdir(folder):
        if f.startswith(prefix):
            path = os.path.join(folder, f)
            adj = np.load(path) if f.endswith(".npy") else pd.read_csv(path, header=None).values
            return (adj > 0.5).astype(int)
    return None

def train_eval(X_tr, X_te, Y_tr, Y_te, in_dim):
    model = SimpleNBeats(in_dim)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = loss_fn(model(X_tr), Y_tr.view(-1, 1))
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        mse = loss_fn(model(X_te), Y_te.view(-1, 1)).item()

    return mse, count_params(model)

# ---------------- MAIN ----------------
def run_benchmark():
    dataset_results = []

    if not os.path.exists(DATA_DIR):
        print(f"Directory not found: {DATA_DIR}")
        return

    csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
    csv_files = runtime.select_files(csv_files)
    print(f"Starting VAR N-BEATS Benchmark on {len(csv_files)} datasets...")
    print("-" * 90)

    for data_file in tqdm(csv_files, desc="Overall Progress"):
        prefix = data_file.replace("_data.csv", "")
        
        adj_m2c = load_m2c_matrix(M2C_DIR, prefix)
        if adj_m2c is None:
            continue

        # Load Data - filter for columns with 'T' (time points)
        df = pd.read_csv(os.path.join(DATA_DIR, data_file))
        raw = df.filter(regex="T").values[:, :NUM_VARS]

        split = int(len(raw) * 0.8)
        scaler = StandardScaler().fit(raw[:split])
        data = torch.tensor(scaler.transform(raw), dtype=torch.float32)

        # Train/Test Split (t -> t+1)
        X_all, Y_all = data[:-1], data[1:]
        X_tr, Y_tr = X_all[:split-1], Y_all[:split-1]
        X_te, Y_te = X_all[split-1:], Y_all[split-1:]

        mse_store = {"Base": [], "M2C": []}
        param_store = {"Base": [], "M2C": []}

        # Node-by-node univariate forecasting
        for i in tqdm(range(NUM_VARS), desc=f"  Vars in {prefix[:12]}", leave=False):
            ytr, yte = Y_tr[:, i], Y_te[:, i]

            # --- 1. BASE (Dense: All 10 variables as input) ---
            mse_b, p_b = train_eval(X_tr, X_te, ytr, yte, NUM_VARS)
            mse_store["Base"].append(mse_b)
            param_store["Base"].append(p_b)

            # --- 2. M2C (Sparse: Gated inputs) ---
            idx = np.where(adj_m2c[:, i] == 1)[0].tolist()
            # Always include self history
            if i not in idx:
                idx.append(i)
            
            idx = sorted([j for j in idx if j < NUM_VARS])
            
            mse_c, p_c = train_eval(X_tr[:, idx], X_te[:, idx], ytr, yte, len(idx))
            mse_store["M2C"].append(mse_c)
            param_store["M2C"].append(p_c)

        # Dataset Aggregation
        avg_mse_b = np.mean(mse_store["Base"])
        avg_mse_m = np.mean(mse_store["M2C"])
        avg_p_b = np.sum(param_store["Base"])
        avg_p_m = np.sum(param_store["M2C"])

        gain = (avg_mse_b - avg_mse_m) / avg_mse_b * 100
        red = (avg_p_b - avg_p_m) / avg_p_b * 100

        dataset_results.append({
            "Dataset": prefix,
            "Base_MSE": avg_mse_b,
            "M2C_MSE": avg_mse_m,
            "Gain_%": gain,
            "Param_Red_%": red
        })
        
        tqdm.write(f"Done: {prefix:<25} | Gain: {gain:>7.2f}% | Param Red: {red:>6.1f}%")

    # ---------------- FINAL SUMMARY ----------------
    if dataset_results:
        print("\n" + "="*95)
        print(f"{'VAR N-BEATS CAUSAL BENCHMARK SUMMARY':^95}")
        print("="*95)
        df_final = pd.DataFrame(dataset_results)
        runtime.save_results(df_final)
        print(df_final.to_string(index=False))
        print("-" * 95)
        print(f"Overall Average Gain         : {df_final['Gain_%'].mean():.2f}%")
        print(f"Overall Average Param Red    : {df_final['Param_Red_%'].mean():.2f}%")
        print("="*95)

if __name__ == "__main__":
    runtime.configure(__file__, globals())
    run_benchmark()
    runtime.ensure_results()
