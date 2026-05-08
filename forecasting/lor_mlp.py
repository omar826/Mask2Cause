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
NUM_VARS = 10  # Benchmarking first 10 variables

BASE_PATH = "data_for_forecasting/lor_data"
DATA_DIR = os.path.join(BASE_PATH, "lor_data")
M2C_DIR = os.path.join(BASE_PATH, "mask2cause_matrices")

# ---------------- MODEL ----------------
class UnivariateMLP(nn.Module):
    def __init__(self, input_dim, pred_len):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, pred_len)
        )

    def forward(self, x):
        return self.net(x)

def count_parameters(input_dim, pred_len):
    return (input_dim * 32 + 32) + (32 * pred_len + pred_len)

# ---------------- HELPERS ----------------
def create_windows(data, target_idx, pred_len):
    X, Y = [], []
    for t in range(len(data) - pred_len):
        X.append(torch.tensor(data[t], dtype=torch.float32))
        Y.append(torch.tensor(
            data[t + 1 : t + 1 + pred_len, target_idx],
            dtype=torch.float32
        ))
    return torch.stack(X), torch.stack(Y)

def train_and_eval(X_tr, X_te, Y_tr, Y_te, in_dim):
    model = UnivariateMLP(in_dim, PRED_LEN)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = loss_fn(model(X_tr), Y_tr)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        mse = loss_fn(model(X_te), Y_te).item()

    return mse, count_parameters(in_dim, PRED_LEN)

def load_m2c_matrix(folder, prefix):
    if not os.path.exists(folder):
        return None
    for f in os.listdir(folder):
        if f.startswith(prefix):
            path = os.path.join(folder, f)
            adj = np.load(path) if f.endswith(".npy") else pd.read_csv(path, header=None).values
            return (adj > 0.5).astype(int)
    return None

# ---------------- MAIN ----------------
def run_benchmark():
    dataset_results = []

    if not os.path.exists(DATA_DIR):
        print(f"Directory not found: {DATA_DIR}")
        return

    csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
    print(f"Starting Lorenz MLP Benchmark on {len(csv_files)} datasets...")

    for data_file in tqdm(csv_files, desc="Overall Progress"):
        prefix = data_file.replace("_data.csv", "")
        
        adj_m2c = load_m2c_matrix(M2C_DIR, prefix)
        if adj_m2c is None:
            continue

        df = pd.read_csv(os.path.join(DATA_DIR, data_file))
        raw = df.values[:, :NUM_VARS]

        split = int(len(raw) * 0.8)
        scaler = StandardScaler().fit(raw[:split])
        train_norm = scaler.transform(raw[:split])
        test_norm = scaler.transform(raw[split:])

        mse_store = {"Base": [], "M2C": []}
        param_store = {"Base": [], "M2C": []}

        # Run univariate forecasting for each variable
        for i in tqdm(range(NUM_VARS), desc=f"  Vars in {prefix[:12]}", leave=False):
            Xtr_full, Ytr = create_windows(train_norm, i, PRED_LEN)
            Xte_full, Yte = create_windows(test_norm, i, PRED_LEN)

            # --- 1. BASE (Dense: All 10 vars as input) ---
            mse_b, p_b = train_and_eval(Xtr_full, Xte_full, Ytr, Yte, NUM_VARS)
            mse_store["Base"].append(mse_b)
            param_store["Base"].append(p_b)

            # --- 2. M2C (Sparse: Gated inputs) ---
            idx = np.where(adj_m2c[:, i] == 1)[0]
            idx = [j for j in idx if j < NUM_VARS]
            # always include self history
            if i not in idx:
                idx.append(i)
            
            idx = sorted(idx)
            mse_c, p_c = train_and_eval(Xtr_full[:, idx], Xte_full[:, idx], Ytr, Yte, len(idx))
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
    print("\n" + "="*95)
    print(f"{'LORENZ MLP CAUSAL BENCHMARK SUMMARY':^95}")
    print("="*95)
    df = pd.DataFrame(dataset_results)
    print(df.to_string(index=False))
    print("-" * 95)
    print(f"Overall Average Gain         : {df['Gain_%'].mean():.2f}%")
    print(f"Overall Average Param Red    : {df['Param_Red_%'].mean():.2f}%")
    print("="*95)

if __name__ == "__main__":
    run_benchmark()