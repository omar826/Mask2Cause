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
EPOCHS = 20
torch.manual_seed(42)

BASE_PATH = str(Path(__file__).resolve().parent / "data_for_forecasting/dream3")
DATA_DIR = os.path.join(BASE_PATH, "Dream3TensorData")
M2C_DIR = os.path.join(BASE_PATH, "mask2cause_matrices")

# ---------------- N-BEATS MODEL ----------------
class NBeatsBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.basis = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.basis(self.fc(x))

class SimpleNBeats(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # 3 blocks for a basic stack
        self.blocks = nn.ModuleList([NBeatsBlock(input_dim) for _ in range(3)])

    def forward(self, x):
        # Basis expansion and summation
        return sum(b(x) for b in self.blocks)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ---------------- HELPERS ----------------
def load_dream3_ts(pt_path):
    """Extract TsData from DREAM3 .pt dictionary."""
    try:
        data_dict = torch.load(pt_path, map_location='cpu', weights_only=False)
    except:
        data_dict = torch.load(pt_path, map_location='cpu')
    X = data_dict['TsData']
    return X.numpy() if torch.is_tensor(X) else X

def load_m2c_matrix(data_file):
    path = os.path.join(M2C_DIR, f"{data_file}_predicted_adj.npy")
    if os.path.exists(path):
        adj = np.load(path)
        return (adj > 0.5).astype(int)
    return None

def train_eval(X_tr, X_te, Y_tr, Y_te, in_dim):
    model = SimpleNBeats(in_dim)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    crit = nn.MSELoss()

    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = crit(model(X_tr), Y_tr.view(-1, 1))
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        mse = crit(model(X_te), Y_te.view(-1, 1)).item()

    return mse, count_params(model)

# ---------------- MAIN ----------------
def run_benchmark():
    dataset_results = []

    if not os.path.exists(DATA_DIR):
        print(f"Directory not found: {DATA_DIR}")
        return

    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".pt")])
    files = runtime.select_files(files)
    print(f"Starting DREAM3 N-BEATS Benchmark on {len(files)} datasets...")

    for f in tqdm(files, desc="Overall Progress"):
        raw_ts = load_dream3_ts(os.path.join(DATA_DIR, f))
        adj_m2c = load_m2c_matrix(f)

        if raw_ts is None or adj_m2c is None:
            continue

        num_vars = raw_ts.shape[1]
        
        # Scaling and Splitting
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(raw_ts)
        data = torch.tensor(data_scaled, dtype=torch.float32)

        X_all, Y_all = data[:-1], data[1:]
        split = int(len(X_all) * 0.8)
        
        X_tr, X_te = X_all[:split], X_all[split:]
        Y_tr, Y_te = Y_all[:split], Y_all[split:]

        mse_store = {"Base": [], "M2C": []}
        param_store = {"Base": [], "M2C": []}

        # Node-by-node (Gene-by-gene) univariate forecasting
        for i in tqdm(range(num_vars), desc=f"  Nodes in {f[:12]}", leave=False):
            ytr, yte = Y_tr[:, i], Y_te[:, i]

            # -------- BASE (Dense N-BEATS) --------
            # Input dim = 100 (Access to all variables)
            mse_b, p_b = train_eval(X_tr, X_te, ytr, yte, num_vars)
            mse_store["Base"].append(mse_b)
            param_store["Base"].append(p_b)

            # -------- M2C (Gated N-BEATS) --------
            # Trivial access to self (i) + gated access to parents
            parents = np.where(adj_m2c[i, :] == 1)[0].tolist()
            if i not in parents:
                parents.append(i)
            
            idx = sorted(parents)
            
            mse_c, p_c = train_eval(
                X_tr[:, idx],
                X_te[:, idx],
                ytr, yte,
                len(idx)
            )
            mse_store["M2C"].append(mse_c)
            param_store["M2C"].append(p_c)

        # Dataset Aggregation
        avg_mse_b = np.mean(mse_store["Base"])
        avg_mse_m = np.mean(mse_store["M2C"])
        avg_p_b = np.mean(param_store["Base"])
        avg_p_m = np.mean(param_store["M2C"])

        gain = (avg_mse_b - avg_mse_m) / avg_mse_b * 100
        red = (avg_p_b - avg_p_m) / avg_p_b * 100

        dataset_results.append({
            "Dataset": f,
            "Base_MSE": avg_mse_b,
            "M2C_MSE": avg_mse_m,
            "Gain_%": gain,
            "Param_Red_%": red
        })
        
        tqdm.write(f"Done: {f:<20} | Gain: {gain:>7.2f}% | Param Red: {red:>6.1f}%")

    # ---------------- FINAL SUMMARY ----------------
    print("\n" + "="*85)
    print(f"{'DREAM3 N-BEATS CAUSAL BENCHMARK SUMMARY':^85}")
    print("="*85)
    df = pd.DataFrame(dataset_results)
    runtime.save_results(df)
    print(df.to_string(index=False))
    print("-" * 85)
    print(f"Overall Average Gain         : {df['Gain_%'].mean():.2f}%")
    print(f"Overall Average Param Red    : {df['Param_Red_%'].mean():.2f}%")
    print("="*85)

if __name__ == "__main__":
    runtime.configure(__file__, globals())
    run_benchmark()
    runtime.ensure_results()
