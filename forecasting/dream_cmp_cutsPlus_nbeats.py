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

BASE_PATH = "data_for_forecasting/dream3"
DATA_DIR = os.path.join(BASE_PATH, "Dream3TensorData")
M2C_DIR = os.path.join(BASE_PATH, "mask2cause_matrices")
CUTS_DIR = os.path.join(BASE_PATH, "cuts_plus_matrices")

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
        return sum(b(x) for b in self.blocks)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ---------------- HELPERS ----------------
def load_dream3_ts(pt_path):
    try:
        data_dict = torch.load(pt_path, map_location='cpu', weights_only=False)
    except:
        data_dict = torch.load(pt_path, map_location='cpu')
    X = data_dict['TsData']
    return X.numpy() if torch.is_tensor(X) else X

def load_mask(directory, filename, is_m2c=True):
    if is_m2c:
        # e.g., Size100Ecoli1.pt -> Size100Ecoli1.pt_predicted_adj.npy
        path = os.path.join(directory, f"{filename}_predicted_adj.npy")
    else:
        # e.g., Size100Ecoli1.pt -> ecoli1.npy (Lowercase/cleaned)
        clean_name = filename.replace("Size100", "").replace(".pt", "").lower()
        path = os.path.join(directory, f"{clean_name}.npy")
    
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
    print(f"Starting DREAM3 N-BEATS Benchmark (Base vs M2C vs Cuts+)...")

    for f in tqdm(files, desc="Overall Progress"):
        raw_ts = load_dream3_ts(os.path.join(DATA_DIR, f))
        adj_m2c = load_mask(M2C_DIR, f, is_m2c=True)
        adj_cuts = load_mask(CUTS_DIR, f, is_m2c=False)

        if raw_ts is None or adj_m2c is None or adj_cuts is None:
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

        mse_store = {"Base": [], "M2C": [], "Cuts": []}
        param_store = {"Base": [], "M2C": [], "Cuts": []}

        for i in tqdm(range(num_vars), desc=f"  Nodes in {f[:12]}", leave=False):
            ytr, yte = Y_tr[:, i], Y_te[:, i]

            # 1. BASE (Full Dense)
            mse_b, p_b = train_eval(X_tr, X_te, ytr, yte, num_vars)
            mse_store["Base"].append(mse_b)
            param_store["Base"].append(p_b)

            # 2. M2C (Gated + Diagonal)
            idx_m = sorted(list(set(np.where(adj_m2c[i, :] == 1)[0].tolist() + [i])))
            mse_m, p_m = train_eval(X_tr[:, idx_m], X_te[:, idx_m], ytr, yte, len(idx_m))
            mse_store["M2C"].append(mse_m)
            param_store["M2C"].append(p_m)

            # 3. CUTS+ (Gated + Diagonal)
            idx_c = sorted(list(set(np.where(adj_cuts[i, :] == 1)[0].tolist() + [i])))
            mse_s, p_s = train_eval(X_tr[:, idx_c], X_te[:, idx_c], ytr, yte, len(idx_c))
            mse_store["Cuts"].append(mse_s)
            param_store["Cuts"].append(p_s)

        # Dataset Aggregation
        avg_b, avg_m, avg_c = np.mean(mse_store["Base"]), np.mean(mse_store["M2C"]), np.mean(mse_store["Cuts"])
        avg_p_b, avg_p_m, avg_p_c = np.mean(param_store["Base"]), np.mean(param_store["M2C"]), np.mean(param_store["Cuts"])

        gain_m = (avg_b - avg_m) / avg_b * 100
        gain_c = (avg_b - avg_c) / avg_b * 100
        red_c = (avg_p_b - avg_p_c) / avg_p_b * 100

        dataset_results.append({
            "Dataset": f[:15],
            "M2C_Gain%": gain_m,
            "Cuts_Gain%": gain_c,
            "Cuts_Param_Red%": red_c,
            "Winner": "M2C" if gain_m > gain_c else "Cuts+"
        })
        
        tqdm.write(f"{f[:12]:<12} | M2C Gain: {gain_m:>6.2f}% | Cuts Gain: {gain_c:>6.2f}%")

    print("\n" + "="*85)
    print(f"{'DREAM3 N-BEATS BENCHMARK: M2C VS CUTS+':^85}")
    print("="*85)
    df = pd.DataFrame(dataset_results)
    print(df.to_string(index=False))
    print("-" * 85)
    print(f"Overall Avg M2C Gain:  {df['M2C_Gain%'].mean():.2f}%")
    print(f"Overall Avg Cuts Gain: {df['Cuts_Gain%'].mean():.2f}%")
    print("="*85)

if __name__ == "__main__":
    run_benchmark()