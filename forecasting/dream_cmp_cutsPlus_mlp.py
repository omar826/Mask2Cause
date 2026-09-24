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
BASE_PATH = str(Path(__file__).resolve().parent / "data_for_forecasting/dream3")
DATA_DIR = os.path.join(BASE_PATH, "Dream3TensorData")
M2C_DIR = os.path.join(BASE_PATH, "mask2cause_matrices")
CUTS_DIR = os.path.join(BASE_PATH, "cuts_plus_matrices")

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
def load_dream3_ts(pt_path):
    try:
        data_dict = torch.load(pt_path, map_location='cpu', weights_only=False)
    except:
        data_dict = torch.load(pt_path, map_location='cpu')
    X = data_dict['TsData']
    return X.numpy() if torch.is_tensor(X) else X

def load_mask(directory, filename, is_m2c=True):
    if is_m2c:
        path = os.path.join(directory, f"{filename}_predicted_adj.npy")
    else:
        # Mapping "Size100Ecoli1.pt" -> "ecoli1.npy"
        clean_name = filename.replace("Size100", "").replace(".pt", "").lower()
        path = os.path.join(directory, f"{clean_name}.npy")
    
    if os.path.exists(path):
        adj = np.load(path)
        return (adj > 0.5).astype(int)
    return None

def create_windows(data, target_idx, pred_len):
    X, Y = [], []
    for t in range(len(data) - pred_len):
        X.append(torch.tensor(data[t], dtype=torch.float32))
        Y.append(torch.tensor(data[t + 1 : t + 1 + pred_len, target_idx], dtype=torch.float32))
    return torch.stack(X), torch.stack(Y)

def train_and_eval(X_tr, X_te, Y_tr, Y_te, in_dim):
    model = UnivariateMLP(in_dim, PRED_LEN)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        output = model(X_tr)
        loss = loss_fn(output, Y_tr)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        mse = loss_fn(model(X_te), Y_te).item()
    return mse, count_parameters(in_dim, PRED_LEN)

# ---------------- MAIN ----------------
def run_benchmark():
    dataset_results = []
    if not os.path.exists(DATA_DIR):
        print(f"Directory not found: {DATA_DIR}")
        return

    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".pt")])
    files = runtime.select_files(files)
    print(f"Starting DREAM3 MLP Benchmark (Base vs M2C vs Cuts+)...")

    for f in tqdm(files, desc="Overall Progress"):
        raw_ts = load_dream3_ts(os.path.join(DATA_DIR, f))
        adj_m2c = load_mask(M2C_DIR, f, is_m2c=True)
        adj_cuts = load_mask(CUTS_DIR, f, is_m2c=False)

        if raw_ts is None or adj_m2c is None or adj_cuts is None:
            continue

        num_vars = raw_ts.shape[1]
        split = int(len(raw_ts) * 0.8)
        scaler = StandardScaler().fit(raw_ts[:split])
        data = scaler.transform(raw_ts)
        
        train_data, test_data = data[:split], data[split:]
        mse_store = {"Base": [], "M2C": [], "Cuts": []}

        for i in tqdm(range(num_vars), desc=f"  Nodes in {f[:12]}", leave=False):
            Xtr_full, Ytr = create_windows(train_data, i, PRED_LEN)
            Xte_full, Yte = create_windows(test_data, i, PRED_LEN)

            # 1. BASE
            mse_b, _ = train_and_eval(Xtr_full, Xte_full, Ytr, Yte, num_vars)
            mse_store["Base"].append(mse_b)

            # 2. M2C (Gated + Diagonal)
            idx_m = sorted(list(set(np.where(adj_m2c[i, :] == 1)[0].tolist() + [i])))
            mse_m, _ = train_and_eval(Xtr_full[:, idx_m], Xte_full[:, idx_m], Ytr, Yte, len(idx_m))
            mse_store["M2C"].append(mse_m)

            # 3. CUTS+ (Gated + Diagonal)
            idx_c = sorted(list(set(np.where(adj_cuts[i, :] == 1)[0].tolist() + [i])))
            mse_s, _ = train_and_eval(Xtr_full[:, idx_c], Xte_full[:, idx_c], Ytr, Yte, len(idx_c))
            mse_store["Cuts"].append(mse_s)

        avg_b, avg_m, avg_c = np.mean(mse_store["Base"]), np.mean(mse_store["M2C"]), np.mean(mse_store["Cuts"])
        gain_m = (avg_b - avg_m) / avg_b * 100
        gain_c = (avg_b - avg_c) / avg_b * 100

        dataset_results.append({
            "Dataset": f,
            "Base_MSE": avg_b,
            "M2C_MSE": avg_m,
            "CUTS_MSE": avg_c,
            "M2C_Gain%": gain_m,
            "Cuts_Gain%": gain_c,
            "Winner": "M2C" if gain_m > gain_c else "Cuts+"
        })
        
        tqdm.write(f"{f[:12]:<12} | M2C Gain: {gain_m:>6.2f}% | Cuts Gain: {gain_c:>6.2f}%")

    print("\n" + "="*85)
    df = pd.DataFrame(dataset_results)
    runtime.save_results(df)
    print(df.to_string(index=False))
    print("="*85)
    print(f"Overall Avg M2C Gain:  {df['M2C_Gain%'].mean():.2f}%")
    print(f"Overall Avg Cuts Gain: {df['Cuts_Gain%'].mean():.2f}%")

if __name__ == "__main__":
    runtime.configure(__file__, globals())
    run_benchmark()
    runtime.ensure_results()
