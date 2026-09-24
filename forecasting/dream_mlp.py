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

# ---------------- MODEL ----------------
class UnivariateMLP(nn.Module):
    def __init__(self, input_dim, pred_len):
        super().__init__()
        # input_dim will vary based on number of causal parents
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, pred_len)
        )

    def forward(self, x):
        return self.net(x)

def count_parameters(input_dim, pred_len):
    # MLP params: (in * 32 + 32) + (32 * out + out)
    return (input_dim * 32 + 32) + (32 * pred_len + pred_len)

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

    # Simple training loop
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
    print(f"Starting DREAM3 MLP Benchmark on {len(files)} datasets...")

    for f in tqdm(files, desc="Overall Progress"):
        raw_ts = load_dream3_ts(os.path.join(DATA_DIR, f))
        adj_m2c = load_m2c_matrix(f)

        if raw_ts is None or adj_m2c is None:
            continue

        num_vars = raw_ts.shape[1]
        
        # Preprocessing
        split = int(len(raw_ts) * 0.8)
        scaler = StandardScaler().fit(raw_ts[:split])
        data = scaler.transform(raw_ts)
        
        train_data = data[:split]
        test_data = data[split:]

        mse_store = {"Base": [], "M2C": []}
        param_store = {"Base": [], "M2C": []}

        # Gene-by-gene univariate prediction
        for i in tqdm(range(num_vars), desc=f"  Nodes in {f[:12]}", leave=False):
            Xtr_full, Ytr = create_windows(train_data, i, PRED_LEN)
            Xte_full, Yte = create_windows(test_data, i, PRED_LEN)

            # -------- BASE (Dense: All variables as input) --------
            mse_b, p_b = train_and_eval(Xtr_full, Xte_full, Ytr, Yte, num_vars)
            mse_store["Base"].append(mse_b)
            param_store["Base"].append(p_b)

            # -------- M2C (Sparse: Gated inputs) --------
            # Predicted parents from mask
            parents = np.where(adj_m2c[i, :] == 1)[0].tolist()
            # Always include self
            if i not in parents:
                parents.append(i)
            
            idx = sorted(parents)
            
            mse_c, p_c = train_and_eval(
                Xtr_full[:, idx],
                Xte_full[:, idx],
                Ytr, Yte,
                len(idx)
            )
            mse_store["M2C"].append(mse_c)
            param_store["M2C"].append(p_c)

        # Aggregate Dataset Results
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
    print(f"{'DREAM3 MLP CAUSAL BENCHMARK SUMMARY':^85}")
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
