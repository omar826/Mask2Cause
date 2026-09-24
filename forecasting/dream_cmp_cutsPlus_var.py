from pathlib import Path
try:
    from . import artifact_runtime as runtime
except ImportError:
    import artifact_runtime as runtime
import pandas as pd
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import os
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------- SETTINGS ----------------
BASE_PATH = str(Path(__file__).resolve().parent / "data_for_forecasting/dream3")
DATA_DIR = os.path.join(BASE_PATH, "Dream3TensorData")
M2C_DIR = os.path.join(BASE_PATH, "mask2cause_matrices")
CUTS_DIR = os.path.join(BASE_PATH, "cuts_plus_matrices")

# ---------------- HELPERS ----------------
def count_linear_params(input_dim):
    # coefficients + intercept (bias)
    return input_dim + 1

def load_dream3_ts(pt_path):
    """Extract TsData from DREAM3 .pt dictionary."""
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
        # e.g., Size100Ecoli1.pt -> ecoli1.npy
        clean_name = filename.replace("Size100", "").replace(".pt", "").lower()
        path = os.path.join(directory, f"{clean_name}.npy")
    
    if os.path.exists(path):
        adj = np.load(path)
        return (adj > 0.5).astype(int)
    return None

# ---------------- MAIN ----------------
def run_dream_linear_benchmark():
    dataset_results = []

    if not os.path.exists(DATA_DIR):
        print(f"Directory not found: {DATA_DIR}")
        return

    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".pt")])
    files = runtime.select_files(files)
    print(f"Starting DREAM3 Linear Regression Benchmark (Base vs M2C vs Cuts+)...")

    for f in tqdm(files, desc="Overall Progress"):
        raw_ts = load_dream3_ts(os.path.join(DATA_DIR, f))
        adj_m2c = load_mask(M2C_DIR, f, is_m2c=True)
        adj_cuts = load_mask(CUTS_DIR, f, is_m2c=False)

        if raw_ts is None or adj_m2c is None or adj_cuts is None:
            continue

        num_vars = raw_ts.shape[1] 
        
        # Scaling and Splitting
        scaler = StandardScaler()
        data = scaler.fit_transform(raw_ts)

        X_all, Y_all = data[:-1], data[1:]
        split = int(len(X_all) * 0.8)
        
        X_tr, X_te = X_all[:split], X_all[split:]
        Y_tr, Y_te = Y_all[:split], Y_all[split:]

        mse_store = {"Base": [], "M2C": [], "Cuts": []}
        param_store = {"Base": [], "M2C": [], "Cuts": []}

        # Gene-by-gene univariate Linear Regression
        for i in tqdm(range(num_vars), desc=f"  Nodes in {f[:12]}", leave=False):
            ytr, yte = Y_tr[:, i], Y_te[:, i]

            # -------- 1. BASE (Dense) --------
            model_b = LinearRegression().fit(X_tr, ytr)
            mse_b = mean_squared_error(yte, model_b.predict(X_te))
            mse_store["Base"].append(mse_b)
            param_store["Base"].append(count_linear_params(num_vars))

            # -------- 2. M2C (Gated/Sparse) --------
            idx_m = sorted(list(set(np.where(adj_m2c[i, :] == 1)[0].tolist() + [i])))
            model_m = LinearRegression().fit(X_tr[:, idx_m], ytr)
            mse_m = mean_squared_error(yte, model_m.predict(X_te[:, idx_m]))
            mse_store["M2C"].append(mse_m)
            param_store["M2C"].append(count_linear_params(len(idx_m)))

            # -------- 3. CUTS+ (Gated/Sparse) --------
            idx_c = sorted(list(set(np.where(adj_cuts[i, :] == 1)[0].tolist() + [i])))
            model_c = LinearRegression().fit(X_tr[:, idx_c], ytr)
            mse_c = mean_squared_error(yte, model_c.predict(X_te[:, idx_c]))
            mse_store["Cuts"].append(mse_c)
            param_store["Cuts"].append(count_linear_params(len(idx_c)))

        # Aggregate Dataset Results
        avg_b = np.mean(mse_store["Base"])
        avg_m = np.mean(mse_store["M2C"])
        avg_c = np.mean(mse_store["Cuts"])

        gain_m = (avg_b - avg_m) / avg_b * 100
        gain_c = (avg_b - avg_c) / avg_b * 100
        
        # Calculate param reduction
        avg_p_b = np.mean(param_store["Base"])
        avg_p_c = np.mean(param_store["Cuts"])
        red_c = (avg_p_b - avg_p_c) / avg_p_b * 100

        dataset_results.append({
            "Dataset": f,
            "Base_MSE": avg_b,
            "M2C_MSE": avg_m,
            "CUTS_MSE": avg_c,
            "M2C_Gain%": gain_m,
            "Cuts_Gain%": gain_c,
            "Cuts_Red%": red_c,
            "Winner": "M2C" if gain_m > gain_c else "Cuts+"
        })
        
        tqdm.write(f"Done: {f[:12]:<12} | M2C Gain: {gain_m:>6.2f}% | Cuts Gain: {gain_c:>6.2f}%")

    # ---------------- FINAL SUMMARY ----------------
    print("\n" + "="*95)
    print(f"{'DREAM3 LINEAR REGRESSION BENCHMARK: M2C VS CUTS+':^95}")
    print("="*95)
    df = pd.DataFrame(dataset_results)
    runtime.save_results(df)
    print(df.to_string(index=False))
    print("-" * 95)
    print(f"Overall Avg M2C Gain   : {df['M2C_Gain%'].mean():.2f}%")
    print(f"Overall Avg Cuts Gain  : {df['Cuts_Gain%'].mean():.2f}%")
    print(f"Overall Avg Cuts Red   : {df['Cuts_Red%'].mean():.2f}%")
    print("="*95)

if __name__ == "__main__":
    runtime.configure(__file__, globals())
    run_dream_linear_benchmark()
    runtime.ensure_results()
