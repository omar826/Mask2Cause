from pathlib import Path
try:
    from . import artifact_runtime as runtime
except ImportError:
    import artifact_runtime as runtime
import pandas as pd
import numpy as np
import torch
import os
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------- SETTINGS ----------------
BASE_PATH = str(Path(__file__).resolve().parent / "data_for_forecasting/dream3")
DATA_DIR = os.path.join(BASE_PATH, "Dream3TensorData")
M2C_DIR = os.path.join(BASE_PATH, "mask2cause_matrices")
CUTS_DIR = os.path.join(BASE_PATH, "cuts_plus_matrices")

# ---------------- HELPERS ----------------
def load_dream3_ts_only(pt_path):
    try:
        data_dict = torch.load(pt_path, map_location='cpu', weights_only=False)
    except Exception:
        data_dict = torch.load(pt_path, map_location='cpu')
    X = data_dict['TsData']
    return X.numpy() if torch.is_tensor(X) else X

def load_mask(directory, filename, is_m2c=True):
    if is_m2c:
        path = os.path.join(directory, f"{filename}_predicted_adj.npy")
    else:
        clean_name = filename.replace("Size100", "").replace(".pt", "").lower()
        path = os.path.join(directory, f"{clean_name}.npy")
    
    if os.path.exists(path):
        adj = np.load(path)
        return (adj > 0.5).astype(int)
    return None

# ---------------- MAIN ----------------
def run_benchmark():
    results = []
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} not found.")
        return

    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".pt")])
    files = runtime.select_files(files)
    print(f"Starting ARIMAX Benchmark: Base vs M2C vs Cuts+")
    print("-" * 85)

    for f in tqdm(files, desc="Overall Progress"):
        raw_ts = load_dream3_ts_only(os.path.join(DATA_DIR, f))
        adj_m2c = load_mask(M2C_DIR, f, is_m2c=True)
        adj_cuts = load_mask(CUTS_DIR, f, is_m2c=False)

        if raw_ts is None or adj_m2c is None or adj_cuts is None:
            continue

        num_vars = raw_ts.shape[1]
        split = int(len(raw_ts) * 0.8)
        scaler = StandardScaler().fit(raw_ts[:split])
        data = scaler.transform(raw_ts)

        X_all, Y_all = data[:-1], data[1:]
        X_tr, Y_tr = X_all[:split-1], Y_all[:split-1]
        X_te, Y_te = X_all[split-1:], Y_all[split-1:]

        mse_base, mse_m2c, mse_cuts = [], [], []

        # Per-node progress printing
        for i in tqdm(range(num_vars), desc=f"  Nodes in {f[:12]}", leave=False):
            y_tr, y_te = Y_tr[:, i], Y_te[:, i]

            # 1. BASE
            idx_b = list(range(num_vars))
            m_base = ARIMA(y_tr, exog=X_tr[:, idx_b], order=(1, 0, 0)).fit()
            p_base = m_base.apply(y_te, exog=X_te[:, idx_b]).fittedvalues
            mse_base.append(mean_squared_error(y_te, p_base))

            # 2. M2C (Diagonal Enforced)
            idx_m = sorted(list(set(np.where(adj_m2c[i, :] == 1)[0].tolist() + [i])))
            m_m2c = ARIMA(y_tr, exog=X_tr[:, idx_m], order=(1, 0, 0)).fit()
            p_m2c = m_m2c.apply(y_te, exog=X_te[:, idx_m]).fittedvalues
            mse_m2c.append(mean_squared_error(y_te, p_m2c))

            # 3. CUTS+ (Diagonal Enforced)
            idx_c = sorted(list(set(np.where(adj_cuts[i, :] == 1)[0].tolist() + [i])))
            m_cuts = ARIMA(y_tr, exog=X_tr[:, idx_c], order=(1, 0, 0)).fit()
            p_cuts = m_cuts.apply(y_te, exog=X_te[:, idx_c]).fittedvalues
            mse_cuts.append(mean_squared_error(y_te, p_cuts))

        avg_b, avg_m, avg_c = np.nanmean(mse_base), np.nanmean(mse_m2c), np.nanmean(mse_cuts)
        gain_m = (avg_b - avg_m) / avg_b * 100
        gain_c = (avg_b - avg_c) / avg_b * 100
        best = "M2C" if gain_m > gain_c else "Cuts+"
        
        tqdm.write(f"Done: {f:<20} | M2C Gain: {gain_m:>6.2f}% | Cuts Gain: {gain_c:>6.2f}% | Winner: {best}")
        results.append({'Dataset': f, 'Base_MSE': avg_b, 'M2C_MSE': avg_m, 'CUTS_MSE': avg_c, 'm2c': gain_m, 'cuts': gain_c})

    if results:
        runtime.save_results(pd.DataFrame(results))
        print("\n" + "="*85)
        print(f"OVERALL AVG M2C IMPROVEMENT:   {np.mean([r['m2c'] for r in results]):.2f}%")
        print(f"OVERALL AVG CUTS+ IMPROVEMENT: {np.mean([r['cuts'] for r in results]):.2f}%")
        print("="*85)

if __name__ == "__main__":
    runtime.configure(__file__, globals())
    run_benchmark()
    runtime.ensure_results()
