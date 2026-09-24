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

# ---------------- HELPERS ----------------
def load_dream3_ts_only(pt_path):
    try:
        data_dict = torch.load(pt_path, map_location='cpu', weights_only=False)
    except Exception:
        data_dict = torch.load(pt_path, map_location='cpu')
    X = data_dict['TsData']
    return X.numpy() if torch.is_tensor(X) else X

def load_m2c_matrix(data_file):
    m2c_path = os.path.join(M2C_DIR, f"{data_file}_predicted_adj.npy")
    if os.path.exists(m2c_path):
        adj = np.load(m2c_path)
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
    print(f"Starting Gated ARIMAX Benchmark (Self-Variable Always Accessible)")
    print("-" * 75)

    for f in tqdm(files, desc="Overall Progress"):
        raw_ts = load_dream3_ts_only(os.path.join(DATA_DIR, f))
        adj_m2c = load_m2c_matrix(f)

        if raw_ts is None or adj_m2c is None:
            continue

        num_vars = raw_ts.shape[1]
        
        # Scaling
        split = int(len(raw_ts) * 0.8)
        scaler = StandardScaler().fit(raw_ts[:split])
        data = scaler.transform(raw_ts)

        X_all, Y_all = data[:-1], data[1:]
        X_tr, Y_tr = X_all[:split-1], Y_all[:split-1]
        X_te, Y_te = X_all[split-1:], Y_all[split-1:]

        mse_base, mse_m2c = [], []
        total_params_base, total_params_m2c = 0, 0

        for i in tqdm(range(num_vars), desc=f"  Processing {f[:12]}", leave=False):
            y_tr, y_te = Y_tr[:, i], Y_te[:, i]

            # --- 1. BASE MODEL (Uses everything) ---
            ex_idx_b = [j for j in range(num_vars)] # Includes self
            ex_tr_b, ex_te_b = X_tr[:, ex_idx_b], X_te[:, ex_idx_b]
            total_params_base += len(ex_idx_b)

            try:
                m_base = ARIMA(y_tr, exog=ex_tr_b, order=(1, 0, 0)).fit()
                p_base = m_base.apply(y_te, exog=ex_te_b).fittedvalues
                mse_base.append(mean_squared_error(y_te, p_base))
            except:
                mse_base.append(np.nan)

            # --- 2. GATED M2C MODEL (Self + Gated Parents) ---
            # Extract parents from mask
            parents_from_mask = np.where(adj_m2c[i, :] == 1)[0].tolist()
            
            # Always include self-index 'i' in the predictors
            if i not in parents_from_mask:
                parents_from_mask.append(i)
            
            ex_idx_m = sorted(parents_from_mask)
            total_params_m2c += len(ex_idx_m)

            try:
                ex_tr_m, ex_te_m = X_tr[:, ex_idx_m], X_te[:, ex_idx_m]
                m_m2c = ARIMA(y_tr, exog=ex_tr_m, order=(1, 0, 0)).fit()
                p_m2c = m_m2c.apply(y_te, exog=ex_te_m).fittedvalues
                mse_m2c.append(mean_squared_error(y_te, p_m2c))
            except:
                mse_m2c.append(mse_base[-1] if len(mse_base)>0 else np.nan)

        # Aggregation
        avg_b = np.nanmean(mse_base)
        avg_m = np.nanmean(mse_m2c)
        
        if not np.isnan(avg_b) and avg_b != 0:
            gain = (avg_b - avg_m) / avg_b * 100
            param_red = (1 - (total_params_m2c / total_params_base)) * 100
            tqdm.write(f"Done: {f:<20} | Gain: {gain:>7.2f}% | Param Red: {param_red:>6.1f}%")
            results.append({'Dataset': f, 'Base_MSE': avg_b, 'M2C_MSE': avg_m, 'gain': gain, 'red': param_red})

    if results:
        runtime.save_results(pd.DataFrame(results))
        print("\n" + "="*75)
        print(f"AVERAGE GAIN: {np.mean([r['gain'] for r in results]):.2f}%")
        print(f"AVERAGE PARAMETER REDUCTION: {np.mean([r['red'] for r in results]):.2f}%")
        print("="*75)

if __name__ == "__main__":
    runtime.configure(__file__, globals())
    run_benchmark()
    runtime.ensure_results()
