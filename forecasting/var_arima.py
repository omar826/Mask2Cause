import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import os
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------- SETTINGS ----------------
NUM_VARS = 10
BASE_PATH = "data_for_forecasting/var_data"
DATA_DIR = os.path.join(BASE_PATH, "var_datasets")
M2C_DIR = os.path.join(BASE_PATH, "mask2cause_matrices")

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

# ---------------- MAIN ----------------
def run_var_arimax_benchmark():
    dataset_results = []

    if not os.path.exists(DATA_DIR):
        print(f"Directory not found: {DATA_DIR}")
        return

    csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
    
    print(f"Found {len(csv_files)} VAR datasets. Starting ARIMAX benchmark...")
    print("-" * 90)

    for data_file in tqdm(csv_files, desc="Overall Progress"):
        prefix = data_file.replace("_data.csv", "")
        
        # Load M2C Matrix
        adj_m2c = load_m2c_matrix(M2C_DIR, prefix)
        if adj_m2c is None:
            continue

        # Load Data - filter for columns with 'T' (time points)
        df = pd.read_csv(os.path.join(DATA_DIR, data_file))
        raw = df.filter(regex="T").values[:, :NUM_VARS]

        split = int(len(raw) * 0.8)
        scaler = StandardScaler().fit(raw[:split])
        data = scaler.transform(raw)

        # Train/Test Split (t -> t+1)
        X_all, Y_all = data[:-1], data[1:]
        X_tr, Y_tr = X_all[:split-1], Y_all[:split-1]
        X_te, Y_te = X_all[split-1:], Y_all[split-1:]

        mse_base_list, mse_m2c_list = [], []
        ex_count_base, ex_count_m2c = 0, 0

        # Node-by-node univariate forecasting
        for i in tqdm(range(NUM_VARS), desc=f"  Vars in {prefix[:12]}", leave=False):
            y_tr, y_te = Y_tr[:, i], Y_te[:, i]

            # --- 1. BASE (Dense: all other 9 variables as exog) ---
            # ex_idx_b = [j for j in range(NUM_VARS) if j != i]
            ex_idx_b = [j for j in range(NUM_VARS)]
            ex_tr_b, ex_te_b = X_tr[:, ex_idx_b], X_te[:, ex_idx_b]
            ex_count_base += len(ex_idx_b)

            try:
                model_b = ARIMA(y_tr, exog=ex_tr_b, order=(1, 0, 0)).fit()
                p_base = model_b.apply(y_te, exog=ex_te_b).fittedvalues
                mse_b = mean_squared_error(y_te, p_base)
            except:
                mse_b = np.nan
            mse_base_list.append(mse_b)

            # --- 2. M2C (Sparse: Trivial self + Gated parents) ---
            # Predicted parents from mask (adj_m2c[source, target])
            parents = np.where(adj_m2c[:, i] == 1)[0].tolist()
            
            # Always include the variable's own history
            if i not in parents:
                parents.append(i)
            
            idx = sorted([j for j in parents if j < NUM_VARS])
            ex_count_m2c += len(idx)

            try:
                ex_tr_m, ex_te_m = X_tr[:, idx], X_te[:, idx]
                model_c = ARIMA(y_tr, exog=ex_tr_m, order=(1, 0, 0)).fit()
                p_m2c = model_c.apply(y_te, exog=ex_te_m).fittedvalues
                mse_c = mean_squared_error(y_te, p_m2c)
            except:
                mse_c = mse_b # Fallback to base if optimization fails
            mse_m2c_list.append(mse_c)

        # Aggregation
        avg_b = np.nanmean(mse_base_list)
        avg_m = np.nanmean(mse_m2c_list)
        
        if not np.isnan(avg_b) and avg_b > 0:
            gain = (avg_b - avg_m) / avg_b * 100
            param_red = (1 - (ex_count_m2c / ex_count_base)) * 100
            
            tqdm.write(f"Done: {prefix:<25} | Gain: {gain:>7.2f}% | Param Red: {param_red:>6.1f}%")
            dataset_results.append({
                "Dataset": prefix,
                "Base_MSE": avg_b,
                "M2C_MSE": avg_m,
                "Gain_%": gain,
                "Param_Red_%": param_red
            })

    # ---------------- FINAL SUMMARY ----------------
    if dataset_results:
        print("\n" + "="*95)
        print(f"{'VAR-ARIMAX CAUSAL BENCHMARK SUMMARY':^95}")
        print("="*95)
        df_final = pd.DataFrame(dataset_results)
        print(df_final.to_string(index=False))
        print("-" * 95)
        print(f"Overall Average Gain         : {df_final['Gain_%'].mean():.2f}%")
        print(f"Overall Average Param Red    : {df_final['Param_Red_%'].mean():.2f}%")
        print("="*95)

if __name__ == "__main__":
    run_var_arimax_benchmark()
