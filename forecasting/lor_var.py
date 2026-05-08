import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import os
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

NUM_VARS = 10  
BASE_PATH = "data_for_forecasting/lor_data"
DATA_DIR = os.path.join(BASE_PATH, "lor_data")
M2C_DIR = os.path.join(BASE_PATH, "mask2cause_matrices")

# ---------------- HELPERS ----------------
def count_linear_params(input_dim):
    # coefficients + intercept (bias)
    return input_dim + 1

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
def run_lor_linear_benchmark():
    dataset_results = []

    if not os.path.exists(DATA_DIR):
        print(f"Directory not found: {DATA_DIR}")
        return

    csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
    
    print(f"Found {len(csv_files)} Lorenz datasets. Starting Linear Regression benchmark...")
    print("-" * 80)

    for data_file in tqdm(csv_files, desc="Overall Progress"):
        prefix = data_file.replace("_data.csv", "")
        
        # Load M2C Matrix
        adj_m2c = load_m2c_matrix(M2C_DIR, prefix)
        if adj_m2c is None:
            continue

        # Load and Scale Data
        df = pd.read_csv(os.path.join(DATA_DIR, data_file))
        raw = df.values[:, :NUM_VARS]

        split = int(len(raw) * 0.8)
        scaler = StandardScaler().fit(raw[:split])
        data = scaler.transform(raw)

        # Train/Test Split (t -> t+1)
        X_all, Y_all = data[:-1], data[1:]
        X_tr, Y_tr = X_all[:split-1], Y_all[:split-1]
        X_te, Y_te = X_all[split-1:], Y_all[split-1:]

        mse_store = {"Base": [], "M2C": []}
        param_store = {"Base": [], "M2C": []}

        # Node-by-node univariate forecasting
        for i in tqdm(range(NUM_VARS), desc=f"  Vars in {prefix[:12]}", leave=False):
            y_tr, y_te = Y_tr[:, i], Y_te[:, i]

            # --- 1. BASE (Dense: All variables as input) ---
            model_b = LinearRegression().fit(X_tr, y_tr)
            mse_b = mean_squared_error(y_te, model_b.predict(X_te))
            p_b = count_linear_params(NUM_VARS)

            mse_store["Base"].append(mse_b)
            param_store["Base"].append(p_b)

            # --- 2. M2C (Sparse: Trivial self + Gated parents) ---
            parents = np.where(adj_m2c[:, i] == 1)[0].tolist()
            
            # Always include the variable's own history
            if i not in parents:
                parents.append(i)
            
            idx = sorted([j for j in parents if j < NUM_VARS])
            
            model_c = LinearRegression().fit(X_tr[:, idx], y_tr)
            mse_c = mean_squared_error(y_te, model_c.predict(X_te[:, idx]))
            p_c = count_linear_params(len(idx))

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
        print(f"{'LORENZ LINEAR REGRESSION CAUSAL SUMMARY':^95}")
        print("="*95)
        df_final = pd.DataFrame(dataset_results)
        print(df_final.to_string(index=False))
        print("-" * 95)
        print(f"Overall Average Gain         : {df_final['Gain_%'].mean():.2f}%")
        print(f"Overall Average Param Red    : {df_final['Param_Red_%'].mean():.2f}%")
        print("="*95)

if __name__ == "__main__":
    run_lor_linear_benchmark()