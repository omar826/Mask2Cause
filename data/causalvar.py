import numpy as np
import os

def generate_mixed_causality_data(
    length=5000, 
    num_vars=10, 
    lag=3, 
    density=0.30,       # 30% of possible edges
    var_edge_ratio=1, # 75% of edges are Variance-only (MSE fails here)
    save_path="heteroscedastic_data_100.npz"
):
    print(f"Generating Mixed Physics Data: N={num_vars}, Density={density}...")
    
    # 1. Setup Ground Truth Matrices
    # mean_adj: X -> Mean(Y)
    # var_adj:  X -> Variance(Y)
    # GC:       Binary Ground Truth (Union of both)
    mean_adj = np.zeros((num_vars, num_vars))
    var_adj = np.zeros((num_vars, num_vars))
    GC = np.zeros((num_vars, num_vars))
    
    # Determine number of edges (excluding self-loops)
    num_possible = num_vars * (num_vars - 1)
    num_edges = int(num_possible * density)
    
    # Select random pairs for edges (excluding diagonal)
    # logic: generate all pairs, remove diagonal, sample k
    all_pairs = [(i, j) for i in range(num_vars) for j in range(num_vars) if i != j]
    selected_indices = np.random.choice(len(all_pairs), num_edges, replace=False)
    selected_pairs = [all_pairs[i] for i in selected_indices]
    
    print(f"Selected {len(selected_pairs)} edges out of {num_possible} possible.")

    # 2. Assign Edge Physics
    count_mean = 0
    count_var = 0
    
    for (src, tgt) in selected_pairs:
        GC[src, tgt] = 1
        
        # Decide Physics Type
        is_variance_edge = np.random.rand() < var_edge_ratio
        
        if is_variance_edge:
            # --- Type B: Pure Variance Edge ---
            # Source affects Target's volatility, but NOT the mean.
            # Mean weight is strictly 0.
            # Variance weight is high to be detectable.
            var_adj[src, tgt] = np.random.uniform(2.0, 3.0)
            mean_adj[src, tgt] = 0.0 
            count_var += 1
        else:
            # --- Type A: Mean Edge ---
            # Standard AR causality.
            mean_adj[src, tgt] = np.random.uniform(0.4, 0.6) * np.random.choice([-1, 1])
            var_adj[src, tgt] = 0.0
            count_mean += 1

    # 3. Add Self-Loops (Stabilizers)
    # Every variable should depend on its own past MEAN to be stable.
    for i in range(num_vars):
        mean_adj[i, i] = 0.4
        GC[i, i] = 1

    print(f"Graph Construction Complete:")
    print(f"  - Mean-Causal Edges (MSE wins): {count_mean}")
    print(f"  - Var-Causal Edges  (NLL wins): {count_var}")
    print(f"  - Self-Loops: {num_vars}")
    
    # 4. Simulation Loop
    X = np.zeros((length, num_vars))
    noise = np.random.randn(length, num_vars)
    
    # Burn-in
    X[:lag, :] = noise[:lag, :] 
    
    for t in range(lag, length):
        prev_x = X[t-lag, :] # Shape (N,)
        
        # A. Mean Component (Linear)
        # mu[t] = prev_x @ mean_adj
        mu_t = prev_x @ mean_adj 
        
        # B. Variance Component (Heteroscedastic)
        # sigma[t] = sqrt(base + (prev_x^2) @ var_adj)
        # Note: If var_adj is 0, sigma is const (homoscedastic).
        # If var_adj > 0, sigma fluctuates based on source magnitude.
        volatility_drive = (prev_x ** 2) @ var_adj
        sigma_t = np.sqrt(0.1 + volatility_drive)
        
        # C. Generate
        X[t, :] = mu_t + sigma_t * noise[t, :]
        
        # Stability clip (just in case of runaway feedback)
        if np.max(np.abs(X[t, :])) > 50:
            X[t, :] = np.clip(X[t, :], -50, 50)

    # 5. Save
    print("\nGround Truth Adjacency Sample (First 5x5):")
    print(GC)
    
    np.savez(save_path, X=X, GC=GC)
    print(f"\nSaved dataset to {save_path}")

if __name__ == "__main__":
    generate_mixed_causality_data()