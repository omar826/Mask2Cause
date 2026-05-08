# Official repository for Mask2Cause : Causal Discovery via Adjacency Constrained Causal Attention
Mask2Cause is a Transformer based **causal discovery** architecture that can be used to enhance **multivariate forecasting** with significant reduction in the parameter set size. It learns a global adjacency matrix (causal graph) that acts as a structural mask for the attention mechanism, ensuring that only truly causal parents can influence the target variable. This provides better forecasting accuracy due to absence of any spurious links that can misguide the model.


## Project Structure

```text
Mask2Cause/
├── causalDiscovery/                                                   # Mask2Cause code (has ablation variants as well)
│   ├── mask2cause_MSE_decoupledFinalProjection_residualPrediction.py  # Ablation Variant
│   ├── mask2cause_MSE_decoupledFinalProjection.py                     # Ablation Variant
│   ├── mask2cause_MSE_residualPrediction.py                           # Ablation Variant
│   ├── mask2cause_MSE.py                                              # Standard MSE model
│   ├── mask2cause_NLL_decoupledFinalProjection.py                     # Ablation Variant
│   ├── mask2cause_NLL.py                                              # Standard NLL model
│   └── run_and_tune.py                                                # Runs the above models and tunes over 11 sets of hyperparameters
|── config/                      
│   ├── environment.yml           
│   ├── requirements.txt        # dependencies to be installed for running Mask2Cause
├── data/                       # Datasets (dataloaders are in the run code itself)
│   ├── causaltime_gen_ver1.0/  # medical, pm25, traffic datasets (folders containing gen_data.npy, graph.npy each)
│   ├── dream3/                 # Dream3TensorData (.pt files) & TrueGeneNetworks (.tsv files)
│   ├── lorenz96/               # Lorenz system data (.npz files)
│   ├── mixed_physics/          # Mixed physics datasets (.npz files)
│   └── var/                    # Vector Autoregression data (.npz files)
│   ├── causalvar.py            # Generating Mixed-Physics dataset  
├── forecasting/                # Forecasting code (Along with CUTS+ comparision)
│   ├── data_for_forecasting/   # dream3, lor_data (lorenz in .csv) and var_data (VAR in .csv) datasets with discovered graphs from M2C, CUTS+
│   ├── dream_arima.py                  # Forecasting on Dream3 using ARIMAX model (M2C causal graph pruning)
│   ├── dream_cmp_cutsPlus_arima.py     # Compare on Dream3 using ARIMAX model (M2C and CUTS+ graph pruning)
│   ├── dream_cmp_cutsPlus_mlp.py       # Compare on Dream3 using MLP model (M2C and CUTS+ graph pruning)
│   └── dream_cmp_cutsPlus_nbeats.py    # Compare on Dream3 using NBEATS model (M2C and CUTS+ graph pruning)
│   ├── dream_cmp_cutsPlus_var.py       # Compare on Dream3 using VAR model (M2C and CUTS+ graph pruning)
│   ├── dream_mlp.py                    # Forecasting on Dream3 using MLP model (M2C causal graph pruning)
│   ├── dream_nbeats.py                 # Forecasting on Dream3 using NBEATS model (M2C causal graph pruning)
│   ├── dream_var.py                    # Forecasting on Dream3 using VAR model (M2C causal graph pruning)
│   ├── lor_arima.py                    # Forecasting on Lorenz using ARIMAX model (M2C and CUTS+ causal graph pruning)
│   └── lor_mlp.py                      # Forecasting on Lorenz using MLP model (M2C cand CUTS+ ausal graph pruning)
│   ├── lor_nbeats.py                   # Forecasting on Lorenz using NBEATS model (M2C cand CUTS+ ausal graph pruning)
│   ├── lor_var.py                      # Forecasting on Lorenz using VAR model (M2C and CUTS+ causal graph pruning)
│   ├── var_arima.py                    # Forecasting on VAR using ARIMAX model (M2C and CUTS+ causal graph pruning)
│   ├── var_mlp.py                      # Forecasting on VAR using MLP model (M2C and CUTS+ causal graph pruning)
│   ├── var_nbeats.py                   # Forecasting on VAR using NBEATS model (M2C and CUTS+ causal graph pruning)
│   └── var_var.py                      # Forecasting on VAR using VAR model (M2C and CUTS+ causal graph pruning)
├── results/
│   └── matrices/               # Saved .npy adjacency matrices stored here (after running)
└── README.md                   # readme file
```
## Set-Up
```bash
# Run the following commands on terminal to set-up the environment for running the files

python -m venv .venv                        # create a venv
.venv\\Scripts\\activate                    # Linux : source .venv/bin/activate  
pip install -r config/requirements.txt      # Install dependencies 
```
## Running Causal Discovery Examples
```bash
# Running Mask2Cause (MSE variant) on one of the lorenz96 dataset files - F_10_T_500_dataset_2.npz
python causalDiscovery/run_and_tune.py -m mask2cause_MSE -lr 0.001 -bs 32 -seq 1 -dm 64 -l1 0.02 -df 100 -e 150 --data_dir data\lorenz96\F_10_T_500_dataset_2.npz

# To change hyperparameters use following syntax
# python causalDiscovery/run_and_tune.py -m "Model Name" -lr "Learning Rate" -bs "Batch Size" -seq "Sequence Length" -dm "D_Model" -l1 "Penalty factor" --l1_anneal "Annealing penalty factor" -df "Diagonal Forcing" -e "Epochs" --data_dir "Relative path to data file" (data folder for the case of causaltime)
# Ex. Run Mask2Cause (NLL variant) with custom hyperparameters on one of Dream3 datasets
python causalDiscovery/run_and_tune.py -m mask2cause_NLL -lr 0.001 -bs 32 -seq 1 -dm 64 -l1 0.02 -df 100 -e 140 --data_dir data\lorenz96\F_10_T_500_dataset_2.npz
```
## Running Forecasting examples
```bash
cd forecasting                   
# Running any of the scripts in forecasting can be done by using "python scripts_name.py"
# Ex. Run ARIMAX on dream3
python lor_arima.py
```

