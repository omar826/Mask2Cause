# Mask2Cause

M2C learns causal graphs from multivariate time series using MSE or Gaussian NLL.
The artifact also includes a two-graph model for mean/variance parent recovery
and causal-pruning forecasting experiments.

## Contents and installation

- `causalDiscovery/`: models, dataset loaders, runner, and graph metrics.
- `data/`: VAR, Lorenz-96, Mixed Physics, and RHINO synthetic benchmarks.
- `forecasting/`: forecasting scripts, VAR/Lorenz data, and causal masks.
- `config/best_configs.json`: model-specific settings for all six benchmarks.
- `tests/`: automated checks of loaders, metrics, models, and CLI behavior.

Use Python 3.10 or newer and install the dependencies:

```sh
python -m pip install -r config/requirements.txt
```

Download CausalTime from its [official website](https://www.causaltime.cc/).
Place each dataset's `gen_data.npy` and `graph.npy` under
`data/causaltime_gen_ver1.0/traffic`, `pm25`, or `medical`.

Download DREAM3 from the [official SRU implementation used by the paper](https://github.com/sakhanna/SRU_for_GCI/tree/master/data).
Place `Size100Ecoli1.pt`, `Size100Ecoli2.pt`, and the three `Size100Yeast*.pt`
files under `data/dream3/Dream3TensorData/`, with the corresponding ground-truth
TSV files under `data/dream3/TrueGeneNetworks/`. Both loaders are included;
these two benchmarks are not bundled.

## Run with saved settings

Supply the model (`mse`, `nll`, or `dual`) and an exact dataset filename:

```sh
python -m causalDiscovery.run_and_tune mse rhino_paper_ER_N10_noinst_history_seed0.npz
python -m causalDiscovery.run_and_tune nll S_30_T_500_dataset_1.npz
python -m causalDiscovery.run_and_tune dual heteroscedastic_data_50_regen.npz
```

The runner selects settings from `config/best_configs.json`. A dataset filename
alone selects one run. Full paths are also accepted; CausalTime uses its dataset
directory. MSE/NLL settings cover all six benchmarks; dual settings cover Mixed
Physics. Use `--device cuda` for GPU execution and `--output PATH` to choose a new
output directory.

## Run with arbitrary settings

Override any saved value directly, or use `--manual` to bypass saved settings:

```sh
python -m causalDiscovery.run_and_tune nll S_30_T_500_dataset_1.npz --lr 0.003 --epochs 50 --dropout 0.0 --adj-init -1
python -m causalDiscovery.run_and_tune mse data/var/S_30_T_500_dataset_1.npz --manual --dataset-type var --lr 0.001 --batch-size 32 --seq-len 3 --d-model 64 --lambda-l1 0.01 --diagonal-force 100 --epochs 10
```

Omitted manual options use the model defaults. Explicitly request repeat seeds
or a Cartesian hyperparameter grid by supplying multiple values:

```sh
python -m causalDiscovery.run_and_tune nll S_30_T_500_dataset_1.npz --seeds 0 1 2 --lr 0.001 0.003 --dropout 0.0 0.1
```

Each requested run saves and displays its own AUROC, AUPRC, SHD, and F1, along with
its configuration, checkpoint, loss history, and adjacency scores. Full-matrix
and off-diagonal metrics are saved; SHD/F1 use fixed-0.5 and true-density top-K
thresholds. Dual runs also report separate parent metrics when labels exist.

Use `--help` for all options, `--dry-run` to display configurations without
running, and `--verbose` to print epoch losses.

## Forecasting

```sh
python forecasting/lor_var.py --output results/forecast_lor
python forecasting/var_mlp.py --output results/forecast_var_mlp
```

The `*_arima.py`, `*_mlp.py`, `*_nbeats.py`, and `*_var.py` scripts use settings
defined in each script and save MSE and gain/parameter comparisons to
`metrics.csv`. Neural scripts accept `--epochs`; all accept `--seed` and `--limit`.
