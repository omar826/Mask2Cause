import itertools
import pandas as pd
import numpy as np
import torch
import traceback
import argparse
import importlib

def get_train_func(module_name):
    """ """
    try:
        module = importlib.import_module(module_name)
        return getattr(module, "train_and_validate")
    except (ImportError, AttributeError) as e:
        print(f"Error: Could not import train_and_validate from module '{module_name}'")
        raise e

def parse_args():
    """ """
    parser = argparse.ArgumentParser(description="Grid Search for Causal Graph Transformer")
    
    parser.add_argument("-m", "--module", type=str, default="mask2cause_MSE",
                        help="Module name to import train_and_validate from")
    
    parser.add_argument("-lr", "--lr", type=float, nargs='+', default=[0.001])
    parser.add_argument("-bs", "--batch_size", type=int, nargs='+', default=[32])
    parser.add_argument("-seq", "--seq_len", type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50])
    parser.add_argument("-dm", "--d_model", type=int, nargs='+', default=[64])
    parser.add_argument("-l1", "--lambda_l1", type=float, nargs='+', default=[0.01])
    parser.add_argument("-df", "--diagonal_force", type=float, nargs='+', default=[100.0])
    parser.add_argument("-e", "--epochs", type=int, nargs='+', default=[10])
    
    parser.add_argument("--data_dir", type=str, nargs='+', default=['data/var/S_30_T_500_dataset_0.npz'])
    parser.add_argument("--l1_anneal", type=int, nargs='+', default=[0])

    return parser.parse_args()

def run_tuning():
    """ """
    args = parse_args()
    
    train_and_validate = get_train_func(args.module)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch, 'backends') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    grid = {
        'lr': args.lr,
        'lambda_l1': args.lambda_l1,
        'l1_anneal_epochs': args.l1_anneal,
        'd_model': args.d_model,
        'n_heads': [4], 
        'e_layers': [2], 
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'epochs': args.epochs,
        'data_dir': args.data_dir,
        'early_stop': [False],
        'diagonal_force': args.diagonal_force
    }

    print(f"Starting Tuning on {device} using logic from: {args.module}")
    
    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    results = []

    print(f"Total configurations to test: {len(combinations)}")

    for i, config in enumerate(combinations):
        print(f"\n--- Running Config {i+1}/{len(combinations)} ---")
        print(config)
        
        seeds = [0] 
        aurocs = []
        best_shds = []

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            try:
                res = train_and_validate(config, device, verbose=True)
                aurocs.append(res['auroc'])
                best_shds.append(res['best_shd'])
            except Exception as e:
                print(f"   Seed {seed}: FAILED with error: {e}")
                traceback.print_exc()

        res_entry = config.copy()
        res_entry['test_auroc'] = np.mean(aurocs) if aurocs else 0.5
        res_entry['test_shd'] = np.mean(best_shds) if best_shds else None
        results.append(res_entry)

    df = pd.DataFrame(results)
    if not df.empty:
        best_config_auc = df.loc[df['test_auroc'].idxmax()]
        df.to_csv("tuning_results_transformer.csv", index=False)
        print("\nFull results saved to tuning_results_transformer.csv")
    else:
        print("No successful runs.")

if __name__ == "__main__":
    run_tuning()
