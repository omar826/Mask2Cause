"""Fixed-epoch training and consistent reporting for every supported benchmark."""
import hashlib
import importlib
import json
import platform
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from causalDiscovery import datasets
from causalDiscovery.metrics import graph_metrics, typed_metrics

MODELS = {'mse': 'mask2cause_MSE', 'nll': 'mask2cause_NLL', 'dual': 'mask2cause_dual_graph_NLL'}
ROOT = Path(__file__).resolve().parents[1]


def resolve_data(path):
    path = Path(path)
    if not path.exists() and not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def infer_dataset(path):
    if path.is_dir() and (path / 'gen_data.npy').exists():
        return 'causaltime'
    if path.is_dir() and (path / 'adj_matrix.npy').exists():
        return 'rhino'
    if path.suffix == '.pt':
        return 'dream3'
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as d:
            if 'train' in d and 'adj_matrix' in d:
                return 'rhino'
            if 'Wmu' in d or 'heteroscedastic' in path.name or 'mixed_physics' in str(path):
                return 'mixed'
    if 'lorenz' in str(path).lower():
        return 'lorenz96'
    if 'var' in str(path).lower():
        return 'var'
    raise ValueError('Cannot infer benchmark; supply --dataset-type')


def load_data(config):
    path = resolve_data(config['data_dir'])
    kind = config.get('dataset_type', 'auto')
    if kind == 'auto':
        kind = infer_dataset(path)
    sl, bs = config['seq_len'], config['batch_size']
    normalization = None
    if kind == 'rhino':
        windows, truth, mean, std = datasets.load_rhino_arrays(str(path), sl)
        loader = DataLoader(TensorDataset(torch.from_numpy(windows[:, :-1]),
                                         torch.from_numpy(windows[:, -1])), batch_size=bs, shuffle=True)
        normalization = dict(mean=mean, std=std)
        if config.get('l1_anneal_epochs', 0):
            raise ValueError('RHINO configurations require annealing off, as in the supplied artifact')
    else:
        funcs = {'mixed': datasets.load_heteroscedastic_data, 'dream3': datasets.load_dream3_data,
                 'causaltime': datasets.load_raw_traffic_data, 'var': datasets.load_var_data,
                 'lorenz96': datasets.load_lorenz_data}
        if kind not in funcs:
            raise ValueError(f'Unknown benchmark: {kind}')
        loader, truth = funcs[kind](str(path), sl, bs)
    if not len(loader.dataset):
        raise ValueError('No training windows; reduce sequence length')
    x, y = loader.dataset.tensors
    if x.ndim != 3 or y.ndim != 2 or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('Loader returned invalid training tensors')
    files = [path] if path.is_file() else [p for p in path.iterdir() if p.name in
                                          ['train.csv', 'adj_matrix.npy', 'gen_data.npy', 'graph.npy']]
    if kind == 'dream3':
        import re
        match = re.fullmatch(r'Size(10|50|100)(Ecoli\d+|Yeast\d+)\.pt', path.name)
        if match:
            files.append(path.parent.parent / 'TrueGeneNetworks' / f'InSilicoSize{match[1]}-{match[2]}.tsv')
    return loader, np.asarray(truth, dtype=int), kind, path, normalization, files


def train(config, device='cpu', model_name='nll', verbose=False):
    config = dict(config)
    defaults = dict(lr=.001, batch_size=32, seq_len=3, d_model=32, n_heads=4, e_layers=2,
                    lambda_l1=.01, diagonal_force=100., l1_anneal_epochs=0, dropout=.1,
                    epochs=20, seed=0, threads=1, shuffle_mode='dataloader', grad_clip=0.)
    for k, v in defaults.items():
        config.setdefault(k, v)
    config.setdefault('adj_init', -1. if model_name == 'mse' else 1.)
    if model_name not in MODELS:
        raise ValueError(f'Unknown model: {model_name}')
    if any(config[k] < 1 for k in ['epochs', 'batch_size', 'seq_len', 'd_model', 'n_heads', 'e_layers', 'threads']):
        raise ValueError('Training dimensions and counts must be positive')
    if config['d_model'] % config['n_heads'] or config['lr'] <= 0 or config['lambda_l1'] < 0:
        raise ValueError('Invalid width/heads, learning rate, or L1 penalty')
    if config['l1_anneal_epochs'] < 0 or not 0 <= config['dropout'] < 1:
        raise ValueError('Invalid annealing duration or dropout')
    if config['shuffle_mode'] not in ['dataloader', 'device']:
        raise ValueError('Unknown shuffle mode')
    destination = Path(config['output_dir'])
    if destination.exists():
        raise FileExistsError(f'Output already exists: {destination}. Choose a new directory.')
    torch.set_num_threads(config['threads'])
    seed = config['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device)
    loader, truth, kind, path, normalization, files = load_data(config)
    config.update(dataset_type=kind, data_dir=str(path), model=model_name, device=str(device))
    x, y = loader.dataset.tensors
    module = importlib.import_module('causalDiscovery.' + MODELS[model_name])
    model = module.CausalGraphTransformer(SimpleNamespace(enc_in=x.shape[-1], **config)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    if config['shuffle_mode'] == 'device':
        x, y = x.to(device), y.to(device)
        generator = torch.Generator(device=device).manual_seed(seed)
    destination.mkdir(parents=True)
    (destination / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    history, start = [], time.perf_counter()
    for epoch in range(config['epochs']):
        model.train()
        if config['shuffle_mode'] == 'device':
            perm = torch.randperm(len(x), device=device, generator=generator)
            batches = ((x[idx], y[idx]) for idx in perm.split(config['batch_size']))
        else:
            batches = loader
        total = 0.
        for bx, by in batches:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            if model_name == 'mse':
                prediction_loss = torch.nn.functional.mse_loss(model(bx), by)
            else:
                mu, var = model(bx)
                prediction_loss = torch.nn.functional.gaussian_nll_loss(mu, by, var)
            graphs = model.get_adjacency_matrices() if model_name == 'dual' else (model.get_adjacency_matrix(),)
            penalty = sum((a * (1-model.self_mask)).sum() /
                          (a.numel()-model.self_mask.sum()+1e-5) for a in graphs) / len(graphs)
            anneal = config['l1_anneal_epochs']
            weight = config['lambda_l1'] * min(1., epoch / anneal) if anneal else config['lambda_l1']
            loss = prediction_loss + weight * penalty
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at epoch {epoch+1}')
            loss.backward()
            if config['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()
            total += prediction_loss.detach().item() * len(bx)
        history.append(dict(epoch=epoch+1, prediction_loss=total/len(loader.dataset)))
        if verbose:
            print(f'Epoch {epoch+1}/{config["epochs"]}: prediction loss {history[-1]["prediction_loss"]:.6f}', flush=True)
    with torch.no_grad():
        if model_name == 'dual':
            am, av = [a.detach().cpu().numpy() for a in model.get_adjacency_matrices()]
            scores = np.maximum(am, av)
            np.save(destination/'mean_scores.npy', am); np.save(destination/'variance_scores.npy', av)
        else:
            scores = model.get_adjacency_matrix().detach().cpu().numpy()
    n = len(truth)
    scored = scores[:n, :n]
    result = graph_metrics(truth, scored)
    result.update(**result['offdiag'])  # Stable short names for CLI/table summaries.
    if model_name == 'dual' and kind == 'mixed':
        with np.load(path, allow_pickle=False) as data:
            if 'Wmu' in data and 'Wsigma' in data:
                result['parents'] = typed_metrics(data['Wmu'], data['Wsigma'], am[:n, :n], av[:n, :n])
    result['metadata'] = dict(dataset_type=kind, model=model_name, seed=seed,
                             epochs=config['epochs'], training_windows=len(loader.dataset),
                             input_variables=x.shape[-1], graph_variables=n,
                             seconds=time.perf_counter()-start, parameters=sum(p.numel() for p in model.parameters()),
                             evaluation='fixed final epoch; primary metrics exclude self-edges; SHD=FP+FN',
                             data_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                             source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                            for p in Path(__file__).parent.glob('*.py')},
                             python=platform.python_version(), torch=torch.__version__, numpy=np.__version__)
    np.save(destination/'scores.npy', scored)
    if scores.shape != scored.shape:
        np.save(destination/'scores_all_variables.npy', scores)
    np.save(destination/'ground_truth.npy', truth)
    torch.save(dict(model=model.state_dict(), config=config, normalization=normalization), destination/'model.pt')
    (destination/'history.json').write_text(json.dumps(history, indent=2) + '\n')
    (destination/'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result
