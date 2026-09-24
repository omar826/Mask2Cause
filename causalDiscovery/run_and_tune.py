"""Run a dataset with saved settings or explicit hyperparameters."""
import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causalDiscovery.training import ROOT, train


def model_name(value):
    aliases = {'mse': 'mse', 'nll': 'nll', 'dual': 'dual',
               'm2c-mse': 'mse', 'm2c-nll': 'nll',
               'mask2cause_mse': 'mse', 'mask2cause_nll': 'nll',
               'mask2cause_dual_graph_nll': 'dual'}
    try:
        return aliases[value.lower()]
    except KeyError:
        raise argparse.ArgumentTypeError('Choose mse, nll, or dual')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('model', type=model_name)
    parser.add_argument('dataset', help='Exact dataset filename, path, or CausalTime directory')
    parser.add_argument('--config', type=Path, default=ROOT/'config/best_configs.json')
    parser.add_argument('--seed', '--seeds', dest='seeds', type=int, nargs='+', help='Explicit random seeds for repeated runs')
    parser.add_argument('--manual', action='store_true', help='Use CLI settings and model defaults instead of saved settings')
    parser.add_argument('--dataset-type', choices=['auto','var','lorenz96','mixed','rhino','dream3','causaltime'], default='auto')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    for flags, dest, typ in [
        (['-lr', '--lr'], 'lr', float),
        (['-bs', '--batch-size', '--batch_size'], 'batch_size', int),
        (['-seq', '--seq-len', '--seq_len', '--lag'], 'seq_len', int),
        (['-dm', '--d-model', '--d_model'], 'd_model', int),
        (['-l1', '--lambda-l1', '--lambda_l1'], 'lambda_l1', float),
        (['-df', '--diagonal-force', '--diagonal_force'], 'diagonal_force', float),
        (['-e', '--epochs'], 'epochs', int),
        (['--l1-anneal', '--l1_anneal'], 'l1_anneal_epochs', int),
        (['--adj-init'], 'adj_init', float),
        (['--dropout'], 'dropout', float),
        (['--n-heads'], 'n_heads', int),
        (['--e-layers'], 'e_layers', int),
        (['--grad-clip'], 'grad_clip', float),
    ]:
        parser.add_argument(*flags, dest=dest, type=typ, nargs='+')
    parser.add_argument('--shuffle-mode', choices=['device', 'dataloader'], nargs='+')
    return parser.parse_args(argv)


def resolve_configs(args):
    if any(c in args.dataset for c in '*?[]'):
        raise ValueError('Choose one exact dataset filename, not a pattern')
    requested = Path(args.dataset)
    key = requested.parent.name if requested.name == 'gen_data.npy' else requested.name
    catalog = json.loads(args.config.read_text(encoding='utf-8-sig')) if not args.manual else {}
    entry = catalog.get('datasets', {}).get(key, {})
    if not args.manual and not entry:
        raise ValueError(f'No saved configuration for {key}; use --manual for arbitrary settings')
    group = catalog.get('configurations', {}).get(entry.get('configuration'), {})
    if not args.manual and args.model not in group['models']:
        raise ValueError(f'No {args.model} settings for {entry["configuration"]}; use --manual')
    path = requested
    if requested.parent == Path('.') and not requested.exists() and entry:
        path = ROOT/entry['path']
    elif not requested.is_absolute() and not requested.exists():
        path = ROOT/requested
    if args.manual and requested.parent == Path('.') and not path.exists():
        matches = list((ROOT/'data').rglob(requested.name))
        if len(matches) > 1:
            raise ValueError('Ambiguous dataset name; supply its full path')
        if matches:
            path = matches[0]
    kind = args.dataset_type if args.dataset_type != 'auto' else group.get('dataset_type', 'auto')
    if path.name == 'gen_data.npy':
        path = path.parent
    if not path.exists():
        raise ValueError(f'Dataset not found: {path}. Download the dataset before running.')
    if path.is_dir() and kind not in ['auto', 'causaltime']:
        raise ValueError('This benchmark requires a dataset file')
    config = dict(group.get('models', {}).get(args.model, {}))
    keys = ['lr', 'batch_size', 'seq_len', 'd_model', 'lambda_l1', 'diagonal_force',
            'epochs', 'l1_anneal_epochs', 'adj_init', 'dropout', 'n_heads', 'e_layers',
            'grad_clip', 'shuffle_mode']
    grid = {key: getattr(args, key) for key in keys if getattr(args, key) is not None}
    variants = [dict(zip(grid, values)) for values in itertools.product(*grid.values())]
    seeds = args.seeds if args.seeds is not None else [entry.get('seed', 0)]
    configs = [dict(config, **variant, seed=seed) for variant in variants for seed in seeds]
    output = args.output or ROOT/'results'/(
        f'{args.model}_{key}_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    for i, config in enumerate(configs):
        config.update(model=args.model, dataset_type=kind, data_dir=str(path.resolve()),
                      threads=args.threads, output_dir=str(output/f'run_{i:04d}' if len(configs)>1 else output),
                      configuration=entry.get('configuration', 'manual'),
                      config_source=str(args.config.resolve()) if not args.manual else 'CLI/model defaults')
    return configs


def resolve_config(args):
    configs = resolve_configs(args)
    if len(configs) != 1:
        raise ValueError('Use resolve_configs for explicit sweeps')
    return configs[0]


def main(argv=None):
    args = parse_args(argv)
    configs = resolve_configs(args)
    if args.dry_run:
        print(json.dumps(configs[0] if len(configs)==1 else configs, indent=2))
        return
    if len(configs)>1:
        Path(configs[0]['output_dir']).parent.mkdir(parents=True, exist_ok=False)
    for config in configs:
        result = train(config, args.device, args.model, args.verbose)
        print(json.dumps(dict(model=args.model, dataset=config['data_dir'], seed=config['seed'],
                             offdiag=result['offdiag'], full=result['full'],
                             **({'parents': result['parents']} if 'parents' in result else {})),
                         indent=2, allow_nan=False))
        print(f'Saved this run to {config["output_dir"]}')


if __name__ == '__main__':
    main()
