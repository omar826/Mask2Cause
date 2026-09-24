"""Small CLI/output adapter; forecasting algorithms remain in their scripts."""
import argparse
import json
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np

OPTIONS = None
CONTEXT = {}


def configure(script, namespace):
    global OPTIONS, CONTEXT
    CONTEXT = namespace
    parser = argparse.ArgumentParser(description=Path(script).stem + ' forecasting benchmark')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, help='Override neural training epochs')
    parser.add_argument('--limit', type=int, help='Run the first N datasets (quick check)')
    parser.add_argument('--output', type=Path)
    OPTIONS = parser.parse_args()
    if OPTIONS.limit is not None and OPTIONS.limit < 1:
        parser.error('--limit must be positive')
    if OPTIONS.epochs is not None:
        if OPTIONS.epochs < 1 or 'EPOCHS' not in namespace:
            parser.error('--epochs is positive and supported only by neural scripts')
        namespace['EPOCHS'] = OPTIONS.epochs
    random.seed(OPTIONS.seed); np.random.seed(OPTIONS.seed)
    if 'torch' in namespace:
        namespace['torch'].manual_seed(OPTIONS.seed)
        namespace['torch'].set_num_threads(1)
    OPTIONS.output = OPTIONS.output or Path(__file__).resolve().parents[1] / 'results' / 'forecasting' / (
        Path(script).stem + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    OPTIONS.output.mkdir(parents=True, exist_ok=False)
    (OPTIONS.output/'config.json').write_text(json.dumps(dict(script=Path(script).name,
        seed=OPTIONS.seed, epochs=namespace.get('EPOCHS'), limit=OPTIONS.limit), indent=2)+'\n')


def select_files(files):
    if OPTIONS:
        usable = []
        for name in files:
            prefix = name.replace('_data.csv', '')
            gene = re.search(r'(Ecoli\d+|Yeast\d+)', name, re.I)
            missing = []
            for key in ['M2C_DIR', 'CUTS_DIR']:
                wanted = gene.group(0).lower() if key == 'CUTS_DIR' and gene else prefix.lower()
                if key in CONTEXT and not any(p.name.lower().startswith(wanted)
                        for p in Path(CONTEXT[key]).glob('*') if p.is_file()):
                    missing.append(key)
            if missing:
                print(f'Skipping {name}: no matching {", ".join(missing)} mask')
            else:
                usable.append(name)
        files = usable
    return files[:OPTIONS.limit] if OPTIONS and OPTIONS.limit else files


def save_results(frame):
    if frame.empty:
        raise RuntimeError('No forecasting results; check dataset and causal-mask paths')
    if OPTIONS:
        frame.to_csv(OPTIONS.output/'metrics.csv', index=False)


def ensure_results():
    if not (OPTIONS.output/'metrics.csv').exists():
        raise RuntimeError('No datasets produced results; check data and supplied masks')
    print(f'Saved forecasting metrics to {OPTIONS.output / "metrics.csv"}')
