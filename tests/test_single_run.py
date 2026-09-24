import contextlib
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch
from causalDiscovery.run_and_tune import ROOT, main, parse_args, resolve_config, resolve_configs


class SingleRunTests(unittest.TestCase):
    def test_objective_specific_settings_and_file_seed(self):
        name = 'rhino_paper_ER_N20_noinst_history_seed4.npz'
        mse = resolve_config(parse_args(['mse', name]))
        nll = resolve_config(parse_args(['nll', name]))
        self.assertEqual((mse['batch_size'], mse['adj_init'], mse['seed']), (32, -1, 4))
        self.assertEqual((nll['batch_size'], nll['dropout'], nll['seed']), (128, 0, 4))

    def test_explicit_sweeps_only(self):
        name = 'S_30_T_500_dataset_1.npz'
        self.assertEqual(len(resolve_configs(parse_args(['mse', name]))), 1)
        configs = resolve_configs(parse_args(['mse', name, '--seeds', '0', '1',
                                               '--lr', '.001', '.003']))
        self.assertEqual(len(configs), 4)
        self.assertEqual({(c['seed'], c['lr']) for c in configs},
                         {(0, .001), (0, .003), (1, .001), (1, .003)})
        self.assertEqual(len({c['output_dir'] for c in configs}), 4)
        with self.assertRaisesRegex(ValueError, 'exact'):
            resolve_config(parse_args(['mse', '*.npz']))

    def test_manual_without_catalog(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)/'custom.npz'
            data.touch()
            c = resolve_config(parse_args(['mse', str(data), '--manual', '--config',
                                          'missing.json', '--dataset-type', 'var', '--lr', '.02']))
            self.assertEqual(c['lr'], .02)
            self.assertEqual(c['configuration'], 'manual')

    def test_catalog_resolves_every_supported_pair(self):
        catalog = json.loads((ROOT/'config/best_configs.json').read_text())
        for name, entry in catalog['datasets'].items():
            if not (ROOT/entry['path']).exists():
                self.assertIn(catalog['configurations'][entry['configuration']]['dataset_type'], ['dream3', 'causaltime'])
                continue
            for model in catalog['configurations'][entry['configuration']]['models']:
                config = resolve_config(parse_args([model, name]))
                self.assertEqual(config['model'], model)
                self.assertEqual(config['configuration'], entry['configuration'])

    def test_scalar_overrides_and_missing_model(self):
        config = resolve_config(parse_args(['nll', 'heteroscedastic_data_50_regen.npz',
                                           '--dropout', '0', '--adj-init', '-1', '--seed', '7']))
        self.assertEqual((config['dropout'], config['adj_init'], config['seed']), (0, -1, 7))
        with self.assertRaisesRegex(ValueError, 'No dual settings'):
            resolve_config(parse_args(['dual', 'S_30_T_500_dataset_1.npz']))

    def test_main_trains_once_and_reports_one_file(self):
        with patch('causalDiscovery.run_and_tune.train', return_value={'offdiag': {}, 'full': {}}) as train:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                main(['mse', 'S_30_T_500_dataset_1.npz'])
            train.assert_called_once()
            self.assertIn('S_30_T_500_dataset_1.npz', output.getvalue())
            self.assertNotIn('std_population', output.getvalue())
