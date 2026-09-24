import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from causalDiscovery.datasets import load_dream3_data, load_rhino_arrays
from causalDiscovery.metrics import graph_metrics
from causalDiscovery.training import train
from causalDiscovery.mask2cause_NLL import CausalGraphTransformer as Single
from causalDiscovery.mask2cause_dual_graph_NLL import CausalGraphTransformer as Dual


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        torch.set_num_threads(1)

    def tearDown(self):
        self.temp.cleanup()

    def rhino(self, instantaneous=False):
        raw = np.arange(2*8*3, dtype=np.float32).reshape(2, 8, 3)
        graph = np.zeros((3, 3, 3), dtype=int)
        graph[1, 0, 1] = 1; graph[2, 2, 0] = 1
        if instantaneous:
            graph[0, 0, 1] = 1
        path = self.root/'rhino.npz'
        np.savez(path, train=raw, test=raw+100000, adj_matrix=graph)
        return path, raw

    def test_rhino_normalization_boundaries_and_orientation(self):
        path, raw = self.rhino()
        windows, truth, mean, std = load_rhino_arrays(path, 2)
        expected = (raw-raw.mean((0, 1)))/(raw.std((0, 1))+1e-5)
        self.assertEqual(windows.shape, (12, 3, 3))
        np.testing.assert_array_equal(windows[5], expected[0, 5:8])
        np.testing.assert_array_equal(windows[6], expected[1, :3])
        np.testing.assert_array_equal(mean, raw.mean((0, 1)))
        self.assertEqual(truth[1, 0], 1)
        self.assertEqual(truth[0, 2], 1)
        self.assertEqual(int(truth.sum()), 2)
        path, _ = self.rhino(instantaneous=True)
        with self.assertRaises(ValueError):
            load_rhino_arrays(path, 2)

    def test_dream_original_flat_windows_and_size50(self):
        data = self.root/'Dream3TensorData'; data.mkdir()
        graph = self.root/'TrueGeneNetworks'; graph.mkdir()
        raw = torch.arange(12*50, dtype=torch.float32).reshape(12, 50)
        torch.save({'TsData': raw}, data/'Size50Ecoli1.pt')
        (graph/'InSilicoSize50-Ecoli1.tsv').write_text('G1\tG2\t+\n')
        loader, truth = load_dream3_data(str(data/'Size50Ecoli1.pt'), 3, 2)
        self.assertEqual(len(loader.dataset), 9)
        self.assertEqual(truth.shape, (50, 50))
        self.assertEqual(truth[1, 0], 1)
        expected = (raw-raw.mean(0))/(raw.std(0)+1e-5)
        torch.testing.assert_close(loader.dataset.tensors[0][5], expected[5:8])

    def test_metrics_hand_calculation_and_empty_graph(self):
        truth = np.zeros((3, 3), dtype=int); truth[1, 0] = 1
        score = np.zeros((3, 3)); score[1, 0] = .9; score[0, 2] = .8
        result = graph_metrics(truth, score)['offdiag']
        self.assertEqual(result['auroc'], 1)
        self.assertEqual(result['auprc'], 1)
        self.assertEqual(result['shd_density_oracle'], 0)
        self.assertEqual(result['shd_0_5'], 1)
        empty = graph_metrics(np.zeros((3, 3)), score)
        self.assertIsNone(empty['offdiag']['auroc'])
        self.assertIsNone(empty['offdiag']['f1_density_oracle'])
        json.dumps(empty, allow_nan=False)

    def test_dual_graph_paths_and_equal_graph_equivalence(self):
        args = SimpleNamespace(enc_in=3, seq_len=2, d_model=8, n_heads=2,
                               e_layers=2, dropout=0., diagonal_force=100.)
        model = Dual(args).eval()
        reference = Single(args).eval()
        reference.load_state_dict({k:v for k,v in model.state_dict().items() if k!='variance_adj_logits'})
        x = torch.randn(4, 2, 3)
        mu, var = model(x)
        ref_mu, ref_var = reference(x)
        torch.testing.assert_close(mu, ref_mu); torch.testing.assert_close(var, ref_var)
        gm, gv = torch.autograd.grad(mu.square().sum(), [model.adj_logits, model.variance_adj_logits], allow_unused=True)
        self.assertIsNotNone(gm); self.assertIsNone(gv)
        mu, var = model(x)
        gm, gv = torch.autograd.grad(var.sum(), [model.adj_logits, model.variance_adj_logits], allow_unused=True)
        self.assertIsNone(gm); self.assertIsNotNone(gv)

    def test_all_models_train_and_save_rhino_without_overwriting(self):
        path, _ = self.rhino()
        for name in ['mse', 'nll', 'dual']:
            config = dict(data_dir=str(path), dataset_type='rhino', seq_len=2, epochs=1,
                          batch_size=4, d_model=8, n_heads=2, output_dir=str(self.root/name))
            result = train(config, model_name=name)
            self.assertEqual(result['metadata']['training_windows'], 12)
            self.assertTrue((self.root/name/'model.pt').exists())
            json.loads((self.root/name/'metrics.json').read_text())
            with self.assertRaises(FileExistsError):
                train(config, model_name=name)


if __name__ == '__main__':
    unittest.main()
