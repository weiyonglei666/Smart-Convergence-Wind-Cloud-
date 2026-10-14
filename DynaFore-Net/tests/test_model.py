"""CPU checks using generated tensors, not station data."""
import importlib.util
import unittest

import numpy as np
import torch


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("dynafore_net"), "Standalone package not created yet")
        from dynafore_net.model import DynaForeNet
        from dynafore_net.augmentation import FeatureAwareAugment
        self.model_class, self.augment_class = DynaForeNet, FeatureAwareAugment
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.stats = {"x_mean": np.zeros(17, np.float32), "x_std": np.ones(17, np.float32),
                      "f_mean": np.zeros(18, np.float32), "f_std": np.ones(18, np.float32)}
        self.stats["x_mean"][0] = 50.0
        self.cfg = {"d_model": 32, "layers": 1, "dropout": .1, "capacity_mw": 100.0,
                    "future_columns": ["nwp__pl925__u", "nwp__pl925__v", "weather__GHI", "weather__GTI"] + [f"f{i}" for i in range(14)]}
        self.x = torch.randn(2, 288, 17)
        self.future = torch.randn(2, 16, 18)

    def test_wind_decodes_station_capacity_inside_output_head(self):
        model = self.model_class(17, 18, 288, 16, self.stats, self.cfg).eval()
        with torch.no_grad():
            pred = model(self.x, self.future)
        expected = (self.x[:, -1, 0] + 50) / 100
        torch.testing.assert_close(pred, expected[:, None].expand(-1, 16))
        self.assertEqual(tuple(pred.shape), (2, 16))
        self.assertTrue(((pred >= 0) & (pred <= 1)).all())

    def test_pv_night_mask_is_in_forward_not_postprocessing(self):
        model = self.model_class(17, 18, 288, 16, self.stats, self.cfg, {"energy_type": "pv"}).eval()
        self.future[..., 2:4] = 0
        self.future[:, 8:, 2:4] = 600
        pred = model(self.x, self.future)
        self.assertTrue((pred[:, :8] == 0).all())
        self.assertTrue((pred[:, 8:] > 0).all())
        self.assertTrue(((pred >= 0) & (pred <= 1)).all())

    def test_autoda_receives_gradients_jointly_with_predictor(self):
        model = self.model_class(17, 18, 288, 16, self.stats, self.cfg)
        augment = self.augment_class(17, self.stats, power_scale=100)
        xa, ya, reg = augment(self.x, torch.full((2, 16), .4))
        loss = (model(xa, self.future) - ya).square().mean() + reg
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in augment.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))
        self.assertEqual(tuple(xa.shape), (2, 288, 17))

    def test_ablation_flags_and_autoda_bypass(self):
        from dynafore_net.training import dynafore_flags, apply_training_augmentation
        flags = dynafore_flags("noPMD-PVA-TRA", "wind")
        self.assertFalse(flags["pmd"] or flags["pva"] or flags["tra"])
        self.assertFalse(dynafore_flags("noNSAN", "pv")["nsan"])
        self.assertFalse(dynafore_flags("noguided correction", "wind")["guided_correction"])
        y = torch.zeros(2, 16)
        xout, yout, reg = apply_training_augmentation(self.x, y, None, "noAutoDA")
        self.assertIs(xout, self.x)
        self.assertIs(yout, y)
        self.assertEqual(float(reg), 0)

    def test_invalid_horizon_and_lookback_are_rejected(self):
        with self.assertRaises(ValueError):
            self.model_class(17, 18, 288, 12, self.stats, self.cfg)
        with self.assertRaises(ValueError):
            self.model_class(17, 18, 290, 16, self.stats, self.cfg)


if __name__ == "__main__":
    unittest.main()
