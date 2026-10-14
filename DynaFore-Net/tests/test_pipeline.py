import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("dynafore_net"), "Standalone package not created yet")

    def test_metrics_use_capacity_and_weighted_acc(self):
        from dynafore_net.metrics import metrics
        y = np.array([[0., 100.] * 8])
        got = metrics(y, np.zeros_like(y), 100.)
        self.assertEqual(got["MSE@16"], 5000.)
        self.assertEqual(got["MAE@16"], 50.)
        self.assertEqual(got["nMAE@16"], .5)
        self.assertAlmostEqual(got["nRMSE@16"], 2 ** -.5)
        self.assertEqual(got["ACC@16"], 0.)
        self.assertEqual(metrics(y, y, 100.)["ACC@16"], 100.)
        with self.assertRaises(ValueError):
            metrics(y, y, 0.)

    def test_iso_and_dayfirst_dates_keep_the_intended_calendar_day(self):
        from dynafore_net.data import read_timeseries
        with tempfile.TemporaryDirectory(prefix="dynafore_date_") as tmp:
            path = Path(tmp) / "times.csv"
            for text in ["2026-04-05 12:00:00", "05/04/2026 12:00:00"]:
                pd.DataFrame({"DATE_TIME": [text], "ACTIVE_POWER": [1.]}).to_csv(path, index=False)
                got = read_timeseries(path)
                self.assertEqual(got.index[0], pd.Timestamp("2026-04-05 12:00:00"))

    def test_weather_gaps_are_not_backfilled_from_future(self):
        from dynafore_net.data import load_station_raw
        with tempfile.TemporaryDirectory(prefix="dynafore_gap_") as tmp:
            raw = Path(tmp) / "raw"
            self.make_raw(raw)
            weather_path = raw / "zdata_DEMO_wind_tower_15min.csv"
            weather = pd.read_csv(weather_path)
            weather.loc[0, "WS_HUBHEIGHT"] = np.nan
            weather.loc[20:24, "WS_HUBHEIGHT"] = np.nan
            weather.loc[25, "WS_HUBHEIGHT"] = 99.
            before = weather.loc[19, "WS_HUBHEIGHT"]
            weather.to_csv(weather_path, index=False)
            base, _, _, _ = load_station_raw("DEMO", "wind", raw / "zdata_DEMO_wind_power_real_15min.csv", raw)
            self.assertTrue(pd.isna(base.iloc[0]["weather__WS_HUBHEIGHT"]))
            self.assertEqual(base.iloc[20]["weather__WS_HUBHEIGHT"], before)
            self.assertTrue(pd.isna(base.iloc[24]["weather__WS_HUBHEIGHT"]))
            self.assertEqual(base.iloc[25]["weather__WS_HUBHEIGHT"], 99.)

    @staticmethod
    def make_raw(path):
        """Runtime-only synthetic fixture; never shipped as a dataset."""
        path.mkdir()
        n = 700
        dates = pd.date_range("2026-01-01", periods=n, freq="15min")
        power = 50 + 10 * np.sin(np.arange(n) / 20)
        pd.DataFrame({"DATE_TIME": dates, "ACTIVE_POWER": power}).to_csv(path / "zdata_DEMO_wind_power_real_15min.csv", index=False)
        weather = pd.DataFrame({"DATE_TIME": dates})
        names = ["WS_HUBHEIGHT", "WS_100", "WS_140", "WS_150", "WS_10", "WD_HUBHEIGHT", "WD_100", "T", "H", "P"]
        for i, name in enumerate(names):
            weather[name] = 10 + i + np.sin(np.arange(n) / 31)
        weather.to_csv(path / "zdata_DEMO_wind_tower_15min.csv", index=False)
        pd.DataFrame({"FARM_ID": ["DEMO"], "CAP": [100.]}).to_csv(path / "zbase_wind_farm_info_encryption.csv", index=False)

    def test_cli_prepare_train_evaluate_and_ablation(self):
        from dynafore_net.data import load_station_arrays
        with tempfile.TemporaryDirectory(prefix="dynafore_test_") as tmp:
            root = Path(tmp)
            raw, out = root / "raw", root / "outputs"
            self.make_raw(raw)
            command = [sys.executable, "-m", "dynafore_net", "run", "--raw-dir", str(raw), "--output", str(out),
                       "--lookback", "32", "--epochs", "1", "--min-epochs", "1", "--batch-size", "128",
                       "--device", "cpu", "--variants", "full", "noAutoDA"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            x, f, y, origins, split, meta = load_station_arrays(out / "data", "DEMO")
            self.assertEqual(x.shape[1:], (32, 17))
            self.assertEqual(f.shape[1:], (16, 18))
            self.assertEqual(meta["future_weather_source"], "measured_future")
            dates = pd.to_datetime(origins)
            for current, nxt in [("train", "val"), ("val", "test")]:
                self.assertLess(dates[split == current].max() + pd.Timedelta(minutes=225), dates[split == nxt].min())
            run = out / "runs" / "DEMO" / "full"
            pred_before = np.load(run / "prediction_test.npy")
            evaluate = subprocess.run([sys.executable, "-m", "dynafore_net", "evaluate", "--output", str(out), "--device", "cpu"], capture_output=True, text=True, timeout=60)
            self.assertEqual(evaluate.returncode, 0, evaluate.stdout + evaluate.stderr)
            np.testing.assert_allclose(np.load(run / "prediction_test.npy"), pred_before, atol=1e-5)
            full = json.loads((run / "metrics_test.json").read_text())
            ablation = json.loads((out / "runs" / "DEMO" / "noAutoDA" / "metrics_test.json").read_text())
            self.assertTrue(full["auto_da_used"])
            self.assertFalse(ablation["auto_da_used"])
            self.assertTrue((out / "results_by_station.csv").exists())


if __name__ == "__main__":
    unittest.main()
