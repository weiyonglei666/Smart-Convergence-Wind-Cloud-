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

    def test_weather_selection_keeps_required_wind_channels(self):
        from dynafore_net.data import choose_weather_columns
        frame = pd.DataFrame({f"WS_{index}": [1.0] for index in range(12)})
        frame["WD_100"] = 90.0
        frame["T"] = 20.0
        frame["H"] = 50.0
        frame["P"] = 1000.0
        chosen = choose_weather_columns(frame, "wind")
        self.assertEqual(len(chosen), 10)
        self.assertTrue(any(name.startswith("WS") for name in chosen))
        self.assertTrue(any(name.startswith("WD") for name in chosen))

    def test_cli_prepare_train_evaluate_and_ablation(self):
        from dynafore_net.data import load_station_arrays
        with tempfile.TemporaryDirectory(prefix="dynafore_test_") as tmp:
            root = Path(tmp)
            raw, out = root / "raw", root / "outputs"
            self.make_raw(raw)
            command = [sys.executable, "-m", "dynafore_net", "run", "--raw-dir", str(raw), "--output", str(out),
                       "--lookback", "32", "--future-source", "measured", "--epochs", "1", "--min-epochs", "1", "--batch-size", "128",
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

    def test_pv_prepare_uses_96_step_history(self):
        with tempfile.TemporaryDirectory(prefix="dynafore_pv_") as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            n = 600
            dates = pd.date_range("2026-01-01", periods=n, freq="15min")
            pd.DataFrame({"DATE_TIME": dates, "ACTIVE_POWER": 10 + np.sin(np.arange(n) / 20)}).to_csv(
                raw / "zdata_PVDEMO_pv_power_real_15min.csv", index=False
            )
            weather = pd.DataFrame({"DATE_TIME": dates})
            names = ["GHI", "GTI", "DNI", "DHI", "WS", "WD", "T", "H", "P", "T_BACKBOARD"]
            for i, name in enumerate(names):
                weather[name] = 100 + i + np.sin(np.arange(n) / 31)
            weather.to_csv(raw / "zdata_PVDEMO_qxz_15min.csv", index=False)
            pd.DataFrame({"FARM_ID": ["PVDEMO"], "CAP_AC": [50.]}).to_csv(
                raw / "zbase_photovoltaic_farm_info.csv", index=False
            )
            out = root / "outputs"
            result = subprocess.run(
                [sys.executable, "-m", "dynafore_net", "prepare", "--raw-dir", str(raw), "--output", str(out),
                 "--future-source", "measured"],
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            from dynafore_net.data import load_station_arrays
            x, f, _, _, _, meta = load_station_arrays(out / "data", "PVDEMO")
            self.assertEqual(meta["lookback"], 96)
            self.assertEqual(x.shape[1:], (96, 17))
            self.assertEqual(f.shape[1:], (16, 18))

    def test_nwp_accepts_ecmwf_style_variable_names(self):
        from dynafore_net.nwp import build_nwp_future
        with tempfile.TemporaryDirectory(prefix="dynafore_nwp_alias_") as tmp:
            path = Path(tmp) / "zdata_DEMO_nwp.csv"
            issue = pd.Timestamp("2026-01-01T00:00:00Z")
            valid = pd.date_range("2026-01-01T00:00:00Z", periods=5, freq="3h")
            rows = []
            for index, stamp in enumerate(valid):
                rows.append({"ISSUE_TIME": issue.isoformat(), "DATE_TIME": stamp.isoformat(),
                             "u10": 3.0, "v10": 4.0, "t2m": float(index), "r": 50.0,
                             "sp": 1000.0, "tcc": 0.0, "tp": 0.0, "ssrd": 100.0 * index})
            pd.DataFrame(rows).to_csv(path, index=False)
            index = pd.date_range("2026-01-01 06:00", periods=48, freq="15min")
            calendar = np.zeros((len(index), 6), dtype=np.float32)
            future, names, _ = build_nwp_future(
                index, np.array([index.get_loc(pd.Timestamp("2026-01-01 12:00"))]),
                16, "wind", path, calendar,
            )
            self.assertEqual(future.shape, (1, 16, 18))
            self.assertIn("nwp__WS", names)
            self.assertIn("nwp__GHI", names)
            self.assertFalse(np.isnan(future).any())

    def test_nwp_future_honours_issue_time_and_interpolates(self):
        from dynafore_net.nwp import build_nwp_future
        with tempfile.TemporaryDirectory(prefix="dynafore_nwp_") as tmp:
            path = Path(tmp) / "zdata_DEMO_nwp.csv"
            issue = pd.Timestamp("2026-01-01T00:00:00Z")
            valid = pd.date_range("2026-01-01T00:00:00Z", periods=5, freq="3h")
            rows = []
            for index, stamp in enumerate(valid):
                row = {"ISSUE_TIME": issue.isoformat(), "DATE_TIME": stamp.isoformat(),
                       "WS": 10.0, "WD": 90.0, "T": 0.0 + index, "H": 50.0, "P": 1000.0,
                       "CLOUD": 0.0, "TP": 0.0, "GHI": 100.0 * index, "GTI": 90.0 * index, "DNI": 80.0 * index}
                rows.append(row)
            pd.DataFrame(rows).to_csv(path, index=False)
            index = pd.date_range("2026-01-01 06:00", periods=48, freq="15min")
            calendar = np.zeros((len(index), 6), dtype=np.float32)
            origin = pd.Timestamp("2026-01-01 12:00")
            position = index.get_loc(origin)
            future, names, _ = build_nwp_future(index, np.array([position]), 16, "wind", path, calendar)
            self.assertEqual(future.shape, (1, 16, 18))
            self.assertEqual(len(names), 18)
            self.assertFalse(np.isnan(future).any())
            u = future[0, :, names.index("future__wind_u")]
            v = future[0, :, names.index("future__wind_v")]
            self.assertTrue(np.allclose(u, 0.0, atol=1e-4))
            self.assertTrue(np.allclose(v, 10.0, atol=1e-4))
            t = future[0, :, names.index("nwp__T")]
            self.assertGreater(t[-1], t[0])
            # An origin before the issue must not use that issue.
            before = build_nwp_future(index, np.array([0]), 16, "wind", path, calendar)[0]
            self.assertTrue(np.isnan(before).all())


if __name__ == "__main__":
    unittest.main()
