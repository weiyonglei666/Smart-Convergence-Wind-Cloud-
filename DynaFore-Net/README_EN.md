# DynaFore-Net

A standalone extraction of the current station-wise, 16-step wind/PV forecasting implementation. Includes the model, feature-aware training augmentation, virtual fine-phase reconstruction, physical output constraints, training, evaluation, and four ablations.

**No datasets, prepared arrays, checkpoints, or station predictions are included. For dataset access, please contact [ts26170007a31@cumt.edu.cn](mailto:ts26170007a31@cumt.edu.cn).**

## Quick start

Python >= 3.10; install an appropriate PyTorch build for your device.

```bash
python -m pip install -e .
dynafore run --raw-dir /path/to/private_csvs --output outputs --device cpu
```

For raw CSV filenames, columns, and capacity metadata, see [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md). Paths are supplied at runtime; no private server path is required.

```bash
dynafore run --raw-dir /path/to/private_csvs --output outputs_ablations \
  --variants full noPMD-PVA-TRA noNSAN noAutoDA "noguided correction"
dynafore evaluate --output outputs
python -m unittest discover -s tests -v
python scripts/check_release.py
```

Each station is trained independently. Defaults: 288 historical samples, 16 forecast steps at 15-minute resolution, chronological 70/15/15 splits with target-overlap purge, train-only scaling, up to 20 epochs, validation ACC early stopping, and ReduceLROnPlateau. Reports include MSE, MAE, nMAE, nRMSE, ACC and per-station first-place counts.

**Weather protocol:** this version preserves the experiment's *future measured weather* input. It is not operationally available NWP. Historical missing weather is forward-filled for at most four steps. See [implementation notes](docs/IMPLEMENTATION.md) for the exact augmentation, metrics, ablations, and extraction changes.

## Terms

At the project owner's direction, the original research-use terms are retained in [LICENSE](LICENSE). No MIT/OSI-approved open-source license is asserted. Dataset access does not itself grant redistribution rights. Contact **ts26170007a31@cumt.edu.cn** for data and authorization enquiries.
