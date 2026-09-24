User page
=============
Welcome to the CNN-Mamba forest typology user main page!

This page is the practical guide to running the software: what you need, what
the commands do, and what they write where. The description of the model
itself is in the [top-level README](../README.md), and the design rationale is
in [High-level Design](design.md).

## Requirements

- Python 3.10 or newer.
- TensorFlow 2.15 or newer. A GPU is not required to run the tests, but
  training at the default settings assumes one.
- Google Cloud credentials with read access to the public ForTy bucket:
  `gcloud auth application-default login`. Credentials are read from the
  environment; never place a service-account key inside the repository.

```bash
python -m venv .venv && source .venv/bin/activate
make install                       # pip install -e .
# pip install -r requirements.txt  # both tracks, including JAX
```

## Training

```bash
make train
```

which is

```bash
python -m mamba_forest.train \
    --train-shards 128 --val-shards 8 --epochs 40 \
    --output-dir runs/mamba_forest
```

Every epoch reads a fresh random subset of `--train-shards` training shards and
the same `--val-shards` validation shards, so validation metrics are comparable
from epoch to epoch. Each epoch prints the learning rate, the training loss,
pixel accuracy, mean IoU, Overall and Forests F1, and the three forest classes
individually.

The flags worth knowing:

| Flag | Default | Effect |
|---|---|---|
| `--train-shards` | 128 | Training shards per epoch; 1024 is the full split. |
| `--val-shards` | 8 | Validation shards, fixed across epochs. |
| `--batch-size` | 16 | Tiles per step. |
| `--epochs` | 40 | Maximum epochs; early stopping may end the run sooner. |
| `--learning-rate` | 2e-4 | Initial rate of the cosine schedule. |
| `--lr-alpha` | 0.1 | Final rate as a fraction of the initial one. |
| `--weight-decay` | 5e-3 | AdamW weight decay. |
| `--d-model` | 32 | Width after fusion; the main capacity knob. |
| `--dropout` | 0.2 | Dropout throughout the network. |
| `--d-state-temporal` | 8 | State dimension of the temporal Mamba block. |
| `--d-state-spatial` | 16 | State dimension of the spatial Mamba block. |
| `--bottleneck-channels` | 512 | Channel width of the bottleneck. |
| `--combo-alpha` | 0.4 | Weight of cross-entropy against the Dice term. |
| `--label-smoothing` | 0.05 | Smoothing of the cross-entropy term. |
| `--no-class-weights` | off | Uniform class weights instead of the defaults. |
| `--grad-clip` | 1.0 | Global-norm gradient clipping; 0 disables it. |
| `--no-mixed-precision` | off | Train in float32 instead of mixed float16. |
| `--patience` | 8 | Epochs without improvement before stopping. |
| `--seed` | 42 | Seed for TensorFlow, NumPy and the shard shuffling. |

Written to `--output-dir`:

- `config.json` — every resolved flag, so the run can be repeated.
- `training_results.csv` — one row per epoch: loss, pixel accuracy, mean IoU,
  Overall and Forests F1, and the per-class F1 scores.
- `best.weights.h5` — the checkpoint with the best Overall F1.
- `last.weights.h5` — the most recent checkpoint.

## Evaluation

```bash
make eval
```

which is

```bash
python -m mamba_forest.evaluate \
    --weights runs/mamba_forest/best.weights.h5 --test-shards 64
```

This prints pixel accuracy, mean IoU and the benchmark columns — Overall,
Forests, N, P, TC — and writes `test_metrics.json` next to the weights. A
checkpoint stores weights and not the architecture, so the four architecture
flags — `--d-model`, `--d-state-temporal`, `--d-state-spatial` and
`--bottleneck-channels` — must be given the same values they had during
training, or `load_weights` will refuse the checkpoint.

## Prediction on your own tiles

```bash
make predict TILES=my_tiles OUT=predictions
```

Applies the checkpoint to a directory of `.npz` tiles and writes a class map per
tile plus `predictions.csv`. The input contract and the output files are
documented in [Mapping Your Own Region](apply.md).

## Plotting

```bash
python scripts/plot_results.py runs/mamba_forest/training_results.csv \
    --output runs/mamba_forest/training_curves.png
```

Two panels: the training loss, and the validation metrics over epochs.

## Tests and style

```bash
make test          # unit tests, CPU only, no dataset access
make lint          # flake8
make clean         # build artefacts and __pycache__
```

## Troubleshooting

**A permission or credential error on the first batch.** The dataset is read
lazily, so an authentication problem surfaces when the first shard is opened
rather than at start-up. Re-run `gcloud auth application-default login`.

**Out of memory.** Lower `--batch-size` first, then `--d-model`. Leaving mixed
precision on matters here: `--no-mixed-precision` roughly doubles activation
memory.

**Slow epochs.** The input pipeline interleaves and prefetches, so throughput
is usually bound by the accelerator rather than by the network. Lowering
`--train-shards` shortens an epoch without changing anything else.
