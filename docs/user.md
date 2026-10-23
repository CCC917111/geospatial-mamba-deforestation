[Documentation](README.md) · [Project overview](../README.md)

# Usage guide

Install the package, run training and evaluation, and apply a checkpoint to
prepared tiles. This guide covers requirements, command options and outputs;
the [design guide](design.md) explains the architecture and preprocessing.

## Requirements

- Python 3.10 to 3.12 (tested with 3.12). A very new interpreter such as 3.14
  has no stable TensorFlow release yet; see [Setup](../README.md#setup).
- TensorFlow 2.15 or newer. A GPU is not required to run the tests, but
  training at the default settings assumes one.
- Google Cloud credentials with read access to the public ForTy bucket:
  `gcloud auth application-default login`. Credentials are read from the
  environment; never place a service-account key inside the repository.

```bash
python -m venv .venv && source .venv/bin/activate
make install                       # pip install -r requirements.txt && pip install -e .
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

Selection is explicit for each zero-based epoch using `seed + epoch`.
Re-reading a dataset for model warmup does not change its shard membership.
Use `--fixed-train-subset` to train on the same seeded subset every epoch;
without it, the 128-shard default is a per-epoch read budget and can cover much
more than 12.5% of the split over a complete run.

The flags worth knowing:

| Flag | Default | Effect |
|---|---|---|
| `--data-root` / `--gcs-root` | ForTy GCS root | Local directory or cloud shard root. |
| `--total-shards` | 1024 | Shards per split; also determines the filename suffix. |
| `--train-shards` | 128 | Training shards per epoch; 1024 is the full split. |
| `--fixed-train-subset` | off | Keep the same seeded training subset for every epoch. |
| `--val-shards` | 8 | Validation shards, fixed across epochs. |
| `--batch-size` | 16 | Tiles per step. |
| `--epochs` | 40 | Maximum epochs; early stopping may end the run sooner. |
| `--learning-rate` | 2e-4 | Initial rate of the cosine schedule. |
| `--lr-alpha` | 0.1 | Final rate as a fraction of the initial one. |
| `--weight-decay` | 5e-3 | AdamW weight decay. |
| `--d-model` | 32 | Width after fusion; the main capacity knob. |
| `--dropout` | 0.2 | Dropout throughout the network. |
| `--d-state-temporal` | 8 | State dimension of the temporal Mamba-inspired selective-SSM block. |
| `--d-state-spatial` | 16 | State dimension of the spatial Mamba-inspired selective-SSM block. |
| `--bottleneck-channels` | 512 | Channel width of the bottleneck. |
| `--combo-alpha` | 0.4 | Weight of cross-entropy against the Dice term. |
| `--label-smoothing` | 0.05 | Smoothing of the cross-entropy term. |
| `--no-class-weights` | off | Uniform class weights instead of the defaults. |
| `--grad-clip` | 1.0 | Global-norm gradient clipping; 0 disables it. |
| `--precision` | auto | float32 on CPU; mixed float16 with a visible GPU. Explicit policies are also accepted. |
| `--no-mixed-precision` | off | Legacy alias for `--precision float32`. |
| `--patience` | 8 | Epochs without improvement before stopping. |
| `--seed` | 42 | Seed for Python, TensorFlow, Keras, NumPy and shard selection. |

Written to `--output-dir`:

- `config.json` — requested flags, the resolved precision and TensorFlow version.
- `checkpoint_selection.json` — exact validation score, epoch and SHA-256 of
  the best checkpoint; CSV scores retain full precision for selection checks.
- `split_manifest.json` — exact validation paths and paths for every started
  training epoch, completion flags, and completed unique shard coverage.
- `training_results.csv` — one row per epoch: loss, pixel accuracy, mean IoU,
  Overall and Forests F1, per-class F1, training/validation seconds, training
  examples and training tiles per second. Timings include input reads and graph
  compilation, and exclude checkpoint writes.
- `best.weights.h5` — the checkpoint with the best Overall F1.
- `last.weights.h5` — the most recent checkpoint.

Use a fresh `--output-dir` for each training run. A directory containing
`*.weights.h5` is rejected before configuration writes or dataset access,
keeping an existing checkpoint paired with its original configuration.
Training does not resume from those weights automatically.

## Local data and offline checks

Once dependencies are installed, code changes, tests, local data preflight,
training and evaluation do not need a network connection. Run `make test` for
synthetic tensors and temporary local TFRecords; it does not read the bucket.
Real-data accuracy still requires real records.

Training, evaluation and preflight accept `--data-root` as an alias for
`--gcs-root`. For a small local dataset, set `--total-shards` to match the
filename suffix in every split. A four-shard split uses names such as
`forty_v1-train.tfrecord-00000-of-00004`; all requested split budgets must be
between 1 and that total. The default remains 1,024 for official ForTy files.
The JSON config key remains `gcs_root` for compatibility, even for local paths.

For an existing local four-shard dataset:

```bash
python -m mamba_forest.validate_data --data-root ./local_forty \
    --total-shards 4 --train-shards 2 --val-shards 1 --test-shards 1 \
    --output runs/offline/preflight.json

python -m mamba_forest.train --data-root ./local_forty \
    --total-shards 4 --train-shards 2 --val-shards 1 --batch-size 1 \
    --epochs 2 --patience 0 --fixed-train-subset --precision float32 \
    --d-model 8 --d-state-temporal 4 --d-state-spatial 4 \
    --bottleneck-channels 32 --output-dir runs/offline

python -m mamba_forest.evaluate --data-root ./local_forty \
    --total-shards 4 --test-shards 1 --batch-size 1 \
    --weights runs/offline/best.weights.h5
```

`--precision auto` selects float32 when no GPU is visible and mixed float16
otherwise. `config.json` records both the requested and actual policy; loss
scaling is enabled only for mixed float16. An explicit policy pins controlled
experiments. `--no-mixed-precision` still selects float32 and conflicts with
explicit `--precision mixed_float16`.

## Evaluation

```bash
make eval
```

which is

```bash
python -m mamba_forest.evaluate \
    --weights runs/mamba_forest/best.weights.h5 --test-shards 64
```

This prints pixel accuracy, mean IoU and the benchmark columns — `Overall`
(macro F1 over the nine classes), `Forests` (mean F1 of the three forest
types) and `N`, `P`, `TC` (the F1 of natural forest, planted forest and tree
crops) — and writes `test_metrics.json` next to the weights. Evaluation and
prediction read the five model settings (`d_model`, `dropout`,
`d_state_temporal`, `d_state_spatial`, `bottleneck_channels`) from `config.json`
beside the weights. Explicit architecture flags override saved values; missing
settings fall back to the original model defaults.

When moving a checkpoint, keep its training configuration beside it or pass
`--config /path/to/training/config.json`. `--no-config` disables automatic
discovery and uses explicit flags plus defaults. It cannot be combined with
`--config`. A missing automatically discovered file is allowed; a missing
explicit file, malformed JSON or an invalid effective model setting is an
error before data reads. The resolved architecture must still match the
weights. Automatic loading does not change data roots, split budgets or
inference precision; those remain controlled by the current command/runtime.

The JSON also records exact test shard paths, checkpoint SHA-256, model and
runtime settings, the training-config path when used, pixel count, per-class
support and the confusion matrix.
Mean IoU excludes classes absent from both truth and prediction; macro F1
includes all nine classes and assigns zero to absent classes. Validation and
test use the first N shards without augmentation. An empty split is an error.

## Measuring compute performance

For a local demonstration and timing that includes reading tiles and writing
predictions, see [Local demo and end-to-end timing](local-demo.md).

For actual source validation, paired objective experiments and full-test
evaluation, follow the [GPU validation runbook](gpu-validation.md). Start with
`python -m mamba_forest.validate_data --require-gpu --output preflight.json`;
this rejects a CPU-only runtime before opening any records and samples one
record from each selected split once a GPU is available. A passing preflight
does not establish whole-dataset integrity or geographic independence.

```bash
python -m mamba_forest.benchmark --mode both --tile-size 32 \
    --batch-size 1 --d-model 8 --bottleneck-channels 32 \
    --steps 10 --output results/performance_cpu.json
```

The benchmark compares eager and graph inference on one unchanged model, and
measures training on a separate model. Warmup and tracing are excluded, every
step is synchronized, and JSON retains raw latencies, median, p95, throughput,
input shapes, precision and device settings. Synthetic in-memory compute
excludes data loading and file writes. Use a different output path to preserve
the checked-in reference measurement. See the
[audit](optimization-audit.md) for measured results and limits.

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

**Out of memory.** Lower `--batch-size` first, then `--d-model`. On a GPU,
mixed float16 can reduce activation memory; savings depend on the model and
runtime. Auto precision uses float32 on CPU.

**Slow epochs.** Inspect the recorded training and validation times. Input IO,
preprocessing and model compute can each dominate depending on the environment.
Lowering `--train-shards` reduces the per-epoch read budget.
