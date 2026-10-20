Results
=============

All numbers are F1 scores in percent, computed over pixels and following the
metrics of the benchmark. For one class, *precision* is the share of the pixels
predicted as that class that really belong to it, *recall* is the share of the
class's pixels that are found, and *F1* is their harmonic mean. **Overall F1**
is the macro average over the nine classes, **Forest mean F1** the average over
the three forest types, and the last three columns are the F1 of natural
forest, planted forest and tree crops. In the code and in `test_metrics.json`
the keys are `overall_f1`, `forests_f1`, `natural_forest_f1`,
`planted_forest_f1` and `tree_crops_f1`, stored as fractions.

## Benchmark

| Model | Overall F1 | Forest mean F1 | Natural forest F1 | Planted forest F1 | Tree crops F1 |
|---|---:|---:|---:|---:|---:|
| UNet3D | 32.4 | 24.2 | 56.2 | 7.5 | 8.8 |
| UTAE | 49.4 | 37.7 | 71.4 | 13.8 | 27.8 |
| MTSViT | 81.1 | 74.9 | 82.8 | 62.9 | 78.9 |
| **CNN-Mamba (this repository)** | **56.89** | **48.81** | **64.67** | **27.64** | **54.12** |

The baseline scores are published ForTy results using the full training split.
The historical CNN-Mamba scores used 128 of 1,024 training shards per epoch,
8 validation shards and 64 test shards. Rotating the training subset makes
12.5% a per-epoch read budget, not total unique data usage. Natural-forest F1
is below UTAE (64.67 versus 71.4); the other displayed values are higher.
The test subsets, training budgets and modalities differ, so these numbers
are contextual references rather than a controlled model comparison.

Only this historical table is retained: there are no source logs, checkpoints,
split manifests or objective ablation artifacts to verify the scores. The
commands below run new experiments; they cannot recreate an undocumented
historical run exactly. New shard sampling is explicit and seeded per epoch,
which also changes the training sequence relative to the earlier pipeline.

## Training configuration

| Setting | Value |
|---|---|
| Training shards | 128 of 1,024 (~12.5%) per epoch, resampled |
| Validation shards | 8 of 1,024, fixed across epochs, used to choose the checkpoint |
| Test shards | 64 of 1,024, used only for the reported numbers |
| Batch size | 16 |
| Optimiser | AdamW, weight decay 5e-3 |
| Learning rate | 2e-4, cosine decay to 10% |
| Gradient clipping | global norm 1.0 |
| Loss | 0.4 x weighted cross-entropy + 0.6 x soft Dice |
| Label smoothing | 0.05 |
| Precision | mixed float16 with dynamic loss scaling |

## Reproducing

```bash
make train                 # writes runs/mamba_forest/training_results.csv
make eval                  # writes runs/mamba_forest/test_metrics.json
python scripts/plot_results.py runs/mamba_forest/training_results.csv
```

`training_results.csv` holds one row per epoch with the training loss, pixel
accuracy, mean IoU, Overall and Forests F1 and the per-class F1 scores;
`test_metrics.json` holds the test metrics, confusion matrix, class support,
exact shard paths, model settings and the weights' SHA-256. Training also writes
`split_manifest.json` with the exact paths for each started epoch, allowing
actual cumulative shard coverage to be counted.

## Effect of the objective

Class weighting and Dice loss are intended to help under-represented classes.
Their effect requires retained ablation runs. For a matched comparison, run
both objectives with the same seed, fixed subset, model, precision, batch size,
epoch limit and validation/test paths:

```bash
python -m mamba_forest.train --seed 42 --fixed-train-subset \
    --output-dir runs/combo
python -m mamba_forest.train --seed 42 --fixed-train-subset \
    --combo-alpha 1 --no-class-weights --label-smoothing 0 \
    --output-dir runs/unweighted_ce
python -m mamba_forest.evaluate --weights runs/combo/best.weights.h5
python -m mamba_forest.evaluate --weights runs/unweighted_ce/best.weights.h5
```

Compare manifests and test support before comparing metrics. Checkpoint
selection uses the same validation macro F1 rule in both runs. Repeat with
several paired seeds, report the actual epochs and training time, and evaluate
the full test split (`--test-shards 1024`) for comparison with full-benchmark
results. Isolate one loss setting at a time when attributing its effect.
