[Documentation](../docs/README.md) · [Project overview](../README.md)

# Results and evidence

Historical benchmark scores, retained CPU measurements and local workflow
checks are recorded separately below.

[Historical scores](#benchmark) · [Compute diagnostics](#retained-compute-measurements) ·
[File-to-file timing](#local-demo-and-file-to-file-prediction-timing)

The historical accuracy table reports F1 scores in percent, computed over
pixels using the benchmark metrics. For one class, *precision* is the share of
pixels predicted as that class that really belong to it, *recall* is the share
of the class's pixels that are found, and *F1* is their harmonic mean. **Overall F1**
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

<a id="reproducing"></a>

## Run new experiments

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

## Effect of the selective state-space blocks

The architecture mode pairs `cnn_ssm` and `cnn_only` with the same ComboLoss,
fixed split memberships, seeds and training budgets:

```bash
python -m mamba_forest.experiment --comparison architecture \
  --run-root runs/architecture-pair
# Review plan.json, then execute on the prepared real-data GPU runtime.
python -m mamba_forest.experiment --comparison architecture \
  --run-root runs/architecture-pair --execute
python -m mamba_forest.compare --comparison architecture \
  --run-root runs/architecture-pair
```

CNN-only uses optical temporal mean pooling and removes both SSM blocks while
retaining normalization, fusion, CNN stages and local SASF. Parameter counts
differ. This is a component ablation under a shared training budget; it does
not isolate recurrence from parameter count or loss of optical temporal order.
The [controlled protocol](../docs/gpu-validation.md#controlled-architecture-comparison)
describes the controls, metrics and remaining geographic validation.
No real-data architecture ranking is established.

## Effect of the objective

Class weighting and Dice loss are intended to help under-represented classes.
Their effect requires retained ablation runs. For a matched comparison, run
both objectives with the same seed, fixed subset, model, precision, batch size,
epoch limit and validation/test paths:

```bash
python -m mamba_forest.experiment --run-root runs/objective-pair
# Review plan.json, then run the same settings on the prepared GPU environment.
python -m mamba_forest.experiment --run-root runs/objective-pair --execute
python -m mamba_forest.compare --run-root runs/objective-pair
```

The default plan uses three seeds, fixed 128/8 training/validation shards,
40 epochs with early stopping disabled, and all 1,024 test shards. All training
finishes before evaluation starts. The comparator checks manifests, example
budgets, test support, model settings, checkpoint hashes, exact best validation
epochs and confusion-derived scores before reporting paired differences and
timings. Plain CE changes Dice, weighting and smoothing together; isolate one
loss setting at a time when attributing its effect.

The [GPU validation runbook](../docs/gpu-validation.md) expands this into a
real-record preflight, smoke run, paired three-seed protocol and full-test
evaluation. These experiments have not run in the current CPU environment.

## Retained compute measurements

- `performance_cpu.json`: reduced 32×32 model, ten measured steps per mode.
- `performance_cpu_128.json`: default 128×128 model, three measured inference
  steps after two warmups; eager median 316.54 ms, graph median 149.01 ms.
- `forty_gpu_preflight.json`: the GPU requirement failed on this CPU runtime;
  no source records were opened and no real-data validation result is implied.
- [`architecture_pipeline_smoke.json`](architecture_pipeline_smoke.json): both architecture
  arms completed two CPU epochs on four synthetic training and two validation
  examples, then evaluated the same two test tiles and passed paired artifact
  checks. CNN + SSM has 3,311,968 parameters; CNN-only has 3,309,724 in this
  reduced configuration. This verifies workflow integration, not real-data
  accuracy or a model ranking.
- `paired_pipeline_smoke.json`: both objectives completed two CPU epochs on
  one synthetic 128×128 tile per split, reloaded saved configurations and
  passed all seven workflow stages and paired artifact checks. Its coverage
  refers to the one-shard synthetic fixture. No real-data score is reported.

`performance_cpu.json` and `performance_cpu_128.json` contain synchronized
CPU compute timings and configuration. They exclude source IO and measure
fixed-shape model compute. The file-to-file benchmark below uses a different
timing boundary. None of these CPU records establishes GPU performance or
real-data accuracy.

## Local demo and file-to-file prediction timing

The [local demo record](local_demo_cpu.json) retains a completed CPU run of
the real training, evaluation and prediction entry points. Generated data use
the standard 128 × 128 record schema, with four training examples, two
validation examples and two test examples. The reduced architecture uses
`d_model=4`, temporal/spatial state dimensions 2/2, bottleneck width 8 and
dropout 0, trained for two epochs with batch size 2 and float32 computation.
The summary identifies the selected checkpoint and the source modules by hash.
The [figure](../docs/figures/local-demo-synthetic.png) displays its actual toy
predictions, which remain nearly uniform at this short training budget.

The [prediction timing record](performance_predict_cpu.json) uses this same
checkpoint and the two synthetic NPZ test tiles, with class-map NPY, PNG and
per-pass CSV output enabled. After one warmup pass, 30 measured passes produced
60 steady-state tile samples. Both TensorFlow thread settings were 2.

| Per-tile stage | Median |
|---|---:|
| NPZ read and validation | 12.16 ms |
| Normalization and tensor preparation | 3.01 ms |
| Graph inference and actual output synchronization | 115.88 ms |
| Class shares and colour-map preparation | 0.36 ms |
| NPY and PNG export | 1.00 ms |
| Per-tile stage total | 133.38 ms |

The per-tile total is calculated before taking the median; it excludes CSV
and initialization. Full measured-pass wall times
include directory creation, CSV writes and close/flush; their combined
throughput was **7.20 tiles/s**. Runtime import took 2.39 seconds separately;
the first prediction tensors were available 1.88 seconds after entering the
prediction function, including model setup and first graph tracing. Process
startup and command-line parsing remain outside these timing boundaries.

Input files are reread on every pass, but operating-system caches are not
cleared. File writes are closed/flushed to the OS cache without `fsync`.
These results apply to the reduced demo architecture and selected exports;
they are not directly comparable with the default-model compute-only timings.
They establish neither real-data accuracy nor performance on a GPU.

The [local guide](../docs/local-demo.md) provides commands to regenerate the
demo and repeat the benchmark on its checkpoint or on another trained model.
