[Documentation](README.md) · [Project overview](../README.md)

# Real-data and GPU validation runbook

This runbook prepares new experiments on the current implementation. It does
not reproduce the historical score table or implement the published UNet3D,
UTAE or MTSViT baselines. Two paired modes address separate questions: `architecture` compares CNN-only
with CNN + selective SSM under the same ComboLoss, while the default
`objective` mode compares ComboLoss with plain cross-entropy on CNN + SSM.

**Execution status, 2026-10-26:** the full-budget, multi-seed real-data
protocols below remain pending. The local workspace is CPU-only and Google
Cloud Storage access is blocked. A separate
[Tesla T4 synthetic comparison](performance.md#retained-tesla-t4-comparison)
completed in Colab and retains compute timing and allocator memory. Those
measurements do not establish real-data accuracy or dataset compatibility.

## Controlled architecture comparison

Generate a plan without reading data or importing TensorFlow:

```bash
python -m mamba_forest.experiment --comparison architecture \
  --run-root runs/architecture-pair
```

The default plan pairs `cnn_ssm` and `cnn_only` for seeds 42, 43 and 44,
40 epochs each, batch size one, fixed 128/8 train/validation shards and all
1,024 test shards. Both arms use the same ComboLoss, normalization,
augmentation protocol, optimizer, widths, precision and validation selection
rule. All training completes and its artifacts pass checks before test
evaluation starts. Review `plan.json`, then use the same settings to execute:

```bash
python -m mamba_forest.experiment --comparison architecture \
  --run-root runs/architecture-pair --execute
python -m mamba_forest.compare --comparison architecture \
  --run-root runs/architecture-pair
```

CNN-only averages optical observations over seasons and removes both selective
SSM blocks. It retains normalization, dynamic modality fusion, the CNN
encoder/decoder and local SASF. This removes optical season ordering as well
as learned SSM processing. **Parameter counts differ**, so this is a component
ablation under a common training budget, not a capacity-matched comparison or
an isolated test of the recurrence alone. Climate still uses its ordered MLP
input in both arms. Evaluation records each model's parameter count.

The comparator checks arm identities, matching objectives, manifests, example
budgets, evaluation support and checkpoint hashes. Its summary includes forest
F1, mean IoU, paired percentage-point changes and sample standard deviation
across seeds. One seed has no sample standard deviation. Epoch and test-process
wall times are diagnostic; they are not steady-state inference latency or GPU
peak memory. Use the compute/prediction benchmarks for a defined latency
boundary and retain actual hardware memory measurements separately.

Use `--data-root /path/to/forty` for a local mirror. Preflight inspects only
one record per selected split. Matching manifests do not establish geographic
independence: inspect official split provenance and tile IDs/coordinates
before claiming spatially independent evaluation. Test outcomes must not guide
model or hyperparameter selection.

## Automated paired objective experiment

The experiment CLI implements the primary comparison below. First generate
the commands and settings without importing TensorFlow or reading the source:

```bash
python -m mamba_forest.experiment --run-root runs/objective-pair
```

Review `runs/objective-pair/plan.json`, then execute with the same flags:

```bash
python -m mamba_forest.experiment --run-root runs/objective-pair --execute
```

The default plan uses seeds 42, 43 and 44, 128 fixed training shards, eight
validation shards, 40 epochs, batch size one, mixed float16 training and all
1,024 test shards. Each seed trains ComboLoss and plain cross-entropy. A strict
GPU operation probe runs before source access; each seed is preflighted before
training. All six training jobs must finish before test evaluation starts.
Evaluation processes use float32 and restore architecture from each run's
configuration.

Use `--data-root /path/to/forty` for a local mirror. Flags also control shard
counts, seeds, epochs, batch size, precision and architecture; `--help` lists
them. Choose a fresh run root for every execution. A failed execution retains
its status, commands and logs and cannot silently restart or overwrite earlier
checkpoints. Plan generation can be repeated with identical settings.

The runner saves source and package metadata, subprocess logs and wall times.
After evaluation it runs the comparator automatically. Recompute a summary
from retained artifacts with:

```bash
python -m mamba_forest.compare --run-root runs/objective-pair
```

The comparator requires complete, matching fixed-split training budgets,
expected objectives, identical test memberships and class support, consistent
model/runtime settings, and the SHA-256 of each actual best checkpoint. It
recomputes metrics from confusion matrices and verifies checkpoint selection
against full-precision training scores and `checkpoint_selection.json`.
The checkpoint hash saved during selection must match both the evaluated
checkpoint hash and the current weight file.
`comparison.json` contains per-seed scores, paired percentage-point changes,
their mean and sample standard deviation, and training/evaluation timing.
One seed produces no sample standard deviation.

`--allow-cpu` enables small offline pipeline checks and selects float32 by
default. Synthetic inputs verify this workflow only; their scores must remain
labeled synthetic and separate from real-data results. The manual steps below
cover environment preparation, calibration and the same experiment protocol.

## Prepare an appropriate execution environment

The [cloud setup guide](cloud-environment.md) covers preparing a fresh CPU or
GPU runtime and its network/identity requirements.
The [free Colab guide](colab-free.md) provides a smaller pipeline check when
a free GPU is available; it does not assume capacity for this full protocol.

Use a GPU-enabled environment with a compatible TensorFlow installation,
driver and CUDA libraries, and authorized read access to the ForTy source.
Allow the required Google Cloud Storage destinations through the supported
network configuration. Use the environment's configured identity or existing
application-default credentials; never place credentials in this repository
or save their contents in experiment artifacts. Installing GPU libraries alone
does not provide GPU hardware or cloud access.

Run from the repository root in Bash. Keep one environment, package version,
GPU, model configuration and precision policy across the comparison. A fresh
run directory prevents accidental reuse or overwriting of earlier artifacts:

```bash
set -euo pipefail
export RUN_ROOT="runs/gpu-validation-$(date -u +%Y%m%dT%H%M%SZ)"
export FORTY_ROOT="gs://forest_typology/forty_v1/1.0.0"
export GPU_BATCH_SIZE=1
mkdir -p "$RUN_ROOT"
git rev-parse HEAD > "$RUN_ROOT/source_commit.txt"
git status --short > "$RUN_ROOT/source_worktree_status.txt"
python -m pip freeze > "$RUN_ROOT/packages.txt"
```

The package must be installed in this interpreter. TensorFlow CPU 2.21.0 and
Python 3.12.14 were used for the existing local checks; GPU compatibility still
requires validation in the chosen runtime. Resolve and retain the GPU
environment's package versions before starting the six comparison runs.

Check that TensorFlow sees and can execute on a GPU before scheduling expensive
jobs. Device placement is enforced for this small probe, preventing a silent
CPU fallback:

```bash
python - <<'PY' | tee "$RUN_ROOT/gpu_preflight.txt"
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
assert gpus, "No TensorFlow GPU is available; stop before training"
tf.config.set_soft_device_placement(False)
with tf.device("/GPU:0"):
    product = tf.matmul(tf.ones((2, 2)), tf.ones((2, 2)))
assert "GPU:0" in product.device, product.device
product.numpy()
print("TensorFlow:", tf.__version__)
print("Visible GPUs:", gpus)
print("Probe device:", product.device)
PY
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv \
    > "$RUN_ROOT/gpu_hardware.csv"
```

## Validate actual source records before training

Use the preflight CLI to select and record the intended split memberships,
then read the first **unaugmented** record of the first selected shard in each
split. The GPU requirement fails before source IO when no GPU is visible.
Check the real schema, shapes, dtypes, finite modalities, label range and
encoding. Stop if any check fails; fix the parser against the source
specification before proceeding.

Each invocation reads only three examples. It does not establish whole-dataset
integrity, class balance, independent geographic blocks or representative test
coverage. The training parser does not retain tile IDs or coordinates;
preflight reports metadata fields when present in the raw record but does not
verify global uniqueness or geographic separation. Test access here is a
software-contract check and must not guide model or hyperparameter selection.

```bash
python -m mamba_forest.validate_data \
  --gcs-root "$FORTY_ROOT" --train-shards 128 --val-shards 8 \
  --test-shards 1024 --seed 42 --require-gpu \
  --output "$RUN_ROOT/source_preflight.json" \
  2>&1 | tee "$RUN_ROOT/source_preflight.log"

python -m mamba_forest.validate_data \
  --gcs-root "$FORTY_ROOT" --train-shards 1 --val-shards 1 \
  --test-shards 1 --seed 42 --require-gpu \
  --output "$RUN_ROOT/smoke_source_preflight.json" \
  2>&1 | tee "$RUN_ROOT/smoke_source_preflight.log"
```

The second invocation checks the tiny run's memberships separately because
sampled shard order can depend on the requested subset size. Retain both
reports, including selected paths, observed counts and label IDs. A report
listing 1,024 test paths means those paths were selected; only one test record
was inspected by preflight. The formal loop below also preflights each seed's
first selected training record before that pair starts.

## Run a two-epoch smoke experiment and reload its checkpoint

Keep the default full model and 128×128 source tiles, but use one training and
one validation shard, batch size one and two epochs. This is a pipeline and
checkpoint test, not an accuracy experiment. Mixed float16 is the training
default; inspect the logs for non-finite losses or GPU execution failures.

```bash
smoke_dir="$RUN_ROOT/smoke_batch${GPU_BATCH_SIZE}"
mkdir -p "$smoke_dir"
/usr/bin/time -f '%e' -o "$smoke_dir/train_wall_seconds.txt" \
  python -m mamba_forest.train \
    --gcs-root "$FORTY_ROOT" --seed 42 --fixed-train-subset \
    --precision mixed_float16 \
    --train-shards 1 --val-shards 1 --batch-size "$GPU_BATCH_SIZE" \
    --epochs 2 --patience 0 \
    --d-model 32 --d-state-temporal 8 --d-state-spatial 16 \
    --bottleneck-channels 512 --dropout 0.2 \
    --output-dir "$smoke_dir" \
    2>&1 | tee "$smoke_dir/train.log"

/usr/bin/time -f '%e' -o "$smoke_dir/test_wall_seconds.txt" \
  python -m mamba_forest.evaluate \
    --gcs-root "$FORTY_ROOT" \
    --weights "$smoke_dir/best.weights.h5" \
    --test-shards 1 --batch-size "$GPU_BATCH_SIZE" \
    --d-model 32 --d-state-temporal 8 --d-state-spatial 16 \
    --bottleneck-channels 512 --dropout 0.2 \
    --output-dir "$smoke_dir/test" \
    2>&1 | tee "$smoke_dir/evaluate.log"
```

Verify two completed manifest entries with identical training paths, finite
loss/timing values, positive example counts, both checkpoint files and a test
confusion matrix whose total equals the reported pixel count. Smoke test
metrics are not evidence for a model comparison and must not be used to tune
the formal experiments.

## Measure 128×128 GPU compute and calibrate batch size

Measure inference in float32, matching a fresh evaluation/prediction CLI
process, and training in explicitly selected mixed float16. Auto precision
also selects mixed float16 on a visible GPU. These
commands use the full default model. Each mode excludes warmup and records
30 synchronized samples:

```bash
python -m mamba_forest.benchmark \
  --mode inference --inference-mode both --precision float32 \
  --tile-size 128 --batch-size "$GPU_BATCH_SIZE" --warmup 5 --steps 30 \
  --d-model 32 --d-state-temporal 8 --d-state-spatial 16 \
  --bottleneck-channels 512 --dropout 0.2 --device /GPU:0 \
  --output "$RUN_ROOT/gpu_inference_128_batch${GPU_BATCH_SIZE}.json"

python -m mamba_forest.benchmark \
  --mode training --precision mixed_float16 \
  --tile-size 128 --batch-size "$GPU_BATCH_SIZE" --warmup 5 --steps 30 \
  --d-model 32 --d-state-temporal 8 --d-state-spatial 16 \
  --bottleneck-channels 512 --dropout 0.2 --device /GPU:0 \
  --output "$RUN_ROOT/gpu_training_128_batch${GPU_BATCH_SIZE}.json"
```

These are synthetic model-compute measurements. They exclude source reads,
preprocessing, checkpoint loading and file output. Inference uses a fixed-shape
graph and a scalar synchronization output; it does not time the production
predictor's dynamic signature or class-map transfer. Check the recorded input
device to ensure `/GPU:0` was used; a requested device alone is insufficient.

Start with batch one. If the selected hardware allows a larger common batch,
set `GPU_BATCH_SIZE` before the formal runs and repeat both benchmarks and the
tiny training smoke with that batch. Keep every calibration artifact. Do not
change batch size, architecture or precision between comparison arms or seeds;
if a formal run fails for memory, restart all affected comparisons under one
consistent configuration. No hardware-independent safe batch or memory/time
estimate is assumed.

The smoke's `training_results.csv` and all formal training CSVs provide
IO-inclusive epoch timing. `train_seconds` includes source reads,
preprocessing and graph compilation; `validation_seconds` includes evaluation
reads and tracing. They exclude checkpoint writes. First and later epochs
should be shown separately because compilation and caches differ. External
wall timers include process startup, reads, computation and checkpoint/JSON
writes. Do not subtract the synthetic compute benchmark from these timings and
call the difference isolated IO latency.

## Run the primary paired objective comparison

Run seeds 42, 43 and 44 sequentially on the same hardware. Each pair reuses
exactly 128 seeded training shards, the same leading eight validation shards,
40 epochs, architecture, batch and precision. Disable early stopping so each
arm receives the same epoch budget; choose its checkpoint using validation
macro F1 only. Keep the test set out of training and checkpoint selection.

The contrast below changes the combined objective: plain cross-entropy has
no Dice term, class weighting or label smoothing. It cannot attribute a
difference specifically to Dice. An optional third arm with only
`--combo-alpha 1` retains weighting and smoothing and isolates the Dice term.
That third arm requires a separate paired run and must be labeled separately.

```bash
printf '%s\n' "$GPU_BATCH_SIZE" > "$RUN_ROOT/formal_batch_size.txt"
for seed in 42 43 44; do
  python -m mamba_forest.validate_data \
    --gcs-root "$FORTY_ROOT" --train-shards 128 --val-shards 8 \
    --test-shards 1024 --seed "$seed" --require-gpu \
    --output "$RUN_ROOT/source_preflight_${seed}.json" \
    2>&1 | tee "$RUN_ROOT/source_preflight_${seed}.log"
  for arm in combo plain_ce; do
    run_dir="$RUN_ROOT/${arm}_${seed}"
    mkdir -p "$run_dir"
    loss_flags=()
    if [ "$arm" = plain_ce ]; then
      loss_flags=(--combo-alpha 1 --no-class-weights --label-smoothing 0)
    fi
    /usr/bin/time -f '%e' -o "$run_dir/train_wall_seconds.txt" \
      python -m mamba_forest.train \
        --gcs-root "$FORTY_ROOT" --seed "$seed" --fixed-train-subset \
        --precision mixed_float16 \
        --train-shards 128 --val-shards 8 --batch-size "$GPU_BATCH_SIZE" \
        --epochs 40 --patience 0 \
        --d-model 32 --d-state-temporal 8 --d-state-spatial 16 \
        --bottleneck-channels 512 --dropout 0.2 \
        "${loss_flags[@]}" --output-dir "$run_dir" \
        2>&1 | tee "$run_dir/train.log"
  done
  diff "$RUN_ROOT/combo_${seed}/split_manifest.json" \
       "$RUN_ROOT/plain_ce_${seed}/split_manifest.json"
done
```

Each seed changes both initialization and the fixed training subset. Pairing
compares objectives under the same seed/subset, rather than isolating
initialization variability. Confirm each manifest contains 40 completed
epochs, exactly 128 unique completed training shards and matching validation
paths. The six runs entail 30,720 training-shard passes and 1,920
validation-shard passes; these are workload counts, not duration estimates.

## Evaluate every selected checkpoint on the full test split

Freeze all configurations and select each best checkpoint before running
these evaluations. Use all 1,024 test shards for every arm and seed. Fresh
evaluation processes use float32 consistently even though training used mixed
float16. Architecture flags must match the saved weights.

```bash
for seed in 42 43 44; do
  for arm in combo plain_ce; do
    run_dir="$RUN_ROOT/${arm}_${seed}"
    /usr/bin/time -f '%e' -o "$run_dir/test_wall_seconds.txt" \
      python -m mamba_forest.evaluate \
        --gcs-root "$FORTY_ROOT" --weights "$run_dir/best.weights.h5" \
        --test-shards 1024 --batch-size "$GPU_BATCH_SIZE" \
        --d-model 32 --d-state-temporal 8 --d-state-spatial 16 \
        --bottleneck-channels 512 --dropout 0.2 \
        --output-dir "$run_dir/test" \
        2>&1 | tee "$run_dir/evaluate_full.log"
  done
done
```

Before comparing results, require equal test shard lists and class support
across all six evaluations, 1,024 unique test paths per run, positive pixel
counts, matching model configurations and the same metric conventions. Check
that each recorded checkpoint SHA-256 corresponds to that run's chosen best
checkpoint. Selected paths are not content hashes; preserve a source release
identifier or object versions/checksums when available.

The automatic check for the directory layout above is:

```bash
python -m mamba_forest.compare --run-root "$RUN_ROOT" --seeds 42 43 44
```

For each seed, report `Combo - plain CE` for Overall F1, Forests F1 and each
forest class, plus the actual epochs, best validation epoch, total train wall
time, summed epoch train/validation times and throughput. Convert the stored
F1 fractions to percentage points consistently. Show the three paired
differences and their mean and sample standard deviation; three seeds do not
support strong significance claims. Retain raw confusion matrices so metric
summaries can be recomputed. Full-test evaluation alone does not match the
published baselines' training budgets or modalities.

Keep the source commit, worktree status, package/GPU metadata, preflight
report, calibration JSONs, every config/manifest/log/CSV, best and last
checkpoints, checkpoint hashes, full-test JSONs, wall timers and paired summary
together. Checkpoints and run data are ignored by Git; use a durable artifact
store or the environment's retained run storage and record its location.
Publish observed measurements only after these runs complete. No result in
this runbook is a substitute for executing the experiment.
