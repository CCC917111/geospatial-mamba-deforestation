# Optimization audit — 2026-10-20

Reviewed the data pipeline, comparison claims, training loop, evaluation and
prediction at baseline commit `0b7b8a2`.

## Data protocol

The code preserves ForTy's `train`, `validation` and `test` shard namespaces.
It does not parse coordinates or geographic block IDs, so the repository
cannot independently verify the benchmark's geographic split or rule out
correlation across block boundaries. No real ForTy records were available in
this environment; the public schema and real-data accuracy remain unverified.

Previously, training reshuffled all 1,024 shards on each dataset iteration and
read 128. A model-build warmup started an extra sampling iteration. Describing
this as using only 12.5% of the available training data was incorrect: it was a
per-epoch read budget. Across 40 independent epochs, expected unique coverage
would be `1 - (1 - 128/1024)^40`, approximately 99.5%. This expectation does
not reconstruct historical run coverage.

Training now selects explicit paths with a seeded `seed + epoch` sampler.
Dataset warmup preserves membership. `--fixed-train-subset` reuses the same
seeded membership for every epoch. `split_manifest.json` records every started
epoch, its training completion status, validation paths, sampling protocol and
the number of unique shards in completed epochs. This changes the sampling
sequence relative to old runs. Paths identify selected files; they are not
content hashes or a proof of per-record geographic independence.

Validation and test retain their historical first-N-shard selection without
augmentation. This makes comparisons stable but can be unrepresentative if
shard ordering correlates with geography. Invalid, fractional or out-of-range
labels now fail before one-hot encoding instead of silently creating zero
targets. Parsed integer masks retain their integer dtype.

## Model and objective comparisons

The existing table contains published baseline scores and historical
CNN-Mamba scores from a 64-shard test subset. Training budget, test membership
and modalities were not matched. Natural-forest F1 is below UTAE (64.67 versus
71.4), contradicting the earlier claim that all displayed scores were higher.
The table is retained as contextual history with corrected interpretation.

No historical checkpoint, logs, manifests or objective ablation artifacts
were retained, so the claimed data efficiency and loss-function effects cannot
be validated. [The results guide](../results/README.md) provides paired fixed
subset commands for new objective comparisons. Use several paired seeds,
identical split manifests and checkpoint-selection rules, and report actual
epochs and wall time. Full-benchmark comparisons require full-test evaluation
and matched training/data protocols; no new accuracy ranking is claimed here.

Evaluation now saves exact test paths, checkpoint SHA-256, model/runtime
settings, pixel counts, per-class support and the int64 confusion matrix.
Accuracy, mean IoU and F1 derive from the same matrix. Mean IoU excludes classes
with no true or predicted support; macro F1 includes all nine classes and
assigns absent classes zero. Empty validation/test datasets fail explicitly.

## Training and inference performance

The architecture and checkpoint weight structure are unchanged. The sequential
`tf.scan` implementation remains portable; it has device-dependent dispatch
cost and is not equivalent in speed to a fused Mamba CUDA implementation.

Evaluation now compiles inference and confusion updates into a graph, retains
the matrix on the device and copies it once at the end. Prediction uses a
compiled function with dynamic spatial dimensions. Without
`--save-probabilities`, it transfers a uint8 class map and scalar mean
confidence per tile instead of the full nine-channel float32 probability map.
For a 128×128 tile this reduces these output tensors from 589,824 bytes to
16,388 bytes, about 36-fold; this is a transfer-size calculation, not a latency
measurement. The first input tile is also loaded only once. Input validation
rejects an incorrect season count, empty dimensions and non-finite arrays.

Training seeds Python/Keras as well as TensorFlow and NumPy, and logs elapsed
training/validation seconds and training throughput. These epoch timers include
input IO and graph compilation and exclude checkpoint writes. A seed and shard
manifest establish membership, not bitwise determinism on every accelerator.
Use float32 on CPU unless measuring a specific mixed-precision configuration.

The retained [CPU measurement](../results/performance_cpu.json) used TensorFlow
2.21.0, Python 3.12.14, float32, 32×32 tiles, batch size 1, `d_model=8`, state
dimensions 8/16 and bottleneck width 32 (3,413,016 parameters). Both TensorFlow
thread environment settings were 2. Each mode had three warmup steps followed
by ten synchronized measurements; raw samples and runtime settings are in JSON.
Inference modes share one unchanged model/input; training uses a separate
model with AdamW and ComboLoss.

| Synthetic mode | Median step | p95 step | Tiles/s |
|---|---:|---:|---:|
| Eager inference | 89.59 ms | 118.60 ms | 10.94 |
| Graph inference | 14.34 ms | 14.93 ms | 73.48 |
| Graph training | 53.38 ms | 62.60 ms | 18.46 |

Graph inference median was about 6.25 times faster in this measurement. This
compares fixed-shape synthetic model compute plus a scalar output barrier; it
does not time the production predictor's dynamic signature, class-map outputs,
file writes, remote input pipeline, default 128×128 model or GPU execution.
It provides no evidence of an accuracy improvement or a training speedup
against a previous implementation.

Reproduce the measurement with:

```bash
TF_CPP_MIN_LOG_LEVEL=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 \
python -m mamba_forest.benchmark --mode both --tile-size 32 \
    --batch-size 1 --d-model 8 --bottleneck-channels 32 \
    --steps 10 --warmup 3 --device /CPU:0 \
    --output runs/performance_cpu.json
```

## Validation

### Follow-up: default model and real-data readiness

The [official ForTy documentation](https://github.com/google-deepmind/forest_typology/blob/main/forty_v1/README.md)
confirms the bucket root, 1,024-shard naming convention and the modality keys.
Its climate shape is `[4, 1, 1, 14]`; the parser's `[4, 14]` representation
removes only singleton spatial dimensions. These documentation checks do not
verify the live TFRecord encoding or label semantics.

The default 128×128 model (7,732,800 parameters, batch 1, float32) was also
measured on CPU with two warmups and three synchronized samples per mode.
The [raw artifact](../results/performance_cpu_128.json) records eager median
316.54 ms and graph median 149.01 ms, approximately 2.12 times faster for this
synthetic compute comparison. Three samples are a small diagnostic measurement,
not a robust hardware performance study. It has the same scalar-output and
IO exclusions as the reduced-model benchmark above.

The current runtime has no visible GPU, uses a CPU TensorFlow build, and its
enforced network policy excludes Google Cloud Storage. No GPU provisioning
capability or configured cloud identity is available. Consequently no real
ForTy training, full-test evaluation or GPU measurement has been executed.
The retained [preflight report](../results/forty_gpu_preflight.json) records
the GPU-gate failure with zero inspected splits and no source-record IO.
The [runbook](gpu-validation.md) stages those experiments, starting with
`mamba_forest.validate_data`. This preflight retains raw feature diagnostics,
normalization/label checks, available sample IDs and coordinates, and exact
selected paths; `--require-gpu` fails before record IO on a CPU-only runtime.
Its three inspected records can detect a duplicate inspected ID but cannot
establish full-split geographic separation.

### Offline follow-up — 2026-10-21

Training and evaluation now accept a local `--data-root` and configurable
`--total-shards`, with the count propagated to filenames, actual data reads,
selection manifests and evaluation provenance. Preflight accepts the same root
alias. The legacy `--gcs-root` and JSON `gcs_root` key remain supported. The
two-epoch integration test now uses genuine four-shard local filenames rather
than a sparse set with a 1,024-shard suffix.

Training's default `--precision auto` selects float32 on CPU and mixed float16
with a visible GPU. Every run resets the policy and records both requested and
resolved precision. Loss scaling follows the resolved policy. Explicit policies
and the legacy float32 flag remain available; conflicting flags fail before
data IO. Device-discovery tests mock GPU visibility without claiming GPU
execution, while the local training smoke checks actual CPU computation,
checkpoint reload and the absence of loss scaling.

`ComboLoss` previously used the truth value of `class_weights`, which raised an
ambiguous-truth-value error for NumPy arrays. Weight sequences now normalize
to a serializable tuple of Python floats, preserving `None` versus zero-weight
semantics. Invalid class counts, smoothing settings, stabilizers and weights
fail during construction. Numerical loss, gradient and serialization checks
cover NumPy arrays, lists and tuples.

### Checkpoint configuration follow-up — 2026-10-22

Evaluation and prediction now restore the five architecture settings from the
flat training `config.json` beside the weights. Explicit CLI values take
precedence; absent settings use shared, unchanged model defaults. `--config`
selects a moved checkpoint's training configuration, and `--no-config` disables
automatic discovery. Only architecture fields are read; dataset paths and
runtime precision are not inherited. Malformed configurations and invalid
effective values fail before dataset or tile IO.

Training previously wrote a new `config.json` before attempting to read data.
A failed rerun in a directory with old weights could therefore pair those
weights with the new configuration. Training now rejects an output directory
containing `*.weights.h5` before writing configuration or changing precision.
Preflight-only directories remain usable, and existing checkpoints are not
implicitly resumed. A fresh output directory is required for a new run.

Integration tests save actual nondefault model weights and reload them without
manual architecture flags. Evaluation reproduces the original model's confusion
matrix on local synthetic TFRecords. Prediction reproduces the original eager
probabilities through adjacent-config, explicit-config and standalone/manual
paths. Regression tests also preserve existing weight/config bytes after a
rejected training rerun. These checks verify configuration and checkpoint
integrity, not real-data accuracy or GPU performance.

### Automated checks

The baseline's 17 tests passed before changes. New tests cover local TFRecord
split isolation and membership, augmentation alignment, invalid labels,
hand-computed metrics, batch-boundary invariance, empty evaluation, saved
provenance, graph/eager prediction agreement, dynamic tile sizes, predictor
file outputs, benchmark warmup accounting, and a two-epoch training run on
small local records. The smoke run verifies exact manifests, finite timing
metrics and reloads both checkpoints; the final checkpoint reproduces the
trained model's output. Tests use synthetic data and do not replace a real
ForTy validation run.

Final offline checks on TensorFlow CPU 2.21.0 / Python 3.12.14: all 136 tests
passed in 57.87 seconds, flake8 passed and `git diff --check` passed. All 19
Bash blocks in the user guide, prediction guide and GPU runbook passed syntax
checks. The preflight tests
include malformed raw-feature diagnostics, optional metadata, observed duplicate
IDs, invalid inputs and the GPU gate before record IO. Keras checkpoint
serialization emitted 757 NumPy `__array__(copy=...)` deprecation warnings;
checkpoint reload and prediction-equivalence checks passed despite them.

### Paired experiment workflow — 2026-10-22

`mamba_forest.experiment` now generates a framework-free plan and executes
paired fixed-subset ComboLoss/plain-CE runs in fresh directories. It preserves
commands, source/package metadata, stage status, logs and process wall times.
The GPU probe and per-seed data preflight precede training, and all training
artifacts are checked before test evaluation starts. CPU execution is an
explicit option for small offline checks.

`mamba_forest.compare` checks complete training budgets, seeded memberships,
objective settings, effective model/runtime conditions, class support and
checkpoint hashes. It recomputes test scores from confusion matrices before
reporting paired differences in percentage points and sample standard
deviations. It reports coverage and timing scopes alongside the scores.
Shard paths and class support do not establish source-content identity or
geographic independence.

Checkpoint selection formerly used unrounded validation scores while the CSV
rounded them to six decimals. Two different scores could therefore appear
tied in the retained history. Training now writes round-trip precision and
an atomic `checkpoint_selection.json` with the exact best epoch, score and
checkpoint SHA-256. The selection, evaluation and current weight hashes must
agree, so replacing a selected checkpoint invalidates the comparison.
The training regression covers improvements hidden by six-decimal rounding.

The retained [pipeline report](../results/paired_pipeline_smoke.json) records
two CPU epochs per objective on a synthetic 128×128 fixture, followed by saved
configuration reload, test evaluation and verified paired summarization. All
seven stages passed. A separate invocation without `--allow-cpu` failed at the
GPU operation probe before any data preflight. The evaluation graph also now
rejects non-finite predictions before argmax can turn them into class indices.

Final workflow checks: 183 tests passed in 53.65 seconds; flake8 and
`git diff --check` passed; all 25 Bash examples passed syntax checks. Keras
emitted 865 NumPy deprecation warnings. These checks establish offline workflow
integrity; real-data accuracy, geographic independence and GPU throughput
remain unmeasured.

### New environment preparation — 2026-10-22

The [cloud setup guide](cloud-environment.md), CPU/GPU dependency profiles and
`scripts/setup_cloud_env.sh` prepare a fresh Python 3.12 virtual environment on
an existing host. The GPU profile installs the CUDA-enabled TensorFlow package
with its CUDA and GCS extras; the CPU profile omits the CUDA extra. Direct
dependency versions match the local tested runtime, while target-specific
dependencies are resolved and retained after installation.

The setup script preserves inherited network/credential settings and rejects
existing virtual environments after normalizing their paths. Driver failure
stops GPU installation before environment creation. A standalone readiness
checker verifies GCS registration and finite convolution gradients on the
requested device, with CPU fallback disabled for GPU checks.

All 190 tests passed in 59.85 seconds; lint, shell syntax and diff checks
passed. The standalone checker passed on the current CPU runtime and rejected
it as a GPU runtime. No new host was created, no target installation was run,
and no billable resources were provisioned. Provider, account connection and
budget remain required before deployment; the current environment interface
exposes inspection only.

### Free Colab preparation — 2026-10-23

The [Colab guide](colab-free.md) and unexecuted notebook provide a free-runtime
entry point. The notebook restores source commit `ae6a123` from the supplied
Git recovery archive, keeps Colab's existing TensorFlow/CUDA stack, verifies
GPU convolution gradients and GCS registration, and prepares two objective
arms with two epochs, one shard per split, one seed and batch size one.
Those scores are diagnostic; they do not establish an objective ranking.

An offline rehearsal executed the notebook's upload/restore code against the
actual recovery ZIP, checked the pinned checkout, generated the tiny GPU plan
with the restored source and archived the artifacts. No source records or GPU
training were executed. Twelve notebook tests cover syntax, defaults, real
CLI parsing and reading only the expected bundle from uploaded ZIP content.

Optional Drive output preserves completed artifacts from the start; otherwise
the user must download from temporary storage. Normal interruptions stop the
runner process group before archiving. Forced runtime reclamation may prevent
export, and interrupted training does not resume optimizer state. Free GPU
availability and duration remain subject to the provider. Actual Colab GPU
execution requires the user's signed-in session and remains unexecuted here.

Final preparation checks: 202 tests passed in 54.50 seconds; lint, diff checks
and all 33 documentation Bash blocks passed. Keras emitted 865 upstream NumPy
deprecation warnings. Notebook cells remain unexecuted in the delivered file.
