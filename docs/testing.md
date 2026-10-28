[Documentation](README.md) · [Project overview](../README.md)

# Testing

Check model behavior and CLI integration with small CPU models, synthetic
tensors and temporary local TFRecords. The suite needs neither a GPU nor
remote dataset access. Assertions cover properties that can fail silently,
including axis handling, modality influence and loss behavior, rather than
numerical values that depend on initialization.

## Continuous integration

[CPU checks](../.github/workflows/ci.yml) runs on pull requests to `main` and
pushes to the published development branch or `main`. It installs the pinned
Python 3.12 CPU environment, checks actual convolution gradients and GCS
filesystem registration without source access, then runs lint and the offline
test suite. Package versions, the runtime report and JUnit results are retained
as workflow artifacts. These checks do not run real-data training or GPU jobs.

## Overall structure

```bash
make test                          # python -m pytest tests -q
python tests/test_layers.py        # any file also runs standalone
```

The original layer, model and loss files also work as standalone scripts.
New tests use pytest fixtures and should be run through pytest. Everything the
suite needs is in `requirements.txt`.

## Unit tests

### `tests/test_layers.py`

- **Shape preservation.** `MambaBlock` returns a tensor of the same shape it
  was given, so it can be dropped into a residual stack.
- **Recurrent directionality.** Perturbing the tail of a sequence must not
  change the outputs at the head, and must change the outputs at the tail —
  the second half of the assertion is what stops the test passing because the
  block ignores its input.
  The block is a recurrence plus a separable convolution of width three, so the
  test perturbs from step 6 on and checks steps 0 to 4, which the convolution
  cannot reach either. This checks directionality outside the convolution's
  local neighbourhood; the complete block is not strictly causal because its
  convolution mixes adjacent positions on both sides.
  The perturbation varies across channels on purpose: the
  block normalises over the channel axis, so a shift applied equally to every
  channel would be removed before the recurrence ever saw it.
- **The state carries information.** Perturbing the first step must change the
  output at the last step; a block whose decay saturated to zero would pass the
  directionality test and fail this one.
- **SASF locality.** A perturbation at one pixel must reach its direct
  neighbours and must not reach a distant pixel — the property that separates a
  neighbourhood fusion from a global mixing layer.
- **SASF uses the five-point neighbourhood.** With the neighbour weights at
  zero and the centre weight at one, the layer doubles its input: the pixel's
  own state is a term of the fusion, as in the equation.
- **Fusion contract.** The fused map has the expected shape, the wrong number
  of modalities raises `ValueError`, and scaling any one modality changes the
  output, which catches a fusion that silently drops an input.

### `tests/test_model.py`

- **Output resolution.** The network returns a map at the resolution of the
  input tile, which is what makes the skip connections and the decoder stages
  line up.
- **Valid probabilities.** Outputs are non-negative and sum to one over the
  class axis.
- **Every modality matters.** Perturbing the optical series, the elevation map
  or the climate series each change the prediction.
- **CNN-only ablation.** The baseline has no SSM layers, keeps finite gradients,
  is invariant to optical season permutation, has fewer parameters and restores
  predictions from its saved architecture and weights.
- **Temporal order matters.** With an optical tile that is constant over space,
  reversing the season axis must change the prediction: the only structure left
  in that input is temporal, so the test fails if the phenological order is not
  actually read.
- **Finite gradients.** A backward pass through the full network produces
  finite gradients for every variable that receives one, which is the cheapest
  guard against a numerically unstable block.

### `tests/test_losses.py`

- **Ranking.** A correct prediction scores lower than a wrong one.
- **Class weighting.** Missing planted forest costs exactly its class weight
  more than missing an unweighted class, checked as an exact ratio rather than
  an inequality.
- **Uniform weights.** With `class_weights=None` the loss is class independent.
- **Dice masking.** With one class present and perfectly predicted, the Dice
  term is zero, which only holds if the eight absent classes are excluded from
  the average.
- **Validation.** An `alpha` outside `[0, 1]` and a weight vector of the wrong
  length are both rejected.

## Data, evaluation, prediction and performance

- `test_data.py` writes local TFRecords to check split isolation, explicit
  epoch membership, warmup stability, label validation and augmentation alignment.
- `test_evaluation.py` checks confusion-derived metrics, batch invariance,
  empty-data errors, non-finite predictions and evaluation provenance.
- `test_training.py` checks fixed/rotating manifests and a two-epoch synthetic
  training run on a four-shard local dataset, with checkpoints, timings and
  automatic CPU float32 selection without loss scaling. It also verifies that
  a rejected rerun preserves the previous checkpoint and configuration, and
  that sub-six-decimal improvements retain the exact selected epoch and hash.
- `test_checkpoint_config.py` checks configuration precedence, legacy defaults,
  explicit config selection, opt-out, malformed files, effective-value
  validation and exclusion of data/precision settings from architecture.
- `test_precision.py` checks compute dtypes, visible-device selection, explicit
  precision, the legacy float32 flag and early conflict rejection. GPU
  discovery is mocked in selection tests; these are not GPU execution tests.
- `test_predict.py` checks graph/eager agreement, dynamic tile sizes and the
  input contract, plus real nondefault checkpoint loading through adjacent,
  explicit and standalone configuration paths.
- `test_predict_timing.py` checks requested tensor synchronization, first-tile
  read accounting, per-pass export separation, checkpoint identity and
  protection of existing timing/configuration files.
- `test_benchmark_predict.py` checks warmup and first-trace exclusion,
  framework-free argument parsing, raw timing summaries and fresh-output
  protection, including normalized path aliases.
- `test_predict_batching.py` compares an actual reduced checkpoint at batch
  one/two across distinct climate inputs and heterogeneous tile sizes, verifies
  final partial groups and output order, checks owned input-buffer lifetimes,
  and rejects invalid batch settings before IO. It checks raw batch timings
  against allocated tile costs and excludes the entire first batch from
  steady-state summaries.
- `test_demo.py` checks the actual 128 × 128 TFRecord schema, distinct split
  examples, seed reproducibility, matching NPZ inputs, output protection,
  failure diagnostics and restoration of the caller's cache environment.
- `test_benchmark.py` checks timing aggregation and warmup accounting.
- `test_revision_performance.py` exercises shared fixture preparation, inference,
  complete optimizer updates, report protection and changed-input rejection even
  under optimized Python. `test_performance_comparison.py` rejects mismatched
  sources, fixtures, runtimes, sample counts, medians and batch throughput.
- `test_revision_profiling.py` captures a real reduced-model CPU XPlane trace,
  verifies replicated-batch shapes/throughput and complete training updates,
  excludes profiler samples from timing, and checks cleanup on capture errors.
- `test_batch_study.py` checks source/fixture/batch integrity, throughput units,
  alternating order, separate capture jobs, OOM exclusion and retained timing
  after a profiling failure. It rejects missing or modified trace artifacts.
- `test_profile_archives.py` verifies lossless bounded trace chunks, restore
  hashes, metadata omissions and safe fresh-output behavior. Notebook tests
  cover process-group cleanup, complete bundle downloads and delayed viewer
  installation with measured-package constraints.
- `test_selection_seed.py` checks one fixed cohort across run seeds while
  preserving example-shuffle randomness, preflight agreement, atomic history
  writes and actual optimizer progress. `test_atomic_checkpoint.py` checks
  failed/partial saves preserve existing H5 bytes and verifies real Keras reload.
- `test_stage_data.py` verifies original shard names, selected-file hashes,
  offline reuse, immutable repair and byte/disk budgets. `test_research.py`
  checks stage continuation, preserved partial attempts, test gating, lock
  recovery and rejection of changed/missing identities or completed outputs.
- `test_research_report.py` validates neutral numeric-ID metrics, the complete
  frozen seed list, retention hashes and the stronger source/runtime/data/GPU
  receipts required for real-data pilot completion. Research notebook tests
  cover paused/completed displays, Drive-first retention and worker cleanup.
- `test_validate_data.py` checks local-record schema diagnostics, sampled
  metadata, invalid labels and inputs, duplicate observed IDs, saved failure
  reports, and the GPU requirement before any record IO.
- `test_experiment.py` checks command planning, execution ordering and failure
  preservation without starting long training jobs.
- `test_comparison.py` checks paired summaries against retained artifact
  contracts and rejects incomplete or inconsistent comparisons.
- `test_cloud_environment.py` checks installation planning without execution,
  protection of existing environments including normalized path aliases,
  driver failure before installation, actual CPU convolution gradients and
  rejection of missing GPU/GCS support. It installs no packages or cloud resources.
- `test_colab_notebook.py` checks unexecuted notebook structure, Python syntax,
  the default architecture budget through the actual experiment argument parser,
  fixed-revision source checkout and reading only the expected Git bundle from
  an optional uploaded ZIP. It does not start
  Colab, install packages or run GPU computation.

## What is not covered

Tests exercise `make_dataset` using local files. They do not verify the public
bucket's schema, geographic block separation, historical benchmark scores or
real-data accuracy. Synthetic timing measurements do not establish performance
on other hardware or include the remote input pipeline.

## Adding a test

Add a `test_*` function to the file that matches the module under test, keep it
free of dataset access, and prefer a property over a hard-coded number: assert
that a perturbation propagates where it should and stops where it should not, rather
than that an output equals a number produced by today's random seed.
