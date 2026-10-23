Testing
=============
Welcome to the CNN-Mamba forest typology testing page!

The suite runs on CPU and needs neither
a GPU nor access to the remote dataset: tests use small models, synthetic
tensors and temporary local TFRecords. The point is to pin down properties
that are easy to break
silently — a sequence folded along the wrong axis, a modality that stops
reaching the output, a loss term that quietly rewards ignoring a class — rather
than to check numerical values that depend on initialisation.

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
- **Causality.** Perturbing the tail of a sequence must not change the outputs
  at the head, and must change the outputs at the tail — the second half of the
  assertion is what stops the test passing because the block ignores its input.
  The block is a recurrence plus a separable convolution of width three, so the
  test perturbs from step 6 on and checks steps 0 to 4, which the convolution
  cannot reach either. The perturbation varies across channels on purpose: the
  block normalises over the channel axis, so a shift applied equally to every
  channel would be removed before the recurrence ever saw it.
- **The state carries information.** Perturbing the first step must change the
  output at the last step; a block whose decay saturated to zero would pass the
  causality test and fail this one.
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
- `test_benchmark.py` checks timing aggregation and warmup accounting.
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
  the small default budget through the actual experiment argument parser, and
  reading only the expected Git bundle from an uploaded ZIP. It does not start
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
