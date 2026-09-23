Testing
=============
Welcome to the CNN-Mamba forest typology testing page!

The suite is written so that it runs on CPU in under a minute and needs neither
a GPU nor access to the dataset: every test builds a small model on synthetic
tensors. The point is to pin down the properties that are easy to break
silently — a sequence folded along the wrong axis, a modality that stops
reaching the output, a loss term that quietly rewards ignoring a class — rather
than to check numerical values that depend on initialisation.

## Overall structure

```bash
make test                          # python -m pytest tests -q
python tests/test_layers.py        # any file also runs standalone
```

Each file works both under pytest and as a script: running it directly executes
every `test_*` function in order and prints one line per test. The JAX track
has its own suite, `python jeo_plugin/tests/test_mamba_mtst_shapes.py`, which
needs jax, flax and einops but no JEO checkout.

## Unit tests

### `tests/test_layers.py`

- **Shape preservation.** `Mamba2Block` returns a tensor of the same shape it
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

## What is not covered

The data module talks to Google Cloud Storage, so `make_dataset` is exercised
by running a training job rather than by a unit test; the parts of it that are
pure tensor manipulation — `normalize`, `augment`, `encode_labels` — take and
return plain tensors and can be called directly on a synthetic example if you
want to extend the suite.

## Adding a test

Add a `test_*` function to the file that matches the module under test, keep it
free of dataset access, and prefer a property over a golden value: assert that
a perturbation propagates where it should and stops where it should not, rather
than that an output equals a number produced by today's random seed.
