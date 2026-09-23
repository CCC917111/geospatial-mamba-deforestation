Coding Style
=============
Welcome to the CNN-Mamba forest typology coding style page!

The Python code follows the
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html),
with the conventions below applied consistently across the package.

## Formatting

- Two-space indentation, 80 columns for code and prose, 88 as the hard limit
  checked by the linter.
- `from __future__ import annotations` at the top of every module, so type
  hints stay readable on older interpreters.
- Imports in three groups — standard library, third party, then this package —
  separated by a blank line and alphabetised within each group.
- One class or one coherent group of functions per module; a module that starts
  needing a table of contents is split instead.

## Naming and interfaces

- `snake_case` for functions and variables, `CamelCase` for classes,
  `UPPER_CASE` for module-level constants.
- A leading underscore marks a helper that is not part of the module's public
  surface, such as `MambaForestSegmenter._encode_optical`.
- Every public function and class carries a docstring that states its intent,
  its arguments and the tensor shapes it expects and returns. Shapes are
  written as `[B, T, H, W, C]` and named consistently across the package.
- Constants that appear in more than one place — channel counts, class names,
  the shard root — live in `data.py` or `model.py` and are imported, never
  duplicated as literals.

## Comments

- Comments explain intent, not mechanics: why the time axis is transposed
  before the reshape, why the Dice term masks absent classes, why the output
  head stays in float32.
- Anything a future reader would find surprising gets a comment, including
  workarounds for framework behaviour such as restoring a static shape after
  `tf.reshape`.

## Correctness habits

- Arguments are validated where a wrong value would otherwise fail deep inside
  a framework call: `ComboLoss` rejects an out-of-range `alpha` and a
  mismatched weight vector, `DynamicMultiModalFusion` rejects the wrong number
  of modalities, and `make_dataset` rejects an impossible shard count.
- Randomness is seeded from a single `--seed` flag, and every configuration is
  dumped to `config.json` next to the checkpoints, so a run can be repeated.
- No credentials, absolute paths or machine-specific settings in the source;
  the dataset root, the output directory and every hyper-parameter are flags
  with defaults.

## Checking

```bash
make lint          # python -m flake8 src tests scripts --max-line-length 88
```
