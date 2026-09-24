Mapping Your Own Region
=============
Welcome to the CNN-Mamba forest typology application page!

The benchmark answers "how good is this model". This page answers the question
someone with their own imagery actually has: **give the model a stack of
satellite tiles, get back a forest-type map and a table of what is in them.**

## What the tool is for

You have satellite coverage of an area — a concession, a catchment, a national
park, a supply-chain sourcing region — and you want to know not just where the
trees are, but *which kind*: natural forest, planted forest or tree crops. That
distinction is the one that matters for deforestation monitoring and
zero-deforestation compliance, and it is the one a normal NDVI threshold or a
binary forest mask cannot give you.

What it is not: a general-purpose segmentation model. It expects the ForTy v1
modality stack — Sentinel-2 seasonal composites, terrain and climate — and it
predicts the nine ForTy land-cover classes. Feeding it single-date imagery, a
different band order or a different class taxonomy will produce output, but not
meaningful output.

## What you need

- A trained checkpoint (`best.weights.h5`). Train one with `make train`, which
  needs read access to the public ForTy bucket; the repository ships the code,
  not the weights.
- Your tiles in the format below.
- A GPU is convenient but not required; prediction on CPU is fine for tens of
  tiles.

## Input format

One `.npz` file per tile, with three arrays:

| Array | Shape | Contents | Units |
|---|---|---|---|
| `s2` | `(4, H, W, 10)` | Sentinel-2 seasonal composites | raw reflectance, the scale the ForTy shards use (about 0-10000) |
| `elevation` | `(H, W, 3)` | elevation, slope, aspect | metres, degrees, degrees |
| `climate` | `(4, 14)` | the 14 climate variables, one row per composite | native TerraClimate units |

Rules: `H` must equal `W` and be divisible by 8, because the encoder pools three
times; 128 x 128 is the size the model was trained on. The four composites are
the four seasons, in a consistent order across your tiles. The band order of
`s2` must be the ForTy v1 band order.

Store the arrays **unnormalised**. `predict.py` calls the same `normalize()`
that the training pipeline calls, so a tile prepared for prediction and a tile
prepared for training go through identical code — there is no second,
drifting copy of the scaling constants.

Building one tile from arrays you already have:

```python
import numpy as np

np.savez_compressed(
    "tiles/aoi_0001.npz",
    s2=s2,                # (4, 128, 128, 10) float32
    elevation=elevation,  # (128, 128, 3)     float32
    climate=climate,      # (4, 14)           float32
)
```

If your imagery is a GeoTIFF mosaic rather than tiles, cut it into 128 x 128
windows first and keep the window offsets in the file name — the class maps come
back in the same array layout, so you can mosaic the predictions with the same
offsets.

## Running it

```bash
make predict TILES=my_tiles OUT=predictions
```

which is

```bash
python -m mamba_forest.predict \
    --weights runs/mamba_forest/best.weights.h5 \
    --input-dir my_tiles \
    --output-dir predictions \
    --save-png
```

The four architecture flags (`--d-model`, `--d-state-temporal`,
`--d-state-spatial`, `--bottleneck-channels`) have to match the values the
checkpoint was trained with; the defaults match `make train`.

## What you get back

For every tile:

- `<tile>_class_map.npy` — an `(H, W)` `uint8` array of class indices, in the
  order of `CLASS_NAMES`: 0 Unknown, 1 Natural Forest, 2 Planted Forest,
  3 Tree Crops, 4 Other Vegetation, 5 Water, 6 Ice, 7 Bare Ground,
  8 Built Areas.
- `<tile>_class_map.png` with `--save-png` — the same map, colour-coded, for a
  quick visual check.
- `<tile>_probabilities.npy` with `--save-probabilities` — the full
  `(H, W, 9)` softmax, if you want to apply your own threshold or measure
  uncertainty.

And one table for the whole run:

- `predictions.csv` — one row per tile with the pixel share of every class, the
  combined `forest_share`, and the mean softmax confidence.

The console prints the same per-tile summary and the average composition over
all tiles, which is usually the number you actually wanted:

```
  aoi_0001: forest  62.4% | natural 41.2% | planted 14.8% | tree crops  6.4%
  aoi_0002: forest  55.1% | natural 12.0% | planted 38.7% | tree crops  4.4%

average composition over all tiles:
  Unknown             1.02%
  Natural Forest     26.60%
  Planted Forest     26.75%
  ...
```

## Turning that into an answer

The class shares are what make the output actionable. A few patterns:

- **Composition of a region.** Average `share_natural_forest`,
  `share_planted_forest` and `share_tree_crops` over the tiles that cover it.
  A concession that is 40% planted forest and 5% natural forest tells a very
  different story from the reverse, even at identical total canopy cover.
- **Change between two dates.** Run the model on tiles built from two different
  years and subtract the class maps. Natural forest turning into tree crops is
  a conversion event; natural forest turning into bare ground is a clearing.
  Both are invisible to a binary forest mask, which sees canopy in the first
  case and simply "less forest" in the second.
- **A shortlist for inspection.** Sort `predictions.csv` by the share of the
  class you care about and look at the top tiles first. With
  `--save-probabilities` you can also rank by mean confidence and inspect the
  least certain tiles, which is where label noise and cloud cover usually sit.

## Applying it to a different label set

The class list, the class count and the forest grouping are three constants in
`src/mamba_forest/data.py` — `CLASS_NAMES`, `NUM_CLASSES` and
`FOREST_CLASS_INDICES`. Changing them changes the network's output head, the
metric aggregation and the colour map together, so retraining on a different
taxonomy is a data-module change rather than a model change.
[Common Tasks](tasks.md) has the full recipe, including adding a fourth
modality such as Sentinel-1 radar.
