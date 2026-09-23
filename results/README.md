# Recorded run

`training_results.csv` holds the per-epoch validation metrics of the training
run shown in the top-level README, taken from the Colab notebook in which the
model was originally developed.

Configuration of that run, as printed in the notebook log:

| Setting | Value |
|---|---|
| Training shards per epoch | 32 (of 1024) |
| Validation shards | 2 (of 1024) |
| Batch size | 16 |
| Optimiser | AdamW, learning rate 1e-4, weight decay 1e-4 |
| Loss | categorical cross-entropy |
| Gradient clipping | values clipped to [-1, 1] |
| Planned epochs | 20 (the run was stopped during epoch 5) |

Two caveats about the columns:

- `f1_class_1`, `f1_class_2`, `f1_class_3` are the per-class F1 scores of class
  indices 1, 2 and 3, which that run treated as the forest classes; `forest_f1`
  is their mean. The dataset documentation lists *Natural Forest* at index 0,
  so the code in `src/mamba_forest/data.py` now uses indices 0-2 and the
  indices used for the recorded numbers should be treated as unverified.
- `src/mamba_forest/train.py` writes the same file with one `f1_class_i`
  column per class, so a new run produces a superset of these columns.

The recorded run also predates two fixes in `src/mamba_forest/`: the optical
time series is now transposed before being folded into per-pixel sequences (the
notebook version fed width-adjacent pixels to the temporal block instead of
time steps), and each climate variable is standardised separately. The numbers
here therefore describe the earlier version of the model and should not be
quoted as results for the code in this repository.
