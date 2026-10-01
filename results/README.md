Results
=============

All numbers are F1 scores in percent, computed over pixels and following the
metrics of the benchmark. For one class, *precision* is the share of the pixels
predicted as that class that really belong to it, *recall* is the share of the
class's pixels that are found, and *F1* is their harmonic mean. **Overall F1**
is the macro average over the nine classes, **Forest mean F1** the average over
the three forest types, and the last three columns are the F1 of natural
forest, planted forest and tree crops. In the code and in `test_metrics.json`
these five columns are named `Overall`, `Forests`, `N`, `P` and `TC`.

## Benchmark

| Model | Overall F1 | Forest mean F1 | Natural forest F1 | Planted forest F1 | Tree crops F1 |
|---|---:|---:|---:|---:|---:|
| UNet3D | 32.4 | 24.2 | 56.2 | 7.5 | 8.8 |
| UTAE | 49.4 | 37.7 | 71.4 | 13.8 | 27.8 |
| MTSViT | 81.1 | 74.9 | 82.8 | 62.9 | 78.9 |
| **CNN-Mamba (this repository)** | **56.89** | **48.81** | **64.67** | **27.64** | **54.12** |

The baseline scores are the ones published with the ForTy benchmark, where
every baseline is trained on the full training split. The CNN-Mamba model was
trained on 128 of the 1,024 training shards, about 12.5% of the data, validated
on 8 validation shards and evaluated on 64 test shards. Under that budget it improves on UNet3D and UTAE in every column,
most clearly on the two classes the benchmark finds hardest — planted forest
(27.64 against 13.8) and tree crops (54.12 against 27.8) — and stays behind
MTSViT.

## Training configuration

| Setting | Value |
|---|---|
| Training shards | 128 of 1,024 (~12.5%), randomly sampled |
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
`test_metrics.json` holds the same metrics for the test split.

## Effect of the objective

Training with a plain unweighted cross-entropy collapses the under-represented
classes: planted forest and tree crops are predicted as natural forest or other
vegetation and their F1 stays near zero, which also makes the validation
metrics swing between epochs. Weighting the cross-entropy by class and adding
the region-level Dice term removes both effects and is what the default
configuration uses.
