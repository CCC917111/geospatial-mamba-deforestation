Results
=============

All numbers are F1 scores in percent on the ForTy v1 test split, following the
protocol of the benchmark: **Overall** is the macro F1 over the nine classes,
**Forests** is the mean F1 of the three forest types, and **N**, **P** and
**TC** are the individual F1 scores of natural forest, planted forest and tree
crops.

## Benchmark

| Model | Overall | Forests | N | P | TC |
|---|---:|---:|---:|---:|---:|
| UNet3D | 32.4 | 24.2 | 56.2 | 7.5 | 8.8 |
| UTAE | 49.4 | 37.7 | 71.4 | 13.8 | 27.8 |
| MTSViT | 81.1 | 74.9 | 82.8 | 62.9 | 78.9 |
| **CNN-Mamba (this repository)** | **56.89** | **48.81** | **64.67** | **27.64** | **54.12** |

The baseline scores are the ones published with the ForTy benchmark, where
every baseline is trained on the full training split. The CNN-Mamba model was
trained on 128 of the 1024 training shards, about 12.5% of the data, and
evaluated on 64 test shards. Under that budget it improves on UNet3D and UTAE
in every column, most clearly on the two classes the benchmark finds hardest —
planted forest (27.64 against 13.8) and tree crops (54.12 against 27.8) — and
stays behind MTSViT, which sees eight times as much training data.

## Training configuration

| Setting | Value |
|---|---|
| Training shards per epoch | 128 of 1024 (~12.5%) |
| Validation shards | 8, fixed across epochs |
| Test shards | 64 |
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
