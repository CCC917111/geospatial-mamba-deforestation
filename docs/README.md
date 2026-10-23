# Documentation

Multi-modal forest type segmentation with convolutional encoders and a
**Mamba-inspired selective state-space model**.

**Project lead:** Yike Chen

**Team:** Qianhua Wan, Cherng Khai Hng, and Jiawei Li

[Repository overview](../README.md) · [Technical summary](project-summary.md) ·
[Results and artifacts](../results/README.md)

## Choose a starting point

| What you want to do | Start here | Continue with |
|---|---|---|
| **Try the workflow locally** | [Synthetic CPU demo](local-demo.md) | [Usage guide](user.md) · [Tile input and output formats](apply.md) |
| **Understand the model** | [Technical summary](project-summary.md) | [Architecture and equations](design.md) |
| **Inspect the evidence** | [Results and retained artifacts](../results/README.md) | [Optimization audit](optimization-audit.md) · [Validation protocol](gpu-validation.md) |
| **Develop the package** | [Developer guide](developer.md) | [API reference](reference.md) · [Tests](testing.md) · [Common tasks](tasks.md) · [Coding style](coding.md) |

## Run the local demonstration

After installing the project as described in the [repository overview](../README.md):

```bash
make demo
```

This generates synthetic 128 × 128 records, trains a small model for two CPU
epochs, evaluates its checkpoint, and exports predictions and a labelled
comparison figure. Each invocation creates a fresh directory under `runs/`.
No dataset download, cloud credentials, or GPU is required.

The [demo guide](local-demo.md) explains every output and the separate
file-to-file prediction benchmark. The generated images and scores demonstrate
software integration; they do not measure accuracy on satellite observations.

## Read the evidence in context

| Evidence | What it supports | Where to inspect it |
|---|---|---|
| **Local software verification** | Automated tests and synthetic CPU training, checkpoint restoration, evaluation, and prediction | [Testing guide](testing.md) · [Completed demo record](../results/local_demo_cpu.json) |
| **CPU timing measurements** | Compute diagnostics and a separately measured file-to-file prediction workflow, each with its own architecture and timing boundary | [Measurement guide](local-demo.md) · [Raw prediction timing](../results/performance_predict_cpu.json) · [Audit](optimization-audit.md) |
| **Historical ForTy scores** | Context from earlier reported results; original logs and weights are unavailable, and literature baselines use different protocols | [Results discussion](../results/README.md) |

Live-data accuracy, controlled architecture comparisons, and GPU performance
remain unverified. The [validation runbook](gpu-validation.md) documents
prepared experiments; it is not a record of completed GPU runs.

## Prepare another runtime when needed

- [Cloud environment setup](cloud-environment.md): dependency profiles, runtime
  checks, and source/artifact recovery.
- [Free Colab guide](colab-free.md): a prepared notebook whose real GPU
  execution remains pending; free GPU availability is not guaranteed.
- [Real-data and GPU validation](gpu-validation.md): preflight, fixed-budget
  paired experiments, and evidence retention.
