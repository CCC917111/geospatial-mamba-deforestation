# Documentation

Multi-modal forest type segmentation with convolutional encoders and a
**Mamba-inspired selective state-space model**.

**Project lead and primary technical contributor:** Yike Chen, working with
Qianhua Wan, Cherng Khai Hng and Jiawei Li.

[Repository overview](../README.md) · [Project summary](project-summary.md) ·
[Results and artifacts](../results/README.md)

## Choose a starting point

| What you want to do | Start here | Continue with |
|---|---|---|
| **Try the workflow locally** | [Synthetic CPU demo](local-demo.md) | [Usage guide](user.md) · [Tile input and output formats](apply.md) |
| **Understand the model** | [Project summary](project-summary.md) | [Architecture and equations](design.md) |
| **Inspect the evidence** | [Results and retained artifacts](../results/README.md) | [Optimization audit](optimization-audit.md) · [Validation protocol](gpu-validation.md) |
| **Run a controlled research study** | [Multi-seed research pilot](research-pilot.md) | [Drive-first Colab notebook](../notebooks/colab_research_pilot.ipynb) · [Application brief](application-brief.md) |
| **Preserve trained weights** | [Checkpoint backup guide](checkpoint-backup.md) | [CPU backup notebook](../notebooks/colab_checkpoint_backup.ipynb) · [Interview preparation](application-interview.md) |
| **Measure runtime performance** | [Performance guide](performance.md) | [GPU profiling and batch scaling](gpu-profiling.md) · [GPU source comparison](../notebooks/colab_performance.ipynb) · [File-to-file timing](local-demo.md) |
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

The [demo guide](local-demo.md) describes the outputs and the separate
file-to-file prediction benchmark. The generated images and scores demonstrate
software integration; they do not measure accuracy on satellite observations.

## Read the evidence in context

| Evidence | What it supports | Where to inspect it |
|---|---|---|
| **Three-seed ForTy pilot** | Fixed-cohort CNN+SSM/CNN-only comparison with a small, seed-sensitive average improvement and higher parameter and training costs | [Audited results and paired-seed figure](../results/architecture_pilot_t4_20261029/README.md) · [Protocol](research-pilot.md) |
| **Local software verification** | Automated tests and synthetic CPU training, checkpoint restoration, evaluation, and prediction | [Testing guide](testing.md) · [Completed demo record](../results/local_demo_cpu.json) |
| **CPU timing measurements** | Compute diagnostics and a separately measured file-to-file prediction workflow, each with its own architecture and timing boundary | [Measurement guide](local-demo.md) · [Raw prediction timing](../results/performance_predict_cpu.json) · [Audit](optimization-audit.md) |
| **T4 compute measurements** | Source comparison, synthetic batch scaling and four real GPU traces; compute timings exclude IO | [Performance guide](performance.md) · [Source comparison](../results/performance_t4_20261026/README.md) · [Batch study](../results/batch_study_t4_20261028/README.md) |
| **Historical ForTy scores** | Context from earlier reported results; original logs and weights are unavailable, and literature baselines use different protocols | [Results discussion](../results/README.md) |

The small real-data architecture pilot is complete, but it does not establish
a stable architecture advantage. Full-scale training and full-test evaluation
remain pending. The [validation runbook](gpu-validation.md) describes those
larger experiments. The separate T4 batch study uses initialized weights and
a repeated synthetic tile; its compute throughput does not measure accuracy.

## Prepare another runtime when needed

- [Cloud environment setup](cloud-environment.md): dependency profiles, runtime
  checks, and source/artifact recovery.
- [Free Colab guide](colab-free.md): a notebook for a small real-source check;
  free GPU availability is not guaranteed.
- [Real-data and GPU validation](gpu-validation.md): preflight, fixed-budget
  architecture/objective comparisons, and evidence retention.
