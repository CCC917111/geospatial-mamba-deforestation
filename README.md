<div align="center">
  <h1>CNN-Mamba</h1>
  <p><strong>Multi-modal forest type segmentation</strong></p>
  <p>Seasonal optical imagery &nbsp;·&nbsp; Terrain &nbsp;·&nbsp; Climate</p>
  <p>
    <a href="#quick-start">Run the local demo</a> &nbsp;·&nbsp;
    <a href="docs/project-summary.md">Project brief</a> &nbsp;·&nbsp;
    <a href="#results">Results &amp; evidence</a> &nbsp;·&nbsp;
    <a href="docs/README.md">Documentation</a>
  </p>
</div>

CNN-Mamba combines convolutional encoders with a **Mamba-inspired selective
state-space model** to map forest types from four-season Sentinel-2 imagery,
terrain and climate. The TensorFlow package includes training, evaluation,
tile prediction and an offline demonstration.

**Project lead:** Yike Chen · **Team:** Qianhua Wan, Cherng Khai Hng and Jiawei Li.

A deep-learning course project at the Chinese University of Hong Kong, Shenzhen.

| Task | Inputs | Implementation |
|---|---|---|
| Nine-class pixel segmentation | Optical time series, terrain and climate | TensorFlow · selective SSMs · U-Net-style decoder |

## Quick start

### Setup

Use Python **3.10–3.12**. The retained checks used Python 3.12 and TensorFlow
2.21; a pinned CPU recipe is available in the
[environment guide](docs/cloud-environment.md#install-into-a-new-virtual-environment).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
make install
```

### Local demonstration

```bash
make demo
```

The demo generates standard 128 × 128 synthetic records, trains a small model
for two epochs on CPU, evaluates the selected checkpoint and predicts two test
tiles. Each run writes a fresh directory under `runs/`.

| Explore the output | What it contains |
|---|---|
| `comparison.png` | Synthetic optical inputs, constructed labels and model predictions |
| `predictions/` | Class maps, PNGs, class-share CSV and stage timings |
| `summary.json` | Run status, checkpoint identity and artifact locations |

> **Demo scope:** Generated arrays exercise the full software workflow.
> The short training run is not a measure of real-world segmentation accuracy.

Follow the [demo and timing guide](docs/local-demo.md) for a retained preview,
output details and repeated file-to-file benchmarks.

### Check the installation

```bash
make test
make lint
```

The checks use CPU computation and local synthetic records. They require no
ForTy download or cloud credentials.

## Problem

Natural forest, planted forest and tree crops can have similar canopy cover
while representing different land uses. ForTy v1 [[1]](#ref-1) frames this
distinction as pixel-level classification across nine land-cover classes.
Seasonal optical observations describe temporal patterns; terrain and climate
provide complementary context.

The implemented task is **forest type segmentation**. Comparing maps across
years remains an extension requiring independent temporal validation.

## Model

![Forest type segmentation architecture](docs/figures/architecture.svg)

*Convolutional features and custom selective-SSM blocks connect multi-modal
inputs to pixel-level predictions. [Open the diagram](docs/figures/architecture.svg)
or read the [design notes](docs/design.md).*

| Stage | What it does |
|---|---|
| **Encode** | A temporal selective SSM summarizes each pixel's four seasons; convolutions encode terrain and an MLP embeds climate. |
| **Fuse** | Sample-dependent weights combine the modalities before a U-Net-style encoder [[3]](#ref-3). |
| **Aggregate** | A spatial selective SSM scans the 16 × 16 bottleneck; Structure-Aware State Fusion [[10]](#ref-10) mixes local neighbours. |
| **Decode** | Skip connections restore spatial detail and a softmax head produces nine-class probabilities. |

CNN-Mamba is a custom TensorFlow adaptation inspired by Mamba [[2]](#ref-2).
The [state-space implementation](docs/design.md#selective-state-space-blocks)
uses a shared low-rank state projection and `tf.scan`. Its recurrence and
device-dependent overhead are documented separately from measured performance.

## What the repository implements

- **Auditable experiments:** seeded data selection, per-epoch split manifests,
  validation-selected checkpoints and confusion-derived F1 metrics.
- **Verified comparisons:** paired architecture and objective plans check data paths, training
  budgets, saved configurations and checkpoint hashes.
- **Practical prediction:** shared preprocessing, saved-architecture restoration,
  optional bounded batches, class maps and a CSV of predicted class shares.
- **Local inspection:** an offline demo, CPU software checks, compute-only
  diagnostics and file-to-file prediction timing.

The [runtime performance guide](docs/performance.md) documents checkpoint-compatible
temporal readout and validation-graph optimizations. In three controlled CPU
rounds, inference median latency fell 6.4%–8.9%; training-step timings did not
show a stable gain. A [three-round Tesla T4 comparison](results/performance_t4_20261026/README.md)
found lower TensorFlow allocator peaks (12.75% for inference, 6.04% for
training), with no stable inference or training latency gain at batch one.
The completed [T4 batch study](results/batch_study_t4_20261028/README.md) found
4.80× inference and 5.82× training throughput at batch eight versus one, using
the same repeated synthetic tile. Four traces contain real GPU events.
These compute-only results do not establish file-to-file throughput or accuracy.

### CNN-only ablation

The default `cnn_ssm` model retains the selective state-space blocks.
`--model-variant cnn_only` uses optical temporal mean pooling and omits both
SSM blocks while keeping fusion, CNN stages and local SASF. Saved checkpoints
retain their variant; older configurations default to `cnn_ssm`.

Generate the controlled architecture comparison without reading data:

```bash
python -m mamba_forest.experiment --comparison architecture \
  --run-root runs/architecture-pair
```

The [validation runbook](docs/gpu-validation.md#controlled-architecture-comparison)
explains execution, shared controls and the different parameter counts.
The [retained research pilot](docs/research-pilot.md) fixes one shard cohort
across three run seeds and keeps all six best checkpoints in Drive, with
verified stage continuation. Its three-seed synthetic CPU workflow check is
complete; real-data multi-seed training remains pending.
The [free Colab notebook](https://colab.research.google.com/github/CCC917111/geospatial-mamba-deforestation/blob/main/notebooks/colab_free_gpu.ipynb)
prepares a smaller real-source GPU check. The separate
[performance notebook](https://colab.research.google.com/github/CCC917111/geospatial-mamba-deforestation/blob/main/notebooks/colab_performance.ipynb)
runs the synthetic before/after GPU comparison recorded above.

## Results

Each result below has a different validation scope. Source artifacts and
protocols are linked from the [results record](results/README.md).

| Evidence | Retained result | Interpretation |
|---|---|---|
| **Software checks** | **486 tests passed** | CPU verification; supports software behavior. |
| **Offline demonstration** | Two training epochs, evaluation and prediction | Generated 128 × 128 inputs and a reduced model. |
| **File-to-file prediction** | **60 tile executions · 7.20 tiles/s** | Two synthetic inputs, 30 passes after one warmup pass; includes PNG and CSV output. |
| **Controlled CPU compute** | Inference median latency **6.4%–8.9% lower** per round | Three alternating rounds, 30 synthetic samples per source/workload/round; no stable training gain. |
| **Controlled T4 compute** | Inference/training allocator peaks **12.75% / 6.04% lower** | Same three-round synthetic protocol; no stable latency gain; memory is not total GPU VRAM. |
| **T4 batch scaling** | Batch-eight throughput **4.80× inference / 5.82× training** versus batch one | Three alternating rounds on one repeated synthetic tile; four real GPU captures; excludes IO. |
| **Historical ForTy report** | Overall F1 **56.89%**; forest mean F1 **48.81%** | Original logs and checkpoint are not retained; scores are not independently reproducible. |

The historical run and published UNet3D, UTAE and MTSViT references differ in
training budget, modalities and test membership, so they do not form a
controlled ranking. Full-scale real ForTy training and full-test evaluation
remain pending. The retained GPU compute record does not establish accuracy.

The next research question is whether selective-SSM modules improve
segmentation over a matched CNN control across seeds. A controlled multi-seed
architecture ranking has not been established.

**Inspect the evidence:** [Demo record](results/local_demo_cpu.json) ·
[Prediction timings](results/performance_predict_cpu.json) ·
[Compute timings](results/performance_cpu_128.json) ·
[Paired workflow check](results/paired_pipeline_smoke.json) ·
[Audit](docs/optimization-audit.md)

For application materials, use the [evidence-backed project brief](docs/application-brief.md).
The 4.80× figure means synthetic inference compute throughput at batch eight
versus batch one on T4, excluding IO; it is separate from architecture accuracy.

## Dataset

[ForTy v1](https://github.com/google-deepmind/forest_typology/tree/main/forty_v1)
[[1]](#ref-1) contains about 200,000 globally sampled 128 × 128 tiles, published
as 1,024 TFRecord shards per split.

| Input | Per-tile shape | Contents |
|---|---|---|
| Sentinel-2 | `(4, 128, 128, 10)` | Four seasonal optical composites |
| Terrain | `(128, 128, 3)` | Elevation, slope and aspect |
| Climate | `(4, 14)` after parsing | Seasonal climate variables |
| Labels | `(128, 128)` | Nine classes, including three forest types |

The benchmark describes geographically blocked splits in an 8:1:1 ratio.
This package preserves the published namespaces. Its model inputs do not
retain coordinates or block IDs, so geographic separation has not been
independently verified here.

### Training, Validation and Test Sets

| Split | Default shard budget | Purpose |
|---|---|---|
| Training | 128 of 1,024, resampled each epoch | Fit weights |
| Validation | 8 of 1,024, fixed | Select the checkpoint |
| Test | 64 of 1,024 | Evaluate the selected checkpoint |

The 12.5% training share is a **per-epoch read budget**. Cumulative unique
coverage can be larger. New runs save exact paths in `split_manifest.json`;
`--fixed-train-subset` keeps training membership fixed. Validation and test
use the first N shards, which may be unrepresentative if ordering correlates
with geography. Full-split evaluation uses `--test-shards 1024`.

## Training

Default-scale training assumes a suitable GPU and authorized ForTy bucket
access. Configure application-default credentials on that host, for example
with `gcloud auth application-default login`, then run:

```bash
make train
make eval
```

Training selects `best.weights.h5` using validation F1. Evaluation reads the
test split. Outputs go to `runs/mamba_forest` by default; trained weights are
not shipped with the repository. The [usage guide](docs/user.md) documents
flags and outputs, including [local-data workflows](docs/user.md#local-data-and-offline-checks).

## Mapping Your Own Region

Prepare raw `.npz` tiles and keep the checkpoint together with its training
`config.json`.

```bash
make predict TILES=my_tiles OUT=predictions
```

The [tile-prediction guide](docs/apply.md) specifies shapes, units, band order
and output interpretation. Reliable mapping requires validation on
representative labelled imagery from the target region.

## Documentation

| Start with | Continue to |
|---|---|
| [Documentation hub](docs/README.md) | A guide map organized by task |
| [Project brief](docs/project-summary.md) | Research context, design and evidence |
| [Local demo](docs/local-demo.md) | An executable example and prediction timing |
| [Usage](docs/user.md) · [Design](docs/design.md) | Commands, data flow and architecture |
| [Results](results/README.md) · [Developer guide](docs/developer.md) | Retained artifacts and implementation details |

## References

1. <a id="ref-1"></a>Y. Jiang and M. Neumann. Not every tree is a forest:
   Benchmarking forest types from satellite remote sensing. 2025.
   [arXiv:2505.01805](https://arxiv.org/abs/2505.01805)
2. <a id="ref-2"></a>A. Gu and T. Dao. Mamba: Linear-time sequence modeling
   with selective state spaces. 2023.
   [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
3. <a id="ref-3"></a>O. Ronneberger, P. Fischer and T. Brox. U-Net:
   Convolutional networks for biomedical image segmentation. *MICCAI*, 2015.
   [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
4. <a id="ref-4"></a>Ö. Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox and
   O. Ronneberger. 3D U-Net: Learning dense volumetric segmentation from sparse
   annotation. *MICCAI*, 2016. [arXiv:1606.06650](https://arxiv.org/abs/1606.06650)
5. <a id="ref-5"></a>V. Sainte Fare Garnot and L. Landrieu. Panoptic
   segmentation of satellite image time series with convolutional temporal
   attention networks. *ICCV*, 2021.
   [arXiv:2107.07933](https://arxiv.org/abs/2107.07933)
6. <a id="ref-6"></a>M. Drusch et al. Sentinel-2: ESA's optical high-resolution
   mission for GMES operational services. *Remote Sensing of Environment* 120,
   2012. [doi:10.1016/j.rse.2011.11.026](https://doi.org/10.1016/j.rse.2011.11.026)
7. <a id="ref-7"></a>J. T. Abatzoglou, S. Z. Dobrowski, S. A. Parks and
   K. C. Hegewisch. TerraClimate, a high-resolution global dataset of monthly
   climate and climatic water balance from 1958–2015. *Scientific Data* 5,
   2018. [doi:10.1038/sdata.2017.191](https://doi.org/10.1038/sdata.2017.191)
8. <a id="ref-8"></a>L. Hawker, P. Uhe, L. Paulo, J. Sosa, J. Savage, C. Sampson
   and J. Neal. A 30 m global map of elevation with forests and buildings
   removed. *Environmental Research Letters* 17, 2022.
   [doi:10.1088/1748-9326/ac4d4f](https://doi.org/10.1088/1748-9326/ac4d4f)
9. <a id="ref-9"></a>Z. Wang, J.-Q. Zheng, Y. Zhang, G. Cui and L. Li.
   Mamba-UNet: UNet-like pure visual Mamba for medical image segmentation.
   2024. [arXiv:2402.05079](https://arxiv.org/abs/2402.05079)
10. <a id="ref-10"></a>C. Xiao, M. Li, Z. Zhang, D. Meng and L. Zhang.
    Spatial-Mamba: Effective visual state space models via structure-aware
    state fusion. *ICLR*, 2025. [arXiv:2410.15091](https://arxiv.org/abs/2410.15091)
11. <a id="ref-11"></a>C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens and
    Z. Wojna. Rethinking the Inception architecture for computer vision.
    *CVPR*, 2016. [arXiv:1512.00567](https://arxiv.org/abs/1512.00567)
12. <a id="ref-12"></a>F. Milletari, N. Navab and S.-A. Ahmadi. V-Net: Fully
    convolutional neural networks for volumetric medical image segmentation.
    *3DV*, 2016. [arXiv:1606.04797](https://arxiv.org/abs/1606.04797)
13. <a id="ref-13"></a>I. Loshchilov and F. Hutter. Decoupled weight decay
    regularization. *ICLR*, 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
14. <a id="ref-14"></a>I. Loshchilov and F. Hutter. SGDR: Stochastic gradient
    descent with warm restarts. *ICLR*, 2017.
    [arXiv:1608.03983](https://arxiv.org/abs/1608.03983)
15. <a id="ref-15"></a>P. Micikevicius et al. Mixed precision training. *ICLR*,
    2018. [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)

## Acknowledgments

The ForTy v1 benchmark and the baseline results are from Jiang and Neumann
[[1]](#ref-1), Google DeepMind.

## License

MIT — see [LICENSE](LICENSE).
