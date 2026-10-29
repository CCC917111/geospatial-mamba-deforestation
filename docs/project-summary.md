[Documentation](README.md) · [Project overview](../README.md)

# Project summary

Multi-modal forest type segmentation with convolutional encoders and a
Mamba-inspired selective state-space model.

Yike Chen led this deep-learning course project as its **primary technical
contributor**, working with Qianhua Wan, Cherng Khai Hng and Jiawei Li at the
Chinese University of Hong Kong, Shenzhen.

## Research question

CNN-Mamba studies forest type segmentation using seasonal satellite observations
and environmental context. Natural forest, planted forest and tree crops can
share canopy cover while representing different land uses. The implemented task
predicts nine land-cover classes for each pixel of a 128 × 128 tile in the
ForTy v1 format.

The research question is whether selective state-space processing improves
segmentation over a CNN-only control under the same data and training budget.
A three-seed pilot found a small, seed-sensitive average improvement with greater
model capacity and training cost.
Deforestation change detection requires separate temporal validation.

## Approach

The TensorFlow architecture combines convolutional encoders with a
**Mamba-inspired selective state-space model**. A temporal block processes four
seasonal Sentinel-2 composites per pixel. Terrain convolutions encode elevation,
slope and aspect; an MLP embeds climate observations. Sample-dependent fusion
weights combine modalities, and a U-Net-style encoder and decoder restore
pixel-level predictions. A spatial selective-SSM block scans the downsampled
bottleneck with local neighborhood fusion.

The state-space recurrence uses input-dependent updates, a shared low-rank
state projection and TensorFlow's `tf.scan`. This custom adaptation differs
from the official Mamba and Mamba-2 implementations. The
[design notes](design.md) describe its equations, scan direction and limits.

## Engineering and reproducibility

The repository provides training, checkpoint restoration, evaluation and tile
prediction. Seeded shard manifests retain data membership; validation selects
checkpoints; evaluation saves confusion matrices, class support, configuration
and checkpoint hashes. Paired architecture and objective runners check shared
training budgets and artifacts before reporting differences. The architecture
comparison uses models with different parameter counts. Geographic independence
of the published splits has not been independently verified here.

A [one-command CPU demo](local-demo.md) generates the standard record schema
and exercises training, evaluation and prediction on synthetic data. It retains
source and data identifiers, checkpoint identity, logs and labelled figures.

## Evidence and limitations

- **Software:** 526 passing tests and synthetic CPU pipeline runs check
  training, evaluation, checkpoint restoration and prediction. Accuracy on
  satellite observations requires separate real-data evaluation.
- **Controlled real-data pilot:** Six Tesla T4 runs compare CNN-only and
  CNN+SSM at seeds 42/43/44 on one fixed 8/4/8-shard training/validation/test
  cohort, with ten epochs per run. Test Macro F1 over all nine numeric IDs is
  **38.19 ± 0.88% versus 39.30 ± 1.38%**, respectively. The paired change is
  **+1.10 ± 2.11 percentage points**, with improvement in two of three seeds.
  Mean IoU is 27.32 ± 0.82% versus 28.52 ± 0.96%. Values are means and sample
  standard deviations, not significance claims. The shared test cohort has
  146 tiles; each run uses 1,195 training examples per epoch. See the
  [audited results](../results/architecture_pilot_t4_20261029/README.md).
- **Model size and training cost:** Parameter count increases from 5,035,392 to
  7,732,800 (+53.6%). Mean training-process wall time was 339.42 seconds for
  CNN-only versus 785.15 seconds for CNN+SSM in one sequential session,
  including startup, validation and checkpoint writes; this is not a kernel
  or steady-state inference benchmark. Numeric ID 7 has 9,770 true test pixels
  but zero predictions in every model. Class semantics and the official
  Unknown protocol remain unconfirmed, so primary results use numeric IDs.
- **Artifact verification:** Recomputed metrics match all six confusion matrices;
  all 95 export-manifest file hashes and 13 source-module hashes were verified.
  The metadata records six best and six last H5 files in Drive, representing
  seven unique weight hashes. The uploaded archive excludes weights and
  TFRecords, so verification of their actual bytes and current Drive availability
  remains pending. This small-cohort pilot does not establish full-benchmark
  performance or a capacity-matched architectural effect.
- **Historical accuracy:** Reported ForTy macro F1 is 56.89% across nine classes
  and 48.81% across the three forest types. Original logs and weights are
  unavailable. Published baselines use different protocols, so these values
  cannot establish a controlled ranking or independently verified performance.
- **Performance:** Retained CPU compute and file-to-file measurements use
  different model configurations and timing boundaries. A three-round
  [Tesla T4 comparison](../results/performance_t4_20261026/README.md) found
  12.75% lower inference and 6.04% lower training TensorFlow allocator peaks,
  with no stable latency gain. A subsequent
  [batch-capacity study](../results/batch_study_t4_20261028/README.md) measured
  4.80× inference and 5.82× training compute throughput at batch eight versus
  one, using initialized weights and one repeated synthetic tile. These timings
  exclude IO, preprocessing, checkpoint loading and exports. Full-scale real-data
  training and full-test evaluation remain pending; synthetic timings do not
  establish accuracy.

The [results record](../results/README.md) links the raw artifacts. The
[validation runbook](gpu-validation.md) describes the broader real-data
protocols for full-scale training and full-test evaluation.

## Next study

The [research pilot](research-pilot.md) provides a Drive-first workflow
with one shard-selection seed, validation-selected checkpoints and checked
stage continuation. The completed pilot remains a fixed reference for further
experiments.

The next study can test longer training and a capacity-matched CNN control,
with additional seeds and a predeclared evaluation cohort. Investigate class
imbalance and underpredicted numeric IDs using training/validation data; use
a new held-out assessment for changes motivated by the observed pilot test
errors. Resolve class semantics and geographic split provenance before
claiming official benchmark comparability or semantic forest-class gains.
