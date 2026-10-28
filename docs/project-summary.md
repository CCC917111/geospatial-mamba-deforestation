[Documentation](README.md) · [Project overview](../README.md)

# Project summary

Multi-modal forest type segmentation with convolutional encoders and a
Mamba-inspired selective state-space model.

**Project lead:** Yike Chen · **Context:** Team deep-learning course project

**Team:** Qianhua Wan, Cherng Khai Hng, and Jiawei Li

## Research question

Developed at the Chinese University of Hong Kong, Shenzhen, CNN-Mamba studies
forest type segmentation using seasonal satellite observations and environmental
context. Natural forest, planted forest and tree crops can share canopy cover
while representing different land uses. The implemented task predicts nine
land-cover classes for each pixel of a 128 × 128 tile in the ForTy v1 format.

The research question is whether selective state-space processing improves
segmentation over a matched convolutional model under the same data and
training budget. A controlled architecture comparison remains to be run.
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
state projection and TensorFlow's `tf.scan`. It is a custom adaptation rather
than an official Mamba or Mamba-2 implementation. The
[design notes](design.md) describe its equations, scan direction and limits.

## Engineering and reproducibility

The repository provides training, checkpoint restoration, evaluation and tile
prediction. Seeded shard manifests retain data membership; validation selects
checkpoints; evaluation saves confusion matrices, class support, configuration
and checkpoint hashes. Paired architecture and objective runners check common training budgets and
artifacts before reporting differences; architecture parameter counts differ. Geographic independence of the
published splits has not been independently verified here.

A [one-command CPU demo](local-demo.md) generates the standard record schema
and exercises training, evaluation and prediction on synthetic data. It retains
source and data identifiers, checkpoint identity, logs and labelled figures.

## Evidence and limitations

- **Software:** CPU verification includes 484 passing tests and
  synthetic CPU pipeline runs. These establish software behavior, not accuracy
  on satellite observations.
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
  4.80× inference and 5.82× training throughput at batch eight versus one,
  using repeated synthetic inputs. Full-scale real-data training and full-test
  evaluation remain unverified; synthetic timings do not establish accuracy.

The [results record](../results/README.md) links the raw artifacts and explains
their scope. The [validation runbook](gpu-validation.md) documents prepared
real-data experiments; it does not report completed runs.

## Next study

The [retained research pilot](research-pilot.md) now provides a Drive-first
three-seed workflow with one shard-selection seed, validation-selected weights
and checked stage continuation. Its synthetic CPU integration retained six
best and six last checkpoints; the real-data GPU pilot remains pending.

Compare the selective-SSM model with a CNN control using identical split
membership, objectives and training budgets, then repeat across seeds. Report
forest-class F1, mean IoU, inference time and memory alongside retained
checkpoints and error analysis. This would test the model's added value and
provide reproducible evidence for subsequent research.
