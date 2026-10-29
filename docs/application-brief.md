[Documentation](README.md) · [Project summary](project-summary.md)

# Application brief

**Project:** Reproducible Multimodal Forest-Type Segmentation and GPU Performance Engineering

**Role:** Yike Chen, Project Lead and Primary Technical Contributor, team project

**Team:** Yike Chen, Qianhua Wan, Cherng Khai Hng and Jiawei Li

The project combines CNNs with a custom **Mamba-inspired selective state-space
model** for forest-type segmentation using seasonal optical, terrain and
climate inputs. The TensorFlow pipeline supports auditable data selection,
checkpoint-aware inference, controlled experiments and GPU performance analysis.

## Verified engineering evidence

- A complete training, validation, checkpoint-restoration, evaluation and
  tile-export workflow, with recorded source, configuration and data identities.
- Tesla T4 batch scaling across three alternating rounds. **Synthetic inference
  compute throughput increased from 10.18 to 48.82 tiles/s, or 4.80×, at batch
  eight versus batch one.** This compute-only benchmark uses one repeated
  synthetic tile and initialized weights, excluding input/output handling and
  exports. It does not measure production file-to-file speedup.
- Four real CUDA profiler captures, raw timing samples and reproducible event
  analysis. Recorded GPU-event fractions are not utilization or occupancy.
- Bounded prediction batches preserve per-tile climate normalization and
  export order. Versioned timing reports separate measured batch latency from
  allocated per-tile costs.

## Audited research pilot

The [real-data pilot](../results/architecture_pilot_t4_20261029/README.md)
completed six Tesla T4 training/evaluation runs on 2026-10-29: CNN-only and
CNN+SSM at seeds 42, 43 and 44. Every run used the same ForTy cohort of eight
training, four validation and eight test shards, the same objective, ten epochs
and batch size eight. Validation selected each checkpoint, and all training
finished before test evaluation.

| Test metric | CNN-only | CNN+SSM |
|---|---:|---:|
| Macro F1 over nine numeric IDs | 38.19 ± 0.88% | 39.30 ± 1.38% |
| Mean IoU | 27.32 ± 0.82% | 28.52 ± 0.96% |

Values are means and sample standard deviations across three seeds. The paired
Macro F1 change was **+1.10 ± 2.11 percentage points**, improving in two of
three seeds. This is a small, seed-sensitive average benefit. CNN+SSM has
7.73M parameters versus 5.04M for CNN-only, a **53.6% increase**; the comparison
changes model capacity and temporal-order handling together. Recorded
training-process wall time was about 2.31 times as long for CNN+SSM, including setup,
validation and saving. Numeric ID 7 had 9,770 true test pixels but received no
predictions from any of the six models.

Independent recomputation from six retained confusion matrices matches the
reported metrics. All 95 export-manifest file hashes and 13 recorded source
module hashes were checked. Metadata records six best and six last checkpoint
files in Drive; their binary weights and the TFRecords are excluded from the
uploaded archive and were not independently rehashed. The study uses a small
cohort and does not establish full-benchmark performance, statistical
significance or geographic independence. Semantic class naming and the official
Unknown protocol await publisher clarification.

## CV wording

> **Project Lead and Primary Technical Contributor — Multimodal Forest-Type Segmentation**

> - Led a four-person team developing a reproducible TensorFlow pipeline
>   for multimodal forest-type segmentation with CNNs and a Mamba-inspired
>   selective state-space model.
> - Fixed-cohort, three-seed architecture pilot:
>   +1.10 percentage-point mean Macro F1 change versus CNN-only, improvement in
>   two of three seeds and 53.6% more parameters.
> - Tesla T4 batch benchmark: 4.80× synthetic inference compute throughput at batch 8
>   versus batch 1 on repeated synthetic input with initialized weights,
>   excluding data IO and exports.

Specific module ownership should be supported by actual commits, design
decisions and collaboration examples, with the team credited alongside them.
The [interview preparation](application-interview.md) includes a three-minute
project talk, a short statement paragraph and technical questions.

## Technical discussion

Shard selection is controlled separately from model randomness, climate
statistics are computed per tile across seasons, and validation selects a
checkpoint before test evaluation. Profiler overhead and whole-batch latency
need to be distinguished from unprofiled per-tile throughput. The negative seed,
parameter overhead and failure to predict ID 7 limit the average gain.
Changes motivated by the observed test errors require a new held-out
assessment; the completed pilot remains an unchanged evidence record.
