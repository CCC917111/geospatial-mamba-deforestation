[Documentation](README.md) · [Project summary](project-summary.md)

# Application brief

**Project:** Reproducible Multimodal Forest-Type Segmentation and GPU Performance Engineering

**Role:** Yike Chen, Project Lead and Primary Technical Contributor, team project

**Team:** Yike Chen, Qianhua Wan, Cherng Khai Hng and Jiawei Li

The project combines CNNs with a **Mamba-inspired selective state-space model**
to segment forest-type benchmark imagery from seasonal optical, terrain and
climate inputs. Its retained strengths are auditable data selection,
checkpoint-aware inference, controlled experiment tooling and GPU performance
analysis. The implemented task is forest-type segmentation.

## Verified engineering evidence

- A complete training, validation, checkpoint-restoration, evaluation and
  tile-export workflow, with source/configuration/data identities.
- Tesla T4 batch scaling across three alternating rounds. **Synthetic inference
  compute throughput increased from 10.18 to 48.82 tiles/s, or 4.80×, at batch
  eight versus batch one.** This excludes input/output handling and uses one
  repeated synthetic tile with initialized weights. It is not an architecture
  accuracy result or a production file-to-file speedup.
- Four real CUDA profiler captures, raw timing samples and reproducible event
  analysis. Recorded GPU-event fractions are not utilization or occupancy.
- Bounded prediction batches preserve per-tile climate normalization and
  export order. Versioned timing reports separate measured batch latency from
  allocated per-tile costs.

## Audited research pilot

The [real-data pilot](../results/architecture_pilot_t4_20261029/README.md)
completed six Tesla T4 training/evaluation runs on 2026-10-29: CNN-only and
CNN+SSM at seeds 42, 43 and 44, using one fixed ForTy cohort, the same objective,
and ten epochs per run. All training completed before test evaluation.

| Test metric | CNN-only | CNN+SSM |
|---|---:|---:|
| Macro F1 over nine numeric IDs | 38.19 ± 0.88% | 39.30 ± 1.38% |
| Mean IoU | 27.32 ± 0.82% | 28.52 ± 0.96% |

Values are means and sample standard deviations across three seeds. The paired
Macro F1 change was **+1.10 ± 2.11 percentage points**, improving in two of
three seeds. This is a small, seed-sensitive average benefit. CNN+SSM has
7.73M parameters versus 5.04M for CNN-only, a **53.6% increase**; the comparison
does not isolate an effect at matched model capacity. Numeric ID 7 received
no predicted pixels in any of the six models despite positive test support.

Independent recomputation from six retained confusion matrices matches the
reported metrics. All 95 export-manifest file hashes and 13 recorded source
module hashes were checked. Metadata records six best and six last checkpoint
files in Drive; their binary weights and the TFRecords are excluded from the
uploaded archive and were not independently rehashed. The study uses a small
cohort and does not establish full-benchmark performance, statistical
significance or geographic independence. Semantic class naming and the official
Unknown protocol remain awaiting publisher clarification.

## CV wording

> **Project Lead and Primary Technical Contributor — Multimodal Forest-Type Segmentation**

> - Led a four-person team project developing a reproducible TensorFlow pipeline
>   for multimodal forest-type segmentation with CNNs and a Mamba-inspired
>   selective state-space model.
> - Fixed-cohort, three-seed architecture pilot:
>   +1.10 percentage-point mean Macro F1 change versus CNN-only, improvement in
>   two of three seeds and 53.6% more parameters.
> - Tesla T4 batch benchmark: 4.80× synthetic inference compute throughput at batch 8
>   versus batch 1 on repeated synthetic input with initialized weights,
>   excluding data IO and exports.

The confirmed role combines project leadership and primary technical
contribution. Describe specific module ownership through actual commits,
design decisions and collaboration; keep the team attribution alongside it.
The [interview preparation](application-interview.md) includes a three-minute
project talk, a short statement paragraph and technical questions.

## Technical discussion

Explain how the shard-selection seed is separated from run randomness, why
climate statistics must be computed per tile over seasons, how validation
chooses a checkpoint before test evaluation, and how profiler overhead and
batch latency differ from unprofiled throughput. Discuss the negative seed,
parameter overhead and unpredicted numeric ID alongside the average benefit.
Any changes motivated by the observed test errors need a new held-out
assessment; the completed pilot should remain an unchanged evidence record.
