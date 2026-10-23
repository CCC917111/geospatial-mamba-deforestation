[Documentation](README.md) · [Project overview](../README.md)

# Technical summary

Multi-modal forest type segmentation with convolutional encoders and a
Mamba-inspired selective state-space model.

**Project lead:** Yike Chen

**Team:** Qianhua Wan, Cherng Khai Hng, and Jiawei Li

## Context

This began as a deep-learning course project at the Chinese University of
Hong Kong, Shenzhen.

Forest type segmentation distinguishes natural forest, planted forest, and
tree crops within a nine-class land-cover taxonomy. These categories describe
different land uses even when all retain tree cover. The project implements
a TensorFlow workflow for the ForTy v1 input format, combining four seasonal
Sentinel-2 composites, elevation with slope and aspect, and climate variables.
Its defined task is pixel-level forest type segmentation. Change detection
and downstream environmental decisions require separate validation.

## Design

The architecture combines convolutional encoders with a **Mamba-inspired
selective state-space model**. Optical observations are rearranged into one
four-step sequence per pixel. Input-dependent projections control state
updates and readout; a shared low-rank projection connects channels to a
compact recurrent state. TensorFlow's `tf.scan` evaluates the recurrence.
This is a custom selective-SSM adaptation, with differences from the reference
Mamba and Mamba-2 architectures. A second block scans the downsampled spatial
grid in raster order. Five-point neighborhood fusion then introduces local
two-dimensional structure. The recurrence proceeds in one direction, while
the preceding convolution mixes neighboring positions. The
[design notes](design.md) explain these choices and implementation limits.

Terrain is encoded by convolutions, and an MLP embeds climate observations
before broadcasting them spatially. A fusion network pools modality features
and predicts sample-dependent weights. A U-Net-style encoder and decoder
combine these features with skip connections and restore the original spatial
resolution. These established components are adapted for heterogeneous
remote-sensing inputs. The selective-SSM component's benefit over a matched
CNN remains untested.

### Class imbalance

Class imbalance motivates the training objective. The default combines
weighted cross-entropy with soft Dice loss, using coefficients of 0.4 and 0.6,
respectively. Cross-entropy uses label smoothing and increased weights for
selected classes. Dice excludes absent classes from its average. These choices
are intended to support minority classes, including planted forest. The
available plain-cross-entropy comparison changes weighting, smoothing, and
Dice together, so any eventual difference would concern the combined objective.

## Implementation

Experiment integrity is an explicit part of the implementation. Seeded shard
selection records exact membership, and fixed-subset mode preserves training
membership across epochs. Validation selects checkpoints. Selection metadata
records the chosen epoch, score, and checkpoint digest. Evaluation retains
the confusion matrix, class support, model settings, runtime information, and
actual checkpoint hash. The paired runner completes all planned training
before test evaluation. Its comparator rejects mismatched budgets, incomplete
runs, or inconsistent metrics. These controls make experiments inspectable;
recorded paths alone do not prove geographic independence.

A [one-command local demo](local-demo.md) generates the actual 128-by-128
record schema and runs CPU training, evaluation, and tile prediction on
synthetic data. It retains source and data identifiers, checkpoint identity,
stage logs, and explicitly labelled input/label/prediction figures.

## Evidence

**Historical scores.**
The [results record](../results/README.md) separates historical scores from
current evidence. Historical ForTy F1 values have no retained original logs
or weights. Published baseline scores use different training and evaluation
conditions, so the table cannot establish a controlled ranking.

**Software verification.** Local verification recorded 225 passing automated
tests and synthetic CPU pipeline checks, including training, checkpoint
restoration, evaluation, and paired artifact validation. These establish
software behavior; live-data accuracy
and GPU performance remain unverified.

**CPU measurements.** The [optimization audit](optimization-audit.md) records
a synthetic CPU diagnostic for the default 128-by-128 model. After two warmup
steps, three synchronized measurements per mode gave median inference times
of 316.54 ms with eager execution and 149.01 ms with graph execution. This small diagnostic
excludes source loading and production output handling. It measures neither
end-to-end prediction latency nor GPU performance.

A separate [file-to-file benchmark](local-demo.md#repeat-the-end-to-end-benchmark)
records cold initialization, steady inference, input reads, and output writes,
with 60 measured tile samples from 30 passes over two synthetic tiles. It uses
the reduced demo checkpoint, so its timings are distinct from the
default-model compute diagnostic.

## Next research question

The next research question is whether selective state-space processing adds
value beyond a CNN under the same data, objective, and training budget. A
matched architecture ablation would examine class-level errors alongside
compute costs and retained experimental artifacts. That study has not run.
