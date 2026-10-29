[Application brief](application-brief.md) · [Audited pilot](../results/architecture_pilot_t4_20261029/README.md)

# Application and interview preparation

**Role:** Yike Chen, Project Lead and Primary Technical Contributor

**Team:** Yike Chen, Qianhua Wan, Cherng Khai Hng and Jiawei Li

The drafts below describe confirmed project results and the primary technical
role. Use actual commits, design decisions and collaboration examples when
answering questions about individual module ownership.

## Three-minute project talk

I led a four-person team project on multimodal forest-type segmentation and
served as its primary technical contributor. Our research question was whether
selective state-space processing could improve segmentation over a
convolutional component control under a common data and training budget.

The TensorFlow pipeline combines four seasonal optical observations with
terrain and climate inputs. Convolutional encoders capture local features,
while custom temporal and spatial selective state-space blocks aggregate
sequence information. I describe the architecture as Mamba-inspired because
its input-dependent recurrence follows that idea, while the implementation
uses a custom shared state projection and TensorFlow scan operations.

We evaluated CNN+SSM against CNN-only in a reduced-data ForTy pilot. The same
eight training, four validation and eight test shards were fixed across model
seeds 42, 43 and 44. Each run used ten epochs and batch eight, with a shared
objective and example budget. Validation selected each checkpoint, and all six
training runs finished before test evaluation. The shared test cohort contained
146 tiles. Source identities, configurations, split membership, histories and
checkpoint digests were retained so the comparison could be inspected.

Macro F1 averaged 39.30 percent for CNN+SSM and 38.19 percent for CNN-only.
The paired change was plus 1.10 percentage points, with a sample standard
deviation of 2.11 points: two seeds improved and one regressed. This supports
a small, seed-sensitive average effect. The SSM model also had 53.6 percent
more parameters and about 2.31 times the recorded training-process wall time,
including setup, validation and saving. Capacity and temporal aggregation
changed together, which limits attribution to the recurrence itself.

Error analysis exposed a coverage failure: numeric ID 7 had 9,770 true test
pixels but received no predictions from any model. We use numeric IDs because
the publisher's class mapping and Unknown protocol remain unconfirmed.

Separately, a Tesla T4 batch study measured 4.80 times synthetic inference
compute throughput at batch eight versus one. It used repeated synthetic
input and initialized weights, excluding file IO and exports. The figure
describes that compute benchmark.

The project now connects model design, controlled evaluation, error analysis
and systems measurement through an auditable record. Next steps include a
capacity-matched control and a larger predeclared study. Changes motivated by
the observed test errors would need a new held-out assessment.

## Short statement paragraph

As Project Lead and Primary Technical Contributor on a four-person
forest-type segmentation project, I worked on the question of whether
Mamba-inspired selective state-space processing adds value to a multimodal
CNN pipeline. A controlled three-seed ForTy pilot connected model evaluation
with reproducible data selection, validation-based checkpoint choice and
retained experiment records. The SSM model showed a small average Macro F1
increase, alongside seed sensitivity and 53.6% more parameters. A separate
Tesla T4 study measured 4.80× synthetic inference compute throughput at batch
eight versus one using repeated synthetic input and initialized weights,
excluding file IO. These results made the tradeoffs between
accuracy, model capacity and execution cost concrete. The project provides a
foundation for graduate research on reproducible geospatial machine learning
and empirical evaluation of efficient sequence models.

## Technical questions

### What was your contribution, and how did the team work together?

I led the project and was its primary technical contributor. Our four-person
team delivered the segmentation pipeline, controlled experiments and
performance evidence. For module-level questions, I would connect my account
to concrete commits, design decisions and examples of collaboration. The
team attribution remains part of the project description.

### Why call the architecture Mamba-inspired?

The temporal and spatial blocks use input-dependent selective recurrence,
implemented with a shared state projection and TensorFlow scan operations.
This is the project's custom adaptation. Its equations and runtime behavior
are documented in the [design notes](design.md); the name does not claim an
official Mamba or Mamba-2 implementation.

### How did you control data selection and avoid using test data for selection?

Shard-selection seed 2026 fixed one cohort independently of the three model
seeds. Validation chose the first checkpoint attaining maximum nine-ID Macro
F1; test evaluation began after all six training runs finished. Published
split namespaces were preserved. Geographic separation has not been
independently verified because the pipeline does not retain geographic block
IDs, and leading validation/test shards may be unrepresentative.

### Does the CNN-only comparison isolate the selective recurrence?

It is a component ablation with the same cohort, objective and example/epoch
budget. CNN-only replaces temporal aggregation with a mean and removes both
SSM blocks, changing temporal-order handling and parameter count together.
CNN+SSM has 7.73M parameters versus 5.04M. A capacity-matched control and a
separate temporal-aggregation control would help attribute the observed effect.

### How convincing is the three-seed improvement?

The paired Macro F1 changes were +3.19, −1.03 and +1.14 percentage points,
giving +1.10 ± 2.11 points as mean and sample standard deviation. These are
descriptive results from three model seeds on one shared test cohort. They
show a small average benefit with a negative seed, rather than a stable or
statistically established advantage. The test pixels are not independent
replications across seeds.

### What did the failure analysis reveal?

Numeric ID 7 had 9,770 true test pixels, or about 0.41%, and zero predictions
in all six runs. ID 5 also varied substantially across seeds. These are
observed errors, without confirmed semantic class names. I would investigate
class support and behavior on training/validation data; changes motivated by
the pilot's test errors require a new held-out assessment.

### What does the 4.80× performance figure measure?

It is synthetic inference compute throughput on Tesla T4 at batch eight
relative to batch one: 48.82 versus 10.18 tiles per second. The study used
one repeated synthetic input, initialized weights and three alternating
rounds, excluding file IO and export. Whole-batch latency and per-tile
throughput answer different questions. The real-data pilot's process times
include setup and checkpoint handling, so they have a separate timing scope.

## Evidence to have ready

Use the [pilot report](../results/architecture_pilot_t4_20261029/README.md),
[paired-seed figure](../results/architecture_pilot_t4_20261029/paired-seed-results.svg),
[batch study](../results/batch_study_t4_20261028/README.md) and
[source implementation](../src/mamba_forest/layers.py) to support the account.
The uploaded pilot archive supports independent metadata and confusion-matrix
checks. Actual H5 weights and TFRecords were excluded, so the local metadata
audit did not reload those checkpoints or repeat training/evaluation.
