[Documentation](README.md) · [Project overview](../README.md)

# Run the small GPU check in free Colab

This notebook prepares a small real-source architecture check on an available
Colab GPU. For the
separate synthetic before/after compute study, see the completed
[Tesla T4 record](../results/performance_t4_20261026/README.md) and
[performance notebook](../notebooks/colab_performance.ipynb).

This workflow uses Colab's free runtime without Colab Pro or enabled cloud
billing. GPU availability, session duration and usage allowances are
not guaranteed. A run may remain incomplete if no free GPU is available or the
runtime is reclaimed.
See the [Colab FAQ](https://research.google.com/colaboratory/faq.html) for the
provider's current resource rules; this workflow assumes no fixed free quota.

## Open the notebook

1. [Open directly in Colab](https://colab.research.google.com/github/CCC917111/geospatial-mamba-deforestation/blob/main/notebooks/colab_free_gpu.ipynb)
   and sign in to your Google account. You can also download the
   [notebook](../notebooks/colab_free_gpu.ipynb) and upload it at
   [Google Colab](https://colab.research.google.com).
2. In **Runtime → Change runtime type**, select an available GPU in the free
   runtime. If none is available, stop and try another session later.
3. Review the settings cell, including the optional Drive setting below. Run
   cells in order. Source is cloned from the public GitHub repository and
   checked out at the notebook's immutable `SOURCE_COMMIT`.
4. Confirm the source identity and GPU probe pass before the data and
   experiment cells continue. Retain the logs if either step fails.

No recovery archive is needed by default. `UPLOAD_SOURCE_BUNDLE = True`
enables the optional upload path for a `.bundle` or recovery `.zip` containing
`geospatial-mamba-deforestation.bundle`. That bundle must contain the pinned
commit; an older recovery archive may not. Both paths reject a source mismatch
or a modified working tree. Use a fresh workspace when changing revisions.

The notebook installs the package editable with `--no-deps`, preserving Colab's
TensorFlow installation. GPU compatibility is checked by an actual GPU
operation, and resolved package versions are retained with the session.

## What the default run does

The default settings use `COMPARISON = "architecture"`, the full-width models,
batch size one and one paired seed. Both arms use the same ComboLoss, seeded
fixed training shard, one validation shard and one test shard:

- CNN + selective SSM (`cnn_ssm`): **two training epochs**.
- CNN-only (`cnn_only`): **two training epochs**.

CNN-only averages optical seasons and omits both SSM blocks; normalization,
fusion, CNN stages and local SASF remain. Parameter counts differ. This is a
component ablation under a common training budget. Set `COMPARISON = "objective"`
to run the separate ComboLoss versus plain-cross-entropy check on CNN + SSM.

The GPU gate precedes source access. Dataset preflight inspects one raw record
from each selected train, validation and test shard for schema, shapes, finite
inputs and valid labels. That check covers three records. It does not verify
the whole source or geographic isolation.

After both arms finish training and their artifacts pass checks, the runner evaluates their selected
checkpoints on the same test shard and compares retained confusion matrices,
checkpoint hashes and experiment settings. The small run's scores are
diagnostic. They do not establish model ranking, reproduce the historical
benchmark or support accuracy claims. Keep test results out of model tuning.

## Save outputs before a runtime ends

`SAVE_TO_DRIVE` defaults to `False`. With that setting, session files under
`/content` are temporary. Download the run-artifact archive produced by the
notebook and retain the pinned source revision and any supplied recovery archive.

To save outputs to Drive, set `SAVE_TO_DRIVE = True` **before running the setup
and experiment cells**. The notebook then asks you to sign in and authorize a
Google Drive mount. Its `SESSION_ROOT` stores run outputs and checkpoints on
Drive from the start. Use an account with enough available space; free Drive
capacity is bounded and storage availability is not guaranteed.

A reclaimed runtime may lose `/content` immediately, and Python `finally`
blocks or final download/archive cells may never run. Drive helps retain
completed files, but an interrupted write can leave a partial checkpoint or
archive. Check the saved status, logs and hashes before using recovered files.

Training does not resume optimizer state after interruption. Start a fresh
session root for a retry and rerun the workflow; preserve the previous root
for diagnosis. Reusing a failed run directory cannot safely continue training.

## Inspect publisher label metadata

[Open the CPU metadata check in Colab](https://colab.research.google.com/github/CCC917111/geospatial-mamba-deforestation/blob/main/notebooks/forty_metadata_check.ipynb)
to collect the source metadata without another training run. Keep hardware
acceleration set to **None / CPU**, run its cells in order, and download the
small `forty-source-metadata-*.zip` report. It uses a fresh directory per
attempt and bounds metadata downloads; no TFRecord shards are read.

Successful collection does not verify class-ID semantics. The report keeps
`mapping_status` unresolved if no explicit names are found. Publisher names,
if present, still require review against stored IDs; the notebook never changes
the training configuration or silently relabels a class.

## Follow-up validation

Retain the complete session artifacts, including source revision, package/GPU
metadata, preflight reports, commands, logs, configs, split manifests,
checkpoints, hashes and comparison JSON. Report actual completion status and
observed measurements. An unavailable GPU or interrupted session remains an
incomplete validation.

The [cloud environment guide](cloud-environment.md) covers another prepared
runtime when resources become available. The [GPU validation runbook](gpu-validation.md)
describes calibration and the larger paired experiment needed beyond this
small Colab check.
