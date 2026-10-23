# Run the small GPU check in free Colab

**Status: prepared; Colab GPU execution is pending.** The current workspace is
CPU-only. Local code and source-restoration checks do not establish that the
notebook has run successfully on Colab or on a GPU.

This workflow uses Colab's free runtime, without a paid API, Colab Pro or
enabling billing. GPU availability, session duration and usage allowances are
not guaranteed. A run may remain incomplete if no free GPU is available or the
runtime is reclaimed.
See the [Colab FAQ](https://research.google.com/colaboratory/faq.html) for the
provider's current resource rules; this workflow assumes no fixed free quota.

## Open the notebook

1. Download the [notebook](../notebooks/colab_free_gpu.ipynb) and the supplied
   `publication-recovery.zip` from the handoff. Keep both files locally. A
   supplied standalone Git bundle can also restore the source.
2. Open [Google Colab](https://colab.research.google.com), choose **Upload
   notebook**, and upload `colab_free_gpu.ipynb`.
3. In **Runtime → Change runtime type**, select an available GPU in the free
   runtime. If none is available, stop and try another session later.
4. Review the settings cell, including the optional Drive setting below. Run
   cells in order. When the source-upload cell opens a file chooser, upload
   `publication-recovery.zip` or the supplied Git bundle.
5. Confirm source restoration and the GPU probe pass before allowing the data
   and experiment cells to continue. Retain the logs if either step fails.

The notebook restores prepared codebase `ae6a123` from the recovery archive and
checks the source revision. Use the supplied archive for this workflow; the
notebook does not depend on the prepared branch already being published to
GitHub. It installs the package editable with `--no-deps`, preserving Colab's
TensorFlow installation. GPU compatibility is established by an actual GPU
operation, and the resolved package versions are retained with the session.

## What the default run does

The default settings use the full model, batch size one and one paired seed.
Both objective arms use the same seeded, fixed training shard, one validation
shard and one test shard:

- ComboLoss: **two training epochs**.
- Plain unweighted cross-entropy: **two training epochs**.

The GPU gate precedes source access. Dataset preflight inspects one raw record
from each selected train, validation and test shard for schema, shapes, finite
inputs and valid labels. That check covers three records. It does not verify
the whole source or geographic isolation.

After both arms finish training, the runner evaluates their selected
checkpoints on the same test shard and compares retained confusion matrices,
checkpoint hashes and experiment settings. The small run's scores are
diagnostic. They do not establish model ranking, reproduce the historical
benchmark or support accuracy claims. Keep test results out of model tuning.

## Save outputs before a runtime ends

`SAVE_TO_DRIVE` defaults to `False`. With that setting, session files under
`/content` are temporary. Download the run-artifact archive produced by the
notebook and retain the source recovery archive separately.

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
