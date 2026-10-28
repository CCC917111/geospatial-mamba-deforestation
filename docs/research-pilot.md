[Documentation](README.md) · [Project summary](project-summary.md)

# Retained multi-seed architecture pilot

This study tests whether the CNN + Mamba-inspired selective state-space model
improves segmentation over the CNN-only component control under one fixed
training cohort and budget. The workflow is implemented and verified with
three-seed synthetic CPU training, continuation and checkpoint checks. **The
real-data multi-seed GPU experiment has not run yet.** No positive architecture
effect or new accuracy score is claimed.

## Run the real-data pilot

[Open the retained research notebook in Colab](https://colab.research.google.com/github/CCC917111/geospatial-mamba-deforestation/blob/main/notebooks/colab_research_pilot.ipynb),
select an available T4 GPU and run the cells in order. The notebook checks the
GPU before data downloads, pins its source and writes model artifacts to
Google Drive. A stable `STUDY_ID` identifies one immutable protocol.

| Setting | Default |
|---|---|
| Architecture arms | `cnn_ssm`, `cnn_only` |
| Run/model seeds | 42, 43, 44 |
| Shard-selection seed | 2026, shared across all six runs |
| Fixed shard budgets | 8 training, 4 validation, 8 test, preserving original 1,024-shard filenames |
| Epoch budget | 10 per run; early stopping disabled |
| Batch / precision | 8 / mixed float16 training, float32 evaluation |
| Objective | Same ComboLoss, weights, smoothing and optimizer settings in both arms |
| Checkpoint selection | First epoch attaining the best validation macro F1 over nine numeric IDs |
| Test boundary | All six complete training runs validate before any model test evaluation |
| Retention | Six best and six last H5 files, configs, histories, manifests, selection receipts, logs and hashes |

The selection seed controls shard membership; run seeds retain initialization,
example shuffle and augmentation randomness. Both arms of each seed share the
protocol and cohort. This does not promise identical random draws or bitwise
training trajectories. Architecture order alternates across seeds. CNN-only
uses temporal mean pooling and omits both SSM blocks, so parameter count and
temporal-order handling differ. This is a component ablation, not a matched
parameter-count experiment.

The pilot stages only selected TFRecord shards into a local cache, with a
16 GiB byte budget and free-disk check. Its immutable data manifest records
source URIs, original filenames, sizes and SHA-256 hashes. A new runtime can
restage missing files only if they match the original receipts. TFRecord data
stays local; checkpoint and report files persist in Drive. Free GPU availability
and session duration vary; only successfully written files can survive
runtime reclamation.

The staged cache contains only the fixed cohort. The research runner always
uses `--fixed-train-subset`; a standalone rotating-subset training command may
request shards that were not staged.

## Continue across sessions

Keep the same source commit, `STUDY_ID` and scientific settings. The runner
rechecks source, runtime versions, data identity and completed-stage hashes.
Completed runs are skipped only after verification. An interrupted arm is
preserved under `attempts/` and restarted from its seed and epoch one. H5
weights do not contain optimizer state; this is stage continuation, not
mid-training optimizer resumption.

`MAX_TRAIN_RUNS` can limit a session to a chosen number of new complete runs.
`STOP_AFTER_TRAINING=True` retains all six runs before evaluation. These
execution limits can change between sessions; cohort, seeds, model settings,
precision and epoch budget cannot. Use a new study ID for a new protocol.

Use one active runtime per study. Normal interruption sends a graceful signal
before terminating remaining workers. After a hard runtime loss, confirm the
old runtime has stopped before setting `RECOVER_LOCK=True`. A leftover
`.research.recovery` reservation requires stopping the old runtime and
removing only that marker. Missing identity receipts, a lost execution ledger
with existing model outputs, or modified completed checkpoints fail validation;
the runner does not infer replacements or silently overwrite completed work.

Checkpoint H5 and epoch CSV writes use temporary siblings followed by
replacement. This preserves a previous complete file when a save fails under
normal filesystem semantics. It does not guarantee remote Drive durability.
Actual optimizer updates and skipped batches are logged per epoch; zero-update
epochs fail. The comparison matches example/epoch budgets and reports update
counts rather than assuming mixed-precision scaling applies every update.

## Interpret results without unconfirmed class names

The publisher has not yet answered the
[integer mapping / Unknown protocol question](https://github.com/google-deepmind/forest_typology/issues/2).
The primary `research_report.json` therefore reports only numeric IDs 0–8:

- Macro F1 averages all nine IDs, assigning absent IDs F1 zero.
- Mean IoU averages IDs with nonzero true or predicted support.
- Pixel accuracy uses all evaluated pixels; no numeric ID is ignored.
- Per-seed results, confusion matrices, common class support, parameter counts,
  update counts and paired differences are retained. Means and sample standard
  deviations across three seeds are descriptive, not significance claims.

Legacy training/test files retain the project's existing class-name convention;
their semantic forest groups are not primary research claims. An authoritative
mapping and benchmark Unknown protocol are needed before publishing semantic
forest-type scores or asserting official ForTy comparability.

The fixed validation/test budgets use leading shards. This may be
unrepresentative if shard order correlates with geography. Published split
namespaces are preserved, but neither sampled preflight records nor shard
hashes independently prove geographic separation. Preflight reads one raw
record per split for format checks; it does not evaluate a model on test data.
The small cohort and ten-epoch budget make this a pilot rather than a full
benchmark. Record a positive, neutral or negative outcome as observed.

## Inspect and retain evidence

`checkpoint_inventory.json` tracks completed training runs during execution.
After completion, `checkpoint_manifest.json` identifies all six selected best
and six last checkpoint files and their hashes, together with configuration,
epoch history, exact split membership, source/runtime/data receipts, closed
worker logs and test metrics. The primary report checks the complete frozen
seed list and never promotes a one-seed subset to the planned three-seed result.

The notebook downloads a bounded metadata-only ZIP below 32 MiB. Checkpoints
remain in Drive; `DOWNLOAD_BEST_CHECKPOINTS=True` downloads selected H5 files
separately with unique variant/seed names and checksum validation. Checkpoints
are not combined into one oversized ZIP. Source data and weights are excluded
from the metadata download, with omissions recorded explicitly.

To use another prepared GPU host, first stage and freeze a cohort:

```bash
python -m mamba_forest.stage_data \
  --source-root gs://forest_typology/forty_v1/1.0.0 \
  --cache-root /data/forty-pilot \
  --manifest runs/retained-pilot/selected_data.json \
  --train-shards 8 --val-shards 4 --test-shards 8 --selection-seed 2026
python -m mamba_forest.research --run-root runs/retained-pilot \
  --data-root /data/forty-pilot \
  --data-manifest runs/retained-pilot/selected_data.json
# Inspect plan.json, then use the identical arguments with --execute.
```

Add `--resume` for subsequent invocations. Source/software identities must
remain consistent. Choose a new run root for a changed experiment.

The retained [CPU software-check receipt](../results/research_pipeline_smoke_20261028.json)
records three seeds, two epochs, a reduced model and generated 128 × 128
records. It completed two training runs, continued with the other four,
verified the first two checkpoints were unchanged, then ran all six test
evaluations. All six best and six last files and 103 retained-file hashes were
verified. This establishes workflow behavior; it provides no real-data
accuracy or GPU result.

## Application wording

The earlier **4.80×** result is **synthetic inference compute throughput at
batch eight versus batch one on Tesla T4**, excluding data IO and exports.
It is separate from this architecture study and does not establish segmentation
accuracy, end-to-end GPU speed or an SSM-versus-CNN improvement. See the
[application brief](application-brief.md) for wording tied to retained evidence.
