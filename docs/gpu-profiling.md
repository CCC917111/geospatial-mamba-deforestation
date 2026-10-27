[Documentation](README.md) · [Performance results](performance.md)

# GPU profiling and batch scaling

The completed [T4 source comparison](../results/performance_t4_20261026/README.md)
found lower TensorFlow allocator peaks without a stable latency gain at batch
one. This next study measures batch capacity and captures execution traces
before selecting further optimizations. Its GPU execution is pending; CPU
integration checks verify the workflow and real CPU trace export/restore.

## Run on Colab

1. [Open the profiling notebook](https://colab.research.google.com/github/CCC917111/geospatial-mamba-deforestation/blob/main/notebooks/colab_gpu_profiling.ipynb).
2. Select **T4 GPU** or another available GPU, then run the cells in order.
   The notebook pins one source commit and preserves Colab's TensorFlow stack.
3. Keep the downloaded `batch-study-bundle.zip`. If the combined files exceed
   the transfer budget, keep `metadata.zip`, `profile-archives.json` and every
   numbered trace ZIP instead. Every ZIP is below 32 MiB; traces are split
   losslessly. A failed run can still export its logs and available traces.

No ForTy download, class mapping, paid API or Colab subscription is required.
Free GPU availability and runtime duration vary. `/content` is temporary;
enable `SAVE_TO_DRIVE` before starting for persistent report/trace writes.
An interrupted session starts a fresh directory and reloads initial weights;
optimizer state is not resumed. A reclaimed runtime can prevent final export.

## Protocol and timing boundaries

| Setting | Default |
|---|---|
| Model | Current optimized CNN + selective SSM, 7,732,800 parameters |
| Input | One seeded synthetic 128 × 128 tile repeated within each batch |
| Batch sizes | 1, 2, 4, 8 |
| Weights | One shared seeded initial checkpoint; optional matching full-model checkpoint |
| Inference / training precision | Float32 / mixed float16 |
| Timing budget | Three alternating batch-order rounds; five warmups and 30 samples per case |
| Isolation | One fresh process per workload, batch and round |
| Profiling | Separate fresh processes after timing, batch one and the largest fully measured batch per workload |
| Capture budget | Five warmups, then three synchronized captured steps |

All batches use the same checkpoint and replicate the same base tile on the
host. Forward predictions are compared with the repeated base golden array
within dtype-specific tolerances; argmax agreement is recorded separately.
This removes data differences from a compute-capacity study. It does not
simulate training diversity or establish convergence at a larger batch size.
Every measured or profiled training step must update the optimizer. Warmup
loss-scale adaptation is recorded separately.

Throughput is `batch_size × measured_steps / sum(step_seconds)`. Batch latency
is the time for the whole batch. Amortized time per tile divides batch latency
by batch size and is not single-request response latency. Inspect all rounds;
the highest observed throughput batch is descriptive, not an optimal training
setting or a statistically significant result.

Only an explicit `ResourceExhaustedError` above batch one is recoverable. That
case is marked out of memory, remaining rounds of the same case are skipped,
and partial samples are excluded from throughput comparisons. Other errors
stop the study. A profiling failure preserves completed timing in `study.json`
with `timing_passed` status and a failed `execution.json`.

Imports, tracing, IO, preprocessing, loading and exports are outside compute
timings. TensorFlow allocator peaks are measured after warmup and include
starting live allocations; they are not total/reserved GPU VRAM. Profiles run
after all timing jobs and are excluded from timing and memory summaries.

## Inspect the profiles

Set `SHOW_PROFILE=True` to enable the optional viewer cell **after timing**.
It installs [XProf](https://openxla.org/xprof) with constraints protecting the
measured TensorFlow, TensorBoard, Keras, NumPy, protobuf and ml-dtypes versions.
Dependency conflicts stop the viewer installation; they do not invalidate or
erase the completed measurements. Exact Colab viewer compatibility is not
assumed. Package versions after viewer installation are recorded separately.
Rerun the download cell after using the viewer to include that environment
record in the refreshed export.

In TensorBoard's **Profile** tab:

- Confirm GPU stream tracks in **Trace Viewer** and populated **GPU Kernel
  Stats** before treating the capture as a GPU profile. The report initially
  marks GPU trace coverage unverified; a nonempty XPlane file can be host-only.
- Inspect host launch gaps, recurrent SSM work, convolution kernels and device
  idle intervals. `profile-*.json` maps model attributes to existing layer
  names, including temporal/spatial SSMs, modality projections and CNN stages.
- Compare batch one with the larger capture. Operation totals and overlapping
  kernels do not directly sum to synchronized wall time. Use traces to explain
  behavior, and use the separate unprofiled rounds for latency and throughput.
- Export **GPU Kernel Stats** as CSV for review alongside the raw traces.

Missing GPU tracks require checking CUPTI loading/symbol/privilege diagnostics;
an initial library-search warning alone does not prove capture failure. The
workflow does not modify CUDA installations or system privileges. See the
official [capture troubleshooting](https://openxla.org/xprof/capturing_profiles),
[GPU Kernel Stats guide](https://openxla.org/xprof/gpu_kernel_stats) and
[TensorFlow capture guidance](https://openxla.org/xprof/tensorflow_profiling).
The default three captured steps stay within the recommended short capture.

## Run on another prepared host

Use a fresh run directory:

```bash
python scripts/study_batches.py --run-root runs/batch-study --device /GPU:0 \
  --batch-sizes 1 2 4 8 --rounds 3 --warmup 5 --steps 30 --profile-steps 3
python scripts/archive_profiles.py --run-root runs/batch-study \
  --output-dir runs/batch-study-export
```

The exporter writes metadata and complete trace chunks into a new directory
outside the run root. Metadata omits checkpoints/arrays and records size
omissions. Each trace chunk has at most 24 MiB of raw bytes. The manifest
records archive, chunk and whole-trace SHA-256 hashes; no protobuf is truncated.
The CLI emits multiple files; the notebook bundles them into one ZIP when it
fits below the transfer budget.

Unzip the notebook's outer bundle first, when present, then restore traces:

```bash
python scripts/archive_profiles.py --restore downloads/profile-archives.json \
  --archive-dir downloads --destination restored-study
```

Output paths preserve the original relative structure. A notebook export uses
`restored-study/study/profiles` as the viewer log directory; a direct CLI
export of the run root uses `restored-study/profiles`. Restore verifies all
hashes and sizes, rejects unsafe paths and refuses to overwrite a destination.

For a small CPU functionality check, use float32 training and reduced inputs:

```bash
TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=2 \
python scripts/study_batches.py --run-root runs/batch-study-cpu --device /CPU:0 \
  --batch-sizes 1 2 --rounds 1 --steps 2 --warmup 1 --profile-steps 2 \
  --training-precision float32 --tile-size 8 --d-model 4 \
  --d-state-temporal 2 --d-state-spatial 2 --bottleneck-channels 8
```

This CPU check exercises nine processes and four captures. It does not measure
T4 throughput, validate CUPTI collection or establish real-data accuracy.
