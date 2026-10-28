# Tesla T4 batch study: 2026-10-28

[Performance guide](../../docs/performance.md) · [Profiling protocol](../../docs/gpu-profiling.md)

All 29 stages completed: preparation, 24 timing processes and four separate
profiler captures. Each timing process has five warmups and 30 measured steps;
three rounds alternate ascending/descending/ascending batch order. The
7,732,800-parameter model uses one initial checkpoint and one synthetic
128 × 128 tile repeated within the batch. Source is the clean pinned commit
`fac8b8da180b6bca2ce7dc6267d850bba551526b`. The runtime is Tesla T4,
TensorFlow 2.21.0, Keras 3.13.2, Python 3.13.16 and NumPy 2.1.3.

## Batch capacity

![T4 throughput by batch size](../../docs/figures/t4-batch-scaling.svg)

The plot shows pooled throughput and the range of the three round rates.
The band is descriptive, not a confidence interval. Regenerate the SVG/PNG
with [`plot_batches.py`](plot_batches.py) in a Matplotlib environment.

| Batch | Inference tiles/s | Training tiles/s | Inference batch median | Training batch median | Inference peak MiB | Training peak MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10.179 | 3.886 | 94.336 ms | 255.112 ms | 64.164 | 238.648 |
| 2 | 19.417 | 7.308 | 99.929 ms | 269.997 ms | 93.000 | 366.768 |
| 4 | 30.981 | 14.069 | 129.682 ms | 279.868 ms | 176.840 | 615.785 |
| 8 | **48.816** | **22.613** | 163.257 ms | 350.433 ms | 445.846 | 1121.292 |

Throughput is `batch_size × samples / sum(step_seconds)`. Batch eight has
**4.80× inference throughput** and **5.82× training throughput** relative to
batch one. Every paired round improves: inference ratios range from 4.22× to
5.53× and training ratios from 5.80× to 5.83×. Batch eight has the highest
tested throughput; larger batches were not tested, so it is not proven optimal.

The whole batch takes longer despite processing more tiles per second. At
batch eight, dividing the pooled median by eight gives 20.41 ms/tile for
inference and 43.80 ms/tile for training. These are amortized times, not
single-request response latencies. Memory columns are medians of the three
TensorFlow allocator peaks after warmup; they are not total/reserved GPU VRAM.
No batch ran out of memory. Training's repeated synthetic examples do not
establish convergence, sample efficiency or a learning-rate recommendation.

All 12 timed training processes completed 30 optimizer updates, in addition
to five warmup updates. Both training captures completed three captured
updates. Forward checks passed their tolerances. FP32 maximum absolute error
was `2.8312e-7`, with identical argmax maps on this fixture. Mixed checks had
maximum error `7.9438e-5` and at most 13 differing argmax pixels per repeated
tile. These are checks before training, not accuracy measurements.

## What the traces show

All four restored XPlane files contain actual `/device:GPU:0` CUDA compute
and copy streams for the T4. No XSpace errors or warnings are recorded.
The uploaded reports conservatively mark GPU coverage unverified; the
separate [`trace-analysis.json`](trace-analysis.json) now verifies event
coverage without altering those original reports.

| Capture, three steps | Compute-kernel events | Recorded GPU activity / profiled host wall | Scan-condition copy duration on GPU |
|---|---:|---:|---:|
| Inference, batch 1 | 4,695 | 8.45% | 1.668 ms |
| Inference, batch 8 | 5,094 | 35.35% | 1.728 ms |
| Training, batch 1 | 26,217 | 14.36% | 2.287 ms |
| Training, batch 8 | 26,427 | 29.35% | 1.575 ms |

Each capture records **2,313 one-byte forward spatial-scan condition
transfers**: 1,542 host-to-device and 771 device-to-host over three steps.
That matches two uploads and one download per condition evaluation for
257 evaluations per 256-token spatial scan. Training also records
TensorList-gradient transfers. This motivates investigating launch/control
overhead, but the short transfer durations do not prove it dominates latency.

The activity fraction merges overlapping recorded GPU-event intervals across
streams and clips them to the synchronized host Trace steps. It is **not GPU
utilization or occupancy**. Profiler wall times include capture overhead and
must not replace the separate unprofiled timing samples. Operation-duration
sums can overlap; scan envelopes are not exclusive scan costs.

The benchmark wraps its entire worker in `tf.device('/GPU:0')`, including
control flow. Normal production prediction/training use automatic placement.
The observed scalar transfers may depend on that difference. A controlled
placement comparison is needed before attributing a deployment bottleneck or
rewriting the recurrence. Production batch throughput must also be measured
separately with input/output handling included.

## Retained evidence

- [`study/study.json`](study/study.json): original summary, independently
  recomputed from all 24 raw timing reports with an identical result.
- `study/inference-*.json`, `study/training-*.json`: all 720 latency samples,
  update counts, forward checks, memory and source identities.
- `study/profile-*.json`: capture configuration, existing layer names and
  complete trace sizes/hashes.
- [`study/plan.json`](study/plan.json), [`study/execution.json`](study/execution.json),
  [`study/fixture/fixture.json`](study/fixture/fixture.json),
  [`source-revision.json`](source-revision.json), [`packages.txt`](packages.txt):
  protocol, commands, device probe, shared fixture and environment.
- [`audit.json`](audit.json) and [`provenance.json`](provenance.json): checks,
  archive identity and hashes of 37 byte-identical uploaded metadata files.
- [`profile-archives.json`](profile-archives.json): lossless trace reconstruction
  manifest. Binary traces and duplicate logs remain in the original uploaded
  bundle and local review directory; they are not committed to Git.
- [`analyze_xplane.py`](analyze_xplane.py): reproducible offline event inspection
  using TensorFlow 2.21's packaged `tsl/profiler/protobuf/xplane.proto` schema,
  with schema/script/trace hashes recorded in the analysis JSON. This is an
  artifact-specific analysis, not a stable project API or XProf Kernel Stats
  export. TensorFlow packaging may change the schema import path.

To repeat event inspection, unpack the outer bundle into `downloads`, unpack
`downloads/metadata.zip` into `review/metadata`, then use the trusted repository
restorer and the analysis script in a TensorFlow 2.21 environment:

```bash
python scripts/archive_profiles.py --restore downloads/profile-archives.json \
  --archive-dir downloads --destination review/restored
python results/batch_study_t4_20261028/analyze_xplane.py --review-root review
```

The original upload excludes binary inputs, weights and golden predictions;
their bytes cannot be revalidated locally. The measurements exclude IO,
preprocessing, checkpoint loading and exports. They support a synthetic
batch-capacity finding, not real-data accuracy or end-to-end deployment gains.
