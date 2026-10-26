# Tesla T4 performance comparison: 2026-10-26

[Performance guide](../../docs/performance.md) · [Results index](../README.md)

The Colab comparison completed all 13 stages on a Tesla T4. Baseline
`82723f3183231a36e736c1c118a866aed5ad9d6b` and candidate
`f49b0042e35593fb120bf97bb292d0de8416fd4d` used the same synthetic inputs,
initial checkpoint and 7,732,800-parameter model. Inputs are 128 × 128, batch
one. Three rounds alternate source order; each source/workload/round runs in
a fresh process with five warmups and 30 measured steps. Inference uses
float32 and training uses mixed float16. The environment was TensorFlow
2.21.0, Keras 3.13.2, Python 3.13.16 and NumPy 2.1.3.

## Findings

| Measurement | Baseline | Candidate | Interpretation |
|---|---:|---:|---|
| Inference pooled median, 90 steps | 94.197 ms | 94.834 ms | 0.68% higher latency; no stable speedup |
| Training pooled median, 90 steps | 251.490 ms | 257.948 ms | 2.57% higher latency; no stable speedup |
| Inference allocator peak, median of three rounds | 73.539 MiB | 64.164 MiB | 12.75% lower |
| Training allocator peak, median of three rounds | 251.887 MiB | 236.676 MiB | 6.04% lower |

Inference latency reductions by paired round were −3.98%, +0.36% and +3.21%;
training reductions were −0.18%, +0.58% and −5.34%. Directions varied for
both workloads. Neither supports a GPU speedup claim. Pooled medians and the
median of paired-round reductions describe different aggregations.

Allocator peaks improved in every round: 12.75% for inference and 3.84%–6.27%
for training. These are TensorFlow allocator peaks measured after warmup and
resetting the peak counter, including starting live allocations. They are
**not total or reserved GPU VRAM**. Peak above starting live allocations is
also retained in each raw report, with a different denominator.

All six training measurements completed 30 optimizer updates. Forward
comparisons before timing passed their tolerances, but GPU class maps were
not exactly identical. Float32 candidate predictions differed from the golden
array by at most `1.0431e-7`, with one different argmax pixel out of 16,384.
Mixed-precision comparisons had maximum error `6.8158e-5` and at most eight
different argmax pixels; baseline mixed-precision outputs also varied across
processes. These checks use `training=False` before training warmup. They do
not establish identical training trajectories or accuracy on real data.

## Evidence and verification

- [`comparison/comparison.json`](comparison/comparison.json): uploaded summary,
  recomputed locally from all 12 raw reports with an identical result.
- `comparison/inference-*.json` and `comparison/training-*.json`: all 360 raw
  latency samples, memory, forward checks, optimizer updates and source hashes.
- [`comparison/fixture/fixture.json`](comparison/fixture/fixture.json): shared
  input/checkpoint identities and configuration. Binary inputs, weights and
  golden arrays were excluded from the upload and cannot be rehashed locally.
- [`comparison/plan.json`](comparison/plan.json),
  [`comparison/execution.json`](comparison/execution.json) and
  [`comparison/preparation.json`](comparison/preparation.json): commands,
  alternating order, preparation and process status.
- [`source-revisions.json`](source-revisions.json) and
  [`packages.txt`](packages.txt): pinned revisions and recorded environment.
- [`audit.json`](audit.json): local checks and recomputed memory summaries.
  Reported module hashes match the pinned Git objects, and the measurement
  harness hash matches the repository script.
- [`provenance.json`](provenance.json): original ZIP SHA-256 and hashes of all
  21 retained, unaltered uploaded files. Duplicate text logs remain in the
  original upload and are omitted here.

The measurements exclude imports, tracing, data IO, preprocessing, checkpoint
loading and output export. They establish neither end-to-end throughput nor
segmentation accuracy, convergence or statistical significance. CPU gains
cannot be transferred to this GPU result. The next performance study should
profile the GPU execution and test larger batches before changing more code.
