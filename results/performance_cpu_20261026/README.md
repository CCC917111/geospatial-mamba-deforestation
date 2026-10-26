# CPU performance comparison: 2026-10-26

These are synchronized synthetic-compute measurements, not real-data accuracy
or GPU results. Baseline source is `82723f3`; the measured core candidate is
`6289f22`. Both used the same 128 x 128 input batch, initial checkpoint and
7,732,800-parameter CNN + selective SSM model. TensorFlow 2.21.0, float32,
batch one, two intra/inter-op threads, five warmups and 30 measured steps per
process were used. Three rounds alternate revision order. Each workload and
revision runs in a fresh process.

- `comparison-validated.json`: summary recomputed from frozen raw reports with
  additional source-path, fixture-shape and sample-count checks.
- `comparison.json`: original summary from the timed harness revision.
- `inference-*.json` / `training-*.json`: raw samples and forward agreement.
- `fixture.json`: source, input/checkpoint identities and model configuration;
  large binary fixture/checkpoint files remain local under the run directory.
- `plan.json`, `execution.json`, `preparation.json`: commands and process status.
- `validation-reuse.json`: separate one-shot vs retained validation-graph check
  on the same optimized model, using 20 repeated in-memory synthetic batches.
  Dataset iteration, tracing and metrics are included. This is not GCS timing.
- `host.json`: hardware/container context collected after measurement.

Paired-round median inference latency fell 7.83%, with individual round
reductions of 6.41% to 8.86%. The pooled inference median changed from 146.07ms
to 129.27ms (11.50%); pooled and paired medians summarize different aspects of
the same samples. Training pooled median was 470.75ms vs 471.39ms, and paired
rounds were inconsistent; no stable training-step speedup is established.
Validation-pass median changed from 3.770s to 3.147s (16.5% reduction), with
identical confusion matrices and one retained graph trace.

All stored forward-agreement checks passed with maximum absolute error 0.0
and argmax agreement 1.0 on this fixed fixture. Dropout sampling shapes differ
during training, so identical seeded training trajectories are not promised.
CPU allocator memory is not reported. GPU timing and allocator memory require
the separate Colab comparison. These descriptive rounds do not establish
statistical significance or performance on another machine.

## Production prediction check

`production/protocol.json` and its six raw JSON reports retain a separate
NPZ-to-NPY/PNG/CSV comparison. Both sources load the same 3,311,968-parameter
toy checkpoint and the same two synthetic input files. Three alternating
rounds each measure 30 full passes after one warmup pass, totaling 180 tile
executions per revision. Pooled throughput is 7.05 vs 7.03 tiles/s; paired
rounds vary in direction, so no stable end-to-end gain is claimed. All 180
measured output class-map pairs match. Input/checkpoint hashes and commands
are retained. OS caches are not cleared and outputs are not fsynced.
