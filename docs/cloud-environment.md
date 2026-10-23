# Prepare a cloud environment

**Status: prepared, not deployed.** The setup files prepare software in a new
virtual environment. No VM, GPU, network policy or cloud identity has been
provisioned by this work. The current tools can inspect environment settings;
resource creation and configuration changes use the provider or Codex UI.

For a free hosted GPU attempt, start with the [Colab notebook](colab-free.md).
It reuses Colab's configured TensorFlow runtime and prepares a small paired
pipeline check. Free GPU allocation and session duration are not guaranteed.

## Choose the execution target

Use Linux x86_64, Bash, GNU coreutils and Python 3.12, with a persistent workspace for
source, checkpoints and run artifacts. For real GPU validation, select a VM
with an NVIDIA GPU and provider-supported drivers. Choose the provider,
region, GPU and disk allocation after the platform and spending limit are
known. Set a spending limit and shutdown policy in that provider's controls.
Batch size, VRAM needs, throughput and experiment cost require measurement on
the selected machine.

A Codex managed environment can support code development without exposing a
GPU. Treat GPU availability as unconfirmed until TensorFlow executes the
strict GPU probe. Installing CUDA dependencies does not provide hardware.

## Configure the environment

For a new Codex cloud task, select repository
`CCC917111/geospatial-mamba-deforestation` and the `codex/...` branch containing
these setup files. Configure a writable persistent workspace and Python 3.12.
Use a separate GPU VM if the chosen Codex runtime cannot expose a GPU.
If the setup files have not been pushed to GitHub, restore the supplied Git
bundle first or publish the prepared branch before using repository checkout.

Through the supported network settings, allow the required GitHub and Python
package destinations, including `github.com`, `raw.githubusercontent.com`,
`codeload.github.com`, `pypi.org` and `files.pythonhosted.org`. Real ForTy access
also needs `storage.googleapis.com`. Permit the identity provider's required
authentication endpoints when applicable; for Google ADC flows using OAuth,
this can include `oauth2.googleapis.com`. Review and apply those settings in
the provider configuration. Editing `/etc/codex/network-policy.json` cannot
grant access.

Use an existing configured provider identity or application-default
credentials (ADC), supplied through secure environment configuration. Preserve
the runtime's injected authentication and proxy settings. Keep credential
values out of the repository, chat, logs and artifact archives.

## Install into a new virtual environment

Run from the repository root. Inspect the planned commands first:

```bash
bash scripts/setup_cloud_env.sh \
  --venv /path/to/new-venv --python python3.12 --dry-run
```

The default GPU profile checks the host driver before installation and requires
a real TensorFlow GPU operation afterward:

```bash
bash scripts/setup_cloud_env.sh \
  --venv /path/to/new-venv --python python3.12
source /path/to/new-venv/bin/activate
make test
make lint
```

The [setup script](../scripts/setup_cloud_env.sh) does not provision hardware,
install host NVIDIA drivers or change global Git identity. It uses the pinned
[GPU dependencies](../cloud/requirements-gpu.txt), including TensorFlow 2.21.0
with its CUDA dependency extra. The GPU probe rejects CPU fallback; a driver
listing alone is insufficient. It executes a convolution forward and backward
pass on the GPU and checks finite gradients. Installing CUDA extras into
`tensorflow-cpu` cannot turn that package into a GPU build.

Both profiles request GCS filesystem support. The runtime checker verifies
that `gs` is registered without reading data or proving network access. The
main dependencies are pinned; target-specific CUDA/GCS dependencies are
resolved during installation. The environment directory retains
`cloud-packages.txt`, `cloud-readiness.json`, source metadata and, for GPU
setup, `cloud-gpu.csv`. Preserve this resolved package list with the runs.
An existing environment directory is rejected before installation. If setup
fails partway through, inspect its artifacts and choose a fresh path.

For CPU development, choose the CPU profile explicitly:

```bash
bash scripts/setup_cloud_env.sh \
  --venv /path/to/new-cpu-venv --python python3.12 --cpu
```

The [CPU dependencies](../cloud/requirements-cpu.txt) install the normal
TensorFlow wheel without the CUDA extra. Both profiles share the same pinned
NumPy/Keras versions. CPU setup and synthetic tests establish software
integration only; they do not establish real ForTy accuracy or GPU performance.

The [runtime checker](../cloud/check_runtime.py) can also be run separately:

```bash
python cloud/check_runtime.py --output runs/runtime-readiness.json
```

Add `--cpu` for a CPU check. A failure writes a diagnostic report and returns
a nonzero exit code. This checks the selected interpreter; it does not create
an instance or grant access to the dataset.

## Validate data, then run experiments

After GPU setup passes, inspect actual records before scheduling training:

```bash
python -m mamba_forest.validate_data \
  --data-root gs://forest_typology/forty_v1/1.0.0 \
  --train-shards 1 --val-shards 1 --test-shards 1 --require-gpu \
  --output runs/source-preflight.json
```

Use `--data-root /path/to/local/forty` for an existing local mirror. A passing
three-record preflight does not establish complete dataset validity or
geographic separation. Follow the [GPU validation runbook](gpu-validation.md)
for checkpoint smoke tests, batch calibration and paired experiments. Retain
the source revision, resolved packages, hardware details and observed timings.

## Preserve and recover work

Keep the repository's Git recovery bundle and the run-artifact archive on
durable storage before replacing an environment. Verify a supplied Git bundle
with `git bundle verify`, restore its recorded branch, and restore run artifacts
with their original directory layout. A Git bundle preserves source history;
ignored checkpoints and run data need their separate artifact archive.

Use the runbook's [artifact-retention instructions](gpu-validation.md#evaluate-every-selected-checkpoint-on-the-full-test-split)
to retain configs, manifests, logs, checkpoint hashes and metric files together.
Record the durable archive location in the handoff, and exclude credentials.
Re-run environment and data preflights after recovery before claiming a new
runtime is ready.
