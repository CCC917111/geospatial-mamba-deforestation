[Research pilot](research-pilot.md) · [Audited results](../results/architecture_pilot_t4_20261029/README.md) · [Application preparation](application-interview.md)

# Preserve and verify the trained checkpoints

The research metadata lists six best and six last checkpoint files in Drive;
the weights are stored separately. This workflow checks those files against
the published retention manifest and packages them with the model
configurations and source records.

**Weight backup verification is pending.** The utility has passed local fixture
tests. To verify the study's actual weights, run it in the signed-in Colab
session, then verify the downloads on the receiving computer. The completed
metadata audit covers the recorded manifests and metrics, not the weight files.

## Run the CPU backup notebook

[Open the checkpoint backup notebook in Colab](https://colab.research.google.com/github/CCC917111/geospatial-mamba-deforestation/blob/main/notebooks/colab_checkpoint_backup.ipynb),
select **CPU**, and run the cells in order. It uses the completed study:

```text
/content/drive/MyDrive/geospatial-mamba/research/forty-architecture-pilot-v1
```

The offline delivery package includes the notebook and
`checkpoint-backup-source.bundle`. You can upload the notebook directly to
Colab. With the default `UPLOAD_SOURCE_BUNDLE=True`, upload the supplied bundle
or full delivery ZIP when prompted. The notebook checks the fixed bundle hash
and restores the pinned helper locally. Set `UPLOAD_SOURCE_BUNDLE=False` to
clone the same helper revision from GitHub instead.

Allow notebook credential access and consent to all requested permissions in
the Google authorization window. The notebook requests a read-only Drive
mount. If Drive is already mounted, the utility still opens study files only
for reading. It uses a separate pinned backup-tool checkout and never invokes
the research runner or changes the frozen study protocol.

The trusted reference is the original
[checkpoint manifest](../results/architecture_pilot_t4_20261029/metadata/checkpoint_manifest.json):

```text
SHA-256 66c668774ddba98f04bdc46a2ff3a5fd7fcd6ee07b208317b7dcd9d778460fb1
```

Each of the twelve weight files is streamed into its own ZIP while its size
and SHA-256 are checked. The ZIP is reopened and verified before finalizing.
The corresponding `config.json` is also checked against its original receipt
and included, preserving the architecture settings needed to load the weights.
Source/runtime identity receipts are included when referenced by the manifest.

The recorded weights total 308,911,968 bytes (294.6 MiB). The largest single
file is 31,163,864 bytes. Each uncompressed ZIP stays strictly below 32 MiB;
the utility checks available local disk space and archive size before writing.
Five best files equal their respective last files, but all twelve paths are
preserved. They represent seven unique recorded weight hashes.

The notebook prepares twelve weight ZIPs and `backup-metadata.zip`. Its default
download cell requests all thirteen files. If the browser blocks multiple
downloads, allow downloads for Colab or request missed archives individually
with the numbered fallback cell. Put the files in one fresh local folder,
keeping their original names. Colab's `/content` output is temporary until
the browser saves it to your computer.

Source files remain unchanged. An interrupted attempt can leave completed
archives without a final index; it is not a complete backup. Stop the previous
process and rerun the configuration cells to choose a fresh destination for
another attempt. Existing output directories are rejected rather than
overwritten.

## Verify on the receiving computer

With Python 3.10 or newer, open a terminal in the folder containing the thirteen
downloaded ZIPs and run:

```bash
python -m zipfile -e backup-metadata.zip .
python verify_backup.py verify \
  --backup-dir . \
  --manifest checkpoint_manifest.json \
  --manifest-sha256 66c668774ddba98f04bdc46a2ff3a5fd7fcd6ee07b208317b7dcd9d778460fb1 \
  --report local-backup-verification.json
```

The verifier uses Python's standard library. It checks every archive, weight,
configuration and receipt against the original manifest. The twelve weight
ZIPs can remain compressed. Keep the successful verification JSON beside the
backups and preserve the original research metadata ZIP as well.

The creation index records `archives_created_local_download_not_verified`.
That historical index remains unchanged after verification. A successful local
check produces a separate `downloaded_backup_verified` receipt for the files
actually read on the receiving computer. A browser download request alone does
not establish this result.

## Use the command-line tool directly

The same utility can run against a local study directory or a mounted Drive:

```bash
python scripts/checkpoint_backup.py create \
  --run-root /path/to/forty-architecture-pilot-v1 \
  --manifest results/architecture_pilot_t4_20261029/metadata/checkpoint_manifest.json \
  --manifest-sha256 66c668774ddba98f04bdc46a2ff3a5fd7fcd6ee07b208317b7dcd9d778460fb1 \
  --output-dir /path/to/fresh-checkpoint-backup

python scripts/checkpoint_backup.py verify \
  --backup-dir /path/to/downloaded-checkpoint-backup \
  --manifest results/architecture_pilot_t4_20261029/metadata/checkpoint_manifest.json \
  --manifest-sha256 66c668774ddba98f04bdc46a2ff3a5fd7fcd6ee07b208317b7dcd9d778460fb1 \
  --report /path/to/local-backup-verification.json
```

Choose a fresh output directory outside the original study. No GPU, TensorFlow,
new training or TFRecord download is needed. Weights are not added to Git.
Archive creation requires a local filesystem with hard-link support, such as
Colab's `/content`, for atomic publication without overwriting files. Offline
verification reads the downloaded archives without that requirement. Choose a
fresh report filename when repeating a local verification.

## Restore a model

After verification, extract the chosen checkpoint ZIP into a **fresh** restore
directory. It contains the original relative checkpoint path and its matching
model configuration. Keep both files together and use the recorded scientific
source `7f55b2c30c830693873e1fe1628c8e52208ed57d`; the source identity is also
available in the [original metadata](../results/architecture_pilot_t4_20261029/metadata/source_identity.json).

These H5 files contain model weights, not optimizer state. They support model
restoration and evaluation, rather than exact mid-training resumption. Source
TFRecords are not part of the checkpoint backup. Confirming a local backup
does not independently repeat the ForTy accuracy experiment or establish
long-term storage durability.
