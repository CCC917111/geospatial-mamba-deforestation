"""Offline source uploads are bounded and checked before a Git checkout."""

import hashlib
import io
import json
from pathlib import Path
import zipfile

import pytest


NOTEBOOK = (Path(__file__).resolve().parents[1] /
            "notebooks/colab_checkpoint_backup.ipynb")
PAYLOAD = b"trusted bundle fixture"
BUNDLE_NAME = "checkpoint-backup-source.bundle"


def reader():
  notebook = json.loads(NOTEBOOK.read_text())
  source = next("".join(c["source"]) for c in notebook["cells"]
                if c["id"] == "source-bundle-helper")
  namespace = {
      "hashlib": hashlib,
      "TRUSTED_BUNDLE_BYTES": len(PAYLOAD),
      "TRUSTED_BUNDLE_SHA256": hashlib.sha256(PAYLOAD).hexdigest(),
  }
  exec(compile(source, "backup-source-reader", "exec"), namespace)
  return namespace["source_bundle_bytes"]


def wrapped(entries):
  output = io.BytesIO()
  with zipfile.ZipFile(output, "w") as archive:
    for name, payload in entries:
      archive.writestr(name, payload)
  return output.getvalue()


@pytest.mark.parametrize("wrapper", [False, True], ids=["bundle", "delivery-zip"])
def test_source_upload_accepts_only_the_trusted_bundle_bytes(wrapper):
  payload = wrapped([("package/" + BUNDLE_NAME, PAYLOAD)]) if wrapper else PAYLOAD
  name = "delivery.zip" if wrapper else BUNDLE_NAME
  assert reader()(name, payload) == PAYLOAD


@pytest.mark.parametrize("failure", ["checksum", "unsafe", "ambiguous", "oversize"])
def test_source_upload_rejects_corrupt_or_unsafe_packages(failure):
  name = "delivery.zip"
  if failure == "checksum":
    name, payload = BUNDLE_NAME, b"x" + PAYLOAD[1:]
  elif failure == "unsafe":
    payload = wrapped([(BUNDLE_NAME, PAYLOAD), ("../escape.txt", b"unsafe")])
  elif failure == "ambiguous":
    payload = wrapped([("a/" + BUNDLE_NAME, PAYLOAD),
                       ("b/" + BUNDLE_NAME, PAYLOAD)])
  else:
    name, payload = BUNDLE_NAME, b"x" * (32 * 1024**2)
  with pytest.raises(ValueError):
    reader()(name, payload)
