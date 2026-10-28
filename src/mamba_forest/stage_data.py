"""Stage a fixed ForTy shard cohort with immutable content receipts.

Planning and local-file copies use only the standard library. TensorFlow's
GCS filesystem is imported only when a remote object must be inspected or
copied. Staging verifies bytes, not record contents, labels or geography.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import numbers
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
from urllib.parse import urlsplit

from mamba_forest.constants import GCS_ROOT, TOTAL_SHARDS

BLOCK_BYTES = 1024 * 1024
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_SELECTED_FILES = 4096
MAX_TOTAL_SHARDS = 100000
DEFAULT_MAX_BYTES = 16 * 1024**3
SCOPE = ("Selected-shard byte staging only; does not validate TFRecord contents, "
         "label semantics, geographic separation or full-dataset coverage. "
         "Verified cached originals may be reused without rereading the source.")


def _integer(value, name, minimum=1, maximum=None):
  invalid = (isinstance(value, bool) or not isinstance(value, numbers.Integral)
             or value < minimum or maximum is not None and value > maximum)
  if invalid:
    raise ValueError(f"Invalid {name}: {value!r}")
  return int(value)


def _source_root(value):
  value = os.fspath(value)
  if not isinstance(value, str) or not value or "\x00" in value:
    raise ValueError("source_root must be a nonempty path or gs:// URI")
  if "://" not in value:
    return str(Path(value).expanduser().resolve())
  parsed = urlsplit(value)
  invalid = (parsed.scheme != "gs"
             or not re.fullmatch(r"[A-Za-z0-9_.-]+", parsed.netloc)
             or parsed.query or parsed.fragment or ".." in parsed.path.split("/"))
  if invalid:
    raise ValueError("Only a plain gs:// source prefix is supported")
  return value.rstrip("/")


def selected_files(source_root=GCS_ROOT, *, total_shards=TOTAL_SHARDS,
                   train_shards=8, val_shards=4, test_shards=8,
                   selection_seed=2026):
  """Plan original filenames without importing a framework or opening data."""
  source_root = _source_root(source_root)
  total = _integer(total_shards, "total_shards", maximum=MAX_TOTAL_SHARDS)
  seed = _integer(selection_seed, "selection_seed", 0, 2**32 - 1)
  budgets = {
      "train": _integer(train_shards, "train_shards", maximum=total),
      "validation": _integer(val_shards, "val_shards", maximum=total),
      "test": _integer(test_shards, "test_shards", maximum=total)}
  if sum(budgets.values()) > MAX_SELECTED_FILES:
    raise ValueError("Selected shard budget exceeds the staging file limit")
  result = []
  for split, budget in budgets.items():
    names = [f"forty_v1-{split}.tfrecord-{index:05d}-of-{total:05d}"
             for index in range(total)]
    chosen = random.Random(seed).sample(names, budget) if split == "train" else (
        names[:budget])
    result.extend({"split": split, "filename": name,
                   "source": source_root + "/" + name} for name in chosen)
  return result


def _digest(path):
  digest = hashlib.sha256()
  with Path(path).open("rb") as handle:
    for block in iter(lambda: handle.read(BLOCK_BYTES), b""):
      digest.update(block)
  return digest.hexdigest()


def _source_stat(path):
  if path.startswith("gs://"):
    import tensorflow as tf
    value = tf.io.gfile.stat(path)
    if value.is_directory:
      raise ValueError(f"Source shard is a directory: {path}")
    return int(value.length), value.mtime_nsec
  local = Path(path)
  if local.is_symlink() or not local.is_file():
    raise ValueError(f"Source shard is missing or unsafe: {path}")
  value = local.stat()
  return value.st_size, value.st_mtime_ns


def _open_source(path):
  if path.startswith("gs://"):
    import tensorflow as tf
    return tf.io.gfile.GFile(path, "rb")
  return Path(path).open("rb")


def _read_manifest(path, config, files):
  if path.is_symlink() or not path.is_file():
    raise ValueError("Existing staging manifest is not a regular file")
  if path.stat().st_size > MAX_MANIFEST_BYTES:
    raise ValueError("Staging manifest exceeds its size limit")
  value = json.loads(path.read_text())
  invalid = (not isinstance(value, dict) or value.get("schema_version") != 1
             or value.get("status") != "ready" or value.get("config") != config)
  if invalid:
    raise ValueError("Existing staging manifest is incomplete or has different config")
  records = value.get("files")
  if not isinstance(records, list) or len(records) != len(files):
    raise ValueError("Staging manifest has incomplete selected files")
  for record, selected in zip(records, files):
    invalid = (not isinstance(record, dict) or any(
        record.get(key) != expected for key, expected in selected.items()))
    if invalid:
      raise ValueError("Staging manifest contains an unsafe or mismatched shard")
    _integer(record.get("bytes"), "manifest shard bytes", minimum=0)
    invalid = (not isinstance(record.get("sha256"), str)
               or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]))
    if invalid:
      raise ValueError("Staging manifest contains an invalid SHA256")
  return value


def _cached_matches(path, record):
  if path.is_symlink() or path.exists() and not path.is_file():
    raise ValueError(f"Cached shard is not a safe regular file: {path}")
  return (path.is_file() and path.stat().st_size == record["bytes"]
          and _digest(path) == record["sha256"])


def _copy_shard(record, destination, expected=None):
  size, marker = _source_stat(record["source"])
  if expected is not None and size != expected["bytes"]:
    raise ValueError(f"Source shard changed size: {record['filename']}")
  if size != record["bytes"]:
    raise ValueError(f"Source shard changed during staging: {record['filename']}")
  if shutil.disk_usage(destination.parent).free < size:
    raise ValueError(f"Insufficient cache disk space for {record['filename']}")
  temporary = None
  try:
    with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False,
                                     prefix=".shard-", suffix=".partial") as handle:
      temporary = Path(handle.name)
      digest, copied = hashlib.sha256(), 0
      with _open_source(record["source"]) as source:
        for block in iter(lambda: source.read(BLOCK_BYTES), b""):
          copied += len(block)
          if copied > size:
            raise ValueError(f"Source shard grew during copy: {record['filename']}")
          handle.write(block)
          digest.update(block)
    if copied != size or _source_stat(record["source"]) != (size, marker):
      raise ValueError(f"Source shard changed during copy: {record['filename']}")
    value = digest.hexdigest()
    if expected is not None and value != expected["sha256"]:
      raise ValueError(f"Source shard changed SHA256: {record['filename']}")
    if temporary.stat().st_size != size or _digest(temporary) != value:
      raise ValueError(f"Cached bytes differ from copied source: {record['filename']}")
    os.replace(temporary, destination)
    return {**record, "sha256": value}
  finally:
    if temporary is not None:
      temporary.unlink(missing_ok=True)


def _write_manifest(path, manifest):
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = None
  try:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False,
                                     prefix=".staging-", suffix=".json") as handle:
      temporary = Path(handle.name)
      json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
      handle.write("\n")
    os.replace(temporary, path)
  finally:
    if temporary is not None:
      temporary.unlink(missing_ok=True)


def stage_dataset(source_root, cache_root, manifest_path, *,
                  total_shards=TOTAL_SHARDS, train_shards=8, val_shards=4,
                  test_shards=8, selection_seed=2026, max_bytes=DEFAULT_MAX_BYTES):
  """Copy only the selected cohort or repair its exact recorded originals.

  An existing ready receipt is immutable. Valid cached originals are reused
  offline. Missing/corrupt copies are replaced only if source bytes match that
  receipt. Partial source copies never become published cache files.
  """
  source_root = _source_root(source_root)
  cache_path = Path(cache_root).expanduser()
  manifest_path = Path(manifest_path).expanduser()
  if cache_path.is_symlink() or manifest_path.is_symlink():
    raise ValueError("Cache root and manifest must not be symlinks")
  cache = cache_path.resolve()
  manifest_path = manifest_path.resolve()
  if "://" not in source_root and cache == Path(source_root):
    raise ValueError("Cache root must differ from the source root")
  limit = _integer(max_bytes, "max_bytes")
  files = selected_files(source_root, total_shards=total_shards,
                         train_shards=train_shards, val_shards=val_shards,
                         test_shards=test_shards, selection_seed=selection_seed)
  config = {"source_root": source_root, "cache_root": str(cache),
            "total_shards": int(total_shards), "selection_seed": int(selection_seed),
            "train_shards": int(train_shards), "val_shards": int(val_shards),
            "test_shards": int(test_shards)}
  reference = (_read_manifest(manifest_path, config, files)
               if manifest_path.exists() else None)
  records = reference["files"] if reference else [
      {**item, "bytes": _source_stat(item["source"])[0]} for item in files]
  if sum(record["bytes"] for record in records) > limit:
    raise ValueError("Selected shard bytes exceed --max-bytes")
  if cache.exists() and not cache.is_dir():
    raise ValueError("Cache root must be a directory")
  cache.mkdir(parents=True, exist_ok=True)
  jobs, growth, peak_extra = [], 0, 0
  for record in records:
    destination = cache / record["filename"]
    if reference is not None and _cached_matches(destination, record):
      jobs.append((record, destination, False))
      continue
    if destination.is_symlink() or destination.exists() and not destination.is_file():
      raise ValueError(f"Cached shard is not a safe regular file: {destination}")
    old_size = destination.stat().st_size if destination.is_file() else 0
    peak_extra = max(peak_extra, growth + record["bytes"])
    growth += record["bytes"] - old_size
    jobs.append((record, destination, True))
  if shutil.disk_usage(cache).free < peak_extra:
    raise ValueError("Insufficient cache disk space for selected shard copies")
  completed = []
  for record, destination, copy_needed in jobs:
    if not copy_needed:
      completed.append(record)
      continue
    print(f"staging {record['split']}: {record['filename']}", flush=True)
    completed.append(_copy_shard(record, destination, record if reference else None))
  if reference is not None:
    return reference
  manifest = {"schema_version": 1, "status": "ready", "config": config,
              "files": completed, "scope": SCOPE}
  # Another incompatible receipt must never be silently overwritten.
  if manifest_path.exists():
    current = _read_manifest(manifest_path, config, files)
    if current["files"] != completed:
      raise ValueError("Staging receipt changed during execution")
    return current
  _write_manifest(manifest_path, manifest)
  return manifest


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source-root", default=GCS_ROOT)
  parser.add_argument("--cache-root", type=Path, required=True)
  parser.add_argument("--manifest", type=Path, required=True)
  parser.add_argument("--total-shards", type=int, default=TOTAL_SHARDS)
  parser.add_argument("--train-shards", type=int, default=8)
  parser.add_argument("--val-shards", type=int, default=4)
  parser.add_argument("--test-shards", type=int, default=8)
  parser.add_argument("--selection-seed", type=int, default=2026)
  parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
  args = parser.parse_args(argv)
  try:
    manifest = stage_dataset(
        args.source_root, args.cache_root, args.manifest,
        total_shards=args.total_shards, train_shards=args.train_shards,
        val_shards=args.val_shards, test_shards=args.test_shards,
        selection_seed=args.selection_seed, max_bytes=args.max_bytes)
  except (ValueError, OSError) as error:
    parser.exit(1, f"staging failed: {error}\n")
  print(json.dumps({"status": manifest["status"], "files": len(manifest["files"]),
                    "bytes": sum(item["bytes"] for item in manifest["files"]),
                    "manifest": str(args.manifest)}, indent=2))
  return manifest


if __name__ == "__main__":
  main()
