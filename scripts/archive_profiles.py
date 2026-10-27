"""Export small report ZIPs and lossless, bounded TensorFlow profile chunks.

The metadata ZIP excludes checkpoints and input arrays. Raw XPlane traces are
split into bounded ZIPs without truncating their contents. Restore verifies
every chunk and whole trace before reporting success; it never runs files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import zipfile


MIB = 1024 * 1024
ARCHIVE_LIMIT = 32 * MIB
CHUNK_BYTES = 24 * MIB
METADATA_BYTES = 16 * MIB
METADATA_FILE_BYTES = 4 * MIB
MANIFEST_BYTES = 16 * MIB
MAX_FILES = 10000
MAX_METADATA_FILES = 2000
MAX_TRACES = 256
MAX_CHUNKS = 512
MAX_TRACE_BYTES = 2 * 1024 * MIB
MAX_TOTAL_TRACE_BYTES = 8 * 1024 * MIB
BLOCK_BYTES = MIB
MANIFEST_NAME = "profile-archives.json"
CHUNK_NAME = "chunk.bin"


def digest(path):
  result = hashlib.sha256()
  with Path(path).open("rb") as handle:
    for block in iter(lambda: handle.read(BLOCK_BYTES), b""):
      result.update(block)
  return result.hexdigest()


def safe_relative(value):
  if not isinstance(value, str) or not value or len(value) > 1024:
    raise ValueError("Invalid relative artifact path")
  path = PurePosixPath(value)
  unsafe = (not path.parts or path.is_absolute() or path.as_posix() != value
            or any(part in (".", "..") for part in path.parts)
            or any(character in value for character in ("\\", ":", "\x00"))
            or any(ord(character) < 32 for character in value))
  if unsafe:
    raise ValueError(f"Unsafe relative artifact path: {value!r}")
  return path


def is_trace(name):
  return (name.endswith(".xplane.pb") or name.endswith(".trace.json.gz")
          or name == "trace.json.gz")


def bounded_integer(value, maximum, name, allow_zero=False):
  minimum = 0 if allow_zero else 1
  invalid = (isinstance(value, bool) or not isinstance(value, int)
             or not minimum <= value <= maximum)
  if invalid:
    raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
  return value


def check_digest(value):
  if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
    raise ValueError("Invalid SHA-256 digest")


def check_archive(path):
  if path.is_symlink() or not path.is_file():
    raise ValueError(f"Missing or unsafe archive: {path.name}")
  if path.stat().st_size >= ARCHIVE_LIMIT:
    raise ValueError(f"Archive exceeds the transfer limit: {path.name}")


def write_trace(path, name, index, destination, chunk_bytes):
  before = path.stat()
  whole = hashlib.sha256()
  chunks = []
  with path.open("rb") as handle:
    remaining = before.st_size
    while remaining or not chunks:
      size = min(remaining, chunk_bytes)
      archive_name = f"trace-{index:04d}-part-{len(chunks):04d}.zip"
      archive_path = destination / archive_name
      part_hash = hashlib.sha256()
      with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED) as archive:
        with archive.open(CHUNK_NAME, "w") as output:
          pending = size
          while pending:
            block = handle.read(min(BLOCK_BYTES, pending))
            if not block:
              raise ValueError(f"Trace changed during export: {name}")
            output.write(block)
            whole.update(block)
            part_hash.update(block)
            pending -= len(block)
      check_archive(archive_path)
      chunks.append({"archive": archive_name, "entry": CHUNK_NAME,
                     "bytes": size, "sha256": part_hash.hexdigest(),
                     "archive_sha256": digest(archive_path)})
      remaining -= size
    after = path.stat()
    changed = (handle.read(1) or before.st_size != after.st_size
               or before.st_mtime_ns != after.st_mtime_ns)
    if changed:
      raise ValueError(f"Trace changed during export: {name}")
  return {"path": name, "bytes": before.st_size, "sha256": whole.hexdigest(),
          "chunks": chunks}


def export_study(run_root, output_dir, chunk_bytes=CHUNK_BYTES,
                 metadata_bytes=METADATA_BYTES,
                 metadata_file_bytes=METADATA_FILE_BYTES):
  """Export a fresh directory and return its JSON-serializable manifest."""
  bounded_integer(chunk_bytes, CHUNK_BYTES, "chunk bytes")
  bounded_integer(metadata_bytes, METADATA_BYTES, "metadata bytes")
  bounded_integer(metadata_file_bytes, METADATA_FILE_BYTES, "metadata file bytes")
  run_root, output_dir = Path(run_root), Path(output_dir)
  if run_root.is_symlink() or not run_root.is_dir():
    raise ValueError("Run root must be a real directory")
  source = run_root.resolve()
  destination = output_dir.resolve()
  if destination.is_relative_to(source):
    raise ValueError("Export directory must be outside the run root")
  if output_dir.exists() or output_dir.is_symlink():
    raise FileExistsError(f"Refusing to overwrite {output_dir}")
  paths = sorted(source.rglob("*"))
  if len(paths) > MAX_FILES:
    raise ValueError("Too many artifacts to export")
  metadata, traces, omitted = [], [], []
  metadata_total = trace_total = 0
  for path in paths:
    name = path.relative_to(source).as_posix()
    safe_relative(name)
    if path.is_symlink():
      omitted.append({"path": name, "reason": "symlink"})
      continue
    if path.is_dir():
      continue
    if not path.is_file():
      omitted.append({"path": name, "reason": "not_regular_file"})
      continue
    size = path.stat().st_size
    if is_trace(path.name):
      bounded_integer(size, MAX_TRACE_BYTES, "trace bytes", allow_zero=True)
      traces.append((path, name))
      trace_total += size
      if len(traces) > MAX_TRACES or trace_total > MAX_TOTAL_TRACE_BYTES:
        raise ValueError("Profile export exceeds the trace resource limits")
    elif name == MANIFEST_NAME:
      omitted.append({"path": name, "reason": "reserved_export_name"})
    elif path.suffix.lower() not in {".json", ".csv", ".log", ".txt"}:
      omitted.append({"path": name, "reason": "excluded_binary_or_extension"})
    elif (size > metadata_file_bytes or metadata_total + size > metadata_bytes
          or len(metadata) >= MAX_METADATA_FILES):
      omitted.append({"path": name, "bytes": size, "reason": "metadata_budget"})
    else:
      metadata.append((path, name, size))
      metadata_total += size
  chunks = sum(max(1, (path.stat().st_size + chunk_bytes - 1) // chunk_bytes)
               for path, _ in traces)
  if chunks > MAX_CHUNKS:
    raise ValueError("Profile export exceeds the chunk resource limit")
  destination.mkdir(parents=True, exist_ok=False)
  trace_records = [write_trace(path, name, index, destination, chunk_bytes)
                   for index, (path, name) in enumerate(traces)]
  downloads = ["metadata.zip", MANIFEST_NAME]
  downloads.extend(chunk["archive"] for trace in trace_records
                   for chunk in trace["chunks"])
  result = {"schema_version": 1,
            "trace_status": "complete" if trace_records else "none",
            "scope": ("Metadata and complete raw profile chunks only; "
                      "checkpoints and arrays remain in the runtime. "
                      "A captured trace does not establish GPU event coverage."),
            "limits": {"archive_bytes_exclusive": ARCHIVE_LIMIT,
                       "trace_chunk_bytes": chunk_bytes,
                       "metadata_bytes": metadata_bytes,
                       "metadata_file_bytes": metadata_file_bytes},
            "metadata": {"archive": "metadata.zip", "saved": [],
                         "omitted": omitted},
            "traces": trace_records, "downloads": downloads}
  archive_path = destination / "metadata.zip"
  with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED) as archive:
    for path, name, size in metadata:
      data_hash = hashlib.sha256()
      total = 0
      with path.open("rb") as handle, archive.open(name, "w") as output:
        for block in iter(lambda: handle.read(BLOCK_BYTES), b""):
          total += len(block)
          if total > size:
            raise ValueError(f"Metadata changed during export: {name}")
          output.write(block)
          data_hash.update(block)
      if total != size:
        raise ValueError(f"Metadata changed during export: {name}")
      result["metadata"]["saved"].append(
          {"path": name, "bytes": size, "sha256": data_hash.hexdigest()})
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if len(encoded.encode()) > MANIFEST_BYTES:
      raise ValueError("Archive manifest exceeds the resource limit")
    archive.writestr(MANIFEST_NAME, encoded)
  check_archive(archive_path)
  (destination / MANIFEST_NAME).write_text(encoded)
  return result


def validated_traces(manifest, archive_dir):
  if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
    raise ValueError("Unsupported profile archive schema")
  traces = manifest.get("traces")
  if not isinstance(traces, list) or len(traces) > MAX_TRACES:
    raise ValueError("Invalid profile trace list")
  paths, archives = set(), set()
  total = chunk_count = 0
  for trace in traces:
    path = safe_relative(trace["path"])
    if not is_trace(path.name) or path in paths:
      raise ValueError("Invalid or duplicate trace output path")
    paths.add(path)
    total += bounded_integer(trace["bytes"], MAX_TRACE_BYTES, "trace bytes", True)
    check_digest(trace["sha256"])
    chunks = trace["chunks"]
    if not isinstance(chunks, list) or not chunks:
      raise ValueError("Missing trace chunks")
    chunk_count += len(chunks)
    if chunk_count > MAX_CHUNKS or total > MAX_TOTAL_TRACE_BYTES:
      raise ValueError("Profile restoration exceeds the resource limits")
    sizes = 0
    for chunk in chunks:
      name = chunk["archive"]
      invalid = (not isinstance(name, str)
                 or not re.fullmatch(r"trace-[0-9]{4}-part-[0-9]{4}\.zip", name)
                 or name in archives or chunk["entry"] != CHUNK_NAME)
      if invalid:
        raise ValueError("Invalid or duplicate trace archive")
      archives.add(name)
      size = bounded_integer(chunk["bytes"], CHUNK_BYTES, "chunk bytes", True)
      if not size and (len(chunks) != 1 or trace["bytes"] != 0):
        raise ValueError("Unexpected empty trace chunk")
      sizes += size
      check_digest(chunk["sha256"])
      check_digest(chunk["archive_sha256"])
      archive_path = archive_dir / name
      check_archive(archive_path)
      if digest(archive_path) != chunk["archive_sha256"]:
        raise ValueError(f"Archive hash mismatch: {name}")
    if sizes != trace["bytes"]:
      raise ValueError("Chunk sizes do not match the trace size")
  return traces


def restore_profiles(manifest_path, archive_dir, destination):
  """Restore only verified raw traces, preserving their safe relative paths."""
  manifest_path, archive_dir = Path(manifest_path), Path(archive_dir)
  destination = Path(destination)
  if destination.exists() or destination.is_symlink():
    raise FileExistsError(f"Refusing to overwrite {destination}")
  invalid = (manifest_path.is_symlink() or not manifest_path.is_file()
             or manifest_path.stat().st_size > MANIFEST_BYTES)
  if invalid:
    raise ValueError("Missing, unsafe or oversized profile manifest")
  if archive_dir.is_symlink() or not archive_dir.is_dir():
    raise ValueError("Archive directory must be a real directory")
  manifest = json.loads(manifest_path.read_text())
  traces = validated_traces(manifest, archive_dir)
  destination.mkdir(parents=True, exist_ok=False)
  try:
    for trace in traces:
      path = destination.joinpath(*safe_relative(trace["path"]).parts)
      path.parent.mkdir(parents=True, exist_ok=True)
      whole = hashlib.sha256()
      with path.open("xb") as output:
        for chunk in trace["chunks"]:
          with zipfile.ZipFile(archive_dir / chunk["archive"]) as archive:
            entries = archive.infolist()
            if (len(entries) != 1 or entries[0].filename != CHUNK_NAME
                or entries[0].is_dir()
                or stat.S_ISLNK(entries[0].external_attr >> 16)
                or entries[0].file_size != chunk["bytes"]
                or entries[0].compress_type not in (
                    zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED)):
              raise ValueError("Unsafe or unmatched ZIP chunk entry")
            part_hash, total = hashlib.sha256(), 0
            with archive.open(entries[0]) as handle:
              for block in iter(lambda: handle.read(BLOCK_BYTES), b""):
                total += len(block)
                if total > chunk["bytes"]:
                  raise ValueError("Decompressed chunk exceeds its declared size")
                output.write(block)
                whole.update(block)
                part_hash.update(block)
            if total != chunk["bytes"] or part_hash.hexdigest() != chunk["sha256"]:
              raise ValueError("Trace chunk hash or size mismatch")
      if path.stat().st_size != trace["bytes"] or whole.hexdigest() != trace["sha256"]:
        raise ValueError("Restored trace hash or size mismatch")
  except Exception:
    shutil.rmtree(destination)
    raise
  return {"status": "restored", "traces": len(traces),
          "destination": str(destination.resolve())}


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  mode = parser.add_mutually_exclusive_group(required=True)
  mode.add_argument("--run-root", type=Path)
  mode.add_argument("--restore", type=Path, metavar="MANIFEST")
  parser.add_argument("--output-dir", type=Path)
  parser.add_argument("--archive-dir", type=Path)
  parser.add_argument("--destination", type=Path)
  args = parser.parse_args(argv)
  if args.run_root is not None:
    if args.output_dir is None or args.archive_dir or args.destination:
      parser.error("Export requires --output-dir and no restore arguments")
    result = export_study(args.run_root, args.output_dir)
  else:
    if args.archive_dir is None or args.destination is None or args.output_dir:
      parser.error("Restore requires --archive-dir and --destination")
    result = restore_profiles(args.restore, args.archive_dir, args.destination)
  print(json.dumps(result, indent=2, allow_nan=False))
  return result


if __name__ == "__main__":
  main()
