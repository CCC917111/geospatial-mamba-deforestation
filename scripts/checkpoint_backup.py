"""Create and verify bounded checkpoint backups without importing TensorFlow.

The trusted retention manifest supplies the expected weight and config hashes.
Creating archives verifies source bytes in the caller's runtime; only running
``verify`` on downloaded files verifies the bytes in that supplied directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
import zipfile


MIB = 1024 * 1024
ARCHIVE_LIMIT = 32 * MIB
BLOCK_BYTES = MIB
MAX_MANIFEST_BYTES = 4 * MIB
MAX_SMALL_BYTES = MIB
MAX_CHECKPOINTS = 128
INDEX_NAME = "backup-index.json"
METADATA_NAME = "backup-metadata.zip"
CREATE_STATUS = "archives_created_local_download_not_verified"
VERIFY_STATUS = "downloaded_backup_verified"
OPTIONAL_RECEIPTS = ("source_identity.json", "runtime_identity.json")


def _require(condition, message):
  if not condition:
    raise ValueError(message)


def _integer(value, maximum, name):
  _require(type(value) is int and 0 < value <= maximum, f"Invalid {name}")
  return value


def _digest(value):
  return hashlib.sha256(value).hexdigest()


def _check_digest(value):
  _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
           "Invalid SHA-256 digest")


def _relative(value):
  _require(isinstance(value, str) and 0 < len(value) <= 1024,
           "Invalid relative path")
  path = PurePosixPath(value)
  _require(bool(path.parts) and not path.is_absolute() and path.as_posix() == value and
           all(part not in (".", "..") for part in path.parts) and
           not any(c in value for c in ("\\", ":", "\x00")) and
           not any(ord(c) < 32 for c in value), "Unsafe relative path")
  return path


def _real_path(value, directory=False):
  path = Path(value).absolute()
  current = Path(path.anchor)
  for part in path.parts[1:]:
    current /= part
    if current.is_symlink():
      raise ValueError(f"Symlinks are not accepted: {current.name}")
  if directory:
    _require(path.is_dir(), "Expected an existing real directory")
  return path.resolve()


def _regular(path):
  path = _real_path(path)
  _require(path.is_file() and stat.S_ISREG(path.stat().st_mode),
           f"Expected a regular file: {path.name}")
  return path


def _read_bounded(path, maximum):
  path = _regular(path)
  _require(path.stat().st_size <= maximum, "File exceeds its byte budget")
  with path.open("rb") as handle:
    data = handle.read(maximum + 1)
  _require(len(data) <= maximum, "File grew beyond its byte budget")
  return data


def _json_bytes(value):
  return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _parse_json(data):
  def pairs(items):
    result = {}
    for key, value in items:
      _require(key not in result, "Duplicate JSON key")
      result[key] = value
    return result

  def constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")

  return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def _file_record(entry):
  _require(isinstance(entry, dict), "Invalid file record")
  _relative(entry.get("path"))
  _check_digest(entry.get("sha256"))
  _integer(entry.get("bytes"), 8 * 1024 * MIB, "recorded file size")
  _require(entry.get("external") is False, "External files are not accepted")
  return {key: entry[key] for key in ("path", "bytes", "sha256")}


def _manifest(path, expected_sha=None):
  data = _read_bounded(path, MAX_MANIFEST_BYTES)
  sha = _digest(data)
  if expected_sha is not None:
    _check_digest(expected_sha)
    _require(sha == expected_sha, "Trusted manifest SHA-256 mismatch")
  document = _parse_json(data)
  _require(isinstance(document, dict) and document.get("schema_version") == 1 and
           document.get("manifest_kind") == "architecture_pilot_checkpoint_retention"
           and document.get("status") == "completed_pilot",
           "Expected a completed architecture retention manifest")
  entries = document.get("checkpoints")
  _require(isinstance(entries, list) and 0 < len(entries) <= MAX_CHECKPOINTS,
           "Invalid checkpoint count")
  all_files = document.get("files")
  _require(isinstance(all_files, list) and len(all_files) <= 4096,
           "Invalid manifest files")
  files = {}
  for entry in all_files:
    record = _file_record(entry)
    _require(record["path"] not in files, "Duplicate manifest path")
    files[record["path"]] = record
  records, identities = [], set()
  for entry in entries:
    record = _file_record(entry)
    seed = entry.get("seed")
    variant, kind = entry.get("model_variant"), entry.get("kind")
    _require(type(seed) is int and 0 <= seed < 2**32 and
             variant in ("cnn_only", "cnn_ssm") and
             kind in ("selected_checkpoint", "last_weights_only"),
             "Invalid checkpoint identity")
    label = "best" if kind == "selected_checkpoint" else "last"
    prefix = f"{variant}_{seed}"
    _require(record["path"] == f"{prefix}/{label}.weights.h5",
             "Checkpoint path does not match its identity")
    identity = (seed, variant, label)
    _require(identity not in identities, "Duplicate checkpoint identity")
    identities.add(identity)
    _require(files.get(record["path"]) == record, "Conflicting checkpoint receipt")
    config = files.get(f"{prefix}/config.json")
    _require(config is not None, "Missing model configuration receipt")
    _integer(config["bytes"], MAX_SMALL_BYTES, "model configuration size")
    records.append({"checkpoint": record, "config": config,
                    "filename": f"{variant}-seed{seed}-{label}.zip"})
  seeds = {seed for seed, _, _ in identities}
  expected = {(seed, variant, label) for seed in seeds
              for variant in ("cnn_only", "cnn_ssm") for label in ("best", "last")}
  _require(identities == expected and
           document.get("selected_checkpoint_count") == len(entries) // 2,
           "Each seed requires both models and their best/last checkpoints")
  return document, data, sha, sorted(records, key=lambda r: r["filename"]), files


def _expected_receipt(record, manifest_sha):
  return {"schema_version": 1, "manifest_sha256": manifest_sha,
          "checkpoint": record["checkpoint"], "config": record["config"],
          "scope": "Model weights and architecture config; no optimizer state."}


def _zip_info(name, size):
  info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
  info.compress_type = zipfile.ZIP_STORED
  info.create_system = 3
  info.external_attr = (stat.S_IFREG | 0o600) << 16
  info.file_size = size
  return info


def _stored_zip_size(entries):
  return 22 + sum(size + 76 + 2 * len(name.encode("utf-8"))
                  for name, size in entries)


def _stream_expected(path, record, output=None):
  path = _regular(path)
  before = path.stat()
  _require(before.st_size == record["bytes"], f"Source size mismatch: {record['path']}")
  checksum = hashlib.sha256()
  with path.open("rb") as handle:
    _require(stat.S_ISREG(os.fstat(handle.fileno()).st_mode), "Nonregular source")
    pending = record["bytes"]
    while pending:
      block = handle.read(min(BLOCK_BYTES, pending))
      _require(bool(block), "Source file became shorter")
      pending -= len(block)
      checksum.update(block)
      if output is not None:
        output.write(block)
    _require(not handle.read(1), "Source file became longer")
  _require(checksum.hexdigest() == record["sha256"],
           f"Source SHA-256 mismatch: {record['path']}")


def _file_sha(path, maximum):
  path = _regular(path)
  _require(path.stat().st_size < maximum, "Archive exceeds the transfer limit")
  checksum = hashlib.sha256()
  size = 0
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(BLOCK_BYTES), b""):
      size += len(block)
      _require(size < maximum, "Archive grew beyond the transfer limit")
      checksum.update(block)
  return size, checksum.hexdigest()


def _check_members(archive, expected):
  members = archive.infolist()
  names = [entry.filename for entry in members]
  _require(len(names) == len(set(names)) and set(names) == set(expected),
           "Unexpected or duplicate ZIP members")
  for entry in members:
    _relative(entry.filename)
    _require(not entry.is_dir() and not entry.flag_bits & 1 and
             entry.compress_type == zipfile.ZIP_STORED and
             stat.S_ISREG(entry.external_attr >> 16) and
             entry.file_size == expected[entry.filename] and
             entry.compress_size == entry.file_size,
             "Unsafe ZIP member or unexpected size")


def _verify_archive(path, record, manifest_sha, limit):
  _file_sha(path, limit)
  receipt = _json_bytes(_expected_receipt(record, manifest_sha))
  expected = {record[k]["path"]: record[k]["bytes"] for k in ("checkpoint", "config")}
  expected["receipt.json"] = len(receipt)
  with zipfile.ZipFile(path) as archive:
    _check_members(archive, expected)
    _require(archive.read("receipt.json") == receipt, "Checkpoint receipt mismatch")
    for key in ("checkpoint", "config"):
      item = record[key]
      checksum = hashlib.sha256()
      count = 0
      with archive.open(item["path"]) as handle:
        for block in iter(lambda: handle.read(BLOCK_BYTES), b""):
          count += len(block)
          _require(count <= item["bytes"], "ZIP member exceeded expected bytes")
          checksum.update(block)
      _require(count == item["bytes"] and checksum.hexdigest() == item["sha256"],
               "Archived checkpoint/config SHA-256 mismatch")


def _new_temporary(destination):
  descriptor, name = tempfile.mkstemp(prefix=".partial-", suffix=".zip",
                                      dir=destination)
  os.close(descriptor)
  return Path(name)


def _publish(temporary, target):
  # Linking is atomic and refuses an existing name, including a concurrent one.
  # Archive creation uses a local output filesystem; verification needs no links.
  try:
    os.link(temporary, target)
  except FileExistsError:
    raise FileExistsError(f"Refusing to overwrite {target.name}") from None
  except OSError as error:
    raise OSError("Atomic publication unavailable; "
                  "choose a local output directory") from error
  temporary.unlink()


def _create_archive(source, destination, record, manifest_sha, limit):
  temporary = _new_temporary(destination)
  try:
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
      for key in ("checkpoint", "config"):
        item = record[key]
        with archive.open(_zip_info(item["path"], item["bytes"]), "w") as output:
          _stream_expected(source / item["path"], item, output)
      receipt = _json_bytes(_expected_receipt(record, manifest_sha))
      archive.writestr(_zip_info("receipt.json", len(receipt)), receipt)
    _verify_archive(temporary, record, manifest_sha, limit)
    target = destination / record["filename"]
    _publish(temporary, target)
    size, sha = _file_sha(target, limit)
    return {"filename": target.name, "bytes": size, "sha256": sha,
            "checkpoint_path": record["checkpoint"]["path"],
            "checkpoint_bytes": record["checkpoint"]["bytes"],
            "checkpoint_sha256": record["checkpoint"]["sha256"],
            "config_path": record["config"]["path"],
            "config_bytes": record["config"]["bytes"],
            "config_sha256": record["config"]["sha256"]}
  finally:
    temporary.unlink(missing_ok=True)


def _instructions(manifest_sha):
  command = ("python verify_backup.py verify --backup-dir . "
             "--manifest checkpoint_manifest.json --manifest-sha256 " + manifest_sha +
             " --report local-backup-verification.json")
  return f"""Checkpoint backup recovery and local verification

Download backup-metadata.zip and every ZIP named by backup-index.json into
one local directory. Extract only backup-metadata.zip first. Then run there:

{command}

The SHA-256 above identifies the original trusted retention manifest. Verify
the entire downloaded folder; a browser download request does not prove that
the files reached your computer. The original backup-index.json deliberately
retains local_download_verified=false. The separate verification report is
created only after every indexed archive, checkpoint and config matches.

After successful verification, standard ZIP tools can restore each checkpoint
and its model configuration at the original relative paths. Best and last
files can have identical hashes; preserve every indexed path. These weights
do not contain optimizer state and do not resume an interrupted optimizer.

Keep the original study directory and original metadata.zip. Source/runtime
identity files are included here when referenced by the retention manifest.
For this repository's recorded ForTy pilot, source identity is also retained at
results/architecture_pilot_t4_20261029/metadata/source_identity.json.

Archive creation verifies source bytes in the caller's runtime; it does not
verify local browser downloads, remote storage durability or source TFRecords.
The verify command checks bytes in the caller-supplied backup directory.
""".encode("utf-8")


def _write_new(path, content):
  _real_path(path)
  created = False
  try:
    with path.open("xb") as handle:
      created = True
      handle.write(content)
  except BaseException:
    if created:
      path.unlink(missing_ok=True)
    raise


def _metadata_zip(destination, index, manifest_bytes, support):
  items = {INDEX_NAME: _json_bytes(index),
           "checkpoint_manifest.json": manifest_bytes,
           "verify_backup.py": Path(__file__).read_bytes(),
           "README.txt": _instructions(index["manifest_sha256"]), **support}
  temporary = _new_temporary(destination)
  try:
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
      for name, value in items.items():
        archive.writestr(_zip_info(name, len(value)), value)
    _file_sha(temporary, ARCHIVE_LIMIT)
    with zipfile.ZipFile(temporary) as archive:
      _check_members(archive, {name: len(value) for name, value in items.items()})
      _require(all(archive.read(name) == value for name, value in items.items()),
               "Backup metadata ZIP mismatch")
    _publish(temporary, destination / METADATA_NAME)
  finally:
    temporary.unlink(missing_ok=True)


def create_backup(run_root, manifest, output_dir, archive_max_bytes=ARCHIVE_LIMIT,
                  manifest_sha256=None):
  """Verify source bytes and create new archives; never modify the source study."""
  limit = _integer(archive_max_bytes, ARCHIVE_LIMIT, "archive byte limit")
  _, raw, sha, records, files = _manifest(manifest, manifest_sha256)
  source = _real_path(run_root, directory=True)
  destination = _real_path(output_dir)
  _require(not destination.is_relative_to(source) and
           not source.is_relative_to(destination), "Source and destination overlap")
  if destination.exists():
    raise FileExistsError("Backup output directory must be fresh")
  _real_path(destination.parent, directory=True)
  configs, estimated = set(), 0
  for record in records:
    for key in ("checkpoint", "config"):
      item = record[key]
      path = _regular(source / item["path"])
      _require(path.stat().st_size == item["bytes"], "Source size mismatch")
    if record["config"]["path"] not in configs:
      _stream_expected(source / record["config"]["path"], record["config"])
      configs.add(record["config"]["path"])
    entries = [(record[k]["path"], record[k]["bytes"])
               for k in ("checkpoint", "config")]
    entries.append(("receipt.json", len(_json_bytes(_expected_receipt(record, sha)))))
    archive_size = _stored_zip_size(entries)
    _require(archive_size < limit, "Checkpoint archive cannot fit the transfer limit")
    estimated += archive_size
  support = {}
  for name in OPTIONAL_RECEIPTS:
    if name in files:
      item = files[name]
      _integer(item["bytes"], MAX_SMALL_BYTES, "identity receipt size")
      _stream_expected(source / name, item)
      support[name] = _read_bounded(source / name, MAX_SMALL_BYTES)
      _require(_digest(support[name]) == item["sha256"], "Identity receipt changed")
  _require(shutil.disk_usage(destination.parent).free >= estimated + 8 * MIB,
           "Insufficient disk space for the complete checkpoint backup")
  destination.mkdir(exist_ok=False)
  archives = [_create_archive(source, destination, record, sha, limit)
              for record in records]
  index = {"schema_version": 1, "status": CREATE_STATUS,
           "checkpoint_count": len(records), "manifest_sha256": sha,
           "source_checkpoint_bytes_verified": True, "local_download_verified": False,
           "archive_max_bytes_exclusive": limit, "archives": archives,
           "unique_checkpoint_hashes": len({r["checkpoint"]["sha256"]
                                            for r in records}),
           "support_files": [{"filename": name, "bytes": len(value),
                              "sha256": _digest(value)}
                             for name, value in support.items()],
           "backup_tool_sha256": _digest(Path(__file__).read_bytes()),
           "verification_scope": "Source and runtime archive bytes; "
           "browser downloads unverified."}
  _metadata_zip(destination, index, raw, support)
  _write_new(destination / INDEX_NAME, _json_bytes(index))
  return index


def _report_path(report, backup, manifest, document):
  if report is None:
    return None
  path = _real_path(report)
  _require(path.suffix == ".json", "Verification report must be a JSON file")
  _real_path(path.parent, directory=True)
  original = document.get("run_root")
  if isinstance(original, str) and Path(original).is_absolute():
    _require(not path.is_relative_to(Path(original).resolve()),
             "Report must be outside the original study")
  protected = {INDEX_NAME, METADATA_NAME, "checkpoint_manifest.json",
               "verify_backup.py", "README.txt", *OPTIONAL_RECEIPTS}
  _require(path != _real_path(manifest) and
           not (path.parent == backup and
                (path.name in protected or path.suffix == ".zip")),
           "Report would overwrite a protected backup artifact")
  if path.exists():
    raise FileExistsError("Verification report must be a new file")
  return path


def verify_backup(backup_dir, manifest, manifest_sha256=None, report=None):
  """Verify all supplied archive bytes against an original trusted manifest."""
  document, raw, sha, records, files = _manifest(manifest, manifest_sha256)
  backup = _real_path(backup_dir, directory=True)
  report_path = _report_path(report, backup, manifest, document)
  index = _parse_json(_read_bounded(backup / INDEX_NAME, MAX_MANIFEST_BYTES))
  _require(index.get("schema_version") == 1 and index.get("status") == CREATE_STATUS and
           index.get("checkpoint_count") == len(records) and
           index.get("manifest_sha256") == sha and
           index.get("source_checkpoint_bytes_verified") is True and
           index.get("local_download_verified") is False, "Invalid backup index")
  limit = _integer(index.get("archive_max_bytes_exclusive"), ARCHIVE_LIMIT,
                   "archive byte limit")
  entries = index.get("archives")
  _require(isinstance(entries, list) and len(entries) == len(records),
           "Incomplete backup archive index")
  expected = {r["filename"]: r for r in records}
  seen = set()
  for entry in entries:
    _require(isinstance(entry, dict), "Invalid archive index entry")
    name = entry.get("filename")
    _require(name in expected and name not in seen, "Unexpected or duplicate archive")
    seen.add(name)
    record = expected[name]
    for prefix in ("checkpoint", "config"):
      item = record[prefix]
      _require(all(entry.get(prefix + "_" + k) == item[k]
                   for k in ("path", "bytes", "sha256")),
               "Archive index receipt mismatch")
    _check_digest(entry.get("sha256"))
    size, checksum = _file_sha(backup / name, limit)
    _require(size == entry.get("bytes") and checksum == entry["sha256"],
             "Backup archive SHA-256/size mismatch")
    _verify_archive(backup / name, record, sha, limit)
  _require(seen == set(expected), "Incomplete checkpoint backup")
  metadata = _regular(backup / METADATA_NAME)
  _file_sha(metadata, ARCHIVE_LIMIT)
  support = {name: files[name] for name in OPTIONAL_RECEIPTS if name in files}
  with zipfile.ZipFile(metadata) as archive:
    sizes = {INDEX_NAME: len(_json_bytes(index)), "checkpoint_manifest.json": len(raw),
             "verify_backup.py": None, "README.txt": len(_instructions(sha))}
    sizes.update({name: item["bytes"] for name, item in support.items()})
    script_info = archive.getinfo("verify_backup.py")
    _integer(script_info.file_size, MAX_SMALL_BYTES, "portable verifier size")
    sizes["verify_backup.py"] = script_info.file_size
    _check_members(archive, sizes)
    _require(archive.read(INDEX_NAME) == _json_bytes(index) and
             archive.read("checkpoint_manifest.json") == raw and
             archive.read("README.txt") == _instructions(sha) and
             _digest(archive.read("verify_backup.py")) ==
             index.get("backup_tool_sha256"),
             "Backup metadata contents mismatch")
    for name, item in support.items():
      _require(_digest(archive.read(name)) == item["sha256"],
               "Identity receipt mismatch")
  result = {"schema_version": 1, "status": VERIFY_STATUS,
            "checkpoint_count": len(records), "manifest_sha256": sha,
            "local_download_verified": True,
            "verification_scope": "All indexed archive, checkpoint and config "
            "bytes in the supplied directory.",
            "source_records_verified": False,
            "checkpoint_bytes_verified": sum(r["checkpoint"]["bytes"] for r in records)}
  if report_path is not None:
    _write_new(report_path, _json_bytes(result))
  return result


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  commands = parser.add_subparsers(dest="command", required=True)
  create = commands.add_parser("create",
                               help="Verify source bytes and prepare bounded ZIPs")
  create.add_argument("--run-root", required=True)
  create.add_argument("--output-dir", required=True)
  create.add_argument("--archive-max-bytes", type=int, default=ARCHIVE_LIMIT)
  verify = commands.add_parser("verify",
                               help="Verify downloaded archives without TensorFlow")
  verify.add_argument("--backup-dir", required=True)
  verify.add_argument("--report")
  for command in (create, verify):
    command.add_argument("--manifest", required=True)
    command.add_argument("--manifest-sha256")
  args = parser.parse_args(argv)
  try:
    if args.command == "create":
      result = create_backup(args.run_root, args.manifest, args.output_dir,
                             args.archive_max_bytes, args.manifest_sha256)
    else:
      result = verify_backup(args.backup_dir, args.manifest,
                             args.manifest_sha256, args.report)
  except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
    parser.exit(2, f"Checkpoint backup failed: {error}\n")
  json.dump(result, sys.stdout, indent=2, sort_keys=True)
  print()


if __name__ == "__main__":
  main()
