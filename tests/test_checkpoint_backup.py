"""Actual-byte backup checks are independent of TensorFlow and browser downloads."""

from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/checkpoint_backup.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_backup", SCRIPT)
backup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backup)


def sha(value):
  return hashlib.sha256(value).hexdigest()


def save_manifest(path, document):
  path.write_text(json.dumps(document, indent=2) + "\n")
  return sha(path.read_bytes())


@pytest.fixture
def study(tmp_path):
  root = tmp_path / "study"
  root.mkdir()
  manifest = tmp_path / "checkpoint_manifest.json"
  entries, files = [], []
  for seed in (42, 43, 44):
    for variant in ("cnn_only", "cnn_ssm"):
      arm = root / f"{variant}_{seed}"
      arm.mkdir()
      config = json.dumps({"seed": seed, "model_variant": variant}).encode()
      (arm / "config.json").write_bytes(config)
      files.append({"path": f"{arm.name}/config.json", "bytes": len(config),
                    "sha256": sha(config), "external": False, "kind": "run_artifact"})
      for label, kind in (("best", "selected_checkpoint"),
                          ("last", "last_weights_only")):
        # Five runs intentionally have identical best/last bytes.
        version = label if (variant, seed) == ("cnn_ssm", 42) else "shared"
        value = (f"{variant}-{seed}-{version}:" * 16).encode()
        path = f"{arm.name}/{label}.weights.h5"
        (root / path).write_bytes(value)
        entry = {"path": path, "bytes": len(value), "sha256": sha(value),
                 "external": False, "kind": kind, "seed": seed,
                 "model_variant": variant}
        entries.append(entry)
        files.append(dict(entry))
  for name, value in (("source_identity.json", b'{"commit": "fixture"}\n'),
                      ("runtime_identity.json", b'{"python": "fixture"}\n')):
    (root / name).write_bytes(value)
    files.append({"path": name, "bytes": len(value), "sha256": sha(value),
                  "external": False, "kind": "provenance_receipt"})
  document = {"schema_version": 1,
              "manifest_kind": "architecture_pilot_checkpoint_retention",
              "status": "completed_pilot", "selected_checkpoint_count": 6,
              "run_root": str(root), "checkpoints": entries, "files": files}
  digest = save_manifest(manifest, document)
  return SimpleNamespace(root=root, manifest=manifest, document=document,
                         sha=digest, output=tmp_path / "backup")


def create(study, **kwargs):
  return backup.create_backup(study.root, study.manifest, study.output,
                              manifest_sha256=study.sha, **kwargs)


def verify(study, **kwargs):
  return backup.verify_backup(study.output, study.manifest,
                              manifest_sha256=study.sha, **kwargs)


def source_snapshot(study):
  return {str(p.relative_to(study.root)): p.read_bytes()
          for p in study.root.rglob("*") if p.is_file()}


def test_twelve_file_roundtrip_and_portable_verifier(study, tmp_path):
  before = source_snapshot(study)
  index = create(study)
  assert index["status"] == "archives_created_local_download_not_verified"
  assert index["source_checkpoint_bytes_verified"] is True
  assert index["local_download_verified"] is False
  assert index["checkpoint_count"] == 12
  assert index["unique_checkpoint_hashes"] == 7
  assert len(index["archives"]) == 12
  assert all(a["bytes"] < backup.ARCHIVE_LIMIT for a in index["archives"])
  with zipfile.ZipFile(study.output / backup.METADATA_NAME) as archive:
    assert set(archive.namelist()) == {
        backup.INDEX_NAME, "checkpoint_manifest.json", "verify_backup.py", "README.txt",
        "source_identity.json", "runtime_identity.json"}
    for name in archive.namelist():
      (study.output / name).write_bytes(archive.read(name))
  report = study.output / "local-backup-verification.json"
  result = verify(study, report=report)
  assert result["status"] == "downloaded_backup_verified"
  assert result["local_download_verified"] is True
  assert result["checkpoint_count"] == 12
  assert result["checkpoint_bytes_verified"] == sum(
      e["bytes"] for e in study.document["checkpoints"])
  assert json.loads(report.read_text()) == result
  retained_index = json.loads((study.output / backup.INDEX_NAME).read_text())
  assert retained_index["local_download_verified"] is False
  portable = subprocess.run(
      [sys.executable, str(study.output / "verify_backup.py"), "verify",
       "--backup-dir", ".",
       "--manifest", "checkpoint_manifest.json", "--manifest-sha256", study.sha],
      cwd=study.output, capture_output=True, text=True, check=True)
  assert json.loads(portable.stdout)["status"] == backup.VERIFY_STATUS
  assert source_snapshot(study) == before


@pytest.mark.parametrize("kind", ["weight", "config", "identity"])
def test_changed_source_fails_without_touching_originals(study, kind):
  entry = study.document["checkpoints"][0]
  names = {"weight": entry["path"],
           "config": str(Path(entry["path"]).parent / "config.json"),
           "identity": "source_identity.json"}
  path = study.root / names[kind]
  content = path.read_bytes()
  path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
  before = source_snapshot(study)
  with pytest.raises(ValueError, match="SHA-256 mismatch"):
    create(study)
  assert source_snapshot(study) == before
  assert not (study.output / backup.INDEX_NAME).exists()
  assert not list(study.output.glob(".partial-*"))


@pytest.mark.parametrize("mutation", ["traversal", "external", "duplicate",
                                      "missing_pair", "bad_size", "bad_digest",
                                      "config_conflict"])
def test_bad_manifest_rejected_before_output(study, mutation):
  d = study.document
  if mutation == "traversal":
    d["checkpoints"][0]["path"] = "../outside.weights.h5"
  elif mutation == "external":
    d["checkpoints"][0]["external"] = True
  elif mutation == "duplicate":
    d["checkpoints"].append(dict(d["checkpoints"][0]))
  elif mutation == "missing_pair":
    d["checkpoints"].pop()
  elif mutation == "bad_size":
    d["checkpoints"][0]["bytes"] = True
  elif mutation == "bad_digest":
    d["checkpoints"][0]["sha256"] = "z" * 64
  else:
    d["files"] = [e for e in d["files"] if not e["path"].endswith("/config.json")]
  study.sha = save_manifest(study.manifest, d)
  with pytest.raises(ValueError):
    create(study)
  assert not study.output.exists()


def test_manifest_pin_and_disk_budget_reject_before_output(study, monkeypatch):
  with pytest.raises(ValueError, match="Trusted manifest"):
    backup.create_backup(study.root, study.manifest, study.output,
                         manifest_sha256="0" * 64)
  monkeypatch.setattr(backup.shutil, "disk_usage", lambda _p: SimpleNamespace(free=1))
  with pytest.raises(ValueError, match="Insufficient disk"):
    create(study)
  assert not study.output.exists()


def test_archive_limit_preflight_and_exact_boundary(study):
  with pytest.raises(ValueError, match="cannot fit"):
    create(study, archive_max_bytes=100)
  assert not study.output.exists()
  index = create(study)
  limit = max(e["bytes"] for e in index["archives"])
  shutil.rmtree(study.output)
  with pytest.raises(ValueError, match="cannot fit"):
    create(study, archive_max_bytes=limit)


@pytest.mark.parametrize("kind", ["source_file", "source_parent", "output_parent"])
def test_symlinks_rejected(study, tmp_path, kind):
  if kind == "source_file":
    path = study.root / study.document["checkpoints"][0]["path"]
    target = tmp_path / "outside.weights.h5"
    path.rename(target)
    path.symlink_to(target)
  elif kind == "source_parent":
    path = study.root / Path(study.document["checkpoints"][0]["path"]).parent
    target = tmp_path / "outside-arm"
    path.rename(target)
    path.symlink_to(target, target_is_directory=True)
  else:
    link = tmp_path / "alias"
    link.symlink_to(tmp_path, target_is_directory=True)
    study.output = link / "backup"
  with pytest.raises(ValueError, match="Symlinks"):
    create(study)


def test_existing_and_overlapping_outputs_are_unchanged(study):
  study.output.mkdir()
  sentinel = study.output / "keep.txt"
  sentinel.write_bytes(b"keep")
  with pytest.raises(FileExistsError):
    create(study)
  assert sentinel.read_bytes() == b"keep"
  for path in (study.root / "new-backup", study.root.parent):
    with pytest.raises(ValueError, match="overlap"):
      backup.create_backup(study.root, study.manifest, path)


def test_concurrent_target_creation_cannot_be_overwritten(study, monkeypatch):
  original = backup.os.link

  def raced(temporary, target):
    target.write_bytes(b"other writer")
    return original(temporary, target)

  monkeypatch.setattr(backup.os, "link", raced)
  before = source_snapshot(study)
  with pytest.raises(FileExistsError, match="Refusing to overwrite"):
    create(study)
  assert [p.read_bytes() for p in study.output.glob("*.zip")] == [b"other writer"]
  assert not (study.output / backup.INDEX_NAME).exists()
  assert not list(study.output.glob(".partial-*"))
  assert source_snapshot(study) == before


def test_interruption_keeps_completed_archives_but_no_success_index(study, monkeypatch):
  original = backup._create_archive
  count = 0

  def interrupt(source, destination, record, digest, limit):
    nonlocal count
    count += 1
    if count == 2:
      old_stream = backup._stream_expected

      def fail(_path, _record, output=None):
        if output is not None:
          output.write(b"partial")
          raise KeyboardInterrupt()
        return old_stream(_path, _record, output)

      monkeypatch.setattr(backup, "_stream_expected", fail)
    return original(source, destination, record, digest, limit)

  before = source_snapshot(study)
  monkeypatch.setattr(backup, "_create_archive", interrupt)
  with pytest.raises(KeyboardInterrupt):
    create(study)
  assert len(list(study.output.glob("*.zip"))) == 1
  assert not (study.output / backup.INDEX_NAME).exists()
  assert not list(study.output.glob(".partial-*"))
  assert source_snapshot(study) == before


@pytest.mark.parametrize("mutation", ["corrupt", "missing", "wrong_index",
                                      "missing_index_entry"])
def test_incomplete_or_corrupt_downloads_never_write_success_report(study, mutation):
  index = create(study)
  first = study.output / index["archives"][0]["filename"]
  if mutation == "corrupt":
    value = first.read_bytes()
    first.write_bytes(value[:-1] + bytes([value[-1] ^ 1]))
  elif mutation == "missing":
    first.unlink()
  elif mutation == "wrong_index":
    index["manifest_sha256"] = "0" * 64
  else:
    index["archives"].pop()
  if mutation in ("wrong_index", "missing_index_entry"):
    (study.output / backup.INDEX_NAME).write_bytes(backup._json_bytes(index))
  report = study.output / "verification.json"
  with pytest.raises((ValueError, OSError)):
    verify(study, report=report)
  assert not report.exists()


@pytest.mark.parametrize("mutation", ["extra", "duplicate", "traversal", "symlink",
                                      "payload"])
def test_unsafe_zip_members_fail_even_with_updated_index_hash(study, mutation):
  index = create(study)
  entry = index["archives"][0]
  path = study.output / entry["filename"]
  with zipfile.ZipFile(path) as old:
    values = [(info, old.read(info.filename)) for info in old.infolist()]
  if mutation == "symlink":
    values[0][0].external_attr = (stat.S_IFLNK | 0o777) << 16
  elif mutation == "payload":
    info, value = values[0]
    values[0] = (info, bytes([value[0] ^ 1]) + value[1:])
  else:
    name = {"extra": "extra.txt", "duplicate": values[0][0].filename,
            "traversal": "../escape"}[mutation]
    values.append((backup._zip_info(name, 4), b"oops"))
  warning = (pytest.warns(UserWarning, match="Duplicate name")
             if mutation == "duplicate" else nullcontext())
  with warning:
    with zipfile.ZipFile(path, "w") as new:
      for info, value in values:
        new.writestr(info, value)
  entry["bytes"], entry["sha256"] = path.stat().st_size, sha(path.read_bytes())
  (study.output / backup.INDEX_NAME).write_bytes(backup._json_bytes(index))
  with pytest.raises(ValueError):
    verify(study)


def test_reports_cannot_replace_source_or_backup_files(study):
  create(study)
  source = source_snapshot(study)
  for target in (study.root / "new-report.json", study.manifest,
                 study.output / backup.INDEX_NAME):
    with pytest.raises((ValueError, FileExistsError)):
      verify(study, report=target)
  existing = study.output / "old-report.json"
  existing.write_bytes(b"keep")
  with pytest.raises(FileExistsError):
    verify(study, report=existing)
  assert existing.read_bytes() == b"keep"
  assert source_snapshot(study) == source


def test_cli_help_requires_no_tensorflow():
  result = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                          capture_output=True, text=True, check=True)
  assert "create" in result.stdout and "verify" in result.stdout
