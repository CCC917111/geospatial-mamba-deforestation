"""Profile exports keep complete traces while bounding every download."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/archive_profiles.py"
SPEC = importlib.util.spec_from_file_location("profile_archives", SCRIPT)
archives = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archives)


def make_export(tmp_path):
  run = tmp_path / "run"
  run.mkdir()
  trace = run / "profiles/inference/plugins/profile/stamp/host.xplane.pb"
  trace.parent.mkdir(parents=True)
  contents = bytes(range(256)) * 13
  trace.write_bytes(contents)
  (run / "report.json").write_text('{"status": "passed"}\n')
  (run / "weights.h5").write_bytes(b"weights stay in runtime")
  (run / "inputs.npz").write_bytes(b"inputs stay in runtime")
  (run / "long.log").write_text("x" * 500)
  output = tmp_path / "download"
  manifest = archives.export_study(
      run, output, chunk_bytes=700, metadata_file_bytes=100)
  return run, trace, contents, output, manifest


def test_profile_roundtrip_preserves_every_byte_and_metadata_limits(tmp_path):
  run, trace, contents, output, manifest = make_export(tmp_path)
  record = manifest["traces"][0]
  assert manifest["trace_status"] == "complete"
  assert len(record["chunks"]) == 5
  assert all(chunk["bytes"] <= 700 for chunk in record["chunks"])
  assert record["sha256"] == hashlib.sha256(contents).hexdigest()
  with zipfile.ZipFile(output / "metadata.zip") as archive:
    assert set(archive.namelist()) == {"report.json", archives.MANIFEST_NAME}
    assert json.loads(archive.read(archives.MANIFEST_NAME)) == manifest
  omissions = {entry["path"]: entry["reason"]
               for entry in manifest["metadata"]["omitted"]}
  assert omissions == {"weights.h5": "excluded_binary_or_extension",
                       "inputs.npz": "excluded_binary_or_extension",
                       "long.log": "metadata_budget"}
  for file in output.glob("*.zip"):
    assert file.stat().st_size < 32 * archives.MIB
  restored = tmp_path / "restored"
  result = archives.restore_profiles(
      output / archives.MANIFEST_NAME, output, restored)
  assert result["status"] == "restored" and result["traces"] == 1
  assert (restored / trace.relative_to(run)).read_bytes() == contents
  assert not (restored / "weights.h5").exists()
  assert not (restored / "report.json").exists()
  with pytest.raises(FileExistsError):
    archives.restore_profiles(output / archives.MANIFEST_NAME, output, restored)
  with pytest.raises(FileExistsError):
    archives.export_study(run, output)


@pytest.mark.parametrize("failure", ["missing", "corrupt_archive", "corrupt_chunk",
                                     "corrupt_trace", "extra_entry", "zip_path",
                                     "symlink_entry", "symlink_archive"])
def test_restore_rejects_incomplete_or_changed_profile(tmp_path, failure):
  _, _, _, output, manifest = make_export(tmp_path)
  record = manifest["traces"][0]
  chunk = record["chunks"][0]
  archive_path = output / chunk["archive"]
  if failure == "missing":
    archive_path.unlink()
  elif failure == "corrupt_archive":
    archive_path.write_bytes(b"not the exported archive")
  elif failure == "corrupt_chunk":
    chunk["sha256"] = "0" * 64
  elif failure == "corrupt_trace":
    record["sha256"] = "0" * 64
  elif failure == "symlink_archive":
    other = output / "other.zip"
    archive_path.rename(other)
    archive_path.symlink_to(other)
  else:
    with zipfile.ZipFile(archive_path) as archive:
      content = archive.read("chunk.bin")
    with zipfile.ZipFile(archive_path, "w") as archive:
      if failure == "zip_path":
        archive.writestr("../chunk.bin", content)
      elif failure == "symlink_entry":
        info = zipfile.ZipInfo("chunk.bin")
        info.create_system = 3
        info.external_attr = 0o120777 << 16
        archive.writestr(info, content)
      else:
        archive.writestr("chunk.bin", content)
        archive.writestr("unexpected.txt", b"extra")
    chunk["archive_sha256"] = archives.digest(archive_path)
  (output / archives.MANIFEST_NAME).write_text(json.dumps(manifest))
  destination = tmp_path / "failed"
  with pytest.raises(ValueError):
    archives.restore_profiles(output / archives.MANIFEST_NAME, output, destination)
  assert not destination.exists()


@pytest.mark.parametrize("path", ["../host.xplane.pb", "/tmp/host.xplane.pb",
                                  "safe/../../host.xplane.pb",
                                  "safe\\host.xplane.pb", "C:/host.xplane.pb",
                                  "./host.xplane.pb", "safe//host.xplane.pb",
                                  "run.py", "safe/host.xplane.pb/../run.py"])
def test_restore_rejects_unsafe_or_non_trace_paths(tmp_path, path):
  _, _, _, output, manifest = make_export(tmp_path)
  manifest["traces"][0]["path"] = path
  (output / archives.MANIFEST_NAME).write_text(json.dumps(manifest))
  destination = tmp_path / "unsafe"
  with pytest.raises(ValueError):
    archives.restore_profiles(output / archives.MANIFEST_NAME, output, destination)
  assert not destination.exists()


def test_export_omits_symlinks_and_rejects_nested_output(tmp_path):
  run = tmp_path / "run"
  run.mkdir()
  elsewhere = tmp_path / "real.xplane.pb"
  elsewhere.write_bytes(b"not followed")
  (run / "linked.xplane.pb").symlink_to(elsewhere)
  output = tmp_path / "output"
  manifest = archives.export_study(run, output)
  assert manifest["trace_status"] == "none"
  assert manifest["metadata"]["omitted"] == [
      {"path": "linked.xplane.pb", "reason": "symlink"}]
  with pytest.raises(ValueError, match="outside"):
    archives.export_study(run, run / "nested")


def test_restore_rejects_false_sizes_and_resource_budgets(tmp_path):
  _, _, _, output, manifest = make_export(tmp_path)
  manifest["traces"][0]["bytes"] += 1
  (output / archives.MANIFEST_NAME).write_text(json.dumps(manifest))
  with pytest.raises(ValueError, match="sizes"):
    archives.restore_profiles(output / archives.MANIFEST_NAME, output,
                              tmp_path / "bad-size")
  with pytest.raises(ValueError, match="chunk bytes"):
    archives.export_study(tmp_path, tmp_path / "never", chunk_bytes=32 * archives.MIB)


def test_zero_length_trace_roundtrip(tmp_path):
  run = tmp_path / "run"
  run.mkdir()
  (run / "empty.xplane.pb").write_bytes(b"")
  output = tmp_path / "export"
  archives.export_study(run, output)
  restored = tmp_path / "restored"
  archives.restore_profiles(output / archives.MANIFEST_NAME, output, restored)
  assert (restored / "empty.xplane.pb").read_bytes() == b""
