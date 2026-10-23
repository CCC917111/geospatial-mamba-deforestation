"""Validate the free-GPU notebook without Colab, network or TensorFlow."""

from __future__ import annotations

import ast
import io
import json
from pathlib import Path
import re
import sys
import warnings
import zipfile

import pytest

from mamba_forest import experiment

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "colab_free_gpu.ipynb"
SOURCE_COMMIT = "ae6a123d0469a2950ce6b9733cf7902b14cb7f5c"
BUNDLE_MEMBER = "geospatial-mamba-deforestation.bundle"


def _notebook():
  return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell):
  value = cell["source"]
  return value if isinstance(value, str) else "".join(value)


def _cell(identifier):
  return next(cell for cell in _notebook()["cells"] if cell["id"] == identifier)


def _literal_configuration():
  values = {}
  tree = ast.parse(_source(_cell("configuration")))
  for statement in tree.body:
    if isinstance(statement, ast.Assign):
      try:
        value = ast.literal_eval(statement.value)
      except (ValueError, TypeError):
        continue
      for target in statement.targets:
        if isinstance(target, ast.Name):
          values[target.id] = value
  return values


def _function(cell_id, name, namespace):
  tree = ast.parse(_source(_cell(cell_id)))
  definitions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == name]
  assert len(definitions) == 1 and not definitions[0].decorator_list
  module = ast.Module(body=definitions, type_ignores=[])
  exec(compile(module, f"{NOTEBOOK.name}:{cell_id}:{name}", "exec"), namespace)
  return namespace[name]


def _zip(entries):
  payload = io.BytesIO()
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    with zipfile.ZipFile(payload, "w") as archive:
      for name, content in entries:
        archive.writestr(name, content)
  return payload.getvalue()


def test_notebook_is_unexecuted_valid_python_with_unique_cell_ids():
  notebook = _notebook()
  assert notebook["nbformat"] == 4
  identifiers = [cell["id"] for cell in notebook["cells"]]
  assert len(identifiers) == len(set(identifiers))
  assert all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value) for value in identifiers)
  for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
      assert cell["execution_count"] is None and cell["outputs"] == []
      # Python compilation rejects shell/IPython magics without running cells.
      compile(_source(cell), f"{NOTEBOOK.name}:{cell['id']}", "exec")


def test_default_configuration_is_small_gpu_run_with_optional_drive_disabled():
  config = _literal_configuration()
  for key in ("TRAIN_SHARDS", "VAL_SHARDS", "TEST_SHARDS", "BATCH_SIZE"):
    assert type(config[key]) is int and config[key] == 1
  assert config["EPOCHS"] == 2
  assert config["SEEDS"] == [42]
  assert config["SAVE_TO_DRIVE"] is False
  assert config["SOURCE_COMMIT"] == SOURCE_COMMIT


def test_notebook_does_not_embed_keys_or_paid_provisioning_sdks():
  denied = ("boto3", "runpod", "vastai", "azure.mgmt.compute", "google.cloud.compute")
  sensitive = re.compile(r"api_?key|access_?token|secret_?key|private_?key", re.I)
  for cell in _notebook()["cells"]:
    if cell["cell_type"] != "code":
      continue
    tree = ast.parse(_source(cell))
    for node in ast.walk(tree):
      if isinstance(node, ast.Import):
        assert all(not alias.name.startswith(denied) for alias in node.names)
      if isinstance(node, ast.ImportFrom):
        assert not (node.module or "").startswith(denied)
      if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        has_secret_name = any(sensitive.fullmatch(name) for name in names)
        assert not (node.value.value and has_secret_name)
    assert "-----BEGIN PRIVATE KEY-----" not in _source(cell)


def test_extracted_builder_produces_real_parser_compatible_gpu_plan(tmp_path):
  namespace = {
      **_literal_configuration(), "sys": sys,
      "RUN_ROOT": tmp_path / "experiment",
  }
  builder = _function("plan_experiment", "experiment_command", namespace)
  default_command = builder()
  execute_command = builder(execute=True)
  assert default_command[:3] == [sys.executable, "-m", "mamba_forest.experiment"]
  assert all(isinstance(value, str) for value in default_command)
  assert "--execute" not in default_command
  assert execute_command == [*default_command, "--execute"]
  args = experiment.parse_args(default_command[3:])
  assert args.execute is False and args.allow_cpu is False
  assert experiment.parse_args(execute_command[3:]).execute is True
  plan = experiment.make_plan(args)
  config = plan["config"]
  assert config["epochs"] == 2 and config["seeds"] == [42]
  assert config["batch_size"] == 1
  assert config["train_shards"] == config["val_shards"] == config["test_shards"] == 1
  assert config["precision"] == "mixed_float16" and config["allow_cpu"] is False
  training = [step for step in plan["steps"] if step["phase"] == "train"]
  assert [step["arm"] for step in training] == ["combo", "plain_ce"]
  for step in training:
    argv = step["argv"]
    assert argv[argv.index("--epochs") + 1] == "2"
    assert "--fixed-train-subset" in argv and "--allow-cpu" not in argv
  assert plan["steps"][0]["argv"][-1] == "gpu"
  preflight = next(step for step in plan["steps"] if step["phase"] == "preflight")
  assert "--require-gpu" in preflight["argv"]
  assert not namespace["RUN_ROOT"].exists()


def test_bundle_restore_returns_only_known_member_without_extracting(monkeypatch):
  restore = _function("restore_source", "bundle_bytes", {"io": io, "zipfile": zipfile})
  payload = b"# v2 git bundle\nexample payload\n"

  def forbidden(*args, **kwargs):
    pytest.fail("ZIP restoration must read bytes without filesystem extraction")

  monkeypatch.setattr(zipfile.ZipFile, "extract", forbidden)
  monkeypatch.setattr(zipfile.ZipFile, "extractall", forbidden)
  assert restore("source.bundle", payload) == payload
  assert restore("source.zip", _zip([(BUNDLE_MEMBER, payload)])) == payload
  archive = _zip([(BUNDLE_MEMBER, payload), ("../outside.txt", b"ignored"),
                  ("nested/other.bundle", b"ignored")])
  assert restore("source.zip", archive) == payload


@pytest.mark.parametrize("entries", [
    [], [("README.txt", b"missing")], [("../" + BUNDLE_MEMBER, b"traversal")],
    [("/" + BUNDLE_MEMBER, b"absolute")], [("nested/" + BUNDLE_MEMBER, b"nested")],
    [(BUNDLE_MEMBER, b"first"), (BUNDLE_MEMBER, b"duplicate")],
])
def test_invalid_or_ambiguous_bundle_archives_rejected(entries):
  restore = _function("restore_source", "bundle_bytes", {"io": io, "zipfile": zipfile})
  with pytest.raises(ValueError):
    restore("source.zip", _zip(entries))


def test_malformed_zip_and_unsupported_upload_rejected():
  restore = _function("restore_source", "bundle_bytes", {"io": io, "zipfile": zipfile})
  with pytest.raises((ValueError, zipfile.BadZipFile)):
    restore("broken.zip", b"not an archive")
  with pytest.raises(ValueError):
    restore("source.tar", b"not a supported bundle")
