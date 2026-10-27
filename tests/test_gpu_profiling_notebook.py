"""Colab exports complete profile downloads and protects measured packages."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import sys
from types import ModuleType
import zipfile

import pytest


NOTEBOOK = (Path(__file__).resolve().parents[1]
            / "notebooks/colab_gpu_profiling.ipynb")


def cells():
  return {cell["id"]: "".join(cell["source"])
          for cell in json.loads(NOTEBOOK.read_text())["cells"]}


def test_notebook_pins_source_and_installs_viewer_only_after_measurements():
  sources = cells()
  config = ast.parse(sources["configuration"])
  commit = next(ast.literal_eval(node.value) for node in config.body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "SOURCE_COMMIT"
                        for target in node.targets))
  assert re.fullmatch("[0-9a-f]{40}", commit)
  code_before_viewer = "\n".join(
      value for key, value in sources.items()
      if key not in ("optional-viewer", "view-instructions"))
  assert '"pip", "install"' not in code_before_viewer
  assert '"--constraint"' in sources["optional-viewer"]
  for package in ("tensorflow", "tensorboard", "keras", "numpy", "protobuf",
                  "ml-dtypes"):
    assert '"' + package + '"' in sources["optional-viewer"]
  assert '"/GPU:0"' in sources["execute-study"]


@pytest.mark.parametrize("bundle_fits", [True, False])
def test_download_bundle_keeps_every_trace_part(tmp_path, bundle_fits):
  function = next(node for node in ast.parse(cells()["archive-helper"]).body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "bundle_downloads")
  namespace = {"zipfile": zipfile}
  exec(compile(ast.Module(body=[function], type_ignores=[]), "bundle", "exec"),
       namespace)
  names = ["metadata.zip", "profile-archives.json", "trace-0000-part-0000.zip"]
  payloads = {name: ("complete " + name).encode() for name in names}
  for name, value in payloads.items():
    (tmp_path / name).write_bytes(value)
  selected = namespace["bundle_downloads"](
      tmp_path, {"downloads": names}, limit=100000 if bundle_fits else 100)
  if bundle_fits:
    assert selected == ["batch-study-bundle.zip"]
    with zipfile.ZipFile(tmp_path / selected[0]) as archive:
      assert archive.namelist() == names
      assert {name: archive.read(name) for name in names} == payloads
  else:
    assert selected == names
    assert not (tmp_path / "batch-study-bundle.zip").exists()


@pytest.mark.parametrize("report", [None, {"status": "timing_passed", "summary": {}}])
def test_result_cell_can_show_incomplete_studies(tmp_path, monkeypatch, report):
  (tmp_path / "execution.json").write_text(json.dumps({"status": "failed"}))
  if report is not None:
    (tmp_path / "study.json").write_text(json.dumps(report))
  captured = []
  display = ModuleType("IPython.display")
  display.Markdown = lambda value: value
  display.display = captured.append
  monkeypatch.setitem(sys.modules, "IPython", ModuleType("IPython"))
  monkeypatch.setitem(sys.modules, "IPython.display", display)
  namespace = {"RUN_ROOT": tmp_path, "json": json,
               "print": lambda *values: captured.append(" ".join(map(str, values)))}
  exec(compile(cells()["read-results"], "results", "exec"), namespace)
  assert "Execution status: failed" in captured
  if report is None:
    assert any("Download" in value for value in captured)
  else:
    assert "Study status: timing_passed" in captured
    assert "GPU trace coverage: not verified: study incomplete" in captured
