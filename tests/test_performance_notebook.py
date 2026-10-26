"""Stopping the Colab driver also terminates its GPU worker process group."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

import pytest


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/colab_performance.ipynb"


@pytest.mark.parametrize("outcome", ["success", "failure", "interrupt", "timeout"])
def test_driver_always_cleans_up_its_process_group(outcome):
  notebook = json.loads(NOTEBOOK.read_text())
  for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
      compile("".join(cell["source"]), cell["id"], "exec")
      assert cell["execution_count"] is None and cell["outputs"] == []
  source = "".join(next(cell for cell in notebook["cells"]
                        if cell["id"] == "archive-helper")["source"])
  function = next(node for node in ast.parse(source).body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "execute_with_cleanup")
  events = []

  class Process:
    pid = 123456

    def __init__(self, argv, **kwargs):
      assert kwargs["start_new_session"] is True
      self.calls = 0

    def wait(self, **kwargs):
      self.calls += 1
      events.append(("wait", kwargs))
      if self.calls == 1 and outcome in ("interrupt", "timeout"):
        raise KeyboardInterrupt()
      if self.calls == 2 and outcome == "timeout":
        raise subprocess.TimeoutExpired(["driver"], 10)
      return 7 if outcome == "failure" else 0

  namespace = {
      "os": SimpleNamespace(killpg=lambda pid, sig: events.append((pid, sig))),
      "signal": signal,
      "subprocess": SimpleNamespace(
          Popen=Process, STDOUT=subprocess.STDOUT,
          TimeoutExpired=subprocess.TimeoutExpired,
          CalledProcessError=subprocess.CalledProcessError),
  }
  exec(compile(ast.Module(body=[function], type_ignores=[]), "cleanup", "exec"),
       namespace)
  run = namespace["execute_with_cleanup"]
  if outcome == "failure":
    with pytest.raises(subprocess.CalledProcessError):
      run(["driver"], "/tmp", object())
  elif outcome in ("interrupt", "timeout"):
    with pytest.raises(KeyboardInterrupt):
      run(["driver"], "/tmp", object())
  else:
    run(["driver"], "/tmp", object())
  assert (123456, signal.SIGTERM) in events
  assert (123456, signal.SIGKILL) in events
  assert events[-1][0] == "wait"
  assert events.index((123456, signal.SIGTERM)) < events.index(
      (123456, signal.SIGKILL))
