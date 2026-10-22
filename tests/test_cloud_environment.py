"""Local checks for environment preparation without installing any packages."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "scripts" / "setup_cloud_env.sh"
SPEC = importlib.util.spec_from_file_location(
    "cloud_runtime_check", ROOT / "cloud" / "check_runtime.py")
runtime_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime_check)


def test_setup_plan_does_not_run_python_or_create_environment(tmp_path):
  target = tmp_path / "venv with spaces"
  result = subprocess.run([
      "bash", str(SETUP), "--dry-run", "--venv", str(target),
      "--python", "interpreter-does-not-exist"],
      text=True, capture_output=True, check=True)
  assert "requirements-gpu.txt" in result.stdout
  assert not target.exists()


def test_setup_rejects_existing_environment_without_changing_it(tmp_path):
  target = tmp_path / "existing"
  target.mkdir()
  marker = target / "original.txt"
  marker.write_bytes(b"existing environment")
  result = subprocess.run([
      "bash", str(SETUP), "--venv", str(target), "--python", sys.executable],
      text=True, capture_output=True)
  assert result.returncode != 0
  assert "already exists" in result.stderr
  assert list(target.iterdir()) == [marker]
  assert marker.read_bytes() == b"existing environment"


def test_failed_host_driver_stops_before_creating_or_installing(tmp_path):
  tools = tmp_path / "tools"
  tools.mkdir()
  driver = tools / "nvidia-smi"
  driver.write_text("#!/bin/sh\necho 'driver unavailable' >&2\nexit 1\n")
  driver.chmod(0o755)
  target = tmp_path / "new-venv"
  environment = {**os.environ, "PATH": str(tools) + os.pathsep + os.environ["PATH"]}
  result = subprocess.run([
      "bash", str(SETUP), "--venv", str(target), "--python", sys.executable],
      env=environment, text=True, capture_output=True)
  assert result.returncode != 0
  assert "driver unavailable" in result.stderr
  assert not target.exists()


def test_normalized_path_cannot_alias_an_existing_environment(tmp_path):
  target = tmp_path / "existing"
  target.mkdir()
  marker = target / "original.txt"
  marker.write_bytes(b"preserve me")
  alias = tmp_path / "missing" / ".." / "existing"
  result = subprocess.run([
      "bash", str(SETUP), "--dry-run", "--venv", str(alias)],
      text=True, capture_output=True)
  assert result.returncode != 0
  assert "already exists" in result.stderr
  assert marker.read_bytes() == b"preserve me"
  assert not (tmp_path / "missing").exists()


def test_cpu_runtime_executes_backward_and_restores_device_policy():
  previous = tf.config.get_soft_device_placement()
  report = runtime_check.check_runtime(require_gpu=False)
  assert report["status"] == "passed", report
  assert report["convolution_backward"] == "passed"
  assert "CPU:0" in report["probe_device"]
  assert tf.config.get_soft_device_placement() == previous


def test_gpu_runtime_rejects_hidden_devices_before_convolution(monkeypatch):
  monkeypatch.setattr(tf.test, "is_built_with_cuda", lambda: True)
  monkeypatch.setattr(tf.config, "get_visible_devices", lambda kind: [])
  monkeypatch.setattr(tf.io.gfile, "get_registered_schemes", lambda: ["file", "gs"])

  def unexpected_convolution(*args, **kwargs):
    raise AssertionError("GPU gate must run before convolution")

  monkeypatch.setattr(tf.nn, "conv2d", unexpected_convolution)
  report = runtime_check.check_runtime()
  assert report["status"] == "failed"
  assert "sees no GPU" in report["error"]


def test_runtime_rejects_missing_gcs_registration(monkeypatch):
  monkeypatch.setattr(tf.io.gfile, "get_registered_schemes", lambda: ["file"])
  report = runtime_check.check_runtime(require_gpu=False)
  assert report["status"] == "failed"
  assert "GCS filesystem support is missing" in report["error"]
