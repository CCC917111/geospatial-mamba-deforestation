"""Check TensorFlow execution and GCS registration without reading source data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def check_runtime(require_gpu: bool = True) -> dict:
  """Returns readiness after a finite convolution backward pass on the device."""
  report = {"profile": "gpu" if require_gpu else "cpu", "status": "failed"}
  try:
    import tensorflow as tf

    report.update({
        "tensorflow_version": tf.__version__,
        "cuda_build": tf.test.is_built_with_cuda(),
        "filesystem_schemes": tf.io.gfile.get_registered_schemes(),
        "visible_gpus": [
            device.name for device in tf.config.get_visible_devices("GPU")],
    })
    if "gs" not in report["filesystem_schemes"]:
      raise RuntimeError("GCS filesystem support is missing")
    if require_gpu:
      if not report["cuda_build"]:
        raise RuntimeError("TensorFlow is a CPU build")
      if not report["visible_gpus"]:
        raise RuntimeError("TensorFlow sees no GPU")
    previous_placement = tf.config.get_soft_device_placement()
    try:
      tf.config.set_soft_device_placement(False)
      with tf.device("/GPU:0" if require_gpu else "/CPU:0"):
        inputs = tf.Variable(tf.ones((1, 8, 8, 3)))
        kernel = tf.Variable(tf.ones((3, 3, 3, 4)))
        with tf.GradientTape() as tape:
          output = tf.nn.conv2d(inputs, kernel, strides=1, padding="SAME")
          loss = tf.reduce_sum(output)
        gradients = tape.gradient(loss, [inputs, kernel])
        for gradient in gradients:
          if gradient is None:
            raise RuntimeError("Convolution gradient is missing")
          tf.debugging.assert_all_finite(gradient, "Non-finite convolution gradient")
          gradient.numpy()
      if require_gpu and ("GPU:0" not in output.device
                          or any("GPU:0" not in value.device for value in gradients)):
        raise RuntimeError("Convolution or its gradients did not execute on GPU")
      report.update({"status": "passed", "probe_device": output.device,
                     "convolution_backward": "passed"})
    finally:
      tf.config.set_soft_device_placement(previous_placement)
  except Exception as error:
    report["error"] = f"{type(error).__name__}: {error}"
  return report


def main() -> None:
  """Writes a pass/fail report and returns failure when the runtime is unready."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--cpu", action="store_true", help="Check CPU execution")
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  report = check_runtime(require_gpu=not args.cpu)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  if report["status"] != "passed":
    raise SystemExit(report["error"])
  print(f"Runtime ready: {report['probe_device']}; GCS filesystem registered")


if __name__ == "__main__":
  main()
