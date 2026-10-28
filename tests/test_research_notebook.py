"""The retained research notebook separates persistence and metadata transfer."""

import ast
import json
from pathlib import Path
import re


NOTEBOOK = (Path(__file__).resolve().parents[1]
            / "notebooks/colab_research_pilot.ipynb")


def cells():
  return json.loads(NOTEBOOK.read_text())["cells"]


def test_research_notebook_freezes_cohort_and_uses_drive_before_training():
  source = {cell["id"]: "".join(cell["source"]) for cell in cells()}
  constants = {}
  for node in ast.parse(source["configuration"]).body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
      try:
        constants[node.targets[0].id] = ast.literal_eval(node.value)
      except (ValueError, TypeError):
        pass
  assert re.fullmatch("[0-9a-f]{40}", constants["SOURCE_COMMIT"])
  assert constants["SEEDS"] == [42, 43, 44]
  assert constants["SELECTION_SEED"] == 2026
  assert constants["EPOCHS"] == 10
  assert constants["MAX_TRAIN_RUNS"] is None
  assert constants["RECOVER_LOCK"] is False
  assert 'drive.mount("/content/drive")' in source["persistent-study"]
  assert '"--data-kind", "real_forty"' in source["persistent-study"]
  assert '"--resume"' in source["persistent-study"]
  assert '"PYTHONPATH": str(SOURCE_ROOT / "src")' in source["restore-and-probe"]
  order = list(source)
  assert order.index("restore-and-probe") < order.index("persistent-study")
  assert order.index("archive-helper") < order.index("stage-selected-data")
  assert order.index("stage-selected-data") < order.index("execute-study")
  assert "execute_with_cleanup(staging" in source["stage-selected-data"]
  assert "archive_profiles.py" in source["archive-helper"]
  assert '"--output-dir", str(destination)' in source["archive-helper"]
  assert "sha256" in source["download-results"]
  assert "-seed" in source["download-results"]


def test_research_result_cell_handles_paused_training_without_false_completion(
        tmp_path):
  source = next("".join(cell["source"]) for cell in cells()
                if cell["id"] == "read-results")
  (tmp_path / "research_state.json").write_text(json.dumps({
      "state": "training_pending", "stages": {
          "train_cnn_ssm_42": {"state": "completed"},
          "train_cnn_only_43": {"state": "pending"}}}))
  (tmp_path / "checkpoint_inventory.json").write_text(json.dumps({
      "completed_training_runs": 1, "required_training_runs": 6}))
  output = []
  namespace = {"STUDY_ROOT": tmp_path, "json": json,
               "print": lambda *values: output.append(" ".join(map(str, values)))}
  exec(compile(source, "research-results", "exec"), namespace)
  assert "Execution state: training_pending" in output
  assert "Verified training runs: 1 / 6" in output
  assert any("pilot is not complete" in value for value in output)


def test_research_result_cell_reads_the_completed_report_schema(tmp_path):
  source = next("".join(cell["source"]) for cell in cells()
                if cell["id"] == "read-results")
  report = {"status": "completed_pilot", "summary": {
      "cnn_ssm": {}, "cnn_only": {}, "paired_delta_percentage_points": {
          "macro_f1_all_ids": {"mean": 1.0, "sample_sd": .2}}}}
  (tmp_path / "research_report.json").write_text(json.dumps(report))
  output = []
  namespace = {"STUDY_ROOT": tmp_path, "json": json,
               "print": lambda *values: output.append(" ".join(map(str, values)))}
  exec(compile(source, "completed-results", "exec"), namespace)
  assert "Report status: completed_pilot" in output
  assert any("Paired deltas" in value for value in output)
  assert not any("pilot is not complete" in value for value in output)
