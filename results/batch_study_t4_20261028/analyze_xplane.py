"""Read-only offline analysis of restored XPlane GPU events.

Uses the protobuf schema bundled with TensorFlow 2.21.0, without converting
through private TensorBoard/XProf APIs. Profiled wall time is not benchmark
latency. Activity unions are trace coverage, not SM occupancy or utilization.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import re
import statistics

import tensorflow as tf
from tensorflow.tsl.profiler.protobuf import xplane_pb2


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def stat_value(plane, stat):
    field = stat.WhichOneof("value")
    if field is None:
        return None
    value = getattr(stat, field)
    if field == "ref_value":
        metadata = plane.stat_metadata.get(value)
        return metadata.name if metadata else "<unknown reference>"
    return value


def event_stats(plane, event):
    return {plane.stat_metadata[stat.metadata_id].name: stat_value(plane, stat)
            for stat in event.stats}


def intervals(events, bounds=None):
    values = []
    for event in events:
        start, end = event["start_ps"], event["end_ps"]
        if bounds:
            start, end = max(start, bounds[0]), min(end, bounds[1])
        if end > start:
            values.append((start, end))
    return values


def union_ps(values):
    if not values:
        return 0
    values = sorted(values)
    total, (start, end) = 0, values[0]
    for left, right in values[1:]:
        if left <= end:
            end = max(end, right)
        else:
            total += end - start
            start, end = left, right
    return total + end - start


def time_stats(events):
    durations = [e["end_ps"] - e["start_ps"] for e in events]
    return {"count": len(events),
            "sum_gpu_duration_ms": sum(durations) / 1e9,
            "union_gpu_duration_ms": union_ps(intervals(events)) / 1e9,
            "median_event_duration_us": statistics.median(durations) / 1e6
            if durations else None}


def analyze(path, record):
    space = xplane_pb2.XSpace()
    space.ParseFromString(path.read_bytes())
    events, steps, gpu_planes = [], [], []
    for plane in space.planes:
        if plane.name.startswith("/device:GPU:"):
            gpu_planes.append({"name": plane.name,
                               "stats": {plane.stat_metadata[s.metadata_id].name:
                                         stat_value(plane, s) for s in plane.stats},
                               "streams": [{"id": line.id, "name": line.name,
                                            "events": len(line.events)}
                                           for line in plane.lines]})
            for line in plane.lines:
                for event in line.events:
                    metadata = plane.event_metadata.get(event.metadata_id)
                    name = metadata.name if metadata else "<unknown>"
                    stats = event_stats(plane, event)
                    kind = ("compute_kernel" if "kernel_details" in stats else
                            name if name.startswith("Memcpy") else
                            "memset" if name.startswith("Memset") else "other")
                    start = line.timestamp_ns * 1000 + event.offset_ps
                    details = stats.get("memcpy_details", "")
                    size = re.search(r"(?:^| )size:(\d+)(?: |$)", details)
                    events.append({"name": name, "kind": kind,
                                   "tf_op": stats.get("tf_op", ""),
                                   "start_ps": start,
                                   "end_ps": start + event.duration_ps,
                                   "stream": line.name,
                                   "copy_bytes": int(size.group(1)) if size else None})
        if plane.name == "/host:CPU":
            for line in plane.lines:
                for event in line.events:
                    metadata = plane.event_metadata.get(event.metadata_id)
                    if metadata and metadata.name in ("inference_step", "training_step"):
                        start = line.timestamp_ns * 1000 + event.offset_ps
                        stats = event_stats(plane, event)
                        steps.append({"step_num": stats["step_num"],
                                      "start_ps": start,
                                      "end_ps": start + event.duration_ps})
    steps.sort(key=lambda x: x["step_num"])
    if len(steps) != record["steps"]:
        raise ValueError("Host step count differs from capture report")
    kernels = [e for e in events if e["kind"] == "compute_kernel"]
    copies = [e for e in events if e["kind"].startswith("Memcpy")]
    condition_copies = [e for e in copies
                        if "/scan/while/cond/" in e["tf_op"]]
    scan_forward = [e for e in events
                    if "/scan/" in e["tf_op"] and "gradient_tape" not in e["tf_op"]]
    scan_backward = [e for e in events
                     if "/scan/" in e["tf_op"] and "gradient_tape" in e["tf_op"]]
    by_kind = {kind: time_stats([e for e in events if e["kind"] == kind])
               for kind in sorted({e["kind"] for e in events})}
    per_step = []
    for step in steps:
        bounds = step["start_ps"], step["end_ps"]
        wall = step["end_ps"] - step["start_ps"]
        active = union_ps(intervals(events, bounds))
        kernel_active = union_ps(intervals(kernels, bounds))
        copy_active = union_ps(intervals(copies, bounds))
        condition_active = union_ps(intervals(condition_copies, bounds))
        row = {"step_num": step["step_num"], "profiled_host_step_ms": wall / 1e9,
               "recorded_gpu_activity_union_ms": active / 1e9,
               "recorded_kernel_activity_union_ms": kernel_active / 1e9,
               "recorded_memcpy_activity_union_ms": copy_active / 1e9,
               "forward_condition_copy_activity_union_ms": condition_active / 1e9,
               "fraction_with_recorded_gpu_activity": active / wall,
               "fraction_without_recorded_gpu_activity": 1 - active / wall}
        for name, group in (("scan_forward", scan_forward),
                            ("scan_backward", scan_backward)):
            within = [e for e in group
                      if e["start_ps"] < bounds[1] and e["end_ps"] > bounds[0]]
            clipped = intervals(within, bounds)
            row[name] = {
                "gpu_event_count": len(within),
                "event_envelope_ms": ((max(t[1] for t in clipped)
                                      - min(t[0] for t in clipped)) / 1e9
                                     if clipped else None),
                "gpu_activity_union_ms": union_ps(clipped) / 1e9}
        per_step.append(row)
    tf_ops = defaultdict(list)
    for event in events:
        tf_ops[(event["kind"], event["tf_op"])].append(event)
    totals = [{"kind": kind, "tf_op": op, **time_stats(items)}
              for (kind, op), items in tf_ops.items()]
    copies_by_op = [item for item in totals if item["kind"].startswith("Memcpy")]
    kernel_by_op = [item for item in totals if item["kind"] == "compute_kernel"]
    total_wall = sum(step["profiled_host_step_ms"] for step in per_step)
    total_active = sum(step["recorded_gpu_activity_union_ms"] for step in per_step)
    return {"name": path.parents[3].name, "mode": record["mode"],
            "batch_size": record["batch_size"], "precision": record["precision"],
            "trace_sha256": digest(path), "trace_bytes": path.stat().st_size,
            "recorded_source_commit": record["source"]["commit"],
            "runtime": record["runtime"], "xspace_errors": list(space.errors),
            "xspace_warnings": list(space.warnings), "gpu_planes": gpu_planes,
            "coverage": ("actual_gpu_stream_and_kernel_events_present"
                         if gpu_planes and kernels else "not_verified"),
            "gpu_event_totals_by_kind": by_kind,
            "capture_summary": {
                "profiled_host_steps_ms": total_wall,
                "recorded_gpu_activity_union_ms_within_steps": total_active,
                "fraction_with_recorded_gpu_activity_within_steps":
                    total_active / total_wall,
                "recorded_kernel_activity_union_ms_within_steps":
                    sum(step["recorded_kernel_activity_union_ms"] for step in per_step),
                "recorded_memcpy_activity_union_ms_within_steps":
                    sum(step["recorded_memcpy_activity_union_ms"] for step in per_step)},
            "steps": per_step,
            "forward_condition_copies": {
                **time_stats(condition_copies),
                "count_by_kind": dict(Counter(e["kind"] for e in condition_copies)),
                "byte_sizes": dict(Counter(e["copy_bytes"] for e in condition_copies)),
                "total_bytes": sum(e["copy_bytes"] or 0 for e in condition_copies)},
            "top_copy_ops_by_count": sorted(copies_by_op,
                                             key=lambda x: -x["count"])[:12],
            "top_kernel_ops_by_sum_duration": sorted(kernel_by_op,
                                                     key=lambda x: -x["sum_gpu_duration_ms"])[:20],
            "top_kernel_ops_by_count": sorted(kernel_by_op,
                                             key=lambda x: -x["count"])[:12]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    root = args.review_root.resolve()
    results = []
    for directory in sorted((root / "restored/study/profiles").iterdir()):
        trace = list(directory.rglob("*.xplane.pb"))
        if len(trace) != 1:
            raise ValueError("Expected one trace per capture")
        record = json.loads((root / "metadata/study" / (directory.name + ".json")).read_text())
        results.append(analyze(trace[0], record))
    output = {
        "schema_version": 1,
        "analysis_runtime": {"tensorflow": tf.__version__,
                             "python": platform.python_version()},
        "protobuf_schema": xplane_pb2.DESCRIPTOR.name,
        "protobuf_descriptor_sha256": hashlib.sha256(
            xplane_pb2.DESCRIPTOR.serialized_pb).hexdigest(),
        "analysis_script_sha256": digest(Path(__file__)),
        "method": {
            "time_axis": "line.timestamp_ns * 1000 + event.offset_ps; all durations in picoseconds",
            "step_boundary": "Host inference_step/training_step Trace annotations including synchronization",
            "kernel_detection": "GPU event carrying kernel_details stat",
            "op_attribution": "GPU event tf_op stat; ref_value resolves through plane.stat_metadata",
            "active_time": "Union of clipped GPU event intervals across all streams, avoiding overlap double counting",
            "scan_categories": "Literal /scan/ tf_op path, separated by presence of gradient_tape",
            "forward_condition_category": "Memcpy event whose tf_op contains /scan/while/cond/"},
        "limitations": [
            "Capture has profiler overhead; profiled host step times are not the unprofiled benchmark latencies.",
            "Recorded GPU activity fraction is not SM occupancy, utilization or global GPU-idle measurement.",
            "Summed operation durations may overlap and cannot be equated with synchronized wall time.",
            "Scan event envelopes are not exclusive operation duration and may include other concurrent work.",
            "Condition-copy counts and short GPU copy durations alone do not prove their host synchronization overhead dominates latency.",
            "The worker places its entire measure scope under explicit tf.device('/GPU:0'); normal automatic production placement needs a controlled counterfactual before deployment-bottleneck claims.",
            "This is direct public-schema event inspection, not the XProf viewer's GPU Kernel Stats aggregation."],
        "captures": results}
    path = root / "trace-analysis.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(path)
    for row in results:
        print(row["name"], row["coverage"], row["forward_condition_copies"])
        print("STEPS", row["steps"])


if __name__ == "__main__":
    main()
