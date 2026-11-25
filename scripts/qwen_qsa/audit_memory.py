#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Summarize the append-only OS/WDDM trace without conflating memory gauges."""

import hashlib
import json

from verify_handoff import ROOT, WORK, sha


def audit_memory():
    folder = WORK / "memory-v1"
    if not (folder / "samples.jsonl").exists():
        return None
    protocol = json.loads((folder / "protocol.json").read_text())
    assert not protocol["gpu_context_created"]
    assert sha(folder / "sampler.py") == protocol["sampler_sha256"]
    raw = (folder / "samples.jsonl").read_bytes()
    # A live writer may be midway through its final line. Pin an immutable
    # complete-line prefix and its length/hash; later appends stay auditable.
    end = raw.rfind(b"\n") + 1
    prefix = raw[:end]
    frames = [json.loads(line) for line in prefix.splitlines()]
    assert frames and all(frame["unix"] >= protocol["started_unix"] for frame in frames)
    summary = {
        "protocol_sha256": sha(folder / "protocol.json"),
        "snapshot_bytes": end,
        "snapshot_sha256": hashlib.sha256(prefix).hexdigest(),
        "samples": len(frames),
        "interval_seconds": protocol["interval_seconds"],
        "scope": protocol["scope"],
        "notes": protocol["notes"],
        "rows": [],
    }
    stop_path = folder / "status.json"
    sampler_status = json.loads(stop_path.read_text()) if stop_path.exists() else None
    for label in protocol["labels"]:
        selected = [
            frame for frame in frames if frame["stage"] == label and frame["processes"]
        ]
        if not selected:
            continue
        stage = json.loads(
            (ROOT / "_local/qsa/stages" / label / "status.json").read_text()
        )
        before_start = protocol["started_unix"] <= stage["started_unix"]
        stage_end = (
            stage["started_unix"] + stage["elapsed_seconds"]
            if "elapsed_seconds" in stage
            else None
        )
        spans_end = bool(
            stage_end and sampler_status and sampler_status["ended_unix"] >= stage_end
        )
        coverage = (
            "startup through completion (sampled)"
            if before_start and spans_end
            else "in progress, attached before startup"
            if before_start
            else "partial, attached mid-run"
        )
        if stage["state"] == "failed":
            coverage = (
                "startup through failed attempt"
                if before_start
                else "partial failed attempt"
            )
        row = {
            "stage": label,
            "context": int(label.split("-")[-2]),
            "batch": 1,
            "coverage": coverage,
            "stage_state": stage["state"],
            "samples": len(selected),
            "first_sample_unix": selected[0]["unix"],
            "last_sample_unix": selected[-1]["unix"],
            "peak_sampled_rss_tree_bytes": max(
                frame["rss_tree_sum_bytes"] for frame in selected
            ),
            "peak_sampled_private_commit_tree_bytes": max(
                frame["private_commit_tree_sum_bytes"] for frame in selected
            ),
            "max_worker_lifetime_peak_rss_bytes": max(
                process["lifetime_peak_rss_bytes"]
                for frame in selected
                for process in frame["processes"]
            ),
            "max_worker_lifetime_peak_private_commit_bytes": max(
                process["lifetime_peak_private_commit_bytes"]
                for frame in selected
                for process in frame["processes"]
            ),
            "minimum_system_ram_available_bytes": min(
                frame["system_ram_available_bytes"] for frame in selected
            ),
            "maximum_system_ram_used_bytes": max(
                frame["system_ram_used_bytes"] for frame in selected
            ),
            "maximum_sample_gap_seconds": max(
                (
                    right["unix"] - left["unix"]
                    for left, right in zip(selected, selected[1:])
                ),
                default=0,
            ),
            "gpu_counter_sample_count": sum(
                bool(frame["wddm"].get("process")) for frame in selected
            ),
        }
        for field in [
            "dedicated_usage_bytes",
            "shared_usage_bytes",
            "total_committed_bytes",
        ]:
            values = [
                sum(
                    process[field]
                    for process in frame["wddm"].get("process", {}).values()
                    if field in process
                )
                for frame in selected
                if any(
                    field in process
                    for process in frame["wddm"].get("process", {}).values()
                )
            ]
            row["peak_sampled_wddm_" + field] = max(values) if values else None
        summary["rows"].append(row)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
