#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Independently check completed native timing rows and their paired statistics."""

import json
import math

import numpy as np
from verify_handoff import WORK, sha


def audit_performance():
    task = WORK / "native-gqa"
    timing = json.loads((WORK / "timing-v1/manifest.json").read_text())
    validation = json.loads((task / "full-model-validation.json").read_text())
    results = []
    for path in sorted((task / "paired-performance").glob("*/summary.json")):
        folder = path.parent
        summary = json.loads(path.read_text())
        protocol = json.loads((folder / "protocol.json").read_text())
        assert summary["status"] == "complete"
        assert (
            summary["replay_exact"]
            and summary["sessions_unchanged"]
            and summary["cache_pointers_unchanged"]
        )
        assert summary["rows_sha256"] == sha(folder / "rows.jsonl")
        assert summary["protocol_sha256"] == sha(folder / "protocol.json")
        assert protocol["binary_sha256"] == validation["binary_sha256"]
        assert protocol["stream_sha256"] == timing["streams_sha256"]
        assert protocol["timing_manifest_sha256"] == sha(
            WORK / "timing-v1/manifest.json"
        )
        assert protocol["warmup_pairs"] == timing["warmup_pairs"] == 2
        assert protocol["measured_pairs"] == timing["measured_pairs"] == 6
        assert protocol["decode_calls"] == timing["decode_calls"] == 64
        B, context = protocol["batch"], protocol["context"]
        assert B == summary["batch"] and context == summary["context"]
        assert protocol["capacity"] == context + 64
        rows = [
            json.loads(line)
            for line in (folder / "rows.jsonl").read_text().splitlines()
        ]
        assert (
            len(rows) == 12
            and len({(row["pair"], row["variant"]) for row in rows}) == 12
        )
        for pair in range(6):
            expected = (
                ["Original", "Changed"] if pair % 2 == 0 else ["Changed", "Original"]
            )
            pair_rows = [row for row in rows if row["pair"] == pair]
            assert [row["variant"] for row in pair_rows] == expected
            assert [row["order"] for row in pair_rows] == [0, 1]
        samples = np.random.default_rng(42).integers(0, 6, size=(10000, 6))
        for phase, tokens in [("prefill", B * context), ("decode", B * 64)]:
            for row in rows:
                seconds = row[phase + "_seconds"]
                assert row["batch"] == B and row["context"] == context
                assert math.isfinite(seconds) and seconds > 0
                assert math.isclose(
                    row[phase + "_tokens_s"], tokens / seconds, rel_tol=1e-12
                )
            times = {
                variant: np.array(
                    [
                        next(
                            row[phase + "_seconds"]
                            for row in rows
                            if row["pair"] == pair and row["variant"] == variant
                        )
                        for pair in range(6)
                    ]
                )
                for variant in ["Original", "Changed"]
            }
            change = 100 * (times["Original"].mean() / times["Changed"].mean() - 1)
            gains = 100 * (
                times["Original"][samples].mean(1) / times["Changed"][samples].mean(1)
                - 1
            )
            reported = summary["phases"][phase]
            assert math.isclose(
                reported["throughput_change_percent"], change, abs_tol=1e-10
            )
            assert np.allclose(
                reported["paired_95_interval_percent"],
                np.quantile(gains, [0.025, 0.975]),
                rtol=1e-12,
                atol=1e-10,
            )
            for variant in times:
                assert math.isclose(
                    reported[variant]["mean_seconds"],
                    float(times[variant].mean()),
                    rel_tol=1e-12,
                )
                assert math.isclose(
                    reported[variant]["tokens_s"],
                    float(tokens / times[variant].mean()),
                    rel_tol=1e-12,
                )
        results.append(
            {
                "batch": B,
                "context": context,
                "phases": summary["phases"],
                "summary_sha256": sha(path),
                "protocol_sha256": sha(folder / "protocol.json"),
                "rows_sha256": sha(folder / "rows.jsonl"),
            }
        )
    return sorted(results, key=lambda row: (row["batch"], row["context"]))
