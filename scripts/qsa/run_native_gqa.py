#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Run native-QSA validation, paired timing and the retained quality gate serially."""

import json
import subprocess
import sys

from prepare_baseline import ROOT, SELECTOR_SHA, WORK

STAGES = [
    ("native-gqa-errors-status", ["scripts/qsa/validate_native_gqa_errors.py"]),
    ("native-gqa-final-validation", ["scripts/qsa/validate_native_gqa.py"]),
    (
        "native-gqa-4k-performance",
        ["scripts/qsa/benchmark_native_gqa.py", "--context", "4096", "--pairs", "6"],
    ),
    ("native-gqa-dolma-quality", ["scripts/qsa/benchmark_native_gqa.py", "--dolma"]),
    (
        "native-gqa-16k-performance",
        ["scripts/qsa/benchmark_native_gqa.py", "--context", "16384", "--pairs", "6"],
    ),
    (
        "native-gqa-32k-performance",
        ["scripts/qsa/benchmark_native_gqa.py", "--context", "32768", "--pairs", "6"],
    ),
    (
        "native-gqa-64k-performance",
        ["scripts/qsa/benchmark_native_gqa.py", "--context", "65472", "--pairs", "6"],
    ),
]


def main():
    status = {
        "state": "running",
        "batch": 1,
        "training_updates": 0,
        "completed_stages": [],
    }
    target = WORK / "native-gqa"
    try:
        for label, command in STAGES:
            status["stage"] = label
            (target / "campaign.json").write_text(json.dumps(status, indent=2) + "\n")
            previous = WORK / "stages" / label / "status.json"
            if previous.exists():
                done = json.loads(previous.read_text())
                assert done["state"] == "complete", (
                    "Investigate previous failure; do not retry automatically"
                )
                assert done["command"][2:] == command
                assert done["selector_sha256_after"] == SELECTOR_SHA
            else:
                print("Starting", label, flush=True)
                subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        "scripts/qsa/run_stage.py",
                        label,
                        "--",
                        *command,
                    ],
                    cwd=ROOT,
                    check=True,
                )
            status["completed_stages"].append(label)
            if (target / "paired-quality/dolma-summary.json").exists() and (
                target / "paired-performance/4096/status.json"
            ).exists():
                subprocess.run(
                    [sys.executable, "scripts/qsa/report_native_gqa.py"],
                    cwd=ROOT,
                    check=True,
                )
        status["state"] = "complete"
    except BaseException as exc:
        status.update(state="failed", error=repr(exc))
        raise
    finally:
        (target / "campaign.json").write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
