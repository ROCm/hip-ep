#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Run the remaining authorized B1 comparison stages serially; never retry failures."""

import json
import subprocess
import sys

from prepare_baseline import ROOT, SELECTOR_SHA, WORK

STAGES = [
    (
        "paired-native-4k-performance",
        ["scripts/qsa/benchmark_pair.py", "--context", "4096", "--pairs", "6"],
    ),
    ("paired-gsm8k-quality", ["scripts/qsa/evaluate_pair.py", "gsm8k"]),
    (
        "paired-16k-performance",
        ["scripts/qsa/benchmark_pair.py", "--context", "16384", "--pairs", "6"],
    ),
    (
        "paired-32k-performance",
        ["scripts/qsa/benchmark_pair.py", "--context", "32768", "--pairs", "6"],
    ),
    (
        "paired-isolated-64k-performance",
        ["scripts/qsa/benchmark_isolated.py", "--context", "65472"],
    ),
]


def main():
    assert (
        json.loads((WORK / "qsa-long-validation.json").read_text())["status"]
        == "passed"
    )
    status_path = WORK / "remaining-campaign.json"
    receipt = {
        "state": "running",
        "batch": 1,
        "training_updates": 0,
        "completed_stages": [],
    }
    try:
        for label, command in STAGES:
            receipt["stage"] = label
            status_path.write_text(json.dumps(receipt, indent=2) + "\n")
            previous = WORK / "stages" / label / "status.json"
            if previous.exists():
                done = json.loads(previous.read_text())
                assert done["state"] == "complete", (
                    "A failed/running stage requires investigation; no automatic retry"
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
            receipt["completed_stages"].append(label)
            subprocess.run(
                [sys.executable, "scripts/qsa/report_results.py"], cwd=ROOT, check=True
            )
        receipt["state"] = "complete"
    except BaseException as exc:
        receipt.update(state="failed", error=repr(exc))
        raise
    finally:
        status_path.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
