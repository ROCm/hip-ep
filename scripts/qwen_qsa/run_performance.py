#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Run Qwen timing candidates and long-quality controls as bounded serial stages."""

import json
import subprocess
import sys
from pathlib import Path

from verify_handoff import ROOT, WORK, sha


def main():
    report = json.loads((WORK / "report.json").read_text())
    assert all(
        item["state"] == "complete"
        for item in report["matched_native_quality"].values()
    )
    assert report["matched_native_comparison"]["ppl_gate_passed"]
    assert report["matched_native_comparison"]["gsm_gate_passed"]
    scripts = Path(__file__).parent
    names = [
        "benchmark_native.py",
        "native_inference.py",
        "evaluate_long_native.py",
        "native_long_inference.py",
    ]
    frozen = {name: sha(scripts / name) for name in names}
    stages = []
    for batch, contexts in [(1, [4096, 16384, 32768, 65536]), (4, [4096]), (8, [4096])]:
        for context in contexts:
            stages.append(
                (
                    f"qwen-native-perf-{context}-b{batch}",
                    "scripts/qwen_qsa/benchmark_native.py",
                    ["--context", str(context), "--batch", str(batch)],
                )
            )
    for context in [16384, 32768]:
        for variant in ["Original", "Changed"]:
            stages.append(
                (
                    f"qwen-native-long-{variant.lower()}-{context}",
                    "scripts/qwen_qsa/evaluate_long_native.py",
                    ["--context", str(context), "--variant", variant],
                )
            )
    for label, script, arguments in stages:
        assert all(sha(scripts / name) == expected for name, expected in frozen.items())
        status_path = ROOT / "_local/qsa/stages" / label / "status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            assert status["state"] == "complete", (
                f"Investigate retained stage {label}; no automatic retry or overlap"
            )
            print("Already complete", label, flush=True)
            continue
        command = [
            sys.executable,
            "-u",
            "scripts/qsa/run_stage.py",
            label,
            "--",
            script,
            *arguments,
        ]
        print("Starting", label, flush=True)
        with (WORK / (label + "-coordinator.log")).open("x", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
            while process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print("Running", label, flush=True)
        assert process.returncode == 0, (
            f"Stage failed: {label}; evidence retained, later stages not started"
        )
        subprocess.run(
            [sys.executable, "scripts/qwen_qsa/report_results.py"], cwd=ROOT, check=True
        )
        print("Completed", label, flush=True)
    print("Native Qwen timing and long-quality stages complete", flush=True)


if __name__ == "__main__":
    main()
