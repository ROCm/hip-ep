#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Run matched native Qwen quality serially; stop on any failed bounded stage."""

import json
import subprocess
import sys
from pathlib import Path

from verify_handoff import ROOT, WORK, sha


def counts():
    return {
        variant: {
            task: len(
                (WORK / "native-gqa/quality" / variant / (task + ".jsonl"))
                .read_text(encoding="utf-8")
                .splitlines()
            )
            if (WORK / "native-gqa/quality" / variant / (task + ".jsonl")).exists()
            else 0
            for task in ["ppl", "gsm8k"]
        }
        for variant in ["Original", "Changed"]
    }


def main():
    scripts = Path(__file__).parent
    frozen = {
        name: sha(scripts / name)
        for name in ["evaluate_native.py", "native_inference.py"]
    }
    stages = [
        (variant, task, start, count)
        for task, limit, count in [("ppl", 64, 8), ("gsm8k", 128, 32)]
        for variant in ["Original", "Changed"]
        for start in range(0, limit, count)
    ]
    for variant, task, start, count in stages:
        assert all(sha(scripts / name) == expected for name, expected in frozen.items())
        label = f"qwen-native-{variant.lower()}-{task}-{start:02d}-{start + count:02d}"
        status = ROOT / "_local/qsa/stages" / label / "status.json"
        path = WORK / "native-gqa/quality" / variant / (task + ".jsonl")
        if status.exists():
            assert json.loads(status.read_text())["state"] == "complete", (
                f"Investigate retained failed stage {label}; no automatic retry"
            )
            done = {
                json.loads(line)["index"]
                for line in path.read_text(encoding="utf-8").splitlines()
            }
            assert set(range(start, start + count)) <= done
            print("Already complete", label, counts(), flush=True)
            continue
        command = [
            sys.executable,
            "-u",
            "scripts/qsa/run_stage.py",
            label,
            "--",
            "scripts/qwen_qsa/evaluate_native.py",
            "--variant",
            variant,
            "--task",
            task,
            "--start",
            str(start),
            "--count",
            str(count),
        ]
        print("Starting", label, counts(), flush=True)
        with (WORK / (label + ".log")).open("x", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
            while process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print("Progress", label, counts(), flush=True)
        assert process.returncode == 0, f"Stage failed: {label}; evidence retained"
        subprocess.run(
            [sys.executable, "scripts/qwen_qsa/report_results.py"], cwd=ROOT, check=True
        )
        print("Completed", label, counts(), flush=True)
    for variant in ["Original", "Changed"]:
        audited = json.loads(
            (WORK / "native-gqa/quality" / variant / "audit.json").read_text()
        )
        assert audited["state"] == "complete"
    print("Matched native Qwen quality complete", counts(), flush=True)


if __name__ == "__main__":
    main()
