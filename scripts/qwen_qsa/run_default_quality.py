#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Complete the frozen default quality campaign, one bounded GPU stage at a time."""

import json
import subprocess
import sys
import time

from verify_handoff import ROOT, WORK


def counts():
    result = {}
    for task in ["ppl", "gsm8k"]:
        path = WORK / "quality/Original" / (task + ".jsonl")
        result[task] = (
            len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0
        )
    return result


def main():
    stages = [("ppl", start, 8) for start in range(0, 64, 8)]
    stages += [("gsm8k", start, 32) for start in range(0, 128, 32)]
    for task, start, count in stages:
        label = f"qwen-default-{task}-{start:02d}-{start + count:02d}"
        status = ROOT / "_local/qsa/stages" / label / "status.json"
        path = WORK / "quality/Original" / (task + ".jsonl")
        if status.exists():
            receipt = json.loads(status.read_text())
            assert receipt["state"] == "complete", (
                f"Retained failed stage {label}; investigate, do not automatically retry"
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
            "scripts/qwen_qsa/evaluate_default.py",
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
        assert process.returncode == 0, (
            f"Stage failed: {label}; partial evidence retained"
        )
        subprocess.run(
            [sys.executable, "scripts/qwen_qsa/report_results.py"], cwd=ROOT, check=True
        )
        print("Completed", label, counts(), flush=True)
        time.sleep(0.1)
    subprocess.run(
        [sys.executable, "scripts/qwen_qsa/report_results.py"], cwd=ROOT, check=True
    )
    audited = json.loads((WORK / "quality/Original/audit.json").read_text())
    assert audited["state"] == "complete"
    print("Fresh native default quality complete", json.dumps(audited), flush=True)


if __name__ == "__main__":
    main()
