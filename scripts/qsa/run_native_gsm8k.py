#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Complete the native GSM8K pilot in serial, resumable 32-pair stages."""

import json
import subprocess
import sys

from prepare_baseline import ROOT, SELECTOR_SHA, WORK


def main():
    task = WORK / "native-gqa"
    audit = json.loads((task / "final-audit.json").read_text())
    from prepare_baseline import sha

    for name, expected in audit["binary_sha256"].items():
        assert sha(task / "install/bin" / name) == expected, name
    for limit in [32, 64, 96, 128]:
        label = f"native-gqa-gsm8k-{limit}"
        command = ["scripts/qsa/evaluate_native_gsm8k.py", "--limit", str(limit)]
        path = WORK / "stages" / label / "status.json"
        if path.exists():
            previous = json.loads(path.read_text())
            assert previous["state"] == "complete", "Investigate the prior stage"
            assert previous["command"][2:] == command
            assert previous["selector_sha256_after"] == SELECTOR_SHA
            continue
        subprocess.run(
            [sys.executable, "-u", "scripts/qsa/run_stage.py", label, "--", *command],
            cwd=ROOT,
            check=True,
        )
    summary = json.loads((task / "gsm8k/gsm8k-summary.json").read_text())
    assert summary["complete"]
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
