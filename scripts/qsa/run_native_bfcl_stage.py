#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Run one native GPU worker and one official BFCL CPU client serially."""

import argparse
import json
import subprocess
import sys
import time

from prepare_baseline import ROOT, SELECTOR_SHA, WORK, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--category", default="simple_python")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--replay-label", action="append", default=[])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--cpu-python", required=True)
    args = parser.parse_args()
    assert args.label.replace("-", "").replace("_", "").isalnum()
    task = WORK / "native-gqa"
    audit = json.loads((task / "final-audit.json").read_text())
    for name, expected in audit["binary_sha256"].items():
        assert sha(task / "install/bin" / name) == expected, name
    queue = task / "bfcl-v4/queues" / args.label
    queue.mkdir(parents=True, exist_ok=False)
    linux_root = "/mnt/" + ROOT.drive[0].lower() + ROOT.as_posix()[2:]
    relative = queue.relative_to(ROOT).as_posix()
    worker_log = (queue / "worker.log").open("w", encoding="utf-8")
    client_log = (queue / "client.log").open("w", encoding="utf-8")
    client_error_log = (queue / "client.stderr.log").open("w", encoding="utf-8")
    print("BFCL worker/client logs:", queue, flush=True)
    worker = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "scripts/qsa/native_generation_worker.py",
            "--queue",
            str(queue),
        ],
        cwd=ROOT,
        stdout=worker_log,
        stderr=subprocess.STDOUT,
    )
    client = None
    try:
        deadline = time.monotonic() + 600
        while not (queue / "ready.json").exists():
            assert worker.poll() is None, (
                "Native generation worker failed to initialize"
            )
            assert time.monotonic() < deadline, "Native worker initialization timed out"
            time.sleep(0.2)
        client = subprocess.Popen(
            [
                "wsl",
                "-d",
                "Ubuntu",
                "--",
                args.cpu_python,
                "-u",
                "-X",
                "faulthandler",
                linux_root + "/scripts/qsa/bfcl_native_eval.py",
                "--queue",
                linux_root + "/" + relative,
                "--category",
                args.category,
                "--start",
                str(args.start),
                "--count",
                str(args.count),
            ]
            + (["--pilot"] if args.pilot else [])
            + [
                value
                for label in args.replay_label
                for value in [
                    "--replay-queue",
                    linux_root + "/_local/qsa/native-gqa/bfcl-v4/queues/" + label,
                ]
            ],
            cwd=ROOT,
            stdout=client_log,
            stderr=client_error_log,
        )
        while client.poll() is None:
            if worker.poll() is not None:
                assert worker.returncode == 0 and (queue / "stop.json").exists(), (
                    "Native worker failed"
                )
            time.sleep(0.2)
        assert client.returncode == 0, (
            f"Official BFCL client exit {client.returncode}; preserve queue and logs"
        )
        assert worker.wait(timeout=60) == 0
        stopped = json.loads((queue / "stopped.json").read_text())
        assert stopped["selector_sha256"] == SELECTOR_SHA
    finally:
        if client is not None and client.poll() is None:
            client.terminate()
            client.wait(timeout=30)
        if worker.poll() is None:
            worker.terminate()
            worker.wait(timeout=30)
        worker_log.close()
        client_log.close()
        client_error_log.close()


if __name__ == "__main__":
    main()
