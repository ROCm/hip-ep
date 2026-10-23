#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""External WDDM sampler; observes only this campaign's current process."""

from pathlib import Path
import argparse
import json
import time
from windows_gpu_memory import GpuMemory

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "results/profile").mkdir(parents=True, exist_ok=True)
D = ROOT / "results/profile"
p = argparse.ArgumentParser()
p.add_argument("--variant")
p.add_argument("--length", type=int)
args = p.parse_args()
pid = None
gpu = None
records = []
errors = []
lastsave = 0
started = time.time()
output = D / (
    "wddm-" + args.variant + "-" + str(args.length) if args.variant else "wddm-campaign"
)


def save():
    peaks = {}
    for r in records:
        key = f"{r['variant']}/{r['context']}/{r['phase']}"
        if key not in peaks or r["committed"] > peaks[key]["committed"]:
            peaks[key] = r
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "started": started,
                "sample_count": len(records),
                "peaks": peaks,
                "errors": errors,
            },
            indent=2,
        )
        + "\n"
    )


with output.with_suffix(".jsonl").open("w") as log:
    while True:
        try:
            if args.variant:
                active = {"variant": args.variant, "length": args.length}
            else:
                campaign = json.loads((D / "campaign.json").read_text())
                if campaign["status"] == "complete":
                    break
                active = campaign.get("active")
                if not active:
                    time.sleep(0.05)
                    continue
            variant = active["variant"].removeprefix("synthetic-")
            n = active["length"]
            receipt = (
                D
                / variant
                / (str(n) + "-synthetic" if n == 262144 else str(n))
                / "receipt.json"
            )
            if not receipt.exists():
                time.sleep(0.05)
                continue
            r = json.loads(receipt.read_text())
            if args.variant and r["status"] in ("passed", "failed") and records:
                break
            if r["status"] in ("passed", "failed"):
                time.sleep(0.05)
                continue
            if pid != r["pid"]:
                pid = r["pid"]
                gpu = GpuMemory(pid)
            gpu.phase = (
                str(r.get("active_repetition", 0))
                + ":"
                + r.get("active_phase", r["status"])
            )
            row = gpu.collect()
            row.update(variant=variant, context=n, pid=pid, wall_time=time.time())
            records.append(row)
            log.write(json.dumps(row) + "\n")
            # Keep the sampler's internal buffer bounded; records owns the evidence.
            gpu.samples.clear()
            if time.time() - lastsave > 5:
                save()
                log.flush()
                lastsave = time.time()
        except (OSError, ValueError, KeyError, AssertionError) as exc:
            # The instance may not exist while the process is creating its GPU context.
            message = repr(exc)
            if not errors or errors[-1]["error"] != message:
                errors.append({"time": time.time(), "error": message})
        time.sleep(0.02)
save()
print("WDDM samples", len(records), flush=True)
