#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Memory-bounded paired timing with one variant resident in each worker."""
# isort: skip_file

import argparse
import json
import math
import subprocess
import sys
import time

from prepare_baseline import REFERENCE, ROOT, SELECTOR_SHA, WORK, original_folder, sha


def worker(variant, trials, context, folder):
    # The coordinator never initializes a GPU runtime.
    from runtime import run

    import torch

    from benchmark_original import prepared
    from original_inference import Original
    from partitioned_inference import PartitionedOriginal

    torch.set_num_threads(8)
    with torch.inference_mode():
        if variant == "Original":
            model = Original(original_folder(), context + 64)
            sessions = [model.sess]

            def prepare(ids, past):
                return prepared(model, ids, past)

            def invoke(plan):
                run(model.sess, plan["io"])
                return plan["logits"]

        else:
            model = PartitionedOriginal(context + 64)
            model.enable_qsa()
            model.variant = "Changed"
            sessions = model.sessions
            prepare, invoke = model.prepare, model.invoke
        session_ids = [id(s) for s in sessions]
        pointers = [t.data_ptr() for t in model.caches.values()]
        source = REFERENCE / "data/long-timing-64k-v1/tokens.json"
        assert (
            sha(source)
            == json.loads((source.parent / "manifest.json").read_text())[
                "tokens_sha256"
            ]
        )
        tokens = json.loads(source.read_text())
        prefill = prepare(
            torch.tensor([tokens[:context]], dtype=torch.int64, device="cuda"), 0
        )
        decode = [
            prepare(
                torch.tensor([[tokens[context + i]]], dtype=torch.int64, device="cuda"),
                context + i,
            )
            for i in range(64)
        ]
        values = []
        for repeat in range(2):
            outputs = [invoke(prefill).clone()]
            outputs.extend(invoke(plan).clone() for plan in decode)
            value = torch.cat(outputs, dim=1)
            assert value.isfinite().all()
            values.append(value)
            print("Isolated warmup", variant, trials, repeat + 1, flush=True)
        assert torch.equal(*values)
        targets = torch.tensor(tokens[context : context + 64], device="cuda")
        nll = torch.nn.functional.cross_entropy(
            values[0][:, :64].float().reshape(64, -1), targets
        ).item()
        validation = {
            "finite": True,
            "replay_bit_exact": True,
            "continuation_targets": 64,
            "continuation_nll": nll,
            "continuation_ppl": math.exp(nll),
            "continuation_scope": "single reused timing-stream continuation; diagnostic only; untimed warmup logits",
        }
        for pair, order in trials:
            torch.cuda.synchronize()
            start = time.perf_counter()
            invoke(prefill)
            torch.cuda.synchronize()
            pref = time.perf_counter() - start
            start = time.perf_counter()
            for plan in decode:
                invoke(plan)
            torch.cuda.synchronize()
            dec = time.perf_counter() - start
            row = {
                "variant": variant,
                "pair": pair,
                "order": order,
                "context": context,
                "batch": 1,
                "prefill_seconds": pref,
                "decode_seconds": dec,
                "decode_calls": 64,
                "prefill_tokens_s": context / pref,
                "decode_aggregate_tokens_s": 64 / dec,
                "decode_per_request_tokens_s": 64 / dec,
                "isolated_worker": f"{variant}-{trials[0][0]}",
            }
            with (folder / "rows.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        assert session_ids == [id(s) for s in sessions]
        assert pointers == [t.data_ptr() for t in model.caches.values()]
        receipt = {
            "variant": variant,
            "trials": trials,
            "sessions_unchanged": True,
            "cache_pointers_unchanged": True,
            "validation": validation,
        }
        if variant == "Changed":
            receipt["selector_integrity"] = model.bank.integrity()
        (folder / f"worker-{variant}-{trials[0][0]}.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=int, default=65472)
    parser.add_argument("--worker", choices=["Original", "Changed"])
    parser.add_argument("--trials")
    args = parser.parse_args()
    folder = WORK / "paired-performance" / str(args.context)
    if args.worker:
        worker(args.worker, json.loads(args.trials), args.context, folder)
        return
    import numpy as np

    assert args.context == 65472
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / "protocol.json").exists(), "Preserve previous evidence"
    # Adjacent equal variants are coalesced, preserving chronological paired
    # order C/O, O/C, C/O, O/C, C/O, O/C with three first requests each.
    schedule = []
    for pair in range(6):
        variants = ["Changed", "Original"] if pair % 2 == 0 else ["Original", "Changed"]
        for order, variant in enumerate(variants):
            if schedule and schedule[-1][0] == variant:
                schedule[-1][1].append([pair, order])
            else:
                schedule.append((variant, [[pair, order]]))
    source = REFERENCE / "data/long-timing-64k-v1/tokens.json"
    protocol = {
        "batch": 1,
        "context": args.context,
        "capacity": args.context + 64,
        "pairs": 6,
        "decode_calls": 64,
        "training_updates": 0,
        "original_execution": "complete native hip-ep graph",
        "changed_execution": "controlled attention boundary runner; integration/orchestration cost included",
        "process_isolation": True,
        "worker_schedule": schedule,
        "warmup_requests_per_worker": 2,
        "kv_storage": "identical layout/capacity; persistent buffers within each isolated worker",
        "projection_implementation": "identical packed weights and native MatMulNBits implementation; unchanged QKV/O/MLPs",
        "selector_sha256": SELECTOR_SHA,
        "selector_dtype": "BF16",
        "backbone_export_sha256": sha(original_folder() / "export.json"),
        "kernel_sha256": sha(WORK / "native/qsa_kernels_gfx1151.dll"),
        "token_stream_sha256": sha(source),
        "timer": "HIP-synchronized CPU wall time",
        "prefill": "preallocated cache; model load, tensor/binding creation excluded",
        "decode": "64 fixed calls; prefill/cache preparation/sampling/tensor and binding creation excluded",
        "output_head": "included at final position of every call",
        "statistics": "paired mean-time ratio; bootstrap 10000 resamples seed42; includes worker variation",
    }
    (folder / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for variant, trials in schedule:
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(ROOT / "scripts/qsa/benchmark_isolated.py"),
                "--context",
                str(args.context),
                "--worker",
                variant,
                "--trials",
                json.dumps(trials),
            ],
            cwd=ROOT,
            check=True,
        )
    raw = [
        json.loads(line) for line in (folder / "rows.jsonl").read_text().splitlines()
    ]
    assert len(raw) == 12
    summary = {"context": args.context, "batch": 1, "pairs": 6, "phases": {}}
    for phase, work in [("prefill", args.context), ("decode", 64)]:
        elapsed = {
            v: np.array(
                [
                    next(
                        r[phase + "_seconds"]
                        for r in raw
                        if r["variant"] == v and r["pair"] == p
                    )
                    for p in range(6)
                ]
            )
            for v in ["Original", "Changed"]
        }
        samples = np.random.default_rng(42).integers(0, 6, size=(10000, 6))
        changes = 100 * (
            elapsed["Original"][samples].mean(1) / elapsed["Changed"][samples].mean(1)
            - 1
        )
        summary["phases"][phase] = {
            v: {
                "mean_seconds": float(t.mean()),
                "aggregate_tokens_s": float(work / t.mean()),
                "per_request_tokens_s": float(work / t.mean()),
            }
            for v, t in elapsed.items()
        }
        summary["phases"][phase].update(
            throughput_change_percent=float(
                100 * (elapsed["Original"].mean() / elapsed["Changed"].mean() - 1)
            ),
            paired_95_interval_percent=np.quantile(changes, [0.025, 0.975]).tolist(),
        )
    receipts = [json.loads(p.read_text()) for p in sorted(folder.glob("worker-*.json"))]
    assert len(receipts) == len(schedule)
    validation = {}
    for v in ["Original", "Changed"]:
        values = [r["validation"] for r in receipts if r["variant"] == v]
        assert all(x == values[0] for x in values)
        validation[v] = values[0]
    status = {
        "status": "complete",
        "rows": 12,
        "selector_integrity": next(
            r["selector_integrity"] for r in receipts if r["variant"] == "Changed"
        ),
        "cache_pointers_unchanged": all(
            r["cache_pointers_unchanged"] for r in receipts
        ),
        "isolated_workers_sessions_unchanged": all(
            r["sessions_unchanged"] for r in receipts
        ),
        "native_original_session_unchanged": all(
            r["sessions_unchanged"] for r in receipts if r["variant"] == "Original"
        ),
    }
    for name, value in [
        ("summary", summary),
        ("model-validation", validation),
        ("status", status),
    ]:
        (folder / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    main()
