#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Batch-one complete native Original versus integrated Changed phase timings."""
# isort: skip_file

import argparse
import json
import math
import time

from runtime import WORK, run
import numpy as np
import torch

from partitioned_inference import PartitionedOriginal
from original_inference import Original
from benchmark_original import prepared as prepare_original
from prepare_baseline import REFERENCE, SELECTOR_SHA, original_folder, sha


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context", type=int, choices=[4096, 16384, 32768, 65472], required=True
    )
    parser.add_argument("--pairs", type=int, default=6)
    args = parser.parse_args()
    assert args.pairs >= 4 and args.pairs % 2 == 0
    torch.set_num_threads(8)
    assert (
        json.loads((WORK / "changed-validation.json").read_text())["status"] == "passed"
    )
    context = args.context
    if context > 4096:
        assert (
            json.loads((WORK / "qsa-long-validation.json").read_text())["status"]
            == "passed"
        )
    folder = WORK / "paired-performance" / str(context)
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / "rows.jsonl").exists(), (
        "Preserve completed or partial timing trials"
    )
    model = PartitionedOriginal(context + 64)
    model.enable_qsa()
    original = Original(original_folder(), context + 64, caches=model.caches)
    source = REFERENCE / "data/long-timing-64k-v1/tokens.json"
    manifest = json.loads((source.parent / "manifest.json").read_text())
    assert sha(source) == manifest["tokens_sha256"]
    tokens = json.loads(source.read_text())
    prefix_ids = torch.tensor([tokens[:context]], device="cuda", dtype=torch.int64)
    decode_ids = [
        torch.tensor([[tokens[context + step]]], device="cuda", dtype=torch.int64)
        for step in range(64)
    ]
    prefill = {
        "Changed": model.prepare(prefix_ids),
        "Original": prepare_original(original, prefix_ids, 0),
    }
    decode = {
        "Changed": [
            model.prepare(ids, past=context + step)
            for step, ids in enumerate(decode_ids)
        ],
        "Original": [
            prepare_original(original, ids, context + step)
            for step, ids in enumerate(decode_ids)
        ],
    }

    def invoke(variant, plan):
        if variant == "Original":
            run(original.sess, plan["io"])
            return plan["logits"]
        model.variant = "Changed"
        return model.invoke(plan)

    sessions = [id(s) for s in model.sessions]
    original_session = id(original.sess)
    cache_pointers = [t.data_ptr() for t in model.caches.values()]
    protocol = {
        "batch": 1,
        "context": context,
        "capacity": context + 64,
        "decode_calls": 64,
        "pairs": args.pairs,
        "warmup_requests_per_variant": 2,
        "training_updates": 0,
        "original_execution": "complete native hip-ep graph",
        "changed_execution": "controlled attention boundary runner; integration/orchestration cost included",
        "projection_implementation": "identical packed weights and native MatMulNBits implementation; unchanged QKV/O/MLPs",
        "kv_storage": "same persistent buffers shared by the serial variants",
        "selector_sha256": SELECTOR_SHA,
        "selector_dtype": "BF16",
        "backbone_export_sha256": sha(original_folder() / "export.json"),
        "kernel_sha256": sha(WORK / "native/qsa_kernels_gfx1151.dll"),
        "token_stream_sha256": sha(source),
        "timer": "HIP-synchronized CPU wall time",
        "order": "adjacent Original/Changed pairs; alternate which variant goes first",
        "output_head": "included at the final position of every call",
        "prefill": "preallocated cache; excludes model load, tensor/binding creation",
        "decode": "64 fixed teacher-forced calls; excludes prefill, cache preparation, sampling, tensor/binding creation",
        "aggregate_and_per_request_rates": "equal because batch=1",
        "cache_payload_bytes": sum(
            t.numel() * t.element_size() for t in model.caches.values()
        ),
        "statistics": "ratio of paired mean elapsed times; paired bootstrap 10000 resamples, seed42, 95%; within-run uncertainty",
    }
    (folder / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    receipt = {"status": "running", "rows": 0}
    try:
        warmup_values = {v: [] for v in ["Original", "Changed"]}
        for repeat in range(2):
            for variant in ["Original", "Changed"]:
                model.variant = variant
                values = [invoke(variant, prefill[variant]).clone()]
                for plan in decode[variant]:
                    values.append(invoke(variant, plan).clone())
                result = torch.cat(values, dim=1)
                assert result.isfinite().all()
                warmup_values[variant].append(result)
                print("Exact-shape warmup", context, repeat + 1, variant, flush=True)
        # Actual native-batch prefill and sequential cache replay, not cache
        # replication or fabricated batched data. Preserve deterministic replay.
        validation = {}
        for variant in ["Original", "Changed"]:
            assert torch.equal(*warmup_values[variant])
            targets = torch.tensor(
                tokens[context : context + 64], device="cuda", dtype=torch.int64
            )
            continuation_nll = torch.nn.functional.cross_entropy(
                warmup_values[variant][0][:, :64].float().reshape(64, -1), targets
            ).item()
            validation[variant] = {
                "finite": True,
                "replay_bit_exact": True,
                "continuation_targets": 64,
                "continuation_nll": continuation_nll,
                "continuation_ppl": math.exp(continuation_nll),
                "continuation_scope": "single reused timing-stream continuation; diagnostic only; untimed warmup logits",
            }
        (folder / "model-validation.json").write_text(
            json.dumps(validation, indent=2) + "\n"
        )
        rows = []
        for pair in range(args.pairs):
            variants = (
                ["Original", "Changed"] if pair % 2 == 0 else ["Changed", "Original"]
            )
            for order, variant in enumerate(variants):
                model.variant = variant
                torch.cuda.synchronize()
                start = time.perf_counter()
                invoke(variant, prefill[variant])
                torch.cuda.synchronize()
                pref = time.perf_counter() - start
                torch.cuda.synchronize()
                start = time.perf_counter()
                for plan in decode[variant]:
                    invoke(variant, plan)
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
                }
                with (folder / "rows.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                rows.append(row)
                receipt["rows"] = len(rows)
                (folder / "status.json").write_text(
                    json.dumps(receipt, indent=2) + "\n"
                )
                print(json.dumps(row), flush=True)
        assert sessions == [id(s) for s in model.sessions]
        assert original_session == id(original.sess)
        assert cache_pointers == [t.data_ptr() for t in model.caches.values()]
        assert cache_pointers == [t.data_ptr() for t in original.caches.values()]
        summary = {"context": context, "batch": 1, "pairs": args.pairs, "phases": {}}
        for phase, work in [("prefill", context), ("decode", 64)]:
            elapsed = {
                v: np.array(
                    [
                        next(
                            r[phase + "_seconds"]
                            for r in rows
                            if r["variant"] == v and r["pair"] == p
                        )
                        for p in range(args.pairs)
                    ]
                )
                for v in ["Original", "Changed"]
            }
            rng = np.random.default_rng(42)
            samples = rng.integers(0, args.pairs, size=(10000, args.pairs))
            changes = 100 * (
                elapsed["Original"][samples].mean(1)
                / elapsed["Changed"][samples].mean(1)
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
                paired_95_interval_percent=np.quantile(
                    changes, [0.025, 0.975]
                ).tolist(),
            )
        (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        receipt.update(
            status="complete",
            selector_integrity=model.bank.integrity(),
            shared_sessions_unchanged=True,
            native_original_session_unchanged=True,
            cache_pointers_unchanged=True,
        )
    except BaseException as exc:
        receipt.update(status="failed", error=repr(exc))
        raise
    finally:
        (folder / "status.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
