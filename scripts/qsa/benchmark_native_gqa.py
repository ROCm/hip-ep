#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Matched full-graph native GQA/QSA timings or the retained paired Dolma gate."""
# isort: skip_file
# Select the native EP and DLL paths before importing the GPU runtime.
# ruff: noqa: E402

import argparse
import json
import os
import time
from pathlib import Path

TASK = Path(__file__).resolve().parents[2] / "_local/qsa/native-gqa"
os.environ["QSA_HIP_EP_INSTALL"] = str(TASK / "install")
os.environ["QSA_MODEL_CACHE"] = str(TASK / "model-cache")
os.environ.pop("QSA_ORT_PROFILE_DIR", None)

from runtime import run
import numpy as np
import torch

from benchmark_original import prepared
from original_inference import Original
from prepare_baseline import selector_fingerprint
from prepare_baseline import REFERENCE, SELECTOR_SHA, sha


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context", type=int, choices=[4096, 16384, 32768, 65472], default=4096
    )
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--dolma", action="store_true")
    args = parser.parse_args()
    assert args.pairs >= 4 and args.pairs % 2 == 0
    assert json.loads((TASK / "validation.json").read_text())["status"] == "passed"
    torch.set_num_threads(8)
    context = args.context
    original = Original(TASK / "models/Original", context + 64)
    changed = Original(TASK / "models/Changed", context + 64, caches=original.caches)
    models = {"Original": original, "Changed": changed}
    sessions = {k: id(v.sess) for k, v in models.items()}
    pointers = [v.data_ptr() for v in original.caches.values()]
    if args.dolma:
        assert context == 4096
        from evaluate_pair import dolma

        class Pair:
            variant = "Original"

            def forward(self, *a, **kw):
                return models[self.variant].forward(*a, **kw)

        folder = TASK / "paired-quality"
        folder.mkdir(exist_ok=True)
        dolma(Pair(), folder, 64, validate_baseline=False)
        assert selector_fingerprint() == SELECTOR_SHA
        (folder / "integrity.json").write_text(
            json.dumps(
                {
                    "selector_sha256": SELECTOR_SHA,
                    "selector_dtype": "BF16",
                    "training_updates": 0,
                    "execution": "one native session/forward each; identical packed backbone projections",
                },
                indent=2,
            )
            + "\n"
        )
        return
    folder = TASK / "paired-performance" / str(context)
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / "rows.jsonl").exists(), "Preserve prior trials"
    source = REFERENCE / "data/long-timing-64k-v1/tokens.json"
    assert (
        sha(source)
        == json.loads((source.parent / "manifest.json").read_text())["tokens_sha256"]
    )
    tokens = json.loads(source.read_text())
    prefix = torch.tensor([tokens[:context]], dtype=torch.int64, device="cuda")
    ids = [
        torch.tensor([[tokens[context + i]]], dtype=torch.int64, device="cuda")
        for i in range(64)
    ]
    plans = {
        v: (
            prepared(m, prefix, 0),
            [prepared(m, x, context + i) for i, x in enumerate(ids)],
        )
        for v, m in models.items()
    }

    def invoke(variant, plan):
        run(models[variant].sess, plan["io"])
        return plan["logits"]

    protocol = {
        "batch": 1,
        "context": context,
        "capacity": context + 64,
        "pairs": args.pairs,
        "decode_calls": 64,
        "warmup_requests_per_variant": 2,
        "training_updates": 0,
        "original_execution": "one complete native ONNX graph/session",
        "changed_execution": "one complete native ONNX graph/session; four native QSA GQA nodes",
        "ep_dispatches_per_forward": 1,
        "kv_storage": "same persistent buffers shared serially",
        "selector_sha256": SELECTOR_SHA,
        "selector_dtype": "BF16",
        "token_stream_sha256": sha(source),
        "timer": "HIP-synchronized CPU wall time",
        "prefill": "model load and tensor/binding allocation excluded; final-position output head included",
        "decode": "64 fixed teacher-forced calls; prefill/cache preparation/sampling/tensor and binding allocation excluded",
        "statistics": "paired mean-time ratio; 10000 paired bootstrap resamples seed42; within-run uncertainty",
    }
    (folder / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for v in models:
        values = []
        for repeat in range(2):
            prefill, decode = plans[v]
            result = torch.cat(
                [invoke(v, prefill).clone(), *[invoke(v, d).clone() for d in decode]],
                dim=1,
            )
            assert result.isfinite().all()
            values.append(result)
            print("Native warmup", context, v, repeat + 1, flush=True)
        assert torch.equal(*values)
    rows = []
    for pair in range(args.pairs):
        variants = ["Original", "Changed"] if pair % 2 == 0 else ["Changed", "Original"]
        for order, v in enumerate(variants):
            prefill, decode = plans[v]
            torch.cuda.synchronize()
            start = time.perf_counter()
            invoke(v, prefill)
            torch.cuda.synchronize()
            pref = time.perf_counter() - start
            start = time.perf_counter()
            for d in decode:
                invoke(v, d)
            torch.cuda.synchronize()
            dec = time.perf_counter() - start
            row = {
                "variant": v,
                "pair": pair,
                "order": order,
                "batch": 1,
                "context": context,
                "prefill_seconds": pref,
                "decode_seconds": dec,
                "decode_calls": 64,
                "prefill_tokens_s": context / pref,
                "decode_tokens_s": 64 / dec,
            }
            with (folder / "rows.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            rows.append(row)
            print(json.dumps(row), flush=True)
    assert sessions == {k: id(v.sess) for k, v in models.items()}
    assert pointers == [v.data_ptr() for v in changed.caches.values()]
    assert selector_fingerprint() == SELECTOR_SHA
    summary = {"batch": 1, "context": context, "pairs": args.pairs, "phases": {}}
    for phase, count in [("prefill", context), ("decode", 64)]:
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
            for v in models
        }
        samples = np.random.default_rng(42).integers(
            0, args.pairs, size=(10000, args.pairs)
        )
        changes = 100 * (
            elapsed["Original"][samples].mean(1) / elapsed["Changed"][samples].mean(1)
            - 1
        )
        summary["phases"][phase] = {
            v: {"tokens_s": float(count / t.mean()), "mean_seconds": float(t.mean())}
            for v, t in elapsed.items()
        }
        summary["phases"][phase].update(
            throughput_change_percent=float(
                100 * (elapsed["Original"].mean() / elapsed["Changed"].mean() - 1)
            ),
            paired_95_interval_percent=np.quantile(changes, [0.025, 0.975]).tolist(),
        )
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (folder / "status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "rows": len(rows),
                "sessions_unchanged": True,
                "cache_pointers_unchanged": True,
                "replay_exact": True,
                "selector_sha256": SELECTOR_SHA,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
