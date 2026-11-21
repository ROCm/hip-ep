#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Time locked teacher-forced Qwen prefill/decode with balanced native pairs."""
# ruff: noqa: E402

import argparse
import json
import os
import time
from pathlib import Path

os.environ.pop("QSA_ORT_PROFILE_DIR", None)
from native_inference import Native
from runtime import binding, run
from verify_handoff import WORK, sha

import numpy as np
import torch


def prepared(model, ids, past):
    B, S = ids.shape
    mask = torch.zeros(B, model.capacity, device="cuda", dtype=torch.int64)
    mask[:, : past + S] = 1
    positions = (
        torch.arange(past, past + S, device="cuda", dtype=torch.int64)[None, None]
        .expand(3, B, -1)
        .contiguous()
    )
    selected = torch.tensor([S - 1], device="cuda", dtype=torch.int64)
    logits = torch.empty(B, 1, 248320, device="cuda", dtype=torch.float16)
    inputs = {
        **model.caches,
        "input_ids": ids,
        "attention_mask": mask,
        "position_ids": positions,
        "logits_positions": selected,
    }
    outputs = {
        "logits": logits,
        **{
            name.replace("past_key_values.", "present."): value
            for name, value in model.caches.items()
        },
    }
    return {
        "io": binding(model.sess, inputs, outputs),
        "logits": logits,
        "owners": [*inputs.values(), logits],
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context", type=int, choices=[4096, 16384, 32768, 65536], required=True
    )
    parser.add_argument("--batch", type=int, choices=[1, 4, 8], default=1)
    args = parser.parse_args()
    torch.set_num_threads(2)
    task = WORK / "native-gqa"
    validation = json.loads((task / "full-model-validation.json").read_text())
    assert validation["status"] == "passed"
    assert (
        json.loads((task / "operator-validation.json").read_text())["status"]
        == "passed"
    )
    for name, expected in validation["binary_sha256"].items():
        assert sha(task / "install/bin" / name) == expected
    data = WORK / "timing-v1"
    manifest = json.loads((data / "manifest.json").read_text())
    assert (
        manifest["state"] == "locked"
        and sha(data / "streams.json") == manifest["streams_sha256"]
    )
    streams = json.loads((data / "streams.json").read_text())
    assert len(streams) == 8
    # The locked stream file contains token lists, or rows with input_ids.
    tokens = [
        row["input_ids"] if isinstance(row, dict) else row
        for row in streams[: args.batch]
    ]
    assert all(len(row) == 65600 for row in tokens)
    ids = torch.tensor(tokens, device="cuda", dtype=torch.int64)
    context, B = args.context, args.batch
    folder = task / "paired-performance" / f"{context}-b{B}"
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / "rows.jsonl").exists(), "Preserve prior trials"
    original = Native(task / "models/Original", context + 64, batch=B)
    changed = Native(
        task / "models/Changed", context + 64, batch=B, caches=original.caches
    )
    models = {"Original": original, "Changed": changed}
    pointers = [value.data_ptr() for value in original.caches.values()]
    sessions = {name: id(model.sess) for name, model in models.items()}
    plans = {
        name: [
            prepared(model, ids[:, :context].contiguous(), 0),
            *[
                prepared(
                    model,
                    ids[:, context + step : context + step + 1].contiguous(),
                    context + step,
                )
                for step in range(64)
            ],
        ]
        for name, model in models.items()
    }
    protocol = {
        "context": context,
        "batch": B,
        "capacity": context + 64,
        "warmup_pairs": 2,
        "measured_pairs": 6,
        "decode_calls": 64,
        "timing_manifest_sha256": sha(data / "manifest.json"),
        "stream_sha256": manifest["streams_sha256"],
        "binary_sha256": validation["binary_sha256"],
        "benchmark_sha256": sha(Path(__file__)),
        "inference_sha256": sha(Path(__file__).with_name("native_inference.py")),
        "timer": "Independent HIP-synchronized wall phase timers",
        "prefill": "Full session with final-position head; allocation/binding and recurrent-state zeroing outside timer",
        "decode": "64 common teacher-forced full-session calls; prefill, sampling and allocation/binding outside timer",
        "cache": "Identical resident cache buffers shared serially; native selector state owned separately",
        "statistics": "Paired mean-time ratio and 10000 paired bootstrap resamples seed42; within-run uncertainty",
        "training_updates": 0,
    }
    (folder / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    rows, warmups = [], {}
    for pair in range(-2, 6):
        variants = ["Original", "Changed"] if pair % 2 == 0 else ["Changed", "Original"]
        for order, variant in enumerate(variants):
            model, plan = models[variant], plans[variant]
            for name, value in model.caches.items():
                if name.endswith((".conv_state", ".recurrent_state")):
                    value.zero_()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            run(model.sess, plan[0]["io"])
            prefill = time.perf_counter() - begin
            begin = time.perf_counter()
            for call in plan[1:]:
                run(model.sess, call["io"])
            decode = time.perf_counter() - begin
            output = torch.cat([call["logits"].clone() for call in plan], 1)
            assert output.isfinite().all()
            if pair < 0:
                if variant in warmups:
                    assert torch.equal(output, warmups[variant])
                warmups[variant] = output
                print("warmup", pair, variant, flush=True)
                continue
            row = {
                "pair": pair,
                "order": order,
                "variant": variant,
                "context": context,
                "batch": B,
                "prefill_seconds": prefill,
                "decode_seconds": decode,
                "prefill_tokens_s": B * context / prefill,
                "decode_tokens_s": B * 64 / decode,
            }
            rows.append(row)
            with (folder / "rows.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(row, flush=True)
    summary = {"context": context, "batch": B, "pairs": 6, "phases": {}}
    samples = np.random.default_rng(42).integers(0, 6, size=(10000, 6))
    for phase, count in [("prefill", B * context), ("decode", B * 64)]:
        elapsed = {
            name: np.array(
                [
                    next(
                        row[phase + "_seconds"]
                        for row in rows
                        if row["variant"] == name and row["pair"] == pair
                    )
                    for pair in range(6)
                ]
            )
            for name in models
        }
        changes = 100 * (
            elapsed["Original"][samples].mean(1) / elapsed["Changed"][samples].mean(1)
            - 1
        )
        summary["phases"][phase] = {
            name: {
                "mean_seconds": float(values.mean()),
                "tokens_s": float(count / values.mean()),
            }
            for name, values in elapsed.items()
        }
        summary["phases"][phase].update(
            throughput_change_percent=float(
                100 * (elapsed["Original"].mean() / elapsed["Changed"].mean() - 1)
            ),
            paired_95_interval_percent=np.quantile(changes, [0.025, 0.975]).tolist(),
        )
    assert sessions == {name: id(model.sess) for name, model in models.items()}
    assert pointers == [value.data_ptr() for value in changed.caches.values()]
    summary.update(
        status="complete",
        replay_exact=True,
        sessions_unchanged=True,
        cache_pointers_unchanged=True,
        protocol_sha256=sha(folder / "protocol.json"),
        rows_sha256=sha(folder / "rows.jsonl"),
    )
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
