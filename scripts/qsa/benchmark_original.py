#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Initial Original baseline, separately timed prefill and teacher-forced decode."""
# isort: skip_file

import argparse
import json
import statistics
import time

from runtime import WORK, binding, run
import torch

from original_inference import Original
from prepare_baseline import REFERENCE, original_folder, sha


def prepared(model, ids, past):
    """All inputs, outputs and bindings are prepared outside measured intervals."""
    count = ids.shape[1]
    lengths = torch.full(
        (model.batch,), past + count - 1, dtype=torch.int32, device="cuda"
    )
    positions = torch.tensor([count - 1], device="cuda", dtype=torch.int64)
    logits = torch.empty((model.batch, 1, 100278), dtype=torch.float16, device="cuda")
    inputs = {
        **model.caches,
        "input_ids": ids,
        "position_ids": torch.arange(
            past, past + count, device="cuda", dtype=torch.int64
        )[None]
        .expand(model.batch, -1)
        .contiguous(),
        "seqlens_k": lengths,
        "capacity": model.cap,
        "logits_positions": positions,
    }
    outputs = {
        "logits": logits,
        **{
            name.replace("past_", "present_", 1): value
            for name, value in model.caches.items()
        },
    }
    return {
        "io": binding(model.sess, inputs, outputs),
        "owners": [*inputs.values(), logits],
        "logits": logits,
    }


def invoke(model, call):
    run(model.sess, call["io"])


@torch.inference_mode()
def benchmark(context, batch, trials):
    original = original_folder()
    assert json.loads((original / "validation.json").read_text())["status"] == "passed"
    model = Original(original, context + 64, batch=batch)
    source = REFERENCE / "data/long-timing-64k-v1/tokens.json"
    tokens = json.loads(source.read_text())
    ids = torch.tensor([tokens[:context]], device="cuda", dtype=torch.int64).repeat(
        batch, 1
    )
    prefill = prepared(model, ids, 0)
    decode = [
        prepared(
            model,
            torch.tensor(
                [[tokens[context + step]]], dtype=torch.int64, device="cuda"
            ).repeat(batch, 1),
            context + step,
        )
        for step in range(64)
    ]
    folder = WORK / "baseline-w4-rope-v2" / f"timing-{context}-b{batch}"
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / "rows.jsonl").exists(), (
        "Preserve existing timing trials; choose a new run directory"
    )
    protocol = {
        "variant": "Original",
        "context": context,
        "batch": batch,
        "decode_calls": 64,
        "warmups": 2,
        "trials": trials,
        "input_sha256": sha(source),
        "timer": "HIP-synchronized CPU wall time",
        "output_head": "included, final position each call",
        "decode_excludes": [
            "prefill",
            "cache preparation",
            "sampling",
            "input/output creation",
            "binding creation",
        ],
        "prefill_excludes": [
            "model load",
            "preallocated cache creation",
            "input/output creation",
            "binding creation",
        ],
        "precision": "RTN W4A16 group128; DP4A disabled",
        "training_updates": 0,
    }
    (folder / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for index in range(2 + trials):
        start = time.perf_counter()
        invoke(model, prefill)
        pref = time.perf_counter() - start
        start = time.perf_counter()
        for call in decode:
            invoke(model, call)
        dec = time.perf_counter() - start
        if index < 2:
            print("Exact-shape warmup", index + 1, flush=True)
            continue
        row = {
            "variant": "Original",
            "trial": index - 2,
            "context": context,
            "batch": batch,
            "prefill_seconds": pref,
            "decode_seconds": dec,
            "decode_calls": 64,
            "prefill_aggregate_tokens_s": batch * context / pref,
            "decode_aggregate_tokens_s": batch * 64 / dec,
            "decode_per_request_tokens_s": 64 / dec,
        }
        with (folder / "rows.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    rows = [
        json.loads(line) for line in (folder / "rows.jsonl").read_text().splitlines()
    ]
    summary = {
        "status": "complete",
        "variant": "Original",
        "context": context,
        "batch": batch,
        "trials": len(rows),
        **{
            key: statistics.median(r[key] for r in rows)
            for key in [
                "prefill_seconds",
                "decode_seconds",
                "prefill_aggregate_tokens_s",
                "decode_aggregate_tokens_s",
                "decode_per_request_tokens_s",
            ]
        },
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--batch", type=int, choices=[1], default=1)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(8)
    benchmark(args.context, args.batch, args.trials)
