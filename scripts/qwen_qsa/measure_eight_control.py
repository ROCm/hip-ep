#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Measure a bounded fresh default control when historical machine drift is suspected."""
# ruff: noqa: E402

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from measure_eight import TASK, identity
from benchmark_native import prepared
from export_eight import save
from native_inference import Native
from runtime import run
from verify_handoff import WORK, sha

import torch


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context", type=int, choices=[4096, 16384, 32768], required=True
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--measured-requests", type=int, choices=range(1, 7), default=6)
    args = parser.parse_args()
    os.environ.pop("QSA_ORT_PROFILE_DIR", None)
    torch.set_num_threads(2)
    context = args.context
    folder = (args.output_root or TASK / "fresh-original") / str(context)
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / "protocol.json").exists()
    row = json.loads((WORK / "timing-v1/streams.json").read_text())[0]
    tokens = row["input_ids"] if isinstance(row, dict) else row
    protocol = dict(
        **identity(),
        variant="Original",
        context=context,
        batch=1,
        capacity=context + 64,
        warmups=2,
        measured_requests=args.measured_requests,
        decode_calls=64,
        reason="Quantify machine drift suggested by new eight-selector timing and default validation profile",
        comparison="Serial across-run control; no paired interval",
        measurement_sha256_control=sha(Path(__file__)),
        stream_sha256=sha(WORK / "timing-v1/streams.json"),
    )
    original_export = json.loads(
        (WORK / "native-gqa/models/Original/export.json").read_text()
    )
    protocol.update(
        phase_policy="native default full attention in prefill and decode; no QSA nodes",
        graph_sha256=original_export["onnx_sha256"],
        selector_layers=[],
        selector_export_sha256=None,
        export_receipt_sha256=sha(WORK / "native-gqa/models/Original/export.json"),
        reference_eight_export_sha256=identity()["export_receipt_sha256"],
    )
    save(folder / "protocol.json", protocol)
    begin = time.perf_counter()
    model = Native(WORK / "native-gqa/models/Original", context + 64)
    load_seconds = time.perf_counter() - begin
    session_id = id(model.sess)
    pointers = [t.data_ptr() for t in model.caches.values()]
    ids = torch.tensor([tokens[: context + 64]], device="cuda", dtype=torch.int64)
    plan = [prepared(model, ids[:, :context].contiguous(), 0)]
    plan.extend(
        prepared(model, ids[:, context + i : context + i + 1].contiguous(), context + i)
        for i in range(64)
    )
    rows = []
    reference = None
    for trial in range(-2, args.measured_requests):
        if args.stop_file and args.stop_file.exists():
            save(
                folder / "stopped.json",
                dict(state="stopped_at_request_boundary", next_trial=trial),
            )
            return
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
        if reference is None:
            reference = output
        else:
            assert torch.equal(reference, output)
        result = dict(
            trial=trial,
            warmup=trial < 0,
            variant="Original",
            context=context,
            batch=1,
            prefill_seconds=prefill,
            decode_seconds=decode,
            prefill_tokens_s=context / prefill,
            decode_tokens_s=64 / decode,
            finite=True,
            replay_exact=True,
        )
        with (folder / "rows.jsonl").open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        rows.append(result)
        print(result, flush=True)
    assert pointers == [
        t.data_ptr() for t in model.caches.values()
    ] and session_id == id(model.sess)
    summary = dict(
        status="complete",
        context=context,
        batch=1,
        measured_requests=args.measured_requests,
        load_compile_seconds=load_seconds,
        cache_pointers_unchanged=True,
        sessions_unchanged=True,
        replay_exact=True,
        protocol_sha256=sha(folder / "protocol.json"),
        rows_sha256=sha(folder / "rows.jsonl"),
        phases={},
    )
    for phase, count in [("prefill", context), ("decode", 64)]:
        values = [r[phase + "_seconds"] for r in rows if not r["warmup"]]
        mean = statistics.mean(values)
        summary["phases"][phase] = dict(
            mean_seconds=mean,
            tokens_s=count / mean,
            min_seconds=min(values),
            max_seconds=max(values),
            std_seconds=statistics.stdev(values),
        )
    save(folder / "summary.json", summary)


if __name__ == "__main__":
    main()
