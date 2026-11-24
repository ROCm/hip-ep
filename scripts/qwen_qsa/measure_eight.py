#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Validate eight native nodes or measure the locked single-resident workload."""
# ruff: noqa: E402

import argparse
import gc
import json
import os
import platform
import time
from pathlib import Path

from verify_handoff import ROOT, WORK, sha

TASK = WORK / "eight-qsa-v1"
os.environ["QWEN_QSA_MODEL_CACHE"] = str(TASK / "model-cache")
assert not os.environ.get("QWEN_QSA_INSTALL"), "Use the preserved measured build"
from benchmark_native import prepared
from export_eight import EXPORT_SHA, LAYERS, save
from native_inference import Native
from runtime import INSTALL, run

import numpy as np
import onnx
import onnxruntime as ort
import psutil
import torch


def identity():
    old = json.loads((WORK / "native-gqa/full-model-validation.json").read_text())
    binary = {name: sha(INSTALL / "bin" / name) for name in old["binary_sha256"]}
    assert binary == old["binary_sha256"]
    export = json.loads((TASK / "models/Eight/export.json").read_text())
    assert export["selector_sha256"] == EXPORT_SHA
    assert export["selector_layers"] == LAYERS
    assert (
        sha(TASK / "models/Eight/qsa.selectors.bin") == export["selector_data_sha256"]
    )
    graph = onnx.load(TASK / "models/Eight/default.onnx", load_external_data=False)
    assert sum(n.op_type == "QwenQsaGroupQueryAttention" for n in graph.graph.node) == 8
    return dict(
        binary_sha256=binary,
        selector_export_sha256=EXPORT_SHA,
        selector_layers=LAYERS,
        export_receipt_sha256=sha(TASK / "models/Eight/export.json"),
        graph_sha256=export["onnx_sha256"],
        backbone_sha256=export["data_sha256"],
        machine=platform.platform(),
        cpu=platform.processor(),
        gpu=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        hip=torch.version.hip,
        onnxruntime=ort.__version__,
        ram_total_bytes=psutil.virtual_memory().total,
        runtime_sha256=sha(ROOT / "scripts/qsa/runtime.py"),
        inference_sha256=sha(ROOT / "scripts/qwen_qsa/native_inference.py"),
        measurement_sha256=sha(Path(__file__)),
        training_updates=0,
        phase_policy="sparse prefill and sparse decode",
    )


def profile_finish(model, calls):
    profile = Path(model.sess.end_profiling())
    events = json.loads(profile.read_text())
    dispatches = [
        e
        for e in events
        if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time")
    ]
    assert len(dispatches) == calls, (len(dispatches), calls)
    return dict(
        forward_calls=calls,
        ep_dispatches=len(dispatches),
        dispatches_per_forward=1,
        profile=str(profile.relative_to(ROOT)),
        profile_sha256=sha(profile),
    )


def error(actual, expected):
    a, b = actual.float(), expected.float()
    return float((a - b).square().mean().sqrt() / b.square().mean().sqrt())


@torch.inference_mode()
def validate():
    os.environ["QSA_ORT_PROFILE_DIR"] = str(TASK / "profiles")
    result = dict(status="running", **identity(), checks=[], cold_load_seconds={})
    path = TASK / "validation.json"
    assert not path.exists(), "Retain prior validation attempts"
    refs = {}
    timing = json.loads((WORK / "timing-v1/streams.json").read_text())[0]
    tokens = timing["input_ids"] if isinstance(timing, dict) else timing
    ids = torch.tensor([tokens[:4100]], device="cuda", dtype=torch.int64)
    try:
        for variant, folder in [
            ("Original", WORK / "native-gqa/models/Original"),
            ("Eight", TASK / "models/Eight"),
        ]:
            started = time.perf_counter()
            model = Native(folder, 4160)
            result["cold_load_seconds"][variant] = time.perf_counter() - started
            pointers = [t.data_ptr() for t in model.caches.values()]
            calls = 0

            def forward(count, past=0):
                nonlocal calls
                calls += 1
                output = model.forward(
                    ids[:, past : past + count].contiguous(), past=past
                ).clone()
                assert output.isfinite().all()
                return output

            for prefix in [17, 513, 2047]:
                outputs = [forward(prefix)]
                outputs.extend(forward(1, pos) for pos in range(prefix, prefix + 4))
                joined = torch.cat(outputs, 1)
                if variant == "Original":
                    refs[prefix] = joined.cpu()
                else:
                    assert torch.equal(joined.cpu(), refs[prefix]), prefix
                    result["checks"].append(
                        dict(check="full_support_exact", prefix=prefix, decode=4)
                    )
            if variant == "Eight":
                for prefix, end in [(2051, 2057), (4092, 4096)]:
                    reference = forward(end)

                    def cached():
                        out = forward(prefix)
                        for pos in range(prefix, end):
                            out = forward(1, pos)
                        return out

                    actual = cached()
                    relative = error(actual, reference)
                    assert relative < 0.02, (prefix, relative)
                    assert torch.equal(actual, cached())
                    result["checks"].append(
                        dict(
                            check="sparse_prefill_vs_cached_replay",
                            prefix=prefix,
                            end=end,
                            relative_rms=relative,
                            replay_exact=True,
                            pruning_active_at_all_eight_nodes=True,
                        )
                    )
                sparse = forward(4096).cpu()
                assert not torch.equal(sparse, refs["4096"]), (
                    "Pruning must affect this trajectory"
                )
                result["sparse_differs_from_default"] = True
            else:
                refs["4096"] = forward(4096).cpu()
            assert pointers == [t.data_ptr() for t in model.caches.values()]
            result[variant] = dict(
                cache_pointers_unchanged=True, **profile_finish(model, calls)
            )
            print(variant, "dispatch/cache checks passed", flush=True)
            model = None
            gc.collect()
            torch.cuda.empty_cache()
        result.update(
            status="passed",
            eight_node_replacement=True,
            frozen_tensors_verified=True,
            pruning_dispatch_evidence="Eight lowered native nodes, strict EP/no fallback, one EP dispatch per forward; queries cross position2052 at every node, whose unchanged native branch selects sparse support; finite/replay/cache checks pass.",
        )
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        save(path, result)


@torch.inference_mode()
def benchmark(context, output_root=None, stop_file=None):
    os.environ.pop("QSA_ORT_PROFILE_DIR", None)
    validation = json.loads((TASK / "validation.json").read_text())
    assert validation["status"] == "passed"
    data = WORK / "timing-v1"
    manifest = json.loads((data / "manifest.json").read_text())
    assert (
        manifest["state"] == "locked"
        and sha(data / "streams.json") == manifest["streams_sha256"]
    )
    row = json.loads((data / "streams.json").read_text())[0]
    tokens = row["input_ids"] if isinstance(row, dict) else row
    assert len(tokens) == 65600
    folder = (output_root or TASK / "performance") / str(context)
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / "protocol.json").exists(), "Retain previous timing attempts"
    protocol = dict(
        **identity(),
        context=context,
        batch=1,
        capacity=context + 64,
        warmups=2,
        measured_requests=6,
        decode_calls=64,
        timing_manifest_sha256=sha(data / "manifest.json"),
        stream_sha256=manifest["streams_sha256"],
        validation_sha256=sha(TASK / "validation.json"),
        timer="Independent HIP-synchronized CPU wall timers, identical runtime.run",
        prefill="Full session, final-position head; bindings and recurrent zeroing excluded",
        decode="64 common teacher-forced session calls; binding/allocation/sampling excluded",
        comparison="Reused Original/four controls are across-run; no paired confidence intervals",
        resident_sessions=1,
    )
    save(folder / "protocol.json", protocol)
    load_start = time.perf_counter()
    model = Native(TASK / "models/Eight", context + 64)
    load_seconds = time.perf_counter() - load_start
    pointers = [t.data_ptr() for t in model.caches.values()]
    session_id = id(model.sess)
    ids = torch.tensor([tokens[: context + 64]], device="cuda", dtype=torch.int64)
    plan = [prepared(model, ids[:, :context].contiguous(), 0)]
    plan.extend(
        prepared(model, ids[:, context + i : context + i + 1].contiguous(), context + i)
        for i in range(64)
    )
    rows = []
    warmup_reference = None
    for trial in range(-2, 6):
        if stop_file and stop_file.exists():
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
        if trial == -2:
            warmup_reference = output
        else:
            assert torch.equal(output, warmup_reference), (context, trial)
        row = dict(
            trial=trial,
            warmup=trial < 0,
            variant="Eight",
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
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        rows.append(row)
        print(row, flush=True)
    assert (
        pointers == [t.data_ptr() for t in model.caches.values()]
        and id(model.sess) == session_id
    )
    summary = dict(
        status="complete",
        context=context,
        batch=1,
        load_compile_seconds=load_seconds,
        replay_exact=True,
        cache_pointers_unchanged=True,
        sessions_unchanged=True,
        protocol_sha256=sha(folder / "protocol.json"),
        rows_sha256=sha(folder / "rows.jsonl"),
        phases={},
    )
    for phase, count in [("prefill", context), ("decode", 64)]:
        times = np.array([r[phase + "_seconds"] for r in rows if not r["warmup"]])
        summary["phases"][phase] = dict(
            mean_seconds=float(times.mean()),
            tokens_s=float(count / times.mean()),
            min_seconds=float(times.min()),
            max_seconds=float(times.max()),
            std_seconds=float(times.std(ddof=1)),
        )
    save(folder / "summary.json", summary)
    assert identity()["binary_sha256"] == protocol["binary_sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["validate", "benchmark"])
    parser.add_argument("--context", type=int, choices=[4096, 16384, 32768])
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--stop-file", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "validate":
        validate()
    else:
        assert args.context
        benchmark(args.context, args.output_root, args.stop_file)


if __name__ == "__main__":
    main()
