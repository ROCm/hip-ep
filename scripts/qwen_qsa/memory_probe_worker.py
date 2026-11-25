#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""One isolated native model, one prefill and eight decode calls for memory only."""

import argparse
import json
import os
import time
from pathlib import Path

from verify_handoff import WORK, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["Original", "Changed"], required=True)
    parser.add_argument(
        "--context", type=int, choices=[4096, 16384, 32768], required=True
    )
    parser.add_argument("--batch", type=int, choices=[1, 4, 8], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def phase(name):
        event = {"phase": name, "unix": time.time(), "pid": os.getpid()}
        with (args.output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
        temporary = args.output / "phase.tmp"
        temporary.write_text(json.dumps(event) + "\n")
        temporary.replace(args.output / "phase.json")
        print("phase", name, flush=True)

    result = {
        "variant": args.variant,
        "context": args.context,
        "batch": args.batch,
        "capacity": args.context + 64,
        "prefill_calls": 1,
        "decode_calls": 8,
        "worker_pid": os.getpid(),
        "worker_sha256": sha(Path(__file__)),
        "purpose": "Memory only; no warmup/repeated performance trials",
        "training_updates": 0,
    }
    try:
        phase("runtime_import")
        from native_inference import Native
        import torch

        torch.set_num_threads(2)
        validation = json.loads(
            (WORK / "native-gqa/full-model-validation.json").read_text()
        )
        assert validation["status"] == "passed"
        for name, expected in validation["binary_sha256"].items():
            assert sha(WORK / "native-gqa/install/bin" / name) == expected
        data = WORK / "timing-v1"
        manifest = json.loads((data / "manifest.json").read_text())
        assert sha(data / "streams.json") == manifest["streams_sha256"]
        streams = json.loads((data / "streams.json").read_text())
        tokens = [row["input_ids"][: args.context + 8] for row in streams[: args.batch]]
        phase("model_load")
        model = Native(
            WORK / "native-gqa/models" / args.variant,
            capacity=args.context + 64,
            batch=args.batch,
        )
        with torch.inference_mode():
            ids = torch.tensor(tokens, device="cuda", dtype=torch.int64)
            pointers = [value.data_ptr() for value in model.caches.values()]
            torch.cuda.synchronize()
            phase("ready")
            time.sleep(0.4)
            phase("prefill")
            logits = model.forward(ids[:, : args.context].contiguous())
            assert logits.isfinite().all()
            torch.cuda.synchronize()
            phase("prefill_resident")
            time.sleep(0.4)
            phase("decode")
            for step in range(8):
                logits = model.forward(
                    ids[:, args.context + step : args.context + step + 1].contiguous(),
                    past=args.context + step,
                )
                assert logits.isfinite().all()
            torch.cuda.synchronize()
            phase("decode_resident")
            time.sleep(0.4)
            assert pointers == [value.data_ptr() for value in model.caches.values()]
            result.update(
                status="complete",
                finite=True,
                cache_pointers_unchanged=True,
                binary_sha256=validation["binary_sha256"],
                stream_sha256=manifest["streams_sha256"],
                graph_sha256=model.export["onnx_sha256"],
                torch_allocator_subset={
                    "allocated_bytes": torch.cuda.memory_allocated(),
                    "reserved_bytes": torch.cuda.memory_reserved(),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                },
            )
        phase("complete")
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        phase("failed")
        raise
    finally:
        (args.output / "worker-result.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
