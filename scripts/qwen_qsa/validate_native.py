#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Check matched full-model dense support, sparse cache continuity and dispatches."""
# ruff: noqa: E402

import argparse
import gc
import json
import os
from pathlib import Path

from verify_handoff import SOURCE, WORK, sha

os.environ.setdefault("QSA_ORT_PROFILE_DIR", str(WORK / "native-gqa/ort-profiles"))
from native_inference import Native
from runtime import INSTALL

import torch


def statistics(actual, expected):
    a, b = actual.float(), expected.float()
    return {
        "relative_rms": float(
            (a - b).square().mean().sqrt() / b.square().mean().sqrt()
        ),
        "max_absolute": float((a - b).abs().max()),
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=WORK / "native-gqa")
    parser.add_argument(
        "--kernel-receipt",
        type=Path,
        default=WORK / "native-gqa/kernel-validation.json",
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    task = WORK / "native-gqa"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "full-model-validator.py").write_bytes(
        Path(__file__).read_bytes()
    )
    kernel = json.loads(args.kernel_receipt.read_text())
    assert kernel["status"] == "passed"
    assert sha(INSTALL / "bin/custom_kernels_gfx1151.dll") == kernel["kernel_sha256"]
    names = json.loads((WORK / "default-4-support.json").read_text())["binary_sha256"]
    result = {
        "status": "running",
        "training_updates": 0,
        "rows": [],
        "binary_sha256": {name: sha(INSTALL / "bin" / name) for name in names},
        "validator_sha256": sha(Path(__file__)),
        "kernel_validation_sha256": sha(args.kernel_receipt),
        "install": str(INSTALL),
    }
    try:
        data = json.loads((SOURCE / "data/qwen-selector-v1/test.json").read_text())
        tokens = torch.tensor([data[0]["input_ids"]], device="cuda", dtype=torch.int64)
        counts = {"Original": 0, "Changed": 0}
        models, pointers = {}, {}

        def forward(variant, ids, past=0, all_logits=False):
            counts[variant] += 1
            output = (
                models[variant]
                .forward(ids.contiguous(), past=past, all_logits=all_logits)
                .clone()
            )
            assert output.isfinite().all()
            return output

        def finish(variant):
            model = models[variant]
            assert pointers[variant] == [x.data_ptr() for x in model.caches.values()]
            profile = Path(model.sess.end_profiling())
            events = json.loads(profile.read_text())
            dispatches = [
                e
                for e in events
                if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time")
            ]
            assert len(dispatches) == counts[variant], (
                variant,
                len(dispatches),
                counts,
            )
            result[variant] = {
                "forward_calls": counts[variant],
                "ep_node_dispatches": len(dispatches),
                "dispatches_per_forward": 1,
                "cache_pointers_unchanged": True,
                "graph_sha256": sha(task / "models" / variant / "default.onnx"),
                "profile_sha256": sha(profile),
            }

        references = {}
        for variant in counts:
            models[variant] = Native(task / "models" / variant, capacity=4096)
            pointers[variant] = [x.data_ptr() for x in models[variant].caches.values()]
            for prefix in [1, 17, 513, 2047]:
                values = [forward(variant, tokens[:, :prefix])]
                for pos in range(prefix, prefix + 4):
                    values.append(forward(variant, tokens[:, pos : pos + 1], past=pos))
                output = torch.cat(values, 1).cpu()
                del values
                if variant == "Original":
                    references[prefix] = output
                else:
                    check = statistics(output, references[prefix])
                    assert torch.equal(output, references[prefix]), check
                    result["rows"].append(
                        {"check": "full_support_exact", "prefix": prefix, **check}
                    )
                    print(result["rows"][-1], flush=True)
            if variant == "Original":
                finish(variant)
                del models[variant]
                gc.collect()
                torch.cuda.empty_cache()

        # Crossing the first pruning query must preserve all selector/cache state.
        def sequence(prefix):
            values = [forward("Changed", tokens[:, :prefix])]
            for pos in range(prefix, 4096):
                values.append(forward("Changed", tokens[:, pos : pos + 1], past=pos))
            return torch.cat(values, 1)

        full = forward("Changed", tokens, all_logits=True)
        for prefix in [4092]:
            cached = sequence(prefix)
            check = statistics(cached, full[:, prefix - 1 :])
            assert check["relative_rms"] < 0.02, check
            assert torch.equal(cached, sequence(prefix))
            result["rows"].append(
                {
                    "check": "sparse_full_vs_cached",
                    "prefix": prefix,
                    "decode": 4096 - prefix,
                    "replay_exact": True,
                    **check,
                }
            )
        # A bounded boundary sequence directly exercises dense -> sparse dispatch.
        boundary = [forward("Changed", tokens[:, :2051])]
        for pos in range(2051, 2057):
            boundary.append(forward("Changed", tokens[:, pos : pos + 1], past=pos))
        boundary = torch.cat(boundary, 1)
        partial = forward("Changed", tokens[:, :2057], all_logits=True)
        check = statistics(boundary, partial[:, 2050:])
        assert check["relative_rms"] < 0.02, check
        result["rows"].append({"check": "first_pruning_continuity", **check})
        finish("Changed")
        result["status"] = "passed"
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        (args.output_dir / "full-model-validation.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    print("Full-model native Qwen controls passed", flush=True)


if __name__ == "__main__":
    main()
