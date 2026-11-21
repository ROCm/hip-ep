#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Validate the full single-session native QSA model against the boundary reference."""
# isort: skip_file
# Select the native EP and DLL paths before importing the GPU runtime.
# ruff: noqa: E402

import json
import os
from pathlib import Path

TASK = Path(__file__).resolve().parents[2] / "_local/qsa/native-gqa"
os.environ["QSA_HIP_EP_INSTALL"] = str(TASK / "install")
os.environ["QSA_MODEL_CACHE"] = str(TASK / "model-cache")
os.environ["QSA_ORT_PROFILE_DIR"] = str(TASK / "ort-profiles")

from runtime import INSTALL
import torch

from original_inference import Original, stats
from partitioned_inference import PartitionedOriginal
from prepare_baseline import REFERENCE, SELECTOR, SELECTOR_SHA, sha


@torch.inference_mode()
def main():
    assert INSTALL.resolve() == (TASK / "install").resolve()
    torch.set_num_threads(8)
    receipt = {"status": "running", "batch": 1, "training_updates": 0, "rows": []}
    try:
        reference = PartitionedOriginal(4608)
        reference.enable_qsa()
        original = Original(TASK / "models/Original", 4608, caches=reference.caches)
        changed = Original(TASK / "models/Changed", 4608, caches=reference.caches)
        pointers = [t.data_ptr() for t in reference.caches.values()]
        tokens = json.loads(
            (REFERENCE / "data/selector-extended-v1/test.json").read_text(
                encoding="utf-8"
            )
        )[0]["input_ids"]
        tokens = torch.tensor([tokens], device="cuda", dtype=torch.int64)

        def sequence(model, prefix):
            values = [model.forward(tokens[:, :prefix].contiguous()).clone()]
            for pos in range(prefix, prefix + 4):
                values.append(
                    model.forward(
                        tokens[:, pos : pos + 1].contiguous(), past=pos
                    ).clone()
                )
            result = torch.cat(values, dim=1)
            assert result.isfinite().all()
            return result

        for prefix in [17, 517, 1025, 4092]:
            reference.variant = "Original"
            dense_ref = sequence(reference, prefix)
            dense = sequence(original, prefix)
            parity = stats(dense, dense_ref)
            assert parity["relative_l2"] < 0.002 and parity["cosine"] > 0.99999, parity
            receipt["rows"].append(
                {
                    "check": "native_original_vs_dense_boundary",
                    "prefix": prefix,
                    **parity,
                }
            )
            reference.variant = "Changed"
            sparse_ref = sequence(reference, prefix)
            sparse = sequence(changed, prefix)
            parity = stats(sparse, sparse_ref)
            assert parity["relative_l2"] < 0.02 and parity["cosine"] > 0.999, parity
            assert torch.equal(sparse, sequence(changed, prefix))
            receipt["rows"].append(
                {
                    "check": "native_qsa_vs_boundary",
                    "prefix": prefix,
                    "replay_exact": True,
                    **parity,
                }
            )
            print("Native GQA/QSA full-model parity", prefix, parity, flush=True)
        full = changed.forward(tokens, all_logits=True).clone()
        cached = sequence(changed, 4092)
        parity = stats(cached, full[:, 4091:])
        assert parity["relative_l2"] < 0.02 and parity["cosine"] > 0.999, parity
        receipt["rows"].append(
            {"check": "native_qsa_full_vs_cached_4092_plus_four", **parity}
        )
        assert pointers == [t.data_ptr() for t in reference.caches.values()]
        assert pointers == [t.data_ptr() for t in original.caches.values()]
        assert pointers == [t.data_ptr() for t in changed.caches.values()]
        assert sha(SELECTOR) == SELECTOR_SHA
        receipt["dispatch_profiles"] = {}
        for name, model, expected_runs in [
            ("Original", original, 20),
            ("Changed", changed, 46),
        ]:
            profile = Path(model.sess.end_profiling())
            events = json.loads(profile.read_text(encoding="utf-8"))
            dispatches = [
                e
                for e in events
                if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time")
            ]
            assert len(dispatches) == expected_runs, (
                name,
                len(dispatches),
                expected_runs,
            )
            receipt["dispatch_profiles"][name] = {
                "file": str(profile),
                "forward_calls": expected_runs,
                "ep_node_dispatches": len(dispatches),
                "ep_dispatches_per_forward": 1,
            }
        receipt.update(
            status="passed",
            selector_integrity=reference.bank.integrity(),
            execution="one native ONNX session and one run per forward in both variants",
            changed_layers=[7, 11, 15, 23],
            layer_19="dense",
            cache_pointers_unchanged=True,
        )
    except BaseException as exc:
        receipt.update(status="failed", error=repr(exc))
        raise
    finally:
        (TASK / "validation.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
