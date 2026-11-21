#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Validate the integrated frozen Changed model before quality or timing."""
# isort: skip_file

import json
import time

from runtime import WORK
import torch

from original_inference import stats
from partitioned_inference import PartitionedOriginal
from prepare_baseline import REFERENCE, original_folder


@torch.inference_mode()
def main():
    torch.set_num_threads(8)
    folder = original_folder() / "partitioned"
    assert json.loads((folder / "validation.json").read_text())["status"] == "passed"
    assert (
        json.loads((WORK / "qsa-raw-validation.json").read_text())["status"] == "passed"
    )
    receipt = {"status": "running", "batch": 1, "training_updates": 0, "checks": []}
    started = time.perf_counter()
    try:
        model = PartitionedOriginal(4608)
        model.enable_qsa()
        assert sorted(model.qsa) == [7, 11, 15, 23] and 19 not in model.qsa
        session_ids = [id(s) for s in model.sessions]
        cache_pointers = [t.data_ptr() for t in model.caches.values()]
        tokens = json.loads(
            (REFERENCE / "data/selector-extended-v1/test.json").read_text()
        )[0]["input_ids"]
        ids = torch.tensor([tokens], dtype=torch.int64, device="cuda")
        for variant in ["Original", "Changed"]:
            model.variant = variant
            repeats = []
            for _ in range(2):
                parts = [model.forward(ids[:, :517].contiguous()).clone()]
                for pos in range(517, 521):
                    parts.append(
                        model.forward(
                            ids[:, pos : pos + 1].contiguous(), past=pos
                        ).clone()
                    )
                values = torch.cat(parts, dim=1)
                assert values.isfinite().all()
                repeats.append(values)
            assert torch.equal(*repeats)
            receipt["checks"].append(
                {
                    "variant": variant,
                    "name": "prefill_517_plus_four_replay",
                    "bit_exact": True,
                    "finite": True,
                }
            )
        model.variant = "Changed"
        full = model.forward(ids, all_logits=True).clone()
        assert full.isfinite().all()
        parts = [model.forward(ids[:, :4092].contiguous()).clone()]
        for pos in range(4092, 4096):
            parts.append(
                model.forward(ids[:, pos : pos + 1].contiguous(), past=pos).clone()
            )
        cached = torch.cat(parts, dim=1)
        parity = stats(cached, full[:, 4091:])
        receipt["checks"].append(
            {"variant": "Changed", "name": "full_vs_cached_4092_plus_four", **parity}
        )
        assert parity["relative_l2"] < 0.02 and parity["cosine"] > 0.999, parity
        assert session_ids == [id(s) for s in model.sessions]
        assert cache_pointers == [t.data_ptr() for t in model.caches.values()]
        receipt.update(
            status="passed",
            changed_layers=sorted(model.qsa),
            layer_19="Original dense",
            shared_backbone_session_ids_unchanged=True,
            cache_pointers_stable=True,
            selector_integrity=model.bank.integrity(),
            projection_nodes_sha256=model.manifest["projection_nodes_sha256"],
        )
    except BaseException as exc:
        receipt.update(status="failed", error=repr(exc))
        raise
    finally:
        receipt["elapsed_seconds"] = time.perf_counter() - started
        (WORK / "changed-validation.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
