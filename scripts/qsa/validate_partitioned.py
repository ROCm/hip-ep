#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Verify that controlled boundaries preserve the complete dense Original."""
# isort: skip_file

import json
import time

import torch

from original_inference import Original, stats
from partitioned_inference import PartitionedOriginal
from prepare_baseline import REFERENCE, original_folder


@torch.inference_mode()
def main():
    torch.set_num_threads(8)
    folder = original_folder() / "partitioned"
    receipt = {"status": "running", "training_updates": 0, "checks": []}
    started = time.perf_counter()
    try:
        full, split = Original(original_folder(), 4608), PartitionedOriginal(4608)
        data = json.loads(
            (REFERENCE / "data/selector-extended-v1/test.json").read_text()
        )[0]["input_ids"]
        ids = torch.tensor([data], dtype=torch.int64, device="cuda")
        before = [[t.data_ptr() for t in m.caches.values()] for m in [full, split]]
        for count in [517, 4096]:
            a = full.forward(ids[:, :count].contiguous()).clone()
            b = split.forward(ids[:, :count].contiguous()).clone()
            check = stats(a, b)
            receipt["checks"].append(
                {
                    "name": "dense_prefill",
                    "tokens": count,
                    "bit_exact": torch.equal(a, b),
                    **check,
                }
            )
            assert check["relative_l2"] < 0.001 and check["cosine"] > 0.99999, check
            for step in range(4):
                token = ids[:, step : step + 1].contiguous()
                a = full.forward(token, past=count + step).clone()
                b = split.forward(token, past=count + step).clone()
                check = stats(a, b)
                receipt["checks"].append(
                    {
                        "name": "dense_decode",
                        "prefix": count,
                        "step": step,
                        "bit_exact": torch.equal(a, b),
                        **check,
                    }
                )
                assert check["relative_l2"] < 0.001 and check["cosine"] > 0.99999, check
        assert before == [
            [t.data_ptr() for t in m.caches.values()] for m in [full, split]
        ]
        receipt.update(
            status="passed",
            cache_pointers_stable=True,
            shared_projection_nodes_sha256=split.manifest["projection_nodes_sha256"],
        )
    except BaseException as exc:
        receipt.update(status="failed", error=repr(exc))
        raise
    finally:
        receipt["elapsed_seconds"] = time.perf_counter() - started
        (folder / "validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
