#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Long-context B1 QSA decode operator checks with explicit fixture histories."""
# isort: skip_file

import json
import math

from runtime import WORK
import torch

from original_inference import stats
from qsa_runtime import FrozenBank, FrozenQSA
from validate_qsa import attention, normalize, rotate


@torch.inference_mode()
def main():
    torch.set_num_threads(8)
    torch.manual_seed(43)
    receipt = {
        "status": "running",
        "batch": 1,
        "training_updates": 0,
        "fixture_history": True,
        "rows": [],
    }
    try:
        bank = FrozenBank()
        capacity = 65536
        keys = torch.randn((1, 32, capacity, 128), dtype=torch.float16, device="cuda")
        values = torch.randn_like(keys)
        qsa = FrozenQSA(bank, 7, capacity, keys, values)
        qsa.capture = []
        for position in [16383, 32767, 65535]:
            qsa.pool.normal_()
            qsa.tail.normal_()
            before_tail = qsa.tail.clone()
            qsa.seen = position  # Explicit operator fixture, not a model cache attach.
            hidden, query, key, value = [
                torch.randn((1, 1, 4096), dtype=torch.float16, device="cuda")
                for _ in range(4)
            ]
            output = torch.empty_like(query)
            qsa.capture.clear()
            qsa.run(hidden, query, key, value, output, position)
            torch.cuda.synchronize()
            assert torch.equal(keys[0, :, position], key.reshape(32, 128))
            assert torch.equal(values[0, :, position], value.reshape(32, 128))
            projected = qsa.projected[0]
            cos = torch.cat([bank.cos[position], bank.cos[position]])
            sin = torch.cat([bank.sin[position], bank.sin[position]])
            reference_q = rotate(
                normalize(projected[:512].reshape(4, 128), qsa.q_weight), cos, sin
            )
            qerr = stats(qsa.features[0], reference_q)
            before_tail[3] = projected[512:]
            pooled = before_tail.float().mean(0).bfloat16()
            anchor = position - 3
            reference_pool = rotate(
                normalize(pooled, qsa.k_weight),
                torch.cat([bank.cos[anchor], bank.cos[anchor]]),
                torch.cat([bank.sin[anchor], bank.sin[anchor]]),
            )
            perr = stats(qsa.pool[position // 4], reference_pool)
            assert qerr["relative_l2"] < 0.005 and perr["relative_l2"] < 0.005
            selected = qsa.capture[0]["selected"][0].long()
            groups = (position + 1) // 4
            scores = (
                qsa.features[0].float() @ qsa.pool[:groups].float().t()
            ).relu().sum(0) / math.sqrt(128)
            other = torch.ones(groups, dtype=torch.bool, device="cuda")
            other[selected] = False
            gap = float(scores[other].max() - scores[selected].min())
            assert gap < 1e-4 * max(1.0, float(scores.abs().max()))
            ref = attention(query[0, 0], keys, values, selected, position)
            aerr = stats(output[0, 0], ref)
            assert aerr["finite"] and aerr["relative_l2"] < 0.001
            row = {
                "context": position + 1,
                "query_relative_l2": qerr["relative_l2"],
                "pool_relative_l2": perr["relative_l2"],
                "attention_relative_l2": aerr["relative_l2"],
                "topk_gap": gap,
                "workspace_pointers_stable": qsa.pointers() == qsa.workspace_pointers,
            }
            receipt["rows"].append(row)
            print(json.dumps(row), flush=True)
        receipt.update(status="passed", selector_integrity=bank.integrity())
    except BaseException as exc:
        receipt.update(status="failed", error=repr(exc))
        raise
    finally:
        (WORK / "qsa-long-validation.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
