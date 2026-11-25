#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Validate frozen features, causal selection, selected attention and KV append."""
# isort: skip_file

import json
import math

from runtime import WORK
import torch

from original_inference import stats
from qsa_runtime import FrozenBank, FrozenQSA, LAYERS


def normalize(x, weight):
    f = x.float()
    return (
        f * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-6) * weight.float()
    ).bfloat16()


def rotate(x, cos, sin):
    first, second = x.float().chunk(2, dim=-1)
    rotated = torch.cat([-second, first], dim=-1)
    return (x.float() * cos + rotated * sin).bfloat16()


def features(projected, qw, kw, cos, sin):
    count = projected.shape[0]
    c = torch.cat([cos[:count], cos[:count]], -1)
    s = torch.cat([sin[:count], sin[:count]], -1)
    queries = rotate(
        normalize(projected[:, :512].reshape(count, 4, 128), qw), c[:, None], s[:, None]
    )
    groups = count // 4
    pooled = (
        projected[: groups * 4, 512:].float().reshape(groups, 4, 128).mean(1).bfloat16()
    )
    keys = rotate(normalize(pooled, kw), c[: groups * 4 : 4], s[: groups * 4 : 4])
    return queries, keys


def attention(query, keys, values, selected, position):
    groups = (position + 1) // 4
    chosen = min(128, groups)
    ids = (
        selected[:chosen].long()[:, None] * 4 + torch.arange(4, device="cuda")[None]
    ).flatten()
    tail = torch.arange(groups * 4, position + 1, device="cuda")
    ids = torch.cat([ids, tail])
    q = query.reshape(32, 128).float()
    k, v = keys[0, :, ids].float(), values[0, :, ids].float()
    weights = ((q[:, None] * k).sum(-1) / math.sqrt(128)).softmax(-1)
    return (weights[..., None] * v).sum(1).half().reshape(4096)


@torch.inference_mode()
def case(bank, layer, prefix):
    capacity = 4160
    keys = torch.full((1, 32, capacity, 128), 1.25, dtype=torch.float16, device="cuda")
    values = torch.full_like(keys, -2.5)
    expected_k, expected_v = keys.clone(), values.clone()
    qsa = FrozenQSA(bank, layer, capacity, keys, values)
    qsa.capture = []
    hidden = (
        torch.randn((1, prefix + 4, 4096), dtype=torch.float16, device="cuda") * 0.2
    )
    queries, new_k, new_v = [torch.randn_like(hidden) for _ in range(3)]
    projected = torch.nn.functional.linear(hidden.bfloat16(), qsa.weight)[0]
    ref_q, ref_pool = features(
        projected, qsa.q_weight, qsa.k_weight, bank.cos, bank.sin
    )
    rows = []
    for past, count in [(0, prefix), *[(prefix + i, 1) for i in range(4)]]:
        part = slice(past, past + count)
        output = torch.empty((1, count, 4096), dtype=torch.float16, device="cuda")
        qsa.capture.clear()
        qsa.run(
            hidden[:, part].contiguous(),
            queries[:, part].contiguous(),
            new_k[:, part].contiguous(),
            new_v[:, part].contiguous(),
            output,
            past,
        )
        torch.cuda.synchronize()
        expected_k[:, :, part] = (
            new_k[:, part].reshape(1, count, 32, 128).transpose(1, 2)
        )
        expected_v[:, :, part] = (
            new_v[:, part].reshape(1, count, 32, 128).transpose(1, 2)
        )
        assert torch.equal(keys, expected_k) and torch.equal(values, expected_v)
        qerr = stats(qsa.features[:count], ref_q[part])
        pool_count = (past + count) // 4
        perr = (
            stats(qsa.pool[:pool_count], ref_pool[:pool_count])
            if pool_count
            else {"relative_l2": 0.0}
        )
        assert qerr["relative_l2"] < 0.005 and perr["relative_l2"] < 0.005, (qerr, perr)
        captured = torch.cat([r["selected"] for r in qsa.capture], dim=0)
        sampled = sorted({0, count - 1, *range(0, count, 257)})
        max_attention_error, max_gap = 0.0, 0.0
        for row in sampled:
            position = past + row
            groups = (position + 1) // 4
            chosen = min(128, groups)
            selected = captured[row]
            if chosen:
                ids = selected[:chosen].long()
                assert int(ids.min()) >= 0 and int(ids.max()) < groups
                assert torch.equal(ids, ids.unique(sorted=True))
                if groups > 128:
                    # Validate top-k optimality against FP32 score math using
                    # native rounded features; report near-boundary FP32 gaps.
                    scores = (
                        qsa.features[row].float() @ qsa.pool[:groups].float().t()
                    ).relu().sum(0) / math.sqrt(128)
                    mask = torch.ones(groups, dtype=torch.bool, device="cuda")
                    mask[ids] = False
                    gap = float(scores[mask].max() - scores[ids].min())
                    max_gap = max(max_gap, gap)
                    assert gap < 1e-4 * max(1.0, float(scores.abs().max())), gap
            ref = attention(queries[0, position], keys, values, selected, position)
            err = stats(output[0, row], ref)
            max_attention_error = max(max_attention_error, err["relative_l2"])
            assert err["finite"] and err["relative_l2"] < 0.001, err
        rows.append(
            {
                "layer": layer,
                "prefix": prefix,
                "past": past,
                "new": count,
                "query_features_relative_l2": qerr["relative_l2"],
                "pool_relative_l2": perr["relative_l2"],
                "max_attention_relative_l2": max_attention_error,
                "max_topk_optimality_gap": max_gap,
                "cache_append_exact": True,
                "workspaces_stable": qsa.pointers() == qsa.workspace_pointers,
            }
        )
    return rows


@torch.inference_mode()
def ties(bank):
    features = torch.zeros((1, 4, 128), dtype=torch.bfloat16, device="cuda")
    pool = torch.randn((1024, 128), dtype=torch.bfloat16, device="cuda")
    scores = torch.empty((1, 1024), dtype=torch.float32, device="cuda")
    selected = torch.empty((1, 128), dtype=torch.int32, device="cuda")
    status = bank.kernel.select(
        torch.cuda.current_stream().cuda_stream,
        features.data_ptr(),
        pool.data_ptr(),
        scores.data_ptr(),
        selected.data_ptr(),
        1,
        1,
        4095,
        0,
        1,
        1024,
    )
    assert status == 0
    torch.cuda.synchronize()
    assert torch.equal(selected[0], torch.arange(128, dtype=torch.int32, device="cuda"))
    return {"zero_scores": True, "ties_prefer_lower_group_ids": True}


def main():
    torch.set_num_threads(8)
    torch.manual_seed(42)
    receipt = {"status": "running", "batch": 1, "training_updates": 0, "rows": []}
    try:
        bank = FrozenBank()
        for layer in LAYERS:
            for prefix in [1, 3, 4, 511, 512, 516, 517, 519, 1025, 4097]:
                receipt["rows"].extend(case(bank, layer, prefix))
                print("QSA raw validation", layer, prefix, "passed", flush=True)
        receipt["ties"] = ties(bank)
        receipt["integrity"] = bank.integrity()
        receipt["status"] = "passed"
    except BaseException as exc:
        receipt.update(status="failed", error=repr(exc))
        raise
    finally:
        (WORK / "qsa-raw-validation.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        print(
            json.dumps({k: v for k, v in receipt.items() if k != "rows"}, indent=2),
            flush=True,
        )


if __name__ == "__main__":
    main()
