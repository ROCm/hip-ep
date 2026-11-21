#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Validate native Qwen features, stable indices, incremental pools and attention."""
# ruff: noqa: E402

import ctypes
import json
from pathlib import Path

import native_inference
from selector_reference import norm, rotate, token_support
from verify_handoff import PAYLOAD_SHA, SOURCE, WORK, sha

import torch
from safetensors.torch import load_file


def relative(actual, expected):
    return float(
        (actual.float() - expected.float()).square().mean().sqrt()
        / expected.float().square().mean().sqrt().clamp_min(1e-12)
    )


class Kernels:
    def __init__(self):
        self.path = WORK / "native-gqa/install/bin/custom_kernels_gfx1151.dll"
        self.dll = ctypes.CDLL(str(self.path))
        self.stream = torch.cuda.current_stream().cuda_stream

    def call(self, name, pointers, numbers):
        function = getattr(self.dll, "hip_qwen_qsa_" + name)
        function.argtypes = [ctypes.c_void_p] * (len(pointers) + 1) + [
            ctypes.c_int
        ] * len(numbers)
        function.restype = ctypes.c_int
        status = function(self.stream, *[x.data_ptr() for x in pointers], *numbers)
        assert status == 0, (name, status)


@torch.inference_mode()
def main():
    assert native_inference.WORK == WORK
    torch.set_num_threads(2)
    folder = WORK / "selector-fixtures-v1"
    receipt = json.loads((folder / "receipt.json").read_text())
    assert sha(folder / "fixtures.safetensors") == receipt["fixtures_sha256"]
    payload = (
        SOURCE / "checkpoints/hip-ep-qwen/303042-step-0128/selector-bf16.safetensors"
    )
    assert sha(payload) == PAYLOAD_SHA
    fixture = load_file(folder / "fixtures.safetensors", device="cpu")
    weights = load_file(payload, device="cpu")
    kernels = Kernels()
    result = {
        "status": "running",
        "rows": [],
        "training_updates": 0,
        "kernel_sha256": sha(kernels.path),
        "fixtures_sha256": receipt["fixtures_sha256"],
        "selector_sha256": PAYLOAD_SHA,
        "validator_sha256": sha(Path(__file__)),
    }
    path = WORK / "native-gqa/kernel-validation.json"
    try:
        S, C = receipt["tokens"], 2080
        P = (C + 3) // 4
        cos, sin = [
            fixture[name][0, :, :32].contiguous().cuda()
            for name in ["cos_fp32", "sin_fp32"]
        ]
        error = torch.zeros(1, device="cuda", dtype=torch.int32)
        hidden = fixture["hidden_fp16"].cuda()
        positions = (
            torch.arange(S, device="cuda", dtype=torch.int64)[None, None]
            .expand(3, 1, -1)
            .contiguous()
        )
        converted = torch.empty_like(hidden, dtype=torch.bfloat16)
        kernels.call("cast_hidden", [hidden, positions, converted, error], [1, S, 0])
        assert (
            torch.equal(converted.cpu(), fixture["hidden_fp16"].to(torch.bfloat16))
            and error.item() == 0
        )
        malformed = positions.clone()
        malformed[2, 0, 0] = 1
        kernels.call("cast_hidden", [hidden, malformed, converted, error], [1, S, 0])
        assert error.item() != 0
        error.zero_()
        result["rows"].append(
            {"check": "hidden_cast_and_position_rejection", "passed": True}
        )
        for layer in [27, 31, 35, 39]:
            norms = [
                weights[f"selectors.{layer}.{name}.weight"].cuda()
                for name in ["q_norm", "token_k_norm", "pool_k_norm"]
            ]
            original = fixture[f"layer{layer}.projected"].cuda()
            projected = original.clone()
            query = torch.empty(1, S, 512, device="cuda", dtype=torch.bfloat16)
            pool = torch.full(
                (1, P, 128), float("nan"), device="cuda", dtype=torch.bfloat16
            )
            tail = torch.full(
                (1, 4, 128), float("nan"), device="cuda", dtype=torch.bfloat16
            )
            kernels.call(
                "features",
                [projected, *norms, cos, sin, query, pool, tail],
                [1, S, 0, P],
            )
            q_error = relative(
                query.cpu().reshape(1, S, 4, 128), fixture[f"layer{layer}.query"]
            )
            k_error = relative(
                projected[:, :, 512:].cpu(), fixture[f"layer{layer}.token_key"]
            )
            p_error = relative(
                pool[:, : S // 4].cpu(), fixture[f"layer{layer}.pooled_key"]
            )
            assert max(q_error, k_error, p_error) < 0.002, (
                layer,
                q_error,
                k_error,
                p_error,
            )
            # Index math is checked with the frozen source features, independently
            # of projection/norm rounding. Every selected group must match exactly.
            frozen_q = fixture[f"layer{layer}.query"].reshape(1, S, 512).cuda()
            frozen_pool = torch.zeros_like(pool)
            frozen_pool[:, : S // 4] = fixture[f"layer{layer}.pooled_key"].cuda()
            scores = torch.empty(1, 1, P, device="cuda", dtype=torch.float32)
            ids = torch.empty(1, 1, 513, device="cuda", dtype=torch.int32)
            for index, position in enumerate(receipt["positions"]):
                kernels.call(
                    "select",
                    [frozen_q, frozen_pool, scores, ids],
                    [1, S, 0, position, 1, P],
                )
                assert torch.equal(
                    ids[0, 0].cpu(), fixture[f"layer{layer}.selected_groups"][index]
                ), (layer, position)
            # Start with an incomplete group and cross every modulo-four decode
            # boundary, then require exact equality with full-prefill native state.
            incremental_pool = torch.full_like(pool, float("nan"))
            incremental_tail = torch.full_like(tail, float("nan"))
            prefix = 2051
            pre = original[:, :prefix].contiguous()
            pre_q = torch.empty(1, prefix, 512, device="cuda", dtype=torch.bfloat16)
            kernels.call(
                "features",
                [pre, *norms, cos, sin, pre_q, incremental_pool, incremental_tail],
                [1, prefix, 0, P],
            )
            assert torch.equal(pre_q, query[:, :prefix])
            for position in range(prefix, S):
                step = original[:, position : position + 1].contiguous()
                step_q = torch.empty(1, 1, 512, device="cuda", dtype=torch.bfloat16)
                kernels.call(
                    "features",
                    [
                        step,
                        *norms,
                        cos,
                        sin,
                        step_q,
                        incremental_pool,
                        incremental_tail,
                    ],
                    [1, 1, position, P],
                )
                assert torch.equal(step_q, query[:, position : position + 1])
            assert torch.equal(pool[:, : S // 4], incremental_pool[:, : S // 4])
            # The custom single-token projection has the same BF16 contract.
            weight = weights[f"selectors.{layer}.projection.weight"].cuda()
            hidden_step = hidden[:, :1].contiguous()
            position_step = positions[:, :, :1].contiguous()
            projection_step = torch.empty(
                1, 1, 640, device="cuda", dtype=torch.bfloat16
            )
            kernels.call(
                "project_decode",
                [hidden_step, weight, position_step, projection_step, error],
                [1, 0],
            )
            projection_error = relative(
                projection_step.cpu(), fixture[f"layer{layer}.projected"][:, :1]
            )
            assert projection_error < 0.002 and error.item() == 0
            result["rows"].append(
                {
                    "check": "frozen_features_indices_incremental_projection",
                    "layer": layer,
                    "query_relative_rms": q_error,
                    "token_key_relative_rms": k_error,
                    "pooled_key_relative_rms": p_error,
                    "decode_projection_relative_rms": projection_error,
                    "indices_exact": True,
                    "incremental_pool_exact": True,
                }
            )
            print(result["rows"][-1], flush=True)
        for B in [4, 8]:
            length, capacity = 17, 32
            groups = capacity // 4
            cpu_projected = torch.stack(
                [
                    fixture["layer39.projected"][0, b * length : (b + 1) * length]
                    for b in range(B)
                ]
            )
            projected = cpu_projected.cuda()
            q = torch.empty(B, length, 512, device="cuda", dtype=torch.bfloat16)
            p = torch.empty(B, groups, 128, device="cuda", dtype=torch.bfloat16)
            t = torch.empty(B, 4, 128, device="cuda", dtype=torch.bfloat16)
            kernels.call(
                "features",
                [projected, *norms, cos, sin, q, p, t],
                [B, length, 0, groups],
            )
            cpu_cos, cpu_sin = [
                fixture[name][:, :length] for name in ["cos_fp32", "sin_fp32"]
            ]
            expected_q = rotate(
                norm(
                    cpu_projected[:, :, :512].reshape(B, length, 4, 128),
                    weights["selectors.39.q_norm.weight"],
                ),
                cpu_cos,
                cpu_sin,
            )
            expected_k = norm(
                cpu_projected[:, :, 512:], weights["selectors.39.token_k_norm.weight"]
            )
            mean = expected_k[:, :16].float().reshape(B, 4, 4, 128).mean(2).bfloat16()
            expected_p = rotate(
                norm(mean, weights["selectors.39.pool_k_norm.weight"]),
                cpu_cos[:, :16:4],
                cpu_sin[:, :16:4],
            )
            errors = [
                relative(q.cpu().reshape(B, length, 4, 128), expected_q),
                relative(p[:, :4].cpu(), expected_p),
            ]
            assert max(errors) < 0.002, errors
            result["rows"].append(
                {
                    "check": "distinct_batched_features",
                    "batch": B,
                    "query_relative_rms": errors[0],
                    "pooled_key_relative_rms": errors[1],
                }
            )
        # Long stable ties exercise the entire radix range and high positions;
        # B4/B8 here qualify these kernel controls, not full-model memory limits.
        for B, past in [(1, 2052), (4, 32771), (8, 65599)]:
            groups = (past + 4) // 4
            q = torch.zeros(B, 1, 512, device="cuda", dtype=torch.bfloat16)
            p = torch.zeros(B, groups, 128, device="cuda", dtype=torch.bfloat16)
            scores = torch.empty(B, 1, groups, device="cuda")
            ids = torch.empty(B, 1, 513, device="cuda", dtype=torch.int32)
            kernels.call("select", [q, p, scores, ids], [B, 1, past, 0, 1, groups])
            expected = torch.tensor(list(range(512)) + [past // 4], dtype=torch.int32)
            assert torch.equal(ids[:, 0].cpu(), expected[None].expand(B, -1))
            result["rows"].append(
                {
                    "check": "stable_zero_ties",
                    "batch": B,
                    "position": past,
                    "exact": True,
                }
            )
        generator = torch.Generator(device="cpu").manual_seed(20261127)
        cpu_q = torch.randn(1, S, 24, 256, generator=generator).half()
        cpu_k = torch.randn(1, S, 4, 256, generator=generator).half()
        cpu_v = torch.randn(1, S, 4, 256, generator=generator).half()
        q, k, v = [x.cuda() for x in [cpu_q, cpu_k, cpu_v]]
        cache_k = torch.full(
            (1, 4, C, 256), float("nan"), device="cuda", dtype=torch.float16
        )
        cache_v = torch.full_like(cache_k, float("nan"))
        kernels.call("pack_kv", [k, v, cache_k, cache_v], [1, S, 0, C])
        assert torch.equal(cache_k[:, :, :S].cpu(), cpu_k.transpose(1, 2))
        assert torch.equal(cache_v[:, :, :S].cpu(), cpu_v.transpose(1, 2))
        assert cache_k[:, :, S:].isnan().all() and cache_v[:, :, S:].isnan().all()
        partial = torch.empty(1, 1, 24, 16, 256, device="cuda")
        stats = torch.empty(1, 1, 24, 16, 2, device="cuda")
        for position in [0, 7, 2051, 2052, 2055, 2063]:
            i = receipt["positions"].index(position)
            selected = fixture["layer27.selected_groups"][i]
            ids = selected[None, None].cuda()
            support = token_support(selected, position)
            keys = cpu_k[0, support].float().repeat_interleave(6, dim=1).transpose(0, 1)
            values = (
                cpu_v[0, support].float().repeat_interleave(6, dim=1).transpose(0, 1)
            )
            logits = (cpu_q[0, position].float()[:, None] * keys).sum(-1) * 0.0625
            expected = (logits.softmax(-1)[:, :, None] * values).sum(1)
            output = torch.empty_like(q)
            kernels.call(
                "attend",
                [q, cache_k, cache_v, ids, partial, stats, output],
                [1, S, 0, position, 1, C],
            )
            prefill_error = relative(output[0, position].cpu(), expected)
            step_q, step_out = (
                q[:, position : position + 1].contiguous(),
                torch.empty_like(q[:, :1]),
            )
            kernels.call(
                "attend",
                [step_q, cache_k, cache_v, ids, partial, stats, step_out],
                [1, 1, position, 0, 1, C],
            )
            decode_error = relative(step_out[0, 0].cpu(), expected)
            assert max(prefill_error, decode_error) < 0.002, (
                position,
                prefill_error,
                decode_error,
            )
            result["rows"].append(
                {
                    "check": "direct_selected_attention",
                    "position": position,
                    "prefill_relative_rms": prefill_error,
                    "decode_relative_rms": decode_error,
                }
            )
        torch.cuda.synchronize()
        result["status"] = "passed"
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        path.write_text(json.dumps(result, indent=2) + "\n")
    print("Qwen kernel controls passed", flush=True)


if __name__ == "__main__":
    main()
