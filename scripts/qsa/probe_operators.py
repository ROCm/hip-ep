#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""GPU numerical/cache checks for the exact Original export operator contract."""
# isort: skip_file

import argparse
import json

from runtime import WORK, binding, run, session
import numpy as np
import onnx
import torch
from onnx import TensorProto as T, helper as h, numpy_helper as nh

from export_original import quantize


def make_model(name, nodes, inputs, outputs, constants):
    model = h.make_model(
        h.make_graph(
            nodes,
            name,
            inputs,
            outputs,
            [nh.from_array(v, k) for k, v in constants.items()],
        ),
        opset_imports=[h.make_opsetid("", 21), h.make_opsetid("com.microsoft", 1)],
        ir_version=10,
    )
    path = WORK / (name + ".onnx")
    onnx.save(model, path)
    return session(path)


def error(actual, expected):
    a, e = actual.float().cpu(), expected.float().cpu()
    return {
        "relative_l2": float((a - e).norm() / e.norm().clamp_min(1e-12)),
        "max_abs": float((a - e).abs().max()),
        "finite": bool(a.isfinite().all()),
    }


def matmul(large=False):
    torch.manual_seed(42)
    w = torch.randn(4096, 4096).bfloat16() * 0.02
    packed, scales = quantize(w)
    codes = np.empty((4096, 32, 128), dtype=np.uint8)
    codes[..., ::2], codes[..., 1::2] = packed & 15, packed >> 4
    decoded = (
        torch.from_numpy(
            (
                (codes.astype(np.float32) - 8) * scales.astype(np.float32)[..., None]
            ).reshape(4096, 4096)
        )
        .half()
        .cuda()
    )
    result = []
    sess = make_model(
        "matmul-w4a16-probe",
        [
            h.make_node(
                "MatMulNBits",
                ["x", "w", "s"],
                ["y"],
                domain="com.microsoft",
                K=4096,
                N=4096,
                bits=4,
                block_size=128,
                accuracy_level=0,
            )
        ],
        [h.make_tensor_value_info("x", T.FLOAT16, [1, "sequence", 4096])],
        [h.make_tensor_value_info("y", T.FLOAT16, [1, "sequence", 4096])],
        {"w": packed, "s": scales},
    )
    for t in [4096] if large else [1, 17, 128]:
        x = torch.randn((1, t, 4096), device="cuda", dtype=torch.float16)
        y = torch.empty_like(x)
        run(sess, binding(sess, {"x": x}, {"y": y}))
        stats = {"sequence": t, **error(y, x @ decoded.t())}
        assert stats["finite"] and stats["relative_l2"] < 0.005, stats
        result.append({"sequence": t, **stats})
    return result


def attention(window, large=False):
    cap, heads, width = 4160, 32, 128
    constants = {"norm": np.linspace(0.8, 1.2, 4096).astype(np.float16)}
    # No RoPE in this probe: test norm/attention/cache independently. The model
    # probe separately compares real YaRN against the pinned source.
    nodes = [
        h.make_node(
            "SimplifiedLayerNormalization",
            [kind, "norm"],
            [kind + "n"],
            axis=-1,
            epsilon=1e-6,
            stash_type=1,
        )
        for kind in ["q", "k"]
    ]
    nodes.append(
        h.make_node(
            "GroupQueryAttention",
            ["qn", "kn", "v", "pk", "pv", "lengths", "capacity"],
            ["y", "present_k", "present_v"],
            domain="com.microsoft",
            num_heads=heads,
            kv_num_heads=heads,
            do_rotary=0,
            scale=width**-0.5,
            local_window_size=window,
        )
    )
    inputs = [
        h.make_tensor_value_info(k, T.FLOAT16, [1, "sequence", 4096])
        for k in ["q", "k", "v"]
    ]
    inputs += [
        h.make_tensor_value_info(k, T.FLOAT16, [1, heads, cap, width])
        for k in ["pk", "pv"]
    ]
    inputs += [
        h.make_tensor_value_info("lengths", T.INT32, [1]),
        h.make_tensor_value_info("capacity", T.INT32, []),
    ]
    outputs = [h.make_tensor_value_info("y", T.FLOAT16, [1, "sequence", 4096])]
    outputs += [
        h.make_tensor_value_info(k, T.FLOAT16, [1, heads, cap, width])
        for k in ["present_k", "present_v"]
    ]
    sess = make_model(
        "mha-cache-probe-" + str(window), nodes, inputs, outputs, constants
    )
    rows = []
    for past, count in [(0, 4096)] if large else [(0, 17), (17, 1), (4097, 1)]:
        torch.manual_seed(past + 42)
        pk = torch.randn(1, heads, cap, width, dtype=torch.float16, device="cuda")
        pv = torch.randn_like(pk)
        pk_before, pv_before = pk.clone(), pv.clone()
        q, k, v = [
            torch.randn((1, count, 4096), dtype=torch.float16, device="cuda")
            for _ in range(3)
        ]
        scale = torch.from_numpy(constants["norm"]).cuda()

        def norm(t, scale=scale):
            f = t.float()
            return (
                f
                * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-6)
                * scale.float()
            ).half()

        qn, kn = [
            norm(t).reshape(1, count, heads, width).transpose(1, 2) for t in [q, k]
        ]
        vn = v.reshape(1, count, heads, width).transpose(1, 2)
        pk_before[:, :, past : past + count] = kn
        pv_before[:, :, past : past + count] = vn
        sampled = torch.arange(0, count, 127 if large else 1, device="cuda")
        scores = (
            qn[:, :, sampled].float()
            @ pk_before[:, :, : past + count].float().transpose(-1, -2)
            / width**0.5
        )
        qi = (sampled + past)[:, None]
        ki = torch.arange(past + count, device="cuda")[None]
        legal = ki <= qi
        if window > 0:
            legal &= ki > qi - window
        ref = (
            scores.masked_fill(~legal, -float("inf")).softmax(-1)
            @ pv_before[:, :, : past + count].float()
        ).half()
        ref = ref.transpose(1, 2).reshape(1, len(sampled), 4096)
        y = torch.empty_like(q)
        feed = {
            "q": q,
            "k": k,
            "v": v,
            "pk": pk,
            "pv": pv,
            "lengths": torch.tensor(
                [past + count - 1], dtype=torch.int32, device="cuda"
            ),
            "capacity": torch.tensor(cap, dtype=torch.int32, device="cuda"),
        }
        run(sess, binding(sess, feed, {"y": y, "present_k": pk, "present_v": pv}))
        stats = error(y[:, sampled], ref)
        cache = error(pk[:, :, : past + count], pk_before[:, :, : past + count])
        assert stats["finite"] and stats["relative_l2"] < 0.01, stats
        assert cache["relative_l2"] < 0.001, cache
        torch.testing.assert_close(pv, pv_before, rtol=0, atol=0)
        torch.testing.assert_close(
            pk[:, :, past + count :], pk_before[:, :, past + count :], rtol=0, atol=0
        )
        rows.append(
            {
                "past": past,
                "new": count,
                "output": stats,
                "cache": cache,
                "past_present_aliased": True,
                "untouched_cache_suffix": True,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--large-prefill", action="store_true")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    result = {"status": "running", "synthetic_operator_tests_only": True}
    out = WORK / (
        "operator-probe-large-prefill.json"
        if args.large_prefill
        else "operator-probe.json"
    )
    try:
        result["matmul"] = matmul(args.large_prefill)
        print("INT4 projections passed", flush=True)
        result["global"] = attention(-1, args.large_prefill)
        print("MHA global attention/cache passed", flush=True)
        result["sliding"] = attention(4096, args.large_prefill)
        result["status"] = "passed"
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
