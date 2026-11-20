#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric checks for attention merged into one rocMLIR kernel.

With HIPDNN_EP_ROCMLIR=1, hip-fuse-rocmlir-anchor merges a gemm kernel, the
last-axis softmax of its result and the gemm kernel that reads the softmax into
one kernel, which rocMLIR lowers to rock.attention. The LIT test
hip-fuse-rocmlir-attention.mlir only checks which ops end up in the kernel;
these tests check that the merged kernel computes softmax(Q Kᵀ * scale + bias) V.

The merge is gated, so the cases come in pairs that do and don't merge:
- f16 always merges (head dims up to 256).
- f32 merges only when the score matrix is larger than the device's L2 cache,
  so there is one f32 case well above any L2 and one well below.
- A head dim over 256, or probabilities that are also a graph output, keep the
  softmax apart.
- Two identical attention blocks merge into two dispatches, and a score gemm
  repeated without a softmax after it stays a plain gemm kernel.

Without HIPDNN_EP_ROCMLIR=1 the same graphs run on the EP's own gemm and
softmax kernels and must give the same answers. Comparing a run with
HIPDNN_EP_ROCMLIR_NO_ATTENTION=1 isolates the merged kernel.

The reference is fp64 numpy rather than ORT CPU on the same graph: ORT CPU runs
an fp16 softmax in fp16, which drifts further from the exact answer than the
fused kernel does.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type

_TOL = {
    np.float16: dict(atol=5e-3, rtol=1e-2, cos_threshold=0.9999),
    np.float32: dict(atol=1e-4, rtol=1e-3, cos_threshold=0.99999),
}


def _attention_nodes(prefix, q, k, v, out, *, bias=None, k_layout="bhds"):
    """Nodes for ``softmax(q @ kᵀ * scale [+ bias]) @ v``.

    ``k_layout="bhds"`` feeds K already transposed, so the score gemm reads it
    directly; ``"bhsd"`` puts a Transpose in front, which the outliner pulls
    into the gemm as its transB.
    """
    nodes = []
    if k_layout == "bhsd":
        nodes.append(
            helper.make_node("Transpose", [k], [f"{prefix}kt"], perm=[0, 1, 3, 2])
        )
        k = f"{prefix}kt"
    nodes.append(helper.make_node("MatMul", [q, k], [f"{prefix}s"]))
    nodes.append(helper.make_node("Mul", [f"{prefix}s", "scale"], [f"{prefix}m"]))
    logits = f"{prefix}m"
    if bias is not None:
        nodes.append(helper.make_node("Add", [logits, bias], [f"{prefix}b"]))
        logits = f"{prefix}b"
    nodes.append(helper.make_node("Softmax", [logits], [f"{prefix}p"], axis=-1))
    nodes.append(helper.make_node("MatMul", [f"{prefix}p", v], [out]))
    return nodes


def _scale_init(dtype, head_dim):
    return helper.make_tensor(
        "scale", np_to_onnx_type(dtype), [], [1.0 / np.sqrt(head_dim)]
    )


def _make_model(dtype, b, h, s, d, *, bias=False, k_layout="bhds", probs_out=False):
    tp = np_to_onnx_type(dtype)
    k_shape = [b, h, d, s] if k_layout == "bhds" else [b, h, s, d]
    inputs = [
        helper.make_tensor_value_info("q", tp, [b, h, s, d]),
        helper.make_tensor_value_info("k", tp, k_shape),
        helper.make_tensor_value_info("v", tp, [b, h, s, d]),
    ]
    if bias:
        inputs.append(helper.make_tensor_value_info("mask", tp, [b, 1, 1, s]))
    outputs = [helper.make_tensor_value_info("y", tp, [b, h, s, d])]
    if probs_out:
        outputs.append(helper.make_tensor_value_info("p", tp, [b, h, s, s]))
    nodes = _attention_nodes(
        "",
        "q",
        "k",
        "v",
        "y",
        bias="mask" if bias else None,
        k_layout=k_layout,
    )
    return make_model_from_nodes(
        nodes, inputs, outputs, initializers=[_scale_init(dtype, d)]
    )


def _inputs(dtype, b, h, s, d, *, bias=False, k_layout="bhds", seed=0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((b, h, s, d)).astype(dtype)
    k = rng.standard_normal((b, h, s, d)).astype(dtype)
    v = rng.standard_normal((b, h, s, d)).astype(dtype)
    if k_layout == "bhds":
        k_feed = np.ascontiguousarray(k.transpose(0, 1, 3, 2))
    else:
        k_feed = k
    feeds = [q, k_feed, v]
    mask = None
    if bias:
        # Mask out the last quarter of the keys the way a padding mask does.
        mask = np.zeros((b, 1, 1, s), dtype=dtype)
        mask[..., s - s // 4 :] = -10000.0
        feeds.append(mask)
    return feeds, (q, k, v, mask)


def _reference(q, k, v, mask):
    """fp64 ``softmax(q kᵀ / sqrt(d) + mask) v`` and the probabilities."""
    q, k, v = (t.astype(np.float64) for t in (q, k, v))
    logits = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(q.shape[-1])
    if mask is not None:
        logits = logits + mask.astype(np.float64)
    logits -= logits.max(axis=-1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=-1, keepdims=True)
    return p @ v, p


def _run(model_runner, model, feeds):
    """Run *model* on the EP from a fresh per-test work subdir."""
    path = model_runner._next_subdir() / "model.onnx"
    path.write_bytes(model.SerializeToString())
    return model_runner.backend.run(str(path), feeds)


def _check(model_runner, dtype, b, h, s, d, **kw):
    model = _make_model(dtype, b, h, s, d, **kw)
    feeds, (q, k, v, mask) = _inputs(
        dtype,
        b,
        h,
        s,
        d,
        bias=kw.get("bias", False),
        k_layout=kw.get("k_layout", "bhds"),
        seed=s * 1000 + d,
    )
    actual = _run(model_runner, model, feeds)
    y, p = _reference(q, k, v, mask)
    expected = [y, p] if kw.get("probs_out") else [y]
    compare_outputs(actual, expected, **_TOL[dtype])


class TestAttentionFusionF16:
    """f16 has matrix acceleration, so these all merge into one kernel."""

    @pytest.mark.parametrize(
        "s,d", [(64, 32), (128, 64), (256, 64), (128, 128), (64, 256)]
    )
    def test_scaled(self, model_runner, s, d):
        _check(model_runner, np.float16, 1, 4, s, d)

    @pytest.mark.parametrize("s", [64, 200])
    def test_mask_bias(self, model_runner, s):
        """A broadcast additive mask between the scale and the softmax."""
        _check(model_runner, np.float16, 2, 4, s, 64, bias=True)

    def test_transposed_k(self, model_runner):
        """K arrives BHSD and a Transpose feeds the score gemm."""
        _check(model_runner, np.float16, 1, 4, 128, 64, k_layout="bhsd")

    def test_batched_heads(self, model_runner):
        _check(model_runner, np.float16, 2, 12, 128, 64)


class TestAttentionFusionF32:
    """f32 merges only when the score matrix is larger than the L2 cache."""

    def test_large_scores(self, model_runner):
        """1x16x1024x1024 f32 scores are 64 MiB, well above a GPU's L2."""
        _check(model_runner, np.float32, 1, 16, 1024, 64)

    def test_small_scores(self, model_runner):
        """1x2x64x64 f32 scores are 32 KiB, below any L2, so they stay apart."""
        _check(model_runner, np.float32, 1, 2, 64, 32)


class TestAttentionFusionKeptApart:
    """Graphs the merge must leave alone still compute the same answer."""

    def test_head_dim_over_limit(self, model_runner):
        _check(model_runner, np.float16, 1, 2, 64, 512)

    def test_probs_also_output(self, model_runner):
        _check(model_runner, np.float16, 1, 4, 64, 64, probs_out=True)


class TestAttentionFusionRepeatedBlocks:
    """Several blocks of the same shape in one graph."""

    def test_two_identical_blocks(self, model_runner):
        dtype, b, h, s, d = np.float16, 1, 4, 128, 64
        tp = np_to_onnx_type(dtype)
        names = ("q0", "k0", "v0", "q1", "k1", "v1")
        shapes = ([b, h, s, d], [b, h, d, s], [b, h, s, d]) * 2
        inputs = [
            helper.make_tensor_value_info(n, tp, sh) for n, sh in zip(names, shapes)
        ]
        outputs = [
            helper.make_tensor_value_info(n, tp, [b, h, s, d]) for n in ("y0", "y1")
        ]
        nodes = _attention_nodes("a_", "q0", "k0", "v0", "y0") + _attention_nodes(
            "b_", "q1", "k1", "v1", "y1"
        )
        model = make_model_from_nodes(
            nodes, inputs, outputs, initializers=[_scale_init(dtype, d)]
        )

        feeds, expected = [], []
        for seed in (1, 2):
            f, (q, k, v, _) = _inputs(dtype, b, h, s, d, seed=seed)
            feeds += f
            expected.append(_reference(q, k, v, None)[0])
        actual = _run(model_runner, model, feeds)
        compare_outputs(actual, expected, **_TOL[dtype])

    def test_score_gemm_shared_with_plain_matmul(self, model_runner):
        """A second Q Kᵀ * scale with no softmax after it, reading the same K,
        stays a gemm kernel next to the merged attention."""
        dtype, b, h, s, d = np.float16, 1, 4, 128, 64
        tp = np_to_onnx_type(dtype)
        inputs = [
            helper.make_tensor_value_info("q", tp, [b, h, s, d]),
            helper.make_tensor_value_info("k", tp, [b, h, d, s]),
            helper.make_tensor_value_info("v", tp, [b, h, s, d]),
            helper.make_tensor_value_info("q2", tp, [b, h, s, d]),
        ]
        outputs = [
            helper.make_tensor_value_info("y", tp, [b, h, s, d]),
            helper.make_tensor_value_info("s2", tp, [b, h, s, s]),
        ]
        nodes = _attention_nodes("", "q", "k", "v", "y") + [
            helper.make_node("MatMul", ["q2", "k"], ["mm2"]),
            helper.make_node("Mul", ["mm2", "scale"], ["s2"]),
        ]
        model = make_model_from_nodes(
            nodes, inputs, outputs, initializers=[_scale_init(dtype, d)]
        )

        feeds, (q, k, v, _) = _inputs(dtype, b, h, s, d, seed=3)
        q2 = np.random.default_rng(4).standard_normal((b, h, s, d)).astype(dtype)
        actual = _run(model_runner, model, feeds + [q2])
        y, _ = _reference(q, k, v, None)
        s2 = (
            q2.astype(np.float64) @ k.astype(np.float64).transpose(0, 1, 3, 2)
        ) / np.sqrt(d)
        compare_outputs(actual, [y, s2], **_TOL[dtype])
