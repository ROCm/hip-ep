#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric checks for attention merged into one rocMLIR kernel.

Each test turns on HIPDNN_EP_ROCMLIR=1, where hip-fuse-rocmlir-anchor merges a
gemm kernel, the last-axis softmax of its result and the gemm kernel that reads
the softmax into one kernel, which rocMLIR lowers to rock.attention. The LIT
test hip-fuse-rocmlir-attention.mlir only checks which ops end up in the
kernel; these tests check that the merged kernel computes
softmax(Q Kᵀ * scale + bias) V, and the fusion report
(HIPDNN_EP_ROCMLIR_FUSION_REPORT) shows whether each softmax was merged into a
kernel or stayed a host op.

The merge is gated, so the cases come in pairs that do and don't merge:
- f16 always merges (head dims up to 256), also with an f32 softmax between
  casts.
- f32 without matrix acceleration merges only when the score matrix is larger
  than HIPDNN_EP_ROCMLIR_ATTENTION_F32_MIN_SCORE_BYTES, pinned here to 2 MiB;
  an f32 the target accelerates merges at any size.
- A head dim over 256, probabilities that are also a graph output, or
  HIPDNN_EP_ROCMLIR_NO_ATTENTION=1 keep the softmax apart.
- Two identical attention blocks merge into two dispatches, and a score gemm
  repeated without a softmax after it stays a plain gemm kernel.

The reference is fp64 numpy rather than ORT CPU on the same graph: ORT CPU runs
an fp16 softmax in fp16, which drifts further from the exact answer than the
fused kernel does.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type

_TOL = {
    np.float16: dict(atol=5e-3, rtol=1e-2, cos_threshold=0.9999),
    np.float32: dict(atol=1e-4, rtol=1e-3, cos_threshold=0.99999),
}

# Present only when LibHipCompiler is built with ENABLE_ROCMLIRTRITON; a build
# without it ignores HIPDNN_EP_ROCMLIR and would pass on the HIP kernels.
_ROCMLIR_BUILD_MARKER = b"rocMLIR kernel compilation failed"

_F32_MIN_SCORE_BYTES = 2 * 1024 * 1024


def _dll_contains(path: Path, needle: bytes) -> bool:
    if not path.is_file():
        return False
    overlap = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                return False
            if needle in overlap + chunk:
                return True
            overlap = chunk[1 - len(needle) :]


class _RocMlirSession:
    """Runs models with the rocMLIR path on and reads back its fusion."""

    def __init__(self, model_runner, monkeypatch, root: Path):
        self._runner = model_runner
        self._monkeypatch = monkeypatch
        self._root = root
        self._runs = 0
        self.records: list[dict] = []
        self.accel_types: list[str] = []

    def run(self, model, feeds):
        self._runs += 1
        report = self._root / f"report{self._runs}.jsonl"
        dump = self._root / f"pre-fusion{self._runs}"
        self._monkeypatch.setenv("HIPDNN_EP_ROCMLIR_FUSION_REPORT", str(report))
        self._monkeypatch.setenv("HIPDNN_EP_ROCMLIR_PRE_FUSION_DUMP", str(dump))
        path = self._runner._next_subdir() / "model.onnx"
        path.write_bytes(model.SerializeToString())
        outputs = self._runner.backend.run(str(path), feeds)
        if not report.is_file():
            pytest.fail("the rocMLIR fusion wrote no report, so it did not run")
        self.records = [
            json.loads(line) for line in report.read_text().splitlines() if line
        ]
        header = next(iter(sorted(dump.glob("*.mlir"))), None)
        self.accel_types = []
        if header is not None:
            first = header.read_text(errors="replace").splitlines()[0]
            for field in first.split():
                if field.startswith("gemm-gemm-accel-types="):
                    self.accel_types = [t for t in field.split("=", 1)[1].split(",") if t]
        return outputs

    def softmax_counts(self) -> tuple[int, int]:
        """(merged kernels holding a softmax, softmaxes left on the host)."""
        merged = sum(
            1
            for r in self.records
            if r.get("kind") == "kernel" and "hip.miopen.softmax" in r.get("ops", {})
        )
        host = sum(
            r.get("ops", {}).get("hip.miopen.softmax", 0)
            for r in self.records
            if r.get("kind") == "host"
        )
        return merged, host

    def expect(self, merged: int, host: int):
        actual = self.softmax_counts()
        assert actual == (merged, host), (
            f"expected {merged} merged attention kernel(s) and {host} host "
            f"softmax(es), the fusion report shows {actual}"
        )


@pytest.fixture
def rocmlir(model_runner, monkeypatch, request, tmp_path):
    dll = request.config.getoption("--ep-dll")
    if not dll or not _dll_contains(Path(dll), _ROCMLIR_BUILD_MARKER):
        pytest.skip(
            "EP was built without ENABLE_ROCMLIRTRITON; "
            "HIPDNN_EP_ROCMLIR would not run the attention merge"
        )
    monkeypatch.setenv("HIPDNN_EP_ROCMLIR", "1")
    monkeypatch.setenv(
        "HIPDNN_EP_ROCMLIR_ATTENTION_F32_MIN_SCORE_BYTES", str(_F32_MIN_SCORE_BYTES)
    )
    monkeypatch.delenv("HIPDNN_EP_ROCMLIR_NO_ATTENTION", raising=False)
    return _RocMlirSession(model_runner, monkeypatch, tmp_path)


def _attention_nodes(
    prefix, q, k, v, out, *, bias=None, k_layout="bhds", softmax_f32=False
):
    """Nodes for ``softmax(q @ kᵀ * scale [+ bias]) @ v``.

    ``k_layout="bhds"`` feeds K already transposed, so the score gemm reads it
    directly; ``"bhsd"`` puts a Transpose in front, which the outliner pulls
    into the gemm as its transB. ``softmax_f32`` casts the f16 logits to f32
    for the softmax and the probabilities back to f16.
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
    if softmax_f32:
        nodes.append(
            helper.make_node(
                "Cast", [logits], [f"{prefix}l32"], to=TensorProto.FLOAT
            )
        )
        logits = f"{prefix}l32"
    probs = f"{prefix}p"
    nodes.append(helper.make_node("Softmax", [logits], [probs], axis=-1))
    if softmax_f32:
        nodes.append(
            helper.make_node("Cast", [probs], [f"{prefix}p16"], to=TensorProto.FLOAT16)
        )
        probs = f"{prefix}p16"
    nodes.append(helper.make_node("MatMul", [probs, v], [out]))
    return nodes


def _scale_init(dtype, head_dim):
    return helper.make_tensor(
        "scale", np_to_onnx_type(dtype), [], [1.0 / np.sqrt(head_dim)]
    )


def _make_model(
    dtype,
    b,
    h,
    s,
    d,
    *,
    bias=False,
    k_layout="bhds",
    probs_out=False,
    softmax_f32=False,
):
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
        softmax_f32=softmax_f32,
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


def _check(rocmlir, dtype, b, h, s, d, **kw):
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
    actual = rocmlir.run(model, feeds)
    y, p = _reference(q, k, v, mask)
    expected = [y, p] if kw.get("probs_out") else [y]
    compare_outputs(actual, expected, **_TOL[dtype])


def _expect_merged(rocmlir):
    rocmlir.expect(merged=1, host=0)


def _expect_apart(rocmlir):
    rocmlir.expect(merged=0, host=1)


class TestAttentionFusionF16:
    """f16 has matrix acceleration, so these all merge into one kernel."""

    @pytest.mark.parametrize(
        "s,d", [(64, 32), (128, 64), (256, 64), (128, 128), (64, 256)]
    )
    def test_scaled(self, rocmlir, s, d):
        _check(rocmlir, np.float16, 1, 4, s, d)
        _expect_merged(rocmlir)

    @pytest.mark.parametrize("s", [64, 200])
    def test_mask_bias(self, rocmlir, s):
        """A broadcast additive mask between the scale and the softmax."""
        _check(rocmlir, np.float16, 2, 4, s, 64, bias=True)
        _expect_merged(rocmlir)

    def test_transposed_k(self, rocmlir):
        """K arrives BHSD and a Transpose feeds the score gemm."""
        _check(rocmlir, np.float16, 1, 4, 128, 64, k_layout="bhsd")
        _expect_merged(rocmlir)

    def test_batched_heads(self, rocmlir):
        _check(rocmlir, np.float16, 2, 12, 128, 64)
        _expect_merged(rocmlir)

    def test_softmax_in_f32(self, rocmlir):
        """f16 gemms around an f32 softmax: Cast to f32, Softmax, Cast to f16.
        The f32 softmax must keep at least the f16 accuracy."""
        _check(rocmlir, np.float16, 1, 4, 128, 64, bias=True, softmax_f32=True)
        _expect_merged(rocmlir)


class TestAttentionFusionF32:
    """f32 without matrix acceleration merges only above the score cutoff."""

    def _expect(self, rocmlir, score_bytes):
        if "f32" in rocmlir.accel_types or score_bytes > _F32_MIN_SCORE_BYTES:
            _expect_merged(rocmlir)
        else:
            _expect_apart(rocmlir)

    def test_large_scores(self, rocmlir):
        """1x16x1024x1024 f32 scores are 64 MiB, above the cutoff."""
        _check(rocmlir, np.float32, 1, 16, 1024, 64)
        self._expect(rocmlir, 16 * 1024 * 1024 * 4)

    def test_small_scores(self, rocmlir):
        """1x2x64x64 f32 scores are 32 KiB, below the cutoff."""
        _check(rocmlir, np.float32, 1, 2, 64, 32)
        self._expect(rocmlir, 2 * 64 * 64 * 4)


class TestAttentionFusionKeptApart:
    """Graphs the merge must leave alone still compute the same answer."""

    def test_head_dim_over_limit(self, rocmlir):
        _check(rocmlir, np.float16, 1, 2, 64, 512)
        _expect_apart(rocmlir)

    def test_probs_also_output(self, rocmlir):
        _check(rocmlir, np.float16, 1, 4, 64, 64, probs_out=True)
        _expect_apart(rocmlir)

    def test_no_attention(self, rocmlir, monkeypatch):
        """HIPDNN_EP_ROCMLIR_NO_ATTENTION=1 runs the same graph unmerged."""
        monkeypatch.setenv("HIPDNN_EP_ROCMLIR_NO_ATTENTION", "1")
        _check(rocmlir, np.float16, 1, 4, 128, 64, bias=True)
        _expect_apart(rocmlir)


class TestAttentionFusionRepeatedBlocks:
    """Several blocks of the same shape in one graph."""

    def test_two_identical_blocks(self, rocmlir):
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
        actual = rocmlir.run(model, feeds)
        compare_outputs(actual, expected, **_TOL[dtype])
        # Identical blocks may share one kernel, but neither softmax stays.
        merged, host = rocmlir.softmax_counts()
        assert merged >= 1 and host == 0, (merged, host)

    def test_score_gemm_shared_with_plain_matmul(self, rocmlir):
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
        actual = rocmlir.run(model, feeds + [q2])
        y, _ = _reference(q, k, v, None)
        s2 = (
            q2.astype(np.float64) @ k.astype(np.float64).transpose(0, 1, 3, 2)
        ) / np.sqrt(d)
        compare_outputs(actual, [y, s2], **_TOL[dtype])
        _expect_merged(rocmlir)
