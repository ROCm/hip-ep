#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Constant single-index Gather feeding a rocMLIR kernel, with rocMLIR forced on.

hip-rocmlir-pre-fusion turns the Gather into a tensor.extract_slice (plus a
tensor.collapse_shape for a scalar index), and the rocMLIR fusion reads that
view inside the MatMul kernel or the standalone Add -> Neg kernel, so no
gather kernel runs. The reference is ONNX Runtime's CPU EP running the Gather.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

# Present only when LibHipCompiler is built with ENABLE_ROCMLIRTRITON.
# CompilerDriver.cpp emits it from the rocMLIR arm; a build without that
# arm ignores HIPDNN_EP_ROCMLIR and would pass on the gather kernel.
_ROCMLIR_BUILD_MARKER = b"rocMLIR kernel compilation failed"

_ONNX_TYPE = {np.float16: TensorProto.FLOAT16, np.float32: TensorProto.FLOAT}


def _gather_shape(data_shape, axis, index_shape):
    return data_shape[:axis] + index_shape + data_shape[axis + 1 :]


def _make_model(data_shape, axis, index, index_shape, consumer, dtype, seed):
    """Gather(X, constant index) -> MatMul(G, W), or -> Neg(Add(G, B))."""
    rng = np.random.default_rng(seed)
    onnx_type = _ONNX_TYPE[dtype]
    norm_axis = axis % len(data_shape)
    g_shape = _gather_shape(data_shape, norm_axis, index_shape)
    index_init = helper.make_tensor(
        "idx", TensorProto.INT64, index_shape, [index] * max(1, len(index_shape))
    )
    gather = helper.make_node("Gather", ["X", "idx"], ["G"], axis=axis)
    if consumer == "matmul":
        w = rng.uniform(-0.5, 0.5, [g_shape[-1], 16]).astype(dtype)
        inits = [index_init, helper.make_tensor("W", onnx_type, w.shape, w.ravel())]
        nodes = [gather, helper.make_node("MatMul", ["G", "W"], ["Y"])]
        y_shape = g_shape[:-1] + [16]
    else:
        b = rng.uniform(-0.5, 0.5, g_shape).astype(dtype)
        inits = [index_init, helper.make_tensor("B", onnx_type, b.shape, b.ravel())]
        nodes = [
            gather,
            helper.make_node("Add", ["G", "B"], ["S"]),
            helper.make_node("Neg", ["S"], ["Y"]),
        ]
        y_shape = g_shape
    x_info = helper.make_tensor_value_info("X", onnx_type, data_shape)
    y_info = helper.make_tensor_value_info("Y", onnx_type, y_shape)
    return make_model_from_nodes(nodes, [x_info], [y_info], initializers=inits)


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


def _require_rocmlir_build(request) -> None:
    dll = request.config.getoption("--ep-dll")
    if not dll or not _dll_contains(Path(dll), _ROCMLIR_BUILD_MARKER):
        pytest.skip(
            "EP was built without ENABLE_ROCMLIRTRITON; "
            "HIPDNN_EP_ROCMLIR would not run the gather rewrite"
        )


def _assert_gather_absorbed(dump_dir: Path) -> None:
    dumps = sorted(dump_dir.glob("*.mlir"))
    if not dumps:
        pytest.fail(
            "HIPDNN_EP_IR_DUMP_PATH produced no IR, so the slice rewrite was not confirmed"
        )
    ir = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in dumps)
    # The tail dump starts after rocMLIR fusion. An absorbed gather is a slice
    # inside the kernel, whose name lists its ops, and no gather op.
    if "@mlir_extract_slice" not in ir:
        pytest.fail(
            "no rocMLIR kernel reads the slice, so the gather rewrite was not exercised"
        )
    if "hip.gather" in ir or "hip_gather" in ir:
        pytest.fail("compiled IR still contains a gather")


@pytest.mark.parametrize("dtype", [np.float16, np.float32], ids=["f16", "f32"])
@pytest.mark.parametrize(
    "data_shape,axis,index,index_shape,consumer",
    [
        # Scalar index into the innermost axis: a strided slice plus collapse.
        ([32, 64, 2], -1, 1, [], "matmul"),
        # Negative scalar index on a middle axis.
        ([32, 4, 64], 1, -1, [], "matmul"),
        # Scalar index on the outermost axis: a contiguous slice.
        ([3, 32, 64], 0, 2, [], "matmul"),
        # One-element 1-D index keeps the unit dim: a slice only.
        ([5, 32, 64], 0, 2, [1], "add_neg"),
        # Innermost-axis scalar index into a standalone elementwise kernel.
        ([64, 32, 2], 2, 0, [], "add_neg"),
    ],
    ids=[
        "innermost_matmul",
        "neg_middle_matmul",
        "outer_matmul",
        "one_elem_add_neg",
        "innermost_add_neg",
    ],
)
def test_gather_slice_rocmlir(
    model_runner,
    monkeypatch,
    request,
    data_shape,
    axis,
    index,
    index_shape,
    consumer,
    dtype,
):
    _require_rocmlir_build(request)
    monkeypatch.setenv("HIPDNN_EP_ROCMLIR", "1")
    with tempfile.TemporaryDirectory(prefix="gather-slice-rocmlir-") as dump_dir:
        monkeypatch.setenv("HIPDNN_EP_IR_DUMP_PATH", str(Path(dump_dir) / "ir.mlir"))
        monkeypatch.setenv("HIPDNN_EP_IR_DUMP_AFTER_ONLY", "1")
        model = _make_model(
            data_shape, axis, index, index_shape, consumer, dtype, seed=7
        )
        x = np.random.default_rng(3).uniform(-1, 1, data_shape).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x])
        _assert_gather_absorbed(Path(dump_dir))
    assert np.isfinite(actual[0]).all()
    tol = 2e-2 if dtype == np.float16 else 1e-4
    compare_outputs(actual, expected, atol=tol, rtol=tol)
