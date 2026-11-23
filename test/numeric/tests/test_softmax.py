#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric tests for standalone ONNX Softmax -> hip.miopen.softmax.

The runtime picks one of four kernels from the row width: a wavefront per row
(exact vector widths, or masked for any width up to three values per lane), a
256-thread block per row with the row in registers up to 8192, a 1024-thread
block up to 32768, and the three-pass fallback past that. The widths below sit
on each side of every dispatch boundary, plus widths whose byte size selects
16-byte, 8-byte or scalar accesses (4096, 1500, 2049).

Reference strategy: a numpy fp64 softmax of the exact input values, not ORT
CPU, because ORT CPU accumulates fp16 softmax in fp16 and has no bf16 Softmax.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest
from onnx import helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type

# (atol, rtol) per storage type; outputs are probabilities in [0, 1].
TOLERANCE = {
    np.float32: (1e-6, 1e-4),
    np.float16: (1e-3, 1e-2),
    ml_dtypes.bfloat16: (8e-3, 2e-2),
}

BOUNDARY_COLS = [
    1,
    7,
    20,
    32,
    33,
    77,
    128,
    197,
    256,
    257,
    512,
    513,
    577,
    1000,
    1024,
    1025,
    1500,
    2048,
    2049,
    4096,
    8192,
    8193,
    14400,
    16384,
    16385,
    32768,
    32769,
    40000,
]


def _make_softmax_model(shape, dtype, axis=None):
    onnx_type = np_to_onnx_type(dtype)
    x_info = helper.make_tensor_value_info("x", onnx_type, shape)
    y_info = helper.make_tensor_value_info("y", onnx_type, shape)
    attrs = {} if axis is None else {"axis": axis}
    node = helper.make_node("Softmax", ["x"], ["y"], **attrs)
    return make_model_from_nodes([node], [x_info], [y_info], opset=17)


def _softmax_ref(x, axis=-1):
    x = x.astype(np.float64)
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def _random_input(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-8.0, 8.0, size=shape).astype(dtype)


def _persist(model_runner, model):
    sub = model_runner._next_subdir()
    path = sub / "model.onnx"
    path.write_bytes(model.SerializeToString())
    return str(path)


def _run_and_compare(model_runner, x, axis=-1, model_axis=-1):
    dtype = x.dtype.type
    model = _make_softmax_model(list(x.shape), dtype, model_axis)
    actual = model_runner.backend.run(_persist(model_runner, model), [x])
    expected = [_softmax_ref(x, axis).astype(np.float32)]
    atol, rtol = TOLERANCE[dtype]
    compare_outputs(
        [actual[0].astype(np.float32)],
        expected,
        atol=atol,
        rtol=rtol,
        cos_threshold=0.9999,
    )


class TestSoftmax:
    @pytest.mark.parametrize("dtype", [np.float16, np.float32])
    @pytest.mark.parametrize("cols", BOUNDARY_COLS)
    def test_last_axis_widths(self, model_runner, dtype, cols):
        x = _random_input([5, cols], dtype, seed=cols)
        _run_and_compare(model_runner, x)

    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ([4096, 49], np.float32),
            ([2048, 197], np.float16),
            ([256, 4096], np.float32),
            ([64, 14400], np.float16),
        ],
    )
    def test_many_rows(self, model_runner, shape, dtype):
        x = _random_input(shape, dtype, seed=shape[-1])
        _run_and_compare(model_runner, x)

    # The masked wavefront kernel packs several rows into one block, and the
    # block size comes from the device's wave size. A kernel that took its
    # rows per block from a different wave size than the launcher used would
    # leave some rows of a block unwritten, so these row counts sit on and
    # off every multiple of 8 and of 4, with widths below one wave, within it
    # and past it. Each row is compared, so an unwritten row fails the test.
    @pytest.mark.parametrize("dtype", [np.float16, np.float32])
    @pytest.mark.parametrize(
        "rows,cols",
        [
            (8, 77),
            (9, 77),
            (16, 77),
            (4, 77),
            (5, 33),
            (7, 197),
            (17, 300),
            (33, 512),
            (8, 513),
            (13, 1024),
        ],
    )
    def test_every_row_written(self, model_runner, dtype, rows, cols):
        x = _random_input([rows, cols], dtype, seed=rows * 1000 + cols)
        _run_and_compare(model_runner, x)

    # 8193 and 30000 stay on the block kernel (cutoff 32768). 32769 and 40000
    # are past it, so they take the three-pass fallback. Both paths keep the
    # exponents in fp32; the bound below is what checks that.
    @pytest.mark.parametrize("cols", [8193, 30000, 32769, 40000])
    def test_fp16_fallback_rounds_once(self, model_runner, cols):
        # Each fp16 output is the exact softmax rounded once: within half an
        # fp16 step, 2^-11 relative, or 2^-25 among the subnormals. A stored
        # fp16 exponent would round twice and miss this bound on many elements.
        x = _random_input([5, cols], np.float16, seed=cols)
        model = _make_softmax_model(list(x.shape), np.float16, -1)
        actual = model_runner.backend.run(_persist(model_runner, model), [x])
        compare_outputs(
            [actual[0].astype(np.float64)],
            [_softmax_ref(x)],
            atol=1.05 * 2.0**-25,
            rtol=1.2 * 2.0**-11,
            cos_threshold=0.9999,
        )

    # A skip, not an xfail: with bf16 typed as f32 the kernel reads and writes
    # twice the buffer sizes, which can fault or corrupt other allocations.
    @pytest.mark.skip(
        reason="the MorphiZen importer maps ONNX BFLOAT16 to f32 "
        "(onnxElementTypeToMlirElementType), so bf16 data is read as f32",
    )
    @pytest.mark.parametrize("cols", [300, 1500, 4096, 8193])
    def test_bf16(self, model_runner, cols):
        # ORT's Python binding has no numpy bf16 type, so the bf16 tensors are
        # passed as raw 16-bit buffers through IO binding. A Cast around the
        # Softmax instead would let the compiler fold the casts and run fp32.
        import onnxruntime as ort

        bf16 = np_to_onnx_type(ml_dtypes.bfloat16)
        shape = [5, cols]
        x = _random_input(shape, ml_dtypes.bfloat16, seed=cols)
        model = _make_softmax_model(shape, ml_dtypes.bfloat16, -1)
        sess = model_runner.backend.create_session(_persist(model_runner, model))
        out_bits = np.zeros(shape, dtype=np.uint16)
        binding = sess.io_binding()
        binding.bind_ortvalue_input(
            "x",
            ort.OrtValue.ortvalue_from_numpy_with_onnx_type(x.view(np.uint16), bf16),
        )
        binding.bind_ortvalue_output(
            "y", ort.OrtValue.ortvalue_from_numpy_with_onnx_type(out_bits, bf16)
        )
        sess.run_with_iobinding(binding)
        actual = out_bits.view(ml_dtypes.bfloat16).astype(np.float32)
        atol, rtol = TOLERANCE[ml_dtypes.bfloat16]
        compare_outputs(
            [actual],
            [_softmax_ref(x).astype(np.float32)],
            atol=atol,
            rtol=rtol,
            cos_threshold=0.9999,
        )

    @pytest.mark.parametrize(
        "shape,axis",
        [
            ([2, 3, 1500], 2),
            ([2, 3, 1500], -1),
            ([1, 2, 4, 4096], 3),
            ([7, 1000], None),  # default axis on rank 2: the last axis in every opset
        ],
    )
    def test_ranks(self, model_runner, shape, axis):
        x = _random_input(shape, np.float16, seed=len(shape))
        _run_and_compare(model_runner, x, model_axis=axis)

    def test_fully_masked_row(self, model_runner):
        # A row of -inf has no defined softmax; the result is NaN, the same as
        # an fp32 reference. The other rows must be unaffected.
        x = _random_input([3, 2049], np.float32, seed=3)
        x[1, :] = -np.inf
        model = _make_softmax_model(list(x.shape), np.float32, -1)
        y = model_runner.backend.run(_persist(model_runner, model), [x])[0]
        assert np.isnan(y[1]).all()
        ref = _softmax_ref(x[[0, 2]]).astype(np.float32)
        compare_outputs([y[[0, 2]]], [ref], atol=1e-6, rtol=1e-4)

    def test_non_last_axis_not_claimed(self, model_runner, monkeypatch):
        # The kernel normalizes the last axis only. A non-last axis must not
        # compile for the GPU; with CPU fallback disabled the session then
        # fails instead of returning a last-axis softmax. Strict mode would
        # abort the process on that compile failure, so it is cleared here.
        monkeypatch.delenv("HIPDNN_EP_STRICT", raising=False)
        x = _random_input([2, 3, 64], np.float32, seed=11)
        model = _make_softmax_model(list(x.shape), np.float32, 1)
        with pytest.raises(
            Exception, match="(?i)fallback|no execution provider|assign"
        ):
            model_runner.backend.run(_persist(model_runner, model), [x])
