#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the GatherBlockQuantized custom op (com.microsoft) with
FLOAT8E4M3FN data.

Qwen3.8-Flash-Next engram.onnx gathers 16 rows per token from a
[320001536, 160] FLOAT8E4M3FN table with block_size = 0 and a single
[1, 1] fp16 scale.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

SEQ_LENS = [1, 128]

ENGRAM_ROW = 160
ENGRAM_HEADS = 16

FLOAT8_NAN_CODES = (0x7F, 0xFF)


def _float8_codes(rng, shape):
    codes = rng.integers(0, 256, shape, dtype=np.uint8)
    for nan_code in FLOAT8_NAN_CODES:
        codes[codes == nan_code] = 0
    return codes


def _make_gbq_float8_model(
    codes: np.ndarray,
    scales: np.ndarray,
    indices_shape: list[int],
    indices_dtype: int,
    block_size: int,
    quantize_axis: int | None = None,
    gather_axis: int = 0,
):
    data = helper.make_tensor(
        "data",
        TensorProto.FLOAT8E4M3FN,
        list(codes.shape),
        codes.tobytes(),
        raw=True,
    )
    out_dtype = TensorProto.FLOAT16 if scales.dtype == np.float16 else TensorProto.FLOAT
    indices = helper.make_tensor_value_info("indices", indices_dtype, indices_shape)
    output = helper.make_tensor_value_info(
        "output",
        out_dtype,
        list(codes.shape[:gather_axis])
        + indices_shape
        + list(codes.shape[gather_axis + 1 :]),
    )
    attrs = {"gather_axis": gather_axis, "block_size": block_size}
    if quantize_axis is not None:
        attrs["quantize_axis"] = quantize_axis
    node = helper.make_node(
        "GatherBlockQuantized",
        ["data", "indices", "scales"],
        ["output"],
        domain="com.microsoft",
        **attrs,
    )
    return make_model_from_nodes(
        [node],
        [indices],
        [output],
        initializers=[data, numpy_helper.from_array(scales, name="scales")],
        opset=21,
        extra_opsets=[helper.make_opsetid("com.microsoft", 1)],
    )


class TestGatherBlockQuantizedFloat8:
    def test_decode_all_codes(self, model_runner):
        codes = np.array(
            [c for c in range(256) if c not in FLOAT8_NAN_CODES], dtype=np.uint8
        ).reshape(-1, 1)
        model = _make_gbq_float8_model(
            codes,
            np.ones([1, 1], dtype=np.float16),
            [1, codes.shape[0]],
            TensorProto.INT64,
            block_size=0,
            quantize_axis=1,
        )
        indices = np.arange(codes.shape[0], dtype=np.int64).reshape(1, -1)
        actual, expected = model_runner.run_sample(model, [indices])
        compare_outputs(actual, expected, atol=0, rtol=0)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_engram_per_tensor_scale(self, model_runner, seq_len):
        vocab = 4096
        rng = np.random.default_rng(0)
        model = _make_gbq_float8_model(
            _float8_codes(rng, [vocab, ENGRAM_ROW]),
            np.array([[0.0123]], dtype=np.float16),
            [1, seq_len, ENGRAM_HEADS],
            TensorProto.INT64,
            block_size=0,
            quantize_axis=1,
        )
        indices = rng.integers(
            -vocab, vocab, [1, seq_len, ENGRAM_HEADS], dtype=np.int64
        )
        actual, expected = model_runner.run_sample(model, [indices])
        compare_outputs(actual, expected, atol=1e-3, rtol=1e-3)

    def test_blockwise_scale_broadcast_over_rows(self, model_runner):
        vocab = 64
        rng = np.random.default_rng(1)
        model = _make_gbq_float8_model(
            _float8_codes(rng, [vocab, ENGRAM_ROW]),
            rng.uniform(0.01, 0.1, [1, ENGRAM_ROW // 32]).astype(np.float32),
            [8],
            TensorProto.INT32,
            block_size=32,
        )
        indices = rng.integers(0, vocab, [8], dtype=np.int32)
        actual, expected = model_runner.run_sample(model, [indices])
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-5)

    def test_inner_gather_axis_leading_quantize_axis(self, model_runner):
        rng = np.random.default_rng(2)
        model = _make_gbq_float8_model(
            _float8_codes(rng, [4, 64, 32]),
            rng.uniform(0.01, 0.1, [1, 1, 32]).astype(np.float16),
            [2, 3],
            TensorProto.INT64,
            block_size=0,
            quantize_axis=0,
            gather_axis=1,
        )
        indices = rng.integers(-64, 64, [2, 3], dtype=np.int64)
        actual, expected = model_runner.run_sample(model, [indices])
        compare_outputs(actual, expected, atol=1e-3, rtol=1e-3)
