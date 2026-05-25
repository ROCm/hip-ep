#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric verification for ONNX Range (Category-B and Category-C).

Range exercises BOTH branches of the operand-provenance dispatch added
with the dynamic output shapes work:

  * Category B -- when start/limit/delta all originate from EP-visible
    func-arg scalar inputs AND the element type is i64, the OnnxToHip
    conversion emits a DimSpec arithmetic tree
    `CeilDiv(Sub(limit,start), delta)` over `InputValueI64` leaves so
    the EP host-side resolver computes the output length BEFORE
    `main_graph` runs. The runtime path then degenerates to a single
    fill kernel launch (no slot publishing / no D2H sync).

  * Category C -- when at least one of (start, limit, delta) is an
    intermediate value, or when the element type is not i64 (the EP
    resolver only knows how to read InputValueI64 as int64), the
    conversion falls back to a `RuntimeSlot` leaf and the wrap_range_dyn
    code path host-stages the three scalars, computes the length on
    host, publishes the dim into a slot, dyn_pool_allocs the GPU
    buffer, publishes the pointer, then launches the fill kernel.

Both branches must produce the same Range result. Each test below
explicitly labels which category it targets so a future regression
shows up with a clear pointer at the failing path.

Numeric tolerance: Range values are exact for integer dtypes; for fp
dtypes we use a tight tolerance to allow last-ulp difference between
the GPU fill kernel and ORT CPU's reference accumulation.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# numpy dtype -> (ONNX TensorProto enum, comparator atol).
# ONNX Range supports only int32/int64/float/double per the spec, so
# we only map those here.
_DTYPE_INFO = {
    np.dtype(np.int64): (TensorProto.INT64, 0.0),
    np.dtype(np.int32): (TensorProto.INT32, 0.0),
    np.dtype(np.float32): (TensorProto.FLOAT, 1e-5),
}


def _make_range_model_dynamic(dtype: np.dtype):
    """Range with start/limit/delta as FUNC-ARG scalar inputs.

    Category-B-eligible for i64; Category C for everything else (the EP
    resolver only reads InputValueI64 as int64; other dtypes degrade to
    runtime slot publishing).
    """
    np_dtype = np.dtype(dtype)
    onnx_type, _ = _DTYPE_INFO[np_dtype]
    s = helper.make_tensor_value_info("start", onnx_type, [])
    l = helper.make_tensor_value_info("limit", onnx_type, [])
    d = helper.make_tensor_value_info("delta", onnx_type, [])
    Y = helper.make_tensor_value_info("Y", onnx_type, [None])
    node = helper.make_node("Range", ["start", "limit", "delta"], ["Y"])
    return make_model_from_nodes([node], [s, l, d], [Y])


def _make_range_model_via_intermediate(dtype: np.dtype):
    """Range where `start` is computed as `Cast(start_in, to=same dtype)`.

    The Cast output is an intermediate value (the OnnxToHip Cast
    pattern unconditionally emits `hip.cast` even when source and
    target element types match), not a func-arg, so the
    OnnxToHip conversion can NOT prove host-readability of the start
    scalar and falls back to Category C even for i64.

    A same-dtype Cast (rather than Add) is used to avoid pulling in
    MIOpen for the intermediate op -- MIOpen has no i64 kernel, and a
    runtime fallback produces zeroed output that turns into a hard-to-
    diagnose "Range starts at 0 instead of N" assertion. Cast lowers
    to `hip.cast`, which has a generic i64 path. Identity is
    unsuitable here because IdentityForward replaces the SSA value in
    place and the shape-input then traces all the way back to the
    func arg (which would silently flip us into Category B).
    """
    np_dtype = np.dtype(dtype)
    onnx_type, _ = _DTYPE_INFO[np_dtype]
    s = helper.make_tensor_value_info("start_in", onnx_type, [])
    l = helper.make_tensor_value_info("limit", onnx_type, [])
    d = helper.make_tensor_value_info("delta", onnx_type, [])
    Y = helper.make_tensor_value_info("Y", onnx_type, [None])

    cast_node = helper.make_node("Cast", ["start_in"], ["start"], to=onnx_type)
    range_node = helper.make_node("Range", ["start", "limit", "delta"], ["Y"])
    return make_model_from_nodes([cast_node, range_node], [s, l, d], [Y])


def _scalar(value, dtype: np.dtype) -> np.ndarray:
    """0-D numpy array of the given dtype."""
    return np.array(value, dtype=np.dtype(dtype))


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestRange:
    # ------- Category B: i64, all-func-arg --------------------------------
    #
    # Output length is resolved by the EP host-side DimSpec walker
    # (CeilDiv(Sub(limit,start), delta)) BEFORE main_graph runs. Verify
    # the runtime fill agrees with numpy.

    @pytest.mark.parametrize(
        "start,limit,delta",
        [
            (0, 10, 1),
            (5, 25, 2),
            (-3, 4, 1),
            (0, 0, 1),  # empty range
            (10, 0, -2),  # descending
        ],
    )
    def test_range_i64_category_b(self, model_runner, start, limit, delta):
        """Category B: i64 func-arg scalars -> EP pre-resolves the length."""
        dtype = np.int64
        model = _make_range_model_dynamic(dtype)
        actual, expected = model_runner.run_sample(
            model,
            [_scalar(start, dtype), _scalar(limit, dtype), _scalar(delta, dtype)],
            reference="cpu",
        )
        compare_outputs(actual, expected, atol=0, rtol=0)

    # ------- Category C: i32 func-arg (dtype rules out Category B) -------
    #
    # i32 takes the Category C path even though all three operands are
    # func-args, because the EP resolver only knows how to read i64.

    @pytest.mark.parametrize(
        "start,limit,delta",
        [
            (0, 16, 1),
            (4, 100, 7),
            (-5, 5, 1),
            (3, 3, 1),  # empty range
        ],
    )
    def test_range_i32_category_c(self, model_runner, start, limit, delta):
        """Category C via dtype: i32 -> wrap_range_dyn host-stages and slot-publishes."""
        dtype = np.int32
        model = _make_range_model_dynamic(dtype)
        actual, expected = model_runner.run_sample(
            model,
            [_scalar(start, dtype), _scalar(limit, dtype), _scalar(delta, dtype)],
            reference="cpu",
        )
        compare_outputs(actual, expected, atol=0, rtol=0)

    # ------- Category C: i64 via intermediate -----------------------------
    #
    # All three operands ARE EP inputs but `start` is routed through an
    # intermediate `Add` so the operand-provenance check fails the
    # all-func-arg gate, forcing Category C. Sanity for the dispatch
    # fallback when a real model trace uses a tiny op chain to compute
    # one of the bounds.

    @pytest.mark.parametrize(
        "start,limit,delta",
        [(0, 32, 1), (3, 50, 5)],
    )
    def test_range_i64_category_c_via_intermediate(
        self, model_runner, start, limit, delta
    ):
        dtype = np.int64
        model = _make_range_model_via_intermediate(dtype)
        actual, expected = model_runner.run_sample(
            model,
            [_scalar(start, dtype), _scalar(limit, dtype), _scalar(delta, dtype)],
            reference="cpu",
        )
        compare_outputs(actual, expected, atol=0, rtol=0)

    # ------- Category C: float dtypes -------------------------------------
    #
    # Float Range always lands in Category C (EP resolver i64-only).
    # Tight tolerance because the fill kernel emits start + k*delta and
    # numpy CPU does the same -- last-ulp deltas only at extreme k.

    @pytest.mark.parametrize(
        "start,limit,delta",
        [(0.0, 10.0, 0.5), (-2.0, 3.0, 0.25)],
    )
    def test_range_f32_category_c(self, model_runner, start, limit, delta):
        dtype = np.float32
        model = _make_range_model_dynamic(dtype)
        _, atol = _DTYPE_INFO[np.dtype(dtype)]
        actual, expected = model_runner.run_sample(
            model,
            [_scalar(start, dtype), _scalar(limit, dtype), _scalar(delta, dtype)],
            reference="cpu",
        )
        compare_outputs(actual, expected, atol=atol)

    # NOTE: f16 Range is NOT in the ONNX Range schema (the spec lists
    # only int32 / int64 / float / double), so no test exists here.
    # ORT rejects f16 Range with INVALID_GRAPH at session load.

    # ------- Larger sweeps to exercise grid coverage ----------------------

    def test_range_i64_long_category_b(self, model_runner):
        """Long Range -- sanity on a multi-block fill (4096 elements)."""
        dtype = np.int64
        model = _make_range_model_dynamic(dtype)
        actual, expected = model_runner.run_sample(
            model,
            [_scalar(0, dtype), _scalar(4096, dtype), _scalar(1, dtype)],
            reference="cpu",
        )
        compare_outputs(actual, expected, atol=0, rtol=0)
        assert actual[0].shape == (4096,)

    def test_range_i32_long_category_c(self, model_runner):
        """Same as above but via the Category C path."""
        dtype = np.int32
        model = _make_range_model_dynamic(dtype)
        actual, expected = model_runner.run_sample(
            model,
            [_scalar(0, dtype), _scalar(4096, dtype), _scalar(1, dtype)],
            reference="cpu",
        )
        compare_outputs(actual, expected, atol=0, rtol=0)
        assert actual[0].shape == (4096,)
