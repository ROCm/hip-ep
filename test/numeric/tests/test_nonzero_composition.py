#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""End-to-end correctness for multi-op graphs that compose dynamic-shape ops.

The single-op numeric tests verify each Category-C wrapper in
isolation. This file stitches them together to verify the EP host-side
resolver / compose-dim-specs / metadata blob round-trip when:

  1. A Category-C op (NonZero) produces a tensor whose `N` dim is only
     known after compute, and another op CONSUMES that tensor's shape
     via Shape -> something. ONNX `Shape(NonZero(X))` is itself a
     legitimate 1-D output that depends on the data of X. The EP must
     post-compute-resolve NonZero's `N` and re-publish the corresponding
     `Shape` output of length=2 (since NonZero is rank-2).

  2. A Category-B op (Range with func-arg i64 operands) feeds a
     downstream consumer whose dimension is the Range output length.
     Composition of arithmetic-tree DimSpecs is the canonical
     correctness check for the ComposeDimSpecs pass.

These graphs are tiny but they exercise pipeline edges that single-op
tests do not -- specifically:

  * `output_dim_specs` propagation across two ops in the same module
    (composite DimSpec trees with InputDim and RuntimeSlot leaves
    interleaved),
  * the EP marshalling the **shape vector** of a Category-C output as
    a normal i64 model output (verified via Shape(NonZero(X))),
  * the runtime not re-using a stale slot dim across inferences (each
    test calls `run_sample` only once today, but the model loads with
    the slot table initialised to -1; missing init would manifest as
    a stale dim being returned).
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_shape_of_nonzero_model(
    input_dtype: np.dtype, input_shape: list[int]
):
    """`Shape(NonZero(X))` -- output is `[2]` int64 = (rank, N).

    The runtime evaluates NonZero (publishes the dyn N into a slot),
    then Shape consumes the rank-2 NonZero output to emit `[rank, N]`
    as an i64 1-D tensor.  Shape's output dim-0 is statically known
    (= 2 = rank of NonZero output), but its CONTENT depends on a
    Category-C dim, so this is a useful end-to-end check that the
    compose pass propagates the RuntimeSlot dim into a downstream op's
    output.
    """
    from framework.onnx_utils import np_to_onnx_type

    tp = np_to_onnx_type(input_dtype)
    X = helper.make_tensor_value_info("X", tp, input_shape)
    # Intermediate NonZero output (`[rank, N]` int64; N dynamic).
    nz_out = helper.make_tensor_value_info(
        "nz_out", TensorProto.INT64, [len(input_shape), None]
    )
    # Final Shape output (1-D length-2 int64: (rank, N)).
    Y = helper.make_tensor_value_info("Y", TensorProto.INT64, [2])
    nz_node = helper.make_node("NonZero", ["X"], ["nz_out"])
    shape_node = helper.make_node("Shape", ["nz_out"], ["Y"])
    return make_model_from_nodes([nz_node, shape_node], [X], [Y])


def _make_constantofshape_over_range_model():
    """Range -> ConstantOfShape composition (no Shape op required).

    The ONNX `ConstantOfShape` op takes a 1-D i64 tensor giving the
    output shape directly. `Range(start, limit, delta)` of i64 elems
    is itself a 1-D i64 vector whose VALUES are the desired dim
    sizes, so feeding Range's output STRAIGHT INTO ConstantOfShape
    builds:

        shape_tensor = Range(start, limit, delta)
        Y            = ConstantOfShape(shape_tensor, value=1.0)

    Critical properties exercised here:

      * Range with i64 func-arg operands lands in Category B. The
        EP host-side DimSpec resolver computes `Range.dim[0] =
        CeilDiv(Sub(limit, start), delta)` before main_graph runs
        and allocates the output OrtValue at the right rank.

      * ConstantOfShape's `shape` input is NOT a func-arg in this
        graph -- it's the Range intermediate. The conversion picks
        Category C and the runtime wrap_constant_of_shape_dyn D2H-
        stages the Range-produced i64s, publishes per-axis dim
        slots, dyn-pool-allocs the GPU buffer, and launches the
        fill kernel.

      * The N-rank ConstantOfShape output's first dim is whatever
        `Range[0]` was -- so the input values matter, not just the
        Range output LENGTH. (E.g. Range(2,5,1) = [2,3,4] -> output
        shape [2,3,4].)

    This validates the slot table + dyn-pool buffer alias survival
    between two ops on the same GPU stream within a single Compute().
    """
    start = helper.make_tensor_value_info("start", TensorProto.INT64, [])
    limit = helper.make_tensor_value_info("limit", TensorProto.INT64, [])
    delta = helper.make_tensor_value_info("delta", TensorProto.INT64, [])
    # Y is filled with a fp32 constant and has rank = length of Range.
    # We declare a max possible rank dynamically by computing it from
    # the test inputs in the test body; ONNX requires us to declare a
    # rank here. We use rank=3 throughout the parametrised cases below.
    Y = helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT, [None, None, None]
    )

    range_node = helper.make_node(
        "Range", ["start", "limit", "delta"], ["range_out"]
    )
    value_t = numpy_helper.from_array(
        np.array([1.0], dtype=np.float32), name="cof_value"
    )
    cof_node = helper.make_node(
        "ConstantOfShape", ["range_out"], ["Y"], value=value_t
    )
    return make_model_from_nodes(
        [range_node, cof_node],
        [start, limit, delta],
        [Y],
        initializers=[value_t],
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestNonZeroComposition:
    """Compositions involving the Category-C NonZero op."""

    def test_shape_of_nonzero_1d_i64(self, model_runner):
        """Shape(NonZero(1D i64)) -- N varies, output is [2] = (1, N)."""
        x = np.array(
            [0, 5, 0, 3, 0, 0, 7, 8, 0, 9, 0, 0],
            dtype=np.int64,
        )
        model = _make_shape_of_nonzero_model(np.int64, list(x.shape))
        actual, expected = model_runner.run_sample(model, [x], reference="cpu")
        compare_outputs(actual, expected, atol=0, rtol=0)
        # Sanity: the [2]-vector should be (1, count_of_nonzeros).
        assert int(actual[0][0]) == 1
        assert int(actual[0][1]) == int((x != 0).sum())

    def test_shape_of_nonzero_2d_bool(self, model_runner):
        """Shape(NonZero(2D bool)) -- output is [2] = (2, N)."""
        shape = [4, 6]
        rng = np.random.default_rng(2)
        x = (rng.uniform(0, 1, shape) > 0.5).astype(np.bool_)
        model = _make_shape_of_nonzero_model(np.bool_, shape)
        actual, expected = model_runner.run_sample(model, [x], reference="cpu")
        compare_outputs(actual, expected, atol=0, rtol=0)
        assert int(actual[0][0]) == 2
        assert int(actual[0][1]) == int(x.sum())

    def test_shape_of_nonzero_2d_all_zero(self, model_runner):
        """All zeros => N=0, output is [2, 0] -> Shape returns (2, 0)."""
        shape = [4, 6]
        x = np.zeros(shape, dtype=np.bool_)
        model = _make_shape_of_nonzero_model(np.bool_, shape)
        actual, expected = model_runner.run_sample(model, [x], reference="cpu")
        compare_outputs(actual, expected, atol=0, rtol=0)
        assert int(actual[0][0]) == 2
        assert int(actual[0][1]) == 0


class TestRangeConstantOfShapeComposition:
    """Range (Category B) feeds ConstantOfShape (Category C).

    Verifies the two-pass scheme end-to-end on a tiny multi-op graph:

        Range(i64,i64,i64) -> ConstantOfShape(value=1.0f).

    Range's start/limit/delta are i64 func-args, so its output length
    is resolvable PRE-compute as `CeilDiv(Sub(limit,start), delta)` by
    the EP host-side DimSpec walker -- this is Category B.

    The Range output then directly feeds ConstantOfShape's `shape`
    operand. From ConstantOfShape's standpoint, this is an
    INTERMEDIATE i64 vector (not a func-arg), so the conversion takes
    Category C: each of ConstantOfShape's `rank` output dims is a
    RuntimeSlot, and wrap_constant_of_shape_dyn host-stages the Range
    output to read the dim sizes.

    Why this matters end-to-end:

      * Two ops, both contributing to the EP's dim_slot table inside
        one inference_compute() call -- exercises the per-Compute
        slot-table reset and the GPU dyn-pool allocator.

      * The shape vector handed to wrap_constant_of_shape_dyn is a
        kernel-produced buffer, not an EP-marshalled input. The D2H
        stage in the wrap must read it AFTER Range's fill kernel
        completes (both on the same stream, so the in-order
        guarantee + the wrap's hipStreamSynchronize cover it).

      * If a future refactor accidentally aliases the dyn-pool with
        Range's output buffer, the ConstantOfShape read would race
        with the kernel that wrote it.

    Each case picks (start, limit, delta) so Range produces a 3-elem
    vector (matches the static rank=3 declared on Y), and we verify
    both the shape AND the all-1.0 content.
    """

    @pytest.mark.parametrize(
        "start,limit,delta,expected_shape",
        [
            (2, 5, 1, (2, 3, 4)),           # Range = [2,3,4]
            (1, 7, 2, (1, 3, 5)),           # Range = [1,3,5]
            (4, 13, 3, (4, 7, 10)),          # Range = [4,7,10]
        ],
    )
    def test_range_cof_i64(
        self, model_runner, start, limit, delta, expected_shape
    ):
        dtype = np.int64
        model = _make_constantofshape_over_range_model()
        s = np.array(start, dtype=dtype)
        l = np.array(limit, dtype=dtype)
        d = np.array(delta, dtype=dtype)
        actual, expected = model_runner.run_sample(
            model, [s, l, d], reference="cpu"
        )
        compare_outputs(actual, expected, atol=0, rtol=0)
        assert actual[0].shape == expected_shape, (
            f"actual={actual[0].shape} expected={expected_shape}"
        )
        assert np.all(actual[0] == 1.0)
