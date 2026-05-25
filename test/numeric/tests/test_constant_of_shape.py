#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric verification for ONNX ConstantOfShape.

Three paths through the conversion are exercised:

  * Compile-time fold -- shape input is a Constant and result type is
    fully static. The conversion folds the whole op to a single
    `arith.constant` splat at compile time. Runtime sees a literal.

  * Category B -- shape input is a func-arg int64 1-D tensor (the EP
    DimSpec resolver reads it as `InputValueI64`). The output shape is
    resolved BEFORE main_graph runs and the ORT OrtValue is allocated
    up front. wrap_constant_of_shape (non-_dyn) handles the fill.

  * Category C -- shape input is an intermediate value OR an int32
    func-arg (the resolver only reads i64). The conversion attaches
    one RuntimeSlot per output axis, and wrap_constant_of_shape_dyn
    host-stages the shape vector, publishes each axis dim into its
    slot, dyn_pool_allocs the buffer, publishes the pointer, then
    launches the fill kernel. The EP post-compute reader then
    D2H-copies into a freshly allocated OrtValue.

Each test calls out which path it targets so a future regression
points at the right layer.

The output is always a splat (constant value broadcast over the
shape), so the comparator can run with `atol=0, rtol=0` against ORT
CPU. Output dtype coverage spans fp16/fp32/int32/int64/bool to
exercise the full runtime dtype switch.
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


# numpy dtype -> (ONNX TensorProto enum used in `value` attribute,
# value-tensor builder lambda).
_VALUE_BUILDERS = {
    np.dtype(np.float32): TensorProto.FLOAT,
    np.dtype(np.float16): TensorProto.FLOAT16,
    np.dtype(np.int32): TensorProto.INT32,
    np.dtype(np.int64): TensorProto.INT64,
    np.dtype(np.bool_): TensorProto.BOOL,
}


def _make_value_attr(dtype: np.dtype, scalar):
    """Build the `value` TensorProto attribute used by ConstantOfShape.

    The attribute holds a single element. ONNX accepts any output dtype
    here; the test exercises several to ensure the runtime fill kernel
    handles each.
    """
    np_dtype = np.dtype(dtype)
    arr = np.array([scalar], dtype=np_dtype)
    return numpy_helper.from_array(arr, name="value")


def _make_cof_model_constant_shape(
    shape_values: list[int], dtype: np.dtype, scalar
):
    """Compile-time-fold path: shape is a Constant initializer."""
    np_dtype = np.dtype(dtype)
    onnx_out_type = _VALUE_BUILDERS[np_dtype]

    shape_init = numpy_helper.from_array(
        np.array(shape_values, dtype=np.int64), name="shape_init"
    )
    Y = helper.make_tensor_value_info("Y", onnx_out_type, shape_values)
    value_attr = _make_value_attr(np_dtype, scalar)
    cof_node = helper.make_node(
        "ConstantOfShape", ["shape_init"], ["Y"], value=value_attr
    )
    # Empty inputs list; the model has only the constant initializer.
    return make_model_from_nodes(
        [cof_node], inputs=[], outputs=[Y], initializers=[shape_init]
    )


def _make_cof_model_funcarg_shape(
    rank: int,
    dtype: np.dtype,
    scalar,
    shape_dtype: np.dtype = np.int64,
):
    """Category B path: shape is a func-arg 1-D int tensor of fixed length.

    For shape_dtype=int64 the conversion lands in Category B (the EP
    DimSpec resolver reads i64 InputValueI64 leaves directly).
    For shape_dtype=int32 the conversion falls back to Category C (the
    resolver is i64-only); the same test then exercises wrap_constant_of_shape_dyn.
    """
    np_dtype = np.dtype(dtype)
    onnx_out_type = _VALUE_BUILDERS[np_dtype]
    shape_onnx_type = (
        TensorProto.INT64 if np.dtype(shape_dtype) == np.int64 else TensorProto.INT32
    )

    shape_in = helper.make_tensor_value_info("shape", shape_onnx_type, [rank])
    # Output rank is fixed (rank); each dim is dynamic since it comes
    # from the runtime-read shape input.
    Y = helper.make_tensor_value_info("Y", onnx_out_type, [None] * rank)

    value_attr = _make_value_attr(np_dtype, scalar)
    cof_node = helper.make_node(
        "ConstantOfShape", ["shape"], ["Y"], value=value_attr
    )
    return make_model_from_nodes(
        [cof_node], inputs=[shape_in], outputs=[Y]
    )


def _make_cof_model_via_intermediate(
    rank: int, dtype: np.dtype, scalar
):
    """Category C via intermediate shape: shape = Cast(shape_in, to=int64).

    The Cast output is an intermediate value (the OnnxToHip Cast
    pattern unconditionally emits a `hip.cast` op even when source
    and target element types match), so operand-provenance fails the
    func-arg check and the conversion attaches one RuntimeSlot per
    output dim. The runtime wrap_constant_of_shape_dyn must publish
    each dim into its slot.

    A same-type Cast (int64 -> int64) is the minimal way to insert a
    non-trivial EP-supported op between the func-arg shape and
    ConstantOfShape without pulling in:
      * Identity (folded by IdentityForward to the input SSA value),
      * Concat (no OnnxToHip pattern, so the EP cannot claim the
        node and the partition fails with "default CPU EP
        disabled"),
      * Add (lowers to MIOpen, which has no i64 kernel).
    """
    np_dtype = np.dtype(dtype)
    onnx_out_type = _VALUE_BUILDERS[np_dtype]

    shape_in = helper.make_tensor_value_info("shape_in", TensorProto.INT64, [rank])
    Y = helper.make_tensor_value_info("Y", onnx_out_type, [None] * rank)

    cast_node = helper.make_node(
        "Cast", ["shape_in"], ["shape"], to=TensorProto.INT64
    )
    value_attr = _make_value_attr(np_dtype, scalar)
    cof_node = helper.make_node(
        "ConstantOfShape", ["shape"], ["Y"], value=value_attr
    )
    return make_model_from_nodes(
        [cast_node, cof_node],
        inputs=[shape_in],
        outputs=[Y],
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestConstantOfShape:
    # ------- Compile-time fold ------------------------------------------
    #
    # Shape input is a Constant initializer + result type is fully
    # static => ConstantOfShapeFold collapses the op to a single
    # arith.constant splat. Verifies the fast path that transformers
    # rely on for static KV / mask init.

    @pytest.mark.parametrize(
        "shape,dtype,scalar",
        [
            ([4, 8], np.float32, 0.0),
            ([2, 3, 4], np.float16, 1.5),
            ([1, 5], np.int64, 7),
            ([16], np.int32, -3),
        ],
    )
    def test_cof_fold_path(self, model_runner, shape, dtype, scalar):
        """Static shape Constant -> compile-time fold to arith.constant."""
        model = _make_cof_model_constant_shape(shape, dtype, scalar)
        actual, expected = model_runner.run_sample(model, [], reference="cpu")
        compare_outputs(actual, expected, atol=0, rtol=0)
        assert actual[0].shape == tuple(shape)

    # ------- Category B: i64 func-arg shape -----------------------------
    #
    # The EP DimSpec resolver pre-resolves every output dim by reading
    # InputValueI64 leaves directly out of the func-arg int64 shape
    # tensor. wrap_constant_of_shape (non-_dyn) handles the fill.

    @pytest.mark.parametrize(
        "shape,dtype,scalar",
        [
            ([3, 4], np.float32, 0.5),
            ([2, 2, 2], np.float16, -1.0),
            ([5], np.int64, 42),
            ([4, 6], np.int32, 9),
            ([3, 3], np.bool_, True),
        ],
    )
    def test_cof_category_b_i64_shape(
        self, model_runner, shape, dtype, scalar
    ):
        """Category B: i64 func-arg shape -> EP pre-resolves all dims."""
        rank = len(shape)
        model = _make_cof_model_funcarg_shape(
            rank, dtype, scalar, shape_dtype=np.int64
        )
        shape_arr = np.array(shape, dtype=np.int64)
        actual, expected = model_runner.run_sample(
            model, [shape_arr], reference="cpu"
        )
        compare_outputs(actual, expected, atol=0, rtol=0)
        assert actual[0].shape == tuple(shape)

    # ------- Category C: via intermediate shape -------------------------
    #
    # The shape is `shape_in + zero_vec` -- the Add output is not a
    # func-arg, so operand-provenance fails the host-readable check and
    # the conversion attaches `slot_ids` + RuntimeSlot per dim.

    @pytest.mark.parametrize(
        "shape,dtype,scalar",
        [
            ([4, 4], np.float32, 1.0),
            ([2, 3], np.int64, 7),
        ],
    )
    def test_cof_category_c_via_intermediate(
        self, model_runner, shape, dtype, scalar
    ):
        rank = len(shape)
        model = _make_cof_model_via_intermediate(rank, dtype, scalar)
        shape_arr = np.array(shape, dtype=np.int64)
        actual, expected = model_runner.run_sample(
            model, [shape_arr], reference="cpu"
        )
        compare_outputs(actual, expected, atol=0, rtol=0)
        assert actual[0].shape == tuple(shape)

    # ------- Larger shapes (atomic-free fill kernel sanity) -------------

    def test_cof_large_fold(self, model_runner):
        """Bigger static splat -- exercises the fold path's bytes path."""
        model = _make_cof_model_constant_shape(
            [64, 64], np.float32, 0.125
        )
        actual, expected = model_runner.run_sample(model, [], reference="cpu")
        compare_outputs(actual, expected, atol=0, rtol=0)

    def test_cof_large_category_b(self, model_runner):
        """Bigger Category B -- runtime fill kernel must handle 4096 elems."""
        rank = 2
        model = _make_cof_model_funcarg_shape(
            rank, np.float32, -0.5, shape_dtype=np.int64
        )
        shape_arr = np.array([64, 64], dtype=np.int64)
        actual, expected = model_runner.run_sample(
            model, [shape_arr], reference="cpu"
        )
        compare_outputs(actual, expected, atol=0, rtol=0)
