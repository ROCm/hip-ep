#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for data-rearrangement / shape ops: Tile, Expand, Pad, GatherND,
Slice, ScatterND.

All six ops are added by the qwen-vision-kernels PR. They share a common
property: the **shape** of an output depends on a graph-time tensor (repeats
for Tile, shape for Expand, pads for Pad, indices for GatherND, starts/ends
for Slice, indices for ScatterND). The runtime D2H-reads these small tensors
once per call and dispatches a host-built launch -- so the test must pin
both the value tensors and the data tensor in the same model.

Per the runtime dtype tables in lib/Runtime/real/<op>.cpp the supported
data dtype set is:

    Tile      : f16, f32, i32, i64
    Expand    : f16, f32, i32, i64
    Pad       : f16, f32, i32, i64       (modes: constant, reflect, edge, wrap)
    GatherND  : f16, f32, i32, i64       (indices: int64 only)
    Slice     : f16, f32, i32, i64       (starts/ends/axes/steps: int64 only)
    ScatterND : f16, f32, i32, i64       (indices: int64 only;
                                          reductions: none, add, mul, min, max)
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type


# ---------------------------------------------------------------------------
# Concat
# ---------------------------------------------------------------------------
#
# Concat doesn't have a runtime kernel of its own -- ConcatConversion.cpp
# decomposes it into `tensor.empty + tensor.insert_slice`, which bufferize to
# `memref.subview + memref.copy`. For non-leading axes the destination
# `memref.subview` is STRIDED relative to the contiguous source, and the
# runtime `memrefCopy` helper in lib/Runtime/real/hip.cpp must honour those
# strides. A previous flat-memcpy implementation silently corrupted the
# output whenever axis != 0 (the second insert_slice's contiguous source
# overran the first one's slot, producing values that look like a
# stride-shifted version of the last input). Symptom on Qwen 3.5 vision:
# a `Reshape -> Unsqueeze -> Concat(axis=-1) -> Expand -> Tile` chain
# returning only the second input's data shifted by one element.
def _make_concat_model(dtype, input_shapes: list[list[int]], axis: int):
    """Build a model with a single onnx.Concat across N graph inputs."""
    tp = np_to_onnx_type(dtype)
    inputs = [
        helper.make_tensor_value_info(f"X{i}", tp, list(s))
        for i, s in enumerate(input_shapes)
    ]
    rank = len(input_shapes[0])
    norm_axis = axis if axis >= 0 else axis + rank
    out_shape = list(input_shapes[0])
    out_shape[norm_axis] = sum(s[norm_axis] for s in input_shapes)
    Y = helper.make_tensor_value_info("Y", tp, out_shape)
    node = helper.make_node(
        "Concat", [f"X{i}" for i in range(len(input_shapes))], ["Y"], axis=axis
    )
    return make_model_from_nodes([node], inputs, [Y])


class TestConcat:
    @pytest.mark.parametrize(
        "dtype,shapes,axis",
        [
            # Leading-axis concat -- destination subview is contiguous,
            # exercises the flat memcpy fast path.
            (np.float16, [[2, 4], [3, 4]], 0),
            (np.float32, [[1, 8], [2, 8], [3, 8]], 0),
            # Non-leading axes -- destination subview is STRIDED, exercises
            # the hipMemcpy2DAsync tier (rowStart == 1 in memrefCopy).
            (np.float32, [[2, 3], [2, 4], [2, 5]], 1),
            (np.float16, [[2, 3], [2, 4], [2, 5]], -1),
            # The Qwen 3.5 vision reproducer: int64 indices, two rank-2
            # singleton-trailing-dim inputs concatenated along the last
            # axis. Before the strided-memrefCopy fix this returned the
            # second input's values shifted by one element.
            (np.int64, [[1024, 1], [1024, 1]], -1),
            # Higher rank: rank-3 concat on the middle axis. After the
            # contiguous-suffix detection this is also rowStart == 1
            # (the entire last dim plus the per-input middle dim form one
            # contiguous "row" against the outer dim 0).
            (np.float32, [[2, 3, 5], [2, 4, 5]], 1),
            # Higher rank, last-axis concat -- rowStart == 2 in
            # memrefCopy, exercises the generic per-row tier.
            (np.float16, [[2, 3, 4], [2, 3, 5]], -1),
        ],
    )
    def test_concat(self, model_runner, dtype, shapes, axis):
        model = _make_concat_model(dtype, shapes, axis)
        rng = np.random.default_rng(700 + len(shapes))
        feeds = []
        for i, s in enumerate(shapes):
            if np.issubdtype(dtype, np.integer):
                feeds.append(rng.integers(-100, 100, s, dtype=dtype))
            else:
                feeds.append(rng.uniform(-2.0, 2.0, s).astype(dtype))
        actual, expected = model_runner.run_sample(model, feeds)
        compare_outputs(actual, expected, atol=0)


# ---------------------------------------------------------------------------
# Tier-3 strided memref.copy kernel (hip_strided_copy)
# ---------------------------------------------------------------------------
#
# A last-axis Concat on rank >= 3 produces a destination `memref.subview`
# whose contiguous suffix is only the innermost dim, so two or more outer
# dims remain strided (rowStart >= 2 in memrefCopy). That is the generic
# "Tier 3" regime, which the parallel `hip_strided_copy` kernel now handles
# in a single launch instead of one hipMemcpyAsync per row. These cases pin
# that path numerically against the ORT CPU reference, across the element
# widths the kernel templates on (2/4/8 bytes) and with outer extents large
# enough to span multiple thread blocks plus the grid-stride remainder.
class TestStridedCopyTier3:
    @pytest.mark.parametrize(
        "dtype,shapes,axis",
        [
            # rank-3, last axis -> rowStart == 2 (outerRank == 2).
            (np.float16, [[8, 16, 4], [8, 16, 4]], -1),  # 2-byte elems
            (np.float32, [[8, 16, 5], [8, 16, 3]], -1),  # 4-byte elems
            (np.int32, [[6, 9, 4], [6, 9, 7]], -1),  # 4-byte elems
            (np.int64, [[5, 7, 3], [5, 7, 6]], -1),  # 8-byte elems
            # Large outer extent: 4096 rows forces grid clamp + grid-stride loop.
            (np.float16, [[64, 64, 2], [64, 64, 2]], -1),
            # rank-4, last axis -> rowStart == 3 (outerRank == 3).
            (np.float32, [[2, 3, 4, 5], [2, 3, 4, 6]], -1),
            (np.int64, [[2, 2, 3, 4], [2, 2, 3, 4]], -1),
        ],
    )
    def test_concat_last_axis_strided(self, model_runner, dtype, shapes, axis):
        model = _make_concat_model(dtype, shapes, axis)
        rng = np.random.default_rng(800 + sum(shapes[0]))
        feeds = []
        for s in shapes:
            if np.issubdtype(dtype, np.integer):
                feeds.append(rng.integers(-100, 100, s, dtype=dtype))
            else:
                feeds.append(rng.uniform(-2.0, 2.0, s).astype(dtype))
        actual, expected = model_runner.run_sample(model, feeds)
        # Pure data movement -- must be bit-exact, no tolerance.
        compare_outputs(actual, expected, atol=0)


# ---------------------------------------------------------------------------
# Tile
# ---------------------------------------------------------------------------
def _make_tile_model(dtype, input_shape: list[int], repeats: list[int]):
    tp = np_to_onnx_type(dtype)
    X = helper.make_tensor_value_info("X", tp, list(input_shape))
    repeats_init = numpy_helper.from_array(
        np.array(repeats, dtype=np.int64), name="repeats"
    )
    out_shape = [d * r for d, r in zip(input_shape, repeats)]
    Y = helper.make_tensor_value_info("Y", tp, out_shape)
    node = helper.make_node("Tile", ["X", "repeats"], ["Y"])
    return make_model_from_nodes([node], [X], [Y], initializers=[repeats_init])


class TestTile:
    @pytest.mark.parametrize(
        "dtype,shape,repeats",
        [
            (np.float16, [2, 3], [2, 4]),
            (np.float32, [1, 4], [3, 2]),
            (np.int32, [1, 4], [4, 1]),
            (np.int64, [1, 1, 4], [1, 2, 3]),
        ],
    )
    def test_tile(self, model_runner, dtype, shape, repeats):
        model = _make_tile_model(dtype, shape, repeats)
        rng = np.random.default_rng(401)
        if np.issubdtype(dtype, np.integer):
            x = rng.integers(-10, 10, shape, dtype=dtype)
        else:
            x = rng.uniform(-2.0, 2.0, shape).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)


# ---------------------------------------------------------------------------
# Expand
# ---------------------------------------------------------------------------
def _make_expand_model(dtype, input_shape: list[int], out_shape: list[int]):
    tp = np_to_onnx_type(dtype)
    X = helper.make_tensor_value_info("X", tp, list(input_shape))
    shape_init = numpy_helper.from_array(
        np.array(out_shape, dtype=np.int64), name="target_shape"
    )
    Y = helper.make_tensor_value_info("Y", tp, list(out_shape))
    node = helper.make_node("Expand", ["X", "target_shape"], ["Y"])
    return make_model_from_nodes([node], [X], [Y], initializers=[shape_init])


class TestExpand:
    @pytest.mark.parametrize(
        "dtype,in_shape,out_shape",
        [
            # Broadcast leading dim
            (np.float16, [1, 8], [4, 8]),
            # Broadcast trailing dim
            (np.float32, [4, 1], [4, 8]),
            # Pure replication of a singleton
            (np.int64, [1, 1, 1], [2, 3, 4]),
            # Llama-style attention-mask expand: [1, 1, S] -> [1, H, S]
            (np.int32, [1, 1, 16], [1, 8, 16]),
        ],
    )
    def test_expand(self, model_runner, dtype, in_shape, out_shape):
        model = _make_expand_model(dtype, in_shape, out_shape)
        rng = np.random.default_rng(402)
        if np.issubdtype(dtype, np.integer):
            x = rng.integers(-10, 10, in_shape, dtype=dtype)
        else:
            x = rng.uniform(-2.0, 2.0, in_shape).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)


# ---------------------------------------------------------------------------
# Pad
# ---------------------------------------------------------------------------
def _make_pad_model(
    dtype,
    input_shape: list[int],
    pads: list[int],
    mode: str,
    constant_value=None,
):
    """ONNX-18 Pad: data, pads, constant_value? (axes omitted)."""
    tp = np_to_onnx_type(dtype)
    X = helper.make_tensor_value_info("X", tp, list(input_shape))
    pads_init = numpy_helper.from_array(np.array(pads, dtype=np.int64), name="pads")
    initializers = [pads_init]
    input_names = ["X", "pads"]

    if constant_value is not None:
        cval_init = numpy_helper.from_array(
            np.array(constant_value, dtype=dtype), name="constant_value"
        )
        initializers.append(cval_init)
        input_names.append("constant_value")

    rank = len(input_shape)
    out_shape = [input_shape[i] + pads[i] + pads[i + rank] for i in range(rank)]
    Y = helper.make_tensor_value_info("Y", tp, out_shape)
    node = helper.make_node("Pad", input_names, ["Y"], mode=mode)
    return make_model_from_nodes([node], [X], [Y], initializers=initializers)


class TestPad:
    @pytest.mark.parametrize(
        "dtype,shape,pads,cval",
        [
            (np.float16, [2, 3], [1, 1, 1, 1], 0.0),
            (np.float32, [2, 3], [0, 2, 0, 2], 0.0),
            (np.int32, [2, 3], [1, 0, 1, 0], 0),
            # Non-zero constant value (sanity)
            (np.float16, [3, 4], [0, 1, 0, 1], 1.5),
        ],
    )
    def test_pad_constant(self, model_runner, dtype, shape, pads, cval):
        model = _make_pad_model(
            dtype, shape, pads, mode="constant", constant_value=cval
        )
        rng = np.random.default_rng(403)
        if np.issubdtype(dtype, np.integer):
            x = rng.integers(-10, 10, shape, dtype=dtype)
        else:
            x = rng.uniform(-2.0, 2.0, shape).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize("mode", ["reflect", "edge", "wrap"])
    def test_pad_modes(self, model_runner, mode):
        """Cover the three non-constant Pad modes on fp16 input."""
        shape = [3, 4]
        # 'reflect' requires |pad| < |dim| on each side -- pads of 1 on a
        # dim of size 3 / 4 is always safe.
        pads = [1, 1, 1, 1]
        model = _make_pad_model(np.float16, shape, pads, mode=mode)
        rng = np.random.default_rng(404)
        x = rng.uniform(-2.0, 2.0, shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)


# ---------------------------------------------------------------------------
# GatherND
# ---------------------------------------------------------------------------
def _make_gather_nd_model(
    dtype,
    data_shape: list[int],
    indices_shape: list[int],
    indices: np.ndarray,
    output_shape: list[int],
    batch_dims: int = 0,
):
    """GatherND: data + indices (int64 only) -> output.

    indices is materialized as an *initializer*, not a graph input -- this
    matches how vision graphs typically feed constant index tables.
    """
    tp = np_to_onnx_type(dtype)
    X = helper.make_tensor_value_info("X", tp, list(data_shape))
    indices_init = numpy_helper.from_array(
        np.array(indices, dtype=np.int64), name="indices"
    )
    Y = helper.make_tensor_value_info("Y", tp, list(output_shape))
    attrs = {}
    if batch_dims:
        attrs["batch_dims"] = batch_dims
    node = helper.make_node("GatherND", ["X", "indices"], ["Y"], **attrs)
    return make_model_from_nodes([node], [X], [Y], initializers=[indices_init])


class TestGatherND:
    def test_gather_nd_rank2_indices_last1(self, model_runner):
        """data[4,5] f16; indices[3,1] -> output[3,5] (gather rows)."""
        data_shape = [4, 5]
        indices = np.array([[0], [3], [2]], dtype=np.int64)
        out_shape = [3, 5]
        model = _make_gather_nd_model(
            np.float16, data_shape, list(indices.shape), indices, out_shape
        )
        rng = np.random.default_rng(405)
        x = rng.uniform(-2.0, 2.0, data_shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    def test_gather_nd_rank3_indices_last2(self, model_runner):
        """data[2,3,4] f32; indices[5,2] -> output[5,4] (point gather)."""
        data_shape = [2, 3, 4]
        indices = np.array(
            [[0, 0], [0, 2], [1, 1], [1, 2], [0, 1]],
            dtype=np.int64,
        )
        out_shape = [5, 4]
        model = _make_gather_nd_model(
            np.float32, data_shape, list(indices.shape), indices, out_shape
        )
        rng = np.random.default_rng(406)
        x = rng.uniform(-2.0, 2.0, data_shape).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    def test_gather_nd_batch_dims_1(self, model_runner):
        """batch_dims=1 case: data[2,3,4], indices[2,2,1] -> output[2,2,4]."""
        data_shape = [2, 3, 4]
        indices = np.array(
            [[[0], [2]], [[1], [0]]],
            dtype=np.int64,
        )
        out_shape = [2, 2, 4]
        model = _make_gather_nd_model(
            np.float16,
            data_shape,
            list(indices.shape),
            indices,
            out_shape,
            batch_dims=1,
        )
        rng = np.random.default_rng(407)
        x = rng.uniform(-2.0, 2.0, data_shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)


# ---------------------------------------------------------------------------
# Slice
# ---------------------------------------------------------------------------
#
# The graph-constant + positive-stride case is folded to
# `tensor.extract_slice` upstream; this test exercises the runtime fallback
# kernel by:
#   (a) feeding `starts` / `ends` as *graph inputs* (non-constant) -- forces
#       the runtime D2H + host-side resolution path, OR
#   (b) using negative steps -- the fold rejects this case so it falls
#       through to the runtime kernel as well.
#
# Both paths run the same `hip_slice` kernel.


def _make_slice_model(
    dtype,
    input_shape: list[int],
    starts: list[int],
    ends: list[int],
    axes: list[int] | None,
    steps: list[int] | None,
    output_shape: list[int],
    constant_indices: bool,
):
    """Build a Slice model.

    If ``constant_indices`` is True, starts/ends/axes/steps become initializers
    -- but the runtime still sees them as non-constant if any step is
    negative (the compile-time fold only handles positive steps). If
    ``constant_indices`` is False they become graph inputs, guaranteeing the
    runtime kernel path.
    """
    tp = np_to_onnx_type(dtype)
    X = helper.make_tensor_value_info("X", tp, list(input_shape))

    input_names = ["X", "starts", "ends"]
    inputs_list = [X]
    initializers: list = []

    if constant_indices:
        initializers.append(
            numpy_helper.from_array(np.array(starts, dtype=np.int64), name="starts")
        )
        initializers.append(
            numpy_helper.from_array(np.array(ends, dtype=np.int64), name="ends")
        )
    else:
        starts_vi = helper.make_tensor_value_info(
            "starts", TensorProto.INT64, [len(starts)]
        )
        ends_vi = helper.make_tensor_value_info("ends", TensorProto.INT64, [len(ends)])
        inputs_list += [starts_vi, ends_vi]

    if axes is not None:
        input_names.append("axes")
        if constant_indices:
            initializers.append(
                numpy_helper.from_array(np.array(axes, dtype=np.int64), name="axes")
            )
        else:
            axes_vi = helper.make_tensor_value_info(
                "axes", TensorProto.INT64, [len(axes)]
            )
            inputs_list.append(axes_vi)
    if steps is not None:
        input_names.append("steps")
        if constant_indices:
            initializers.append(
                numpy_helper.from_array(np.array(steps, dtype=np.int64), name="steps")
            )
        else:
            steps_vi = helper.make_tensor_value_info(
                "steps", TensorProto.INT64, [len(steps)]
            )
            inputs_list.append(steps_vi)

    Y = helper.make_tensor_value_info("Y", tp, list(output_shape))
    node = helper.make_node("Slice", input_names, ["Y"])
    return (
        make_model_from_nodes([node], inputs_list, [Y], initializers=initializers),
        inputs_list,
    )


class TestSlice:
    @pytest.mark.parametrize(
        "dtype,shape,starts,ends,axes,steps,out_shape",
        [
            # Plain positive-step slice on one axis (graph-input form -- forces
            # runtime kernel; the const+positive-step form folds upstream).
            (np.float16, [8, 6], [2], [6], [0], [1], [4, 6]),
            (np.float32, [4, 5, 3], [1, 0], [3, 2], [0, 2], [1, 1], [2, 5, 2]),
            (np.int32, [10], [3], [9], [0], [1], [6]),
            (np.int64, [4, 4], [0, 1], [4, 4], [0, 1], [1, 1], [4, 3]),
            # Strided positive step (still goes to runtime kernel because
            # we feed via graph inputs).
            (np.float32, [10], [0], [10], [0], [2], [5]),
            (np.float16, [6, 8], [0, 0], [6, 8], [0, 1], [2, 2], [3, 4]),
        ],
    )
    def test_slice_positive_step_runtime(
        self, model_runner, dtype, shape, starts, ends, axes, steps, out_shape
    ):
        """Starts/ends/axes/steps fed as graph inputs -- forces runtime path."""
        model, _ = _make_slice_model(
            dtype,
            shape,
            starts,
            ends,
            axes,
            steps,
            out_shape,
            constant_indices=False,
        )
        rng = np.random.default_rng(501)
        if np.issubdtype(dtype, np.integer):
            x = rng.integers(-50, 50, shape, dtype=dtype)
        else:
            x = rng.uniform(-2.0, 2.0, shape).astype(dtype)
        starts_a = np.array(starts, dtype=np.int64)
        ends_a = np.array(ends, dtype=np.int64)
        feeds = [x, starts_a, ends_a]
        if axes is not None:
            feeds.append(np.array(axes, dtype=np.int64))
        if steps is not None:
            feeds.append(np.array(steps, dtype=np.int64))
        actual, expected = model_runner.run_sample(model, feeds)
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize(
        "dtype,shape,starts,ends,axes,steps,out_shape",
        [
            # Negative step: reverses elements along axis 0. The compile-time
            # fold rejects negative steps -> goes through the runtime kernel
            # even with constant indices.
            #
            # Note on the negative-end sentinel:
            #   ONNX Slice spec normalises `end += dim` for any end < 0,
            #   THEN clamps. So to express "go all the way down to index 0
            #   inclusive" with step<0 you need to pass `end = -(dim+1)`
            #   (which post-normalisation becomes -1, the "before zero"
            #   sentinel that step<0 clamping permits).
            # Reverse 6-element vector: start=5, end=-7=-(6+1) -> end=-1 ->
            #   indices [5,4,3,2,1,0], 6 elements.
            (np.float16, [6], [5], [-7], [0], [-1], [6]),
            # 2-D reverse along both axes: start=(3,4), end=(-5,-6) ->
            #   normalised (-1,-1), step=(-1,-1) -> output [4,5].
            (np.float32, [4, 5], [3, 4], [-5, -6], [0, 1], [-1, -1], [4, 5]),
            # step=-2 walking 7,5,3,1 down to 0 exclusive (so 4 elements):
            #   start=7, end=-9=-(8+1) -> end=-1, step=-2 -> 4 elements.
            (np.int32, [8], [7], [-9], [0], [-2], [4]),
        ],
    )
    def test_slice_negative_step(
        self, model_runner, dtype, shape, starts, ends, axes, steps, out_shape
    ):
        """Negative step ALWAYS goes through the runtime kernel."""
        model, _ = _make_slice_model(
            dtype,
            shape,
            starts,
            ends,
            axes,
            steps,
            out_shape,
            constant_indices=True,
        )
        rng = np.random.default_rng(502)
        if np.issubdtype(dtype, np.integer):
            x = rng.integers(-50, 50, shape, dtype=dtype)
        else:
            x = rng.uniform(-2.0, 2.0, shape).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    def test_slice_negative_indices(self, model_runner):
        """Negative start/end with positive step -- runtime path, fp16."""
        # Slice [-3, -1) of a [6] vector with step 1 -> elements 3..4
        shape = [6]
        starts, ends, axes, steps = [-3], [-1], [0], [1]
        out_shape = [2]
        model, _ = _make_slice_model(
            np.float16,
            shape,
            starts,
            ends,
            axes,
            steps,
            out_shape,
            constant_indices=False,
        )
        rng = np.random.default_rng(503)
        x = rng.uniform(-2.0, 2.0, shape).astype(np.float16)
        feeds = [
            x,
            np.array(starts, dtype=np.int64),
            np.array(ends, dtype=np.int64),
            np.array(axes, dtype=np.int64),
            np.array(steps, dtype=np.int64),
        ]
        actual, expected = model_runner.run_sample(model, feeds)
        compare_outputs(actual, expected, atol=0)


# ---------------------------------------------------------------------------
# ScatterND
# ---------------------------------------------------------------------------
def _make_scatter_nd_model(
    dtype,
    data_shape: list[int],
    indices: np.ndarray,
    updates_shape: list[int],
    reduction: str | None = None,
):
    """ScatterND: data + indices (int64, initializer) + updates -> output.

    Indices live as an initializer (matches the typical graph pattern from
    Qwen / Llama vision); data and updates are graph inputs.
    """
    tp = np_to_onnx_type(dtype)
    X = helper.make_tensor_value_info("X", tp, list(data_shape))
    U = helper.make_tensor_value_info("U", tp, list(updates_shape))
    indices_init = numpy_helper.from_array(
        np.array(indices, dtype=np.int64), name="indices"
    )
    Y = helper.make_tensor_value_info("Y", tp, list(data_shape))
    attrs = {}
    if reduction is not None:
        attrs["reduction"] = reduction
    node = helper.make_node("ScatterND", ["X", "indices", "U"], ["Y"], **attrs)
    return make_model_from_nodes([node], [X, U], [Y], initializers=[indices_init])


class TestScatterND:
    @pytest.mark.parametrize(
        "dtype",
        [np.float16, np.float32, np.int32, np.int64],
    )
    def test_scatter_nd_1d_default(self, model_runner, dtype):
        """data[8], indices[4,1] -> overwrite 4 scalars."""
        data_shape = [8]
        indices = np.array([[4], [3], [1], [7]], dtype=np.int64)
        updates_shape = [4]
        model = _make_scatter_nd_model(dtype, data_shape, indices, updates_shape)
        rng = np.random.default_rng(601)
        if np.issubdtype(dtype, np.integer):
            x = rng.integers(-50, 50, data_shape, dtype=dtype)
            u = rng.integers(-50, 50, updates_shape, dtype=dtype)
        else:
            x = rng.uniform(-2.0, 2.0, data_shape).astype(dtype)
            u = rng.uniform(-2.0, 2.0, updates_shape).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x, u])
        compare_outputs(actual, expected, atol=0)

    def test_scatter_nd_3d_slice(self, model_runner):
        """data[4,4,4] f16; indices[2,1] -> overwrite 2 2-D slices."""
        data_shape = [4, 4, 4]
        indices = np.array([[1], [3]], dtype=np.int64)
        updates_shape = [2, 4, 4]
        model = _make_scatter_nd_model(np.float16, data_shape, indices, updates_shape)
        rng = np.random.default_rng(602)
        x = rng.uniform(-2.0, 2.0, data_shape).astype(np.float16)
        u = rng.uniform(-2.0, 2.0, updates_shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x, u])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize(
        "reduction,dtype",
        [
            ("add", np.float32),
            ("add", np.int32),
            ("mul", np.float32),
            ("min", np.float32),
            ("min", np.int32),
            ("max", np.float32),
            ("max", np.int32),
        ],
    )
    def test_scatter_nd_reductions(self, model_runner, reduction, dtype):
        """Cover the 4 reduction modes against ORT CPU reference.

        ORT CPU does NOT implement fp16/bf16 ``add`` or ``mul`` (see
        onnxruntime/core/providers/cpu/tensor/scatter_nd.cc:174-176), and
        ``ScatterNDReduction`` on the CUDA EP only supports float / fp16.
        We skip fp16 ``add``/``mul`` here because no CPU reference is
        available to compare against -- the GPU path itself is exercised
        indirectly via the fp32 + integer reduction cases (same kernel,
        same atomic-CAS code path, different bit-cast word width).
        """
        data_shape = [6]
        # Use NON-OVERLAPPING indices so the reduction result is deterministic
        # regardless of update order. (Race-correct atomic reductions still
        # produce different bit-exact results across runs when multiple
        # updates land on the same target.)
        indices = np.array([[0], [2], [4]], dtype=np.int64)
        updates_shape = [3]
        model = _make_scatter_nd_model(
            dtype, data_shape, indices, updates_shape, reduction=reduction
        )
        rng = np.random.default_rng(603)
        if np.issubdtype(dtype, np.integer):
            x = rng.integers(1, 10, data_shape, dtype=dtype)
            u = rng.integers(1, 10, updates_shape, dtype=dtype)
        else:
            x = rng.uniform(0.5, 2.0, data_shape).astype(dtype)
            u = rng.uniform(0.5, 2.0, updates_shape).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x, u])
        # fp16 reductions go through float-domain CAS; tolerate one ulp.
        atol = 1e-3 if dtype == np.float16 else 0
        compare_outputs(actual, expected, atol=atol)

    def test_scatter_nd_negative_index(self, model_runner):
        """Negative index normalisation: -1 -> last row."""
        data_shape = [4, 3]
        indices = np.array([[-1], [0]], dtype=np.int64)
        updates_shape = [2, 3]
        model = _make_scatter_nd_model(np.float32, data_shape, indices, updates_shape)
        rng = np.random.default_rng(604)
        x = rng.uniform(-2.0, 2.0, data_shape).astype(np.float32)
        u = rng.uniform(-2.0, 2.0, updates_shape).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x, u])
        compare_outputs(actual, expected, atol=0)
