#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Range op driven by a dynamic input tensor's shape (Category C).

This file targets the in-model Range pattern observed in
``Qwen3.5-35B-A3B/text.onnx`` (and every other Qwen vision-language
checkpoint we've inspected): every transformer layer emits a chain like

    data:f16[B, S, H] -> Shape -> Gather(axis=0, indices=[k]) -> Squeeze
        -> Range(0, <that scalar>, 1) -> i64[<that scalar>]

The point of the chain is to materialise a position-id vector whose
length tracks the input sequence at runtime. The Range op's ``limit``
operand is NOT a graph input -- it's an intermediate value derived
from ``Shape(data)``. Per the "operand-provenance" dispatch in
``lib/Conversion/OnnxToHip/RangeConversion.cpp``, that puts the op
firmly on the Category-C / RuntimeSlot path:

  * Compiler allocates a unique slot id, attaches
    ``DimSpec::makeRuntimeSlot(slot_id)`` to the op's
    ``output_dim_specs`` and emits ``hipdnn.elide_dps_init`` so the
    pool-allocs pass knows the upper-bound DPS init buffer is dead.
  * Runtime ``wrap_range_dyn`` (``lib/Runtime/real/range.cpp::85``)
    D2H-stages ``(start, limit, delta)``, computes ``N`` on the host,
    calls ``hipdnn_ep_state_publish_dim(state, slot_id, N)`` so the
    EP can resolve the output shape post-Compute, allocates
    ``N * elem_bytes`` from the dyn pool, publishes the pointer via
    ``hipdnn_ep_state_publish_buffer``, then launches ``hip_range``.

What this test proves
---------------------

A SINGLE ``onnxruntime.InferenceSession`` (built once over a fully
symbolic ``(B, S)`` model) serves four different ``(B, S)``
combinations back-to-back, and the EP returns the correct
``i64[S]`` Range output for each. Per-call shape switching exercises
every layer of the runtime that has to be re-keyed per call:

  * ``Shape`` reads the freshly-marshalled input shape.
  * ``Gather`` returns the chosen dim as an i64 scalar.
  * ``Squeeze`` (with explicit axes -- the no-axes form trips a
    converter gap, see the gotcha below) flattens to rank-0.
  * ``Range`` writes its output into a slot-allocated buffer whose
    size differs every call.
  * Output ``OrtValue`` is allocated post-Compute at the slot-
    published shape.

If any of those steps cached state from a previous call (a stale
runtime-slot value, a stale pool size, an autotune entry keyed on
the wrong shape), the next call would either crash or produce the
wrong shape. ``assert_subgraph_on_ep`` (via ``make_session``) makes
sure the subgraph actually landed on the MorphiZen EP -- a silent
CPU fallback would still satisfy the bit-exact comparison but
defeat the point of the test.

Two compiler gotchas surfaced while writing this fixture
--------------------------------------------------------

1. ``Squeeze`` MUST be given an explicit ``axes`` input. The opset-13+
   no-axes form ("squeeze every size-1 dim") is recognised by the
   ONNX shape inferencer but is NOT covered by the ``OnnxToHip``
   ``SqueezeToStdTensor`` pattern -- it requires 2 operands in
   ``validateSqueezeUnsqueezeOp``. The failure mode is silent at
   conversion (``onnx.Squeeze`` is left in the IR) and then surfaces
   as ``op was not bufferized`` at the bufferization pass. Diagnosing
   it requires ``tools/dump_imported_mlir.py`` + ``hip-mlir-opt
   --convert-onnx-to-hip``.

2. (Historical, FIXED.) The full in-model pattern multiplies two axes
   (``B * S``) via ``Mul(Gather(Shape, 0), Gather(Shape, 1))`` and
   needs an i64 ``Mul`` kernel. ``wrap_miopenOpTensor`` used to abort
   with "unsupported data_type 4 for MIOpen" (the MIOpen
   ``miopenOpTensor`` API only supports FLOAT / HALF / BFLOAT16), the
   output buffer was left at whatever value preceded the call (zero
   for a fresh allocation), ``Range`` consumed ``limit == 0`` /
   ``limit == 1``, and the output collapsed to a single element.
   Fixed by adding a custom ``hip_elementwise_mul`` HIP kernel
   (``3rd-party/custom_kernels/hip/elementwise_binary_kernel.hip``)
   covering ``i32`` / ``i64`` (FP types too, for symmetry with
   ``hip_elementwise_div``) and routing the integer Mul path through
   it from ``lib/Runtime/real/elementwise.cpp::wrap_miopenOpTensor``.
   The two-axis ``Range(0, B * S, 1)`` test below was previously
   ``xfail(strict=True)``; it now passes bit-exactly.
"""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.ort_cpu_backend import OrtCpuBackend


# ---------------------------------------------------------------------------
# Subgraph builders
# ---------------------------------------------------------------------------


# Hidden dim is fixed (16) so the test doesn't need to invent yet another
# free dim_param; what matters is that batch_size + sequence_length stay
# unbound.
_HIDDEN = 16


def _make_range_from_one_dim_model(dim_to_use: int) -> onnx.ModelProto:
    """Build: ``data:f16[B, S, 16] -> Range(0, data.shape[dim_to_use], 1)``.

    ``dim_to_use=1`` covers the canonical position-id case (Range up
    to the sequence length); ``dim_to_use=0`` covers the (less common
    but structurally equivalent) batch axis. Both are dynamic
    ``dim_param`` slots in the graph, so the EP can never resolve the
    output length at compile time.
    """
    data = helper.make_tensor_value_info(
        "data", TensorProto.FLOAT16, ["batch_size", "sequence_length", _HIDDEN]
    )
    output_dim = "batch_size" if dim_to_use == 0 else "sequence_length"
    out = helper.make_tensor_value_info("out", TensorProto.INT64, [output_dim])

    idx_init = numpy_helper.from_array(
        np.array([dim_to_use], dtype=np.int64), name=f"idx{dim_to_use}"
    )
    axes_init = numpy_helper.from_array(
        np.array([0], dtype=np.int64), name="squeeze_axes"
    )
    start_init = numpy_helper.from_array(np.array(0, dtype=np.int64), name="start")
    delta_init = numpy_helper.from_array(np.array(1, dtype=np.int64), name="delta")

    nodes = [
        helper.make_node("Shape", ["data"], ["shape"]),
        helper.make_node(
            "Gather", ["shape", f"idx{dim_to_use}"], ["dim_rank1"], axis=0
        ),
        helper.make_node("Squeeze", ["dim_rank1", "squeeze_axes"], ["limit"]),
        helper.make_node("Range", ["start", "limit", "delta"], ["out"]),
    ]
    graph = helper.make_graph(
        nodes,
        "range_dyn_from_shape_single",
        [data],
        [out],
        initializer=[idx_init, axes_init, start_init, delta_init],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _make_range_from_two_dim_product_model() -> onnx.ModelProto:
    """Build the full in-model pattern: ``Range(0, B * S, 1)``.

    Exercises the i64 ``Mul`` path: both ``Gather`` outputs are
    rank-1 length-1 i64 tensors and the ``Mul`` runs through the
    custom ``hip_elementwise_mul`` HIP kernel (added to plug the
    MIOpen integer gap -- see gotcha 2 in the file docstring).
    """
    data = helper.make_tensor_value_info(
        "data", TensorProto.FLOAT16, ["batch_size", "sequence_length", _HIDDEN]
    )
    out = helper.make_tensor_value_info("out", TensorProto.INT64, ["batch_seq"])

    idx0_init = numpy_helper.from_array(np.array([0], dtype=np.int64), name="idx0")
    idx1_init = numpy_helper.from_array(np.array([1], dtype=np.int64), name="idx1")
    axes_init = numpy_helper.from_array(
        np.array([0], dtype=np.int64), name="squeeze_axes"
    )
    start_init = numpy_helper.from_array(np.array(0, dtype=np.int64), name="start")
    delta_init = numpy_helper.from_array(np.array(1, dtype=np.int64), name="delta")

    nodes = [
        helper.make_node("Shape", ["data"], ["shape"]),
        helper.make_node("Gather", ["shape", "idx0"], ["batch"], axis=0),
        helper.make_node("Gather", ["shape", "idx1"], ["seq"], axis=0),
        helper.make_node("Mul", ["batch", "seq"], ["bs_rank1"]),
        helper.make_node("Squeeze", ["bs_rank1", "squeeze_axes"], ["limit"]),
        helper.make_node("Range", ["start", "limit", "delta"], ["out"]),
    ]
    graph = helper.make_graph(
        nodes,
        "range_dyn_from_shape_bs_product",
        [data],
        [out],
        initializer=[
            idx0_init,
            idx1_init,
            axes_init,
            start_init,
            delta_init,
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _make_range_from_two_dim_product_rank1_model() -> onnx.ModelProto:
    """Build the EXACT in-model topology observed in Qwen text.onnx layers.

    Reproduces verbatim what every ``*_mrope/range/Range`` node in
    ``D:\\Develop\\m\\models\\Qwen3.5-9B-rtn-int4-int8-128gs-fp16-onnx-gpu\\``
    ``text.onnx`` emits (16 layers x 2 sides {k, q} -> 16 such Range
    nodes). The wire shape that matters: **all three Range operands
    (start, limit, delta) are rank-1 length-1 i64 tensors**, NOT
    rank-0 scalars. The other ``_make_range_from_*`` builders in this
    file insert a ``Squeeze`` to flatten the ``Mul`` output to a
    scalar before Range, because that is what
    ``OnnxToHip/RangeConversion.cpp`` requires today (line 191-196:
    ``op->getOperands() ... t->getRank() != 0`` -> notifyMatchFailure).
    The Qwen graph never inserts that Squeeze -- start and delta come
    from initializers of ``shape=[1]`` and limit comes directly from
    the ``Mul(Gather(Shape, [0]), Gather(Shape, [1]))`` chain whose
    output is rank-1[1].

    Why this matters for the test surface:

    * The MorphiZen EP must either compile this exact topology, OR
      it must be documented as a known limitation that the EP relies
      on a graph-rewrite step upstream (which it currently does NOT
      have -- no L1 pass touches Range).
    * If the conversion bounces, the subgraph silently falls back to
      CPU at session creation (assert_subgraph_on_ep, called from
      framework.ort_ep_backend.make_session, will raise instead --
      ``session.disable_cpu_ep_fallback=1`` is set on the
      InferenceSession). So this test is also the canary for "did
      the EP claim the subgraph at all?".

    The graph below uses ``[B, S, 1024]`` to mirror the k_mrope shape
    family (the q_mrope side uses ``[B, S, 4096]``; the third axis is
    irrelevant to Range because only axes 0 and 1 are Gathered).
    """
    data = helper.make_tensor_value_info(
        "data", TensorProto.FLOAT16, ["batch_size", "sequence_length", 1024]
    )
    out = helper.make_tensor_value_info("out", TensorProto.INT64, ["batch_seq"])

    idx0_init = numpy_helper.from_array(np.array([0], dtype=np.int64), name="idx0")
    idx1_init = numpy_helper.from_array(np.array([1], dtype=np.int64), name="idx1")
    # Start / delta as rank-1[1] initializers (not rank-0). This is the
    # difference from _make_range_from_two_dim_product_model -- and the
    # whole point of this new builder.
    start_init = numpy_helper.from_array(np.array([0], dtype=np.int64), name="start")
    delta_init = numpy_helper.from_array(np.array([1], dtype=np.int64), name="delta")

    nodes = [
        helper.make_node("Shape", ["data"], ["shape"]),
        helper.make_node("Gather", ["shape", "idx0"], ["batch"], axis=0),
        helper.make_node("Gather", ["shape", "idx1"], ["seq"], axis=0),
        helper.make_node("Mul", ["batch", "seq"], ["limit"]),
        # NO Squeeze. limit is rank-1[1]; Range receives three rank-1[1]
        # operands -- exactly as in Qwen text.onnx.
        helper.make_node("Range", ["start", "limit", "delta"], ["out"]),
    ]
    graph = helper.make_graph(
        nodes,
        "range_dyn_from_shape_bs_product_rank1",
        [data],
        [out],
        initializer=[idx0_init, idx1_init, start_init, delta_init],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


# Per-call shapes deliberately mix small / large, batch-1 / batch-N,
# and a power-of-two / non-power-of-two sequence so the autotune cache
# and pool resizing both have to react across calls in a single session.
_SHAPES: list[tuple[int, int]] = [
    (1, 4),
    (2, 8),
    (1, 17),
    (3, 32),
    (1, 128),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRangeDynFromShape:
    """Single InferenceSession serves every ``(B, S)`` in :data:`_SHAPES`."""

    @pytest.mark.parametrize("dim_to_use", [1, 0])
    def test_range_from_single_shape_dim_per_call_switching(
        self, request, model_runner, dim_to_use
    ):
        """``Range(0, data.shape[dim_to_use], 1)`` over a dynamic ``(B, S)`` input.

        ``dim_to_use=1`` matches the position-id case from the Qwen
        mrope chains; ``dim_to_use=0`` covers the analogous batch-axis
        derivation. Both must succeed bit-exactly against ORT CPU on
        every parametrized ``(B, S)`` combination, using exactly one
        EP-bound InferenceSession.
        """
        model = _make_range_from_one_dim_model(dim_to_use)

        sub = model_runner._next_subdir(request.node.name)
        model_path = sub / "model.onnx"
        model_path.write_bytes(model.SerializeToString())

        ep_backend = model_runner.backend
        if not hasattr(ep_backend, "make_session"):
            pytest.skip(
                "Active backend does not expose make_session(); the "
                "single-session-per-shape contract requires the "
                "ort_ep backend."
            )
        ep_sess = ep_backend.make_session(str(model_path))
        cpu_backend = OrtCpuBackend()

        try:
            input_name = ep_sess.get_inputs()[0].name
            for b, s in _SHAPES:
                data = np.zeros((b, s, _HIDDEN), dtype=np.float16)
                expected = cpu_backend.run(str(model_path), [data])
                actual = ep_sess.run(None, {input_name: data})
                compare_outputs(actual, expected, atol=0, rtol=0)

                expected_len = b if dim_to_use == 0 else s
                assert actual[0].shape == (expected_len,), (
                    f"(B,S)=({b},{s}) dim_to_use={dim_to_use}: "
                    f"output shape {actual[0].shape} != ({expected_len},)"
                )
                assert actual[0].tolist() == list(range(expected_len)), (
                    f"(B,S)=({b},{s}) dim_to_use={dim_to_use}: "
                    f"Range values {actual[0].tolist()} != "
                    f"{list(range(expected_len))}"
                )
        finally:
            del ep_sess

    def test_range_from_bs_product_per_call_switching(self, request, model_runner):
        """``Range(0, B * S, 1)`` over a dynamic ``(B, S)`` input.

        Mirrors the position-id flatten pattern observed in the Qwen
        text/vision graphs: a Shape -> Gather -> Mul -> Range chain
        whose intermediate ``Mul`` is on i64. Drives the new
        ``hip_elementwise_mul`` HIP kernel (the MIOpen
        ``miopenOpTensor`` API has no integer path, so this op used
        to silently abort in the runtime; see gotcha 2 in the file
        docstring). Each of the parametrized ``(B, S)`` shapes must
        match ORT CPU bit-exactly over the same persistent
        InferenceSession.
        """
        model = _make_range_from_two_dim_product_model()

        sub = model_runner._next_subdir(request.node.name)
        model_path = sub / "model.onnx"
        model_path.write_bytes(model.SerializeToString())

        ep_backend = model_runner.backend
        if not hasattr(ep_backend, "make_session"):
            pytest.skip(
                "Active backend does not expose make_session(); the "
                "single-session-per-shape contract requires the "
                "ort_ep backend."
            )
        ep_sess = ep_backend.make_session(str(model_path))
        cpu_backend = OrtCpuBackend()

        try:
            input_name = ep_sess.get_inputs()[0].name
            for b, s in _SHAPES:
                data = np.zeros((b, s, _HIDDEN), dtype=np.float16)
                expected = cpu_backend.run(str(model_path), [data])
                actual = ep_sess.run(None, {input_name: data})
                compare_outputs(actual, expected, atol=0, rtol=0)
                assert actual[0].shape == (b * s,), (
                    f"(B,S)=({b},{s}): expected shape ({b * s},), got {actual[0].shape}"
                )
                assert actual[0].tolist() == list(range(b * s))
        finally:
            del ep_sess

    def test_range_from_bs_product_rank1_in_model_topology(self, request, model_runner):
        """Exact in-model topology: ``Range(start[1], B*S[1], delta[1])``.

        The other tests insert a ``Squeeze`` to feed Range a rank-0
        scalar limit, because that's what ``OnnxToHip/RangeConversion``
        wants. The real Qwen text.onnx graph does NOT insert that
        Squeeze; every ``*_mrope/range/Range`` node receives three
        rank-1 length-1 i64 operands. This test pins down whether
        the EP handles the exact production topology end-to-end.

        Per-call shape switching mirrors the other tests in this file:
        a single ``InferenceSession`` must serve every ``(B, S)`` in
        :data:`_SHAPES` without recompiling, and the bit-exact output
        is the position-id vector ``[0, 1, ..., B*S - 1]``.
        """
        model = _make_range_from_two_dim_product_rank1_model()

        sub = model_runner._next_subdir(request.node.name)
        model_path = sub / "model.onnx"
        model_path.write_bytes(model.SerializeToString())

        ep_backend = model_runner.backend
        if not hasattr(ep_backend, "make_session"):
            pytest.skip(
                "Active backend does not expose make_session(); the "
                "single-session-per-shape contract requires the "
                "ort_ep backend."
            )
        ep_sess = ep_backend.make_session(str(model_path))
        cpu_backend = OrtCpuBackend()

        try:
            input_name = ep_sess.get_inputs()[0].name
            for b, s in _SHAPES:
                data = np.zeros((b, s, 1024), dtype=np.float16)
                expected = cpu_backend.run(str(model_path), [data])
                actual = ep_sess.run(None, {input_name: data})
                compare_outputs(actual, expected, atol=0, rtol=0)
                assert actual[0].shape == (b * s,), (
                    f"(B,S)=({b},{s}): expected shape ({b * s},), got {actual[0].shape}"
                )
                assert actual[0].tolist() == list(range(b * s))
        finally:
            del ep_sess

    def test_range_then_reshape_via_concat_shape_vector(self, request, model_runner):
        """End-to-end: ``Range`` -> ``Reshape(Concat([B], [S]))`` -> [B, S] i64.

        Mirrors the post-Range "build positional-id grid" pattern in
        Qwen text.onnx: after ``Range(0, B*S, 1)`` produces a flat
        ``[batch_seq]`` vector, the graph reshapes it back to ``[B, S]``
        with a shape operand built by ``Concat([B], [S])`` where ``B``
        and ``S`` come from earlier ``Gather(Shape(data), 0|1)`` nodes.

        This exercises three previously broken pieces simultaneously:

          * ONNX ``Concat`` lowering (was MISSING -- silently left in
            the IR, failing at bufferisation), now lowered to
            ``tensor.from_elements`` by ``ConcatConversion.cpp``.
          * ``Reshape`` with the shape operand being a runtime
            ``tensor.from_elements`` of i64 scalars: previously the
            same-rank/multi-dyn paths bailed; ``ReshapeConversion`` now
            reads sizes from the shape operand element-by-element when
            ``buildExpandShapeOutputShape``'s single-dyn-per-group
            heuristic isn't enough.
          * Rank-1[batch_seq] -> rank-2[B, S] expand with TWO dyn dims
            in the same reassociation group -- the canonical
            ``buildExpandShapeOutputShapeFromShape`` trigger.

        If any of the three regresses, the test fails at session creation
        (assert_subgraph_on_ep raises -- CPU fallback is disabled) or
        with a shape / value mismatch against the ORT CPU reference.
        """
        data = helper.make_tensor_value_info(
            "data",
            TensorProto.FLOAT16,
            ["batch_size", "sequence_length", _HIDDEN],
        )
        out = helper.make_tensor_value_info(
            "out", TensorProto.INT64, ["batch_size", "sequence_length"]
        )

        idx0_init = numpy_helper.from_array(np.array([0], dtype=np.int64), name="idx0")
        idx1_init = numpy_helper.from_array(np.array([1], dtype=np.int64), name="idx1")
        axes_init = numpy_helper.from_array(
            np.array([0], dtype=np.int64), name="squeeze_axes"
        )
        start_init = numpy_helper.from_array(np.array(0, dtype=np.int64), name="start")
        delta_init = numpy_helper.from_array(np.array(1, dtype=np.int64), name="delta")

        nodes = [
            helper.make_node("Shape", ["data"], ["shape"]),
            helper.make_node("Gather", ["shape", "idx0"], ["B_r1"], axis=0),
            helper.make_node("Gather", ["shape", "idx1"], ["S_r1"], axis=0),
            # Build the flat range Range(0, B*S, 1).
            helper.make_node("Mul", ["B_r1", "S_r1"], ["bs_r1"]),
            helper.make_node("Squeeze", ["bs_r1", "squeeze_axes"], ["limit"]),
            helper.make_node("Range", ["start", "limit", "delta"], ["flat_pos"]),
            # Build the shape vector via Concat -- the previously-missing op.
            helper.make_node("Concat", ["B_r1", "S_r1"], ["pos_shape"], axis=0),
            # Reshape flat -> [B, S]. Both dyn dims live in one reassoc
            # group, so the converter MUST read sizes from `pos_shape`
            # (not derive from input.dim(0)).
            helper.make_node("Reshape", ["flat_pos", "pos_shape"], ["out"]),
        ]
        graph = helper.make_graph(
            nodes,
            "range_then_reshape_via_concat",
            [data],
            [out],
            initializer=[
                idx0_init,
                idx1_init,
                axes_init,
                start_init,
                delta_init,
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        sub = model_runner._next_subdir(request.node.name)
        model_path = sub / "model.onnx"
        model_path.write_bytes(model.SerializeToString())

        ep_backend = model_runner.backend
        if not hasattr(ep_backend, "make_session"):
            pytest.skip(
                "Active backend does not expose make_session(); the "
                "single-session-per-shape contract requires the "
                "ort_ep backend."
            )
        ep_sess = ep_backend.make_session(str(model_path))
        cpu_backend = OrtCpuBackend()

        try:
            input_name = ep_sess.get_inputs()[0].name
            for b, s in _SHAPES:
                data_in = np.zeros((b, s, _HIDDEN), dtype=np.float16)
                expected = cpu_backend.run(str(model_path), [data_in])
                actual = ep_sess.run(None, {input_name: data_in})
                compare_outputs(actual, expected, atol=0, rtol=0)
                assert actual[0].shape == (b, s), (
                    f"(B,S)=({b},{s}): expected shape ({b},{s}), got {actual[0].shape}"
                )
                # Row-major reshape of [0..b*s) into [B,S] is the
                # canonical positional grid; explicit equality catches
                # a reshape that picks up wrong (B,S) ordering.
                expected_grid = np.arange(b * s, dtype=np.int64).reshape(b, s)
                np.testing.assert_array_equal(actual[0], expected_grid)
        finally:
            del ep_sess
