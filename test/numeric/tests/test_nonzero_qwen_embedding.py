#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""End-to-end correctness for the Qwen-VL 9B `embedding.onnx` subgraph.

This is the canonical real-world *multi-op* dynamic-shape composition
in our test surface today. It stitches eleven distinct ONNX ops into
a single graph whose output rank is static but whose ScatterND update
volume is purely data-dependent on the input ``input_ids``:

```
Gather(embed_tokens.weight, input_ids)          -> [B, S, 4096] fp16   (static)
Equal(input_ids, IMAGE_TOKEN_ID=248056)         -> [B, S]   bool       (static)
Unsqueeze(...)                                  -> [B, S, 1]            (static)
Expand(... , Shape(Gather))                     -> [B, S, 4096]         (static)
NonZero(Expand)                                 -> [3, N]               (Category-C: N dynamic)
Transpose(NonZero, perm=[1,0])                  -> [N, 3]               (N dynamic)
Reshape(image_features, [-1])                   -> [K*4096]  fp16       (depends on K)
Shape(Transpose); Gather(Shape, 0); Unsqueeze   -> []  -> [1]           (publishes N)
Slice(Reshape, [0], [N], [0])                   -> [N]                  (Category-C)
ScatterND(Gather, Transpose, Slice)             -> [B, S, 4096] fp16    (static)
```

Why this graph is the highest-value real-world test for the
data-dependent-dynamic-output-shape path:

  * **One Category-C op (NonZero) drives at least four downstream
    consumers** -- Transpose's rank-1 dim, Shape(Transpose)'s VALUE,
    Slice's end argument, and ScatterND's indices size -- exactly the
    fan-out the ``compose-dim-specs`` pass was designed to handle.

  * **The `N` slot is referenced via a chain of Shape/Gather/Unsqueeze
    + Slice, NOT just by Reshape into the consumer**. Stale-slot or
    re-publish bugs in the runtime slot table show up as wrong-size
    Slice output, which then breaks ScatterND's update buffer.

  * **`N` lives in 0 <= N <= B*S*4096 ~= 524,288** for B=1,S=128. The
    upper bound is large enough that the EP's host-side dyn-pool
    sizing must accept it (vs the tiny synthetic cases in
    `test_nonzero_composition.py`).

  * **The graph deliberately handles N=0** through the implicit
    semantics of ScatterND with an empty indices tensor -- useful as
    the cheapest non-trivial dynamic case (the EP must publish N=0,
    allocate a zero-byte ScatterND updates buffer, and skip the
    kernel without crashing).

  * **Mixed dtypes (int64 input_ids, fp16 weights/features, i64
    intermediates from Shape/NonZero/Gather-on-Shape, bool from
    Equal)** exercise the wrap layer's input/output marshalling
    across every dtype path that any other Category-C op in the
    repository uses.

Two scenarios are covered:

  1. **K=1** -- a single IMAGE_TOKEN_ID; ``image_features=(1, 4096)``.
     N = K * 4096 = 4096 = exactly one row. Off-by-one regression
     for the host-side N-publish + Slice/ScatterND indexing.

  2. **K=4** -- four scattered IMAGE_TOKEN_IDs; ``image_features=(4, 4096)``.
     N = K * 4096 = 16384 flat elements; Slice consumes the whole
     reshape output and ScatterND routes those 16384 elements across
     four distinct ``(b, s, h)`` row groups in the embedding tensor.
     Validates the full real ScatterND-update path with non-empty
     data.

The K=0 text-only scenario (no IMAGE_TOKEN_ID, ``image_features=(0, 4096)``)
is intentionally NOT a scenario in this file because the EP MLIR
pipeline currently cannot bufferize a graph that declares a static
zero-extent input (``tensor<0x4096xf16>``) -- the compile-time
bufferization fails before the DLL can be produced. The K=0 behaviour
is fully covered by the dynamic-shape sister test
``test_nonzero_qwen_embedding_dyn.py``, where ``image_features``
stays symbolic at compile time (``tensor<?x4096xf16>``) and the K=0
case is exercised per-call via the runtime's empty-input sentinel
path (see CLAUDE.md "Empty (zero-element) Category-B inputs MUST
get a sentinel buffer, not a null"). When the EP grows support for
static zero-extent inputs, add a K=0 scenario back here -- the
expected output is the pure ``Gather(embed_tokens.weight, input_ids)``
because ScatterND with N=0 is a no-op.

Op support audit (CLAUDE.md `lib/Conversion/OnnxToHip/`):

  * Gather, Equal, Expand, Shape, NonZero, Transpose, Reshape,
    Slice, ScatterND -- all have first-class conversions.
  * Unsqueeze is a zero-cost shape op handled via
    `tensor.expand_shape` (per CLAUDE.md "Adding a New Operator").
  * Constant is folded.

so NO op should fall back to CPU.

History
-------

This test was initially blocked by a framework gap: ``Shape(hip.transpose
(...))`` could not lower because ``hip.transpose`` carried no
``output_dim_specs`` attribute, so ``ShapeToHip``'s walk of the producer
chain returned empty and the ``onnx.Shape`` op survived into bufferize
with ``error: op was not bufferized``. Closed by the per-op **DimSpec
builder registry** in ``shape_interface`` (``HipDialect::initialize``
calls ``populateBuiltinDimSpecBuilders``); see
``docs/design/compiler-runtime-contract.md`` for the design.

The follow-on gap was ``wrap_expand`` not supporting ``bool`` (``ui8``).
The Qwen graph emits ``Equal(input_ids, IMAGE_TOKEN_ID)`` -> 1-byte
predicate -> ``Expand`` to ``[B,S,4096]`` -> ``NonZero``. With Expand a
no-op, NonZero scanned uninitialised pool memory; scenario 1 happened
to find N=0 (coincidentally correct in shape but pointer arithmetic
downstream made the output bit-divergent from CPU), and scenarios 2/3
hung outright. Closed by mirroring the ``wrap_nonzero`` convention --
``HIPDNN_EP_DATATYPE_{INT8,UINT8}`` both map to ``HIP_DTYPE_INT8`` in
``wrap_expand``, and ``hip_expand_kernel`` now instantiates an ``int8_t``
template path (pure 1-byte copy, signedness irrelevant). Same change
shipped together for both the runtime wrap and the kernel switch.

Model contract (2026-05)
------------------------

The current ``Qwen3.5-9B-rtn-int4-int8-128gs-fp16-onnx-gpu/embedding.onnx``
checkpoint declares ``image_features`` as a **graph input** of shape
``[num_logical_patches, 4096]`` fp16 -- NOT a static initializer.
(An older snapshot of the same checkpoint family shipped it as an
initializer of shape ``[<rows>, 2048]`` fp16; the test was originally
authored against that snapshot but has since been rewritten to feed
``image_features`` per scenario in the same way the
``test_nonzero_qwen3_35b_a3b_embedding.py`` sister file does.)

Because the EP today requires fully static input shapes (CLAUDE.md
"Test Models"), each scenario binds ``num_logical_patches`` to its
concrete ``K`` via ``_bind_num_logical_patches`` and feeds an actual
``(K, 4096)`` fp16 buffer. The 1:1 row-to-hidden ratio (``hidden ==
image_features_cols == 4096``) means ``K`` image-token positions
consume exactly ``K`` rows of features after ``Reshape -> [-1]``.

The full model file (``embedding.onnx`` + ``embedding.onnx.data``) is
~2 GB on disk; we load it once at module scope and re-use the
``ModelProto`` across all scenarios via deepcopy.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import onnx
import pytest

from framework.comparator import compare_outputs


_QWEN_EMBEDDING_ONNX = Path(
    "D:/Develop/m/models/Qwen3.5-9B-rtn-int4-int8-128gs-fp16-onnx-gpu/embedding.onnx"
)

# Intrinsic properties of the embedding.onnx graph, verified at
# fixture-load time. If any of these change the model is a different
# graph and this test should be re-authored, not silently adapted.
_IMAGE_TOKEN_ID = 248056  # Equal's constant operand
_HIDDEN = 4096  # embed_tokens.weight cols / inputs_embeds last dim
# image_features cols (same as hidden -> 1:1 row mapping: each
# IMAGE_TOKEN_ID position consumes exactly one row of features after
# Reshape -> [-1]).
_IMAGE_FEATURES_COLS = 4096
_BATCH = 1
_SEQ = 128


@pytest.fixture(scope="module")
def qwen_embedding_model():
    """Load the real Qwen-VL 9B embedding ONNX with all external data inlined.

    Module-scope to avoid a ~2 GB reload between scenarios. The
    returned ``ModelProto`` has ``batch_size`` / ``sequence_length``
    pinned to 1 / 128. ``num_logical_patches`` stays dynamic at the
    fixture level -- each test rebinds it via ``_bind_num_logical_patches``
    because the K differs per scenario and the EP does not accept
    dynamic input shapes (CLAUDE.md "Test Models").
    """
    if not _QWEN_EMBEDDING_ONNX.exists():
        pytest.skip(
            f"Qwen embedding model not found at {_QWEN_EMBEDDING_ONNX}. "
            "This test depends on a local Qwen3.5-9B checkpoint."
        )

    m = onnx.load(str(_QWEN_EMBEDDING_ONNX), load_external_data=True)

    for tv in list(m.graph.input) + list(m.graph.output):
        for d in tv.type.tensor_type.shape.dim:
            if d.dim_param == "batch_size":
                d.Clear()
                d.dim_value = _BATCH
            elif d.dim_param == "sequence_length":
                d.Clear()
                d.dim_value = _SEQ

    _assert_graph_invariants(m)
    return m


def _assert_graph_invariants(m: onnx.ModelProto) -> None:
    """Fail loud if the model isn't shaped like the one this test was authored against."""
    op_counts: dict[str, int] = {}
    for n in m.graph.node:
        op_counts[n.op_type] = op_counts.get(n.op_type, 0) + 1
    expected_ops = {
        "Gather": 2,
        "Equal": 1,
        "Expand": 2,
        "NonZero": 1,
        "Transpose": 1,
        "Reshape": 1,
        "Slice": 1,
        "ScatterND": 1,
        "Shape": 3,
        "Unsqueeze": 2,
        "Constant": 7,
    }
    for op, n in expected_ops.items():
        if op_counts.get(op, 0) != n:
            raise RuntimeError(
                f"embedding.onnx structure mismatch: expected {n}x {op}, "
                f"saw {op_counts.get(op, 0)}. Test was authored for a "
                f"specific graph; please re-verify."
            )

    # IMAGE_TOKEN_ID is the scalar value of the Constant node feeding
    # Equal. If this changes, scenarios with K>0 will silently inject
    # zero image tokens (no Equal matches).
    for n in m.graph.node:
        if n.op_type == "Equal":
            const_in = n.input[1]
            for cn in m.graph.node:
                if cn.op_type == "Constant" and cn.output[0] == const_in:
                    for a in cn.attribute:
                        if a.name == "value":
                            v = onnx.numpy_helper.to_array(a.t)
                            if int(v) != _IMAGE_TOKEN_ID:
                                raise RuntimeError(
                                    f"IMAGE_TOKEN_ID changed: graph says "
                                    f"{int(v)}, test was authored for "
                                    f"{_IMAGE_TOKEN_ID}."
                                )

    # image_features must be a graph input (not an initializer) in the
    # current checkpoint. The old initializer-based snapshot is no
    # longer supported; if you hit this assertion the model on disk is
    # the legacy one and this test needs to be re-pointed (or the dyn
    # sister test ``test_nonzero_qwen_embedding_dyn.py`` used instead).
    has_image_input = any(tv.name == "image_features" for tv in m.graph.input)
    has_image_init = any(ini.name == "image_features" for ini in m.graph.initializer)
    if not has_image_input or has_image_init:
        raise RuntimeError(
            "embedding.onnx contract mismatch: expected image_features as "
            "a graph INPUT (no initializer). Got "
            f"input={has_image_input}, initializer={has_image_init}. "
            "Test was authored for the current snapshot in which "
            "image_features is a [num_logical_patches, 4096] fp16 input."
        )


def _bind_num_logical_patches(m: onnx.ModelProto, k: int) -> onnx.ModelProto:
    """Deep-copy ``m`` and rebind the ``num_logical_patches`` dim_param to k.

    The EP requires fully static input shapes (see CLAUDE.md "Test
    Models" -- compiler does not support dynamic dims). ``k`` is the
    number of image tokens in ``input_ids``; the resulting graph
    accepts a concrete ``image_features`` of shape ``(k, 4096)``.

    Passing ``k=0`` produces a graph that accepts a literal empty
    ``image_features`` tensor (the text-only inference path).
    """
    out = copy.deepcopy(m)
    for tv in list(out.graph.input) + list(out.graph.output):
        for d in tv.type.tensor_type.shape.dim:
            if d.dim_param == "num_logical_patches":
                d.Clear()
                d.dim_value = k
    return out


def _make_image_features(k: int) -> np.ndarray:
    """Synthetic ``[k, 4096]`` fp16 features with deterministic values.

    Each row is filled with a single distinct value (``(r+1) * 0.1``)
    so the post-Scatter routing can be validated bit-exactly: if
    ScatterND or the NonZero scan order routes the wrong row to the
    wrong position, the per-row equality check in the K=4 scenario
    catches it with a precise (i, position) diagnostic.

    ``k=0`` returns the legal empty ``(0, 4096)`` tensor.
    """
    if k == 0:
        return np.zeros((0, _IMAGE_FEATURES_COLS), dtype=np.float16)
    rows = []
    for r in range(k):
        rows.append(
            np.full(_IMAGE_FEATURES_COLS, fill_value=(r + 1) * 0.1, dtype=np.float16)
        )
    return np.stack(rows, axis=0)


class TestQwenEmbeddingComposition:
    """End-to-end NonZero-driven multi-op dynamic-shape composition.

    K=0 (text-only inference, ``image_features=(0, 4096)``) is
    deliberately NOT covered here -- the EP MLIR pipeline currently
    cannot bufferize a graph with a static zero-extent input, so the
    DLL never gets produced. K=0 IS exercised end-to-end in the
    dynamic-shape sister test
    ``test_nonzero_qwen_embedding_dyn.py::TestQwen9bEmbeddingDyn::test_per_call_shape_switching``
    (first iteration of its ``_SHAPES`` list: ``(1, 128, 0)``), where
    ``image_features`` stays symbolic at compile time
    (``tensor<?x4096xf16>``) and the K=0 case is handled by the
    runtime's empty-input sentinel path. See the module docstring at
    the top of this file for the full rationale.
    """

    def test_n_positive_k1_single_image_token(
        self, request, model_runner, qwen_embedding_model
    ):
        """Scenario 2 smallest interesting K=1 case.

        Useful regression for off-by-one host-side N-publish bugs:
        N = 1*4096 = 4096 (exactly one row), so any indexing error
        in the Slice/ScatterND chain manifests as either a missing
        update or a one-row-off update -- both caught structurally.
        """
        k = 1
        image_position = 64
        input_ids = np.full((_BATCH, _SEQ), 1, dtype=np.int64)
        input_ids[0, image_position] = _IMAGE_TOKEN_ID

        image_features = _make_image_features(k)
        assert image_features.shape == (1, _IMAGE_FEATURES_COLS)

        m_bound = _bind_num_logical_patches(qwen_embedding_model, k)

        actual, expected = model_runner.run_sample(
            m_bound, [input_ids, image_features], reference="cache"
        )

        compare_outputs(actual, expected, atol=0.0, rtol=0.0)

        # The IMAGE_TOKEN_ID position must have been overwritten by
        # ScatterND with the (only) image feature row. Every other
        # position must equal the embedding row for token=1.
        assert actual[0].shape == (_BATCH, _SEQ, _HIDDEN)
        baseline = actual[0][0, 0, :]
        assert not np.array_equal(actual[0][0, image_position, :], baseline), (
            f"image-token position {image_position} was not overwritten by ScatterND"
        )
        assert np.array_equal(actual[0][0, image_position, :], image_features[0]), (
            "ScatterND wrote the wrong row to the image position"
        )
        for s in range(_SEQ):
            if s == image_position:
                continue
            assert np.array_equal(actual[0][0, s, :], baseline), (
                f"non-image position {s} differs from token=1 baseline"
            )

    def test_n_positive_k4_image_tokens(
        self, request, model_runner, qwen_embedding_model
    ):
        """Scenario 3: K=4 IMAGE_TOKEN_IDs with multi-destination scatter.

        ``image_features`` is ``[4, 4096]`` fp16 -> Reshape -> [-1]
        gives exactly K * 4096 = 16384 flat elements, which Slice
        consumes whole and ScatterND routes across four distinct
        (b, s, h) row groups in the embedding tensor.

        Validates the full real ScatterND-update path with non-empty
        data, where every byte of the dynamic Slice output gets
        routed to the correct (Transpose-published) destination index
        in the [B,S,4096] embedding tensor.
        """
        k = 4
        image_positions = [10, 50, 100, 120]
        assert len(image_positions) == k and max(image_positions) < _SEQ

        input_ids = np.full((_BATCH, _SEQ), 1, dtype=np.int64)
        for p in image_positions:
            input_ids[0, p] = _IMAGE_TOKEN_ID

        image_features = _make_image_features(k)
        assert image_features.shape == (4, _IMAGE_FEATURES_COLS)

        m_bound = _bind_num_logical_patches(qwen_embedding_model, k)

        actual, expected = model_runner.run_sample(
            m_bound, [input_ids, image_features], reference="cache"
        )

        compare_outputs(actual, expected, atol=0.0, rtol=0.0)

        # Structural sanity beyond compare_outputs.
        assert actual[0].shape == (_BATCH, _SEQ, _HIDDEN)
        baseline = actual[0][0, 0, :]
        # NonZero scan order is row-major over the [B,S,H] mask: for
        # mask cell (0, s, h) with s ascending, every h for s is
        # contiguous. So image_position[i] receives image_features[i].
        # If the EP's NonZero changes scan order this assertion catches
        # it.
        for i, p in enumerate(image_positions):
            assert not np.array_equal(actual[0][0, p, :], baseline), (
                f"image-token position {p} was not overwritten by ScatterND"
            )
            assert np.array_equal(actual[0][0, p, :], image_features[i]), (
                f"image position {p} (i={i}) received the wrong image_features "
                "row -- ScatterND or NonZero scan order is wrong"
            )
        for s in range(_SEQ):
            if s in image_positions:
                continue
            assert np.array_equal(actual[0][0, s, :], baseline), (
                f"non-image position {s} differs from token=1 baseline"
            )
