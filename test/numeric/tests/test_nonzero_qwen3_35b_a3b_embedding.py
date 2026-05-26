#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""End-to-end correctness for the Qwen3.5-35B-A3B `embedding.onnx` subgraph.

Companion to ``test_nonzero_qwen_embedding.py`` (which targets the 9B
variant). The 35B-A3B variant is the same family of Category-C
NonZero-driven dynamic-shape composition but with three concrete
structural differences that make it a distinct test target:

1. ``image_features`` is a **graph input** with a dynamic
   ``num_logical_patches`` first dim, NOT an empty initializer. The
   test must (a) bind that dim_param to a concrete K per scenario and
   (b) feed an actual ``(K, 2048)`` fp16 tensor.
2. The graph adds a ``Where(Equal, ConstantOfShape, input_ids)`` branch
   that replaces IMAGE_TOKEN_ID positions in ``input_ids`` with 0
   *before* Gather. ScatterND then OVERWRITES those positions with the
   real image features. This exercises ``Where`` + ``ConstantOfShape``
   ops the 9B graph doesn't touch.
3. ``hidden = 2048`` (not 4096 like 9B), so the Reshape->[-1]->Slice
   flatten ratio is ``K * 2048`` flat elements <-> ``(K, 2048)`` input
   shape -- a 1:1 row mapping (the 9B test needed a 2:1 mapping
   because hidden=4096 vs image_features_cols=2048).

Graph (25 nodes; 9B is 22 nodes -- the +3 are Where + ConstantOfShape
+ one extra Shape for input_ids):

```
Equal(input_ids, IMAGE_TOKEN_ID=248056)         -> [B, S]   bool
Shape(input_ids); ConstantOfShape(zeros)        -> [B, S]   i64 zeros
Where(Equal, zeros, input_ids)                  -> [B, S]   i64        (mask out IMAGE_TOKEN_IDs)
Gather(embed_tokens.weight, Where)              -> [B, S, 2048] fp16   (static)
Unsqueeze(Equal, -1); Expand; Expand            -> [B, S, 2048] bool   (static)
NonZero(Expand)                                 -> [3, N]   i64        (Category-C: N dynamic)
Transpose(NonZero, perm=[1,0])                  -> [N, 3]   i64
Reshape(image_features, [-1])                   -> [K*2048] fp16
Shape(Transpose); Gather; Unsqueeze; Slice      -> [N]      fp16       (= flattened image_features)
ScatterND(Gather, Transpose, Slice)             -> [B, S, 2048] fp16   (static output rank)
```

Two scenarios match the 9B test conventions:

  1. **K=1** -- single image token; off-by-one regression for
     N-publishing + Slice/ScatterND indexing.
  2. **K=4** -- four scattered image tokens, ``image_features=(4, 2048)``.
     Full scatter path with multiple distinct destinations.

The K=0 text-only scenario (``image_features=(0, 2048)``) is
intentionally NOT a scenario here because the EP MLIR pipeline
currently cannot bufferize a graph with a static zero-extent input
(``tensor<0x2048xf16>``) -- the compile-time bufferization fails
before the DLL can be produced. The K=0 behaviour is fully covered
by the dynamic-shape sister test
``test_nonzero_qwen3_35b_a3b_embedding_dyn.py``, where
``image_features`` stays symbolic at compile time
(``tensor<?x2048xf16>``) and the K=0 case is exercised per-call via
the runtime's empty-input sentinel path (see CLAUDE.md
"Empty (zero-element) Category-B inputs MUST get a sentinel buffer,
not a null"). When the EP grows support for static zero-extent
inputs, add a K=0 scenario back here -- the expected output is the
pure ``Gather(embed_tokens.weight, input_ids)`` because ScatterND
with N=0 is a no-op (and ``Where`` masks nothing since Equal is
all-false).

The full model file (~1 GB ``embedding.onnx.data``) is loaded once at
module scope and shared across scenarios via deepcopy.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import onnx
import pytest

from framework.comparator import compare_outputs


# ---------------------------------------------------------------------------
# Model path / fixture
# ---------------------------------------------------------------------------

_QWEN_35B_A3B_EMBEDDING_ONNX = Path(
    "D:/Develop/m/models/Qwen3.5-35B-A3B_int4_rtn_128gs_cuda/embedding.onnx"
)

# Intrinsic properties of the embedding.onnx graph, verified at
# fixture-load time. If any of these change the model is a different
# graph and this test should be re-authored, not silently adapted.
_IMAGE_TOKEN_ID = 248056  # Equal's constant operand (shared with 9B)
_HIDDEN = 2048  # embed_tokens.weight cols / inputs_embeds last dim
_IMAGE_FEATURES_COLS = 2048  # image_features cols (same as hidden -> 1:1 row mapping)
_BATCH = 1
_SEQ = 128


@pytest.fixture(scope="module")
def qwen35b_a3b_embedding_model():
    """Load the real Qwen3.5-35B-A3B embedding ONNX with all external data inlined.

    Module-scope to avoid a ~1 GB reload between scenarios. The
    returned ``ModelProto`` has ``batch_size`` / ``sequence_length``
    pinned to 1 / 128. ``num_logical_patches`` stays dynamic at the
    fixture level -- each test rebinds it via ``_bind_num_logical_patches``
    because the K differs per scenario and the EP does not accept
    dynamic input shapes (CLAUDE.md "Test Models").
    """
    if not _QWEN_35B_A3B_EMBEDDING_ONNX.exists():
        pytest.skip(
            f"Qwen3.5-35B-A3B embedding model not found at "
            f"{_QWEN_35B_A3B_EMBEDDING_ONNX}. This test depends on a "
            "local Qwen3.5-35B-A3B checkpoint."
        )

    m = onnx.load(str(_QWEN_35B_A3B_EMBEDDING_ONNX), load_external_data=True)

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
    """Fail loud if the model isn't shaped like the one this test was authored against.

    Same defensive-assertion pattern as ``test_nonzero_qwen_embedding.py``:
    catches accidental drift between the local checkpoint and the test's
    structural assumptions, so a mismatched graph fails with a clear
    error rather than silently producing wrong-shape outputs.
    """
    op_counts: dict[str, int] = {}
    for n in m.graph.node:
        op_counts[n.op_type] = op_counts.get(n.op_type, 0) + 1
    # 35B-A3B-specific counts: +1 Where, +1 ConstantOfShape vs the 9B
    # variant, fewer Shape/Unsqueeze ops because the Where branch
    # shares the Equal output.
    expected_ops = {
        "Gather": 2,
        "Equal": 1,
        "Where": 1,
        "ConstantOfShape": 1,
        "Expand": 2,
        "NonZero": 1,
        "Transpose": 1,
        "Reshape": 1,
        "Slice": 1,
        "ScatterND": 1,
        "Shape": 4,
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


def _bind_num_logical_patches(m: onnx.ModelProto, k: int) -> onnx.ModelProto:
    """Deep-copy ``m`` and rebind the ``num_logical_patches`` dim_param to k.

    The EP requires fully static input shapes (see CLAUDE.md "Test
    Models" -- compiler does not support dynamic dims). ``k`` is the
    number of image tokens in ``input_ids``; the resulting graph
    accepts a concrete ``image_features`` of shape ``(k, 2048)``.

    Passing ``k=0`` produces a graph that accepts a literal empty
    ``image_features`` tensor (one of the production paths -- text-only
    inference).
    """
    out = copy.deepcopy(m)
    for tv in list(out.graph.input) + list(out.graph.output):
        for d in tv.type.tensor_type.shape.dim:
            if d.dim_param == "num_logical_patches":
                d.Clear()
                d.dim_value = k
    return out


def _make_image_features(k: int) -> np.ndarray:
    """Synthetic ``[k, 2048]`` fp16 features with deterministic values.

    Using an arange seeded by the row index gives every row a distinct,
    visually identifiable pattern -- if ScatterND routes the wrong row
    to the wrong position the structural assertions in the tests catch
    it.
    """
    if k == 0:
        return np.zeros((0, _IMAGE_FEATURES_COLS), dtype=np.float16)
    rows = []
    for r in range(k):
        rows.append(
            np.full(_IMAGE_FEATURES_COLS, fill_value=(r + 1) * 0.1, dtype=np.float16)
        )
    return np.stack(rows, axis=0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestQwen35bA3bEmbeddingComposition:
    """End-to-end NonZero+Where+ScatterND multi-op dynamic-shape composition.

    Same family as ``TestQwenEmbeddingComposition`` (9B variant) but
    exercises (a) ``image_features`` as a real graph input, (b) the
    ``Where`` + ``ConstantOfShape`` mask-out branch, and (c) the 1:1
    flatten ratio (``hidden == image_features_cols == 2048``).

    K=0 (text-only inference, ``image_features=(0, 2048)``) is
    deliberately NOT covered here -- the EP MLIR pipeline currently
    cannot bufferize a graph with a static zero-extent input, so the
    DLL never gets produced. K=0 IS exercised end-to-end in the
    dynamic-shape sister test
    ``test_nonzero_qwen3_35b_a3b_embedding_dyn.py::TestQwen35bA3bEmbeddingDyn::test_per_call_shape_switching``
    (first iteration of its ``_SHAPES`` list: ``(1, 128, 0)``), where
    ``image_features`` stays symbolic at compile time
    (``tensor<?x2048xf16>``) and the K=0 case is handled by the
    runtime's empty-input sentinel path. See the module docstring at
    the top of this file for the full rationale.
    """

    def test_k_one_single_image_token(
        self, request, model_runner, qwen35b_a3b_embedding_model
    ):
        """Scenario 2: K=1 single image token -- off-by-one regression.

        N = K * hidden = 1 * 2048 = 2048 = one row of image_features.
        Any indexing error in the Slice/ScatterND chain manifests as
        either a missing update (image position == baseline) or a
        wrong position update (a non-image position differs from
        baseline) -- both caught structurally.

        The Where branch is now exercised non-trivially: the IMAGE_TOKEN_ID
        slot is rewritten to 0 before Gather (avoiding an OOV lookup
        for tokens >= vocab_size in real deployments), then overwritten
        by ScatterND with the image feature row.
        """
        k = 1
        image_position = 64

        input_ids = np.full((_BATCH, _SEQ), 1, dtype=np.int64)
        input_ids[0, image_position] = _IMAGE_TOKEN_ID

        image_features = _make_image_features(k)
        assert image_features.shape == (1, _IMAGE_FEATURES_COLS)

        m_bound = _bind_num_logical_patches(qwen35b_a3b_embedding_model, k)

        actual, expected = model_runner.run_sample(
            m_bound, [input_ids, image_features], reference="cache"
        )

        compare_outputs(actual, expected, atol=0.0, rtol=0.0)

        # The IMAGE_TOKEN_ID position must have been overwritten by
        # ScatterND with the actual image feature (row 0 of our
        # synthetic features). Every other position must equal the
        # embedding row for token=1.
        assert actual[0].shape == (_BATCH, _SEQ, _HIDDEN)
        baseline = actual[0][0, 0, :]
        assert not np.array_equal(actual[0][0, image_position, :], baseline), (
            f"image-token position {image_position} was not overwritten by ScatterND"
        )
        # Bit-exact: the overwritten row must equal the (only) image
        # feature row.
        assert np.array_equal(actual[0][0, image_position, :], image_features[0]), (
            "ScatterND wrote the wrong row to the image position"
        )
        for s in range(_SEQ):
            if s == image_position:
                continue
            assert np.array_equal(actual[0][0, s, :], baseline), (
                f"non-image position {s} differs from token=1 baseline"
            )

    def test_k_four_image_tokens(
        self, request, model_runner, qwen35b_a3b_embedding_model
    ):
        """Scenario 3: K=4 scattered image tokens with multi-destination scatter.

        N = K * hidden = 4 * 2048 = 8192 flat elements; image_features
        is ``(4, 2048)`` so the Reshape->[-1] gives exactly 8192
        elements that Slice([0:N]) consumes whole. ScatterND then
        routes those 8192 elements across 4 distinct (b, s, h) row
        groups in the embedding tensor.

        This is the canonical "real ScatterND-update" path: every
        byte of the dynamic Slice output gets routed to the correct
        Transpose-published destination.
        """
        k = 4
        image_positions = [10, 50, 100, 120]
        assert len(image_positions) == k and max(image_positions) < _SEQ

        input_ids = np.full((_BATCH, _SEQ), 1, dtype=np.int64)
        for p in image_positions:
            input_ids[0, p] = _IMAGE_TOKEN_ID

        image_features = _make_image_features(k)
        assert image_features.shape == (4, _IMAGE_FEATURES_COLS)

        m_bound = _bind_num_logical_patches(qwen35b_a3b_embedding_model, k)

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
                f"row -- ScatterND or NonZero scan order is wrong"
            )
        for s in range(_SEQ):
            if s in image_positions:
                continue
            assert np.array_equal(actual[0][0, s, :], baseline), (
                f"non-image position {s} differs from token=1 baseline"
            )
