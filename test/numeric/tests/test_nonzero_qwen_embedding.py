#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""End-to-end correctness for the Qwen-VL `embedding.onnx` subgraph.

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
Reshape(image_features, [-1])                   -> [F]   fp16           (depends on image_features rows)
Shape(Transpose); Gather(Shape, 0); Unsqueeze   -> []  -> [1]           (publishes N)
Slice(Reshape, [0], [N], [0])                   -> [N]                  (Category-C)
ScatterND(Gather, Transpose, Slice)             -> [B, S, 4096] fp16    (static)
```

Why this graph is the highest-value real-world test for the
data-dependent-dynamic-output-shape path:

  * **One Category-C op (NonZero) drives at least four downstream
    consumers** — Transpose's rank-1 dim, Shape(Transpose)'s VALUE,
    Slice's end argument, and ScatterND's indices size — exactly the
    fan-out the ``compose-dim-specs`` pass was designed to handle.

  * **The `N` slot is referenced via a chain of Shape/Gather/Unsqueeze
    + Slice, NOT just by Reshape into the consumer**. Stale-slot or
    re-publish bugs in the runtime slot table show up as wrong-size
    Slice output, which then breaks ScatterND's update buffer.

  * **`N` lives in 0 ≤ N ≤ B·S·4096 ≈ 524,288** for B=1,S=128. The
    upper bound is large enough that the EP's host-side dyn-pool
    sizing must accept it (vs the tiny synthetic cases in
    `test_nonzero_composition.py`).

  * **The graph deliberately handles N=0** through the implicit
    semantics of ScatterND with an empty indices tensor — useful as
    the cheapest non-trivial dynamic case (the EP must publish N=0,
    allocate a zero-byte ScatterND updates buffer, and skip the
    kernel without crashing).

  * **Mixed dtypes (int64 input_ids, fp16 weights/features, i64
    intermediates from Shape/NonZero/Gather-on-Shape, bool from
    Equal)** exercise the wrap layer's input/output marshalling
    across every dtype path that any other Category-C op in the
    repository uses.

Two scenarios are covered:

  1. **N=0** — no IMAGE_TOKEN_ID in ``input_ids``. NonZero's output
     is ``[3, 0]``, Slice's output is ``[0]``, ScatterND is a no-op,
     and the final output equals the pure ``Gather`` lookup. Verifies
     the Category-C "empty" edge case end-to-end.

  2. **N>0** — K=4 IMAGE_TOKEN_IDs scattered through ``input_ids``,
     and ``image_features`` is replaced with a synthetic
     ``[2K, 2048]`` fp16 tensor. After ``Reshape→[-1]`` this gives
     ``2K·2048 = K·4096`` flat elements — exactly enough to feed
     ``Slice([0:K*4096])``. Verifies the actual scatter-update path
     where every byte of the dynamic Slice output gets routed to the
     correct ScatterND destination.

Op support audit (CLAUDE.md `lib/Conversion/OnnxToHip/`):

  * Gather, Equal, Expand, Shape, NonZero, Transpose, Reshape,
    Slice, ScatterND — all have first-class conversions.
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
The Qwen graph emits ``Equal(input_ids, IMAGE_TOKEN_ID)`` → 1-byte
predicate → ``Expand`` to ``[B,S,4096]`` → ``NonZero``. With Expand a
no-op, NonZero scanned uninitialised pool memory; scenario 1 happened
to find N=0 (coincidentally correct in shape but pointer arithmetic
downstream made the output bit-divergent from CPU), and scenarios 2/3
hung outright. Closed by mirroring the ``wrap_nonzero`` convention —
``HIPDNN_EP_DATATYPE_{INT8,UINT8}`` both map to ``HIP_DTYPE_INT8`` in
``wrap_expand``, and ``hip_expand_kernel`` now instantiates an ``int8_t``
template path (pure 1-byte copy, signedness irrelevant). Same change
shipped together for both the runtime wrap and the kernel switch.

The full model file (``embedding.onnx`` + ``embedding.onnx.data``) is
~2 GB on disk; we load it once at module scope and re-use the
``ModelProto`` across both scenarios via deepcopy.
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

_QWEN_EMBEDDING_ONNX = Path(
    "D:/Develop/m/models/Qwen3.5-9B-rtn-int4-int8-128gs-fp16-onnx-gpu/embedding.onnx"
)

# These are intrinsic properties of the embedding.onnx graph and are
# verified at fixture-load time. They MUST match the inspection done
# during test authoring -- if any of them change the model is a
# different graph and this test should be re-authored, not silently
# adapted.
_IMAGE_TOKEN_ID = 248056  # Equal's constant operand
_HIDDEN = 4096  # embed_tokens.weight cols
_IMAGE_FEATURES_COLS = 2048  # image_features cols (after Reshape->[-1] flattens)
_BATCH = 1
_SEQ = 128


@pytest.fixture(scope="module")
def qwen_embedding_model():
    """Load the real Qwen embedding ONNX with all external data inlined.

    Module-scope to avoid a ~2 GB reload between scenarios. The
    returned ``ModelProto`` already has its `batch_size` /
    `sequence_length` dim_params bound to 1 / 128 -- the EP does not
    accept dynamic input shapes today (see CLAUDE.md "Test Models").
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
    # Equal. If this changes, scenario 2 will silently inject 0 image
    # tokens.
    for n in m.graph.node:
        if n.op_type == "Equal":
            for a in n.attribute:
                pass
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


def _substitute_image_features(m: onnx.ModelProto, rows: int) -> onnx.ModelProto:
    """Replace the empty ``image_features`` initializer with a synthetic ``[rows, 2048]`` fp16 tensor.

    Values are a deterministic arange so the test is reproducible
    without depending on any rng seed contract. ``rows`` MUST be at
    least ``2*K`` where ``K`` is the number of IMAGE_TOKEN_ID
    positions in ``input_ids`` -- otherwise the Reshape->[-1]->Slice
    chain underfeeds ScatterND.
    """
    out = copy.deepcopy(m)
    fill = np.arange(rows * _IMAGE_FEATURES_COLS, dtype=np.float16).reshape(
        rows, _IMAGE_FEATURES_COLS
    ) * np.float16(0.01)
    new_init = onnx.numpy_helper.from_array(fill, name="image_features")
    replaced = False
    for i, ini in enumerate(out.graph.initializer):
        if ini.name == "image_features":
            out.graph.initializer[i].CopyFrom(new_init)
            replaced = True
            break
    if not replaced:
        raise RuntimeError("image_features initializer not present in graph")
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestQwenEmbeddingComposition:
    """End-to-end NonZero-driven multi-op dynamic-shape composition."""

    def test_n_zero_no_image_tokens(self, request, model_runner, qwen_embedding_model):
        """Scenario 1: no IMAGE_TOKEN_ID -> NonZero N=0 -> ScatterND no-op.

        Output must equal a pure Gather lookup. This exercises the
        Category-C empty-tensor path through the entire downstream
        chain (Transpose, Shape, Gather, Unsqueeze, Slice, ScatterND
        all see N=0 shapes).
        """
        input_ids = np.full((_BATCH, _SEQ), 1, dtype=np.int64)
        # Defensively confirm no token matches IMAGE_TOKEN_ID (otherwise
        # we'd silently fall into scenario 2 with the default empty
        # image_features and ScatterND would have nothing to scatter
        # anyway, but the slot table would publish N != 0).
        assert (input_ids == _IMAGE_TOKEN_ID).sum() == 0, (
            "scenario 1 must have zero IMAGE_TOKEN_IDs in input_ids"
        )

        actual, expected = model_runner.run_sample(
            qwen_embedding_model, [input_ids], reference="cache"
        )

        # fp16 lookup: bit-exact comparison is fair (it's just a Gather).
        compare_outputs(actual, expected, atol=0.0, rtol=0.0)

        # Sanity: the output should equal the embedding row for token=1,
        # broadcast over every position.
        assert actual[0].shape == (_BATCH, _SEQ, _HIDDEN)
        baseline = actual[0][0, 0, :]
        for s in range(1, _SEQ):
            assert np.array_equal(actual[0][0, s, :], baseline), (
                f"row {s} differs from row 0 although all input_ids are 1"
            )

    def test_n_positive_k4_image_tokens(
        self, request, model_runner, qwen_embedding_model
    ):
        """Scenario 2: K=4 IMAGE_TOKEN_IDs + synthetic image_features.

        ``image_features`` is set to ``[8, 2048]`` fp16 (=2K rows ->
        2K*2048 = K*4096 = N flat elements after Reshape->[-1]),
        which exactly feeds Slice([0:N]) into ScatterND.updates.

        Validates the full real ScatterND-update path with non-empty
        data, where every byte of the dynamic Slice output gets
        routed to the correct (Transpose-published) destination index
        in the [B,S,4096] embedding tensor.
        """
        K = 4
        image_positions = [10, 50, 100, 120]
        assert len(image_positions) == K and max(image_positions) < _SEQ

        input_ids = np.full((_BATCH, _SEQ), 1, dtype=np.int64)
        for p in image_positions:
            input_ids[0, p] = _IMAGE_TOKEN_ID

        m2 = _substitute_image_features(qwen_embedding_model, rows=2 * K)

        actual, expected = model_runner.run_sample(m2, [input_ids], reference="cache")

        # fp16 ScatterND: bit-exact (no arithmetic).
        compare_outputs(actual, expected, atol=0.0, rtol=0.0)

        # Structural sanity beyond compare_outputs.
        assert actual[0].shape == (_BATCH, _SEQ, _HIDDEN)
        baseline_row = actual[0][0, 0, :]
        for s in range(_SEQ):
            row = actual[0][0, s, :]
            if s in image_positions:
                assert not np.array_equal(row, baseline_row), (
                    f"image-token position {s} was not overwritten by ScatterND"
                )
            else:
                assert np.array_equal(row, baseline_row), (
                    f"non-image position {s} differs from token=1 baseline"
                )

    def test_n_positive_k1_single_image_token(
        self, request, model_runner, qwen_embedding_model
    ):
        """Scenario 2 smallest interesting K=1 case.

        Useful regression for off-by-one host-side N-publish bugs:
        N = 1*4096 = 4096 (exactly one row), so any indexing error
        in the Slice/ScatterND chain manifests as either a missing
        update or a one-row-off update -- both caught structurally.
        """
        K = 1
        image_position = 64
        input_ids = np.full((_BATCH, _SEQ), 1, dtype=np.int64)
        input_ids[0, image_position] = _IMAGE_TOKEN_ID

        m2 = _substitute_image_features(qwen_embedding_model, rows=2 * K)

        actual, expected = model_runner.run_sample(m2, [input_ids], reference="cache")

        compare_outputs(actual, expected, atol=0.0, rtol=0.0)

        baseline_row = actual[0][0, 0, :]
        assert not np.array_equal(actual[0][0, image_position, :], baseline_row)
        for s in range(_SEQ):
            if s == image_position:
                continue
            assert np.array_equal(actual[0][0, s, :], baseline_row)
