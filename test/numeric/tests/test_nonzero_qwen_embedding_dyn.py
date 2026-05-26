#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Fully-dynamic-input variant of ``test_nonzero_qwen_embedding.py``.

Where the original test pins ``batch_size`` / ``sequence_length`` to
``1`` / ``128`` before handing the ``ModelProto`` to the EP, this one
leaves both ``dim_param`` slots symbolic and exercises the EP via a
SINGLE :class:`onnxruntime.InferenceSession` that accepts every
parametrized ``(B, S, K)`` combination. The point is to prove (a) the
MorphiZen MLIR compiler can lower the embedding graph without any
``dim_value`` substitution and (b) per-call shape switching works at
the ORT/EP boundary -- the same compiled DLL must service four shape
combinations back-to-back.

The four shape tuples deliberately overlap on some axes and disagree
on others:

  ``(1, 128, 0)``  -- text-only inference, no IMAGE_TOKEN_ID
  ``(1, 128, 4)``  -- same (B, S) as #1 but K flips to 4 image tokens
  ``(2, 64, 4)``   -- batch dim switches, sequence shrinks, K stays at 4
  ``(1, 256, 1)``  -- sequence dim grows, K shrinks back to 1

so any "stale cached shape" bug in the runtime (slot tables, pool sizes,
autotune cache keys, kernel launch geometries) surfaces as a wrong-shape
output rather than silently producing the right answer for one shape
and the wrong one for the next.

Why this matters
----------------

The static sister test (``test_nonzero_qwen_embedding.py``) only proves
the EP can compile *one* fully-bound graph. This test additionally
proves:

1. ``--hip-add-context-arg`` + ``convert-onnx-to-hip`` survive
   ``tensor<?x?xf16>`` etc. all the way through to LLVM lowering.
2. ``Equal`` no longer trips the ``resultIsStatic`` guard (Phase 2a
   of the dyn-input plan) -- the broadcast target shape is now built
   from runtime ``tensor.dim`` ops.
3. The dyn-input ``Unsqueeze`` / ``Slice`` paths (Phase 2c LIT
   coverage) hold under real graph composition rather than just
   isolated single-op MLIR fixtures.
4. The per-Compute runtime state (dyn pool, autotune cache,
   GQA seqlens cache, etc.) does not leak state across shape switches.

``image_features`` (model contract)
-----------------------------------

In the current
``Qwen3.5-9B-rtn-int4-int8-128gs-fp16-onnx-gpu/embedding.onnx`` checkpoint
``image_features`` is a graph INPUT of shape
``[num_logical_patches, 4096]`` fp16, NOT a static initializer.
(Older snapshots of this checkpoint family shipped it as an
initializer with second dim ``2048``; if you hit the in-tree static
test
``test_nonzero_qwen_embedding.py::_substitute_image_features`` raising
``image_features initializer not present in graph`` that is the model
contract drift, not a regression here.)

The per-call ``Reshape(image_features, [-1]) -> Slice([0:K*4096])``
chain expects exactly ``K * 4096`` flat fp16 elements. The natural
sizing is ``image_features.shape = (K, 4096)`` -- one row per
IMAGE_TOKEN_ID slot. ``K=0`` is the legal empty tensor ``(0, 4096)``.
Each test call feeds a fresh, deterministically-filled fp16 buffer of
that exact size; no graph-level shape rewriting is needed.

``assert_subgraph_on_ep`` (Phase 0) is called via ``make_session``
inside the EP backend's session-creation path, so a silent CPU
fallback would fail the test at session construction rather than
yielding (potentially correct-looking) CPU outputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest

from framework.comparator import compare_outputs
from framework.ort_cpu_backend import OrtCpuBackend


# ---------------------------------------------------------------------------
# Model path / fixture (matches the static test for consistency)
# ---------------------------------------------------------------------------

_QWEN_EMBEDDING_ONNX = Path(
    "D:/Develop/m/models/Qwen3.5-9B-rtn-int4-int8-128gs-fp16-onnx-gpu/embedding.onnx"
)

_IMAGE_TOKEN_ID = 248056
_HIDDEN = 4096
# Width of every per-call image_features row. Pinned to 4096 to match
# the current checkpoint's declared dim_param-paired layout
# (image_features: f16[num_logical_patches, 4096]); the Reshape->[-1]
# downstream of it produces exactly K * 4096 flat elements when there
# are K rows, which is what Slice consumes.
_IMAGE_FEATURES_COLS = 4096

# Per-call shape combinations. (B, S, K) where:
#   B = batch_size  (left symbolic in the model)
#   S = sequence_length (left symbolic in the model)
#   K = number of IMAGE_TOKEN_ID positions in input_ids. Each call
#       feeds image_features.shape = (K, 4096) -- the natural per-call
#       sizing for a graph that consumes K * 4096 flat features.
_SHAPES: list[tuple[int, int, int]] = [
    (1, 128, 0),
    (1, 128, 4),
    (2, 64, 4),
    (1, 256, 1),
]


@pytest.fixture(scope="module")
def qwen_embedding_model_dyn():
    """Load the 9B embedding ONNX with ``batch_size`` / ``sequence_length`` symbolic.

    Both dim_params stay unbound at the graph level (the whole point of
    this file). ``image_features`` is a graph input in this checkpoint
    so there is nothing to substitute here -- the per-call feed dict
    supplies the right ``(K, 4096)`` block.
    """
    if not _QWEN_EMBEDDING_ONNX.exists():
        pytest.skip(
            f"Qwen embedding model not found at {_QWEN_EMBEDDING_ONNX}. "
            "This test depends on a local Qwen3.5-9B checkpoint."
        )

    m = onnx.load(str(_QWEN_EMBEDDING_ONNX), load_external_data=True)

    # DO NOT bind batch_size / sequence_length / num_logical_patches --
    # that is the entire contract of this test. Leave every dim_param
    # in place so the EP gets a graph with `tensor<?x?xi64>` for
    # input_ids, `tensor<?x4096xf16>` for image_features, and the
    # corresponding `?x?x4096` for the embedding output.
    return m


def _make_image_features(k: int) -> np.ndarray:
    """Synthesize ``image_features`` for ``K`` image-token positions.

    Shape is exactly ``(K, 4096)`` fp16 -- the per-call sizing the
    Reshape->Slice->ScatterND chain expects. The ``K=0`` case is a
    legal empty ONNX tensor ``(0, 4096)`` and exercises the EP's
    empty-input handling: ``hipdnn_ep_tensor_prepare_input`` detects
    the zero-element shape, allocates a 1-byte sentinel from the
    pool, and skips the H2D copy (the model still receives the
    correct shape descriptor and downstream
    ``Reshape->Slice->ScatterND`` no-ops). See CLAUDE.md
    "Cat-C slot publishers MUST publish a non-null buffer even when
    N=0" for the analogous documented pattern.

    Values are a deterministic arange * 0.01 payload so each non-empty
    row is distinguishable and the post-Scatter routing can be
    validated structurally.
    """
    if k == 0:
        return np.zeros((0, _IMAGE_FEATURES_COLS), dtype=np.float16)
    return np.arange(k * _IMAGE_FEATURES_COLS, dtype=np.float16).reshape(
        k, _IMAGE_FEATURES_COLS
    ) * np.float16(0.01)


# ---------------------------------------------------------------------------
# CPU reference (per-shape; the EP side uses one shared session)
# ---------------------------------------------------------------------------


def _build_input_ids(b: int, s: int, k: int) -> tuple[list[int], np.ndarray]:
    """Construct an ``[b, s] int64`` tensor with exactly ``k`` IMAGE_TOKEN_IDs.

    Position layout is deterministic across (b, s, k): the first batch
    row gets ``k`` evenly-spaced image tokens; every other (batch, pos)
    cell is filled with ``1`` (a known-valid embed table index). This
    keeps the bit-exact comparison structural -- if any non-image
    position diverges from the per-batch baseline the test fails with
    a precise (b, s) coordinate.

    Returns ``(positions, arr)`` so the caller can iterate the image
    positions when validating the ScatterND outcome.
    """
    arr = np.full((b, s), 1, dtype=np.int64)
    if k == 0:
        return [], arr
    if k > s - 1:
        raise ValueError(f"K={k} exceeds available positions S-1={s - 1}")
    # Evenly space K positions across [1, s); skip pos 0 so we always
    # have a "baseline row" at index 0 for the structural check below.
    step = max(1, (s - 1) // k)
    positions = [1 + i * step for i in range(k)]
    if max(positions) >= s:
        positions[-1] = s - 1
    for p in positions:
        arr[0, p] = _IMAGE_TOKEN_ID
    return positions, arr


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class TestQwen9bEmbeddingDyn:
    """Single InferenceSession serves every (B, S, K) in :data:`_SHAPES`.

    The test body is intentionally one method: building four sessions
    (one per shape) would defeat the "per-call shape switching" check.
    Per-shape outputs are compared bit-exactly against a fresh ORT CPU
    reference run on the same dynamic graph.
    """

    def test_per_call_shape_switching(
        self, request, model_runner, qwen_embedding_model_dyn
    ):
        # Save the dyn model once; both backends (EP shared session, CPU
        # per-shape) need a path on disk.
        sub = model_runner._next_subdir(request.node.name)
        model_path = sub / "model.onnx"
        model_path.write_bytes(qwen_embedding_model_dyn.SerializeToString())

        # ONE EP session, reused across every shape -- the contract
        # this test exists to validate. `make_session` does the
        # `assert_subgraph_on_ep` check inside, so a silent CPU
        # fallback fails right here.
        ep_backend = model_runner.backend  # framework/ort_ep_backend.OrtEpBackend
        if not hasattr(ep_backend, "make_session"):
            pytest.skip(
                "Active backend does not expose make_session(); the "
                "single-session-per-shape contract requires the "
                "ort_ep backend."
            )
        ep_sess = ep_backend.make_session(str(model_path))

        cpu_backend = OrtCpuBackend()

        try:
            # Pin the input names to what the model actually exports
            # (input_ids first, image_features second per the graph).
            ep_input_names = [i.name for i in ep_sess.get_inputs()]
            assert ep_input_names == ["input_ids", "image_features"], (
                f"unexpected input ordering from the EP session: "
                f"{ep_input_names}. Model contract drift?"
            )

            for b, s, k in _SHAPES:
                positions, input_ids = _build_input_ids(b, s, k)
                image_features = _make_image_features(k)

                # CPU reference: fresh CPU session per shape is OK
                # because the CPU side isn't what this test is gating on.
                # OrtCpuBackend.run() zips inputs to graph-input order,
                # which matches our [input_ids, image_features] tuple.
                expected = cpu_backend.run(str(model_path), [input_ids, image_features])

                # EP: same session, new shape. ORT routes this back into
                # the cached MorphiZen-compiled DLL whose runtime reads
                # the per-call sizes from input_shapes.
                actual = ep_sess.run(
                    None,
                    {"input_ids": input_ids, "image_features": image_features},
                )

                compare_outputs(actual, expected, atol=0.0, rtol=0.0)

                # Structural assertions, repeating the pattern of the
                # static tests but parametrised by k.
                assert actual[0].shape == (b, s, _HIDDEN), (
                    f"output shape mismatch for (B,S,K)=({b},{s},{k}): "
                    f"got {actual[0].shape}, expected ({b},{s},{_HIDDEN})"
                )
                baseline = actual[0][0, 0, :]
                for p in range(s):
                    if p in positions:
                        assert not np.array_equal(actual[0][0, p, :], baseline), (
                            f"(B,S,K)=({b},{s},{k}): image-token position "
                            f"{p} was not overwritten by ScatterND"
                        )
                    elif p == 0:
                        continue
                    else:
                        assert np.array_equal(actual[0][0, p, :], baseline), (
                            f"(B,S,K)=({b},{s},{k}): non-image position "
                            f"{p} differs from baseline row 0"
                        )
                # For B > 1 every non-first batch row must also be a
                # pure baseline (no image tokens injected outside row 0).
                for bi in range(1, b):
                    for p in range(s):
                        assert np.array_equal(actual[0][bi, p, :], baseline), (
                            f"(B,S,K)=({b},{s},{k}): batch {bi} row {p} "
                            "diverged from baseline (only row 0 should "
                            "see image-token rewrites)"
                        )
        finally:
            del ep_sess
