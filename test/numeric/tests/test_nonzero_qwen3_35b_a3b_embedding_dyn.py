#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Fully-dynamic-input variant of ``test_nonzero_qwen3_35b_a3b_embedding.py``.

Companion to ``test_nonzero_qwen_embedding_dyn.py`` (which covers the
9B graph) for the 35B-A3B graph. Differences from the 9B dyn test:

  * ``image_features`` is a real graph input here, not an initializer.
    Its first dim, ``num_logical_patches``, is left symbolic at the
    ModelProto level alongside ``batch_size`` / ``sequence_length`` --
    so the single InferenceSession must accept three independent
    symbolic dim_params at the input boundary.
  * The graph contains a ``Where`` + ``ConstantOfShape`` branch that
    masks out IMAGE_TOKEN_ID positions before Gather (the 9B graph
    doesn't have this), so the Equal output ALSO drives the Where
    inside the dyn path.
  * ``hidden == image_features_cols == 2048`` -- 1:1 row mapping
    between flattened ``image_features`` and ScatterND updates (the
    9B graph needed 2:1).

K=0 is omitted from the parametrization here. The static sister test
documents (and gates with ``xfail``) that the EP MLIR pipeline cannot
bufferize a graph with a literal zero-extent static input; in a
dyn-input session ``num_logical_patches`` is left symbolic so the
graph itself is OK to compile, but the per-call K=0 input STILL trips
that pipeline path inside the runtime view machinery. Re-enabling K=0
here would not exercise anything the 9B dyn test does not already
cover for the Cat-C empty-output edge case. The interesting overlap
this file adds is the K>0 ``Where`` + ``ConstantOfShape`` path.

Shapes (B, S, K):

  ``(1, 128, 1)`` -- single image token (smallest K>0 path).
  ``(1, 128, 4)`` -- same (B, S) as #1, K flips to 4.
  ``(2, 64, 4)``  -- batch dim switches, sequence shrinks.
  ``(1, 256, 1)`` -- sequence grows, K shrinks back to 1.

Single InferenceSession is reused across all four, so any "stale
cached shape" bug surfaces as a per-call ScatterND / Where / Equal
mismatch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest

from framework.comparator import compare_outputs
from framework.ort_cpu_backend import OrtCpuBackend


# ---------------------------------------------------------------------------
# Model path / fixture
# ---------------------------------------------------------------------------

_QWEN_35B_A3B_EMBEDDING_ONNX = Path(
    "D:/Develop/m/models/Qwen3.5-35B-A3B_int4_rtn_128gs_cuda/embedding.onnx"
)

_IMAGE_TOKEN_ID = 248056
_HIDDEN = 2048
_IMAGE_FEATURES_COLS = 2048

_SHAPES: list[tuple[int, int, int]] = [
    (1, 128, 1),
    (1, 128, 4),
    (2, 64, 4),
    (1, 256, 1),
]


@pytest.fixture(scope="module")
def qwen35b_a3b_embedding_model_dyn():
    """Load the 35B-A3B embedding ONNX with EVERY dim_param symbolic.

    ``batch_size``, ``sequence_length``, and ``num_logical_patches`` are
    all left unbound -- the entire promise of this test is that the
    EP services every parametrized ``(B, S, K)`` via a single compiled
    DLL with no per-K recompile.
    """
    if not _QWEN_35B_A3B_EMBEDDING_ONNX.exists():
        pytest.skip(
            f"Qwen3.5-35B-A3B embedding model not found at "
            f"{_QWEN_35B_A3B_EMBEDDING_ONNX}. Test depends on a local checkpoint."
        )
    return onnx.load(str(_QWEN_35B_A3B_EMBEDDING_ONNX), load_external_data=True)


def _make_image_features(k: int) -> np.ndarray:
    """Synthetic ``[k, 2048]`` fp16 features with deterministic row values.

    Same convention as the static 35B-A3B test: row ``r`` is filled
    with ``(r + 1) * 0.1`` so a wrong ScatterND row routing is visible
    in the structural assertions.
    """
    if k == 0:
        return np.zeros((0, _IMAGE_FEATURES_COLS), dtype=np.float16)
    rows = []
    for r in range(k):
        rows.append(
            np.full(_IMAGE_FEATURES_COLS, fill_value=(r + 1) * 0.1, dtype=np.float16)
        )
    return np.stack(rows, axis=0)


def _build_input_ids(b: int, s: int, k: int) -> tuple[list[int], np.ndarray]:
    """Construct an ``[b, s] int64`` tensor with exactly ``k`` IMAGE_TOKEN_IDs.

    Same per-batch / per-position layout as the 9B dyn test so the
    structural validation pattern carries over. Image tokens are only
    injected in batch 0, evenly spaced across ``[1, s)``.
    """
    arr = np.full((b, s), 1, dtype=np.int64)
    if k == 0:
        return [], arr
    if k > s - 1:
        raise ValueError(f"K={k} exceeds available positions S-1={s - 1}")
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


class TestQwen35bA3bEmbeddingDyn:
    """Single InferenceSession serves every (B, S, K) in :data:`_SHAPES`.

    The CPU reference is recomputed per shape (fresh CPU session) but
    the EP side reuses ONE session across all calls -- the actual
    contract under test. ``assert_subgraph_on_ep`` runs inside
    ``make_session`` so a silent CPU fallback aborts the test before
    any comparison.
    """

    def test_per_call_shape_switching(
        self, request, model_runner, qwen35b_a3b_embedding_model_dyn
    ):
        sub = model_runner._next_subdir(request.node.name)
        model_path = sub / "model.onnx"
        model_path.write_bytes(qwen35b_a3b_embedding_model_dyn.SerializeToString())

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
            input_names = [i.name for i in ep_sess.get_inputs()]
            # Sanity: the dyn graph carries two real inputs --
            # ``input_ids`` and ``image_features``. If the model file
            # ever drifts to a different signature the test should
            # fail with a clear error rather than silently feeding
            # wrong arguments.
            assert len(input_names) == 2, (
                f"35B-A3B dyn graph expected 2 inputs, found "
                f"{len(input_names)}: {input_names}"
            )

            for b, s, k in _SHAPES:
                positions, input_ids = _build_input_ids(b, s, k)
                image_features = _make_image_features(k)
                assert image_features.shape == (k, _IMAGE_FEATURES_COLS)

                inputs = {
                    input_names[0]: input_ids,
                    input_names[1]: image_features,
                }
                expected = cpu_backend.run(str(model_path), [input_ids, image_features])
                actual = ep_sess.run(None, inputs)

                compare_outputs(actual, expected, atol=0.0, rtol=0.0)

                assert actual[0].shape == (b, s, _HIDDEN), (
                    f"output shape mismatch for (B,S,K)=({b},{s},{k}): "
                    f"got {actual[0].shape}, expected ({b},{s},{_HIDDEN})"
                )

                # Structural sanity on the scatter routing -- the
                # 35B-A3B graph adds the Where+ConstantOfShape branch,
                # which would corrupt the baseline if Equal misfires
                # under a dynamic shape. Validate it explicitly.
                baseline = actual[0][0, 0, :]
                for i, p in enumerate(positions):
                    assert not np.array_equal(actual[0][0, p, :], baseline), (
                        f"(B,S,K)=({b},{s},{k}): image-token position "
                        f"{p} was not overwritten by ScatterND"
                    )
                    # 1:1 hidden==image_features_cols mapping: position
                    # ``positions[i]`` receives image_features[i].
                    assert np.array_equal(actual[0][0, p, :], image_features[i]), (
                        f"(B,S,K)=({b},{s},{k}): position {p} (i={i}) "
                        "received the wrong image_features row -- "
                        "ScatterND or NonZero scan order regressed"
                    )
                for p in range(s):
                    if p in positions:
                        continue
                    assert np.array_equal(actual[0][0, p, :], baseline), (
                        f"(B,S,K)=({b},{s},{k}): non-image position {p} "
                        "differs from baseline (Where mask or Equal "
                        "misbehaved)"
                    )
                for bi in range(1, b):
                    for p in range(s):
                        assert np.array_equal(actual[0][bi, p, :], baseline), (
                            f"(B,S,K)=({b},{s},{k}): batch {bi} row {p} "
                            "diverged from baseline"
                        )
        finally:
            del ep_sess
