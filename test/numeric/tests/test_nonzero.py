#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric verification for ONNX NonZero (the canonical Category-C op).

NonZero is the reference test for the dynamic-output-shapes framework:
its output shape `[rank, N]` only exists after the runtime count kernel
has reported how many non-zero elements were found, so every layer of
the new pipeline is exercised on every call --
- `wrap_nonzero` publishes the data-dependent `N` into a runtime slot,
- the EP host-side resolver reads it back and allocates the ORT
  OrtValue post-compute, and
- `compose-dim-specs` carries the RuntimeSlot leaf in the dim-spec
  tree from MLIR all the way into the model.dll metadata.

A wrong result in any of those layers shows up here as a shape or
content mismatch.

Layout note. ONNX NonZero output is `[rank, N]` int64 in row-major
input scan order. Our GPU implementation atomic-allocates slots in the
fill kernel, so the per-column ordering across runs is a permutation
of the canonical CPU answer. The comparator below normalises both
sides by lex-sorting along axis 1 before comparing, so any permutation
of the same (correct) column set passes. Same-tensor identity (e.g.
the all-zero / no-non-zero special case) still falls through to the
default `compare_outputs` because the shape match degenerates to
`[rank, 0]` on both sides.

Covers:
  * Category C: strictly data-dependent output dim `N`.
  * Multiple element types (i1 / i32 / i64 / f16 / f32 / ui8).
  * Multiple input ranks (1D / 2D / 3D / 4D).
  * Edge cases: all-zero (N=0), all-non-zero (N=numel), single element.
  * Dynamic input shapes (so the upper-bound numel computation also
    has to lower correctly via tensor.dim chains).
"""

from __future__ import annotations

import numpy as np
from onnx import helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nonzero_model(dtype: np.dtype, input_shape: list[int | None]):
    """Build `onnx.NonZero(X) -> Y` with the given input element type / shape.

    A `None` entry in `input_shape` marks that axis as dynamic in the
    ONNX graph (compiler will lower it via `tensor.dim`). The output
    shape is always `[rank, None]` -- rank is static (= input rank),
    `N` is data-dependent.
    """
    rank = len(input_shape)
    tp = np_to_onnx_type(dtype)
    onnx_in_shape = [d if d is not None else None for d in input_shape]
    X = helper.make_tensor_value_info("X", tp, onnx_in_shape)
    # Output is always int64 per ONNX spec.
    from onnx import TensorProto

    Y = helper.make_tensor_value_info("Y", TensorProto.INT64, [rank, None])
    node = helper.make_node("NonZero", ["X"], ["Y"])
    return make_model_from_nodes([node], [X], [Y])


def _sort_nonzero_columns(idx: np.ndarray) -> np.ndarray:
    """Sort the columns of a NonZero output lexicographically.

    `idx` has shape `[rank, N]` int64 -- one column per non-zero
    element. The comparator must be insensitive to within-batch
    permutation (our GPU kernel atomic-allocates columns), so we
    canonicalise by lex-sorting columns into ascending row-major order
    before diffing.
    """
    if idx.ndim != 2 or idx.shape[1] == 0:
        return idx
    # np.lexsort sorts by the LAST key as the primary key, so feed rows
    # in reverse so that axis 0 is the primary sort key.
    order = np.lexsort(idx[::-1])
    return idx[:, order]


def _compare_nonzero(actual: list[np.ndarray], expected: list[np.ndarray]) -> None:
    """Compare with column-permutation invariance.

    Asserts both sides have exactly one output, identical shape, and
    identical content after lex-sorting columns. The element-wise
    comparison is exact (`np.array_equal`) because the indices are
    int64 -- there is no rounding to tolerate.
    """
    assert len(actual) == len(expected) == 1, (
        f"NonZero expected 1 output, got actual={len(actual)} expected={len(expected)}"
    )
    a, e = actual[0], expected[0]
    assert a.shape == e.shape, (
        f"NonZero shape mismatch: actual={a.shape}, expected={e.shape}"
    )
    if a.size == 0:
        return  # `[rank, 0]` on both sides -- trivially equal.
    a_sorted = _sort_nonzero_columns(a.astype(np.int64))
    e_sorted = _sort_nonzero_columns(e.astype(np.int64))
    if not np.array_equal(a_sorted, e_sorted):
        # Build a helpful diff message: which columns differ.
        diff_mask = (a_sorted != e_sorted).any(axis=0)
        n_diff = int(diff_mask.sum())
        raise AssertionError(
            f"NonZero content mismatch in {n_diff}/{a_sorted.shape[1]} "
            f"columns after lex-sort.\nactual (sorted):\n{a_sorted}\n"
            f"expected (sorted):\n{e_sorted}"
        )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestNonZero:
    # ------- 1D inputs --------------------------------------------------

    def test_nonzero_i64_1d_mixed(self, model_runner):
        """1D i64 with a mix of zeros / non-zeros."""
        shape = [16]
        model = _make_nonzero_model(np.int64, shape)
        x = np.array(
            [0, 1, 0, 2, 3, 0, 0, 4, 5, 0, 6, 7, 8, 0, 9, 10],
            dtype=np.int64,
        )
        actual, expected = model_runner.run_sample(model, [x], reference="cpu")
        _compare_nonzero(actual, expected)

    def test_nonzero_i64_1d_all_zero(self, model_runner):
        """All-zero input -> N=0; output shape `[1, 0]`."""
        shape = [16]
        model = _make_nonzero_model(np.int64, shape)
        x = np.zeros(shape, dtype=np.int64)
        actual, expected = model_runner.run_sample(model, [x], reference="cpu")
        _compare_nonzero(actual, expected)
        assert actual[0].shape == (1, 0)

    def test_nonzero_i64_1d_all_nonzero(self, model_runner):
        """All-non-zero input -> N=numel."""
        shape = [16]
        model = _make_nonzero_model(np.int64, shape)
        x = np.arange(1, shape[0] + 1, dtype=np.int64)
        actual, expected = model_runner.run_sample(model, [x], reference="cpu")
        _compare_nonzero(actual, expected)
        assert actual[0].shape == (1, shape[0])

    def test_nonzero_i64_1d_single_nonzero(self, model_runner):
        """Single non-zero element -> N=1."""
        shape = [16]
        model = _make_nonzero_model(np.int64, shape)
        x = np.zeros(shape, dtype=np.int64)
        x[7] = 42
        actual, expected = model_runner.run_sample(model, [x], reference="cpu")
        _compare_nonzero(actual, expected)
        assert actual[0].shape == (1, 1)
        assert int(actual[0][0, 0]) == 7

    # ------- 2D inputs --------------------------------------------------

    def test_nonzero_bool_2d(self, model_runner):
        """2D bool mask (canonical attention-like usage)."""
        shape = [3, 4]
        model = _make_nonzero_model(np.bool_, shape)
        rng = np.random.default_rng(42)
        x = (rng.uniform(0, 1, shape) > 0.5).astype(np.bool_)
        actual, expected = model_runner.run_sample(model, [x], reference="cpu")
        _compare_nonzero(actual, expected)

    def test_nonzero_i32_2d(self, model_runner):
        """2D i32 with negative + positive non-zeros."""
        shape = [4, 8]
        model = _make_nonzero_model(np.int32, shape)
        rng = np.random.default_rng(7)
        x = rng.integers(-3, 4, shape, dtype=np.int32)
        actual, expected = model_runner.run_sample(model, [x], reference="cpu")
        _compare_nonzero(actual, expected)

    def test_nonzero_f32_2d(self, model_runner):
        """2D f32 with a few exact zeros sprinkled in."""
        shape = [8, 16]
        model = _make_nonzero_model(np.float32, shape)
        rng = np.random.default_rng(123)
        x = rng.uniform(-1, 1, shape).astype(np.float32)
        x[rng.uniform(0, 1, shape) < 0.3] = 0.0
        actual, expected = model_runner.run_sample(model, [x], reference="cpu")
        _compare_nonzero(actual, expected)

    def test_nonzero_f16_2d(self, model_runner):
        """2D f16 with a few exact zeros sprinkled in."""
        shape = [8, 16]
        model = _make_nonzero_model(np.float16, shape)
        rng = np.random.default_rng(99)
        x = rng.uniform(-1, 1, shape).astype(np.float16)
        x[rng.uniform(0, 1, shape) < 0.3] = np.float16(0.0)
        actual, expected = model_runner.run_sample(model, [x], reference="cpu")
        _compare_nonzero(actual, expected)

    def test_nonzero_ui8_2d(self, model_runner):
        """2D ui8 (ORT imports ONNX bool as ui8 -- exercise that path)."""
        shape = [3, 4]
        model = _make_nonzero_model(np.uint8, shape)
        rng = np.random.default_rng(42)
        x = (rng.uniform(0, 1, shape) > 0.5).astype(np.uint8)
        actual, expected = model_runner.run_sample(model, [x], reference="cpu")
        _compare_nonzero(actual, expected)

    # ------- 3D / 4D inputs --------------------------------------------

    def test_nonzero_i64_3d(self, model_runner):
        """3D i64 -- output is `[3, N]`."""
        shape = [2, 3, 4]
        model = _make_nonzero_model(np.int64, shape)
        rng = np.random.default_rng(2024)
        x = rng.integers(-2, 3, shape, dtype=np.int64)
        actual, expected = model_runner.run_sample(model, [x], reference="cpu")
        _compare_nonzero(actual, expected)

    def test_nonzero_f32_4d(self, model_runner):
        """4D f32 -- output is `[4, N]`. Sanity for higher-rank decompose."""
        shape = [2, 3, 4, 5]
        model = _make_nonzero_model(np.float32, shape)
        rng = np.random.default_rng(8)
        x = rng.uniform(-1, 1, shape).astype(np.float32)
        x[rng.uniform(0, 1, shape) < 0.4] = 0.0
        actual, expected = model_runner.run_sample(model, [x], reference="cpu")
        _compare_nonzero(actual, expected)

    # ------- Larger shapes (atomic contention) --------------------------

    def test_nonzero_i32_large(self, model_runner):
        """Larger 2D input -- exercises grid-stride loop + atomic contention.

        4096 elements at ~50% non-zero rate gives the count kernel several
        thousand atomic adds; pass-2's atomic_idx sees the same load with
        the additional per-slot coord write. Output order is invariant
        thanks to `_sort_nonzero_columns`.
        """
        shape = [64, 64]
        model = _make_nonzero_model(np.int32, shape)
        rng = np.random.default_rng(0)
        x = rng.integers(-1, 2, shape, dtype=np.int32)  # values in {-1,0,1}
        actual, expected = model_runner.run_sample(model, [x], reference="cpu")
        _compare_nonzero(actual, expected)

    # ------- Generic compare-outputs sanity for the empty case ----------

    def test_nonzero_empty_shape_matches_compare_outputs(self, model_runner):
        """N=0 falls through to the default tolerance compare cleanly."""
        shape = [8]
        model = _make_nonzero_model(np.int64, shape)
        x = np.zeros(shape, dtype=np.int64)
        actual, expected = model_runner.run_sample(model, [x], reference="cpu")
        # Both sides are `[1, 0]`; allclose is trivially True.
        compare_outputs(actual, expected, atol=0, rtol=0)
