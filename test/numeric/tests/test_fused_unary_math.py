#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric checks for unary math ops as they are lowered inside a fused kernel.

The tests in test_unary_math.py build a bare `Atan` or `Round` graph, which
leaves the op on the runtime's own kernel (lib/Runtime/real/{atan,round}.cpp).
An op that follows a MatMul takes a different route entirely: FuseROCMlir
outlines the pair into a `rock.kernel` function, hip-to-tosa rewrites the op
into TOSA inside it, and rocMLIR compiles the result. So the two paths run
different arithmetic for the same graph, and the standalone tests say nothing
about the fused one.

Atan is the case that matters most. TOSA has no arctangent, so the conversion
does not map the op onto anything -- it expands it into a Cephes minimax
polynomial behind a two-threshold domain fold. A wrong coefficient, a
threshold on the wrong side of its comparison, or a dropped sign would all
survive the LIT coverage, which only asserts the shape of the emitted graph.
Round is the cheaper companion: its lowering is exact, so it is checked at
zero tolerance, with the sign of the half-tie zeros asserted separately --
np.allclose cannot see the difference between -0.0 and 0.0.

Each test puts a MatMul in front of the op purely as a fusion anchor. The
weight is the identity, so the product is the crafted input exactly and what
is being compared is the unary op rather than the matmul's own rounding.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from onnx import helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type

# The two thresholds the atan expansion folds the domain on: above tan(3pi/8)
# it reflects through pi/2, above tan(pi/8) it shifts by pi/4, and below that
# it evaluates the polynomial directly.
TAN_PI_8 = math.tan(math.pi / 8)
TAN_3PI_8 = math.tan(3 * math.pi / 8)


def _make_fused_unary_model(op_type: str, dtype, rows: int, width: int):
    """MatMul against an identity weight, then `op_type`.

    The MatMul is the fusion anchor -- FuseROCMlir outlines an anchor plus the
    pointwise op behind it, and both Atan and Round are on its pointwise list.
    The identity weight keeps the product exact, so the value reaching the
    unary op is the input value unchanged.
    """
    tp = np_to_onnx_type(dtype)
    X = helper.make_tensor_value_info("X", tp, [rows, width])
    Y = helper.make_tensor_value_info("Y", tp, [rows, width])
    matmul = helper.make_node("MatMul", ["X", "W"], ["M"])
    unary = helper.make_node(op_type, ["M"], ["Y"])
    weight = numpy_helper.from_array(np.eye(width, dtype=dtype), name="W")
    return make_model_from_nodes([matmul, unary], [X], [Y], initializers=[weight])


def _atan_probe_row(dtype) -> np.ndarray:
    """Values that sit on, and just either side of, each fold boundary.

    A threshold that drifts across its comparison sends the bracketing pair
    down different branches, and the two branches agree only if the fold
    identity was applied correctly, so the pair catches it.
    """
    # Wide enough to straddle the boundary in fp16, whose spacing near 2.4 is
    # about 2e-3, while staying tight in fp32.
    step = 8e-3 if dtype == np.float16 else 1e-5
    big = 1e4 if dtype == np.float16 else 1e30
    small = 1e-3 if dtype == np.float16 else 1e-20
    values = [
        0.0,
        -0.0,
        small,
        -small,
        TAN_PI_8 - step,
        TAN_PI_8,
        TAN_PI_8 + step,
        -TAN_PI_8,
        1.0,
        -1.0,
        TAN_3PI_8 - step,
        TAN_3PI_8,
        TAN_3PI_8 + step,
        -TAN_3PI_8,
        big,
        -big,
    ]
    return np.array(values, dtype=dtype).reshape(1, len(values))


def _assert_round_half_tie_signs(x: np.ndarray, out: np.ndarray) -> None:
    """Check the sign on the zeros that the half-to-even ties produce.

    compare_outputs ends in np.allclose, which holds -0.0 equal to 0.0, so a
    lowering that dropped the sign on the negative tie would still pass it.
    ONNX Round sends 0.5 to +0.0 and -0.5 to -0.0, and those are the only two
    lanes here that reach a zero from a nonzero input. The +-0.0 inputs cannot
    carry this check: the identity matmul in front of the op adds -0.0 to +0.0
    and yields +0.0 before Round ever sees it.
    """
    for value, want_negative in ((0.5, False), (-0.5, True)):
        lanes = out[x == value]
        assert lanes.size, f"no lane holds the input {value}"
        assert np.all(lanes == 0), f"round({value}) must be a zero, got {lanes}"
        sign = "-" if want_negative else "+"
        assert np.all(np.signbit(lanes) == want_negative), (
            f"round({value}) must be {sign}0.0, got {lanes} "
            f"with sign bits {np.signbit(lanes)}"
        )


class TestFusedAtan:
    """Atan behind a matmul anchor, which is the TOSA expansion rather than
    the device atanf the standalone tests reach."""

    @pytest.mark.parametrize("dtype", [np.float16, np.float32])
    def test_atan_fused_fold_boundaries(self, model_runner, dtype):
        x = _atan_probe_row(dtype)
        model = _make_fused_unary_model("Atan", dtype, x.shape[0], x.shape[1])
        actual, expected = model_runner.run_sample(model, [x], reference="cpu")
        atol = 1e-3 if dtype == np.float16 else 1e-5
        compare_outputs(actual, expected, atol=atol, rtol=1e-3)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32])
    def test_atan_fused_sweep(self, model_runner, dtype):
        """A spread across all three branches at once, rather than at their
        edges, so an error in the interior of a branch shows up too."""
        shape = [8, 64]
        rng = np.random.default_rng(931)
        x = rng.uniform(-20.0, 20.0, shape).astype(dtype)
        model = _make_fused_unary_model("Atan", dtype, shape[0], shape[1])
        actual, expected = model_runner.run_sample(model, [x], reference="cpu")
        atol = 1e-3 if dtype == np.float16 else 1e-5
        compare_outputs(actual, expected, atol=atol, rtol=1e-3)


class TestFusedRound:
    """Round behind the same anchor. Its lowering is exact -- a floor with the
    halfway cases pushed to even -- so it is compared at zero tolerance."""

    @pytest.mark.parametrize("dtype", [np.float16, np.float32])
    def test_round_fused_ties(self, model_runner, dtype):
        x = np.array(
            [[0.0, -0.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 0.9, -0.9, 2.3, -2.3]],
            dtype=dtype,
        )
        model = _make_fused_unary_model("Round", dtype, x.shape[0], x.shape[1])
        actual, expected = model_runner.run_sample(model, [x], reference="cpu")
        compare_outputs(actual, expected, atol=0)
        _assert_round_half_tie_signs(x, actual[0].reshape(x.shape))

    @pytest.mark.parametrize("dtype", [np.float16, np.float32])
    def test_round_fused_sweep(self, model_runner, dtype):
        shape = [8, 64]
        rng = np.random.default_rng(932)
        x = rng.uniform(-8.0, 8.0, shape).astype(dtype)
        model = _make_fused_unary_model("Round", dtype, shape[0], shape[1])
        actual, expected = model_runner.run_sample(model, [x], reference="cpu")
        compare_outputs(actual, expected, atol=0)
