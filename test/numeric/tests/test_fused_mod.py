#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric checks for Mod as it is lowered inside a fused kernel.

`TestMod` in test_binary_elementwise.py builds a bare `Mod`, which leaves the
op on the runtime's own kernel (`wrap_mod`). Behind a MatMul it takes a
different route entirely: FuseROCMlir outlines the chain into a `rock.kernel`
function, hip-to-tosa rewrites the op into TOSA inside it, and rocMLIR
compiles the result. The two paths run different arithmetic for the same
graph, so the standalone test says nothing about the fused one.

The fused spelling is not a single op. TOSA has no modulo, so the lowering
builds one from the remainder identity: a truncating `tosa.intdiv`, then
`lhs - (lhs / rhs) * rhs`, then a correction adding one divisor back wherever
the remainder and the divisor have opposite signs. ONNX Mod with `fmod = 0`
wants the sign to follow the divisor while C's `%` makes it follow the
dividend, and that correction is the entire difference between them. A dropped
correction, an inverted predicate, or a zero remainder that wrongly gains a
divisor would all survive the LIT coverage, which only asserts the shape of
the emitted graph.

Reaching the fused path needs a cast. `hip.mod` accepts signless i32 and i64
only, and every fusion anchor (matmul, conv, gemm) is floating point, so the
graph here is MatMul -> Cast -> Mod. Cast is on FuseROCMlir's pointwise list
too, so all three are outlined into a single kernel.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type

# Dividend/divisor pairs chosen so every branch of the lowering is taken.
#
# `None` stands for the dtype's most negative value, filled in per dtype. That
# is the overflow case: the truncated quotient of INT_MIN / -1 is not
# representable, and `tosa.intdiv` becomes an LLVM `sdiv`, where the pair is
# undefined behaviour rather than a wrong number. The lowering avoids it by
# dividing by 1 instead, which is sound because every remainder by -1 is zero.
_MOD_CASES = [
    # (dividend, divisor, what it covers)
    (None, -1, "INT_MIN / -1, the overflow the lowering sidesteps"),
    (None, 3, "INT_MIN with a correction applied"),
    (-7, 3, "negative dividend, positive divisor -- correction applied"),
    (7, 3, "both positive -- no correction"),
    (-7, -3, "both negative -- no correction"),
    (7, -3, "positive dividend, negative divisor -- correction applied"),
    (12, 4, "exact zero remainder, both positive"),
    (-12, 4, "exact zero remainder, signs differ -- must stay zero"),
    (12, -4, "exact zero remainder, signs differ -- must stay zero"),
    (-12, -4, "exact zero remainder, both negative"),
    (0, 5, "zero dividend"),
    (0, -5, "zero dividend, negative divisor"),
]


def _mod_case_arrays(dtype) -> tuple[np.ndarray, np.ndarray]:
    """The dividend and divisor rows for `_MOD_CASES`, as (f32, dtype).

    The dividend is returned as float32 because it enters the graph through
    the MatMul anchor. Every value here is either small or exactly -2**31 /
    -2**63, all of which are exactly representable in f32, so the cast back
    to an integer recovers the dividend unchanged. That is what makes an
    `atol=0` comparison meaningful -- a dividend needing more than 24 bits of
    mantissa would be perturbed by the anchor rather than by Mod.
    """
    int_min = np.iinfo(dtype).min
    dividends = [int_min if a is None else a for a, _, _ in _MOD_CASES]
    divisors = [b for _, b, _ in _MOD_CASES]
    a = np.array([dividends], dtype=np.float32)
    b = np.array([divisors], dtype=dtype)
    return a, b


def _make_fused_mod_model(dtype, rows: int, width: int, divisor_rows: int):
    """MatMul against an identity weight, then Cast, then Mod.

    The MatMul is only there as a fusion anchor and the weight is the
    identity, so the value reaching Mod is the dividend unchanged and what is
    being compared is the modulo rather than the matmul's own rounding.

    `divisor_rows` is the divisor's leading dimension: pass `rows` for an
    elementwise divisor, or 1 to exercise the broadcast path, which is where
    the lowering has to equalize ranks before the intdiv.
    """
    onnx_int = np_to_onnx_type(dtype)
    onnx_f32 = np_to_onnx_type(np.float32)
    X = helper.make_tensor_value_info("X", onnx_f32, [rows, width])
    B = helper.make_tensor_value_info("B", onnx_int, [divisor_rows, width])
    Y = helper.make_tensor_value_info("Y", onnx_int, [rows, width])
    matmul = helper.make_node("MatMul", ["X", "W"], ["M"])
    cast = helper.make_node("Cast", ["M"], ["C"], to=onnx_int)
    mod = helper.make_node("Mod", ["C", "B"], ["Y"], fmod=0)
    weight = numpy_helper.from_array(np.eye(width, dtype=np.float32), name="W")
    return make_model_from_nodes(
        [matmul, cast, mod], [X, B], [Y], initializers=[weight]
    )


class TestFusedMod:
    """Mod behind a matmul anchor, which is the TOSA remainder expansion
    rather than the wrap_mod the standalone tests reach. The lowering is
    exact, so everything here is compared at zero tolerance."""

    @pytest.mark.parametrize("dtype", [np.int32, np.int64])
    def test_mod_fused_sign_and_zero_cases(self, model_runner, dtype):
        """Every sign combination, the exact-zero remainders, and INT_MIN."""
        a, b = _mod_case_arrays(dtype)
        model = _make_fused_mod_model(dtype, a.shape[0], a.shape[1], b.shape[0])
        actual, expected = model_runner.run_sample(model, [a, b], reference="cpu")
        compare_outputs(actual, expected, atol=0)

        # Pin the lanes the lowering treats specially, rather than leaving
        # them to a comparison that would also pass if both sides were wrong
        # together: a remainder by -1 is zero, and an exact division stays
        # zero instead of gaining a divisor from the sign correction.
        out = np.asarray(actual[0]).reshape(a.shape)
        int_min = np.iinfo(dtype).min
        assert out[0, 0] == 0, (
            f"{int_min} % -1 must be 0, got {out[0, 0]} -- the intdiv overflow "
            "guard is not doing its job"
        )
        for col in (6, 7, 8, 9):
            dividend, divisor, why = _MOD_CASES[col]
            assert out[0, col] == 0, (
                f"{dividend} % {divisor} must be 0 ({why}), got {out[0, col]}"
            )

    @pytest.mark.parametrize("dtype", [np.int32, np.int64])
    def test_mod_fused_sweep(self, model_runner, dtype):
        """A spread of magnitudes and signs, rather than the edges alone."""
        shape = [8, 64]
        rng = np.random.default_rng(933)
        a = rng.integers(-5000, 5000, shape).astype(np.float32)
        magnitude = rng.integers(1, 97, shape, dtype=dtype)
        sign = rng.integers(0, 2, shape, dtype=dtype) * 2 - 1
        b = (magnitude * sign).astype(dtype)
        model = _make_fused_mod_model(dtype, shape[0], shape[1], shape[0])
        actual, expected = model_runner.run_sample(model, [a, b], reference="cpu")
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize("dtype", [np.int32, np.int64])
    def test_mod_fused_broadcast_divisor(self, model_runner, dtype):
        """A row-broadcast divisor, which the lowering has to rank-equalize."""
        rows, width = 4, 16
        rng = np.random.default_rng(934)
        a = rng.integers(-500, 500, [rows, width]).astype(np.float32)
        magnitude = rng.integers(1, 13, [1, width], dtype=dtype)
        sign = rng.integers(0, 2, [1, width], dtype=dtype) * 2 - 1
        b = (magnitude * sign).astype(dtype)
        model = _make_fused_mod_model(dtype, rows, width, 1)
        actual, expected = model_runner.run_sample(model, [a, b], reference="cpu")
        compare_outputs(actual, expected, atol=0)
