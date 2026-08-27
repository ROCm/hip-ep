#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the Conv and ConvTranspose operations.

`TestConv` covers grouped / depthwise convolution. The weight descriptor must
use the per-group input-channel count (input_c / group), not input_c. For
group == 1 the two are equal, but for grouped (1 < group < C) and depthwise
(group == C) convolutions, using input_c describes the wrong filter shape and
produces silently incorrect results.

`TestConvShapeSweep` drives the shape sweep in conv_cases.py, which is where
the cases and the reasoning behind them live. They sit in their own module
because the same sweep also runs outside pytest, against hip-onnx-runner, on
machines whose ORT build cannot load the EP as a plugin.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from conv_cases import CONV_CASES, CONV_TRANSPOSE_CASES, build
from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


def _make_conv_model(
    in_c: int,
    out_c: int,
    group: int,
    k: int = 3,
    spatial: int = 8,
    seed: int = 7,
):
    """Build an fp16 2D Conv model with weight+bias initializers.

    input:   [1, in_c, spatial, spatial] f16
    weights: [out_c, in_c // group, k, k] f16  (initializer)
    bias:    [out_c] f16                        (initializer)
    output:  [1, out_c, spatial, spatial] f16   (stride 1, pad k//2 keeps size)
    """
    assert in_c % group == 0 and out_c % group == 0
    pad = k // 2

    inp = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT16, [1, in_c, spatial, spatial]
    )
    out = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT16, [1, out_c, spatial, spatial]
    )

    rng = np.random.default_rng(seed)
    w = rng.uniform(-0.1, 0.1, [out_c, in_c // group, k, k]).astype(np.float16)
    b = rng.uniform(-0.1, 0.1, [out_c]).astype(np.float16)
    w_init = numpy_helper.from_array(w, name="weight")
    b_init = numpy_helper.from_array(b, name="bias")

    node = helper.make_node(
        "Conv",
        ["input", "weight", "bias"],
        ["output"],
        group=group,
        kernel_shape=[k, k],
        strides=[1, 1],
        pads=[pad, pad, pad, pad],
        dilations=[1, 1],
    )
    return make_model_from_nodes([node], [inp], [out], initializers=[w_init, b_init])


class TestConv:
    @pytest.mark.parametrize(
        "in_c,out_c,group",
        [
            (4, 8, 1),  # standard conv (group=1: input_c/group == input_c)
            (8, 16, 2),  # grouped conv (intermediate group)
            (8, 16, 4),  # grouped conv (intermediate group)
            (8, 8, 8),  # depthwise conv (group == channels)
            (16, 16, 16),  # depthwise conv
        ],
    )
    def test_conv_grouped(self, model_runner, in_c, out_c, group):
        """Conv across group regimes; weights are [out_c, in_c/group, k, k].

        Regression guard: using input_c instead of input_c/group for the MIOpen
        weight descriptor yields uncorrelated output (cosine ~0) for group > 1.
        """
        model = _make_conv_model(in_c, out_c, group)

        rng = np.random.default_rng(123)
        x = rng.uniform(-1.0, 1.0, [1, in_c, 8, 8]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-2, rtol=1e-2, cos_threshold=0.999)


def _check(model_runner, case):
    model, x = build(case)
    actual, expected = model_runner.run_sample(model, [x])

    # Assert finiteness before correlating. A NaN makes the cosine NaN too, so
    # comparing first reports "cosine below threshold" for what is really an
    # arithmetic fault, and the count is the diagnostic that separates a kernel
    # writing garbage from one that is merely imprecise.
    for i, a in enumerate(actual):
        bad = int(np.count_nonzero(~np.isfinite(a)))
        if not bad:
            continue
        finite = a[np.isfinite(a)]
        lo = float(finite.min()) if finite.size else float("nan")
        hi = float(finite.max()) if finite.size else float("nan")
        raise AssertionError(
            f"output {i} has {bad} non-finite values of {a.size} "
            f"({100.0 * bad / a.size:.2f}%); finite range [{lo:.4g}, {hi:.4g}]"
        )

    tol = 2e-2 if case.get("dtype", "float16") == "float16" else 1e-4
    compare_outputs(actual, expected, atol=tol, rtol=tol, cos_threshold=0.999)


class TestConvShapeSweep:
    @pytest.mark.parametrize("case", CONV_CASES, ids=[c["id"] for c in CONV_CASES])
    def test_conv_shape(self, model_runner, case):
        _check(model_runner, case)

    @pytest.mark.parametrize(
        "case", CONV_TRANSPOSE_CASES, ids=[c["id"] for c in CONV_TRANSPOSE_CASES]
    )
    def test_conv_transpose_shape(self, model_runner, case):
        _check(model_runner, case)
