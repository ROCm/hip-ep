#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the Conv operation, focused on grouped / depthwise convolution.

The MIOpen weight descriptor must use the per-group input-channel count
(input_c / group), not input_c. For group == 1 the two are equal, but for
grouped (1 < group < C) and depthwise (group == C) convolutions, using input_c
describes the wrong filter shape and produces silently incorrect results. These
tests exercise all three regimes against the ORT CPU reference.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


def _make_conv_model(
    in_c: int,
    out_c: int,
    group: int,
    k: int = 3,
    spatial: int = 8,
    seed: int = 7,
    dynamic: bool = False,
):
    """Build an fp16 2D Conv model with weight+bias initializers.

    input:   [1, in_c, spatial, spatial] f16
    weights: [out_c, in_c // group, k, k] f16  (initializer)
    bias:    [out_c] f16                        (initializer)
    output:  [1, out_c, spatial, spatial] f16   (stride 1, pad k//2 keeps size)
    """
    assert in_c % group == 0 and out_c % group == 0
    pad = k // 2

    input_shape = [None, in_c, None, None] if dynamic else [1, in_c, spatial, spatial]
    output_shape = (
        [None, out_c, None, None] if dynamic else [1, out_c, spatial, spatial]
    )
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT16, input_shape)
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT16, output_shape)

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

    def test_conv_dynamic_spatial_valid(self, model_runner):
        """A valid dynamic Conv keeps the same GPU-vs-CPU numeric result."""
        model = _make_conv_model(4, 8, 1, dynamic=True)
        rng = np.random.default_rng(321)
        x = rng.uniform(-1.0, 1.0, [1, 4, 8, 8]).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-2, rtol=1e-2, cos_threshold=0.999)
