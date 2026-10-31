#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for BatchNormalization folded into the producing Conv weights.

hip-fold-batchnorm rewrites an inference BatchNormalization with constant
parameters into the filter and bias of the Conv(s) feeding it, directly or
through an Add of two Convs. A wrong per-channel scale, a shift added twice,
or a shift dropped all give outputs that stay correlated with the reference,
so these compare tightly against the ORT CPU reference.

The fold reads only inline constants, and every filter and BN parameter vector
here (64 channels, 256 B in f32) is above the importer's default 127-byte
inline threshold. Run with
`--ep-option XLNX_model_clone_external_data_threshold=1073741824` (or the
`XLNX_ABI_2_0_CLONE_EXTERNAL_DATA_THRESHOLD` environment variable) to exercise
the fold; without it the same graphs check the unfolded path.
"""

import numpy as np
import pytest
from onnx import helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type

CHANNELS = 64
SPATIAL = 8


def _bn_initializers(rng, dtype, prefix="bn"):
    """BatchNorm scale, B, mean, var with a spread wide enough to matter."""
    return [
        numpy_helper.from_array(
            rng.uniform(0.5, 2.0, [CHANNELS]).astype(dtype), f"{prefix}_scale"
        ),
        numpy_helper.from_array(
            rng.uniform(-1.0, 1.0, [CHANNELS]).astype(dtype), f"{prefix}_b"
        ),
        numpy_helper.from_array(
            rng.uniform(-0.5, 0.5, [CHANNELS]).astype(dtype), f"{prefix}_mean"
        ),
        numpy_helper.from_array(
            rng.uniform(0.1, 2.0, [CHANNELS]).astype(dtype), f"{prefix}_var"
        ),
    ]


def _bn_node(x, y, prefix="bn"):
    return helper.make_node(
        "BatchNormalization",
        [x, f"{prefix}_scale", f"{prefix}_b", f"{prefix}_mean", f"{prefix}_var"],
        [y],
        epsilon=1e-5,
    )


def _conv_node(x, w, y, bias=None, group=1):
    return helper.make_node(
        "Conv",
        [x, w] + ([bias] if bias else []),
        [y],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        dilations=[1, 1],
        group=group,
    )


def _weight(rng, dtype, name, in_c, group=1):
    w = rng.uniform(-0.2, 0.2, [CHANNELS, in_c // group, 3, 3]).astype(dtype)
    return numpy_helper.from_array(w, name)


_TOLERANCE = {
    np.float32: dict(atol=1e-3, rtol=1e-3, cos_threshold=0.99999),
    np.float16: dict(atol=3e-2, rtol=2e-2, cos_threshold=0.999),
}


class TestConvBatchNorm:
    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    @pytest.mark.parametrize("with_bias", [False, True])
    @pytest.mark.parametrize("group", [1, 4])
    def test_conv_bn(self, model_runner, dtype, with_bias, group):
        """Conv -> BatchNormalization folds into one conv with a bias."""
        in_c = 16
        tp = np_to_onnx_type(dtype)
        rng = np.random.default_rng(11)
        inits = [_weight(rng, dtype, "w", in_c, group)] + _bn_initializers(rng, dtype)
        if with_bias:
            inits.append(
                numpy_helper.from_array(
                    rng.uniform(-0.5, 0.5, [CHANNELS]).astype(dtype), "cb"
                )
            )
        nodes = [
            _conv_node("x", "w", "c", bias="cb" if with_bias else None, group=group),
            _bn_node("c", "y"),
        ]
        model = make_model_from_nodes(
            nodes,
            [helper.make_tensor_value_info("x", tp, [1, in_c, SPATIAL, SPATIAL])],
            [helper.make_tensor_value_info("y", tp, [1, CHANNELS, SPATIAL, SPATIAL])],
            initializers=inits,
        )
        x = rng.uniform(-1.0, 1.0, [1, in_c, SPATIAL, SPATIAL]).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, **_TOLERANCE[dtype])

    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    def test_add_conv_conv_bn_relu(self, model_runner, dtype):
        """Relu(BN(Conv(x) + Conv(z))): both filters scaled, shift added once."""
        in_x, in_z = 16, 48
        tp = np_to_onnx_type(dtype)
        rng = np.random.default_rng(23)
        inits = [
            _weight(rng, dtype, "wa", in_x),
            _weight(rng, dtype, "wb", in_z),
        ] + _bn_initializers(rng, dtype)
        nodes = [
            _conv_node("x", "wa", "ca"),
            _conv_node("z", "wb", "cb"),
            helper.make_node("Add", ["ca", "cb"], ["s"]),
            _bn_node("s", "n"),
            helper.make_node("Relu", ["n"], ["y"]),
        ]
        model = make_model_from_nodes(
            nodes,
            [
                helper.make_tensor_value_info("x", tp, [1, in_x, SPATIAL, SPATIAL]),
                helper.make_tensor_value_info("z", tp, [1, in_z, SPATIAL, SPATIAL]),
            ],
            [helper.make_tensor_value_info("y", tp, [1, CHANNELS, SPATIAL, SPATIAL])],
            initializers=inits,
        )
        x = rng.uniform(-1.0, 1.0, [1, in_x, SPATIAL, SPATIAL]).astype(dtype)
        z = rng.uniform(-1.0, 1.0, [1, in_z, SPATIAL, SPATIAL]).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x, z])
        compare_outputs(actual, expected, **_TOLERANCE[dtype])
