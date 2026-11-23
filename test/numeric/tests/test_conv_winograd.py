#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric checks for fp16 3x3 convs run as fused Winograd F(2x2, 3x3) kernels.

With HIPDNN_EP_ROCMLIR=1 and HIPDNN_EP_WINOGRAD=1 on gfx1151 or gfx12, an fp16
3x3 stride-1 pad-1 conv that the shape gate admits, with the LeakyRelu or ReLU
that alone consumes it, runs as one hip.winograd_conv kernel built for its
shape. These cases cover each epilogue (bias + LeakyRelu, ReLU without a bias,
bias alone), odd spatial sizes (partial 2x2 output tiles), a batch of two, an
input channel count that is not a multiple of 16, and small images whose
configuration splits the channel sum across waves. The reference is the ORT
CPU EP. Winograd accumulates in fp32 through WMMA but transforms the input in
fp16, so the tolerance is that of an fp16 conv.
"""

import os

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

pytestmark = pytest.mark.skipif(
    os.environ.get("HIPDNN_EP_ROCMLIR") != "1"
    or os.environ.get("HIPDNN_EP_WINOGRAD") != "1",
    reason="Winograd convs run only with HIPDNN_EP_ROCMLIR=1 and HIPDNN_EP_WINOGRAD=1",
)


def _make_model(n, in_c, out_c, h, w, bias, activation, seed=7):
    """fp16 Conv 3x3, stride 1, pad 1, then the optional activation."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [n, in_c, h, w])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [n, out_c, h, w])
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(9.0 * in_c)
    inits = [
        numpy_helper.from_array(
            rng.uniform(-scale, scale, [out_c, in_c, 3, 3]).astype(np.float16), "w"
        )
    ]
    conv_inputs = ["x", "w"]
    if bias:
        inits.append(
            numpy_helper.from_array(
                rng.uniform(-0.5, 0.5, [out_c]).astype(np.float16), "b"
            )
        )
        conv_inputs.append("b")
    conv_out = "y" if activation == "none" else "c"
    nodes = [
        helper.make_node(
            "Conv",
            conv_inputs,
            [conv_out],
            kernel_shape=[3, 3],
            strides=[1, 1],
            pads=[1, 1, 1, 1],
            dilations=[1, 1],
            group=1,
        )
    ]
    if activation == "leaky_relu":
        nodes.append(helper.make_node("LeakyRelu", ["c"], ["y"], alpha=0.2))
    elif activation == "relu":
        nodes.append(helper.make_node("Relu", ["c"], ["y"]))
    return make_model_from_nodes(nodes, [x], [y], initializers=inits)


class TestConvWinograd:
    @pytest.mark.parametrize(
        "n,in_c,out_c,h,w,bias,activation",
        [
            (1, 64, 64, 72, 84, True, "leaky_relu"),
            (1, 32, 48, 33, 45, False, "relu"),
            (1, 48, 48, 64, 64, True, "none"),
            (2, 32, 32, 40, 24, True, "leaky_relu"),
            (1, 24, 40, 17, 19, True, "leaky_relu"),
            (1, 112, 112, 96, 112, True, "leaky_relu"),
            # Few output tiles: the channel sum splits across two waves.
            (1, 456, 456, 12, 21, True, "leaky_relu"),
            (1, 768, 128, 36, 36, True, "none"),
        ],
    )
    def test_winograd_conv(self, model_runner, n, in_c, out_c, h, w, bias, activation):
        model = _make_model(n, in_c, out_c, h, w, bias, activation)
        rng = np.random.default_rng(123)
        x = rng.uniform(-1.0, 1.0, [n, in_c, h, w]).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=2e-2, rtol=2e-2, cos_threshold=0.999)
