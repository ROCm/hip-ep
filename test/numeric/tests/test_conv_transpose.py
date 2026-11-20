#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for ConvTranspose, which hip-decompose-conv-transpose splits into
stride-1 convolutions, one per output phase, and an interleave.

The cases cover the three shapes of that split against the ORT CPU reference:

- equal phases (k4 s2): one wide convolution, with the bias folded into it;
- unequal phases within the padding limit (k3 s2, k5 s3): one wide
  convolution over zero-padded phase filters;
- unequal phases over the limit (k5 s4): one convolution per phase.

A trailing LeakyRelu checks the pointwise op that moves above the interleave
into the convolution's epilogue.

The "add" bias is the exporters' form of a bias: a bias-free ConvTranspose,
then Add(Reshape([1,1,1,M] -> [1,M,1,1])), folded into the convolution's bias.
Like the filter, the fold reads only constants at or below the importer's
inline threshold; run with `XLNX_ABI_2_0_CLONE_EXTERNAL_DATA_THRESHOLD`
raised to exercise it, otherwise the same graphs check the unfolded path.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_TYPES = {
    np.float16: TensorProto.FLOAT16,
    np.float32: TensorProto.FLOAT,
}


def _out_size(size: int, k: int, s: int, p: int, out_pad: int) -> int:
    return (size - 1) * s - 2 * p + k + out_pad


def _make_conv_transpose_model(
    in_c: int,
    out_c: int,
    k: int,
    s: int,
    p: int,
    out_pad: int,
    spatial: list,
    dtype,
    bias: str,
    leaky: bool,
    seed: int = 7,
    next_conv: bool = False,
):
    """Build a ConvTranspose model with a weight initializer.

    bias: "none", "const" (initializer), "input" (second graph input), or
    "add" (a bias-free ConvTranspose, then Add of a reshaped initializer).
    next_conv: follow with a 3x3 convolution, out_c -> out_c, same size.
    """
    rank = len(spatial)
    elem = _TYPES[dtype]
    out_spatial = [_out_size(d, k, s, p, out_pad) for d in spatial]
    inputs = [helper.make_tensor_value_info("input", elem, [1, in_c, *spatial])]
    out = helper.make_tensor_value_info("output", elem, [1, out_c, *out_spatial])

    rng = np.random.default_rng(seed)
    w = rng.uniform(-0.2, 0.2, [in_c, out_c, *([k] * rank)]).astype(dtype)
    inits = [numpy_helper.from_array(w, name="weight")]
    node_inputs = ["input", "weight"]
    if bias == "const":
        b = rng.uniform(-0.5, 0.5, [out_c]).astype(dtype)
        inits.append(numpy_helper.from_array(b, name="bias"))
        node_inputs.append("bias")
    elif bias == "input":
        inputs.append(helper.make_tensor_value_info("bias", elem, [out_c]))
        node_inputs.append("bias")

    nodes = [
        helper.make_node(
            "ConvTranspose",
            node_inputs,
            ["convt"],
            kernel_shape=[k] * rank,
            strides=[s] * rank,
            pads=[p] * (2 * rank),
            output_padding=[out_pad] * rank,
        )
    ]
    if bias == "add":
        b = rng.uniform(-0.5, 0.5, [1] * (rank + 1) + [out_c]).astype(dtype)
        shape = np.array([1, out_c] + [1] * rank, dtype=np.int64)
        inits.append(numpy_helper.from_array(b, name="bias_nhwc"))
        inits.append(numpy_helper.from_array(shape, name="bias_shape"))
        nodes.append(
            helper.make_node("Reshape", ["bias_nhwc", "bias_shape"], ["bias_nchw"])
        )
        nodes.append(helper.make_node("Add", ["convt", "bias_nchw"], ["biased"]))
    if leaky:
        nodes.append(
            helper.make_node("LeakyRelu", [nodes[-1].output[0]], ["act"], alpha=0.2)
        )
    if next_conv:
        w2 = rng.uniform(-0.2, 0.2, [out_c, out_c] + [3] * rank).astype(dtype)
        inits.append(numpy_helper.from_array(w2, name="next_weight"))
        nodes.append(
            helper.make_node(
                "Conv",
                [nodes[-1].output[0], "next_weight"],
                ["next"],
                kernel_shape=[3] * rank,
                pads=[1] * (2 * rank),
            )
        )
    nodes[-1].output[0] = "output"
    return make_model_from_nodes(nodes, inputs, [out], initializers=inits)


class TestConvTranspose:
    @pytest.mark.parametrize("dtype", [np.float16, np.float32])
    @pytest.mark.parametrize(
        "in_c,out_c,k,s,p,out_pad,spatial,bias,leaky",
        [
            # Equal phases: one wide convolution.
            (8, 16, 4, 2, 1, 0, [8, 8], "const", False),
            (16, 8, 4, 2, 1, 0, [8, 8], "const", True),
            (8, 8, 4, 2, 1, 0, [8, 8], "none", True),
            (8, 4, 4, 2, 1, 0, [6, 6], "input", False),
            # Unequal phases, zero-padded to one wide convolution.
            (8, 4, 3, 2, 1, 1, [8, 8], "const", False),
            (4, 8, 3, 2, 1, 1, [7, 7], "none", True),
            (4, 8, 5, 3, 1, 0, [6, 6], "const", True),
            # Unequal phases over the padding limit: one convolution each.
            (4, 8, 5, 4, 0, 0, [6, 6], "const", False),
            (4, 8, 5, 4, 0, 0, [6, 6], "input", True),
        ],
    )
    def test_conv_transpose(
        self, model_runner, dtype, in_c, out_c, k, s, p, out_pad, spatial, bias, leaky
    ):
        model = _make_conv_transpose_model(
            in_c, out_c, k, s, p, out_pad, spatial, dtype, bias, leaky
        )
        rng = np.random.default_rng(123)
        feeds = [rng.uniform(-1.0, 1.0, [1, in_c, *spatial]).astype(dtype)]
        if bias == "input":
            feeds.append(rng.uniform(-0.5, 0.5, [out_c]).astype(dtype))

        actual, expected = model_runner.run_sample(model, feeds)
        tol = 1e-2 if dtype == np.float16 else 1e-4
        compare_outputs(actual, expected, atol=tol, rtol=tol, cos_threshold=0.999)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32])
    @pytest.mark.parametrize(
        "in_c,out_c,k,s,p,spatial,leaky,next_conv",
        [
            # The upsampling block of the gnet models: ConvTranspose,
            # Add(Reshape(bias)), LeakyRelu, 3x3 Conv.
            (8, 16, 4, 2, 1, [8, 8], True, True),
            (16, 8, 4, 2, 1, [8, 8], False, False),
            # Zero-padded phases, and phases over the padding limit.
            (4, 8, 3, 2, 1, [7, 7], True, True),
            (4, 8, 5, 4, 0, [6, 6], True, False),
        ],
    )
    def test_conv_transpose_bias_add(
        self, model_runner, dtype, in_c, out_c, k, s, p, spatial, leaky, next_conv
    ):
        out_pad = 1 if (k, s) == (3, 2) else 0
        model = _make_conv_transpose_model(
            in_c,
            out_c,
            k,
            s,
            p,
            out_pad,
            spatial,
            dtype,
            "add",
            leaky,
            next_conv=next_conv,
        )
        rng = np.random.default_rng(123)
        feeds = [rng.uniform(-1.0, 1.0, [1, in_c, *spatial]).astype(dtype)]

        actual, expected = model_runner.run_sample(model, feeds)
        tol = 1e-2 if dtype == np.float16 else 1e-4
        compare_outputs(actual, expected, atol=tol, rtol=tol, cos_threshold=0.999)
