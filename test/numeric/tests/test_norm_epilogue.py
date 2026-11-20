#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for a Relu, LeakyRelu or residual Add fused into a norm's epilogue.

InstanceNormalization, GroupNormalization and com.microsoft GroupNorm
followed by Relu, LeakyRelu, Add(norm, r), Add(r, norm) or Relu(Add(norm, r))
compile to one hip.group_norm that applies y = act(norm + r) in its output
pass. The shapes reach each kernel path: register-resident rows (small and
large groups), the split reduction, the scalar fallback for rows that are not
16-byte multiples, and channels-last.

The reference runs on ORT CPU in fp32 (fp16 inputs and parameters are cast up,
the output cast back). ORT CPU has no GroupNorm kernel, so GroupNorm is spelled
as Reshape + InstanceNormalization + affine there.
"""

import zlib

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_TENSOR_PROTO = {
    np.float16: TensorProto.FLOAT16,
    np.float32: TensorProto.FLOAT,
}
_TOLERANCE = {np.float16: (4e-3, 4e-3), np.float32: (1e-4, 1e-4)}
_EPSILON = 1e-5
_LEAKY_ALPHA = 0.2


def _norm_node(kind, groups, x, out):
    """kind: "in" (InstanceNormalization), "gn" (GroupNormalization-21) or
    "ms_nhwc" (com.microsoft GroupNorm, channels last)."""
    if kind == "in":
        return helper.make_node(
            "InstanceNormalization", [x, "scale", "bias"], [out], epsilon=_EPSILON
        )
    if kind == "gn":
        return helper.make_node(
            "GroupNormalization",
            [x, "scale", "bias"],
            [out],
            num_groups=groups,
            epsilon=_EPSILON,
        )
    return helper.make_node(
        "GroupNorm",
        [x, "scale", "bias"],
        [out],
        domain="com.microsoft",
        groups=groups,
        epsilon=_EPSILON,
        activation=0,
        channels_last=1,
    )


def _reference_norm(kind, groups, shape, x, out):
    """The norm from fp32 ops ORT CPU implements. GroupNorm is Reshape to
    (N, G, -1), InstanceNormalization, Reshape back, then the affine;
    channels-last input is transposed around it. Returns (nodes, inits) and
    reads "scale_c" / "bias_c" shaped for NCHW broadcasting."""
    if kind == "in":
        return [
            helper.make_node(
                "InstanceNormalization", [x, "scale", "bias"], [out], epsilon=_EPSILON
            )
        ], []
    rank = len(shape)
    nchw = [shape[0], shape[-1], *shape[1:-1]] if kind == "ms_nhwc" else list(shape)
    inits = [
        numpy_helper.from_array(np.array([0, groups, -1], np.int64), "grouped"),
        numpy_helper.from_array(np.array(nchw, np.int64), "nchw"),
        numpy_helper.from_array(np.ones([groups], np.float32), "ones"),
        numpy_helper.from_array(np.zeros([groups], np.float32), "zeros"),
    ]
    nodes = []
    cur = x
    if kind == "ms_nhwc":
        perm = [0, rank - 1, *range(1, rank - 1)]
        nodes.append(helper.make_node("Transpose", [cur], ["xt"], perm=perm))
        cur = "xt"
    nodes += [
        helper.make_node("Reshape", [cur, "grouped"], ["xg"]),
        helper.make_node(
            "InstanceNormalization", ["xg", "ones", "zeros"], ["ng"], epsilon=_EPSILON
        ),
        helper.make_node("Reshape", ["ng", "nchw"], ["nr"]),
        helper.make_node("Mul", ["nr", "scale_c"], ["scaled"]),
        helper.make_node(
            "Add", ["scaled", "bias_c"], ["affine" if kind == "ms_nhwc" else out]
        ),
    ]
    if kind == "ms_nhwc":
        perm = [0, *range(2, rank), 1]
        nodes.append(helper.make_node("Transpose", ["affine"], [out], perm=perm))
    return nodes, inits


def _epilogue_nodes(epilogue, n, out):
    """epilogue: "relu", "leaky", "add", "add_swapped" or "add_relu"."""
    if epilogue == "relu":
        return [helper.make_node("Relu", [n], [out])]
    if epilogue == "leaky":
        return [helper.make_node("LeakyRelu", [n], [out], alpha=_LEAKY_ALPHA)]
    if epilogue == "add":
        return [helper.make_node("Add", [n, "R"], [out])]
    if epilogue == "add_swapped":
        return [helper.make_node("Add", ["R", n], [out])]
    return [
        helper.make_node("Add", [n, "R"], ["sum"]),
        helper.make_node("Relu", ["sum"], [out]),
    ]


def _uses_residual(epilogue):
    return epilogue.startswith("add")


def _params(shape, kind, dtype, seed):
    channels = shape[-1] if kind == "ms_nhwc" else shape[1]
    rng = np.random.default_rng(seed)
    scale = rng.uniform(0.5, 1.5, [channels]).astype(dtype)
    bias = rng.uniform(-0.5, 0.5, [channels]).astype(dtype)
    return scale, bias


def _model(shape, kind, groups, epilogue, dtype, seed, reference=False):
    """The fused graph in *dtype*, or (reference=True) the same graph computed
    in fp32 between casts."""
    tp = _TENSOR_PROTO[dtype]
    scale, bias = _params(shape, kind, dtype, seed)
    nodes = []
    x, r = "X", "R"
    if not reference:
        inits = [
            numpy_helper.from_array(scale, "scale"),
            numpy_helper.from_array(bias, "bias"),
        ]
        nodes.append(_norm_node(kind, groups, x, "n"))
    else:
        affine_shape = [scale.size] + [1] * (len(shape) - 2)
        inits = [
            numpy_helper.from_array(scale.astype(np.float32), "scale"),
            numpy_helper.from_array(bias.astype(np.float32), "bias"),
            numpy_helper.from_array(
                scale.astype(np.float32).reshape(affine_shape), "scale_c"
            ),
            numpy_helper.from_array(
                bias.astype(np.float32).reshape(affine_shape), "bias_c"
            ),
        ]
        if dtype != np.float32:
            nodes.append(helper.make_node("Cast", ["X"], ["xf"], to=TensorProto.FLOAT))
            x = "xf"
            if _uses_residual(epilogue):
                nodes.append(
                    helper.make_node("Cast", ["R"], ["rf"], to=TensorProto.FLOAT)
                )
                r = "rf"
        norm_nodes, norm_inits = _reference_norm(kind, groups, shape, x, "n")
        nodes += norm_nodes
        inits += norm_inits
    cast_out = reference and dtype != np.float32
    epi = _epilogue_nodes(epilogue, "n", "yf" if cast_out else "Y")
    for node in epi:
        node.input[:] = [r if name == "R" else name for name in node.input]
    nodes += epi
    if cast_out:
        nodes.append(helper.make_node("Cast", ["yf"], ["Y"], to=tp))

    inputs = [helper.make_tensor_value_info("X", tp, shape)]
    if _uses_residual(epilogue):
        inputs.append(helper.make_tensor_value_info("R", tp, shape))
    extra = (
        [helper.make_opsetid("com.microsoft", 1)]
        if kind == "ms_nhwc" and not reference
        else None
    )
    return make_model_from_nodes(
        nodes,
        inputs,
        [helper.make_tensor_value_info("Y", tp, shape)],
        initializers=inits,
        opset=21,
        extra_opsets=extra,
    )


# (shape, kind, groups, epilogue, dtype)
_CASES = [
    # register-resident rows, small groups
    ([1, 8, 16, 16], "in", 8, "relu", np.float32),
    ([2, 16, 32, 32], "in", 16, "add", np.float16),
    ([1, 32, 64, 64], "in", 32, "add_relu", np.float32),
    ([2, 64, 32, 32], "gn", 8, "leaky", np.float16),
    # register-resident rows at the split size (16K elements, many rows)
    ([1, 128, 128, 128], "in", 128, "relu", np.float32),
    ([1, 128, 128, 128], "in", 128, "add_swapped", np.float32),
    # split reduction with the finalize merged into the apply
    ([1, 32, 512, 512], "in", 32, "relu", np.float32),
    ([1, 64, 256, 256], "in", 64, "add_relu", np.float32),
    ([1, 32, 128, 128], "gn", 4, "leaky", np.float32),
    # rows that are not 16-byte multiples: one-block-per-group fallback
    ([1, 8, 7, 7], "in", 8, "relu", np.float32),
    ([2, 6, 9, 9], "in", 6, "add_relu", np.float16),
    # channels last
    ([1, 16, 16, 64], "ms_nhwc", 8, "relu", np.float16),
    ([1, 64, 64, 128], "ms_nhwc", 32, "leaky", np.float32),
]


def _case_id(case):
    shape, kind, groups, epilogue, dtype = case
    dims = "x".join(map(str, shape))
    return f"{np.dtype(dtype).name}-{kind}-{dims}-g{groups}-{epilogue}"


class TestNormEpilogue:
    @pytest.mark.parametrize("case", _CASES, ids=[_case_id(c) for c in _CASES])
    def test_norm_epilogue(self, model_runner, case):
        shape, kind, groups, epilogue, dtype = case
        seed = zlib.crc32(_case_id(case).encode())
        model = _model(shape, kind, groups, epilogue, dtype, seed)
        reference = _model(shape, kind, groups, epilogue, dtype, seed, reference=True)
        rng = np.random.default_rng(seed)
        # A per-channel offset keeps |mean| above the stddev; the epilogue then
        # sees values on both sides of zero after normalization.
        channels = shape[-1] if kind == "ms_nhwc" else shape[1]
        offset_shape = (
            [channels] if kind == "ms_nhwc" else [channels] + [1] * (len(shape) - 2)
        )
        offset = rng.uniform(-3, 3, offset_shape)
        inputs = [(rng.uniform(-1, 1, shape) + offset).astype(dtype)]
        if _uses_residual(epilogue):
            inputs.append(rng.uniform(-1, 1, shape).astype(dtype))
        actual, expected = model_runner.run_sample(
            model, inputs, reference_model=reference
        )
        atol, rtol = _TOLERANCE[dtype]
        compare_outputs(actual, expected, atol=atol, rtol=rtol)
