#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for com.microsoft GroupNorm.

The kernel reduces groups of at least 16K elements in a split (stats,
finalize, apply) and smaller ones in one block per group. These shapes cover
both paths in both layouts: odd channels per group, G = 1, a group wider
than one thread block, and Stable Diffusion VAE sizes.

ORT CPU has no GroupNorm kernel, so the reference is an equivalent graph of
standard ops computed in fp32: Reshape to (N, G, -1), InstanceNormalization,
Reshape back, then the per-channel affine and the optional SiLU.
"""

import zlib

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import fp64_normalize_last_axis, make_model_from_nodes

_TENSOR_PROTO = {
    np.float16: TensorProto.FLOAT16,
    np.float32: TensorProto.FLOAT,
}

_EPSILON = 1e-5


def _affine(channels: int, dtype, seed: int):
    rng = np.random.default_rng(seed)
    gamma = rng.uniform(0.5, 1.5, [channels]).astype(dtype)
    beta = rng.uniform(-0.5, 0.5, [channels]).astype(dtype)
    return gamma, beta


def _group_norm_model(shape, groups, channels_last, activation, dtype, seed):
    channels = shape[-1] if channels_last else shape[1]
    gamma, beta = _affine(channels, dtype, seed)
    tp = _TENSOR_PROTO[dtype]
    node = helper.make_node(
        "GroupNorm",
        ["X", "gamma", "beta"],
        ["Y"],
        domain="com.microsoft",
        groups=groups,
        epsilon=_EPSILON,
        activation=activation,
        channels_last=channels_last,
    )
    return make_model_from_nodes(
        [node],
        [helper.make_tensor_value_info("X", tp, shape)],
        [helper.make_tensor_value_info("Y", tp, shape)],
        initializers=[
            numpy_helper.from_array(gamma, name="gamma"),
            numpy_helper.from_array(beta, name="beta"),
        ],
        extra_opsets=[helper.make_opsetid("com.microsoft", 1)],
    )


def _reference_model(
    shape, groups, channels_last, activation, dtype, seed, compute=np.float32
):
    """The same GroupNorm from ops ORT CPU implements, in *compute* precision
    (fp32 through InstanceNormalization, or fp64 from primitives)."""
    rank = len(shape)
    channels = shape[-1] if channels_last else shape[1]
    gamma, beta = _affine(channels, dtype, seed)
    nchw = [shape[0], shape[-1], *shape[1:-1]] if channels_last else list(shape)
    affine_shape = [channels] + [1] * (rank - 2)
    tp = _TENSOR_PROTO[dtype]
    fp64 = compute == np.float64

    inits = [
        numpy_helper.from_array(np.array([0, groups, -1], np.int64), "grouped"),
        numpy_helper.from_array(np.array(nchw, np.int64), "nchw"),
        numpy_helper.from_array(gamma.astype(compute).reshape(affine_shape), "gamma"),
        numpy_helper.from_array(beta.astype(compute).reshape(affine_shape), "beta"),
    ]
    to = TensorProto.DOUBLE if fp64 else TensorProto.FLOAT
    nodes = [helper.make_node("Cast", ["X"], ["xf"], to=to)]
    cur = "xf"
    if channels_last:
        to_first = [0, rank - 1, *range(1, rank - 1)]
        nodes.append(helper.make_node("Transpose", [cur], ["xt"], perm=to_first))
        cur = "xt"
    nodes.append(helper.make_node("Reshape", [cur, "grouped"], ["xg"]))
    if fp64:
        norm_nodes, norm_inits = fp64_normalize_last_axis("xg", "ng", _EPSILON)
        nodes += norm_nodes
        inits += norm_inits
    else:
        inits += [
            numpy_helper.from_array(np.ones([groups], np.float32), "ones"),
            numpy_helper.from_array(np.zeros([groups], np.float32), "zeros"),
        ]
        nodes.append(
            helper.make_node(
                "InstanceNormalization",
                ["xg", "ones", "zeros"],
                ["ng"],
                epsilon=_EPSILON,
            )
        )
    nodes += [
        helper.make_node("Reshape", ["ng", "nchw"], ["nr"]),
        helper.make_node("Mul", ["nr", "gamma"], ["scaled"]),
        helper.make_node("Add", ["scaled", "beta"], ["yf"]),
    ]
    cur = "yf"
    if activation:
        nodes += [
            helper.make_node("Sigmoid", [cur], ["sig"]),
            helper.make_node("Mul", [cur, "sig"], ["silu"]),
        ]
        cur = "silu"
    if channels_last:
        to_last = [0, *range(2, rank), 1]
        nodes.append(helper.make_node("Transpose", [cur], ["yl"], perm=to_last))
        cur = "yl"
    nodes.append(helper.make_node("Cast", [cur], ["Y"], to=tp))
    return make_model_from_nodes(
        nodes,
        [helper.make_tensor_value_info("X", tp, shape)],
        [helper.make_tensor_value_info("Y", tp, shape)],
        initializers=inits,
    )


_TOLERANCE = {np.float16: (4e-3, 4e-3), np.float32: (1e-4, 1e-4)}

# (shape, groups, channels_last, activation, dtype)
_CASES = [
    # small groups: one block per group
    ([1, 8, 4, 4], 4, 0, 0, np.float32),
    ([2, 8, 4, 4], 4, 0, 1, np.float32),
    ([1, 4, 4, 8], 4, 1, 0, np.float32),
    ([1, 8, 4, 4], 2, 0, 1, np.float16),
    ([1, 8, 8, 64], 32, 1, 1, np.float16),
    ([2, 64, 8, 8], 32, 0, 0, np.float16),
    # split reduction, channels last
    ([1, 64, 64, 512], 32, 1, 1, np.float16),
    ([2, 48, 48, 320], 32, 1, 0, np.float16),
    ([1, 64, 64, 40], 8, 1, 1, np.float32),
    ([1, 8, 8, 4096], 1, 1, 0, np.float16),
    # split reduction, channels first
    ([1, 512, 64, 64], 32, 0, 1, np.float16),
    ([2, 128, 31, 31], 1, 0, 0, np.float32),
    ([1, 40, 64, 64], 8, 0, 1, np.float32),
    # Stable Diffusion VAE decoder, fp32 I/O
    ([1, 96, 96, 512], 32, 1, 1, np.float32),
    ([1, 512, 96, 96], 32, 0, 1, np.float32),
]


def _case_id(case):
    shape, groups, channels_last, activation, dtype = case
    layout = "nhwc" if channels_last else "nchw"
    dims = "x".join(map(str, shape))
    return f"{np.dtype(dtype).name}-{layout}-{dims}-g{groups}-silu{activation}"


class TestGroupNorm:
    @pytest.mark.parametrize("case", _CASES, ids=[_case_id(c) for c in _CASES])
    def test_group_norm(self, model_runner, case):
        shape, groups, channels_last, activation, dtype = case
        seed = zlib.crc32(_case_id(case).encode())
        model = _group_norm_model(shape, groups, channels_last, activation, dtype, seed)
        reference = _reference_model(
            shape, groups, channels_last, activation, dtype, seed
        )
        rng = np.random.default_rng(seed)
        # An offset per channel keeps |mean| well above the stddev.
        channels = shape[-1] if channels_last else shape[1]
        offset_shape = (
            [channels] if channels_last else [channels] + [1] * (len(shape) - 2)
        )
        offset = rng.uniform(-3, 3, offset_shape)
        x = (rng.uniform(-1, 1, shape) + offset).astype(dtype)
        actual, expected = model_runner.run_sample(
            model, [x], reference_model=reference
        )
        atol, rtol = _TOLERANCE[dtype]
        compare_outputs(actual, expected, atol=atol, rtol=rtol)

    # |mean| / stddev near 260 with sparse outliers: E[x^2] - mean^2 in fp32
    # would miss the tolerance by about 7x on these inputs. The reference is
    # fp64, since ORT CPU's fp32 mean of such long rows is itself off by more
    # than the tolerance.
    @pytest.mark.parametrize(
        "case",
        [
            ([2, 64, 8, 8], 32, 0, 0, np.float32),
            ([1, 64, 64, 512], 32, 1, 0, np.float32),
            ([1, 512, 64, 64], 32, 0, 1, np.float32),
        ],
        ids=_case_id,
    )
    def test_group_norm_large_offset(self, model_runner, case):
        shape, groups, channels_last, activation, dtype = case
        seed = zlib.crc32(_case_id(case).encode())
        model = _group_norm_model(shape, groups, channels_last, activation, dtype, seed)
        reference = _reference_model(
            shape, groups, channels_last, activation, dtype, seed, compute=np.float64
        )
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1, 1, shape) + 200.0
        x.flat[::61] += 4.0
        actual, expected = model_runner.run_sample(
            model, [x.astype(dtype)], reference_model=reference
        )
        compare_outputs(actual, expected, atol=2e-4, rtol=2e-4)
