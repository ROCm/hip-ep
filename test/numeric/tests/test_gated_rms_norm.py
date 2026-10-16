#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""SiLU-gated per-head RMS norm (hip.gated_rms_norm).

A Qwen3.5 gated-delta-net layer gates its normalized attention output before
the output projection:

    n = Reshape(SimplifiedLayerNormalization(Reshape(x, [B, T, H, D]), w))
    y = Cast<f16>(Cast<f32>(n) * (zf * Sigmoid(zf)))      zf = Cast<f32>(z)

hip-fusion-transform folds the chain into one hip.gated_rms_norm, and with
an int4 projection consuming y at decode the norm also writes y's int8
quantization (hip-hoist-dynamic-quant).
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


def _make_gated_norm_model(
    seq_len: int, heads: int, head_dim: int, proj_n: int, rng: np.random.Generator
):
    """The gated norm, plus an int4 MatMulNBits on its output when proj_n > 0."""
    tp = TensorProto.FLOAT16
    width = heads * head_dim
    inits = [
        helper.make_tensor("heads_shape", TensorProto.INT64, [4], [0, 0, heads, head_dim]),
        helper.make_tensor("flat_shape", TensorProto.INT64, [3], [0, 0, -1]),
        numpy_helper.from_array(
            rng.uniform(0.5, 1.5, head_dim).astype(np.float16), "gamma"
        ),
    ]
    nodes = [
        helper.make_node("Reshape", ["x", "heads_shape"], ["x_h"]),
        helper.make_node(
            "SimplifiedLayerNormalization",
            ["x_h", "gamma"],
            ["n_h"],
            axis=-1,
            epsilon=1e-6,
            stash_type=1,
        ),
        helper.make_node("Reshape", ["n_h", "flat_shape"], ["n"]),
        helper.make_node("Cast", ["n"], ["n32"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["z"], ["z32"], to=TensorProto.FLOAT),
        helper.make_node("Sigmoid", ["z32"], ["z_sig"]),
        helper.make_node("Mul", ["z32", "z_sig"], ["z_silu"]),
        helper.make_node("Mul", ["n32", "z_silu"], ["y32"]),
        helper.make_node("Cast", ["y32"], ["y"], to=TensorProto.FLOAT16),
    ]
    outputs = [helper.make_tensor_value_info("y", tp, [1, seq_len, width])]
    if proj_n:
        block = 32
        blocks = width // block
        inits += [
            numpy_helper.from_array(
                rng.integers(0, 256, [proj_n, blocks, block // 2], dtype=np.uint8),
                "w_q4",
            ),
            numpy_helper.from_array(
                rng.uniform(-0.002, 0.002, [proj_n, blocks]).astype(np.float16),
                "w_scales",
            ),
        ]
        nodes.append(
            helper.make_node(
                "MatMulNBits",
                ["y", "w_q4", "w_scales"],
                ["p"],
                domain="com.microsoft",
                K=width,
                N=proj_n,
                bits=4,
                block_size=block,
                accuracy_level=4,
            )
        )
        outputs.append(helper.make_tensor_value_info("p", tp, [1, seq_len, proj_n]))
    inputs = [
        helper.make_tensor_value_info(name, tp, [1, seq_len, width]) for name in ("x", "z")
    ]
    return make_model_from_nodes(
        nodes,
        inputs,
        outputs,
        initializers=inits,
        opset=17,
        extra_opsets=[helper.make_opsetid("com.microsoft", 1)],
    )


class TestGatedRmsNorm:
    """Head sizes 16..512 cover every threads-per-head variant of the kernel
    (one item per 32, 64, 128 and 256 threads, the last also striding)."""

    @pytest.mark.parametrize(
        "seq_len,heads,head_dim",
        [
            (1, 32, 128),  # Qwen3.5-35B-A3B decode
            (128, 32, 128),
            (3, 4, 16),
            (2, 2, 256),
            (1, 2, 512),
        ],
    )
    def test_gated_rms_norm(self, model_runner, seq_len, heads, head_dim):
        rng = np.random.default_rng(seq_len * 1000 + heads + head_dim)
        model = _make_gated_norm_model(seq_len, heads, head_dim, 0, rng)
        width = heads * head_dim
        x = rng.standard_normal((1, seq_len, width)).astype(np.float16)
        z = (rng.standard_normal((1, seq_len, width)) * 2).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x, z])
        compare_outputs(actual, expected, atol=1e-2, rtol=1e-2, cos_threshold=0.9999)

    # S=1 and S=2 project y through the norm's int8 quantization, S=128 runs
    # the fp projection, for which the norm writes none.
    @pytest.mark.parametrize("seq_len", [1, 2, 128])
    def test_gated_rms_norm_projection(self, model_runner, seq_len):
        rng = np.random.default_rng(23 + seq_len)
        heads, head_dim = 32, 128
        model = _make_gated_norm_model(seq_len, heads, head_dim, 96, rng)
        width = heads * head_dim
        x = rng.standard_normal((1, seq_len, width)).astype(np.float16)
        z = (rng.standard_normal((1, seq_len, width)) * 2).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x, z])
        compare_outputs(actual, expected, atol=5e-2, rtol=1e-2, cos_threshold=0.999)
