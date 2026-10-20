#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Sigmoid-gated product (hip.sigmoid_mul).

A gated attention layer (Qwen3.5) scales its fp16 output by the sigmoid of a
gate projection before the output projection:

    y = Mul(x, Sigmoid(g))

hip-fusion-transform folds the pair into one hip.sigmoid_mul, and with an
int4 projection consuming y at decode the op also writes y's int8
quantization (hip-hoist-dynamic-quant).
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


def _make_sigmoid_mul_model(
    seq_len: int, width: int, proj_n: int, rng: np.random.Generator
):
    """The gate, plus an int4 MatMulNBits on its output when proj_n > 0."""
    tp = TensorProto.FLOAT16
    inits = []
    nodes = [
        helper.make_node("Sigmoid", ["g"], ["g_sig"]),
        helper.make_node("Mul", ["x", "g_sig"], ["y"]),
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
        helper.make_tensor_value_info(name, tp, [1, seq_len, width]) for name in ("x", "g")
    ]
    return make_model_from_nodes(
        nodes,
        inputs,
        outputs,
        initializers=inits,
        opset=17,
        extra_opsets=[helper.make_opsetid("com.microsoft", 1)],
    )


class TestSigmoidMul:
    # 4096 is the Qwen3.5-35B-A3B attention width; 37 has an odd row, which
    # takes the scalar kernel.
    @pytest.mark.parametrize("seq_len,width", [(1, 4096), (128, 4096), (3, 37)])
    def test_sigmoid_mul(self, model_runner, seq_len, width):
        rng = np.random.default_rng(seq_len * 1000 + width)
        model = _make_sigmoid_mul_model(seq_len, width, 0, rng)
        x = rng.standard_normal((1, seq_len, width)).astype(np.float16)
        g = (rng.standard_normal((1, seq_len, width)) * 4).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x, g])
        compare_outputs(actual, expected, atol=2e-3, rtol=1e-3)

    # S=1 and S=2 project y through the gate's int8 quantization, S=128 runs
    # the fp projection, for which the gate writes none.
    @pytest.mark.parametrize("seq_len", [1, 2, 128])
    def test_sigmoid_mul_projection(self, model_runner, seq_len):
        rng = np.random.default_rng(31 + seq_len)
        width = 4096
        model = _make_sigmoid_mul_model(seq_len, width, 96, rng)
        x = rng.standard_normal((1, seq_len, width)).astype(np.float16)
        g = (rng.standard_normal((1, seq_len, width)) * 4).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x, g])
        compare_outputs(actual, expected, atol=5e-2, rtol=1e-2, cos_threshold=0.999)
