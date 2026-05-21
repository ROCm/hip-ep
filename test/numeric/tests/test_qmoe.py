#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the QMoE custom op (com.microsoft).

QMoE implements a quantized Mixture-of-Experts layer with top-k routing
and 4-bit quantized expert weights.

Dimensions and weight distributions match the GPT-OSS-20B ONNX model:
  hidden = 2880, intermediate = 5760 (gate + up fused)
  num_experts = 32, top_k = 4
  expert_weight_bits = 4, block_size = 32
  activation_type = swiglu

Weight statistics (from real model, layer 0):
  gate_up scales: all positive, mean |s| ~ 0.015, max ~ 0.29
  gate_up bias:   range [-3.7, 1.8], mean ~ -0.68 (skewed negative)
  down scales:    all positive, mean |s| ~ 0.011, max ~ 3.4 (sparse outliers)
  down bias:      range [-4.2, 5.4], mean ~ 0

Input context:
  hidden input:   post-layernorm, approximately normalised (range [-1, 1])
  router weights: raw logits from router projection (range [-5, 5], std ~ 1.2)
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

SEQ_LENS = [1, 128]
HIDDEN = 2880
INTERMEDIATE = 5760
NUM_EXPERTS = 32
TOP_K = 4
BLOCK_SIZE = 32
EXPERT_WEIGHT_BITS = 4


def _make_qmoe_model(
    batch: int,
    seq_len: int,
    hidden: int,
    intermediate: int,
    num_experts: int,
    top_k: int,
    block_size: int = 32,
    gate_up_scale_range: tuple[float, float] = (0.001, 0.05),
    gate_up_bias_range: tuple[float, float] = (-1.0, 0.5),
    down_scale_range: tuple[float, float] = (0.001, 0.02),
    down_bias_range: tuple[float, float] = (-0.5, 0.5),
    seed: int = 42,
):
    """Build a QMoE ONNX model with random quantized expert weights.

    Gate/up projection: hidden -> intermediate (per expert, fused).
    Down projection:    intermediate // 2 -> hidden (per expert).
    Weights are 4-bit quantized (packed into uint8).

    Scale and bias ranges are tuneable to match real model distributions.
    Scales are always positive (matching the real GPT-OSS-20B pattern).
    """
    down_in = intermediate // 2
    packed_gate_up = hidden // 2
    packed_down = down_in // 2
    gate_up_num_blocks = hidden // block_size
    down_num_blocks = down_in // block_size

    x = helper.make_tensor_value_info(
        "input",
        TensorProto.FLOAT16,
        [batch, seq_len, hidden],
    )
    router = helper.make_tensor_value_info(
        "router_weights",
        TensorProto.FLOAT16,
        [seq_len, num_experts],
    )
    output = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT16,
        [batch, seq_len, hidden],
    )

    rng = np.random.default_rng(seed)

    gate_up_qw = rng.integers(
        0,
        256,
        [num_experts, intermediate, packed_gate_up],
        dtype=np.uint8,
    )
    gate_up_scales = rng.uniform(
        gate_up_scale_range[0],
        gate_up_scale_range[1],
        [num_experts, intermediate, gate_up_num_blocks],
    ).astype(np.float16)
    gate_up_bias = rng.uniform(
        gate_up_bias_range[0],
        gate_up_bias_range[1],
        [num_experts, intermediate],
    ).astype(np.float16)

    down_qw = rng.integers(
        0,
        256,
        [num_experts, hidden, packed_down],
        dtype=np.uint8,
    )
    down_scales = rng.uniform(
        down_scale_range[0],
        down_scale_range[1],
        [num_experts, hidden, down_num_blocks],
    ).astype(np.float16)
    down_bias = rng.uniform(
        down_bias_range[0],
        down_bias_range[1],
        [num_experts, hidden],
    ).astype(np.float16)

    initializers = [
        numpy_helper.from_array(gate_up_qw, name="gate_up_qweight"),
        numpy_helper.from_array(gate_up_scales, name="gate_up_scales"),
        numpy_helper.from_array(gate_up_bias, name="gate_up_bias"),
        numpy_helper.from_array(down_qw, name="down_qweight"),
        numpy_helper.from_array(down_scales, name="down_scales"),
        numpy_helper.from_array(down_bias, name="down_bias"),
    ]

    node = helper.make_node(
        "QMoE",
        [
            "input",
            "router_weights",
            "gate_up_qweight",
            "gate_up_scales",
            "gate_up_bias",
            "down_qweight",
            "down_scales",
            "down_bias",
            "",
            "",
            "",
        ],
        ["output"],
        domain="com.microsoft",
        activation_type="swiglu",
        activation_alpha=1.702,
        activation_beta=1.0,
        expert_weight_bits=4,
        k=top_k,
        normalize_routing_weights=1,
        swiglu_fusion=1,
        use_sparse_mixer=0,
        block_size=block_size,
        swiglu_limit=7.0,
    )

    ms_opset = helper.make_opsetid("com.microsoft", 1)
    return make_model_from_nodes(
        [node],
        [x, router],
        [output],
        initializers=initializers,
        opset=21,
        extra_opsets=[ms_opset],
    )


class TestQMoE:
    @pytest.mark.parametrize(
        "seq_len,hidden,intermediate,num_experts,top_k",
        [
            (4, 64, 128, 4, 2),
            (8, 64, 128, 4, 2),
        ],
    )
    def test_qmoe(
        self,
        model_runner,
        seq_len,
        hidden,
        intermediate,
        num_experts,
        top_k,
    ):
        """QMoE with small shapes for fast sanity checking."""
        model = _make_qmoe_model(
            1,
            seq_len,
            hidden,
            intermediate,
            num_experts,
            top_k,
        )

        rng = np.random.default_rng(99)
        x = rng.uniform(-1, 1, [1, seq_len, hidden]).astype(np.float16)
        router = rng.standard_normal([seq_len, num_experts]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x, router])
        compare_outputs(actual, expected, atol=5e-2, rtol=1e-2, cos_threshold=0.999)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_qmoe_gpt_oss_shape(self, model_runner, seq_len):
        """QMoE with GPT-OSS-20B shapes: 32 experts, top-4, hidden=2880.

        Input: post-layernorm hidden states (range [-1, 1]).
        Router: raw logits (std ~ 1.2, range ~ [-5, 5]).
        Two chained quantised matmuls per expert path + SwiGLU in between,
        so accumulated error is higher than a single MatMulNBits.
        """
        model = _make_qmoe_model(
            1,
            seq_len,
            HIDDEN,
            INTERMEDIATE,
            NUM_EXPERTS,
            TOP_K,
            BLOCK_SIZE,
            gate_up_scale_range=(0.001, 0.05),
            gate_up_bias_range=(-1.0, 0.5),
            down_scale_range=(0.001, 0.02),
            down_bias_range=(-0.5, 0.5),
        )

        rng = np.random.default_rng(99)
        x = rng.uniform(-1, 1, [1, seq_len, HIDDEN]).astype(np.float16)
        router = (rng.standard_normal([seq_len, NUM_EXPERTS]) * 1.2).astype(
            np.float16,
        )

        actual, expected = model_runner.run_sample(model, [x, router])
        compare_outputs(actual, expected, atol=1e-1, rtol=1e-2, cos_threshold=0.99)
