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


def _make_qmoe_shared_expert_model(
    seq_len: int,
    hidden: int,
    inter: int,
    num_experts: int,
    top_k: int,
    shared_inter: int,
    block_size: int = 32,
    seed: int = 5,
):
    """Qwen3.5-MoE block: QMoE plus a sigmoid-gated shared expert.

      y = QMoE(x) + Sigmoid(x @ Wl) * ((x @ Wg) * Sigmoid(x @ Wg) * (x @ Wu)) @ Wd

    All shared projections are plain int4 MatMulNBits reading x, as in the
    exported model; the compiler folds them into the hip.qmoe.
    """
    rng = np.random.default_rng(seed)
    ms = "com.microsoft"
    x = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT16, [1, seq_len, hidden]
    )
    router = helper.make_tensor_value_info(
        "router_weights", TensorProto.FLOAT16, [seq_len, num_experts]
    )
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT16, [1, seq_len, hidden]
    )

    def expert_weights(name, rows, k, scale_range):
        w = rng.integers(0, 256, [num_experts, rows, k // 2], dtype=np.uint8)
        s = rng.uniform(*scale_range, [num_experts, rows, k // block_size])
        b = rng.uniform(-0.2, 0.2, [num_experts, rows])
        return [
            numpy_helper.from_array(w, name=f"{name}_qweight"),
            numpy_helper.from_array(s.astype(np.float16), name=f"{name}_scales"),
            numpy_helper.from_array(b.astype(np.float16), name=f"{name}_bias"),
        ]

    # Without a swiglu_limit nothing clamps the activations, so the scales are
    # chosen to keep outputs O(1) like the real model's.
    initializers = expert_weights("gate_up", 2 * inter, hidden, (0.001, 0.01))
    initializers += expert_weights("down", hidden, inter, (0.001, 0.01))
    nodes = [
        helper.make_node(
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
            ["routed"],
            domain=ms,
            activation_type="swiglu",
            activation_alpha=1.0,
            activation_beta=0.0,
            expert_weight_bits=4,
            k=top_k,
            normalize_routing_weights=1,
            swiglu_fusion=1,
            use_sparse_mixer=0,
            block_size=block_size,
            swiglu_limit=float("inf"),
        )
    ]

    def projection(name, a, n, k, scale_range=(-0.01, 0.01)):
        w = rng.integers(0, 256, [n, k // block_size, block_size // 2], dtype=np.uint8)
        s = rng.uniform(*scale_range, [n, k // block_size]).astype(np.float16)
        initializers.extend(
            [
                numpy_helper.from_array(w, name=f"{name}_w"),
                numpy_helper.from_array(s, name=f"{name}_s"),
            ]
        )
        nodes.append(
            helper.make_node(
                "MatMulNBits",
                [a, f"{name}_w", f"{name}_s"],
                [name],
                domain=ms,
                K=k,
                N=n,
                bits=4,
                block_size=block_size,
                accuracy_level=4,
            )
        )

    projection("sh_gate", "input", shared_inter, hidden)
    projection("sh_up", "input", shared_inter, hidden)
    nodes += [
        helper.make_node("Sigmoid", ["sh_gate"], ["sh_gate_sig"]),
        helper.make_node("Mul", ["sh_gate", "sh_gate_sig"], ["sh_silu"]),
        helper.make_node("Mul", ["sh_silu", "sh_up"], ["sh_act"]),
    ]
    projection("sh_down", "sh_act", hidden, shared_inter)
    projection("sh_logit", "input", 1, hidden, scale_range=(-0.1, 0.1))
    nodes += [
        helper.make_node("Sigmoid", ["sh_logit"], ["sh_prob"]),
        helper.make_node("Mul", ["sh_down", "sh_prob"], ["shared"]),
        helper.make_node("Add", ["routed", "shared"], ["output"]),
    ]
    return make_model_from_nodes(
        nodes,
        [x, router],
        [output],
        initializers=initializers,
        opset=21,
        extra_opsets=[helper.make_opsetid(ms, 1)],
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

    @pytest.mark.parametrize("seq_len", [2, 37, 512])
    def test_qmoe_gpt_oss_shape_skewed_routing(self, model_runner, seq_len):
        """GPT-OSS-20B shapes with routing skewed toward low expert ids, so
        expert slices range from a single row to hundreds of rows. Prefill
        cuts experts of <= 16 rows into one tile and larger ones into several
        32-row tiles; this covers both kinds, an expert spanning many tiles,
        and the routed-pair reduction over unevenly filled experts.
        """
        model = _make_qmoe_model(
            1,
            seq_len,
            HIDDEN,
            INTERMEDIATE,
            NUM_EXPERTS,
            TOP_K,
            BLOCK_SIZE,
        )

        rng = np.random.default_rng(8)
        x = rng.uniform(-1, 1, [1, seq_len, HIDDEN]).astype(np.float16)
        skew = np.linspace(3.0, -3.0, NUM_EXPERTS)
        router = (rng.standard_normal([seq_len, NUM_EXPERTS]) * 1.2 + skew).astype(
            np.float16,
        )
        # A tie at the k-th logit makes the expert choice backend-defined.
        ranked = -np.sort(-router.astype(np.float32), axis=1)
        assert not (ranked[:, TOP_K - 1] == ranked[:, TOP_K]).any()

        actual, expected = model_runner.run_sample(model, [x, router])
        compare_outputs(actual, expected, atol=1e-1, rtol=1e-2, cos_threshold=0.99)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    @pytest.mark.parametrize(
        "shared_inter", [512, 256], ids=["shared_slot", "shared_separate"]
    )
    def test_qmoe_shared_expert_qwen35_shape(self, model_runner, seq_len, shared_inter):
        """Qwen3.5-35B-A3B MoE block (hidden 2048, expert inter 512, top-8)
        with its sigmoid-gated shared expert, at 32 instead of 256 experts.

        A shared expert as wide as the routed ones runs as one more slot of
        the fused decode at S=1; a narrower one, and every S=128 run, goes
        through the separate gate/up, swiglu, down and combine launches.
        """
        hidden, inter, num_experts, top_k = 2048, 512, 32, 8
        model = _make_qmoe_shared_expert_model(
            seq_len, hidden, inter, num_experts, top_k, shared_inter
        )
        rng = np.random.default_rng(17)
        x = rng.uniform(-1, 1, [1, seq_len, hidden]).astype(np.float16)
        router = (rng.standard_normal([seq_len, num_experts]) * 1.2).astype(
            np.float16
        )
        ranked = -np.sort(-router.astype(np.float32), axis=1)
        assert not (ranked[:, top_k - 1] == ranked[:, top_k]).any()

        actual, expected = model_runner.run_sample(model, [x, router])
        compare_outputs(actual, expected, atol=1e-1, rtol=1e-2, cos_threshold=0.99)
