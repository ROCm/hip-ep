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

TestQMoERelu2 covers swiglu_fusion=0 with activation_type="relu2". ORT CPU
has no relu2 MoE kernel, so it compares against an equivalent graph.
"""

import os
from dataclasses import dataclass

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

# 1 is decode and 128 is prefill; the runtime picks its MoE strategy on token count
# (hipdnn_ep_qmoe_fused_max_tokens): 2..8 run the fused decode kernels as one burst
# over all token*k slots, 9 and up take the grouped path. Speculative verify widths
# live in the first band.
SEQ_LENS = [1, 2, 5, 8, 9, 128]
FUSED_MAX_TOKENS = int(os.environ.get("HIPDNN_EP_QMOE_FUSED_MAX_TOKENS", "8"))
HIDDEN = 2880
INTERMEDIATE = 5760
NUM_EXPERTS = 32
TOP_K = 4
BLOCK_SIZE = 32
EXPERT_WEIGHT_BITS = 4


def _compare_gpt_oss_shape(actual, expected, seq_len):
    """The fused decode kernels quantize activations to int8 per 32-element
    group by default (W4A8, HIPDNN_EP_MATMUL_DP4A), an error that scales with
    the output (|output| reaches ~100 at these shapes). Fused token counts are
    bounded at 2% of the largest |output| with a tighter cosine instead of the
    grouped path's fixed atol.
    """
    if seq_len <= FUSED_MAX_TOKENS:
        scale = float(np.abs(expected[0].astype(np.float32)).max())
        compare_outputs(
            actual, expected, atol=2e-2 * scale, rtol=1e-2, cos_threshold=0.9999
        )
    else:
        compare_outputs(actual, expected, atol=1e-1, rtol=1e-2, cos_threshold=0.99)


def _make_qmoe_model(
    batch: int,
    seq_len: int,
    hidden: int,
    intermediate: int,
    num_experts: int,
    top_k: int,
    block_size: int = 32,
    expert_weight_bits: int = EXPERT_WEIGHT_BITS,
    gate_up_scale_range: tuple[float, float] = (0.001, 0.05),
    gate_up_bias_range: tuple[float, float] = (-1.0, 0.5),
    down_scale_range: tuple[float, float] = (0.001, 0.02),
    down_bias_range: tuple[float, float] = (-0.5, 0.5),
    seed: int = 42,
    with_mix_weights: bool = False,
    with_zero_points: bool = False,
    normalize_routing_weights: int = 1,
    swiglu_limit: float | None = 7.0,
):
    """Build a QMoE ONNX model with random quantized expert weights.

    Gate/up projection: hidden -> intermediate (per expert, fused).
    Down projection:    intermediate // 2 -> hidden (per expert).
    Weights are packed into uint8.

    Scale and bias ranges are tuneable to match real model distributions.
    Scales are always positive (matching the real GPT-OSS-20B pattern).

    with_mix_weights adds the optional QMoE router_weights input (input 14) as
    a graph input named "mix_weights": the router logits then only select the
    experts and the mixing weights are gathered from it.
    """
    down_in = intermediate // 2
    values_per_byte = 8 // expert_weight_bits
    packed_gate_up = (hidden + values_per_byte - 1) // values_per_byte
    packed_down = (down_in + values_per_byte - 1) // values_per_byte
    gate_up_num_blocks = (hidden + block_size - 1) // block_size
    down_num_blocks = (down_in + block_size - 1) // block_size

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

    def make_zero_points(shape: tuple[int, int, int]) -> np.ndarray:
        values_per_byte = 8 // expert_weight_bits
        values = rng.integers(0, 1 << expert_weight_bits, shape, dtype=np.uint8)
        packed_shape = (
            *shape[:-1],
            (shape[-1] + values_per_byte - 1) // values_per_byte,
        )
        packed = np.zeros(packed_shape, dtype=np.uint8)
        for i in range(shape[-1]):
            packed[..., i // values_per_byte] |= values[..., i] << (
                (i % values_per_byte) * expert_weight_bits
            )
        return packed

    initializers = [
        numpy_helper.from_array(gate_up_qw, name="gate_up_qweight"),
        numpy_helper.from_array(gate_up_scales, name="gate_up_scales"),
        numpy_helper.from_array(gate_up_bias, name="gate_up_bias"),
        numpy_helper.from_array(down_qw, name="down_qweight"),
        numpy_helper.from_array(down_scales, name="down_scales"),
        numpy_helper.from_array(down_bias, name="down_bias"),
    ]
    if with_zero_points:
        initializers += [
            numpy_helper.from_array(
                make_zero_points((num_experts, intermediate, gate_up_num_blocks)),
                name="gate_up_zero_points",
            ),
            numpy_helper.from_array(
                make_zero_points((num_experts, hidden, down_num_blocks)),
                name="down_zero_points",
            ),
        ]

    node_inputs = [
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
    ]
    node_inputs += [
        "gate_up_zero_points" if with_zero_points else "",
        "down_zero_points" if with_zero_points else "",
        "",
    ]
    graph_inputs = [x, router]
    if with_mix_weights:
        node_inputs.append("mix_weights")
        graph_inputs.append(
            helper.make_tensor_value_info(
                "mix_weights",
                TensorProto.FLOAT16,
                [seq_len, num_experts],
            )
        )

    attributes = {
        "activation_type": "swiglu",
        "activation_alpha": 1.702,
        "activation_beta": 1.0,
        "expert_weight_bits": expert_weight_bits,
        "k": top_k,
        "normalize_routing_weights": normalize_routing_weights,
        "swiglu_fusion": 1,
        "use_sparse_mixer": 0,
        "block_size": block_size,
    }
    if swiglu_limit is not None:
        attributes["swiglu_limit"] = swiglu_limit
    node = helper.make_node(
        "QMoE",
        node_inputs,
        ["output"],
        domain="com.microsoft",
        **attributes,
    )

    ms_opset = helper.make_opsetid("com.microsoft", 1)
    return make_model_from_nodes(
        [node],
        graph_inputs,
        [output],
        initializers=initializers,
        opset=21,
        extra_opsets=[ms_opset],
    )


class TestQMoE:
    @pytest.mark.parametrize("seq_len", [1, 8])
    @pytest.mark.parametrize("with_zero_points", [False, True])
    @pytest.mark.parametrize("swiglu_limit", [7.0, None])
    def test_two_bit(self, model_runner, seq_len, with_zero_points, swiglu_limit):
        hidden = 64
        intermediate = 128
        num_experts = 4
        top_k = 2
        model = _make_qmoe_model(
            1,
            seq_len,
            hidden,
            intermediate,
            num_experts,
            top_k,
            block_size=32,
            expert_weight_bits=2,
            with_zero_points=with_zero_points,
            swiglu_limit=swiglu_limit,
        )

        rng = np.random.default_rng(17)
        x = rng.uniform(-1, 1, [1, seq_len, hidden]).astype(np.float16)
        router = rng.standard_normal([seq_len, num_experts]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x, router])
        compare_outputs(actual, expected, atol=1e-1, rtol=2e-2, cos_threshold=0.99)

    @pytest.mark.parametrize("seq_len", [1, 8])
    @pytest.mark.parametrize("expert_weight_bits", [2, 4])
    def test_partial_quantization_block(
        self, model_runner, seq_len, expert_weight_bits
    ):
        hidden = 96
        intermediate = 192
        num_experts = 4
        model = _make_qmoe_model(
            1,
            seq_len,
            hidden,
            intermediate,
            num_experts,
            top_k=2,
            block_size=64,
            expert_weight_bits=expert_weight_bits,
        )

        rng = np.random.default_rng(23)
        x = rng.uniform(-1, 1, [1, seq_len, hidden]).astype(np.float16)
        router = rng.standard_normal([seq_len, num_experts]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x, router])
        compare_outputs(actual, expected, atol=1e-1, rtol=2e-2, cos_threshold=0.99)

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

    @pytest.mark.parametrize("seq_len", [1, 37])
    @pytest.mark.parametrize("top_k", [4, 8])
    @pytest.mark.parametrize("num_experts", [64, 128, 256])
    def test_qmoe_many_experts(self, model_runner, num_experts, top_k, seq_len):
        """Expert counts of gpt-oss-120b (128) and Qwen3.5-35B-A3B (256) and
        between, at small hidden/intermediate sizes. Routing holds 2, 4 or 8
        experts per wave lane for these counts; seq_len=1 is the fused decode
        path, 37 the prefill path.
        """
        model = _make_qmoe_model(1, seq_len, 64, 128, num_experts, top_k)

        rng = np.random.default_rng(7)
        x = rng.uniform(-1, 1, [1, seq_len, 64]).astype(np.float16)
        # Each token's logits are a shuffle of distinct fp16 values: a tie at
        # the k-th logit would make the expert choice backend-defined.
        levels = np.linspace(-4.0, 4.0, num_experts).astype(np.float16)
        assert len(np.unique(levels)) == num_experts
        router = rng.permuted(np.tile(levels, (seq_len, 1)), axis=1)

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
        _compare_gpt_oss_shape(actual, expected, seq_len)

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
        _compare_gpt_oss_shape(actual, expected, seq_len)

    @pytest.mark.parametrize("normalize", [0, 1])
    @pytest.mark.parametrize(
        "seq_len",
        [
            pytest.param(
                1,
                marks=pytest.mark.xfail(
                    reason="fused decode exceeds atol=0.1 here, as "
                    "test_qmoe_gpt_oss_shape[1] does without router_weights",
                    strict=False,
                ),
            ),
            37,
            128,
        ],
    )
    def test_qmoe_gpt_oss_shape_mix_weights(self, model_runner, seq_len, normalize):
        """router_weights (QMoE input 14): the logits only pick the top-k
        experts; the mixing weights are gathered from a separate tensor and
        renormalised only when normalize_routing_weights=1. The mixing
        weights are uncorrelated with the logits, so taking the softmax of
        the logits instead would not match. Their top-k sums vary around 1 so
        the unnormalised case keeps the output scale (and fp16 error) of the
        normalised one. seq_len=1 covers the fused decode path, the others
        the prefill paths.
        """
        model = _make_qmoe_model(
            1,
            seq_len,
            HIDDEN,
            INTERMEDIATE,
            NUM_EXPERTS,
            TOP_K,
            BLOCK_SIZE,
            with_mix_weights=True,
            normalize_routing_weights=normalize,
        )

        rng = np.random.default_rng(123)
        x = rng.uniform(-1, 1, [1, seq_len, HIDDEN]).astype(np.float16)
        router = (rng.standard_normal([seq_len, NUM_EXPERTS]) * 1.2).astype(
            np.float16,
        )
        ranked = -np.sort(-router.astype(np.float32), axis=1)
        assert not (ranked[:, TOP_K - 1] == ranked[:, TOP_K]).any()
        mix = rng.uniform(0.05, 0.45, [seq_len, NUM_EXPERTS]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x, router, mix])
        compare_outputs(actual, expected, atol=1e-1, rtol=1e-2, cos_threshold=0.99)


@dataclass(frozen=True)
class Relu2Config:
    seq_len: int
    hidden: int
    inter: int
    num_experts: int
    top_k: int
    with_bias: bool = False
    with_mix_weights: bool = False


def _make_relu2_weights(cfg: Relu2Config, seed: int = 42) -> dict[str, np.ndarray]:
    """QMoE-layout weights [E, N, K/2] and scales [E, N, K/block]."""
    rng = np.random.default_rng(seed)
    e = cfg.num_experts
    w = {}

    def quantize(name: str, n: int, k: int, gain: float) -> None:
        w[f"{name}_qweight"] = rng.integers(0, 256, [e, n, k // 2], dtype=np.uint8)
        # Scales ~ 1/sqrt(K) keep each projection near unit magnitude, where
        # an absolute fp16 tolerance still discriminates.
        base = gain / np.sqrt(k)
        w[f"{name}_scales"] = rng.uniform(
            0.5 * base, 1.5 * base, [e, n, k // BLOCK_SIZE]
        ).astype(np.float16)

    quantize("fc1", cfg.inter, cfg.hidden, gain=0.37)
    quantize("fc2", cfg.hidden, cfg.inter, gain=0.2)
    if cfg.with_bias:
        w["fc1_bias"] = rng.uniform(-0.5, 0.5, [e, cfg.inter]).astype(np.float16)
        w["fc2_bias"] = rng.uniform(-0.5, 0.5, [e, cfg.hidden]).astype(np.float16)
    return w


def _relu2_io(cfg: Relu2Config):
    inputs = [
        helper.make_tensor_value_info(
            "input", TensorProto.FLOAT16, [1, cfg.seq_len, cfg.hidden]
        ),
        helper.make_tensor_value_info(
            "router_logits", TensorProto.FLOAT16, [cfg.seq_len, cfg.num_experts]
        ),
    ]
    if cfg.with_mix_weights:
        inputs.append(
            helper.make_tensor_value_info(
                "mix_weights", TensorProto.FLOAT16, [cfg.seq_len, cfg.num_experts]
            )
        )
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT16, [1, cfg.seq_len, cfg.hidden]
    )
    return inputs, output


def _make_relu2_qmoe_model(cfg: Relu2Config, w: dict[str, np.ndarray]):
    """A single com.microsoft::QMoE node with swiglu_fusion=0 and relu2."""
    inputs, output = _relu2_io(cfg)
    names = [
        "fc1_qweight",
        "fc1_scales",
        "fc1_bias",
        "fc2_qweight",
        "fc2_scales",
        "fc2_bias",
    ]
    initializers = [numpy_helper.from_array(w[n], name=n) for n in names if n in w]
    node_inputs = ["input", "router_logits"] + [n if n in w else "" for n in names]
    # fc3 weights/scales/bias (inputs 8-10) and zero points (11-13) stay absent.
    node_inputs += [""] * 6
    if cfg.with_mix_weights:
        node_inputs.append("mix_weights")

    node = helper.make_node(
        "QMoE",
        node_inputs,
        ["output"],
        domain="com.microsoft",
        activation_type="relu2",
        expert_weight_bits=EXPERT_WEIGHT_BITS,
        k=cfg.top_k,
        normalize_routing_weights=1,
        swiglu_fusion=0,
        block_size=BLOCK_SIZE,
    )
    return make_model_from_nodes(
        [node],
        inputs,
        [output],
        initializers=initializers,
        opset=21,
        extra_opsets=[helper.make_opsetid("com.microsoft", 1)],
    )


def _make_relu2_reference_model(cfg: Relu2Config, w: dict[str, np.ndarray]):
    """The same computation out of ops ORT CPU implements.

    Experts are the top-k logits, weighted by their softmax (or mix_weights)
    renormalised over k. Every expert runs on every token and is masked by
    whether the token selected it, which reproduces the sparse dispatch.
    """
    inputs, output = _relu2_io(cfg)
    nodes = []
    inits = []
    n_blk_hidden = cfg.hidden // BLOCK_SIZE
    n_blk_inter = cfg.inter // BLOCK_SIZE

    def const(array: np.ndarray, name: str) -> str:
        inits.append(numpy_helper.from_array(array, name=name))
        return name

    const(np.array([cfg.top_k], np.int64), "top_k")
    const(np.array([-1], np.int64), "last_axis")
    nodes += [
        helper.make_node("Cast", ["router_logits"], ["logits"], to=TensorProto.FLOAT),
        helper.make_node(
            "TopK", ["logits", "top_k"], ["topk_logits", "sel"], axis=-1, largest=1
        ),
    ]
    if cfg.with_mix_weights:
        nodes.append(
            helper.make_node("Cast", ["mix_weights"], ["source"], to=TensorProto.FLOAT)
        )
    else:
        nodes.append(helper.make_node("Softmax", ["logits"], ["source"], axis=-1))
    nodes += [
        helper.make_node("GatherElements", ["source", "sel"], ["sel_w"], axis=-1),
        helper.make_node("ReduceSum", ["sel_w", "last_axis"], ["w_sum"], keepdims=1),
        helper.make_node("Div", ["sel_w", "w_sum"], ["w_norm"]),
        helper.make_node("Cast", ["w_norm"], ["slot_w"], to=TensorProto.FLOAT16),
    ]

    def matmul_nbits(out: str, inp: str, prefix: str, k: int, n: int) -> None:
        nodes.append(
            helper.make_node(
                "MatMulNBits",
                [inp, f"{prefix}_w", f"{prefix}_s"],
                [out],
                domain="com.microsoft",
                K=k,
                N=n,
                bits=EXPERT_WEIGHT_BITS,
                block_size=BLOCK_SIZE,
            )
        )

    acc = None
    for e in range(cfg.num_experts):
        const(
            w["fc1_qweight"][e].reshape(cfg.inter, n_blk_hidden, BLOCK_SIZE // 2),
            f"e{e}_fc1_w",
        )
        const(w["fc1_scales"][e].reshape(-1), f"e{e}_fc1_s")
        const(
            w["fc2_qweight"][e].reshape(cfg.hidden, n_blk_inter, BLOCK_SIZE // 2),
            f"e{e}_fc2_w",
        )
        const(w["fc2_scales"][e].reshape(-1), f"e{e}_fc2_s")

        matmul_nbits(f"e{e}_t", "input", f"e{e}_fc1", cfg.hidden, cfg.inter)
        fc1_out = f"e{e}_t"
        if cfg.with_bias:
            const(w["fc1_bias"][e], f"e{e}_fc1_b")
            nodes.append(
                helper.make_node("Add", [fc1_out, f"e{e}_fc1_b"], [f"e{e}_tb"])
            )
            fc1_out = f"e{e}_tb"
        nodes += [
            helper.make_node("Relu", [fc1_out], [f"e{e}_relu"]),
            helper.make_node("Mul", [f"e{e}_relu", f"e{e}_relu"], [f"e{e}_a"]),
        ]
        matmul_nbits(f"e{e}_u", f"e{e}_a", f"e{e}_fc2", cfg.inter, cfg.hidden)
        fc2_out = f"e{e}_u"
        if cfg.with_bias:
            const(w["fc2_bias"][e], f"e{e}_fc2_b")
            nodes.append(
                helper.make_node("Add", [fc2_out, f"e{e}_fc2_b"], [f"e{e}_ub"])
            )
            fc2_out = f"e{e}_ub"

        const(np.array(e, np.int64), f"e{e}_id")
        nodes += [
            helper.make_node("Equal", ["sel", f"e{e}_id"], [f"e{e}_hit"]),
            helper.make_node(
                "Cast", [f"e{e}_hit"], [f"e{e}_mask"], to=TensorProto.FLOAT16
            ),
            helper.make_node("Mul", ["slot_w", f"e{e}_mask"], [f"e{e}_masked"]),
            helper.make_node(
                "ReduceSum", [f"e{e}_masked", "last_axis"], [f"e{e}_w"], keepdims=1
            ),
            helper.make_node("Mul", [fc2_out, f"e{e}_w"], [f"e{e}_out"]),
        ]
        if acc is None:
            acc = f"e{e}_out"
        else:
            nodes.append(helper.make_node("Add", [acc, f"e{e}_out"], [f"acc{e}"]))
            acc = f"acc{e}"

    nodes.append(helper.make_node("Identity", [acc], ["output"]))
    return make_model_from_nodes(
        nodes,
        inputs,
        [output],
        initializers=inits,
        opset=21,
        extra_opsets=[helper.make_opsetid("com.microsoft", 1)],
    )


def _run_relu2(model_runner, cfg: Relu2Config, seed: int = 99):
    weights = _make_relu2_weights(cfg)
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, [1, cfg.seq_len, cfg.hidden]).astype(np.float16)
    router = (rng.standard_normal([cfg.seq_len, cfg.num_experts]) * 1.2).astype(
        np.float16
    )
    # A tie at the k-th logit makes the expert choice backend-defined.
    ranked = -np.sort(-router.astype(np.float32), axis=1)
    assert not (ranked[:, cfg.top_k - 1] == ranked[:, cfg.top_k]).any()
    inputs = [x, router]
    if cfg.with_mix_weights:
        inputs.append(
            rng.uniform(0.05, 0.45, [cfg.seq_len, cfg.num_experts]).astype(np.float16)
        )
    return model_runner.run_sample(
        _make_relu2_qmoe_model(cfg, weights),
        inputs,
        reference_model=_make_relu2_reference_model(cfg, weights),
    )


class TestQMoERelu2:
    """seq_len 1 without bias takes the fused relu2 decode; with bias, and
    every longer sequence, take the bucketed prefill path."""

    @pytest.mark.parametrize("with_mix_weights", [False, True])
    @pytest.mark.parametrize("with_bias", [False, True])
    @pytest.mark.parametrize("seq_len", [1, 37])
    def test_qmoe_relu2(self, model_runner, seq_len, with_bias, with_mix_weights):
        cfg = Relu2Config(
            seq_len=seq_len,
            hidden=64,
            inter=128,
            num_experts=8,
            top_k=2,
            with_bias=with_bias,
            with_mix_weights=with_mix_weights,
        )
        actual, expected = _run_relu2(model_runner, cfg)
        compare_outputs(actual, expected, atol=5e-2, rtol=1e-2, cos_threshold=0.999)

    @pytest.mark.parametrize("seq_len", [1, 128])
    def test_qmoe_relu2_large_shape(self, model_runner, seq_len):
        """Production-sized expert shapes; 16 experts keep the reference small."""
        cfg = Relu2Config(
            seq_len=seq_len,
            hidden=2688,
            inter=1856,
            num_experts=16,
            top_k=6,
            with_mix_weights=True,
        )
        actual, expected = _run_relu2(model_runner, cfg)
        compare_outputs(actual, expected, atol=1e-1, rtol=1e-2, cos_threshold=0.99)


def _make_sigmoid_bias_qmoe_model(
    hidden: int,
    inter: int,
    num_experts: int,
    top_k: int,
    block_size: int,
    scale: float,
    seed: int = 7,
):
    """QMoE fed by the sigmoid + correction-bias router that exporters emit
    outside the op: TopK over sigmoid(logits) + bias picks the experts, the
    mixing weights are the picked sigmoid scores renormalised and scaled, and
    two ScatterElements spread both into the dense router_probs (-inf except
    1.0 at the picks) and router_weights inputs. The compiler folds the chain
    into the op, so the reference is the unfused graph on ORT CPU. Symmetric
    int4 experts without bias or zero points, swiglu with default attributes.
    """
    rng = np.random.default_rng(seed)
    e = num_experts
    inits = [
        numpy_helper.from_array(
            rng.uniform(-0.2, 0.2, [e]).astype(np.float32), "correction_bias"
        ),
        numpy_helper.from_array(np.array([top_k], np.int64), "k"),
        numpy_helper.from_array(np.array([-1], np.int64), "axes"),
        numpy_helper.from_array(np.array(scale, np.float32), "scale"),
        numpy_helper.from_array(np.array(-np.inf, np.float16), "neg_inf"),
        numpy_helper.from_array(np.array(1.0, np.float16), "one"),
        numpy_helper.from_array(np.array(0.0, np.float16), "zero"),
        numpy_helper.from_array(np.array([0], np.int64), "s0"),
        numpy_helper.from_array(np.array([-1], np.int64), "s1"),
        numpy_helper.from_array(np.array([e], np.int64), "num_e"),
        numpy_helper.from_array(np.array([-1, e], np.int64), "flat"),
    ]
    for name, n, k in (("fc1", 2 * inter, hidden), ("fc2", hidden, inter)):
        inits.append(
            numpy_helper.from_array(
                rng.integers(0, 256, [e, n, k // 2], dtype=np.uint8), f"{name}_w"
            )
        )
        inits.append(
            numpy_helper.from_array(
                rng.uniform(0.002, 0.02, [e, n, k // block_size]).astype(np.float16),
                f"{name}_s",
            )
        )

    nodes = [
        helper.make_node("Sigmoid", ["logits"], ["sig"]),
        helper.make_node("Add", ["sig", "correction_bias"], ["sel"]),
        helper.make_node(
            "TopK", ["sel", "k"], ["top_v", "idx"], axis=-1, largest=1, sorted=1
        ),
        helper.make_node("GatherElements", ["sig", "idx"], ["g"], axis=-1),
        helper.make_node("ReduceSum", ["g", "axes"], ["g_sum"], keepdims=1),
        helper.make_node("Div", ["g", "g_sum"], ["g_norm"]),
        helper.make_node("Mul", ["g_norm", "scale"], ["g_scaled"]),
        helper.make_node("Cast", ["g_scaled"], ["w"], to=TensorProto.FLOAT16),
        helper.make_node("Shape", ["idx"], ["idx_shape"]),
        helper.make_node("Slice", ["idx_shape", "s0", "s1"], ["lead"]),
        helper.make_node("Concat", ["lead", "num_e"], ["dense_shape"], axis=0),
        helper.make_node("Expand", ["neg_inf", "dense_shape"], ["probs_fill"]),
        helper.make_node("Expand", ["one", "idx_shape"], ["probs_upd"]),
        helper.make_node(
            "ScatterElements", ["probs_fill", "idx", "probs_upd"], ["probs"], axis=-1
        ),
        helper.make_node("Expand", ["zero", "dense_shape"], ["wts_fill"]),
        helper.make_node("ScatterElements", ["wts_fill", "idx", "w"], ["wts"], axis=-1),
        helper.make_node("Reshape", ["probs", "flat"], ["probs_2d"]),
        helper.make_node("Reshape", ["wts", "flat"], ["wts_2d"]),
        helper.make_node(
            "QMoE",
            ["input", "probs_2d", "fc1_w", "fc1_s", "", "fc2_w", "fc2_s"]
            + [""] * 7
            + ["wts_2d"],
            ["output"],
            domain="com.microsoft",
            activation_type="swiglu",
            expert_weight_bits=4,
            k=top_k,
            normalize_routing_weights=0,
            swiglu_fusion=1,
            block_size=block_size,
        ),
    ]
    # A symbolic token count, as in the exports: with static shapes ORT folds
    # the Shape/Expand fills into initializers before the EP sees the graph.
    inputs = [
        helper.make_tensor_value_info("input", TensorProto.FLOAT16, [1, "T", hidden]),
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "T", e]),
    ]
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT16, [1, "T", hidden]
    )
    return make_model_from_nodes(
        nodes,
        inputs,
        [output],
        initializers=inits,
        opset=21,
        extra_opsets=[helper.make_opsetid("com.microsoft", 1)],
    )


class TestQMoESigmoidBiasRouting:
    """256 experts, top-10, block 128: the routing geometry of the sigmoid +
    correction-bias MoE exports, at a reduced hidden size so the ORT CPU
    reference stays small. seq_len 1 and 5 take the fused decode kernels, 9
    and up the grouped prefill path; every token count goes through the
    folded routing kernel."""

    @pytest.mark.parametrize("seq_len", [1, 5, 9, 128, 512])
    def test_qmoe_sigmoid_bias_routing(self, model_runner, seq_len):
        e, k = 256, 10
        model = _make_sigmoid_bias_qmoe_model(
            hidden=512, inter=256, num_experts=e, top_k=k, block_size=128, scale=2.5
        )
        rng = np.random.default_rng(seq_len)
        x = rng.uniform(-1, 1, [1, seq_len, 512]).astype(np.float16)
        logits = (rng.standard_normal([1, seq_len, e]) * 2.0).astype(np.float32)
        # A tie at the k-th selection score makes the expert choice
        # backend-defined.
        bias = numpy_helper.to_array(
            next(i for i in model.graph.initializer if i.name == "correction_bias")
        )
        sel = 1.0 / (1.0 + np.exp(-logits[0])) + bias
        ranked = -np.sort(-sel, axis=1)
        assert not (ranked[:, k - 1] == ranked[:, k]).any()

        actual, expected = model_runner.run_sample(model, [x, logits])
        if seq_len <= FUSED_MAX_TOKENS:
            # W4A8 decode (HIPDNN_EP_MATMUL_DP4A, default on): ten int8-quantized
            # expert paths per token measure cosine ~0.99988 here, and 0.9999
            # or better with HIPDNN_EP_MATMUL_DP4A=0.
            scale = float(np.abs(expected[0].astype(np.float32)).max())
            compare_outputs(
                actual, expected, atol=2e-2 * scale, rtol=1e-2, cos_threshold=0.9995
            )
        else:
            _compare_gpt_oss_shape(actual, expected, seq_len)
