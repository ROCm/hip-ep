#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the QMoE custom op (com.amd).

A latent-space Mixture-of-Experts block: sigmoid + correction-bias
routing, a compressed latent space the experts work in, squared-ReLU
("relu2") activation, and a shared expert every token passes through.
Distinct from com.microsoft::QMoE (test_qmoe.py), which routes with
softmax, activates with SwiGLU, and has neither latent projection nor
shared expert.

ORT CPU has no kernel for this op -- it only exists as a GPU kernel in
this repo, and the CPU side is a stub that raises ORT_NOT_IMPLEMENTED.
So the reference is a second graph, built here from the same weights
out of ops the CPU does implement (com.microsoft::MatMulNBits for the
quantized projections, standard ONNX for everything else), and handed
to the runner via ``reference_model=``. That graph transcribes the
expansion an exporter emits for this block -- the subgraph the op is
there to replace -- rather than inventing an equivalent formulation, so
its agreement is evidence about the op and not about the transcription.
The op's math is specified on Hip_QMoEAmdOp in
include/hip/Dialect/IR/HipOps.td.

The parametrization straddles the runtime's dispatch boundaries so the
suite covers every path rather than whichever one a single shape
happens to select:

  seq_len      1 takes the fused single-token decode path; >1 takes grouped
               WMMA prefill on wave32 and fixed-sequence routing-slot dispatch
               on wave64.
  num_experts  at seq_len 1, a multiple of the routing kernel's
               expert-tile width takes the parallel routing fast path
               and anything else takes the generic one-block-per-token
               router. That fast path is decode-only, so longer
               sequences route generically whatever the expert count.

Shapes stay small (hidden=64) but keep latent and the MoE intermediate a
multiple of block_size, as required by both grouped prefill paths.
"""

from dataclasses import dataclass

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

# Added to the top-k weight sum before normalising, matching the op so
# the reference divides by the same denominator.
ROUTING_EPS = 1e-20

BLOCK_SIZE = 32
EXPERT_WEIGHT_BITS = 4
ROUTED_SCALING_FACTOR = 2.0

# One uniform range for every quantized projection. Picked so the output
# lands around unit magnitude after four chained matmuls and two relu2
# squarings -- small scales here drive the result to ~1e-3, where an
# absolute tolerance stops discriminating.
SCALE_RANGE = (0.01, 0.08)

# Names in schema order. The reference graph reuses them for the tensors
# it can consume unchanged.
_INPUT_NAMES = [
    "fc1_experts_weights",
    "fc1_experts_scales",
    "fc2_experts_weights",
    "fc2_experts_scales",
    "fc1_latent_weights",
    "fc1_latent_scales",
    "fc2_latent_weights",
    "fc2_latent_scales",
    "shared_fc1_weights",
    "shared_fc1_scales",
    "shared_fc2_weights",
    "shared_fc2_scales",
    "router_weight",
    "correction_bias",
]


@dataclass(frozen=True)
class Config:
    batch: int
    seq_len: int
    hidden: int
    latent: int
    moe_intermediate: int
    shared_intermediate: int
    num_experts: int
    top_k: int
    block_size: int = BLOCK_SIZE


def _quantize(rng, n: int, k: int, block_size: int, leading: tuple[int, ...] = ()):
    """Random 4-bit weights in MatMulNBits layout for an [K] -> [N] projection.

    Weight shape is ``[*leading, N, ceil(K / block_size), block_size // 2]``
    (two nibbles per byte) and scales are ``[*leading, N, ceil(K / block_size)]``.
    ``leading`` carries the expert axis for the per-expert projections;
    slicing it off yields exactly the layout a single MatMulNBits node
    wants, which is what lets the reference graph reuse these bytes.

    There is no zero-point tensor: this op is always symmetric around 8,
    which is also MatMulNBits' default.
    """
    n_blocks = (k + block_size - 1) // block_size
    qweight = rng.integers(
        0,
        256,
        [*leading, n, n_blocks, block_size // 2],
        dtype=np.uint8,
    )
    scales = rng.uniform(
        SCALE_RANGE[0],
        SCALE_RANGE[1],
        [*leading, n, n_blocks],
    ).astype(np.float16)
    return qweight, scales


def _make_weights(cfg: Config, seed: int = 42) -> dict[str, np.ndarray]:
    """Build one set of weights, shared by the op graph and the reference."""
    rng = np.random.default_rng(seed)
    experts = (cfg.num_experts,)
    w: dict[str, np.ndarray] = {}

    # Experts read and write the latent space: hidden never reaches them.
    w["fc1_experts_weights"], w["fc1_experts_scales"] = _quantize(
        rng, cfg.moe_intermediate, cfg.latent, cfg.block_size, experts
    )
    w["fc2_experts_weights"], w["fc2_experts_scales"] = _quantize(
        rng, cfg.latent, cfg.moe_intermediate, cfg.block_size, experts
    )
    w["fc1_latent_weights"], w["fc1_latent_scales"] = _quantize(
        rng, cfg.latent, cfg.hidden, cfg.block_size
    )
    w["fc2_latent_weights"], w["fc2_latent_scales"] = _quantize(
        rng, cfg.hidden, cfg.latent, cfg.block_size
    )
    w["shared_fc1_weights"], w["shared_fc1_scales"] = _quantize(
        rng, cfg.shared_intermediate, cfg.hidden, cfg.block_size
    )
    w["shared_fc2_weights"], w["shared_fc2_scales"] = _quantize(
        rng, cfg.hidden, cfg.shared_intermediate, cfg.block_size
    )

    # The router is dense fp16, not quantized.
    w["router_weight"] = rng.standard_normal([cfg.hidden, cfg.num_experts]).astype(
        np.float16
    )
    w["correction_bias"] = (
        rng.standard_normal([cfg.num_experts]).astype(np.float16) * 0.1
    )
    return w


def _io(cfg: Config):
    x = helper.make_tensor_value_info(
        "hidden_states",
        TensorProto.FLOAT16,
        [cfg.batch, cfg.seq_len, cfg.hidden],
    )
    y = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT16,
        [cfg.batch, cfg.seq_len, cfg.hidden],
    )
    return x, y


def _make_qmoe_amd_model(cfg: Config, w: dict[str, np.ndarray]):
    """A single com.amd::QMoE node -- the graph under test."""
    x, y = _io(cfg)
    node = helper.make_node(
        "QMoE",
        ["hidden_states", *_INPUT_NAMES],
        ["output"],
        domain="com.amd",
        k=cfg.top_k,
        expert_weight_bits=EXPERT_WEIGHT_BITS,
        block_size=cfg.block_size,
        normalize_routing_weights=1,
        use_correction_bias=1,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
        activation_type="relu2",
        routing_type="sigmoid",
    )
    return make_model_from_nodes(
        [node],
        [x],
        [y],
        initializers=[numpy_helper.from_array(w[n], name=n) for n in _INPUT_NAMES],
        opset=21,
        extra_opsets=[helper.make_opsetid("com.amd", 1)],
    )


def _make_reference_model(cfg: Config, w: dict[str, np.ndarray]):
    """The same computation out of ops ORT CPU implements.

    A transcription of how the exporter writes this block before the op
    replaces it: routing in fp32 down to the scaled weight, everything
    from there on in fp16, and the per-token weight for an expert
    recovered by masking the top-k slots against that expert's index
    rather than by gathering. Keeping the expansion literal is what
    makes it a trustworthy reference -- an idiom of our own would need
    its own equivalence argument.

    Takes the identical graph input, so the runner can feed one set of
    tensors to both graphs. Per-expert weights are sliced with numpy at
    build time and emitted as one initializer pair per expert, which
    keeps Slice out of the graph and hands each MatMulNBits exactly the
    layout it expects.
    """
    x, y = _io(cfg)
    nodes = []
    inits = []

    def const(array: np.ndarray, name: str) -> str:
        inits.append(numpy_helper.from_array(array, name=name))
        return name

    # --- Routing -----------------------------------------------------
    # Router and bias are stored fp16 and widened in the graph, so the
    # reference consumes the same bytes the op does.
    const(w["router_weight"], "router_f16")
    const(w["correction_bias"], "corr_f16")
    const(np.array([cfg.top_k], np.int64), "top_k")
    const(np.array([-1], np.int64), "last_axis")
    const(np.array(ROUTING_EPS, np.float32), "routing_eps")
    const(np.array(ROUTED_SCALING_FACTOR, np.float32), "routed_scale")

    nodes += [
        helper.make_node("Cast", ["hidden_states"], ["x_f32"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["router_f16"], ["router_f32"], to=TensorProto.FLOAT),
        helper.make_node("MatMul", ["x_f32", "router_f32"], ["logits"]),
        helper.make_node("Sigmoid", ["logits"], ["probs"]),
        # Selection reads the biased score; the weight is the raw prob.
        helper.make_node("Cast", ["corr_f16"], ["corr_f32"], to=TensorProto.FLOAT),
        helper.make_node("Add", ["probs", "corr_f32"], ["biased"]),
        helper.make_node(
            "TopK", ["biased", "top_k"], ["biased_topk", "sel"], axis=-1, largest=1
        ),
        helper.make_node("GatherElements", ["probs", "sel"], ["sel_w"], axis=-1),
        helper.make_node("ReduceSum", ["sel_w", "last_axis"], ["w_sum"], keepdims=1),
        helper.make_node("Add", ["w_sum", "routing_eps"], ["w_denom"]),
        helper.make_node("Div", ["sel_w", "w_denom"], ["w_norm"]),
        helper.make_node("Mul", ["w_norm", "routed_scale"], ["w_scaled"]),
        # Routing narrows to fp16 here and stays there for the rest of
        # the block, accumulation across experts included.
        helper.make_node("Cast", ["w_scaled"], ["slot_w"], to=TensorProto.FLOAT16),
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
                block_size=cfg.block_size,
            )
        )

    def relu2(out: str, inp: str) -> None:
        nodes.append(helper.make_node("Relu", [inp], [f"{out}_relu"]))
        nodes.append(helper.make_node("Mul", [f"{out}_relu", f"{out}_relu"], [out]))

    # --- Latent projection down, experts, projection back up ----------
    const(w["fc1_latent_weights"], "lat_down_w")
    const(w["fc1_latent_scales"], "lat_down_s")
    matmul_nbits("h", "hidden_states", "lat_down", cfg.hidden, cfg.latent)

    acc = None
    for e in range(cfg.num_experts):
        const(w["fc1_experts_weights"][e], f"e{e}_fc1_w")
        const(w["fc1_experts_scales"][e], f"e{e}_fc1_s")
        const(w["fc2_experts_weights"][e], f"e{e}_fc2_w")
        const(w["fc2_experts_scales"][e], f"e{e}_fc2_s")

        matmul_nbits(f"e{e}_t", "h", f"e{e}_fc1", cfg.latent, cfg.moe_intermediate)
        relu2(f"e{e}_a", f"e{e}_t")
        matmul_nbits(
            f"e{e}_u", f"e{e}_a", f"e{e}_fc2", cfg.moe_intermediate, cfg.latent
        )

        # Every expert runs on every token. Masking the top-k slots
        # against this expert's index and summing leaves its weight for
        # tokens that selected it and zero for the rest, so the sum over
        # experts reproduces the op's sparse gather.
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
            helper.make_node("Mul", [f"e{e}_u", f"e{e}_w"], [f"e{e}_out"]),
        ]
        if acc is None:
            acc = f"e{e}_out"
        else:
            nodes.append(helper.make_node("Add", [acc, f"e{e}_out"], [f"acc{e}"]))
            acc = f"acc{e}"

    const(w["fc2_latent_weights"], "lat_up_w")
    const(w["fc2_latent_scales"], "lat_up_s")
    matmul_nbits("routed_out", acc, "lat_up", cfg.latent, cfg.hidden)

    # --- Shared expert, taken by every token regardless of routing ----
    const(w["shared_fc1_weights"], "shared_fc1_w")
    const(w["shared_fc1_scales"], "shared_fc1_s")
    const(w["shared_fc2_weights"], "shared_fc2_w")
    const(w["shared_fc2_scales"], "shared_fc2_s")
    matmul_nbits(
        "shared_t", "hidden_states", "shared_fc1", cfg.hidden, cfg.shared_intermediate
    )
    relu2("shared_a", "shared_t")
    matmul_nbits(
        "shared_out", "shared_a", "shared_fc2", cfg.shared_intermediate, cfg.hidden
    )

    nodes.append(helper.make_node("Add", ["routed_out", "shared_out"], ["output"]))

    return make_model_from_nodes(
        nodes,
        [x],
        [y],
        initializers=inits,
        opset=21,
        extra_opsets=[helper.make_opsetid("com.microsoft", 1)],
    )


class TestQMoEAmd:
    @pytest.mark.parametrize("num_experts", [8, 6])
    @pytest.mark.parametrize("seq_len", [1, 128])
    def test_qmoe_amd(self, model_runner, seq_len, num_experts):
        """com.amd QMoE against an equivalent graph of CPU-implemented ops.

        Four chained quantized matmuls plus two squaring activations per
        token, so the error accumulates well past a single MatMulNBits.
        Both sides dequantize the same stored nibbles with the same
        scales, though, so what is left is accumulation order and fp16
        rounding rather than quantization noise.
        """
        cfg = Config(
            batch=1,
            seq_len=seq_len,
            hidden=64,
            latent=64,
            moe_intermediate=64,
            shared_intermediate=64,
            num_experts=num_experts,
            top_k=2,
        )
        weights = _make_weights(cfg)

        rng = np.random.default_rng(99)
        x = rng.uniform(-0.5, 0.5, [cfg.batch, seq_len, cfg.hidden]).astype(np.float16)

        actual, expected = model_runner.run_sample(
            _make_qmoe_amd_model(cfg, weights),
            [x],
            reference_model=_make_reference_model(cfg, weights),
        )
        compare_outputs(actual, expected, atol=1e-1, rtol=1e-2, cos_threshold=0.999)
