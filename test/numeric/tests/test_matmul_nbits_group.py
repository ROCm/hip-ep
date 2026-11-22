#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""MatMulNBits projections the fusion transform folds into hip.matmul_nbits_group.

Shapes come from the Llama-2-13B and Phi-3.5-mini int4 decoders (separate
q/k/v and gate/up MatMulNBits nodes reading one normalized activation, the MLP
combined as Mul(Mul(gate, Sigmoid(gate)), up)), Llama-3.1-8B AWQ (asymmetric
weights: packed uint8 zero points, block_size 128) and Qwen3.8-Flash-Next
(four projections of one row, outputs split along N, hyper-connection ops
producing the activation).

A group whose activation comes from SkipSimplifiedLayerNormalization,
HyperConnectionPreMix or HyperConnectionPostMix + BranchwiseRMSNorm reads the
int8 copy of the row that the producer's kernel writes. That copy has to be
bit-identical to the group quantizing the row itself, which
TestGroupQuantCopyExact checks for every producer path, fallbacks included.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

BITS = 4
BLOCK_SIZE = 32
LLAMA2_HIDDEN = 5120
LLAMA2_INTERMEDIATE = 13824
PHI35_HIDDEN = 3072
PHI35_INTERMEDIATE = 8192
LLAMA31_HIDDEN = 4096
LLAMA31_KV = 1024
LLAMA31_INTERMEDIATE = 14336
AWQ_BLOCK_SIZE = 128
FLASH_NEXT_BRANCHES = 4
FLASH_NEXT_HIDDEN = 2560


def _q4_initializers(
    rng,
    prefix: str,
    K: int,
    N: int,
    scale_range=(-0.02, 0.02),
    block_size: int = BLOCK_SIZE,
    asym: bool = False,
):
    n_blocks = K // block_size
    q_weight = rng.integers(0, 256, [N, n_blocks, block_size // 2], dtype=np.uint8)
    scales = rng.uniform(scale_range[0], scale_range[1], [N, n_blocks]).astype(
        np.float16
    )
    inits = [
        numpy_helper.from_array(q_weight, name=f"{prefix}_Q4"),
        numpy_helper.from_array(scales, name=f"{prefix}_scales"),
    ]
    if asym:
        # Two 4-bit zero points per byte, each row padded to whole bytes.
        zp = rng.integers(0, 256, [N * ((n_blocks + 1) // 2)], dtype=np.uint8)
        inits.append(numpy_helper.from_array(zp, name=f"{prefix}_zp"))
    return inits


def _q4_node(
    prefix: str,
    x: str,
    y: str,
    K: int,
    N: int,
    block_size: int = BLOCK_SIZE,
    asym: bool = False,
):
    inputs = [x, f"{prefix}_Q4", f"{prefix}_scales"]
    if asym:
        inputs.append(f"{prefix}_zp")
    return helper.make_node(
        "MatMulNBits",
        inputs,
        [y],
        domain="com.microsoft",
        K=K,
        N=N,
        bits=BITS,
        block_size=block_size,
        accuracy_level=4,
    )


def _model(nodes, inputs, outputs, inits):
    return make_model_from_nodes(
        nodes,
        inputs,
        outputs,
        initializers=inits,
        opset=21,
        extra_opsets=[helper.make_opsetid("com.microsoft", 1)],
    )


def _f16_info(name: str, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT16, shape)


def _make_qkv_model(
    seq_len: int,
    K: int,
    n_q: int,
    n_kv: int,
    seed: int = 7,
    block_size: int = BLOCK_SIZE,
    asym: bool = False,
):
    rng = np.random.default_rng(seed)
    outs, nodes, inits = [], [], []
    for name, n in (("q", n_q), ("k", n_kv), ("v", n_kv)):
        inits += _q4_initializers(rng, name, K, n, block_size=block_size, asym=asym)
        nodes.append(_q4_node(name, "X", name.upper(), K, n, block_size, asym))
        outs.append(_f16_info(name.upper(), [1, seq_len, n]))
    return _model(nodes, [_f16_info("X", [1, seq_len, K])], outs, inits)


def _make_gate_up_silu_model(
    seq_len: int,
    K: int,
    N: int,
    seed: int = 11,
    block_size: int = BLOCK_SIZE,
    asym: bool = False,
    split_input: bool = False,
):
    """split_input feeds up from a second input X2, so the projections share
    no activation and stay ungrouped."""
    rng = np.random.default_rng(seed)
    q4 = dict(block_size=block_size, asym=asym)
    inits = _q4_initializers(rng, "gate", K, N, **q4) + _q4_initializers(
        rng, "up", K, N, **q4
    )
    up_in = "X2" if split_input else "X"
    nodes = [
        _q4_node("gate", "X", "G", K, N, **q4),
        _q4_node("up", up_in, "U", K, N, **q4),
        helper.make_node("Sigmoid", ["G"], ["S"]),
        helper.make_node("Mul", ["G", "S"], ["GS"]),
        helper.make_node("Mul", ["GS", "U"], ["Y"]),
    ]
    inputs = [_f16_info("X", [1, seq_len, K])]
    if split_input:
        inputs.append(_f16_info("X2", [1, seq_len, K]))
    return _model(nodes, inputs, [_f16_info("Y", [1, seq_len, N])], inits)


class TestMatMulNBitsGroup:
    """seq_len 1 takes the grouped dp4a kernel; 7 takes the per-projection
    fallback (and, for gate/up, the separate SiLU*mul kernel)."""

    @pytest.mark.parametrize("seq_len", [1, 7])
    @pytest.mark.parametrize(
        "K,n_q,n_kv",
        [
            (LLAMA2_HIDDEN, LLAMA2_HIDDEN, LLAMA2_HIDDEN),
            (PHI35_HIDDEN, PHI35_HIDDEN, PHI35_HIDDEN),
            (512, 512, 128),
        ],
        ids=["llama2", "phi35", "gqa"],
    )
    def test_qkv(self, model_runner, seq_len, K, n_q, n_kv):
        model = _make_qkv_model(seq_len, K, n_q, n_kv)
        rng = np.random.default_rng(99)
        x = rng.uniform(-1, 1, [1, seq_len, K]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-2, rtol=1e-2, cos_threshold=0.999)

    @pytest.mark.parametrize("seq_len", [1, 7])
    @pytest.mark.parametrize(
        "K,N",
        [(LLAMA2_HIDDEN, LLAMA2_INTERMEDIATE), (PHI35_HIDDEN, PHI35_INTERMEDIATE)],
        ids=["llama2", "phi35"],
    )
    def test_gate_up_silu(self, model_runner, seq_len, K, N):
        model = _make_gate_up_silu_model(seq_len, K, N)
        rng = np.random.default_rng(99)
        x = rng.uniform(-1, 1, [1, seq_len, K]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-2, rtol=1e-2, cos_threshold=0.999)

    @pytest.mark.parametrize("seq_len", [7, 128])
    @pytest.mark.parametrize("asym", [False, True], ids=["sym", "asym"])
    def test_gate_up_silu_matches_ungrouped(self, model_runner, seq_len, asym):
        """Above one row the group runs each projection as the ungrouped graph
        does, so its SiLU * mul has to round as the graph does too: the output
        is bit-identical to the same weights with the activation split in two."""
        K, N = PHI35_HIDDEN, PHI35_INTERMEDIATE
        rng = np.random.default_rng(99)
        x = rng.uniform(-1, 1, [1, seq_len, K]).astype(np.float16)
        grouped, _ = model_runner.run_sample(
            _make_gate_up_silu_model(seq_len, K, N, asym=asym), [x]
        )
        ungrouped, _ = model_runner.run_sample(
            _make_gate_up_silu_model(seq_len, K, N, asym=asym, split_input=True),
            [x, x.copy()],
        )
        np.testing.assert_array_equal(grouped[0], ungrouped[0])

    @pytest.mark.parametrize("seq_len", [1, 7])
    @pytest.mark.parametrize(
        "K,n_q,n_kv,block_size",
        [
            (LLAMA31_HIDDEN, LLAMA31_HIDDEN, LLAMA31_KV, AWQ_BLOCK_SIZE),
            (512, 512, 128, BLOCK_SIZE),
        ],
        ids=["llama31_awq", "gqa_bs32"],
    )
    def test_qkv_zero_points(self, model_runner, seq_len, K, n_q, n_kv, block_size):
        model = _make_qkv_model(seq_len, K, n_q, n_kv, block_size=block_size, asym=True)
        rng = np.random.default_rng(99)
        x = rng.uniform(-1, 1, [1, seq_len, K]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-2, rtol=1e-2, cos_threshold=0.999)

    @pytest.mark.parametrize("seq_len", [1, 7])
    @pytest.mark.parametrize(
        "K,N,block_size",
        [
            (LLAMA31_HIDDEN, LLAMA31_INTERMEDIATE, AWQ_BLOCK_SIZE),
            (512, 1024, BLOCK_SIZE),
        ],
        ids=["llama31_awq", "bs32"],
    )
    def test_gate_up_silu_zero_points(self, model_runner, seq_len, K, N, block_size):
        model = _make_gate_up_silu_model(
            seq_len, K, N, block_size=block_size, asym=True
        )
        rng = np.random.default_rng(99)
        x = rng.uniform(-1, 1, [1, seq_len, K]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-2, rtol=1e-2, cos_threshold=0.999)

    @pytest.mark.parametrize(
        "seq_len,ns",
        [
            pytest.param(1, (1, 512, 640, 640), id="four-1"),
            pytest.param(
                7,
                (1, 512, 640, 640),
                id="four-7",
                # Strict, so it fails once the dp4a small-M fix (#1606) lands
                # and this mark can go.
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="hip.matmul_nbits itself is wrong for N=1 at M=5..7, "
                    "grouped or not; fixed by #1606",
                ),
            ),
            pytest.param(1, (640, 6144, 7168), id="three-1"),
            pytest.param(7, (640, 6144, 7168), id="three-7"),
            pytest.param(1, (96, 16384), id="two-1"),
            pytest.param(7, (96, 16384), id="two-7"),
        ],
    )
    def test_shared_input_flash_next(self, model_runner, seq_len, ns):
        """Qwen3.8-Flash-Next's projections sharing one K=2560 activation,
        including a four-member group and an N=1 member."""
        K = 2560
        rng = np.random.default_rng(5)
        nodes, inits, outs = [], [], []
        for i, n in enumerate(ns):
            inits += _q4_initializers(rng, f"p{i}", K, n)
            nodes.append(_q4_node(f"p{i}", "X", f"Y{i}", K, n))
            outs.append(_f16_info(f"Y{i}", [1, seq_len, n]))
        model = _model(nodes, [_f16_info("X", [1, seq_len, K])], outs, inits)
        x = rng.uniform(-1, 1, [1, seq_len, K]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-2, rtol=1e-2, cos_threshold=0.999)

    @pytest.mark.parametrize("seq_len", [1, 7])
    @pytest.mark.parametrize(
        "K,splits,asym",
        [
            (10240, [(320, 4)], False),
            (2560, [(10240, 6144), (48, 48)], False),
            (480, [(40, 24)], True),
            (96, [(5, 7)], False),
            (96, [(5, 7)], True),
        ],
        ids=[
            "flash_next_hc_gate",
            "flash_next_in_proj",
            "zp_odd_blocks",
            "odd_scale_offset",
            "odd_scale_offset_zp",
        ],
    )
    def test_split_output(self, model_runner, seq_len, K, splits, asym):
        """Projections whose outputs are only Split along N become one group
        member per piece, each reading its rows of the shared weight. K=480 has
        15 blocks, so each packed zero-point row carries a padding nibble. K=96
        has 3 blocks, so the second piece of a (5, 7) split starts 15 fp16
        scales in, an odd element offset."""
        rng = np.random.default_rng(17)
        nodes, inits, outs = [], [], []
        for i, pieces in enumerate(splits):
            n = sum(pieces)
            inits += _q4_initializers(rng, f"p{i}", K, n, asym=asym)
            inits.append(
                numpy_helper.from_array(
                    np.array(pieces, dtype=np.int64), name=f"p{i}_split"
                )
            )
            nodes.append(_q4_node(f"p{i}", "X", f"Y{i}", K, n, asym=asym))
            names = [f"Y{i}_{j}" for j in range(len(pieces))]
            nodes.append(
                helper.make_node("Split", [f"Y{i}", f"p{i}_split"], names, axis=-1)
            )
            outs += [_f16_info(name, [1, seq_len, p]) for name, p in zip(names, pieces)]
        model = _model(nodes, [_f16_info("X", [1, seq_len, K])], outs, inits)
        x = rng.uniform(-1, 1, [1, seq_len, K]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-2, rtol=1e-2, cos_threshold=0.999)


def _make_norm_fed_model(seq_len: int, K: int, kind: str, n_q=0, n_kv=0, N=0):
    """SkipSimplifiedLayerNormalization feeding q/k/v (kind="qkv") or the
    SiLU(gate) * up MLP (kind="gate_up"), as in a decoder layer. The residual
    sum is a graph output so the norm keeps both of its results."""
    rng = np.random.default_rng(23)
    gamma = rng.uniform(0.5, 1.5, [K]).astype(np.float16)
    inits = [numpy_helper.from_array(gamma, name="gamma")]
    nodes = [
        helper.make_node(
            "SkipSimplifiedLayerNormalization",
            ["X", "R", "gamma"],
            ["H", "", "", "SUM"],
            domain="com.microsoft",
            epsilon=1e-5,
        )
    ]
    outs = [_f16_info("SUM", [1, seq_len, K])]
    if kind == "qkv":
        for name, n in (("q", n_q), ("k", n_kv), ("v", n_kv)):
            inits += _q4_initializers(rng, name, K, n)
            nodes.append(_q4_node(name, "H", name.upper(), K, n))
            outs.append(_f16_info(name.upper(), [1, seq_len, n]))
    else:
        inits += _q4_initializers(rng, "gate", K, N) + _q4_initializers(rng, "up", K, N)
        nodes += [
            _q4_node("gate", "H", "G", K, N),
            _q4_node("up", "H", "U", K, N),
            helper.make_node("Sigmoid", ["G"], ["S"]),
            helper.make_node("Mul", ["G", "S"], ["GS"]),
            helper.make_node("Mul", ["GS", "U"], ["Y"]),
        ]
        outs.append(_f16_info("Y", [1, seq_len, N]))
    inputs = [_f16_info("X", [1, seq_len, K]), _f16_info("R", [1, seq_len, K])]
    return _model(nodes, inputs, outs, inits)


class TestMatMulNBitsGroupNormQuant:
    """A group fed by SkipSimplifiedLayerNormalization reads the int8 copy of
    the row that the norm kernel writes (block_size 32).

    seq_len 1 is the only shape where the norm writes the copy and the grouped
    kernel reads it; 7 and 128 must ignore the unwritten buffers and quantize
    the activation themselves, through the per-projection fallback (dp4a at 7,
    the prefill kernels at 128, plus the in-place SiLU*mul for gate/up).

    The normalized activation drives q/k/v to |y| ~ 20 and SiLU(gate) * up to
    |y| ~ 200, so the absolute error of int8 activation quantization scales
    with them: atol is ~1% of the output range, and cosine >= 0.9995 catches a
    real regression (a wrong scale or layout drops it far below that).
    """

    @pytest.mark.parametrize("seq_len", [1, 7, 128])
    @pytest.mark.parametrize(
        "K,n_q,n_kv",
        [
            (LLAMA2_HIDDEN, LLAMA2_HIDDEN, LLAMA2_HIDDEN),
            (LLAMA31_HIDDEN, LLAMA31_HIDDEN, LLAMA31_KV),
        ],
        ids=["llama2", "llama31_gqa"],
    )
    def test_qkv(self, model_runner, seq_len, K, n_q, n_kv):
        model = _make_norm_fed_model(seq_len, K, "qkv", n_q=n_q, n_kv=n_kv)
        rng = np.random.default_rng(99)
        x = rng.uniform(-1, 1, [1, seq_len, K]).astype(np.float16)
        r = rng.uniform(-1, 1, [1, seq_len, K]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x, r])
        compare_outputs(actual, expected, atol=2e-1, rtol=1e-2, cos_threshold=0.9995)

    @pytest.mark.parametrize("seq_len", [1, 7, 128])
    @pytest.mark.parametrize(
        "K,N",
        [(LLAMA2_HIDDEN, LLAMA2_INTERMEDIATE), (PHI35_HIDDEN, PHI35_INTERMEDIATE)],
        ids=["llama2", "phi35"],
    )
    def test_gate_up_silu(self, model_runner, seq_len, K, N):
        model = _make_norm_fed_model(seq_len, K, "gate_up", N=N)
        rng = np.random.default_rng(99)
        x = rng.uniform(-1, 1, [1, seq_len, K]).astype(np.float16)
        r = rng.uniform(-1, 1, [1, seq_len, K]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x, r])
        compare_outputs(actual, expected, atol=2.0, rtol=1e-2, cos_threshold=0.9995)


def _hc_producer(kind: str, seq_len: int, C: int, H: int, rng):
    """HyperConnectionPreMix (kind="pre_mix", activation width H) or
    HyperConnectionPostMix + BranchwiseRMSNorm, fused into one post-mix norm
    (kind="post_norm", width C * H over the flattened streams), writing "H".
    Returns (nodes, inputs, initializers, extra outputs, width)."""
    hc = dict(domain="com.microsoft", num_branches=C)
    S = _f16_info("S", [1, seq_len, C * H])
    if kind == "pre_mix":
        G = _f16_info("G", [1, seq_len, C * H])
        node = helper.make_node(
            "HyperConnectionPreMix", ["S", "G"], ["H"], reduction_scale=1.0 / C, **hc
        )
        return [node], [S, G], [], [], H
    B = _f16_info("B", [1, seq_len, H])
    G = _f16_info("G", [1, seq_len, C])
    scale = rng.uniform(0.5, 1.5, [C * H]).astype(np.float16)
    nodes = [
        helper.make_node("HyperConnectionPostMix", ["S", "B", "G"], ["RES"], **hc),
        helper.make_node(
            "BranchwiseRMSNorm", ["RES", "norm_scale"], ["H"], epsilon=1e-6, **hc
        ),
    ]
    inits = [numpy_helper.from_array(scale, name="norm_scale")]
    return nodes, [S, B, G], inits, [_f16_info("RES", [1, seq_len, C * H])], C * H


def _hc_feeds(kind: str, seq_len: int, C: int, H: int, rng):
    streams = rng.uniform(-1, 1, [1, seq_len, C * H]).astype(np.float16)
    if kind == "pre_mix":
        return [streams, rng.uniform(0, 1, [1, seq_len, C * H]).astype(np.float16)]
    return [
        streams,
        rng.uniform(-1, 1, [1, seq_len, H]).astype(np.float16),
        rng.uniform(0, 1, [1, seq_len, C]).astype(np.float16),
    ]


def _make_hc_fed_model(seq_len: int, kind: str):
    """Flash-Next hyper-connection ops feeding two MatMulNBits. The post-mix
    residual is a graph output."""
    rng = np.random.default_rng(29)
    nodes, inputs, inits, outs, K = _hc_producer(
        kind, seq_len, FLASH_NEXT_BRANCHES, FLASH_NEXT_HIDDEN, rng
    )
    for name, n in (("u", 512), ("v", 256)):
        inits += _q4_initializers(rng, name, K, n)
        nodes.append(_q4_node(name, "H", name.upper(), K, n))
        outs.append(_f16_info(name.upper(), [1, seq_len, n]))
    return _model(nodes, inputs, outs, inits)


class TestMatMulNBitsGroupHyperConnectionQuant:
    """A group fed by a Flash-Next hyper-connection op reads the int8 copy the
    op's kernel writes, as with TestMatMulNBitsGroupNormQuant. seq_len 1 is the
    shape where the copy is written and read; at 7 neither side uses it."""

    @pytest.mark.parametrize("seq_len", [1, 7])
    @pytest.mark.parametrize("kind", ["pre_mix", "post_norm"])
    def test_hc_feeds_group(self, model_runner, seq_len, kind):
        model = _make_hc_fed_model(seq_len, kind)
        rng = np.random.default_rng(31)
        feeds = _hc_feeds(kind, seq_len, FLASH_NEXT_BRANCHES, FLASH_NEXT_HIDDEN, rng)

        actual, expected = model_runner.run_sample(model, feeds)
        compare_outputs(actual, expected, atol=2e-1, rtol=1e-2, cos_threshold=0.9995)


def _projections(rng, K: int):
    """Two projections of "H" (a two-member group) and their initializers."""
    inits, nodes, outs = [], [], []
    for name, n in (("u", 512), ("v", 256)):
        inits += _q4_initializers(rng, name, K, n)
        nodes.append(_q4_node(name, "H", name.upper(), K, n))
        outs.append(_f16_info(name.upper(), [1, 1, n]))
    return nodes, inits, outs


class TestGroupQuantCopyExact:
    """At one row the producer's int8 copy replaces the group's own
    quantization of the same fp16 row, so the projections must be
    bit-identical to the group fed that row as a graph input. Each case runs
    the producer with its output H also returned, then the projections alone
    on that H.
    """

    def _assert_exact(self, model_runner, nodes, inputs, inits, extra_outs, K, feeds):
        rng = np.random.default_rng(41)
        proj_nodes, proj_inits, proj_outs = _projections(rng, K)
        fed = _model(
            nodes + proj_nodes,
            inputs,
            [_f16_info("H", [1, 1, K])] + extra_outs + proj_outs,
            inits + proj_inits,
        )
        produced, _ = model_runner.run_sample(fed, feeds)
        h = produced[0]
        standalone = _model(
            proj_nodes, [_f16_info("H", [1, 1, K])], proj_outs, proj_inits
        )
        alone, _ = model_runner.run_sample(standalone, [h])
        n_extra = len(extra_outs)
        for got, want in zip(produced[1 + n_extra :], alone):
            np.testing.assert_array_equal(got, want)

    @pytest.mark.parametrize("K", [2560, LLAMA2_HIDDEN])
    @pytest.mark.parametrize("offset", [0, 1], ids=["aligned", "odd_offset"])
    def test_skip_norm(self, model_runner, K, offset):
        """offset 1 slices the norm input one fp16 element into a wider row,
        which leaves the vector path to the scalar norm plus the standalone
        quantizer when the slice reaches the kernel uncopied."""
        rng = np.random.default_rng(43)
        gamma = rng.uniform(0.5, 1.5, [K]).astype(np.float16)
        inits = [
            numpy_helper.from_array(gamma, name="gamma"),
            numpy_helper.from_array(np.array([offset], dtype=np.int64), "starts"),
            numpy_helper.from_array(np.array([offset + K], dtype=np.int64), "ends"),
            numpy_helper.from_array(np.array([2], dtype=np.int64), "axes"),
        ]
        nodes = [
            helper.make_node("Slice", ["XW", "starts", "ends", "axes"], ["X"]),
            helper.make_node(
                "SkipSimplifiedLayerNormalization",
                ["X", "R", "gamma"],
                ["H"],
                domain="com.microsoft",
                epsilon=1e-5,
            ),
        ]
        inputs = [_f16_info("XW", [1, 1, K + 1]), _f16_info("R", [1, 1, K])]
        feeds = [
            rng.uniform(-1, 1, [1, 1, K + 1]).astype(np.float16),
            rng.uniform(-1, 1, [1, 1, K]).astype(np.float16),
        ]
        self._assert_exact(model_runner, nodes, inputs, inits, [], K, feeds)

    @pytest.mark.parametrize(
        "kind,C,H",
        [
            ("pre_mix", FLASH_NEXT_BRANCHES, FLASH_NEXT_HIDDEN),
            # 1024-thread wide kernel with 3, 2 and 4 elements per thread.
            ("post_norm", FLASH_NEXT_BRANCHES, FLASH_NEXT_HIDDEN),
            ("post_norm", 4, 2048),
            ("post_norm", 2, 4096),
            # Branches off the 32-element group boundary: plain kernel plus
            # the standalone quantizer over the flattened row.
            ("post_norm", 8, 20),
        ],
        ids=["pre_mix", "post_norm_it3", "post_norm_it2", "post_norm_it4", "hidden20"],
    )
    def test_hyper_connection(self, model_runner, kind, C, H):
        rng = np.random.default_rng(47)
        nodes, inputs, inits, extra_outs, K = _hc_producer(kind, 1, C, H, rng)
        feeds = _hc_feeds(kind, 1, C, H, rng)
        self._assert_exact(model_runner, nodes, inputs, inits, extra_outs, K, feeds)
