#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Shared model/device constants and dispatch-stream segmentation.

The analysis scripts all start from one `*_dispatches.csv` emitted by
tools/rgp_parser. That CSV has no notion of layers, MoE regions or expert token
counts -- it is a flat list of dispatches. Everything structural here is
recovered from the stream itself, with no instrumentation in the build:

  region      topk_routing opens the MoE region; the attention/layernorm kernels
              that never appear inside the expert loop close it. Which kernels
              those are is per-model (`ModelSpec.markers`): ew_bcast4d closes the
              region on gpt-oss but scales the routing weights *inside* it on
              Qwen3.6, so one global set would mis-segment one of them.
  expert M    the gather_tokens opening each expert block launches
              ceil(M*hidden/256)*256 threads, so M = round(threads/hidden).
              Models whose MoE runs as bucketed grouped GEMMs instead of a
              per-expert gather/scatter loop have no such blocks at all; that is
              reported as a fused region rather than as zero work.
  lm_head     the only MatMulNBits dispatch with a vocab-sized thread count.
  layers      one topk_routing per layer, so a partial capture window can be
              scaled to a whole layer stack. On a hybrid stack that single
              multiplier is wrong for the attention kernels -- see `scale_for`.

That matters because the alternative -- HIPDNN_EP_PERF -- costs about 4% and is
forbidden for throughput work.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Device:
    """Ceilings for the part under test, not best-observed rates.

    Ranking by best-observed rate is circular: it hides a ceiling the whole
    stack is missing. Defaults are Radeon 8060S / gfx1151 (Strix Halo).

    compute   40 CU RDNA 3.5 @ 2.9 GHz x 512 FLOP/clk/CU = 59.4 TFLOP/s fp16.
              The same model gives 122.9 TFLOP/s for a 7900 XTX against AMD's
              published 122.8, which is the check that it is the right model.
              Note hipInfo reports multiProcessorCount=20: those are WGPs, two
              CUs each.
    bandwidth 256-bit LPDDR5X-8000 = 256 GB/s.
    """

    bw_bytes_s: float = 256e9
    peak_flops: float = 59.4e12

    def floor_s(self, byts: float, flop: float) -> float:
        """Lower bound on time for work that must move `byts` and do `flop`."""
        return max(byts / self.bw_bytes_s, flop / self.peak_flops)


@dataclass(frozen=True)
class ModelSpec:
    """Shapes needed to turn a dispatch stream into work. Defaults: gpt-oss-20b."""

    hidden: int = 2880
    inter: int = 2880
    vocab: int = 201088
    layers: int = 24
    qkv_n: int = 5120
    o_proj_k: int = 4096
    router_n: int = 32
    experts: int = 32
    topk: int = 4
    group_size: int = 32  # int4 quant block -> one fp16 scale per N weights
    chunk_tokens: int = 512  # chunked prefill granularity
    chunks: int = 32  # 16k prefill = 32 chunks
    heads: int = 64
    kv_heads: int = 8
    head_dim: int = 64
    sliding_window: int = 128
    full_attn_layers: int = 12  # the rest are sliding-window
    # Gemma-4 gives its global-attention layers a different KV geometry from its
    # sliding ones, so the floor cannot assume one kv_heads/head_dim for all
    # layers. These default to the sliding values, i.e. homogeneous.
    full_kv_heads: int = 0  # 0 -> same as kv_heads
    full_head_dim: int = 0  # 0 -> same as head_dim
    # Some MoE models also carry a dense MLP per layer alongside the experts.
    dense_inter: int = 0  # 0 -> no dense MLP

    # A hybrid stack replaces most of its softmax attention with a linear
    # (recurrent) operator, so the two layer groups share neither a cost model
    # nor a dispatch signature. Non-zero here switches on the hybrid paths:
    # per-layer-type window scaling in Capture and the recurrence floor in
    # headroom.py. Zero keeps every model that predates it on the old path.
    linear_attn_layers: int = 0
    la_value_heads: int = 0  # state is la_value_heads x la_head_dim x la_head_dim
    la_head_dim: int = 0
    la_chunk: int = 0  # tokens per chunk in the chunked (WMMA) formulation
    la_window_chunks: int = 0  # chunks per scan window (LA_WINDOW_CHUNKS)
    la_proj_n: int = 0  # in_proj width of a linear layer (qkv + z + a + b)
    la_out_k: int = 0  # out_proj K of a linear layer
    conv_channels: int = 0  # short causal conv ahead of the recurrence
    conv_kernel: int = 0

    # Kernel families that close the MoE region, and the ones that mark a layer
    # of each type. Empty means "use the module-level defaults".
    markers: frozenset = frozenset()
    linear_markers: frozenset = frozenset()
    full_markers: frozenset = frozenset()

    @property
    def hybrid(self) -> bool:
        return self.linear_attn_layers > 0

    @property
    def full_kv(self) -> int:
        return self.full_kv_heads or self.kv_heads

    @property
    def full_hd(self) -> int:
        return self.full_head_dim or self.head_dim

    @property
    def la_state_bytes(self) -> float:
        """One layer's recurrent state, the thing a chunk has to read and write."""
        return self.fp16(self.la_value_heads * self.la_head_dim * self.la_head_dim)

    def fp16(self, n: float) -> float:
        return n * 2

    def int4w(self, n: float) -> float:
        """Packed int4 weights plus their fp16 scales."""
        return n * 0.5 + n / self.group_size * 2

    @property
    def expert_weight_bytes(self) -> float:
        return self.int4w(self.hidden * 2 * self.inter) + self.int4w(
            self.inter * self.hidden
        )

    @property
    def expert_flop_per_token(self) -> float:
        return 2.0 * (self.hidden * 2 * self.inter + self.inter * self.hidden)


# Kernels that only ever run outside the MoE expert loop; seeing one means the
# expert region has closed.
DENSE_MARKERS = frozenset(
    {
        "Op2dTensorGeneric",
        "T5LayernormFwdContiguous",
        "split_qkv",
        "ropex2",
        "rope",
        "kv_cache_append",
        "gqa_flash_prefill_v5",
        "ew_bcast4d",
        "gather",
        "elementwise_sub_i64",
        "cast_i64_to_i32",
        "reduce_sum_i64",
    }
)

# One dispatch of these per linear-attention / full-attention layer, so counting
# them says how many layers of each type a window actually holds. Chosen for
# being exactly one per layer: the la_pf_* kernels are launched once per window
# (LA_WINDOW_CHUNKS), so they count windows, not layers, and cannot be used here.
LINEAR_LAYER_MARKERS = frozenset({"causal_conv_prefill", "causal_conv_step"})
FULL_LAYER_MARKERS = frozenset(
    {"gqa_flash_prefill_v5", "gqa_flash_prefill_v8", "gqa_flash_decode"}
)

# Kernels belonging to one layer type, for attributing a window's time. The
# recurrence and the conv only run in linear layers; the flash kernels, the KV
# append and rope only run in full ones. Layout kernels (strided_copy,
# transpose_tiled) are absent because who owns them is model-specific: a preset
# that has checked its stream can claim them via ModelSpec.linear_markers, and
# otherwise they take the generic per-layer scale rather than a guessed owner.
LINEAR_ATTN_FAMILIES = frozenset(
    {
        "la_pf_pass1_local",
        "la_pf_pass2",
        "la_pf_pass3_wmma",
        "la_pf_scan",
        "la_decode",
        "causal_conv_prefill",
        "causal_conv_step",
    }
)
FULL_ATTN_FAMILIES = frozenset(
    {
        "gqa_flash_prefill_v5",
        "gqa_flash_prefill_v8",
        "gqa_flash_decode",
        "kv_cache_append",
        "rope",
        "ropex2",
        "split_qkv",
    }
)

# Qwen3.6's MoE region contains ew_bcast4d (routing-weight and shared-expert-gate
# scaling) and MIOpenActiveFwdLite, both of which close the region on gpt-oss, so
# it needs its own set. Everything here is a kernel that only ever runs outside
# the region on this model, verified against the dispatch stream.
QWEN_MARKERS = frozenset(
    {
        "Op2dTensorGeneric",
        "T5LayernormFwdContiguous",
        "transpose_tiled",
        "strided_copy",
        "kv_cache_append",
        "cast_f16_to_f32",
    }
    | LINEAR_ATTN_FAMILIES
    | FULL_LAYER_MARKERS
)

# The int4 matmul stack: the GEMMs plus the ancillary kernels they drag along.
INT4_FAMILIES = frozenset(
    {
        "MatMulNBitsWMMA_NoZP",
        "MatMulNBitsFp16GEMM",
        "matmul_nbits_gemv",
        "dequant_u4_to_fp16",
        "matmul_nbits_add_bias_rowmajor",
        "transpose2d_fp16",
    }
)

GEMM_FAMILIES = ("MatMulNBitsWMMA_NoZP", "MatMulNBitsFp16GEMM", "matmul_nbits_gemv")

# Buckets chosen to straddle the dispatch thresholds in matmul_nbits_kernel.hip
# (row-major GEMV, col-major GEMV, WMMA), so a routing change shows up as a
# bucket moving rather than as a diffuse shift.
M_BUCKETS = [(1, 1), (2, 15), (16, 63), (64, 255), (256, 10**9)]


def linear_attn_floor(
    spec: ModelSpec, dev: Device, tokens: int
) -> tuple[float, float, float]:
    """Seconds, bytes, FLOP for every linear-attention layer over `tokens`.

    Covers the whole linear layer group: the 4-tap causal conv plus the chunked
    delta-rule recurrence. Unlike softmax attention this is linear in context --
    each chunk sees the fixed-size recurrent state instead of the whole history --
    so the cost that dominates is state traffic, not a growing KV read.

    The state term assumes the chunked formulation the kernels actually use: one
    state read and write per chunk of `la_chunk` tokens. That is a property of
    the algorithm, not of the hardware -- a serial recurrence would trade the
    traffic for no parallelism -- so it is a floor for this algorithm, and a
    larger chunk is a real lever on it rather than an inefficiency to recover.
    """
    n = spec.linear_attn_layers
    if not n:
        return 0.0, 0.0, 0.0
    hv, d, c = spec.la_value_heads, spec.la_head_dim, spec.la_chunk
    chunks = max(1, -(-tokens // c))  # ceil

    # Per chunk per head: readout Q@S, the state update K^T@U and the delta
    # correction K@S are each 2*c*d*d; the intra-chunk QK^T and its @V are
    # 2*c*c*d apiece.
    flop = n * chunks * hv * (3 * 2 * c * d * d + 2 * 2 * c * c * d)

    state_b = n * chunks * spec.la_state_bytes * 2  # read + write per chunk
    stream_b = n * spec.fp16(tokens * (spec.la_proj_n + spec.la_out_k))
    conv_b = n * spec.fp16(tokens * spec.conv_channels) * 2  # in + out
    conv_flop = n * 2 * tokens * spec.conv_channels * spec.conv_kernel

    byts = state_b + stream_b + conv_b
    flop += conv_flop
    return dev.floor_s(byts, flop), byts, flop


@dataclass
class ExpertBlock:
    """One expert served: gather_tokens through scatter_add.

    The whole block is the honest unit -- it is what serving one expert costs,
    including the bias/swiglu/scatter kernels, not just the two GEMMs.
    """

    m: int
    dur_us: float = 0.0
    dequant_us: float = 0.0
    gemm_us: float = 0.0
    kernels: set = field(default_factory=set)


class Capture:
    """A decoded dispatch CSV, segmented into regions, layers and expert blocks."""

    def __init__(self, path: str, spec: ModelSpec):
        self.path = path
        self.spec = spec
        self.rows = list(csv.DictReader(open(path)))
        self.markers = spec.markers or DENSE_MARKERS
        self.linear_fams = spec.linear_markers or LINEAR_ATTN_FAMILIES
        self.full_fams = spec.full_markers or FULL_ATTN_FAMILIES
        self.total_us = sum(float(r["dur_us"]) for r in self.rows)
        fams = [r["family"] for r in self.rows]
        self.linear_in_window = sum(1 for f in fams if f in LINEAR_LAYER_MARKERS)
        self.full_in_window = sum(1 for f in fams if f in FULL_LAYER_MARKERS)
        # A window can catch the recurrence without the conv that dates it -- the
        # la_pf_* kernels run once per scan window, so a capture triggered inside
        # them holds no layer marker at all. The window count per layer is fixed
        # by the config, so divide it out instead of giving up.
        if not self.linear_in_window and spec.la_window_chunks:
            wins = sum(1 for f in fams if f == "la_pf_pass1_local")
            span = spec.la_chunk * spec.la_window_chunks
            per_layer = max(1, -(-spec.chunk_tokens // span))
            if wins:
                self.linear_in_window = max(1, round(wins / per_layer))
        self.layers_in_window = sum(1 for f in fams if f == "topk_routing")
        # A window can hold whole layers without holding their topk_routing --
        # any capture triggered inside the attention half of a layer does. Those
        # captures used to be unusable; the attention markers date them instead.
        self.layers_from_attn = self.linear_in_window + self.full_in_window
        # lm_head: the only MatMulNBits with a vocab-sized thread count.
        self.lm_head_idx = [
            i
            for i, r in enumerate(self.rows)
            if r["family"].startswith("MatMulNBits") and int(r["threads"]) > 1_500_000
        ]
        self.lm_head_us = sum(float(self.rows[i]["dur_us"]) for i in self.lm_head_idx)
        self.regions = self._segment()
        self.blocks = self._expert_blocks()

    @property
    def layer_groups(self) -> int:
        """Whole layers in the window, from topk_routing or the attention markers."""
        return self.layers_in_window or self.layers_from_attn

    @property
    def layer_scale(self) -> float:
        """Multiplier taking the captured window to one full layer stack.

        Right for anything that runs once per layer regardless of layer type --
        the MoE region, the norms, the projections. On a hybrid stack it is wrong
        for the attention kernels, so use `scale_for` when the family is known.
        """
        if not self.layer_groups:
            raise ValueError(
                f"{self.path}: no topk_routing or attention-layer dispatches; "
                "cannot infer layer count from this window"
            )
        return self.spec.layers / self.layer_groups

    def scale_for(self, family: str) -> float:
        """Window-to-stack multiplier for one kernel family.

        A hybrid stack holds two layer types in different proportions, and a
        capture window holds them in a third. Scaling a full-attention kernel by
        `layers / layer_groups` claims the model has `layers` of them: on
        Qwen3.6-35B-A3B, a window with one linear and one full layer scaled its
        one gqa_flash dispatch to 40 layers of flash attention against a true 10,
        a 4x overstatement, while understating the recurrence. Each family is
        scaled by its own layer type's census instead.
        """
        if self.spec.hybrid:
            if family in self.linear_fams and self.linear_in_window:
                return self.spec.linear_attn_layers / self.linear_in_window
            if family in self.full_fams and self.full_in_window:
                return self.spec.full_attn_layers / self.full_in_window
        return self.layer_scale

    def scaled_us(self, i: int) -> float:
        """Dispatch i's duration, scaled to the whole layer stack."""
        if i in self.lm_head_idx:  # once per chunk, not once per layer
            return float(self.rows[i]["dur_us"])
        return float(self.rows[i]["dur_us"]) * self.scale_for(self.rows[i]["family"])

    @property
    def chunk_us(self) -> float:
        """The whole window scaled to one chunk of prefill."""
        return sum(self.scaled_us(i) for i in range(len(self.rows)))

    @property
    def moe_region_us(self) -> float:
        """Measured MoE-region time in this window, scaled to one chunk."""
        return sum(
            self.scaled_us(i)
            for i in range(len(self.rows))
            if self.regions[i] == "qmoe" and i not in self.lm_head_idx
        )

    @property
    def fused_moe(self) -> bool:
        """True when the MoE runs as bucketed grouped GEMMs, not expert blocks.

        The expert-block segmentation keys on gather_tokens. A model that routes
        by bucketing tokens and then running one grouped GEMM per shard never
        emits it, and reporting that as zero MoE work would be a lie -- the
        region is there, it just is not divided into per-expert blocks.
        """
        return not self.blocks and any(r["family"] == "topk_routing" for r in self.rows)

    def _segment(self) -> list[str]:
        out, region = [], "dense"
        for r in self.rows:
            fam = r["family"]
            if fam == "topk_routing":
                region = "qmoe"
            elif fam in self.markers:
                region = "dense"
            out.append(region)
        return out

    def _expert_blocks(self) -> list[ExpertBlock]:
        out: list[ExpertBlock] = []
        cur: ExpertBlock | None = None
        for i, r in enumerate(self.rows):
            if self.regions[i] != "qmoe" or i in self.lm_head_idx:
                continue
            fam, dur = r["family"], float(r["dur_us"])
            if fam == "gather_tokens":
                if cur:
                    out.append(cur)
                cur = ExpertBlock(m=round(int(r["threads"]) / self.spec.hidden))
            elif cur is not None:
                cur.dur_us += dur
                if fam in GEMM_FAMILIES:
                    cur.kernels.add(fam)
                    cur.gemm_us += dur
                elif fam == "dequant_u4_to_fp16":
                    cur.dequant_us += dur
        if cur:
            out.append(cur)
        return out


# Whole-model geometries, so a run does not need a dozen --flags to be correct.
# Every field is checkable against the dispatch stream: the per-layer GEMM N/K
# values, one router dispatch of N=experts per layer, one topk_routing per layer.
PRESETS: dict[str, dict] = {
    "gpt-oss-20b": {},
    # google/gemma-4-26B-A4B-it. 30 layers: 25 sliding GQA (kv8 x 256) and
    # 5 global MQA (kv2 x 512). 128 experts, top-8, expert inter 704, plus a
    # dense MLP of inter 2112 per layer.
    "gemma4-26b-a4b": dict(
        hidden=2816,
        inter=704,
        vocab=262144,
        layers=30,
        qkv_n=4096 + 2048 + 2048,
        o_proj_k=4096,
        router_n=128,
        experts=128,
        topk=8,
        group_size=32,
        heads=16,
        kv_heads=8,
        head_dim=256,
        sliding_window=1024,
        full_attn_layers=5,
        full_kv_heads=2,
        full_head_dim=512,
        dense_inter=2112,
    ),
    # Qwen/Qwen3.6-35B-A3B, int4 gs32. A hybrid stack: 40 layers as 30
    # linear-attention (gated delta rule + a 4-tap causal conv over 8192
    # channels) and 10 full-attention every 4th layer, all 40 carrying a
    # 256-expert top-8 MoE plus a shared expert of inter 512.
    #
    # Checks against the graph: q_proj N=8192 is gated attention, i.e. 16 heads x
    # 256 for q plus an equal gate, so the attention itself is 16 x 256 with
    # o_proj K=4096 and k/v N=512 each (kv2 x 256). The recurrent state is
    # [32, 128, 128], matching linear_num_value_heads x linear_key_head_dim.
    # Prefill is not chunked on this model -- one causal_conv_prefill dispatch
    # per layer covers the whole prompt -- so chunks=1 and chunk_tokens is the
    # measured operating point of the ab_interleaved harness.
    "qwen36-35b-a3b": dict(
        hidden=2048,
        inter=512,
        vocab=248320,
        layers=40,
        qkv_n=8192 + 512 + 512,
        o_proj_k=4096,
        router_n=256,
        experts=256,
        topk=8,
        group_size=32,
        chunk_tokens=3985,
        chunks=1,
        heads=16,
        kv_heads=2,
        head_dim=256,
        sliding_window=0,  # no sliding group; the other 30 layers are linear
        full_attn_layers=10,
        dense_inter=512,  # shared expert, every layer
        linear_attn_layers=30,
        la_value_heads=32,
        la_head_dim=128,
        la_chunk=32,
        la_window_chunks=16,  # 512 tokens per window -> 8 windows at 3985
        la_proj_n=8192 + 4096 + 32 + 32,
        la_out_k=4096,
        conv_channels=8192,
        conv_kernel=4,
        markers=QWEN_MARKERS,
        # transpose_tiled is the [B,L,C] <-> [B,C,L] pair bracketing the conv:
        # two per linear layer, none in a full-attention layer, so it belongs to
        # the linear group. strided_copy serves both and is left generic.
        linear_markers=LINEAR_ATTN_FAMILIES | {"transpose_tiled"},
    ),
}

# Fields settable from the command line; None means "fall back to the preset".
_OVERRIDES = (
    "hidden",
    "inter",
    "vocab",
    "layers",
    "chunks",
    "chunk_tokens",
    "experts",
    "topk",
    "heads",
    "kv_heads",
    "head_dim",
    "full_attn_layers",
    "full_kv_heads",
    "full_head_dim",
    "sliding_window",
    "dense_inter",
    "linear_attn_layers",
)


def add_common_args(ap: argparse.ArgumentParser, *, many: bool = False) -> None:
    """Capture path(s) plus the handful of shapes worth overriding per model."""
    ap.add_argument(
        "captures" if many else "capture",
        nargs="+" if many else None,
        help="*_dispatches.csv from tools/rgp_parser",
    )
    ap.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="gpt-oss-20b",
        help="model geometry to start from; --flags override it",
    )
    for name in _OVERRIDES:
        ap.add_argument(f"--{name.replace('_', '-')}", type=int, default=None)
    ap.add_argument("--bw-gbs", type=float, default=256.0, help="memory roofline, GB/s")
    ap.add_argument(
        "--peak-tflops", type=float, default=59.4, help="fp16 compute roofline"
    )


def specs_from_args(args) -> tuple[ModelSpec, Device]:
    fields = dict(PRESETS[getattr(args, "preset", "gpt-oss-20b")])
    for name in _OVERRIDES:
        val = getattr(args, name, None)
        if val is not None:
            fields[name] = val
    spec = ModelSpec(**fields)
    dev = Device(bw_bytes_s=args.bw_gbs * 1e9, peak_flops=args.peak_tflops * 1e12)
    return spec, dev


def bucket_label(lo: int, hi: int) -> str:
    return f">={lo}" if hi >= 10**9 else f"{lo}..{hi}"
