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
              that never appear inside the expert loop close it.
  expert M    the gather_tokens opening each expert block launches
              ceil(M*hidden/256)*256 threads, so M = round(threads/hidden).
  lm_head     the only MatMulNBits dispatch with a vocab-sized thread count.
  layers      one topk_routing per layer, so a partial capture window can be
              scaled to a whole layer stack.

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

    @property
    def full_kv(self) -> int:
        return self.full_kv_heads or self.kv_heads

    @property
    def full_hd(self) -> int:
        return self.full_head_dim or self.head_dim

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
        self.total_us = sum(float(r["dur_us"]) for r in self.rows)
        self.layers_in_window = sum(
            1 for r in self.rows if r["family"] == "topk_routing"
        )
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
    def layer_scale(self) -> float:
        """Multiplier taking the captured window to one full layer stack."""
        if not self.layers_in_window:
            raise ValueError(
                f"{self.path}: no topk_routing dispatches; "
                "cannot infer layer count from this window"
            )
        return self.spec.layers / self.layers_in_window

    def _segment(self) -> list[str]:
        out, region = [], "dense"
        for r in self.rows:
            fam = r["family"]
            if fam == "topk_routing":
                region = "qmoe"
            elif fam in DENSE_MARKERS:
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
