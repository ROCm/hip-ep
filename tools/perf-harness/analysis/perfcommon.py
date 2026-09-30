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
  layers      one LAYER_MARKERS dispatch per layer, so a partial capture window
              can be scaled to a whole layer stack.

That matters because the alternative -- HIPDNN_EP_PERF -- costs about 4% and is
forbidden for throughput work.
"""

from __future__ import annotations

import argparse
import collections
import csv
import re
import statistics
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
    # Quantisation is per-tensor in practice, not whole-model: an export can
    # leave the lm_head and the MoE router in fp16 while quantising everything
    # else. Assuming int4 throughout understates decode traffic badly -- on
    # qwen3-30b-a3b the fp16 lm_head alone is 622 MB per token, four times what
    # the int4 assumption predicts and ~30% of the whole step.
    lm_head_fp16: bool = False
    router_fp16: bool = False
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

    # ---- decode (M=1) per-token traffic -------------------------------------
    #
    # At M=1 every GEMM is a GEMV: one pass over the weights for a single row of
    # activations, so arithmetic intensity is ~2 FLOP/byte and the floor is
    # bytes/BW with the FLOP term never binding. That makes the whole decode
    # step a memory-traffic budget, and these are its line items. Splitting them
    # out is the point -- a component's share of runtime says nothing about
    # whether it is near its own floor.

    def decode_bytes(self, kv_len: int) -> dict[str, float]:
        """Bytes each component must move to produce one token at `kv_len`.

        Only the KV term depends on the sequence length; every weight term is
        constant, which is why decode TPS degrades so gently with context until
        the KV read overtakes the weights.
        """
        w = self.int4w
        experts = (
            self.topk
            * self.layers
            * (w(self.hidden * 2 * self.inter) + w(self.inter * self.hidden))
        )
        # q/k/v/o. kv_heads*head_dim twice for K and V.
        qkvo = self.layers * (
            w(self.hidden * self.heads * self.head_dim)
            + 2 * w(self.hidden * self.kv_heads * self.head_dim)
            + w(self.heads * self.head_dim * self.hidden)
        )
        router_w = self.hidden * self.experts
        router = self.layers * (
            self.fp16(router_w) if self.router_fp16 else w(router_w)
        )
        lm_w = self.hidden * self.vocab
        lm_head = self.fp16(lm_w) if self.lm_head_fp16 else w(lm_w)
        # K and V, fp16, read once per layer: every past position on a global
        # layer, at most the window on a sliding one, each with its own head
        # geometry -- the split headroom.py's prefill attention floor uses.
        # Charging every layer the full context overstated gemma-4-26B's 2K
        # KV read about 2x.
        nfull = self.layers
        if self.sliding_window:
            nfull = min(self.full_attn_layers, self.layers)
        win = min(kv_len, self.sliding_window)
        kv = (
            (
                nfull * kv_len * self.full_kv * self.full_hd
                + (self.layers - nfull) * win * self.kv_heads * self.head_dim
            )
            * 2
            * 2
        )
        # Any FFN outside the expert loop: the shared MLP beside the experts, or
        # on a dense preset (experts=0) the whole FFN, which the expert term
        # carries as topk=1. Same rule as headroom.py's prefill floor.
        dense = self.layers * (
            w(self.hidden * 2 * self.dense_inter) + w(self.dense_inter * self.hidden)
        )
        if not self.experts:
            dense, experts = dense + experts, 0.0
        return {
            "moe_experts": experts,
            "dense_mlp": dense,
            "attn_proj": qkvo,
            "router": router,
            "lm_head": lm_head,
            "kv_cache": kv,
        }

    def decode_bytes_total(self, kv_len: int) -> float:
        return sum(self.decode_bytes(kv_len).values())


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
# Both WMMA variants belong here. The kernel splits on whether the quantisation
# carries a zero point, which is an export-time property, not a different kind of
# work: the qmoe 26B export is symmetric and emits _NoZP, the dense 12B export is
# asymmetric and emits _ZP for every projection it has. Listing only _NoZP drops
# 84% of a 12B prefill window out of the int4 rollup and out of the --dense-ms
# that headroom.py ranks on.
INT4_FAMILIES = frozenset(
    {
        "MatMulNBitsWMMA_NoZP",
        "MatMulNBitsWMMA_ZP",
        "MatMulNBitsFp16GEMM",
        "matmul_nbits_gemv",
        "dequant_u4_to_fp16",
        "matmul_nbits_add_bias_rowmajor",
        "transpose2d_fp16",
    }
)

GEMM_FAMILIES = (
    "MatMulNBitsWMMA_NoZP",
    "MatMulNBitsWMMA_ZP",
    "MatMulNBitsFp16GEMM",
    "matmul_nbits_gemv",
)

# Families that run exactly once per layer, tried in order. This is what turns a
# partial capture window into a whole layer stack, so being wrong here scales
# every number in every report.
#
# topk_routing is first because on an MoE model it is unambiguous: it opens the
# expert region and nothing else emits it. A dense model has no router at all,
# and gemma-4-12b has no `gqa` family either -- its attention is decomposed into
# gemm/softmax/rope/kv_cache_append, so there is no single attention dispatch to
# count. skip_rms_norm is the residual norm every transformer layer runs once,
# and it holds across both phases: on gemma-4-12b it counts 5 in a prefill
# window and 28 in a decode window, matching softmax_f32_to_out and
# elementwise_gelu_f16 in each. softmax is kept as a second fallback for models
# whose norm is fused away.
LAYER_MARKERS = ("topk_routing", "skip_rms_norm", "softmax_f32_to_out")

# EP op name (the OP_PROFILE / HIPDNN_EP_PERF_OPS name) -> the rgp_parser kernel
# families its wrapper launches: exact names from the kernels each op's own .hip
# file defines, plus prefixes for the templated variants. Only ops whose kernels
# are unique to them are listed. `gemm` is deliberately absent: the hipBLASLt
# family is launched by `matmul`, by `gemm`, and by decomposed attention inside
# `gqa`, so its owner depends on the model. Assign it with --map once a run pins
# it down. On gemma-4-26B decode a HIPDNN_EP_PERF_OPS=gqa,matmul,gemm run shows
# only gqa (30 calls/step) in the decoder, so its 60 gemm/step are the
# attention GEMMs: --map gqa+=gemm.
OP_KERNEL_FAMILIES: dict[str, dict[str, tuple[str, ...]]] = {
    "matmul_nbits": {
        "exact": (
            "dequant_u2_to_fp16",
            "dequant_u3_to_fp16",
            "dequant_u4_to_fp16",
            "transpose2d_fp16",
        ),
        "prefix": ("matmul_nbits", "MatMulNBits"),
    },
    "qmoe": {
        "exact": (
            "topk_routing",
            "bucket_tokens",
            "gather_tokens",
            "scatter_add",
            "swiglu",
            "add_bias",
        ),
        "prefix": ("qmoe_",),
    },
    "gqa": {
        "exact": (
            "add_attention_bias",
            "bias_key_extent",
            "causal_mask_kernel_impl",
            "dequant_kv_i8_to_fp16",
            "expand_kv",
            "rope",
            "softmax_f32_to_out",
            "softmax_inplace",
            "split_qkv",
            "transpose_mid_dims",
        ),
        "prefix": ("gqa", "kv_cache_"),
    },
}


def op_family_map(extra: list[str] | None = None) -> dict[str, dict[str, tuple]]:
    """OP_KERNEL_FAMILIES plus --map overrides: 'op=fam,fam' replaces the op's
    families, 'op+=fam' adds to them. Exact names; a trailing '*' is a prefix."""
    out = {k: dict(v) for k, v in OP_KERNEL_FAMILIES.items()}
    for spec in extra or []:
        op, _, fams = spec.partition("=")
        append = op.endswith("+")
        op = op.rstrip("+").strip()
        if not op or not fams:
            raise SystemExit(
                f"--map wants op=family[,family*] or op+=family, got {spec!r}"
            )
        names = [f.strip() for f in fams.split(",") if f.strip()]
        exact = tuple(n for n in names if not n.endswith("*"))
        prefix = tuple(n[:-1] for n in names if n.endswith("*"))
        # A family belongs to one op; moving it must take it off its old owner.
        for other, m in out.items():
            if other != op:
                m["exact"] = tuple(n for n in m["exact"] if n not in exact)
        base = (
            out.get(op, {"exact": (), "prefix": ()})
            if append
            else {"exact": (), "prefix": ()}
        )
        out[op] = {"exact": base["exact"] + exact, "prefix": base["prefix"] + prefix}
    return out


def op_for_family(family: str, fmap: dict[str, dict[str, tuple]]) -> str | None:
    for op, m in fmap.items():
        if family in m["exact"] or any(family.startswith(p) for p in m["prefix"]):
            return op
    return None


def load_dispatches(path: str) -> list[dict]:
    """Dispatch rows in time order. The parser's artifact rows (no waves
    attributed, so no trustworthy duration) are kept with zero duration and
    r["artifact"] set: the dispatch still happened, so dropping it undercounts
    layer markers (15 of 72 topk_routing on a gemma-4-26B 16K capture, which
    failed every step window) and breaks the adjacency rules in decode_model."""
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["artifact"] = r.get("is_artifact") == "1"
        if r["artifact"]:
            r["dur_us"] = "0"
        r["t0"] = float(r["ts_us"])
        r["t1"] = r["t0"] + float(r["dur_us"])
    rows.sort(key=lambda r: r["t0"])
    return rows


def decode_step_windows(
    rows: list[dict], layers: int = 0
) -> tuple[list[tuple[int, int]], str, int]:
    """Complete decode steps in a capture, as [start, end) row windows.

    A VLM decode step runs on two queues: the embedding session's dispatches,
    then the decoder's. A step therefore starts where the stream switches from
    the decoder's (busiest) queue to another one, and only windows between two
    such switches are complete. Time on the other queue belongs to the other
    session; it is not the decoder's GPU idle, which is what summing every
    gap_before_us in the window used to claim.

    With `layers`, a window is kept only if its layer-marker count is within
    10% of `layers`, so a stray dispatch on another queue mid-step cannot split
    a step (half a step shows about half the markers). Not exact equality: the
    gemma-4-26B decode step runs 29 topk_routing for 30 layers.
    Returns (windows, decoder queue, windows rejected by that check). A
    single-queue capture has no switches and returns no windows.
    """
    queues = collections.Counter(r.get("queue", "") for r in rows)
    main_q = queues.most_common(1)[0][0] if queues else ""
    if len(queues) < 2:
        return [], main_q, 0
    bounds = [
        i
        for i in range(1, len(rows))
        if rows[i].get("queue") != main_q and rows[i - 1].get("queue") == main_q
    ]
    wins = list(zip(bounds, bounds[1:]))
    if not layers:
        return wins, main_q, 0
    counts = collections.Counter(r["family"] for r in rows)
    marker = next((f for f in LAYER_MARKERS if counts.get(f)), None)
    if marker is None:
        return wins, main_q, 0
    tol = max(1, layers // 10)
    kept = [
        (s, e)
        for s, e in wins
        if abs(sum(1 for r in rows[s:e] if r["family"] == marker) - layers) <= tol
    ]
    return kept, main_q, len(wins) - len(kept)


_PERF_BORDER = re.compile(r"^\[PERF\] =+\s*$")
# Parent rows only: the op name sits two spaces in, shape rows four.
_PERF_ROW = re.compile(
    r"^\[PERF\]  (\S+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s*$"
)


def perf_op_tables(path: str) -> list[dict[str, tuple[int, float]]]:
    """Every per-Compute `[PERF] ===` op table in a log: {op: (calls, gpu_ms)}."""
    tables: list[dict[str, tuple[int, float]]] = []
    cur: dict[str, tuple[int, float]] | None = None
    # Windows PowerShell 5.1's Tee-Object writes UTF-16; read as UTF-8 it
    # matches nothing and the log looks like it has no tables.
    with open(path, "rb") as f:
        enc = "utf-16" if f.read(2) in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
    with open(path, encoding=enc, errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if _PERF_BORDER.match(line):
                if cur is None:
                    cur = {}
                else:
                    tables.append(cur)
                    cur = None
                continue
            if cur is not None:
                m = _PERF_ROW.match(line)
                if m:
                    cur[m.group(1)] = (int(m.group(2)), float(m.group(3)))
    return tables


def decode_op_stats(
    tables: list[dict[str, tuple[int, float]]], op: str, drop_top: float = 0.05
) -> dict | None:
    """Per-decode-step calls and GPU ms of `op`, from its per-Compute tables.

    Prefill and decode Runs execute the same graph, so the call count cannot
    separate them; the GPU time can -- a prefill Run is many times a decode
    step. Tables outside [median/3, 3*median] go, then the slowest `drop_top`
    (warmup, autotune), then any whose call count is not the mode.
    """
    vals = [t[op] for t in tables if op in t]
    if not vals:
        return None
    med = statistics.median(v[1] for v in vals)
    keep = [v for v in vals if med / 3 <= v[1] <= 3 * med]
    keep.sort(key=lambda v: v[1])
    keep = keep[: max(1, int(round(len(keep) * (1 - drop_top))))]
    mode = collections.Counter(v[0] for v in keep).most_common(1)[0][0]
    keep = [v for v in keep if v[0] == mode]
    return {
        "tables": len(vals),
        "steps": len(keep),
        "calls": mode,
        "gpu_ms": statistics.median(v[1] for v in keep),
    }


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
    # Bytes the parser measured moving, from SPM: mem_gbps x duration, summed over
    # the block's dispatches. Carried so a bandwidth floor can be checked against
    # what the hardware actually moved instead of being believed on its own -- see
    # `achieved_gbps` and the note on ModelSpec.expert_weight_bytes.
    mem_bytes: float = 0.0

    @property
    def achieved_gbps(self) -> float:
        """Measured bandwidth over the whole block, 0 when SPM was not captured."""
        return (self.mem_bytes / (self.dur_us * 1e-6) / 1e9) if self.dur_us else 0.0


class Capture:
    """A decoded dispatch CSV, segmented into regions, layers and expert blocks."""

    def __init__(self, path: str, spec: ModelSpec):
        self.path = path
        self.spec = spec
        self.rows = list(csv.DictReader(open(path)))
        self.total_us = sum(float(r["dur_us"]) for r in self.rows)
        counts = collections.Counter(r["family"] for r in self.rows)
        self.layer_marker = next(
            (f for f in LAYER_MARKERS if counts.get(f)), LAYER_MARKERS[0]
        )
        self.layers_in_window = counts.get(self.layer_marker, 0)
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
                f"{self.path}: none of {', '.join(LAYER_MARKERS)} appear; "
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
                # mem_gbps is absent without -Counters and 0 where the parser
                # could not classify; either way it contributes nothing.
                try:
                    cur.mem_bytes += float(r.get("mem_gbps") or 0.0) * 1e9 * dur * 1e-6
                except ValueError:
                    pass
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
    # google/gemma-4-12b-it, rtn int4 block-32. Dense, not MoE:
    # enable_moe_block is false in model_config.json and the decoder trace has no
    # qmoe or router GEMM at all. The FFN is carried by the expert term with
    # topk=1, which is the same arithmetic one layer's FFN already does -- one
    # gate and one up of hidden x inter, one down of inter x hidden -- so the
    # byte counts come out right and no separate dense-MLP term is needed.
    #
    # 48 layers: 40 sliding (kv8 x 256, window 1024) and 8 global. The global
    # layers project K and V once and share the result (attention_k_eq_v), so
    # they carry kv1 x 512 rather than two separate heads. Per prefill Run the
    # trace runs, all on matmul_nbits:
    #   n=4096,k=3840 x40 (q)   n=2048,k=3840 x80 (k,v)  n=3840,k=4096 x40 (o)
    #   n=8192,k=3840 x8  (q)   n=512,k=3840   x8 (kv)   n=3840,k=8192 x8  (o)
    #   n=15360,k=3840 x96 (gate,up)                     n=3840,k=15360 x48 (down)
    # and one n=262144,k=3840 at m=1, so the lm_head is int4 and is already
    # pruned to the last row in prefill -- do not pass --lm-head-ms for it.
    #
    # qkv_n and o_proj_k are the per-layer averages over those two geometries
    # ((40*8192 + 8*8704)/48 and (40*4096 + 8*8192)/48) rather than the sliding
    # values. int4w() is linear in the element count, so the average reproduces
    # the exact total weight bytes across all 48 layers; taking the sliding
    # numbers for every layer would understate the o_proj term by 17%.
    "gemma4-12b": dict(
        hidden=3840,
        inter=15360,
        vocab=262144,
        layers=48,
        qkv_n=8277,
        o_proj_k=4779,
        router_n=0,
        experts=0,
        topk=1,
        group_size=32,
        heads=16,
        kv_heads=8,
        head_dim=256,
        sliding_window=1024,
        full_attn_layers=8,
        full_kv_heads=1,
        full_head_dim=512,
        dense_inter=0,
    ),
    # Qwen/Qwen3-30B-A3B, rtn int4 g128 + fp16, lm_head pruned but NOT
    # quantised. 48 identical layers, no sliding window, no dense MLP alongside
    # the experts. Every field below is confirmed against the decode trace:
    # matmul_nbits runs n=4096,k=2048 (q) / n=512,k=2048 x2 (k,v) /
    # n=2048,k=4096 (o) per layer, matmul runs n=128,k=2048 (router) x48 plus
    # one n=151936,k=2048 (lm_head), and gqa reports h=32,d=128.
    "qwen3-30b-a3b": dict(
        hidden=2048,
        inter=768,
        vocab=151936,
        layers=48,
        qkv_n=4096 + 512 + 512,
        o_proj_k=4096,
        router_n=128,
        experts=128,
        topk=8,
        group_size=128,
        heads=32,
        kv_heads=4,
        head_dim=128,
        sliding_window=0,
        full_attn_layers=48,
        dense_inter=0,
        # quantization_config in model_config.json sets "lm_head": false, and
        # the trace confirms it: the lm_head lands on `matmul`, not
        # `matmul_nbits`. The router does too.
        lm_head_fp16=True,
        router_fp16=True,
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
    # 32 vs 128 changes every int4w() byte count by ~3%, so a wrong default here
    # silently shifts every floor in the report.
    "group_size",
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
    add_model_args(ap)


def add_model_args(ap: argparse.ArgumentParser) -> None:
    """Just the geometry/ceiling flags, for scripts that name their own input."""
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
