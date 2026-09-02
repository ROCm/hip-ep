#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Format a HIPDNN EP OGA model_benchmark log into a structured profiling report.

The report is the locked-down rendering of three independent measurement streams
that the EP + OGA emit into a single log when ``HIPDNN_EP_PERF=1`` is set:

    1. ``[PERF SUMMARY]`` - per-Compute aggregates (n samples, min/median/p99/max)
       emitted by ``MlirCustomOp.cpp::PerfCollector::dump_summary`` at library
       unload (the per-model compiled shared object: ``.so`` on Linux,
       ``.dll`` on Windows).
    2. model_benchmark stats block - OGA's own Prompt processing / Token generation
       / Token sampling / E2E summary printed by ``model_benchmark``.
    3. ``[PERF] === ... ===`` per-op tables - one per Compute(), emitted by
       ``op_profile_resolve_and_print`` in ``lib/Runtime/op_profile.cpp``.

The script REQUIRES model_benchmark output (stream 2) to render any report --
when the log has no ``Batch size:`` line the renderer returns the empty string
and the CLI exits with code 2 so the caller can fall back to a raw tail.
Streams 1 and 3 are EP-side and would be present under any host tool that
exercises the MorphiZen EP with ``HIPDNN_EP_PERF=1``, but # 1 / # 2 / footer
rendering is gated on the model_benchmark stats block being present.

Output layout (reading order is headline-first - most-skimmable numbers up top,
forensic distribution at the bottom):

    # 1  HEADLINE                       OGA's own throughput / latency table
    # 2  STEADY-STATE DECODE BREAKDOWN  synthesized "where does the time go" tree
    # 3  PER-OP GPU BREAKDOWN           op_profile.cpp's table verbatim
    # 4  PER-CALL DISTRIBUTION          raw EP Compute() distribution

Stability contract (treat as public API for downstream tooling):

    - Section markers always start with ``  # N  TITLE`` where N is a single
      digit. Tools may grep for ``^  # \\d  `` to find section boundaries.
    - The bottom banner is a single greppable line: ``^\\[OGA\\] prefill=``.
    - Section / banner widths are stable; the indent prefix is the only
      whitespace the caller may modify (via ``--indent``).
    - Each section silently no-ops when its source data is absent (running
      against a ``--mode none`` log prints only # 1 + bottom banner); the
      caller is responsible for raw-tail fallback when no
      ``Batch size:`` line is present (this script then exits with code 2).

USAGE
-----
    tools/perf-report/format_perf_report.py <log-file>
    tools/perf-report/format_perf_report.py <log-file> --no-banner  # CI embed
    tools/perf-report/format_perf_report.py <log-file> --indent 2   # indent N
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# --- parsers -----------------------------------------------------------------

# All five stats live on a single [PERF SUMMARY] row; parse via labeled tokens
# so we are robust to: (a) column reordering, (b) future-added stats (e.g.
# p95), (c) metric-name renames -- the renderer below addresses metrics by
# name, not position, so a new column shows up in __dict__ and is ignored
# unless the renderer is teaching about it.
_STAT_RE = re.compile(r"\b(min|mean|median|p99|max)=([-+0-9.eE]+)")


@dataclass
class PerfSummary:
    """Aggregated per-Compute metrics from the ``[PERF SUMMARY]`` block.

    ``metrics[name]`` is a dict with keys min / mean / median / p99 / max, all ms.
    Names are kept as written in the log (wall_ms, gpu_ms, etc.) so they
    cross-reference 1:1 with the source code in ``MlirCustomOp.cpp``.
    """

    total_inferences: int = 0
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    METRIC_ORDER = (
        "wall_ms",
        "compute_cpu_ms",
        "gpu_ms",
        "marshal_in_ms",
        "marshal_out_ms",
        "fence_residual_ms",
    )

    @classmethod
    def parse(cls, lines: list[str]) -> "PerfSummary":
        out = cls()
        for line in lines:
            if not line.startswith("[PERF SUMMARY]"):
                continue
            m = re.match(r"\[PERF SUMMARY\] total_inferences=(\d+)", line)
            if m:
                out.total_inferences = int(m.group(1))
                continue
            # Format: `[PERF SUMMARY] <name>  min=X  mean=X  median=X  p99=X  max=X`
            name_m = re.match(r"\[PERF SUMMARY\]\s+(\S+)\s+", line)
            if not name_m:
                continue
            stats = {k: float(v) for k, v in _STAT_RE.findall(line)}
            if stats:
                out.metrics[name_m.group(1)] = stats
        return out

    def __bool__(self) -> bool:
        return bool(self.metrics)


@dataclass
class OgaHeadline:
    """``model_benchmark`` stats block (Prompt processing, Token generation, etc.).

    Each section is parsed into a dict with the raw labels as keys (e.g.
    ``avg_us``, ``avg_tps``, ``p50_us``, ``stddev_us``, ``n``). Sections that
    were absent in the log resolve to None so ``__bool__`` can gate rendering.
    """

    batch_size: int = 0
    prompt_tokens: int = 0
    tokens_to_generate: int = 0
    prefill: Optional[dict] = None
    decode: Optional[dict] = None
    sampling: Optional[dict] = None
    e2e: Optional[dict] = None
    peak_ws_bytes: Optional[int] = None

    # Block header -> field name. Headers exactly match what model_benchmark prints.
    _BLOCKS = {
        "Prompt processing (time to first token):": "prefill",
        "Token generation:": "decode",
        "Token sampling:": "sampling",
        "E2E generation (entire generation loop):": "e2e",
    }

    @classmethod
    def parse(cls, lines: list[str]) -> "OgaHeadline":
        out = cls()
        cur_block: Optional[dict] = None

        # Pre-compile per-row patterns. model_benchmark uses literal tabs
        # before each row, but we strip leading whitespace before matching.
        # The unit token (us/ms/tokens/s) is OPTIONAL because the `n` row
        # has no unit annotation; the non-capturing `(?:...)?` wrapper
        # lets the regex still match those rows and leaves group(2) as
        # None, which the dispatch below interprets as the raw-value path.
        row_re = re.compile(
            r"^(avg|p50|stddev|n)\s*"
            r"(?:\((us|ms|tokens/s)\))?"
            r":\s*"
            r"(.+?)\s*$"
        )

        for line in lines:
            stripped = line.lstrip()

            # Single-line headers and tail rows
            m = re.match(
                r"Batch size:\s*(\d+),\s*prompt tokens:\s*(\d+),"
                r"\s*tokens to generate:\s*(\d+)",
                line,
            )
            if m:
                out.batch_size = int(m.group(1))
                out.prompt_tokens = int(m.group(2))
                out.tokens_to_generate = int(m.group(3))
                cur_block = None
                continue

            m = re.match(r"Peak working set size:\s*(\d+)\s*bytes", line)
            if m:
                out.peak_ws_bytes = int(m.group(1))
                cur_block = None
                continue

            # Block-start headers - always unindented
            if not line.startswith(("\t", " ")):
                field_name = cls._BLOCKS.get(line.rstrip())
                if field_name is not None:
                    cur_block = {}
                    setattr(out, field_name, cur_block)
                    continue
                # Any other unindented line ends the current block
                cur_block = None
                continue

            if cur_block is None:
                continue
            m = row_re.match(stripped)
            if not m:
                continue
            label, unit, value = m.group(1), m.group(2), m.group(3)
            # n is the only non-numeric value (e.g. "3 * 128 token(s)"); keep raw
            if label == "n":
                cur_block["n"] = value
                continue
            # Numeric - normalize the key (avg -> avg_us / avg_tps / avg_ms,
            # depending on the unit annotation). model_benchmark sometimes
            # emits multiple `avg` rows in one block (us + tokens/s).
            if unit == "us":
                cur_block[f"{label}_us"] = float(value)
            elif unit == "ms":
                cur_block[f"{label}_ms"] = float(value)
            elif unit == "tokens/s":
                cur_block[f"{label}_tps"] = float(value)
            else:
                cur_block[label] = float(value)
        return out

    def __bool__(self) -> bool:
        return self.batch_size > 0


@dataclass
class PerOpTable:
    """Last ``[PERF] === ... ===`` block in the log + median GPU TOTAL across run.

    The last block is what users see as "steady-state decode": for a typical
    OGA run of (warmup x W) + (reps x R) x (1 prefill + N decodes), the last
    Compute() bracketed by ``[PERF] ===`` borders is the last decode of the
    last rep -- past warmup, past first-call kernel autotune, past
    pool-grow. The TOTAL gpu (ms) MEDIAN across ALL Computes in the log
    (typically several hundred samples for a transformer decoder run) is
    computed from every TOTAL row and feeds the # 2 breakdown so cold-start
    outliers don't poison the derived "GPU outside OP_PROFILE" residual.
    """

    last_block_lines: list[str] = field(default_factory=list)
    total_gpu_ms_samples: list[float] = field(default_factory=list)

    # TOTAL row format: `[PERF]  TOTAL                          <gpu>      <cpu>`
    # Trailing two columns are intentionally locked: op_profile.cpp may grow
    # or shrink intermediate columns (per-op counts, shape tags, ...) but the
    # last two are always (gpu_ms, cpu_ms). Anchoring on that suffix is
    # cheaper and more robust than chasing column widths or labels.
    _TOTAL_RE = re.compile(r"^\[PERF\]\s+TOTAL\s+([\d.]+)\s+([\d.]+)\s*$")

    @classmethod
    def parse(cls, lines: list[str]) -> "PerOpTable":
        out = cls()
        # Find all border line indices, then slice between the last two.
        border_idx = [i for i, ln in enumerate(lines) if ln.startswith("[PERF] ===")]
        if len(border_idx) >= 2:
            top, bot = border_idx[-2], border_idx[-1]
            out.last_block_lines = lines[top : bot + 1]

        for line in lines:
            m = cls._TOTAL_RE.match(line)
            if m:
                out.total_gpu_ms_samples.append(float(m.group(1)))
        return out

    @property
    def total_gpu_median_ms(self) -> Optional[float]:
        if not self.total_gpu_ms_samples:
            return None
        return statistics.median(self.total_gpu_ms_samples)

    def __bool__(self) -> bool:
        return bool(self.last_block_lines)


# --- derived breakdown for # 2 -----------------------------------------------


@dataclass
class DecodeBreakdown:
    """Row values for # 2 STEADY-STATE DECODE BREAKDOWN.

    All inputs come from independent distributions (OGA self-reports n=381, EP
    [PERF SUMMARY] medians from n~639 mixed prefill+decode, per-op TOTAL median
    from the same n samples). Derived rows are therefore approximate at steady
    state, NOT strict per-call subtractions - this is why every difference
    clamps to >= 0 instead of letting negative residuals propagate.
    """

    oga_p50_ms: float
    sampling_avg_ms: float
    wall_ms: float
    marshal_in_ms: float
    marshal_out_ms: float
    compute_cpu_ms: float
    gpu_ms: float
    fence_ms: float
    perop_gpu_median_ms: float

    @classmethod
    def build(
        cls, head: OgaHeadline, perf: PerfSummary, perop: PerOpTable
    ) -> Optional["DecodeBreakdown"]:
        if not (head.decode and perf and perop.total_gpu_median_ms is not None):
            return None
        oga_p50_us = head.decode.get("p50_us")
        sample_avg_us = head.sampling.get("avg_us") if head.sampling else 0.0
        if oga_p50_us is None:
            return None

        def med(name: str) -> Optional[float]:
            row = perf.metrics.get(name)
            return row.get("median") if row else None

        wall = med("wall_ms")
        cpu = med("compute_cpu_ms")
        gpu = med("gpu_ms")
        if wall is None or cpu is None or gpu is None:
            return None

        return cls(
            oga_p50_ms=oga_p50_us / 1000.0,
            sampling_avg_ms=(sample_avg_us or 0.0) / 1000.0,
            wall_ms=wall,
            marshal_in_ms=med("marshal_in_ms") or 0.0,
            marshal_out_ms=med("marshal_out_ms") or 0.0,
            compute_cpu_ms=cpu,
            gpu_ms=gpu,
            fence_ms=med("fence_residual_ms") or 0.0,
            perop_gpu_median_ms=perop.total_gpu_median_ms,
        )

    # -- derived rows (clamp to >= 0 since these are between-distribution diffs)
    @property
    def oga_overhead_ms(self) -> float:
        return max(0.0, self.oga_p50_ms - self.wall_ms)

    @property
    def other_oga_ms(self) -> float:
        return max(0.0, self.oga_overhead_ms - self.sampling_avg_ms)

    @property
    def unprofiled_gpu_ms(self) -> float:
        return max(0.0, self.gpu_ms - self.perop_gpu_median_ms)

    @property
    def cpu_overhead_ms(self) -> float:
        return max(0.0, self.compute_cpu_ms - self.gpu_ms)


# --- utilities ---------------------------------------------------------------


def human_bytes(b: Optional[int]) -> str:
    if b is None or b < 0:
        return "?"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    val = float(b)
    i = 0
    while val >= 1024 and i < len(units) - 1:
        val /= 1024
        i += 1
    return f"{int(val)} {units[i]}" if i == 0 else f"{val:.2f} {units[i]}"


def model_name_from_log_path(p: Path) -> str:
    """`<model>_<tool>_<shape>_p<L>g<G>_<mode>_<ts>.log` -> `<model>`."""
    stem = p.stem
    for marker in ("_oga_", "_perftest_"):
        idx = stem.find(marker)
        if idx > 0:
            return stem[:idx]
    return stem


def run_params_from_log_path(p: Path) -> dict:
    """Extract tool / shape / prompt / gen / mode tags from the log filename.

    Returns whatever it can decode; absent keys are simply missing. Used to
    populate the identity banner since the bench-wrapper '===' header line is
    NOT redirected into the log file itself.
    """
    out: dict = {}
    stem = p.stem
    for marker, key in (("_oga_", "tool"), ("_perftest_", "tool")):
        if marker in stem:
            out["tool"] = marker.strip("_")
            tail = stem.split(marker, 1)[1]
            parts = tail.split("_")
            # parts: [shape, "p128g128", mode, date, time] (approx)
            if parts:
                out["shape"] = parts[0]
            for part in parts[1:]:
                m = re.match(r"p(\d+)g(\d+)", part)
                if m:
                    out["prompt"] = int(m.group(1))
                    out["gen"] = int(m.group(2))
                elif part in ("none", "perf", "debug"):
                    out["mode"] = part
            break
    return out


# --- renderers ---------------------------------------------------------------


# ASCII tree glyphs for the # 2 call stack (one place to tweak).
TREE = {"branch": "+-", "last": "+-", "pipe": "| ", "blank": "  "}

SECTION_RULE = "-" * 74

# Width of the left column in # 2 (label + tree prefix). Picked so the widest
# label ("model.dll inference_compute()") plus deepest indent (4 levels =
# "|   |   | ") still fits without wrapping in a standard 100-col terminal.
LABEL_WIDTH = 42


def render_banner(
    model: str, run_params: dict, perf: PerfSummary, indent: str
) -> list[str]:
    bar = "=" * (74 + len(indent))
    bits = [f"HIPDNN EP profile  *  {model}"]
    # Show shape only when it's worth flagging. "dynamic" is the default for
    # every OGA log this script handles (run_bench.sh rejects --oga --shape
    # static), and even if the formatter is later wired to perftest dynamic
    # runs the token would still be redundant - readers assume dynamic. Any
    # other value (static, future shape modes) IS noteworthy, so it stays.
    if run_params.get("shape") and run_params["shape"] != "dynamic":
        bits.append(run_params["shape"])
    if "prompt" in run_params and "gen" in run_params:
        bits.append(f"prompt={run_params['prompt']} gen={run_params['gen']}")
    if perf.total_inferences:
        # Match the source field name (`total_inferences=` in [PERF SUMMARY])
        # so anyone who greps either the banner or the raw log finds the same
        # token. "n=" was ambiguous against # 1's per-block "samples" column
        # (3 reps, 381 decode tokens, etc.) - readers couldn't tell whether
        # the banner counted timed-window or whole-run.
        bits.append(f"inferences={perf.total_inferences}")
    return [f"{indent}{bar}", f"{indent}  {'  *  '.join(bits)}", f"{indent}{bar}"]


def render_section_header(
    num: int, title: str, subtitle: str, indent: str
) -> list[str]:
    return [
        f"{indent}  # {num}  {title}  -  {subtitle}",
        f"{indent}  {SECTION_RULE}",
    ]


def render_section_1_headline(head: OgaHeadline, indent: str) -> list[str]:
    """# 1 HEADLINE - single five-row table consolidating model_benchmark's four
    indented blocks (Prompt processing / Token generation / Token sampling /
    E2E / Peak WS). Throughput in tok/s, p50 in ms, stddev in ms, samples raw.

    Source:        ``OgaHeadline`` -> model_benchmark stats block.
    Self-gates:    rendered unconditionally when ``head`` is truthy (caller
                   already skipped the whole report on ``not head``); absent
                   sub-blocks within head render the literal ``(absent)``.
    """
    lines = render_section_header(
        1,
        "HEADLINE",
        f"model_benchmark (batch={head.batch_size}, prompt={head.prompt_tokens}, "
        f"gen={head.tokens_to_generate})",
        indent,
    )
    # Column widths chosen so all five rows align with the header.
    fmt = "    {label:<18} {tps:>13}   {p50:>13}   {std:>11}   {n}"
    lines.append(
        indent
        + fmt.format(
            label="", tps="throughput", p50="p50 latency", std="stddev", n="samples"
        ).rstrip()
    )

    def row(
        label: str, block: Optional[dict], lat_unit: str, *, has_tps: bool = True
    ) -> str:
        if block is None:
            return indent + f"    {label:<18} (absent)"
        tps = (
            f"{block.get('avg_tps', 0):.2f} t/s"
            if has_tps and "avg_tps" in block
            else "-"
        )
        p50_key = f"p50_{lat_unit}"
        p50 = f"{block.get(p50_key, 0):.2f} {lat_unit}" if p50_key in block else "-"
        std_key = f"stddev_{lat_unit}"
        std = f"{block.get(std_key, 0):.2f} {lat_unit}" if std_key in block else "-"
        n = block.get("n", "-")
        return indent + fmt.format(label=label, tps=tps, p50=p50, std=std, n=n).rstrip()

    lines.append(row("Prefill (TTFT)", head.prefill, "us"))
    lines.append(row("Decode", head.decode, "us"))
    lines.append(row("Sampling", head.sampling, "us"))
    lines.append(row("E2E generation", head.e2e, "ms", has_tps=False))
    lines.append(
        indent + f"    {'Peak working set':<18} "
        f"{human_bytes(head.peak_ws_bytes)} "
        f"({head.peak_ws_bytes or '?'} bytes)"
    )
    return lines


def render_section_2_breakdown(bd: DecodeBreakdown, indent: str) -> list[str]:
    """# 2 STEADY-STATE DECODE BREAKDOWN - drawn as an actual tree.

    Source:        ``DecodeBreakdown`` -> joined OgaHeadline + PerfSummary +
                   PerOpTable medians. Each row carries a ``<- # N`` arrow
                   pointing to the section that owns its raw number, so a
                   skeptical reader can always trace a value back one hop.
    Self-gates:    caller ensures ``bd`` is not None (build() returns None
                   when any of the three upstream feeds is missing); this
                   renderer assumes all medians are present.

    Indentation in the tree matches visual depth so readers skimming
    top-to-bottom never have to count whitespace.
    """
    tc = TREE
    arrow = "<-"
    lines = render_section_header(
        2,
        "STEADY-STATE DECODE BREAKDOWN",
        "1 decode token, median across run",
        indent,
    )
    lines.append(
        indent + f"    {'':<{LABEL_WIDTH}} "
        f"{'latency':>10}   {'share':>6}   {'source':<20}"
    )

    def row(prefix: str, label: str, ms: float, share: float, src: str = "") -> str:
        pad = LABEL_WIDTH - len(prefix) - len(label)
        if pad < 1:
            pad = 1
        return (
            f"{indent}    {prefix}{label}{' ' * pad} "
            f"{ms:>7.3f} ms   {share:>5.1f} %   {src}"
        ).rstrip()

    total = bd.oga_p50_ms

    def pct(x: float) -> float:
        return 100.0 * x / total if total else 0.0

    # -- Tree row prefixes (hard-coded because the tree shape is fixed).
    # Each prefix encodes both this row's hook (+- vs +-) and whether each
    # ancestor at every level above had more siblings (|  ) or was the last
    # one (3-space blank). Compare against the visual layout below:
    #
    #   OGA::GenerateNextToken                       <- root, no prefix
    #   +- OGA + ORT framework overhead              <- L1 not-last
    #   |  +- Token sampling                         <- L2 under L1-not-last
    #   |  +- Other (last L2 under L1-not-last)
    #   +- EP MlirCustomOp::Compute()                <- L1 LAST (no pipe below)
    #      +- Marshal in                             <- L2 under L1-LAST
    #      +- Marshal out
    #      +- model.dll inference_compute()
    #      |  +- GPU: OP_PROFILE kernels (sum)       <- L3 under L1-LAST + L2-not-last
    #      |  +- GPU: unwrapped + glue kernels
    #      |  +- CPU: host dispatch + sync poll      <- last L3
    #      +- Trailing fence (hipStreamSync)         <- last L2 under L1-LAST
    PIPE = f"{tc['pipe']} "  # "|  " - parent has more siblings below
    BLANK = "   "  # parent was last - no pipe through this column
    BRANCH = f"{tc['branch']} "  # "+- " - this row has siblings below
    LAST = f"{tc['last']} "  # "+- " - last sibling at this level

    lines.append(
        row(
            "",
            "OGA::GenerateNextToken",
            bd.oga_p50_ms,
            100.0,
            f"{arrow} # 1 Decode p50",
        )
    )
    lines.append(
        row(
            BRANCH,
            "OGA + ORT framework overhead",
            bd.oga_overhead_ms,
            pct(bd.oga_overhead_ms),
        )
    )
    lines.append(
        row(
            PIPE + BRANCH,
            "Token sampling",
            bd.sampling_avg_ms,
            pct(bd.sampling_avg_ms),
            f"{arrow} # 1 Sampling avg",
        )
    )
    lines.append(
        row(
            PIPE + LAST,
            "Other (IoBinding rebind, etc.)",
            bd.other_oga_ms,
            pct(bd.other_oga_ms),
        )
    )
    lines.append(
        row(
            LAST,
            "EP MlirCustomOp::Compute()",
            bd.wall_ms,
            pct(bd.wall_ms),
            f"{arrow} # 4 wall_ms",
        )
    )
    lines.append(
        row(
            BLANK + BRANCH,
            "Marshal in (input descriptors)",
            bd.marshal_in_ms,
            pct(bd.marshal_in_ms),
            f"{arrow} # 4 marshal_in_ms",
        )
    )
    lines.append(
        row(
            BLANK + BRANCH,
            "Marshal out (output desc.)",
            bd.marshal_out_ms,
            pct(bd.marshal_out_ms),
            f"{arrow} # 4 marshal_out_ms",
        )
    )
    lines.append(
        row(
            BLANK + BRANCH,
            "model.dll inference_compute()",
            bd.compute_cpu_ms,
            pct(bd.compute_cpu_ms),
            f"{arrow} # 4 compute_cpu_ms",
        )
    )
    lines.append(
        row(
            BLANK + PIPE + BRANCH,
            "GPU: OP_PROFILE kernels (sum)",
            bd.perop_gpu_median_ms,
            pct(bd.perop_gpu_median_ms),
            f"{arrow} # 3 TOTAL",
        )
    )
    lines.append(
        row(
            BLANK + PIPE + BRANCH,
            "GPU: outside # 3 scopes",
            bd.unprofiled_gpu_ms,
            pct(bd.unprofiled_gpu_ms),
        )
    )
    lines.append(
        row(
            BLANK + PIPE + LAST,
            "CPU: host dispatch + sync poll",
            bd.cpu_overhead_ms,
            pct(bd.cpu_overhead_ms),
        )
    )
    lines.append(
        row(
            BLANK + LAST,
            "Trailing fence (hipStreamSync)",
            bd.fence_ms,
            pct(bd.fence_ms),
            f"{arrow} # 4 fence_residual_ms",
        )
    )

    # Notes for the two "residual" rows that don't have a direct # N cross-ref
    # (they are subtractive - what's left after the named rows are accounted
    # for). Without these notes, first-time readers ask "what's in there?" for
    # both rows. We deliberately do NOT annotate every row - the named ones
    # speak for themselves once you've followed their # N arrow.
    lines.append("")
    lines.append(indent + "    notes:")
    lines.append(
        indent + '      - "GPU: outside # 3 scopes" = # 4 gpu_ms - # 3 '
        "TOTAL: GPU work not bracketed by any"
    )
    lines.append(
        indent + "        OP_PROFILE wrapper. Typical sources: MIOpen / "
        "hipBLASLt internal helper"
    )
    lines.append(
        indent + "        launches, ops lacking an OP_PROFILE scope, "
        "hipMemset / glue between ops. If"
    )
    lines.append(
        indent + "        this row grows after a runtime change, "
        "OP_PROFILE coverage has regressed."
    )
    lines.append(
        indent + '      - "Other (IoBinding rebind, etc.)" = # 1 Decode '
        "p50 - sampling - EP wall: OGA"
    )
    lines.append(
        indent + "        framework overhead between Generator steps. "
        "Dominated by IoBinding's per-token"
    )
    lines.append(
        indent + "        rebind of all input/output tensors (scales "
        "with KV buffer size, not seq len)."
    )
    return lines


def render_section_3_perop(perop: PerOpTable, indent: str) -> list[str]:
    """# 3 PER-OP GPU BREAKDOWN - passes the runtime's own [PERF] table through
    verbatim.

    Source:        ``PerOpTable.last_block_lines`` -> the last
                   ``[PERF] === ... ===`` block emitted by
                   ``op_profile_resolve_and_print`` in
                   ``lib/Runtime/op_profile.cpp``.
    Self-gates:    returns ``[]`` when ``perop`` is empty (the log was run
                   with ``HIPDNN_EP_PERF`` unset or no per-op table was
                   ever printed).

    The op_profile.cpp formatter already aligns columns nicely, so we only
    strip the ``[PERF] `` prefix (the section header tells the reader which
    subsystem these rows came from) and add the section frame.
    """
    if not perop:
        return []
    lines = render_section_header(
        3,
        "PER-OP GPU BREAKDOWN",
        "last Compute() = steady-state decode token",
        indent,
    )
    for line in perop.last_block_lines:
        # Drop the leading `[PERF] ` tag; the section header tells the reader
        # which subsystem these rows came from.
        body = line.removeprefix("[PERF] ")
        lines.append(f"{indent}    {body}")
    return lines


def render_section_4_distribution(perf: PerfSummary, indent: str) -> list[str]:
    """# 4 PER-CALL DISTRIBUTION - six metrics x five stats, column-aligned.

    Source:        ``PerfSummary.metrics`` -> the ``[PERF SUMMARY]`` block
                   emitted by ``PerfCollector::dump_summary`` at DLL unload
                   (see ``backend-mlir-compiler/.../MlirCustomOp.cpp``).
    Self-gates:    returns ``[]`` when ``perf`` is empty (no [PERF SUMMARY]
                   block in the log -- typically a ``--mode none`` run).

    The cold-start note at the bottom is intentional: # 4's ``max`` column
    almost always shows a multi-second outlier (the very first Compute()
    that does kernel autotune + GPU pool first-grow) and this surprises
    readers who haven't seen the report before. Documenting it in-band
    pre-empts the predictable "why is max 1000x the median?" question.
    """
    if not perf:
        return []
    lines = render_section_header(
        4,
        "PER-CALL DISTRIBUTION",
        f"EP MlirCustomOp::Compute() over {perf.total_inferences} invocations (all ms)",
        indent,
    )
    lines.append(
        indent + f"    {'':<18} {'min':>10} {'median':>10} {'p99':>10} {'max':>12}"
    )
    for name in PerfSummary.METRIC_ORDER:
        stats = perf.metrics.get(name)
        if not stats:
            continue
        lines.append(
            indent + f"    {name:<18} "
            f"{stats['min']:>10.3f} {stats['median']:>10.3f}"
            f" {stats['p99']:>10.3f} {stats['max']:>12.3f}"
        )
    # The two-line note below catches the two most-common "wait, what?" moments
    # readers have when first comparing # 4 against # 1:
    #   (a) why is `max` 1000x the median? -> cold start; see also # 2 caveat
    #   (b) why doesn't this count match # 1's "samples" column? -> different
    #       windows: # 1 reports only OGA's timed iterations, # 4 reports every
    #       EP call including warmup + verbose-mode display generation.
    lines.append(indent + "    notes:")
    lines.append(
        indent + "      - `max` includes first-Compute cold start "
        "(kernel autotune + pool grow);"
    )
    lines.append(indent + "        # 2 uses median to skip those outliers.")
    lines.append(
        indent + "      - this count covers ALL EP calls (warmup + "
        "verbose-display + timed reps);"
    )
    lines.append(
        indent + "        # 1's per-block `samples` column is the "
        "OGA-timed subset only."
    )
    return lines


def render_footer(head: OgaHeadline, indent: str) -> list[str]:
    """Single greppable tail line - public contract; keep stable across runs.

    Output format (treat as public API for downstream tooling):

        [OGA] prefill=<P> tok/s  TTFT=<T> ms  decode=<D> tok/s  \\
              peak_WS=<H> (<B> bytes)

    Where <P>/<D> are floats in tokens/s, <T> is a float in ms, <H> is the
    human-readable peak working-set (e.g. "1.34 GiB"), and <B> is the raw
    byte count. Tools may grep for ``^\\[OGA\\] prefill=`` to locate the
    line and parse the named fields. Peak working-set bytes are kept in
    parens for greppability even though the GiB form is shown first.
    """
    if not head:
        return []
    bar = "=" * (74 + len(indent))
    prefill_tps = head.prefill.get("avg_tps") if head.prefill else None
    decode_tps = head.decode.get("avg_tps") if head.decode else None
    # TTFT is the raw prompt-processing wall time -- `prefill=... tok/s` is
    # derived from it (prompt_tokens / TTFT_seconds) but the time itself is
    # the more directly meaningful number when comparing across prompt
    # lengths or against external baselines. Use `avg_us` (matching what
    # `prefill tok/s` is derived from) for internal consistency.
    ttft_us = head.prefill.get("avg_us") if head.prefill else None
    ttft_str = f"{ttft_us / 1000.0:.2f} ms" if ttft_us else "?"
    peak_bytes = head.peak_ws_bytes if head.peak_ws_bytes is not None else "?"
    return [
        f"{indent}{bar}",
        f"{indent}  [OGA] prefill={prefill_tps or '?'} tok/s  "
        f"TTFT={ttft_str}  "
        f"decode={decode_tps or '?'} tok/s  "
        f"peak_WS={human_bytes(head.peak_ws_bytes)} ({peak_bytes} bytes)",
        f"{indent}{bar}",
    ]


# --- main --------------------------------------------------------------------


def render_report(
    log_path: Path,
    *,
    show_banner: bool = True,
    indent: str = "",
) -> str:
    lines = log_path.read_text(errors="replace").splitlines()

    perf = PerfSummary.parse(lines)
    head = OgaHeadline.parse(lines)
    perop = PerOpTable.parse(lines)

    if not head:
        # No model_benchmark stats - likely a failed run. Caller prints a raw
        # tail instead of this report; signal absence by returning empty.
        return ""

    breakdown = DecodeBreakdown.build(head, perf, perop)

    out: list[str] = []
    model = model_name_from_log_path(log_path)
    run_params = run_params_from_log_path(log_path)

    if show_banner:
        out += render_banner(model, run_params, perf, indent)
        out.append("")

    out += render_section_1_headline(head, indent)
    out.append("")

    if breakdown is not None:
        out += render_section_2_breakdown(breakdown, indent)
        out.append("")
    if perop:
        out += render_section_3_perop(perop, indent)
        out.append("")
    if perf:
        out += render_section_4_distribution(perf, indent)
        out.append("")

    if show_banner:
        out += render_footer(head, indent)

    # Trim trailing blank line for cleaner embedding in bash output.
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("log", type=Path, help="Path to a HIPDNN EP / OGA bench log")
    p.add_argument(
        "--no-banner", action="store_true", help="Omit the top/bottom identity banner"
    )
    p.add_argument(
        "--indent",
        type=int,
        default=0,
        help="Indent every output line by N spaces (default: 0)",
    )
    args = p.parse_args(argv)

    if not args.log.is_file():
        print(f"error: log not found: {args.log}", file=sys.stderr)
        return 1

    report = render_report(
        args.log,
        show_banner=not args.no_banner,
        indent=" " * args.indent,
    )
    if not report:
        print(
            f"warning: log has no model_benchmark stats block: {args.log}",
            file=sys.stderr,
        )
        return 2
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
