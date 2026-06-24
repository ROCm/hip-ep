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
    perf_multimodal_report.py <log-file>
    perf_multimodal_report.py <log-file> <csv-file> --reps N   # multimodal CSV headline
    perf_multimodal_report.py <log-file> --no-banner           # CI embed
    perf_multimodal_report.py <log-file> --indent 2            # indent N
"""

from __future__ import annotations

import argparse
import csv as _csv
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

    # Block header > field name. Headers exactly match what model_benchmark prints.
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

            m = re.match(r"Peak working set size \(bytes\):\s*(\d+)", line)
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
            # Numeric - normalize the key (avg > avg_us / avg_tps / avg_ms,
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

    @classmethod
    def from_multimodal_csv(cls, csv_path: Path, reps: int = 1) -> "OgaHeadline":
        """Synthesize a headline from benchmark_multimodal.py's CSV output.

        benchmark_multimodal.py (the Python OGA bench used for vision/audio
        models) emits a CSV instead of model_benchmark's stats block, so its
        logs have no ``Batch size:`` line and ## 1/2/footer would be skipped.
        This maps the CSV's averaged columns onto the same dict shape
        ``parse()`` produces, letting the rest of the renderer run unchanged.

        Limitations (the CSV captures averages only, not per-iter dists):
          * ``p50`` is filled with the average and ``stddev`` with 0.0 -- the
            real per-call distribution lives in # 4 ([PERF SUMMARY]).
          * batch_size is hard-coded to 1 (benchmark_multimodal.py is batch-1).
          * # 2's "OGA + ORT framework overhead" row subtracts the EP wall
            *median* (mixed across vision/embedding/text Computes for a
            multimodal run) from this CSV decode latency, so that row is
            approximate -- see the # 2 notes.
        """
        with csv_path.open(newline="") as f:
            rows = list(_csv.DictReader(f))
        if not rows:
            raise ValueError(f"{csv_path} has no data rows")
        row = rows[-1]

        def num(col: str) -> float:
            # Tolerant column lookup: exact, else first header containing col.
            if col in row:
                return float(row[col])
            for k, v in row.items():
                if col.lower() in k.lower():
                    return float(v)
            raise KeyError(f"column matching {col!r} not found in {csv_path}")

        out = cls()
        out.batch_size = 1
        out.prompt_tokens = int(num("Prompt Length"))
        out.tokens_to_generate = int(num("Tokens Generated"))
        # benchmark_multimodal.py times i in [1 .. gen-1] decode tokens.
        decode_calls = max(out.tokens_to_generate - 1, 1)

        prompt_us = num("Prompt Latency") * 1000.0
        decode_us = num("Token Generation Latency") * 1000.0
        sample_us = num("Sampling Latency") * 1000.0
        wall_ms = num("Wall Clock Time") * 1000.0
        prefill_tps = (out.prompt_tokens / (prompt_us / 1e6)) if prompt_us else 0.0

        out.prefill = {
            "avg_us": prompt_us,
            "p50_us": prompt_us,
            "stddev_us": 0.0,
            "n": f"{reps} * {out.prompt_tokens} token(s)",
            "avg_tps": prefill_tps,
        }
        out.decode = {
            "avg_us": decode_us,
            "p50_us": decode_us,
            "stddev_us": 0.0,
            "n": f"{reps} * {decode_calls} token(s)",
            "avg_tps": num("Token Generation Throughput"),
        }
        out.sampling = {
            "avg_us": sample_us,
            "p50_us": sample_us,
            "stddev_us": 0.0,
            "n": f"{reps} * 1 token(s)",
            "avg_tps": num("Sampling Throughput"),
        }
        out.e2e = {
            "avg_ms": wall_ms,
            "p50_ms": wall_ms,
            "stddev_ms": 0.0,
            "n": f"{reps}",
        }
        out.peak_ws_bytes = int(num("Memory Usage") * 1024**3)
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
    # Every [PERF] === block (one per Compute), so --all-ops can render the
    # vision / embedding / text sub-model tables, not just the last (decode) one.
    all_blocks: list[list[str]] = field(default_factory=list)

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

        # Blocks come in border pairs (open/close); slice each one out so
        # --all-ops can show every Compute (vision / embedding / text).
        for i in range(0, len(border_idx) - 1, 2):
            out.all_blocks.append(lines[border_idx[i] : border_idx[i + 1] + 1])

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
            # Use the SAME (last decode token) block that # 7 prints, so the
            # # 2 row "GPU: OP_PROFILE kernels < # 7 TOTAL" reconciles exactly
            # with # 7's TOTAL instead of being a median across all decode
            # Computes (which differed slightly from the displayed last token).
            perop_gpu_median_ms=_block_total_gpu(perop.last_block_lines),
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


# ASCII tree glyphs for the # 2 / # 3 call stacks (one place to tweak).
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
                   PerOpTable medians. Each row carries a ``< # N`` arrow
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
        3,
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
            f"{arrow} # 6 wall_ms",
        )
    )
    lines.append(
        row(
            BLANK + BRANCH,
            "Marshal in (input descriptors)",
            bd.marshal_in_ms,
            pct(bd.marshal_in_ms),
            f"{arrow} # 6 marshal_in_ms",
        )
    )
    lines.append(
        row(
            BLANK + BRANCH,
            "Marshal out (output desc.)",
            bd.marshal_out_ms,
            pct(bd.marshal_out_ms),
            f"{arrow} # 6 marshal_out_ms",
        )
    )
    lines.append(
        row(
            BLANK + BRANCH,
            "model.dll inference_compute()",
            bd.compute_cpu_ms,
            pct(bd.compute_cpu_ms),
            f"{arrow} # 6 compute_cpu_ms",
        )
    )
    lines.append(
        row(
            BLANK + PIPE + BRANCH,
            "GPU: OP_PROFILE kernels (sum)",
            bd.perop_gpu_median_ms,
            pct(bd.perop_gpu_median_ms),
            f"{arrow} # 7 TOTAL",
        )
    )
    lines.append(
        row(
            BLANK + PIPE + BRANCH,
            "GPU: outside # 7 scopes",
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
            f"{arrow} # 6 fence_residual_ms",
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
        indent + '      - "GPU: outside # 7 scopes" = # 6 gpu_ms - # 7 '
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


# --- per-Compute samples + prefill/decode classification ---------------------

# One line per Compute(), emitted by PerfCollector::record (MlirCustomOp.cpp).
# Carries all six metrics for a SINGLE call -- unlike [PERF SUMMARY], which is
# pre-aggregated across every Compute (prefill + decode + vision + ...) and so
# cannot be re-split by stage. We re-derive per-stage distributions from these.
_PERCALL_RE = re.compile(
    r"\[PERF\] #(\d+) wall=([\d.]+) marshal_in=([\d.]+) marshal_out=([\d.]+) "
    r"compute_cpu=([\d.]+) gpu=([\d.]+) fence_residual=([\d.]+)"
)


@dataclass
class ComputeSample:
    """One Compute() call: its six metrics + its [PERF] === op block + stage."""

    wall_ms: float
    marshal_in_ms: float
    marshal_out_ms: float
    compute_cpu_ms: float
    gpu_ms: float
    fence_residual_ms: float
    block: list[str] = field(default_factory=list)
    stage: str = "prefill"  # "prefill" | "decode"


def _classify_stage(block: list[str]) -> str:
    """prefill vs decode from the op block's attention sequence length.

    Decode processes exactly one new token, so its attention op carries
    ``sq=1,``. Every attention variant the EP emits prints ``sq=`` -- ``gqa``,
    ``multi_head_attention``, and ``linear_attention`` all do -- so the tell is
    attention-kind-agnostic. Prefill (text prompt, vision encoder) runs
    attention over many tokens (``sq=301,``, ``sq=1196,``), and the
    embedding/glue Computes have no single-token attention -- all bucketed as
    prefill. The ``sq=1,`` trailing comma avoids matching ``sq=1196``.

    Do NOT also key decode on ``m=1,`` (matmul_nbits): an ``m=1`` matmul is NOT
    unique to decode. A prefill that slices its lm_head to the LAST token only
    (a common optimization -- logits are needed for just the final prompt
    position to sample the first token) emits ``m=1,n=<vocab>,k=...`` while its
    attention is still ``sq=<prompt_len>``. Keying on ``m=1,`` then misfiled
    that whole text-prefill Compute as decode, zeroing the "Text prefill" row in
    the # 2 stage breakdown. Models whose prefill lm_head is NOT sliced
    (``m=<prompt_len>``) were unaffected, which is why it showed up on some
    models and not others. ``sq=1,`` (true single-token attention) is the only
    reliable autoregressive tell, present in every decode block and no prefill
    block.
    """
    text = "\n".join(block)
    return "decode" if "sq=1," in text else "prefill"


# Prefill sub-model buckets, by op signature (Qwen-VL-shaped; the text-prefill
# / vision tells are fairly general, embedding/glue is a catch-all):
#   vision       -> multi_head_attention (the ViT encoder)
#   text_prefill -> matmul_nbits (the quantized LLM prefill)
#   embedding    -> image-feature scatter (nonzero / scatter_nd)
#   glue         -> everything else (tiny inter-op Computes)
def _prefill_substage(block: list[str]) -> str:
    text = "\n".join(block)
    if "multi_head_attention" in text:
        return "vision"
    if "matmul_nbits" in text:
        return "text_prefill"
    if "nonzero" in text or "scatter_nd" in text:
        return "embedding"
    return "glue"


def parse_compute_samples(lines: list[str]) -> list[ComputeSample]:
    """Pair each ``[PERF] #N`` metric line with its following ``[PERF] ===``
    block (same Compute, printed in order) and classify the stage.

    Pairing is by LINE POSITION -- each metric takes the next op-block that
    *opens after* it -- NOT by sequential index. This matters when a metric
    line is dropped: e.g. a console-wrapped ``[PERF] #1 wall=...`` line whose
    ``fence_residual=`` spilled onto the next physical line no longer matches
    ``_PERCALL_RE``, so it's missing from the metric list. With index pairing
    that single drop shifts EVERY subsequent metric onto the previous Compute's
    block, misclassifying a cold multi-second prefill Compute as a "decode"
    token (and blowing up the # 2 decode share to thousands of percent).
    Positional pairing instead just skips the one orphaned Compute.
    """
    metric_pos = [(i, m) for i, ln in enumerate(lines) if (m := _PERCALL_RE.search(ln))]
    border_idx = [i for i, ln in enumerate(lines) if ln.startswith("[PERF] ===")]
    pairs = [
        (border_idx[j], border_idx[j + 1]) for j in range(0, len(border_idx) - 1, 2)
    ]
    out: list[ComputeSample] = []
    pi = 0
    for pos, m in metric_pos:
        # Advance past any op-block that opened before this metric line (e.g.
        # the very first Compute's block, when its own metric line was dropped).
        while pi < len(pairs) and pairs[pi][0] < pos:
            pi += 1
        blk = lines[pairs[pi][0] : pairs[pi][1] + 1] if pi < len(pairs) else []
        if pi < len(pairs):
            pi += 1
        s = ComputeSample(
            wall_ms=float(m.group(2)),
            marshal_in_ms=float(m.group(3)),
            marshal_out_ms=float(m.group(4)),
            compute_cpu_ms=float(m.group(5)),
            gpu_ms=float(m.group(6)),
            fence_residual_ms=float(m.group(7)),
            block=blk,
        )
        s.stage = _classify_stage(blk)
        out.append(s)
    return out


def stage_perf(samples: list[ComputeSample]) -> PerfSummary:
    """Build a PerfSummary (min/mean/median/p99/max per metric) from a list of
    Computes -- the per-stage analog of the EP's all-Computes [PERF SUMMARY]."""
    ps = PerfSummary()
    ps.total_inferences = len(samples)
    for name in PerfSummary.METRIC_ORDER:
        vals = sorted(getattr(s, name) for s in samples)
        if not vals:
            continue
        n = len(vals)
        ps.metrics[name] = {
            "min": vals[0],
            "max": vals[-1],
            "mean": sum(vals) / n,
            "median": statistics.median(vals),
            "p99": vals[max(0, int(round(0.99 * (n - 1))))],
        }
    return ps


# Op rows in a [PERF] === block. Parent rows are indented 2 spaces after the
# tag, shape sub-rows 4 spaces; both end in (calls, gpu, cpu[, gpu%]). The
# label is captured lazily because shape labels can contain spaces (e.g.
# "n=8192 rank=3"). TOTAL/header rows don't match (no integer `calls` column).
_OP_PARENT_RE = re.compile(r"^\[PERF\]  (\S.*?)\s+(\d+)\s+(\S+)\s+(\S+)(?:\s+\S+)?\s*$")
_OP_SHAPE_RE = re.compile(
    r"^\[PERF\]    (\S.*?)\s+(\d+)\s+(\S+)\s+(\S+)(?:\s+\S+)?\s*$"
)


def _accum(slot: list, gpu_tok: str, cpu_tok: str) -> None:
    for idx, tok in ((1, gpu_tok), (2, cpu_tok)):
        if tok != "n/a":
            try:
                slot[idx] += float(tok)
            except ValueError:
                pass


_PREFILL_SUBS = ("vision", "embedding", "text_prefill", "glue")
_PREFILL_METRICS = ("wall", "mi", "mo", "cpu", "gpu", "fence", "optot")


def _segment_prefill_generations(samples: list[ComputeSample]) -> list[dict]:
    """Segment prefill Computes into generations -- a new generation begins at
    each vision-substage Compute (the ViT runs once per rep) -- summing each
    sub-stage's metrics per generation.

    Shared by the # 3 waterfall and the # 5 per-generation normalization so
    their generation counts can NEVER disagree. (They previously could: when a
    first metric line is dropped/wrapped, the first parsed prefill Compute may
    be non-vision, which opens a leading generation that a naive vision-count
    would miss -- off-by-one against # 3.)
    """

    def blank() -> dict:
        return {sub: {m: 0.0 for m in _PREFILL_METRICS} for sub in _PREFILL_SUBS}

    gens: list[dict] = []
    cur: Optional[dict] = None
    for s in samples:
        sub = _prefill_substage(s.block) if s.stage == "prefill" else None
        if sub == "vision":
            if cur is not None:
                gens.append(cur)
            cur = blank()
        if s.stage == "prefill":
            if cur is None:
                cur = blank()
            d = cur[sub]
            d["wall"] += s.wall_ms
            d["mi"] += s.marshal_in_ms
            d["mo"] += s.marshal_out_ms
            d["cpu"] += s.compute_cpu_ms
            d["gpu"] += s.gpu_ms
            d["fence"] += s.fence_residual_ms
            d["optot"] += _block_total_gpu(s.block)
    if cur is not None and any(d["wall"] for d in cur.values()):
        gens.append(cur)
    return gens


def _count_prefill_generations(samples: list[ComputeSample]) -> int:
    """Generation count from the shared segmentation (>= 1)."""
    return len(_segment_prefill_generations(samples)) or 1


def render_perop_aggregate(
    blocks: list[list[str]],
    indent: str,
    *,
    num: int,
    subtitle: str,
    generations: int = 1,
) -> list[str]:
    """Sum each op's gpu/cpu/calls across every block in a stage, keeping the
    op -> shape hierarchy so the discriminating shapes stay visible.

    Used for the PREFILL per-op view: a stage's prefill work spans several
    Computes (vision encoder + embedding + text prefill), so a single block is
    not representative -- the sum is the true "where did TTFT go". The shape
    sub-rows are preserved (e.g. matmul_nbits -> m=301,... for text prefill)
    so the stage is identifiable. Cold-start/autotune is part of prefill and
    intentionally included.
    """
    # name -> [calls, gpu, cpu, {shape: [calls, gpu, cpu]}]
    agg: dict[str, list] = {}
    for blk in blocks:
        cur: Optional[list] = None
        for ln in blk:
            mp = _OP_PARENT_RE.match(ln)
            if mp and mp.group(1).strip() != "TOTAL":
                cur = agg.setdefault(mp.group(1).strip(), [0, 0.0, 0.0, {}])
                cur[0] += int(mp.group(2))
                _accum(cur, mp.group(3), mp.group(4))
                continue
            ms = _OP_SHAPE_RE.match(ln)
            if ms and cur is not None:
                e = cur[3].setdefault(ms.group(1).strip(), [0, 0.0, 0.0])
                e[0] += int(ms.group(2))
                _accum(e, ms.group(3), ms.group(4))
    if not agg:
        return []
    # Normalize GPU/CPU ms to PER GENERATION so this table's TOTAL reconciles
    # with the per-generation # 3 waterfall that references it (# 3 is one
    # generation; the raw sum here spans every rep). `calls` stays a run total
    # (the exact launch count); only the time columns are divided.
    div = generations if generations and generations > 0 else 1
    title = (
        "PER-OP GPU BREAKDOWN (per generation)"
        if div > 1
        else "PER-OP GPU BREAKDOWN (stage total)"
    )
    lines = render_section_header(num, title, subtitle, indent)
    tot_gpu = sum(v[1] for v in agg.values())
    tot_cpu = sum(v[2] for v in agg.values())
    lines.append(
        indent + f"    {'op / shape':<40}{'calls':>7}{'gpu (ms)':>10}"
        f"{'cpu (ms)':>10}{'gpu %':>7}"
    )
    for name, (calls, gpu, cpu, shapes) in sorted(
        agg.items(), key=lambda kv: kv[1][1], reverse=True
    ):
        pct = (100.0 * gpu / tot_gpu) if tot_gpu else 0.0
        lines.append(
            indent + f"    {name:<40}{calls:>7}{gpu / div:>10.1f}"
            f"{cpu / div:>10.1f}{pct:>6.1f}%"
        )
        for shp, (scalls, sgpu, scpu) in sorted(
            shapes.items(), key=lambda kv: kv[1][1], reverse=True
        ):
            spct = (100.0 * sgpu / tot_gpu) if tot_gpu else 0.0
            lines.append(
                indent + f"      {shp:<38}{scalls:>7}{sgpu / div:>10.1f}"
                f"{scpu / div:>10.1f}{spct:>6.1f}%"
            )
    lines.append(
        indent + f"    {'TOTAL':<40}{'':>7}{tot_gpu / div:>10.1f}{tot_cpu / div:>10.1f}"
    )
    return lines


def stage_perop(samples: list[ComputeSample]) -> PerOpTable:
    """Build a PerOpTable for a stage: representative (last) block + the
    stage's TOTAL gpu samples (feeds # 2's GPU-median residual)."""
    pop = PerOpTable()
    blocks = [s.block for s in samples if s.block]
    if blocks:
        pop.last_block_lines = blocks[-1]
        pop.all_blocks = blocks
    for blk in blocks:
        for ln in blk:
            m = PerOpTable._TOTAL_RE.match(ln)
            if m:
                pop.total_gpu_ms_samples.append(float(m.group(1)))
    return pop


def _block_total_gpu(block: list[str]) -> float:
    """The op_profile TOTAL gpu (ms) for one [PERF] === block, else 0."""
    for ln in block:
        m = PerOpTable._TOTAL_RE.match(ln)
        if m:
            return float(m.group(1))
    return 0.0


def render_prefill_breakdown(
    samples: list[ComputeSample], head: OgaHeadline, indent: str, *, num: int
) -> list[str]:
    """Prefill waterfall: split TTFT's EP work into vision / embedding /
    text-prefill / glue, and give the two big sub-models (vision, text-prefill)
    a # 2-style call stack (marshal -> model.dll compute [GPU op_profile / GPU
    outside / CPU dispatch] -> fence).

    All values are the MEDIAN across generations so the one cold warmup
    generation drops out (needs several reps -- e.g. -r 10 -- to be meaningful).
    Generations are segmented at each vision Compute (every rep re-runs the
    full vision->embedding->text-prefill->decode pipeline); per generation we
    sum each sub-stage's metrics, then take the median generation.

    Rows sum to the median per-generation EP prefill wall, NOT to TTFT:
    benchmark_multimodal.py books text prefill under its own "sampling" phase,
    so its phase boundaries don't map 1:1 onto these EP sub-stages. TTFT is a
    reference line only.
    """
    prefill = [s for s in samples if s.stage == "prefill"]
    if not prefill:
        return []

    subs = _PREFILL_SUBS
    metrics = _PREFILL_METRICS
    gens = _segment_prefill_generations(samples)
    if not gens:
        return []

    med = {
        sub: {m: statistics.median([g[sub][m] for g in gens]) for m in metrics}
        for sub in subs
    }
    total = sum(med[sub]["wall"] for sub in subs)
    if total <= 0:
        return []

    tc = TREE
    arrow = "<-"
    PIPE, BLANK, BRANCH, LAST = (
        f"{tc['pipe']} ",
        "   ",
        f"{tc['branch']} ",
        f"{tc['last']} ",
    )
    lines = render_section_header(
        num,
        "PREFILL STAGE BREAKDOWN",
        f"median of {len(gens)} generation(s); EP wall per sub-model + call stack",
        indent,
    )
    lines.append(
        indent + f"    {'':<{LABEL_WIDTH}} {'latency':>10}   {'share':>6}   source"
    )

    def row(prefix: str, label: str, ms: float, src: str = "") -> str:
        pad = max(1, LABEL_WIDTH - len(prefix) - len(label))
        share = 100.0 * ms / total if total else 0.0
        return (
            f"{indent}    {prefix}{label}{' ' * pad} "
            f"{ms:>7.3f} ms   {share:>5.1f} %   {src}"
        ).rstrip()

    def callstack(sub: str, is_last: bool) -> list[str]:
        d = med[sub]
        hook = LAST if is_last else BRANCH
        cont = BLANK if is_last else PIPE
        gpu_outside = max(0.0, d["gpu"] - d["optot"])
        cpu_dispatch = max(0.0, d["cpu"] - d["gpu"])
        return [
            row(hook, _SUBLABEL[sub], d["wall"]),
            row(cont + BRANCH, "Marshal in + out", d["mi"] + d["mo"]),
            row(cont + BRANCH, "model.dll compute()", d["cpu"]),
            row(
                cont + PIPE + BRANCH,
                "GPU: OP_PROFILE kernels",
                d["optot"],
                f"{arrow} # 5 per-op",
            ),
            row(cont + PIPE + BRANCH, "GPU: outside scopes", gpu_outside),
            row(cont + PIPE + LAST, "CPU: host dispatch", cpu_dispatch),
            row(cont + LAST, "Trailing fence", d["fence"]),
        ]

    lines.append(row("", "EP prefill wall (per generation)", total))
    # Two big sub-models get a call stack (largest wall first); embedding+glue
    # are tiny -> one combined leaf, always last.
    big = sorted(("vision", "text_prefill"), key=lambda s: med[s]["wall"], reverse=True)
    for sub in big:
        lines += callstack(sub, is_last=False)
    eg = med["embedding"]["wall"] + med["glue"]["wall"]
    lines.append(row(LAST, "Embedding + glue (image-feature scatter, etc.)", eg))

    ttft_us = head.prefill.get("avg_us") if (head and head.prefill) else None
    if ttft_us:
        lines.append("")
        lines.append(
            indent + f"    reference: CSV TTFT = {ttft_us / 1000.0:.1f} ms "
            "(OGA end-to-end prompt latency)."
        )
        lines.append(
            indent + "    NOTE: rows sum to EP prefill wall, not TTFT --"
            " benchmark_multimodal.py books text"
        )
        lines.append(
            indent + "    prefill under its 'sampling' phase, so TTFT's"
            " phase split differs from these EP sub-stages."
        )
    return lines


_SUBLABEL = {
    "vision": "Vision encoder (ViT)",
    "text_prefill": "Text prefill (LLM prompt)",
}


def render_section_3_perop(
    perop: PerOpTable,
    indent: str,
    *,
    num: int = 3,
    subtitle: str = "last Compute() = steady-state decode token",
) -> list[str]:
    """PER-OP GPU BREAKDOWN - passes the runtime's own [PERF] table through
    verbatim.

    Source:        ``PerOpTable.last_block_lines`` -> the last
                   ``[PERF] === ... ===`` block emitted by
                   ``op_profile_resolve_and_print`` in
                   ``lib/Runtime/op_profile.cpp``.
    Self-gates:    returns ``[]`` when ``perop`` is empty (the log was run
                   with ``HIPDNN_EP_PERF`` unset or no per-op table was
                   ever printed).

    ``num`` / ``subtitle`` let the caller render one of these per stage
    (prefill / decode) with a stage-specific header.

    The op_profile.cpp formatter already aligns columns nicely, so we only
    strip the ``[PERF] `` prefix (the section header tells the reader which
    subsystem these rows came from) and add the section frame.
    """
    if not perop:
        return []
    lines = render_section_header(num, "PER-OP GPU BREAKDOWN", subtitle, indent)
    blocks = [perop.last_block_lines]
    for block in blocks:
        for line in block:
            # Drop the leading `[PERF] ` tag; the section header tells the reader
            # which subsystem these rows came from.
            body = line.removeprefix("[PERF] ")
            lines.append(f"{indent}    {body}")
    return lines


def render_section_4_distribution(
    perf: PerfSummary, indent: str, *, num: int = 4, subtitle: Optional[str] = None
) -> list[str]:
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
        num,
        "PER-CALL DISTRIBUTION",
        subtitle
        if subtitle is not None
        else f"EP MlirCustomOp::Compute() over {perf.total_inferences} invocations (all ms)",
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


def _read_log_text(log_path: Path) -> str:
    """Decode a log robustly. PowerShell `Tee-Object` writes UTF-16LE (with
    BOM); plain redirects/our runner write UTF-8. Detect the BOM (or a high
    NUL ratio for BOM-less UTF-16) so we never silently produce garbage."""
    raw = log_path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    if raw[:3] == b"\xef\xbb\xbf":
        return raw.decode("utf-8-sig")
    if raw.count(b"\x00") > len(raw) // 4:
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")


def render_report(
    log_path: Path,
    *,
    show_banner: bool = True,
    indent: str = "",
    csv_path: Optional[Path] = None,
    reps: int = 1,
) -> str:
    lines = _read_log_text(log_path).splitlines()

    head = OgaHeadline.parse(lines)

    # When a benchmark_multimodal.py CSV is supplied and the log itself has no
    # model_benchmark stats block, synthesize the headline from the CSV so
    # ## 1/2/footer (TTFT/TPS + OGA/ORT flow overhead) render. A real
    # model_benchmark block in the log always wins.
    if csv_path is not None and not head:
        head = OgaHeadline.from_multimodal_csv(csv_path, reps=reps)

    # Split every Compute() into prefill vs decode, then recompute the per-call
    # distribution + per-op table PER STAGE. This is the fix for mixing: the
    # EP's single [PERF SUMMARY] pools prefill (huge, with JIT/vision) and
    # decode (small) into one median, so # 2's "EP wall" was wrong. Re-deriving
    # decode-only medians from the raw [PERF] #N lines makes the OGA/ORT
    # overhead (# 2) honest.
    samples = parse_compute_samples(lines)
    prefill = [s for s in samples if s.stage == "prefill"]
    decode = [s for s in samples if s.stage == "decode"]

    if not (head or samples):
        # Nothing parseable - likely a failed run. Caller prints a raw tail.
        return ""

    # # 2 uses the DECODE-stage wall median (not the all-Compute mix).
    dec_perf = stage_perf(decode)
    dec_perop = stage_perop(decode)
    breakdown = DecodeBreakdown.build(head, dec_perf, dec_perop) if decode else None

    out: list[str] = []
    model = model_name_from_log_path(log_path)
    run_params = run_params_from_log_path(log_path)

    if show_banner:
        out += render_banner(model, run_params, stage_perf(samples), indent)
        out.append("")

    if head:
        out += render_section_1_headline(head, indent)
        out.append("")

    # # 2 PREFILL breakdown -- chronologically first (prefill precedes decode).
    if prefill:
        wf = render_prefill_breakdown(samples, head, indent, num=2)
        if wf:
            out += wf
            out.append("")

    # # 3 DECODE breakdown.
    if breakdown is not None:
        out += render_section_2_breakdown(breakdown, indent)
        out.append("")

    # # 4/5 PREFILL dist + per-op, # 6/7 DECODE dist + per-op -- each stage
    # from its own Computes only.
    if prefill:
        out += render_section_4_distribution(
            stage_perf(prefill),
            indent,
            num=4,
            subtitle=f"PREFILL stage -- {len(prefill)} Compute(s) "
            "(vision + embedding + text prefill), all ms",
        )
        out.append("")
        # Prefill spans multiple Computes per generation -> sum per-op across
        # them, then normalize to per-generation so the TOTAL matches # 2.
        n_gen = _count_prefill_generations(samples)
        out += render_perop_aggregate(
            [s.block for s in prefill if s.block],
            indent,
            num=5,
            generations=n_gen,
            subtitle=f"gpu/cpu per generation (avg over {n_gen} gen(s)); "
            f"calls = run total over {len(prefill)} prefill Compute(s)",
        )
        out.append("")
    if decode:
        out += render_section_4_distribution(
            dec_perf,
            indent,
            num=6,
            subtitle=f"DECODE stage -- {len(decode)} Compute(s) "
            "(steady-state tokens), all ms",
        )
        out.append("")
        # Decode is per-token steady state -> show the last (clean) token, not
        # a sum (which would fold in the cold first-token LM-head autotune).
        out += render_section_3_perop(
            dec_perop,
            indent,
            num=7,
            subtitle="representative = last decode token (steady state)",
        )
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
        "csv",
        type=Path,
        nargs="?",
        default=None,
        help="Optional benchmark_multimodal.py CSV. When the log has "
        "no model_benchmark stats block (e.g. multimodal Python "
        "runs), the headline (# 1 TTFT/TPS), # 2 OGA/ORT flow "
        "overhead, and footer are synthesized from this CSV.",
    )
    p.add_argument(
        "--reps",
        type=int,
        default=1,
        help="Repetitions used in the bench (matches "
        "benchmark_multimodal.py -r); only labels the CSV-derived "
        "`samples` column (default: 1)",
    )
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
    if args.csv is not None and not args.csv.is_file():
        print(f"error: csv not found: {args.csv}", file=sys.stderr)
        return 1

    report = render_report(
        args.log,
        show_banner=not args.no_banner,
        indent=" " * args.indent,
        csv_path=args.csv,
        reps=args.reps,
    )
    if not report:
        print(
            f"warning: log has no parseable EP perf streams: {args.log}",
            file=sys.stderr,
        )
        return 2
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
