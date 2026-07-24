#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Shrink huge MorphiZen / ONNX MLIR for viewers (Netron, diff, editor).

1. Per-channel ``!quant.uniform<...>{...}`` brace lists → compact placeholders
   (same behavior as the frontend noconstargs tool).
2. ``onnx.Constant`` (and similar) ``dense<...>`` attribute payloads → placeholders.

Output is for human viewing / diffing only; elided regions are not valid MLIR.
Processes the input line-by-line so multi-GB files need not fit in RAM.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_QUANT_BRACE_RE = re.compile(r"(!quant\.uniform<[^<>{}]*)\{([^{}]*)\}")


def _count_values(body: str) -> int:
    body = body.strip()
    if not body:
        return 0
    return body.count(",") + 1


def strip_quant_brace_lists(text: str) -> tuple[str, int, int]:
    stats = {"lists": 0, "values": 0}

    def repl(m: re.Match[str]) -> str:
        header, body = m.group(1), m.group(2)
        n = _count_values(body)
        stats["lists"] += 1
        stats["values"] += n
        return f"{header}{{...{n} values elided...}}"

    stripped = _QUANT_BRACE_RE.sub(repl, text)
    return stripped, stats["lists"], stats["values"]


def _find_dense_payload_end(line: str, payload_start: int) -> int:
    """Return index after closing ``>`` of ``dense<...>``, or -1."""
    if payload_start >= len(line):
        return -1
    ch = line[payload_start]
    if ch == '"':
        end_quote = line.find('"', payload_start + 1)
        if end_quote == -1:
            return -1
        if end_quote + 1 >= len(line) or line[end_quote + 1] != ">":
            return -1
        return end_quote + 2
    if ch == "[":
        depth = 1
        i = payload_start + 1
        while i < len(line) and depth:
            c = line[i]
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
            i += 1
        if i >= len(line) or line[i] != ">":
            return -1
        return i + 1
    gt = line.find(">", payload_start)
    if gt == -1:
        return -1
    return gt + 1


def shrink_dense_attributes(line: str) -> tuple[str, int, int]:
    """Replace each ``dense<...>`` payload on the line with a placeholder."""
    needle = "dense<"
    out: list[str] = []
    pos = 0
    replacements = 0
    bytes_elided = 0

    while True:
        idx = line.find(needle, pos)
        if idx == -1:
            out.append(line[pos:])
            break
        payload_start = idx + len(needle)
        end = _find_dense_payload_end(line, payload_start)
        if end == -1:
            out.append(line[pos : payload_start + 1])
            pos = payload_start + 1
            continue
        payload_len = end - payload_start - 1  # exclude closing >
        out.append(line[pos:idx])
        out.append(f'dense<"...{payload_len} chars elided...">')
        bytes_elided += max(payload_len, 0)
        replacements += 1
        pos = end

    return "".join(out), replacements, bytes_elided


def process_line(line: str) -> tuple[str, dict[str, int]]:
    stats = {
        "quant_lists": 0,
        "quant_values": 0,
        "dense_repls": 0,
        "dense_chars": 0,
    }
    line, ql, qv = strip_quant_brace_lists(line)
    stats["quant_lists"] = ql
    stats["quant_values"] = qv
    line, dr, dc = shrink_dense_attributes(line)
    stats["dense_repls"] = dr
    stats["dense_chars"] = dc
    return line, stats


def default_output_path(input_path: Path) -> Path:
    name = input_path.name
    if name.endswith(".frontend.mlir"):
        base = name[: -len(".frontend.mlir")]
        return input_path.with_name(f"{base}.noconstargs.frontend.mlir")
    stem = input_path.stem
    return input_path.with_name(f"{stem}.view.mlir")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="input .mlir file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output path (default: <stem>.view.mlir or "
        "<base>.noconstargs.frontend.mlir next to input)",
    )
    args = parser.parse_args(argv)

    if not args.input.is_file():
        parser.error(f"input file not found: {args.input}")

    out_path = args.output or default_output_path(args.input)
    if out_path.resolve() == args.input.resolve():
        parser.error("refusing to overwrite the input file; choose a different -o")

    in_bytes = args.input.stat().st_size
    totals = {
        "quant_lists": 0,
        "quant_values": 0,
        "dense_repls": 0,
        "dense_chars": 0,
        "lines": 0,
    }
    out_bytes = 0

    with (
        args.input.open("r", encoding="utf-8", errors="replace") as fin,
        out_path.open("w", encoding="utf-8", newline="\n") as fout,
    ):
        for line in fin:
            totals["lines"] += 1
            new_line, st = process_line(line)
            for k in ("quant_lists", "quant_values", "dense_repls", "dense_chars"):
                totals[k] += st[k]
            fout.write(new_line)
            out_bytes += len(new_line.encode("utf-8"))

    print(f"input : {args.input}  ({in_bytes:,} bytes)")
    print(f"output: {out_path}  ({out_bytes:,} bytes)")
    print(
        f"quant: {totals['quant_lists']} list(s), "
        f"{totals['quant_values']:,} value(s) elided"
    )
    print(
        f"dense: {totals['dense_repls']:,} attribute(s), "
        f"~{totals['dense_chars']:,} payload char(s) elided"
    )
    print(f"removed ~{max(in_bytes - out_bytes, 0):,} bytes (approx.)")
    print("NOTE: output is for viewing/diff only — not valid MLIR for compilation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
