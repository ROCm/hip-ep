#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Explain why each leftover operator did not convert.

The conversion probe proves *that* an operator stayed an onnx op. It cannot say
why: converters report their refusals through `notifyMatchFailure`, which is
compiled out of a release build, so no reason reaches the log.

This step separates the two cases that read very differently in a report:

  no converter exists          the operator is genuinely unimplemented
  a converter exists           it is implemented but refused these instances,
                               which usually means one dtype, rank or operand
                               outside what it accepts

For the second case it lists the refusals the converter can emit, ordered so
that constraints the observed operand types violate come first, each with a
file and line to confirm. The ordering is a hint, not a verdict: the agent
confirms the actual constraint with the playbook in diagnose.md.
"""

import argparse
import json
import re
from pathlib import Path

# Element-type spellings that appear both in MLIR type signatures and in
# converter messages ("Expected i8 (packed uint4) or f16").
_TYPE_TOKENS = [
    "bf16",
    "f16",
    "f32",
    "f64",
    "i1",
    "i8",
    "i16",
    "i32",
    "i64",
    "ui8",
    "ui16",
    "ui32",
    "ui64",
]

_STRING_LITERAL_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
_NOTIFY_RE = re.compile(r"notifyMatchFailure\s*\(")
# How far above a notifyMatchFailure call its message may be built. Messages
# assembled through raw_string_ostream span several lines.
_MESSAGE_LOOKBEHIND = 14


def find_converter_files(conversion_dir: Path, op_type: str):
    """Source files that match this operator by name.

    Converters select an operator with a string literal, either the op name
    ("onnx.Cast") or, for onnx.Custom containers, the function name
    ("MatMulNBits"), so only quoted occurrences count. A bare substring would
    match unrelated identifiers.
    """
    wanted = {f'"{op_type}"', f'"onnx.{op_type}"'}
    hits = []
    for path in sorted(conversion_dir.rglob("*.cpp")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if any(token in text for token in wanted):
            hits.append((path, text))
    return hits


def clean_message(message: str) -> str:
    """Tidy a message whose dynamic parts (a printed type) are not available."""
    message = re.sub(r"\s+", " ", message).strip()
    # "unsupported element type: " + <printed type> + ". Expected ..." leaves a
    # colon with nothing after it.
    message = re.sub(r"\s*:\s*(?=[.,])", "", message)
    return message.strip(" :")


def extract_refusals(path: Path, text: str):
    """Messages the file can pass to notifyMatchFailure, with line numbers."""
    lines = text.splitlines()
    refusals = []
    previous_notify = -1
    for index, line in enumerate(lines):
        if not _NOTIFY_RE.search(line):
            continue
        # Stay inside this message: never reach back past the previous refusal,
        # whose literals belong to it.
        start = max(previous_notify + 1, index - _MESSAGE_LOOKBEHIND, 0)
        previous_notify = index
        pieces = []
        for window_line in lines[start : index + 2]:
            # Skip comments so explanatory prose is not read as a message.
            if window_line.strip().startswith("//"):
                continue
            pieces.extend(_STRING_LITERAL_RE.findall(window_line))
        message = clean_message(" ".join(p.strip() for p in pieces if p.strip()))
        if message:
            refusals.append(
                {
                    "message": message,
                    "location": f"{path.name}:{index + 1}",
                }
            )
    return refusals


def observed_element_types(entry):
    """Element types seen on the leftover instances' operands and results."""
    found = []
    for sample in entry.get("samples") or []:
        for body in re.findall(r"tensor<([^<>]*)>", sample.get("type_signature", "")):
            elem = body.rsplit("x", 1)[-1].strip() if "x" in body else body.strip()
            if elem and elem not in found:
                found.append(elem)
    return found


def rank_refusals(refusals, observed):
    """Put refusals the observed types violate first.

    A message that lists accepted types is a likely cause when an observed type
    is missing from that list, and an unlikely one when every observed type is
    covered.
    """

    def score(refusal):
        message = refusal["message"]
        mentioned = [t for t in _TYPE_TOKENS if re.search(rf"\b{t}\b", message)]
        if not mentioned:
            return 0
        violating = [t for t in observed if t not in mentioned]
        return 2 if violating else 1

    ordered = sorted(refusals, key=lambda r: -score(r))
    for refusal in ordered:
        refusal["likely"] = score(refusal) == 2
    return ordered


def explain(entry, conversion_dir: Path, max_refusals: int):
    op_type = entry.get("op_type", "")
    files = find_converter_files(conversion_dir, op_type)
    observed = observed_element_types(entry)

    identity = {
        "key": entry.get("key"),
        "op_type": op_type,
        "domain": entry.get("domain", ""),
    }

    if not files:
        return {
            **identity,
            "converter_found": False,
            "observed_element_types": observed,
            "converter_files": [],
            "refusals": [],
        }

    refusals = []
    for path, text in files:
        refusals.extend(extract_refusals(path, text))
    refusals = rank_refusals(refusals, observed)[:max_refusals]

    return {
        **identity,
        "converter_found": True,
        "observed_element_types": observed,
        "converter_files": [path.name for path, _ in files],
        "refusals": refusals,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("leftover_json", help="leftover_onnx.json from analyze_conversion")
    ap.add_argument("repo_root", help="hip-ep repository root")
    ap.add_argument("output_dir", help="Directory for leftover_reasons.json")
    ap.add_argument(
        "--max-refusals",
        type=int,
        default=6,
        help="Candidate constraints to keep per operator (default 6)",
    )
    args = ap.parse_args()

    leftovers = json.loads(Path(args.leftover_json).read_text(encoding="utf-8"))
    conversion_dir = Path(args.repo_root) / "lib" / "Conversion" / "OnnxToHip"
    if not conversion_dir.is_dir():
        raise SystemExit(f"Conversion directory not found: {conversion_dir}")

    rows = [
        explain(entry, conversion_dir, args.max_refusals)
        for entry in leftovers.get("unconverted") or []
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "leftover_reasons.json"
    out_path.write_text(
        json.dumps(
            {"conversion_dir": str(conversion_dir), "rows": rows},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"[OK] {out_path}")
    for row in rows:
        if not row["converter_found"]:
            print(f"  {row['key']}: no converter in {conversion_dir.name}")
            continue
        likely = [r for r in row["refusals"] if r.get("likely")]
        head = (
            likely[0] if likely else (row["refusals"][0] if row["refusals"] else None)
        )
        detail = f"{head['message']} ({head['location']})" if head else "unknown"
        print(f"  {row['key']}: converter exists, rejected all; likely: {detail}")


if __name__ == "__main__":
    main()
