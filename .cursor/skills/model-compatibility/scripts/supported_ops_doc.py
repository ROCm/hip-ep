#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Parse docs/supported-operations.md into a lookup table.

The doc is the project's curated list of ONNX operations the ONNX-to-HIP
pipeline handles, together with how each one is implemented (vendor library,
custom HIP kernel, decomposition, or a compile-time tensor transform).

Two consumers:

- The compatibility probe uses *presence* in this table to tell `blocked`
  (no conversion happened, but an implementation exists -> extend it) apart
  from `unsupported` (no implementation at all -> write a new operator).
- The report renderer uses the implementation text as the backend attribution
  column.

Operator name and domain form the key: `RotaryEmbedding` and `Attention` each
appear twice with different domains.

Usage:
  python supported_ops_doc.py <repo_root>/docs/supported-operations.md [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# A markdown table row; we only need the first two cells. The
# compiler-optimized table has a third "Notes" cell that we drop.
_ROW = re.compile(r"^\|([^|]+)\|([^|]+)\|")
_SECTION = re.compile(r"^##\s+(.*?)\s*$")
# Domain marker inside the operator cell: (`com.microsoft`) or
# (`ai.onnx`, opset 23/24). Requires a dot so it cannot match prose.
_DOMAIN = re.compile(r"\(\s*`?([a-z][a-z0-9]*(?:\.[a-z0-9]+)+)`?")
# Everything from the first parenthesis onward is qualifier, not name.
_QUALIFIER = re.compile(r"\s*\(.*$")


def _norm_domain(domain: str) -> str:
    return "onnx" if domain in ("", "ai.onnx") else domain


def _impl_text(cell: str) -> str:
    # Backticks stay: they are markdown the report renders as-is, and a cell
    # can hold several inline-code spans.
    return cell.strip()


def load_supported_ops(doc_path: Path) -> dict[tuple[str, str], dict]:
    """Return {(op, domain): {"impl": str, "section": str}}."""
    table: dict[tuple[str, str], dict] = {}
    section = ""
    for line in doc_path.read_text(encoding="utf-8").splitlines():
        sec = _SECTION.match(line)
        if sec:
            section = sec.group(1)
            continue
        row = _ROW.match(line.strip())
        if not row:
            continue
        op_cell, impl_cell = row.group(1).strip(), row.group(2).strip()
        # Skip the header and the |---|---| separator.
        if op_cell in ("Operation", "") or set(op_cell) <= set("-: "):
            continue
        dm = _DOMAIN.search(op_cell)
        domain = _norm_domain(dm.group(1) if dm else "")
        op = _QUALIFIER.sub("", op_cell).strip().strip("`")
        if not op:
            continue
        table[(op, domain)] = {"impl": _impl_text(impl_cell), "section": section}
    return table


def default_doc_path(repo_root: Path) -> Path:
    return repo_root / "docs" / "supported-operations.md"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("doc_path", type=Path)
    ap.add_argument("--json", type=Path, default=None, help="Write the table as JSON.")
    args = ap.parse_args()

    table = load_supported_ops(args.doc_path)

    by_section: dict[str, int] = {}
    by_domain: dict[str, int] = {}
    for (_op, domain), meta in table.items():
        by_section[meta["section"]] = by_section.get(meta["section"], 0) + 1
        by_domain[domain] = by_domain.get(domain, 0) + 1

    print(f"entries: {len(table)}")
    print("by section:")
    for sec, n in by_section.items():
        print(f"   {n:4d}  {sec}")
    print("by domain:")
    for dom, n in sorted(by_domain.items(), key=lambda x: -x[1]):
        print(f"   {n:4d}  {dom}")

    dupes = [op for op, _ in table if sum(1 for o, _ in table if o == op) > 1]
    if dupes:
        print(f"operators present under multiple domains: {sorted(set(dupes))}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {f"{op}|{dom}": meta for (op, dom), meta in table.items()},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
