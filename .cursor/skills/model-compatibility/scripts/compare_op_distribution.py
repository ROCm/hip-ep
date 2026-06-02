#!/usr/bin/env python3
"""
Compare step1 ONNX op distributions (original model vs EP-dumped onnx.onnx).

Usage:
  python compare_op_distribution.py <original_step1.json> <ep_step1.json> <output_dir>
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple


def load_counts(step1_path: Path) -> Tuple[Dict[str, int], int]:
    data = json.loads(step1_path.read_text(encoding="utf-8"))
    counts: Dict[str, int] = {}
    total = 0
    for op_type, info in data.items():
        if op_type.startswith("_") or not isinstance(info, dict):
            continue
        c = int(info.get("count", 0))
        counts[op_type] = c
        total += c
    return counts, total


def build_comparison(
    original_path: Path,
    ep_path: Path,
    original_model: str,
    ep_model: str,
) -> dict:
    orig_counts, orig_total = load_counts(original_path)
    ep_counts, ep_total = load_counts(ep_path)

    all_ops = sorted(set(orig_counts) | set(ep_counts))
    rows = []
    only_original = []
    only_ep = []
    changed = []

    for op in all_ops:
        o = orig_counts.get(op, 0)
        e = ep_counts.get(op, 0)
        delta = e - o
        row = {
            "op_type": op,
            "original_count": o,
            "ep_count": e,
            "delta": delta,
        }
        rows.append(row)
        if o > 0 and e == 0:
            only_original.append(op)
        elif e > 0 and o == 0:
            only_ep.append(op)
        elif delta != 0:
            changed.append(row)

    rows.sort(key=lambda r: (-max(r["original_count"], r["ep_count"]), r["op_type"]))

    return {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "original_model": original_model,
            "ep_model": ep_model,
            "original_step1": str(original_path),
            "ep_step1": str(ep_path),
        },
        "summary": {
            "original_total_nodes": orig_total,
            "ep_total_nodes": ep_total,
            "node_delta": ep_total - orig_total,
            "original_unique_ops": len(orig_counts),
            "ep_unique_ops": len(ep_counts),
            "only_in_original": only_original,
            "only_in_ep": only_ep,
            "count_changed_ops": len(changed),
        },
        "rows": rows,
        "changed": changed,
    }


def write_markdown(comp: dict, out_md: Path) -> None:
    meta = comp["meta"]
    summary = comp["summary"]
    lines = [
        "# Original vs EP input — operator distribution comparison\n\n",
        f"- **Original model:** `{meta['original_model']}`\n",
        f"- **EP input (dumped):** `{meta['ep_model']}`\n",
        f"- **Generated UTC:** `{meta['generated_at_utc']}`\n\n",
        "## Summary\n\n",
        f"| Metric | Original | EP input | Delta |\n",
        f"|---|---:|---:|---:|\n",
        f"| Total node instances | {summary['original_total_nodes']} | "
        f"{summary['ep_total_nodes']} | {summary['node_delta']:+d} |\n",
        f"| Unique operator types | {summary['original_unique_ops']} | "
        f"{summary['ep_unique_ops']} | "
        f"{summary['ep_unique_ops'] - summary['original_unique_ops']:+d} |\n\n",
    ]

    if summary["only_in_original"]:
        lines.append("### Operators only in original\n\n")
        for op in summary["only_in_original"]:
            lines.append(f"- `{op}`\n")
        lines.append("\n")

    if summary["only_in_ep"]:
        lines.append("### Operators only in EP input\n\n")
        for op in summary["only_in_ep"]:
            lines.append(f"- `{op}`\n")
        lines.append("\n")

    lines.append("## Full distribution\n\n")
    lines.append("| Op Type | Original | EP input | Delta |\n")
    lines.append("|---|---:|---:|---:|\n")
    for row in comp["rows"]:
        d = row["delta"]
        mark = ""
        if d > 0:
            mark = " **+**"
        elif d < 0:
            mark = " **-**"
        lines.append(
            f"| {row['op_type']} | {row['original_count']} | {row['ep_count']} | {d:+d}{mark} |\n"
        )

    if comp["changed"]:
        lines.append("\n## Operators with count changes (both present)\n\n")
        lines.append("| Op Type | Original | EP input | Delta |\n")
        lines.append("|---|---:|---:|---:|\n")
        for row in comp["changed"]:
            if row["original_count"] == 0 or row["ep_count"] == 0:
                continue
            lines.append(
                f"| {row['op_type']} | {row['original_count']} | {row['ep_count']} | {row['delta']:+d} |\n"
            )

    lines.append(
        "\n---\n\n"
        "Compatibility analysis (step2–final) uses **EP input** (`onnx.onnx`) as the graph seen by the EP.\n"
    )
    out_md.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare step1 op distributions")
    ap.add_argument("original_step1", type=Path)
    ap.add_argument("ep_step1", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--original-model", default="", help="Label for original model path")
    ap.add_argument("--ep-model", default="", help="Label for EP onnx path")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    orig_label = args.original_model or str(args.original_step1.parent)
    ep_label = args.ep_model or str(args.ep_step1.parent)

    comp = build_comparison(args.original_step1, args.ep_step1, orig_label, ep_label)

    json_path = args.output_dir / "op_distribution_comparison.json"
    md_path = args.output_dir / "op_distribution_comparison.md"
    json_path.write_text(json.dumps(comp, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(comp, md_path)
    print(f"[OK] {json_path}")
    print(f"[OK] {md_path}")


if __name__ == "__main__":
    main()
