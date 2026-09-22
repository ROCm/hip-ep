#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Render the compatibility report from report_input.json.

The report leads with a worklist rather than a distribution table. Knowing
that an operator is unsupported is only useful alongside what to do about
it, and the four non-supported statuses call for four different things --
writing an operator, relaxing a restriction in one, adding a lowering, or
teaching a converter an attribute.

Reads only report_input.json (plus the optional distribution comparison);
every verdict was decided upstream.

Usage:
  python generate_final_reports.py <analysis_dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

WORKLIST_SECTIONS = [
    (
        "unsupported",
        "Implement the operator",
        "No implementation exists anywhere in the tree.",
    ),
    (
        "blocked",
        "Extend an existing operator",
        "An implementation exists but rejects this model's variant. Usually a "
        "dtype, shape or attribute restriction in the converter.",
    ),
    (
        "lowering-broken",
        "Complete the lowering chain",
        "The front end converts the operator but the back end does not follow. "
        "Missing HipToLLVM lowering or runtime function.",
    ),
    (
        "partial",
        "Handle an ignored attribute",
        "Conversion succeeds, but the converter never reads an attribute this "
        "model sets, so the generated code silently means something else.",
    ),
]

EVIDENCE_BADGE = {
    "A": None,
    "B": "Whole-graph probe failed; results come from per-operator slices and "
    "under-report support, since fusions that need surrounding context cannot fire.",
    "C": "Whole-graph MLIR was unavailable; results come from single-operator "
    "models built from the original ONNX.",
    "D": "**Support was read from documentation only.** No compilation happened, "
    "so operators whose implementation rejects this model's variant are reported "
    "as supported. Treat the numbers as an upper bound.",
}


def fmt(value, dash: str = "—") -> str:
    if value is None or value == "" or value == []:
        return dash
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def render_summary(summary: dict) -> list[str]:
    inst = summary["instances"]
    types = summary["operator_types"]
    out = [
        "## Summary\n\n",
        f"- Total node instances: {summary['total_node_instances']}",
    ]
    excluded = summary.get("weight_constants_excluded", 0)
    if excluded:
        out.append(f" (excluding {excluded} weight constants)")
    out.append("\n")
    out.append(
        f"- Supported: **{inst['supported']} ({summary['supported_pct']}%)** "
        f"across {types['supported']} operator types\n"
    )
    for key, label, _ in WORKLIST_SECTIONS:
        out.append(
            f"- {label.split()[0]} ({key}): {inst[key]} instances, "
            f"{types[key]} operator types\n"
        )
    out.append("\n")
    return out


def render_worklist(worklist: dict) -> list[str]:
    total = sum(len(v) for v in worklist.values())
    out = ["## What needs doing\n\n"]
    if not total:
        out.append("Nothing: every operator in this model is fully supported.\n\n")
        return out

    for key, label, why in WORKLIST_SECTIONS:
        items = worklist.get(key) or []
        if not items:
            continue
        out.append(
            f"### {label} — {len(items)} operator(s), "
            f"{sum(i['count'] for i in items)} instances\n\n"
        )
        out.append(f"{why}\n\n")
        for item in items:
            dom = f" ({item['domain']})" if item["domain"] != "onnx" else ""
            out.append(f"**{item['onnx_op']}**{dom} — {item['count']} instances\n\n")
            if item.get("signature"):
                out.append(f"- Signature: `{item['signature']}`\n")
            if item.get("data_types"):
                out.append(f"- Data types: {fmt(item['data_types'])}\n")
            if item.get("shape_types"):
                out.append(f"- Shapes: {fmt(item['shape_types'])}\n")
            if item.get("attributes"):
                attrs = ", ".join(f"{k}={v}" for k, v in item["attributes"].items())
                out.append(f"- Attributes: `{attrs}`\n")
            if item.get("implementation"):
                out.append(f"- Existing implementation: {item['implementation']}\n")
            if item.get("blocking_operand"):
                out.append(f"- **Why it fails**: {item['blocking_operand']}\n")
            if item.get("ignored_attributes"):
                names = ", ".join(
                    f"`{a['name']}={a['value']}`" for a in item["ignored_attributes"]
                )
                out.append(f"- Ignored by the converter: {names}\n")
            if item.get("error"):
                out.append(f"- Error: `{item['error']}`\n")
            if item.get("source_line"):
                out.append(f"- EP input MLIR line: {item['source_line']}\n")
            out.append("- Converter source: _to be filled in_\n")
            out.append("- Root cause: _to be filled in_\n\n")
    return out


def render_distribution(rows: list[dict]) -> list[str]:
    out = [
        "## Operator distribution\n\n",
        "| Op Type | Domain | Count | Data Types | Shapes | Status | Target | "
        "Backend (runtime) | Description |\n",
        "|---|---|---:|---|---|---|---|---|---|\n",
    ]
    for r in rows:
        out.append(
            f"| {r['onnx_op']} | {r['domain']} | {r['count']} | "
            f"{fmt(r.get('data_types'))} | {fmt(r.get('shape_types'))} | "
            f"{r['status']} | {fmt(r.get('target'))} | {fmt(r.get('backend'))} | "
            f"{fmt(r.get('op_description'))} |\n"
        )
    out.append("\n")
    return out


def render_capability_gaps(gaps: list[dict]) -> list[str]:
    if not gaps:
        return []
    out = [
        "## Capability gaps\n\n",
        "Attributes the converter does not read. Proven by changing the value "
        "and re-converting: the output was byte-for-byte identical.\n\n",
        "| Op Type | Target | Attribute | Value in this model | Set by the model |\n",
        "|---|---|---|---|---|\n",
    ]
    for g in gaps:
        out.append(
            f"| {g['onnx_op']} | {fmt(g.get('target'))} | `{g['attribute']}` | "
            f"`{fmt(g.get('value'))}` | {'yes' if g.get('set_by_model') else 'no'} |\n"
        )
    out.append(
        "\nA gap with `set_by_model = no` does not affect this model -- the value "
        "came from ORT filling in a schema default -- but a model that sets it "
        "would be miscompiled without warning.\n\n"
    )
    return out


def render_doc_stale(stale: list[dict]) -> list[str]:
    if not stale:
        return []
    out = [
        "## Documentation drift\n\n",
        "Operators that compile but are absent from `docs/supported-operations.md`:\n\n",
    ]
    for s in stale:
        dom = f" ({s['domain']})" if s["domain"] != "onnx" else ""
        out.append(f"- `{s['onnx_op']}`{dom} lowers to {fmt(s.get('target'))}\n")
    out.append("\n")
    return out


def render_comparison(comp: dict) -> list[str]:
    """Full operator distribution on both sides of the EP's rewrites.

    Every operator is listed, not just the ones whose count moved: the point
    is to see the distribution, with the rewrites visible inside it.
    """
    summary = comp.get("summary", {})
    orig_types = summary.get("original_unique_ops", 0)
    ep_types = summary.get("ep_unique_ops", 0)
    out = [
        "## Original ONNX vs EP input\n\n",
        "The EP rewrites the graph before compiling it. Compatibility above "
        "describes the EP input; this is how it differs from the model as "
        "authored.\n\n",
        "| Metric | Original | EP input | Delta |\n|---|---:|---:|---:|\n",
        f"| Node instances | {summary.get('original_total_nodes', 0)} | "
        f"{summary.get('ep_total_nodes', 0)} | {summary.get('node_delta', 0):+d} |\n",
        f"| Operator types | {orig_types} | {ep_types} | {ep_types - orig_types:+d} |\n\n",
    ]

    only_orig = summary.get("only_in_original") or []
    only_ep = summary.get("only_in_ep") or []
    if only_orig:
        out.append(
            "Removed by the rewrites: "
            + ", ".join(f"`{x}`" for x in only_orig)
            + "\n\n"
        )
    if only_ep:
        out.append(
            "Introduced by the rewrites: "
            + ", ".join(f"`{x}`" for x in only_ep)
            + "\n\n"
        )

    out.append("| Op Type | Original | EP input | Delta |\n|---|---:|---:|---:|\n")
    for r in comp.get("rows", []):
        d = r.get("delta", 0)
        mark = " **+**" if d > 0 else (" **-**" if d < 0 else "")
        out.append(
            f"| {r['op_type']} | {r['original_count']} | {r['ep_count']} | "
            f"{d:+d}{mark} |\n"
        )
    out.append("\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the compatibility report")
    ap.add_argument("analysis_dir", type=Path)
    args = ap.parse_args()

    data = json.loads(
        (args.analysis_dir / "report_input.json").read_text(encoding="utf-8")
    )
    meta, summary = data["meta"], data["summary"]

    comp = None
    for candidate in (
        args.analysis_dir / "op_distribution_comparison.json",
        args.analysis_dir.parent / "op_distribution_comparison.json",
    ):
        if candidate.is_file():
            comp = json.loads(candidate.read_text(encoding="utf-8"))
            break

    lines = ["# Model compatibility report\n\n"]
    if meta.get("model_path"):
        lines.append(f"- Model: `{meta['model_path']}`\n")
    lines.append(f"- EP input: `{meta.get('ep_input_path', '')}`\n")
    lines.append(f"- Generated: `{meta['generated_at_utc']}`\n")
    lines.append(f"- Evidence level: **{meta.get('evidence_level', 'A')}**\n\n")

    badge = EVIDENCE_BADGE.get(meta.get("evidence_level", "A"))
    if badge:
        lines.append(f"> {badge}\n\n")
    if meta.get("evidence_note"):
        lines.append(f"> {meta['evidence_note']}\n\n")

    lines += render_summary(summary)
    lines += render_worklist(data.get("worklist", {}))
    if comp:
        lines += render_comparison(comp)
    lines += render_distribution(data.get("operator_distribution", []))
    lines += render_capability_gaps(data.get("capability_gaps", []))
    lines += render_doc_stale(data.get("doc_stale", []))
    lines.append("Per-operator detail is in `model_compatibility_details.md`.\n")

    report = args.analysis_dir / "model_compatibility_report.md"
    report.write_text("".join(lines), encoding="utf-8")

    # Details: the full per-operator record, including what the report omits.
    det = ["# Model compatibility details\n\n"]
    det.append(f"- EP input: `{meta.get('ep_input_path', '')}`\n")
    det.append(f"- Generated: `{meta['generated_at_utc']}`\n\n")
    det.append("## Every operator\n\n")
    det.append(
        "| Op Type | Domain | Count | Status | Target | Runtime | Data Types | "
        "Ignored Attributes |\n|---|---|---:|---|---|---|---|---|\n"
    )
    for r in data.get("operator_distribution", []):
        det.append(
            f"| {r['onnx_op']} | {r['domain']} | {r['count']} | {r['status']} | "
            f"{fmt(r.get('target'))} | {fmt(r.get('runtime_func'))} | "
            f"{fmt(r.get('data_types'))} | {fmt(r.get('ignored_attributes'))} |\n"
        )
    det.append("\n")
    details = args.analysis_dir / "model_compatibility_details.md"
    details.write_text("".join(det), encoding="utf-8")

    print(f"[OK] {report}")
    print(f"[OK] {details}")


if __name__ == "__main__":
    main()
