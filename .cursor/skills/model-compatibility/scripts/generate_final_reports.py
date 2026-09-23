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
    (
        "import-blocked",
        "Extend the MorphiZen ONNX importer",
        "The EP turned the operator away before any conversion ran, so no HIP "
        "operator would help. The work is in the importer, not in this tree.",
    ),
    (
        "unverified",
        "Check the lowering by hand",
        "Conversion succeeds and the operator is counted as supported, but the "
        "probe could not build a standalone module to check the rest of the "
        "chain. Read the finding and decide whether it needs looking into.",
    ),
]

# Headline for a degraded run. The assembler's note carries the specific
# cause and is appended, so the two do not repeat each other.
EVIDENCE_BADGE = {
    "A": None,
    "B": "**Whole-graph conversion failed**, so each operator was probed in "
    "isolation. Support is a lower bound.",
    "C": "**Whole-graph MLIR was unavailable**, so results come from "
    "single-operator models built from the original ONNX.",
    "D": "**Nothing was compiled**, so support was read from documentation "
    "alone. Operators whose implementation rejects this model read as "
    "supported: treat the numbers as an upper bound.",
}


def fmt(value, dash: str = "—") -> str:
    if value is None or value == "" or value == []:
        return dash
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def render_stages(stages: list[dict]) -> list[str]:
    """Which steps ran, and what any failure said.

    The evidence level says results are degraded; this says which step
    degraded them and why, which is the part that suggests a fix.
    """
    if not stages:
        return []
    out = [
        "## Pipeline\n\n",
        "| Stage | Result | Detail |\n|---|---|---|\n",
    ]
    for s in stages:
        result = s.get("result", "")
        mark = "**failed**" if result == "failed" else result
        out.append(f"| {s.get('stage', '')} | {mark} | {fmt(s.get('detail'), '')} |\n")
    out.append("\n")
    return out


# Whether the model compiles as a whole is a separate question from how many
# of its operators are supported, and the evidence level already answers it.
# Saying only the percentage invites reading a high one as "this model runs".
WHOLE_MODEL_VERDICT = {
    "A": "compiles as a whole",
    "B": "**does not compile as a whole**; the figures below are per operator",
    "C": "**cannot be imported as a whole**; the figures below are per operator "
    "type, not a verdict on the model",
    "D": "**was never compiled**; nothing below was verified",
}


def render_summary(summary: dict, evidence: str = "A") -> list[str]:
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
    # The unverified count sits inside the supported figure rather than
    # beside it, because those operators did convert. Stating it here stops
    # the headline from claiming more was checked than was.
    unverified = summary.get("lowering_unverified_instances", 0)
    n_types = summary.get("lowering_unverified_types", 0)
    caveat = (
        f" — {unverified} of them, in {n_types} operator "
        f"type{'s' if n_types != 1 else ''}, were not checked end to end"
        if unverified
        else ""
    )
    verdict = WHOLE_MODEL_VERDICT.get(evidence)
    if verdict:
        out.append(f"- The model {verdict}\n")
    out.append(
        f"- Supported: **{inst['supported']} ({summary['supported_pct']}%)** "
        f"across {types['supported']} operator types{caveat}\n"
    )
    for key, label, _ in WORKLIST_SECTIONS:
        if key not in inst:
            continue
        out.append(
            f"- {label.split()[0]} ({key}): {inst[key]} instances, "
            f"{types[key]} operator types\n"
        )
    # Import-blocked operators are already counted above, as blocked or
    # unsupported depending on the documentation. Saying so here keeps the
    # reader from planning HIP work for them.
    blocked_at_import = summary.get("import_blocked_instances", 0)
    if blocked_at_import:
        n = summary.get("import_blocked_types", 0)
        out.append(
            f"- Of those, {blocked_at_import} instances in {n} operator "
            f"type{'s' if n != 1 else ''} never reached the compiler: the "
            "importer rejected them, so the work is on the importer\n"
        )
    out.append("\n")
    return out


STATUS_ORDER = ["supported", "partial", "lowering-broken", "blocked", "unsupported"]

STATUS_HEADING = {
    "supported": ("Supported", ""),
    "partial": ("Partial", "Converts, but ignores an attribute this model sets."),
    "lowering-broken": (
        "Lowering broken",
        "Converts, but the chain does not reach a runtime symbol.",
    ),
    "blocked": ("Blocked", "Implemented, but this variant is rejected."),
    "unsupported": ("Unsupported", "No implementation in the tree."),
}


def _one_liner(row: dict) -> str:
    """What to say about an operator in a grouped list."""
    if row["status"] == "supported":
        target = row.get("target") or "—"
        if row.get("lowering_unverified"):
            why = row.get("lowering_unverified_reason") or "could not be isolated"
            return f"{target} — converts; lowering unverified ({why})"
        if row.get("runtime_func"):
            return f"{target} → `{row['runtime_func']}`"
        # A folded operator's target already says so; do not say it twice.
        if row.get("compile_time") and "compile time" not in target:
            return f"{target} (compile-time)"
        return target
    if row.get("ignored_attributes"):
        return "converter ignores " + ", ".join(
            f"`{a}`" for a in row["ignored_attributes"]
        )
    return row.get("backend") or "—"


def render_compatibility_summary(rows: list[dict]) -> list[str]:
    """Operators grouped by status, one line each.

    The distribution table holds more per operator but is hard to scan; this
    answers "what is in each bucket" directly.
    """
    by_status: dict[str, list[dict]] = {}
    for r in rows:
        by_status.setdefault(r["status"], []).append(r)

    out = ["### Compatibility summary\n\n"]
    for status in STATUS_ORDER:
        group = by_status.get(status)
        if not group:
            continue
        instances = sum(r["count"] for r in group)
        heading, note = STATUS_HEADING[status]
        types = f"{len(group)} operator type" + ("s" if len(group) != 1 else "")
        out.append(f"#### {heading} ({types}, {instances} instances)\n\n")
        if note:
            out.append(f"{note}\n\n")
        for r in sorted(group, key=lambda x: -x["count"]):
            dom = f" ({r['domain']})" if r["domain"] != "onnx" else ""
            out.append(f"- `{r['onnx_op']}`{dom} ×{r['count']} — {_one_liner(r)}\n")
        out.append("\n")
    return out


def _finding(item: dict) -> str:
    """One sentence saying why this operator is not simply supported."""
    if item.get("blocking_operand"):
        return item["blocking_operand"]
    if item.get("ignored_attributes"):
        return "converter ignores " + ", ".join(
            f"`{a['name']}={a['value']}`" for a in item["ignored_attributes"]
        )
    if item.get("error"):
        return f"`{item['error']}`"
    return "no implementation in the tree"


def render_worklist(worklist: dict) -> list[str]:
    """One row per operator needing work, grouped by the kind of work.

    A table rather than a section each: lowering failures vary too much in
    shape to give every one its own heading, and the supporting evidence
    reads better collected in the details file than scattered here.
    """
    total = sum(len(v) for v in worklist.values())
    out = ["## What needs doing\n\n"]
    if not total:
        out.append("Nothing: every operator in this model is fully supported.\n\n")
        return out

    out.append("| Operator | Instances | Work | Finding |\n|---|---:|---|---|\n")
    for key, label, _why in WORKLIST_SECTIONS:
        for item in worklist.get(key) or []:
            dom = f" ({item['domain']})" if item["domain"] != "onnx" else ""
            out.append(
                f"| {item['onnx_op']}{dom} | {item['count']} | {label} | "
                f"{_finding(item)} |\n"
            )
    out.append("\n")

    for key, label, why in WORKLIST_SECTIONS:
        if worklist.get(key):
            out.append(f"- **{label}** — {why}\n")
    out.append(
        "\nSignatures, attributes and errors per operator are in "
        "`model_compatibility_details.md`.\n\n"
    )
    return out


def render_distribution(rows: list[dict]) -> list[str]:
    out = [
        "## Operator distribution\n\n",
        "| Op Type | Domain | Count | Data Types | Shapes | Status | Target | "
        "Backend (runtime) | Description |\n",
        "|---|---|---:|---|---|---|---|---|---|\n",
    ]
    for r in rows:
        status = r["status"]
        if r.get("lowering_unverified"):
            status += " (lowering unverified)"
        out.append(
            f"| {r['onnx_op']} | {r['domain']} | {r['count']} | "
            f"{fmt(r.get('data_types'))} | {fmt(r.get('shape_types'))} | "
            f"{status} | {fmt(r.get('target'))} | {fmt(r.get('backend'))} | "
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
    ep_input = meta.get("ep_input_path", "")
    if ep_input.endswith(".mlir"):
        lines.append(f"- EP input: `{ep_input}`\n")
    else:
        # No dump, so the original ONNX is what was analyzed. Calling that
        # the EP input would misdescribe it.
        lines.append("- EP input: not available; analyzed the original ONNX\n")
    lines.append(f"- Generated: `{meta['generated_at_utc']}`\n")
    lines.append(f"- Evidence level: **{meta.get('evidence_level', 'A')}**\n\n")

    badge = EVIDENCE_BADGE.get(meta.get("evidence_level", "A"))
    note = meta.get("evidence_note") or ""
    if badge or note:
        lines.append("> " + " ".join(x for x in (badge, note) if x) + "\n\n")

    lines += render_stages(data.get("pipeline_stages", []))
    lines += render_summary(summary, meta.get("evidence_level", "A"))
    if comp:
        lines += render_comparison(comp)
    lines += render_distribution(data.get("operator_distribution", []))
    lines += render_compatibility_summary(data.get("operator_distribution", []))
    lines += render_worklist(data.get("worklist", {}))
    lines += render_capability_gaps(data.get("capability_gaps", []))
    lines += render_doc_stale(data.get("doc_stale", []))
    lines.append("Per-operator detail is in `model_compatibility_details.md`.\n")

    report = args.analysis_dir / "model_compatibility_report.md"
    report.write_text("".join(lines), encoding="utf-8")

    # Details: the full per-operator record, including what the report omits.
    det = ["# Model compatibility details\n\n"]
    det.append(
        f"- EP input: `{ep_input}`\n"
        if ep_input.endswith(".mlir")
        else "- EP input: not available; analyzed the original ONNX\n"
    )
    det.append(f"- Generated: `{meta['generated_at_utc']}`\n\n")

    worklist = data.get("worklist", {})
    if any(worklist.values()):
        det.append("## Evidence per operator needing work\n\n")
        for key, label, _why in WORKLIST_SECTIONS:
            for item in worklist.get(key) or []:
                dom = f" ({item['domain']})" if item["domain"] != "onnx" else ""
                det.append(
                    f"### {item['onnx_op']}{dom} — {item['count']} instances, "
                    f"{label.lower()}\n\n"
                )
                if item.get("signature"):
                    det.append(f"- Signature: `{item['signature']}`\n")
                if item.get("data_types"):
                    det.append(f"- Data types: {fmt(item['data_types'])}\n")
                if item.get("shape_types"):
                    det.append(f"- Shapes: {fmt(item['shape_types'])}\n")
                if item.get("attributes"):
                    attrs = ", ".join(f"{k}={v}" for k, v in item["attributes"].items())
                    det.append(f"- Attributes: `{attrs}`\n")
                if item.get("implementation"):
                    det.append(
                        f"- Documented implementation: {item['implementation']}\n"
                    )
                if item.get("blocking_operand"):
                    det.append(f"- Blocking operand: {item['blocking_operand']}\n")
                if item.get("ignored_attributes"):
                    names = ", ".join(
                        f"`{a['name']}={a['value']}`"
                        for a in item["ignored_attributes"]
                    )
                    det.append(f"- Ignored by the converter: {names}\n")
                if item.get("error"):
                    det.append(f"- Error: `{item['error']}`\n")
                if item.get("source_line"):
                    det.append(f"- EP input MLIR line: {item['source_line']}\n")
                det.append("\n")

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
