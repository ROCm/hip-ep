#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
import json
import sys
import re
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_op_distribution_comparison(analysis_dir: Path):
    """Load op_distribution_comparison.json from analysis_dir or its parent (EP pipeline layout)."""
    candidates = [
        analysis_dir / "op_distribution_comparison.json",
        analysis_dir.parent / "op_distribution_comparison.json",
    ]
    for path in candidates:
        if path.is_file():
            try:
                return read_json(path)
            except Exception:
                continue
    return None


def render_op_distribution_comparison_section(comp: dict) -> list:
    """Why the analyzed graph differs from the packaged ONNX.

    This is the only rendering of the comparison: op_distribution_comparison
    stays a JSON input, and the details report points here instead of
    repeating the table.
    """
    meta = comp.get("meta") or {}
    summary = comp.get("summary") or {}
    rows = comp.get("rows") or []
    only_orig = summary.get("only_in_original") or []
    only_ep = summary.get("only_in_ep") or []

    out = ["## Original vs EP input\n\n"]
    out.append(
        "Compatibility is analysed on the **EP input** (`ep_input.mlir`), the "
        "graph hip-ep receives from ONNX Runtime once its own optimizations "
        "have run. Initializers are carrier ops in that form and are excluded "
        "from its counts.\n\n"
    )
    out.append(f"- **Original model:** `{meta.get('original_model', '—')}`\n")
    out.append(f"- **EP input (analyzed):** `{meta.get('ep_model', '—')}`\n\n")

    out.append("| Metric | Original | EP input | Delta |\n")
    out.append("|---|---:|---:|---:|\n")
    out.append(
        f"| Total node instances | {summary.get('original_total_nodes', 0)} | "
        f"{summary.get('ep_total_nodes', 0)} | {summary.get('node_delta', 0):+d} |\n"
    )
    original_types = int(summary.get("original_unique_ops", 0))
    ep_types = int(summary.get("ep_unique_ops", 0))
    out.append(
        f"| Unique operator types | {original_types} | {ep_types} | "
        f"{ep_types - original_types:+d} |\n\n"
    )

    if only_orig:
        out.append(
            "**Operators only in original:** "
            + ", ".join(f"`{x}`" for x in only_orig)
            + "\n\n"
        )
    if only_ep:
        out.append(
            "**Operators only in the EP input:** "
            + ", ".join(f"`{x}`" for x in only_ep)
            + "\n\n"
        )

    out.append("| Op Type | Original | EP input | Delta |\n")
    out.append("|---|---:|---:|---:|\n")
    for row in rows:
        delta = int(row.get("delta", 0))
        mark = " **+**" if delta > 0 else (" **-**" if delta < 0 else "")
        out.append(
            f"| {row.get('op_type', '')} | {row.get('original_count', 0)} | "
            f"{row.get('ep_count', 0)} | {delta:+d}{mark} |\n"
        )
    out.append("\n")
    return out


def status_display(status: str) -> str:
    return "supported" if status == "full" else status


def compile_time_reason(reason_texts):
    terms = [
        "compile time",
        "compile-time",
        "constant folding",
        "shape inference",
        "shape manipulation",
        "type-canonicalization",
    ]
    merged = " ".join(reason_texts or []).lower()
    return any(t in merged for t in terms)


def load_reco_rules(script_dir: Path):
    cfg = script_dir / "unsupported_reco_rules.json"
    if cfg.exists():
        try:
            return read_json(cfg)
        except Exception:
            pass
    return {}


def compile_time_reason_with_rules(reason_texts, rules):
    terms = (
        ((rules or {}).get("compile_time_rule") or {}).get("when_reason_contains_any")
    ) or []
    if not terms:
        return compile_time_reason(reason_texts)
    merged = " ".join(reason_texts or []).lower()
    return any(str(t).lower() in merged for t in terms)


def format_reco(path: str, target: str):
    p = (path or "").strip()
    t = (target or "").strip()
    if not p:
        return "Custom Hip Kernel"
    if not t or t in {"none", "-"}:
        return p
    return f"{p} (extend `{t}`)"


def infer_unsupported_reco(op_name: str, op_description: str, rules):
    name_l = (op_name or "").lower()
    desc_l = (op_description or "").lower()

    def has_keyword(text: str, kw: str) -> bool:
        return (
            re.search(rf"(^|[^a-z0-9]){re.escape(kw)}([^a-z0-9]|$)", text) is not None
        )

    # Exact op override has the highest priority.
    for item in rules.get("op_overrides") or []:
        if (item.get("op_name") or "").lower() == name_l:
            return {
                "recommended": format_reco(
                    item.get("recommended_path"), item.get("wrapper_extension_target")
                ),
                "source": "op_override",
                "matched_rule": item.get("op_name"),
                "rationale": item.get("rationale", ""),
            }

    # Family routing by op name / description semantics.
    for fam in rules.get("family_routing") or []:
        kws = [str(x).lower() for x in (fam.get("op_name_keywords_any") or [])]
        if any(k and (has_keyword(name_l, k) or has_keyword(desc_l, k)) for k in kws):
            return {
                "recommended": format_reco(
                    fam.get("preferred_path"), fam.get("wrapper_extension_target")
                ),
                "source": "family_routing",
                "matched_rule": fam.get("family"),
                "rationale": fam.get("notes", ""),
            }

    # Fallback
    default_fb = rules.get("default_fallback") or {}
    return {
        "recommended": format_reco(
            default_fb.get("recommended_path"),
            default_fb.get("wrapper_extension_target"),
        ),
        "source": "default_fallback",
        "matched_rule": "default_fallback",
        "rationale": default_fb.get("rationale", ""),
    }


def recommended_impl_with_trace(op_row, compat_row, reco_rules):
    status = op_row.get("status")
    hip_op = op_row.get("hip_op")
    backend = op_row.get("backend")
    runtime = op_row.get("runtime_func")
    codes = (compat_row or {}).get("reason_codes") or []
    if status in {"full", "partial"} and "COMPILE_TIME_TENSOR_OP" in codes:
        # Covers both forms: converted into a non-runtime op, and folded away
        # with no op left to point at.
        return {
            "recommended": "Compile Time Optimization",
            "source": "compile_time",
            "matched_rule": hip_op or "folded",
            "rationale": "No runtime call comes out of this operator.",
        }
    if status in {"full", "partial"}:
        # The conversion already picked the implementation, so report what it
        # produced instead of guessing from the rule table.
        if hip_op and not hip_op.startswith("hip."):
            return {
                "recommended": "Compile Time Optimization",
                "source": "supported_compile_time_fold",
                "matched_rule": hip_op,
                "rationale": "Converted to a compile-time op, not a runtime kernel.",
            }
        if hip_op and backend and runtime:
            return {
                "recommended": f"{backend} (`{runtime}`)",
                "source": "supported_backend_runtime",
                "matched_rule": hip_op,
                "rationale": "",
            }
        if hip_op:
            return {
                "recommended": f"Hip Dialect (`{hip_op}`)",
                "source": "supported_hip_op",
                "matched_rule": hip_op,
                "rationale": "",
            }
        return {
            "recommended": "Unknown",
            "source": "supported_unknown",
            "matched_rule": "unknown",
            "rationale": "",
        }

    if "CONVERSION_REJECTED_INSTANCES" in codes:
        # The operator is implemented; the fix is to widen what the existing
        # conversion accepts, not to write a new kernel.
        return {
            "recommended": "Extend the existing conversion",
            "source": "conversion_rejected",
            "matched_rule": "CONVERSION_REJECTED_INSTANCES",
            "rationale": "A conversion exists but refused this model's instances.",
        }

    if compile_time_reason_with_rules(
        (compat_row or {}).get("reason_texts") or [], reco_rules
    ):
        return {
            "recommended": "Compile Time Optimization",
            "source": "compile_time_reason",
            "matched_rule": "compile_time_rule",
            "rationale": ((reco_rules.get("compile_time_rule") or {}).get("rationale"))
            or "",
        }
    return infer_unsupported_reco(
        op_row.get("onnx_op", ""),
        resolve_op_description(op_row),
        reco_rules or {},
    )


def fmt_data_types(dtypes):
    return ", ".join(dtypes) if dtypes else "-"


def humanize_camel(name: str) -> str:
    out = []
    for i, ch in enumerate(name):
        if i > 0 and ch.isupper() and (not name[i - 1].isupper()):
            out.append(" ")
        out.append(ch)
    return "".join(out)


def resolve_op_description(op_row):
    op = op_row.get("onnx_op", "") or ""
    domain = op_row.get("domain", "") or ""
    existing = op_row.get("op_description")
    if isinstance(existing, str) and existing.strip() and existing.strip() != "—":
        return existing.strip()

    # Try ONNX schema docs first for standard domain.
    if domain in {"onnx", "ai.onnx", ""}:
        try:
            from onnx import defs

            schema = defs.get_schema(op, domain="")
            doc = " ".join((schema.doc or "").split())
            if doc:
                return doc.split(". ")[0].strip().rstrip(".")
        except Exception:
            pass

    # Curated custom-domain descriptions.
    custom_map = {
        "CausalConvWithState": "Causal convolution with recurrent state cache update (com.microsoft)",
        "LinearAttention": "Linear attention operator with stateful/key-value efficient computation (com.microsoft)",
    }
    if op in custom_map:
        return custom_map[op]

    # Last-resort semantic description from op name.
    return f"{humanize_camel(op)} operation ({domain})"


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit(
            "Usage: python generate_final_reports.py <analysis_dir> [report_dir]"
        )

    analysis_dir = Path(sys.argv[1])
    # The markdown is what a person opens, so it can live above the analysis
    # JSON instead of being written twice.
    report_dir = Path(sys.argv[2]) if len(sys.argv) == 3 else analysis_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    reco_rules = load_reco_rules(script_dir)
    report_input = read_json(analysis_dir / "report_input.json")
    op_dist_comparison = load_op_distribution_comparison(analysis_dir)

    meta = report_input["meta"]
    summary = report_input["summary"]
    op_dist = report_input["operator_distribution"]
    mapping_chain = report_input["mapping_chain"]
    compatibility = report_input["compatibility"]

    compat_map = {(c.get("onnx_op"), c.get("domain")): c for c in compatibility}

    total_instances = summary["total_node_instances"]
    supported_instances = summary["supported_instances"]
    supported_pct = (
        (supported_instances / total_instances * 100.0) if total_instances else 0.0
    )

    # Main report
    lines = []
    lines.append("# Model compatibility report\n")

    # A failed stage is the finding, so it goes before anything a reader could
    # mistake for a verdict.
    failure = meta.get("failure")
    evidence = (meta.get("tool_versions") or {}).get("support_evidence", "")
    if failure:
        # A model that does not compile is the finding either way, but which
        # operator stopped it is a different statement from none of them being
        # known, and the slices tell those apart.
        if evidence == "single-operator conversions":
            lines.append(
                f"\n> **The {failure['stage']} step failed on the whole graph: "
                f"{failure['headline']}.** The model does not compile as it "
                "stands. Support below comes from converting each operator on "
                "its own, so it says which operators are not what stopped it, "
                "not that the model runs.\n\n"
            )
        else:
            lines.append(
                f"\n> **The {failure['stage']} step failed: {failure['headline']}.** "
                "No operator support was verified; the counts below describe the "
                "graph, not what hip-ep can run.\n\n"
            )

    lines.append(f"- **Analyzed graph:** `{meta['model_path']}`\n")
    if op_dist_comparison:
        orig_path = (op_dist_comparison.get("meta") or {}).get("original_model", "")
        if orig_path:
            lines.append(f"- **Original model:** `{orig_path}`\n")
    lines.append(f"- Generated UTC: `{meta['generated_at_utc']}`\n\n")

    if failure:
        lines.append("## Where it failed\n\n")
        lines.append(f"- **Step:** {failure['stage']}\n")
        if failure.get("command"):
            lines.append(f"- **Command:** `{failure['command']}`\n")
        if failure.get("exit_code") is not None:
            lines.append(f"- **Exit code:** {failure['exit_code']}\n")
        lines.append(f"- **Reason:** {failure['headline']}\n")
        if failure.get("log"):
            lines.append(f"- **Log:** `{failure['log']}`\n")
        if failure.get("details"):
            lines.append("\nFrom the log:\n\n")
            for detail in failure["details"]:
                lines.append(f"- `{detail}`\n")
        lines.append("\n")

    lines.append("## Summary\n\n")
    lines.append(f"- Total node instances: {summary['total_node_instances']}\n")
    lines.append(
        f"- Supported instances: {summary['supported_instances']} ({supported_pct:.1f}%)\n"
    )
    lines.append(f"- Unsupported instances: {summary['unsupported_instances']}\n")
    lines.append(f"- Total Operator Types: {summary['total_operator_types']}\n")
    lines.append(f"- Fully Compatible: {summary['fully_compatible_operator_types']}\n")
    lines.append(
        f"- Partially Compatible: {summary['partially_compatible_operator_types']}\n"
    )
    lines.append(f"- Unsupported: {summary['unsupported_operator_types']}\n\n")

    if op_dist_comparison:
        lines.extend(render_op_distribution_comparison_section(op_dist_comparison))

    lines.append("## Operator Distribution with Compatibility Status\n\n")
    lines.append(
        "_Counts and status below refer to the **EP input** graph only._\n\n"
        if op_dist_comparison
        else ""
    )
    lines.append(
        "| Op Type | Domain | Count | Data Types | Recommended Rocm Implementation | Status | Op Description |\n"
    )
    lines.append("|---|---|---:|---|---|---|---|\n")

    unsupported_buckets = {}
    supported_ops = []
    partial_ops = []
    unsupported_ops = []
    for row in op_dist:
        key = (row.get("onnx_op"), row.get("domain"))
        comp = compat_map.get(key, {})
        reco_trace = recommended_impl_with_trace(row, comp, reco_rules)
        reco = reco_trace["recommended"]
        disp = status_display(row.get("status", ""))
        op = row.get("onnx_op", "")
        dom = row.get("domain", "")
        cnt = row.get("count", 0)
        desc = resolve_op_description(row)
        lines.append(
            f"| {op} | {dom} | {cnt} | {fmt_data_types(row.get('data_types') or [])} | {reco} | {disp} | {desc} |\n"
        )
        if disp == "supported":
            supported_ops.append(
                {
                    "onnx_op": op,
                    "hip_op": row.get("hip_op") or "-",
                }
            )
        elif disp == "partial":
            partial_ops.append(op)
        else:
            reason_texts = comp.get("reason_texts") or []
            reason_text = (
                "; ".join(reason_texts)
                if reason_texts
                else "No Hip Dialect implementation available."
            )
            unsupported_ops.append(
                {
                    "onnx_op": op,
                    "reason": reason_text,
                }
            )
            unsupported_buckets.setdefault(reco, []).append(op)

    lines.append("\n### Compatibility Summary\n\n")
    lines.append(f"#### Fully Compatible Operator ({len(supported_ops)})\n\n")
    for item in supported_ops:
        lines.append(f"- `{item['onnx_op']}` -> `{item['hip_op']}`\n")
    lines.append("\n")
    lines.append(f"#### Partially Compatible Operators ({len(partial_ops)}):\n\n")
    for p in partial_ops:
        c = compat_map.get(
            (p, next((r["domain"] for r in op_dist if r["onnx_op"] == p), "onnx")), {}
        )
        reasons = c.get("reason_texts") or []
        reason_text = "; ".join(reasons) if reasons else "Schema mismatch."
        lines.append(f"- `{p}`: {reason_text}\n")
    lines.append("\n")
    lines.append(f"#### Unsupported Operators ({len(unsupported_ops)}):\n\n")
    for item in unsupported_ops:
        lines.append(f"- {item['onnx_op']}: {item['reason']}\n")
    lines.append("\n")

    lines.append("Unsupported operator recommendation buckets\n\n")
    for k, ops in sorted(unsupported_buckets.items(), key=lambda x: x[0]):
        lines.append(f"- {k}: " + ", ".join(f"`{x}`" for x in ops) + "\n")
    lines.append("\n")

    lines.append("## ONNX to Hip to runtime mapping\n\n")
    lines.append(
        "_Hip op observed in the conversion; runtime function and backend read "
        "from its HIP-to-LLVM lowering and runtime implementation._\n\n"
    )
    lines.append(
        "| ONNX Op | Domain | Hip Op | Runtime Func | Backend | Instances | Status |\n"
    )
    lines.append("|---|---|---|---|---|---:|---|\n")
    for m in mapping_chain:
        lines.append(
            f"| {m.get('onnx_op', '')} | {m.get('domain', '')} | {m.get('hip_op', '')} "
            f"| {m.get('runtime_func') or '—'} | {m.get('backend') or '—'} "
            f"| {m.get('instances', 0)} | {status_display(m.get('status', ''))} |\n"
        )
    lines.append(
        "\nDetailed compatibility diagnostics are in model_compatibility_details.md\n"
    )

    (report_dir / "model_compatibility_report.md").write_text(
        "".join(lines), encoding="utf-8"
    )

    # Details report
    # Everything the main report already states is left out on purpose: this
    # file exists for the per-operator evidence behind a non-supported row.
    d = []
    d.append("# Model compatibility details\n\n")
    d.append(f"- **Analyzed graph:** `{meta['model_path']}`\n")
    d.append(f"- Generated UTC: `{meta['generated_at_utc']}`\n")
    d.append("- Summary and operator distribution: `model_compatibility_report.md`\n\n")

    d.append("## Partially compatible details\n\n")
    d.append("| Op Type | Domain | Reason Codes | Reason Texts | Evidence |\n")
    d.append("|---|---|---|---|---|\n")
    for row in compatibility:
        if row.get("status") == "partial":
            ev = row.get("evidence") or []
            ev_text = (
                "; ".join(
                    f"{e.get('source_file', '')}:{e.get('json_pointer', '')}"
                    for e in ev
                )
                if ev
                else "-"
            )
            d.append(
                f"| {row.get('onnx_op', '')} | {row.get('domain', '')} | {', '.join(row.get('reason_codes') or []) or '-'} | {'; '.join(row.get('reason_texts') or []) or '-'} | {ev_text} |\n"
            )

    d.append("\n## Data quality notes\n\n")
    notes = []
    attr_path = analysis_dir / "attr_transfer.json"
    if attr_path.is_file():
        attr_data = read_json(attr_path)
        orphans = attr_data.get("orphan_hip_ops") or {}
        for hip_op, count in sorted(orphans.items()):
            # A runtime op no onnx op accounts for means the pairing lost
            # track, which is why unpaired operators below stay unresolved.
            notes.append(
                f"- `{hip_op}`: {count} runtime op(s) trace back to no ONNX "
                "operator, so pairing is incomplete for this model.\n"
            )
        for key, count in sorted((attr_data.get("unpaired_instances") or {}).items()):
            notes.append(
                f"- `{key}`: {count} instance(s) had no location match after "
                "conversion, so their attribute transfer was not verified.\n"
            )
        for row in attr_data.get("rows") or []:
            if row.get("folded_instances") and not row.get("paired_instances"):
                notes.append(
                    f"- `{row['key']}`: {row['folded_instances']} instance(s) left "
                    "no runtime op; reported as compile-time because every hip op "
                    "in the module is accounted for.\n"
                )
    else:
        notes.append(
            "- Conversion was not probed, so operator support is unverified.\n"
        )
    d.extend(notes or ["- None.\n"])

    (report_dir / "model_compatibility_details.md").write_text(
        "".join(d), encoding="utf-8"
    )
    print(f"Wrote {(report_dir / 'model_compatibility_report.md')}")
    print(f"Wrote {(report_dir / 'model_compatibility_details.md')}")


if __name__ == "__main__":
    main()
