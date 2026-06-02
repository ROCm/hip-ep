#!/usr/bin/env python3
import json
import sys
import re
from pathlib import Path
from datetime import datetime, timezone


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
    """Markdown lines for original vs EP input (from compare_op_distribution.py JSON)."""
    meta = comp.get("meta") or {}
    summary = comp.get("summary") or {}
    rows = comp.get("rows") or []

    out = []
    out.append("## Original vs EP input (operator distribution)\n\n")
    out.append(
        "Compatibility analysis below uses the **EP input** graph (`onnx.onnx`). "
        "This section compares it to the packaged **original** ONNX.\n\n"
    )
    out.append(f"- **Original model:** `{meta.get('original_model', '—')}`\n")
    out.append(f"- **EP input (analyzed):** `{meta.get('ep_model', '—')}`\n\n")

    out.append("| Metric | Original | EP input | Delta |\n")
    out.append("|---|---:|---:|---:|\n")
    out.append(
        f"| Total node instances | {summary.get('original_total_nodes', 0)} | "
        f"{summary.get('ep_total_nodes', 0)} | {summary.get('node_delta', 0):+d} |\n"
    )
    out.append(
        f"| Unique operator types | {summary.get('original_unique_ops', 0)} | "
        f"{summary.get('ep_unique_ops', 0)} | "
        f"{int(summary.get('ep_unique_ops', 0)) - int(summary.get('original_unique_ops', 0)):+d} |\n\n"
    )

    only_orig = summary.get("only_in_original") or []
    only_ep = summary.get("only_in_ep") or []
    if only_orig:
        out.append("**Operators only in original:** " + ", ".join(f"`{x}`" for x in only_orig) + "\n\n")
    if only_ep:
        out.append("**Operators only in EP input:** " + ", ".join(f"`{x}`" for x in only_ep) + "\n\n")

    out.append("| Op Type | Original | EP input | Delta |\n")
    out.append("|---|---:|---:|---:|\n")
    for row in rows:
        d = int(row.get("delta", 0))
        mark = " **+**" if d > 0 else (" **-**" if d < 0 else "")
        out.append(
            f"| {row.get('op_type', '')} | {row.get('original_count', 0)} | "
            f"{row.get('ep_count', 0)} | {d:+d}{mark} |\n"
        )
    out.append("\n")
    return out


def parse_step2_operator_summary(step2_md: Path):
    if not step2_md.exists():
        return []
    lines = step2_md.read_text(encoding="utf-8").splitlines()
    in_section = False
    table_lines = []
    for line in lines:
        if line.strip() == "## Operator Summary":
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if in_section and line.strip().startswith("|"):
            table_lines.append(line)
    return table_lines


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
    terms = (((rules or {}).get("compile_time_rule") or {}).get("when_reason_contains_any")) or []
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
        return re.search(rf"(^|[^a-z0-9]){re.escape(kw)}([^a-z0-9]|$)", text) is not None

    # Exact op override has the highest priority.
    for item in (rules.get("op_overrides") or []):
        if (item.get("op_name") or "").lower() == name_l:
            return {
                "recommended": format_reco(item.get("recommended_path"), item.get("wrapper_extension_target")),
                "source": "op_override",
                "matched_rule": item.get("op_name"),
                "rationale": item.get("rationale", ""),
            }

    # Family routing by op name / description semantics.
    for fam in (rules.get("family_routing") or []):
        kws = [str(x).lower() for x in (fam.get("op_name_keywords_any") or [])]
        if any(k and (has_keyword(name_l, k) or has_keyword(desc_l, k)) for k in kws):
            return {
                "recommended": format_reco(fam.get("preferred_path"), fam.get("wrapper_extension_target")),
                "source": "family_routing",
                "matched_rule": fam.get("family"),
                "rationale": fam.get("notes", ""),
            }

    # Fallback
    default_fb = rules.get("default_fallback") or {}
    return {
        "recommended": format_reco(default_fb.get("recommended_path"), default_fb.get("wrapper_extension_target")),
        "source": "default_fallback",
        "matched_rule": "default_fallback",
        "rationale": default_fb.get("rationale", ""),
    }


def recommended_impl_with_trace(op_row, compat_row, reco_rules):
    status = op_row.get("status")
    hip_op = op_row.get("hip_op")
    backend = op_row.get("backend")
    runtime = op_row.get("runtime_func")
    if status in {"full", "partial"}:
        if hip_op and hip_op.startswith("tensor."):
            return {
                "recommended": "Compile Time Optimization",
                "source": "supported_tensor_compile_time",
                "matched_rule": "tensor.*",
                "rationale": "Mapped tensor op handled at compile time.",
            }
        if backend and backend != "Unknown" and runtime:
            return {
                "recommended": f"{backend} (`{runtime}`)",
                "source": "supported_backend_runtime",
                "matched_rule": "backend+runtime",
                "rationale": "",
            }
        if backend and backend != "Unknown":
            return {
                "recommended": backend,
                "source": "supported_backend_only",
                "matched_rule": "backend",
                "rationale": "",
            }
        if runtime:
            return {
                "recommended": f"`{runtime}`",
                "source": "supported_runtime_only",
                "matched_rule": "runtime",
                "rationale": "",
            }
        return {
            "recommended": "Unknown",
            "source": "supported_unknown",
            "matched_rule": "unknown",
            "rationale": "",
        }

    if compile_time_reason_with_rules((compat_row or {}).get("reason_texts") or [], reco_rules):
        return {
            "recommended": "Compile Time Optimization",
            "source": "compile_time_reason",
            "matched_rule": "compile_time_rule",
            "rationale": ((reco_rules.get("compile_time_rule") or {}).get("rationale")) or "",
        }
    return infer_unsupported_reco(
        op_row.get("onnx_op", ""),
        resolve_op_description(op_row),
        reco_rules or {},
    )


def recommended_impl(op_row, compat_row, reco_rules):
    return recommended_impl_with_trace(op_row, compat_row, reco_rules)["recommended"]


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
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python generate_final_reports.py <analysis_dir>")

    analysis_dir = Path(sys.argv[1])
    script_dir = Path(__file__).resolve().parent
    reco_rules = load_reco_rules(script_dir)
    report_input = read_json(analysis_dir / "report_input.json")
    step2_table = parse_step2_operator_summary(analysis_dir / "step2_hip_ops.md")
    op_dist_comparison = load_op_distribution_comparison(analysis_dir)

    meta = report_input["meta"]
    summary = report_input["summary"]
    op_dist = report_input["operator_distribution"]
    mapping_chain = report_input["mapping_chain"]
    compatibility = report_input["compatibility"]

    compat_map = {(c.get("onnx_op"), c.get("domain")): c for c in compatibility}

    total_instances = summary["total_node_instances"]
    supported_instances = summary["supported_instances"]
    supported_pct = (supported_instances / total_instances * 100.0) if total_instances else 0.0

    # Main report
    lines = []
    lines.append("# Model compatibility report\n")
    lines.append(f"- **EP input (compatibility target):** `{meta['model_path']}`\n")
    if op_dist_comparison:
        orig_path = (op_dist_comparison.get("meta") or {}).get("original_model", "")
        if orig_path:
            lines.append(f"- **Original model:** `{orig_path}`\n")
    lines.append(f"- Generated UTC: `{meta['generated_at_utc']}`\n\n")
    lines.append("## Summary\n\n")
    lines.append(f"- Total node instances: {summary['total_node_instances']}\n")
    lines.append(f"- Supported instances: {summary['supported_instances']} ({supported_pct:.1f}%)\n")
    lines.append(f"- Unsupported instances: {summary['unsupported_instances']}\n")
    lines.append(f"- Total Operator Types: {summary['total_operator_types']}\n")
    lines.append(f"- Fully Compatible: {summary['fully_compatible_operator_types']}\n")
    lines.append(f"- Partially Compatible: {summary['partially_compatible_operator_types']}\n")
    lines.append(f"- Unsupported: {summary['unsupported_operator_types']}\n\n")

    if op_dist_comparison:
        lines.extend(render_op_distribution_comparison_section(op_dist_comparison))

    lines.append("## Operator Distribution with Compatibility Status\n\n")
    lines.append(
        "_Counts and status below refer to the **EP input** graph only._\n\n"
        if op_dist_comparison
        else ""
    )
    lines.append("| Op Type | Domain | Count | Data Types | Recommended Rocm Implementation | Status | Op Description |\n")
    lines.append("|---|---|---:|---|---|---|---|\n")

    unsupported_buckets = {}
    supported_ops = []
    partial_ops = []
    unsupported_ops = []
    unsupported_runtime_entries = []

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
            reason_text = "; ".join(reason_texts) if reason_texts else "No Hip Dialect implementation available."
            unsupported_ops.append(
                {
                    "onnx_op": op,
                    "reason": reason_text,
                }
            )
            unsupported_buckets.setdefault(reco, []).append(op)
            unsupported_runtime_entries.append(
                {
                    "onnx_op": op,
                    "domain": dom,
                    "count": row.get("count", 0),
                    "op_description": desc,
                    "reason_codes": comp.get("reason_codes") or [],
                    "reason_texts": comp.get("reason_texts") or [],
                    "recommended_path": reco,
                    "recommendation_source": reco_trace.get("source", ""),
                    "matched_rule": reco_trace.get("matched_rule", ""),
                    "rationale": reco_trace.get("rationale", ""),
                }
            )

    lines.append("\n### Compatibility Summary\n\n")
    lines.append(f"#### Fully Compatible Operator ({len(supported_ops)})\n\n")
    for item in supported_ops:
        lines.append(f"- `{item['onnx_op']}` -> `{item['hip_op']}`\n")
    lines.append("\n")
    lines.append(f"#### Partially Compatible Operators ({len(partial_ops)}):\n\n")
    for p in partial_ops:
        c = compat_map.get((p, next((r["domain"] for r in op_dist if r["onnx_op"] == p), "onnx")), {})
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

    lines.append("## Hip Ops Summary\n\n")
    if step2_table:
        for tl in step2_table:
            lines.append(tl + "\n")
    else:
        lines.append("| Metric | Value |\n|---|---:|\n| Hip operator summary | — |\n")
    lines.append("\n")

    lines.append("## ONNX-HIP-RUNTIME Mapping\n\n")
    lines.append("| ONNX Op | Domain | Hip Op | Runtime Func | Backend |\n")
    lines.append("|---|---|---|---|---|\n")
    for m in mapping_chain:
        runtime = m.get("runtime_func") or "-"
        backend = m.get("backend") or "Unknown"
        lines.append(
            f"| {m.get('onnx_op','')} | {m.get('domain','')} | {m.get('hip_op','')} | {runtime} | {backend} |\n"
        )
    lines.append("\nDetailed compatibility diagnostics are in model_compatibility_details.md\n")

    (analysis_dir / "model_compatibility_report.md").write_text("".join(lines), encoding="utf-8")

    # Details report
    d = []
    d.append("# Model compatibility details\n\n")
    d.append(f"- **EP input (compatibility target):** `{meta['model_path']}`\n")
    if op_dist_comparison:
        orig_path = (op_dist_comparison.get("meta") or {}).get("original_model", "")
        if orig_path:
            d.append(f"- **Original model:** `{orig_path}`\n")
    d.append(f"- Generated UTC: `{meta['generated_at_utc']}`\n\n")
    if op_dist_comparison:
        d.extend(render_op_distribution_comparison_section(op_dist_comparison))

    d.append("## Supported operators table (full)\n\n")
    d.append("| Op Type | Domain | Count | Recommended Rocm Implementation |\n")
    d.append("|---|---|---:|---|\n")
    for row in op_dist:
        if status_display(row.get("status", "")) == "supported":
            comp = compat_map.get((row.get("onnx_op"), row.get("domain")), {})
            d.append(
                f"| {row.get('onnx_op','')} | {row.get('domain','')} | {row.get('count',0)} | {recommended_impl(row, comp, reco_rules)} |\n"
            )

    d.append("\n## Partially compatible details\n\n")
    d.append("| Op Type | Domain | Reason Codes | Reason Texts | Evidence |\n")
    d.append("|---|---|---|---|---|\n")
    for row in compatibility:
        if row.get("status") == "partial":
            ev = row.get("evidence") or []
            ev_text = "; ".join(f"{e.get('source_file','')}:{e.get('json_pointer','')}" for e in ev) if ev else "-"
            d.append(
                f"| {row.get('onnx_op','')} | {row.get('domain','')} | {', '.join(row.get('reason_codes') or []) or '-'} | {'; '.join(row.get('reason_texts') or []) or '-'} | {ev_text} |\n"
            )

    d.append("\n## Unsupported operators\n\n")
    d.append("| Op Type | Domain | Count | Reason |\n")
    d.append("|---|---|---:|---|\n")
    for row in op_dist:
        if status_display(row.get("status", "")) == "unsupported":
            comp = compat_map.get((row.get("onnx_op"), row.get("domain")), {})
            reason = "; ".join(comp.get("reason_texts") or []) or "No Hip Dialect implementation available."
            d.append(f"| {row.get('onnx_op','')} | {row.get('domain','')} | {row.get('count',0)} | {reason} |\n")

    d.append("\n## Data quality notes\n\n")
    d.append("- None.\n")

    (analysis_dir / "model_compatibility_details.md").write_text("".join(d), encoding="utf-8")
    runtime_json = {
        "meta": {
            "model_path": meta.get("model_path"),
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "rules_version": reco_rules.get("version", "unknown"),
        },
        "summary": {
            "unsupported_operator_types": len(unsupported_runtime_entries),
            "unsupported_node_instances": int(
                sum(int(x.get("count", 0)) for x in unsupported_runtime_entries)
            ),
        },
        "unsupported_recommendations": unsupported_runtime_entries,
    }
    (analysis_dir / "unsupported_reco_runtime.json").write_text(
        json.dumps(runtime_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {(analysis_dir / 'model_compatibility_report.md')}")
    print(f"Wrote {(analysis_dir / 'model_compatibility_details.md')}")
    print(f"Wrote {(analysis_dir / 'unsupported_reco_runtime.json')}")


if __name__ == "__main__":
    main()
