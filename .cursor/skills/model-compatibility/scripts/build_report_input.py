#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Normalize the pipeline outputs into report_input.json (step 5).

Support comes from what the compiler did, not from reading its source:

  unsupported  every instance was still an onnx op after convert-onnx-to-hip
  partial      some instances converted, or an ONNX attribute with a
               non-default value did not reach the HIP op
  full         converted, with every meaningful attribute accounted for

Without the conversion probe (the -SkipDump path, which has no EP-input
MLIR to convert) no operator can be classified, so every row is reported as
unknown-by-omission: the status stays `partial` with an explicit reason. The
report then carries the badge the orchestrator adds.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

FALLBACK_OP_DESCRIPTIONS = {
    "MatMulNBits": "Quantized N-bit matrix multiplication (com.microsoft)",
    "RotaryEmbedding": "Rotary position embedding (RoPE)",
    "SkipSimplifiedLayerNormalization": "Skip connection + RMS normalization",
    "GroupQueryAttention": "Group Query Attention mechanism",
    "QMoE": "Mixture-of-Experts routing and fused expert computation",
}

REASON_CATALOG = [
    ("NO_HIP_DIALECT_IMPL", "No Hip Dialect implementation available."),
    (
        "CONVERSION_REJECTED_INSTANCES",
        "A conversion exists but refused this model's instances.",
    ),
    ("COMPILE_TIME_TENSOR_OP", "Handled at compile time."),
    ("EXTRA_ONNX_ATTR_NOT_IN_HIP", "Extra ONNX attributes not in Hip op."),
    (
        "PARTIAL_INSTANCE_CONVERSION",
        "Some instances converted and others did not.",
    ),
    ("CONVERSION_NOT_PROBED", "Conversion was not run; support is unknown."),
]

# Anything outside the hip dialect is a compile-time fold rather than a
# runtime kernel, whichever dialect the conversion happened to pick.
_RUNTIME_DIALECT_PREFIX = "hip."


def norm_domain(domain: str) -> str:
    if not domain or domain in {"ai.onnx", "onnx", ""}:
        return "onnx"
    return domain


def onnx_op_description(op: str, domain: str) -> str:
    try:
        from onnx import defs

        schema = defs.get_schema(
            op, domain="" if domain in {"", "onnx", "ai.onnx"} else domain
        )
        doc = " ".join((schema.doc or "").split())
        if doc:
            return doc.split(". ")[0].strip().rstrip(".")
    except Exception:
        pass
    return FALLBACK_OP_DESCRIPTIONS.get(op, "—")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_against_schema(report_input: dict) -> None:
    """Fail here rather than let a malformed report reach the reader.

    The schema is the contract between this file and everything downstream, so
    a new reason code or a renamed field should stop the run. jsonschema is
    optional: without it the check is skipped and says so.
    """
    schema_path = Path(__file__).with_name("report_input.schema.json")
    try:
        import jsonschema
    except ImportError:
        print("  (jsonschema not installed; report_input.json was not validated)")
        return
    jsonschema.validate(report_input, read_json(schema_path))


def load_optional(path: Path):
    return read_json(path) if path.is_file() else None


def index_by_key(entries, key_fields=("op_type", "domain")):
    """Index rows on (op_type, normalized domain)."""
    indexed = {}
    for entry in entries or []:
        key = (entry.get(key_fields[0], ""), norm_domain(entry.get(key_fields[1], "")))
        indexed[key] = entry
    return indexed


def leftover_reason(reason_row):
    """Reason code and text for an operator no instance of which converted.

    A missing converter and a converter that refused every instance are
    different findings: the first needs a new implementation, the second needs
    an existing one widened, so they must not share a reason text.
    """
    if not reason_row or not reason_row.get("converter_found"):
        return "NO_HIP_DIALECT_IMPL", ["No Hip Dialect implementation available."]

    files = (
        ", ".join(reason_row.get("converter_files") or []) or "lib/Conversion/OnnxToHip"
    )
    observed = ", ".join(reason_row.get("observed_element_types") or [])
    text = (
        f"A conversion exists ({files}) but refused every instance. "
        f"Operand element types in this model: {observed or 'unknown'}."
    )

    likely = [r for r in reason_row.get("refusals") or [] if r.get("likely")]
    if likely:
        candidates = "; ".join(f"{r['message']} [{r['location']}]" for r in likely)
        # Candidates, not a verdict: the refusal reason is compiled out of a
        # release build, so these are the constraints that the observed types
        # contradict, to be confirmed against the source.
        text += f" Candidate constraints: {candidates}."
    return "CONVERSION_REJECTED_INSTANCES", [text]


def classify(op, domain, count, leftover, attr_row, reason_row, probed):
    """Status, reason codes and texts for one operator type."""
    if not probed:
        return (
            "partial",
            ["CONVERSION_NOT_PROBED"],
            [
                "Conversion was not run (no EP-input MLIR), so support "
                "for this operator was not verified."
            ],
        )

    leftover_count = int((leftover or {}).get("count", 0))
    if leftover_count >= count:
        code, texts = leftover_reason(reason_row)
        return "unsupported", [code], texts

    codes, texts = [], []
    if leftover_count:
        codes.append("PARTIAL_INSTANCE_CONVERSION")
        texts.append(
            f"{leftover_count} of {count} instance(s) were not converted; the "
            "conversion pattern bailed out on them."
        )

    dropped = (attr_row or {}).get("dropped_attrs") or {}
    if dropped:
        codes.append("EXTRA_ONNX_ATTR_NOT_IN_HIP")
        # Show the value: an attribute whose schema declares no default cannot
        # be proven harmless, so the reader needs to see what was set.
        values = (attr_row or {}).get("dropped_attr_values") or {}
        detail = ", ".join(
            f"{name}={'/'.join(values[name])} ({cnt} instance(s))"
            if values.get(name)
            else f"{name} ({cnt} instance(s))"
            for name, cnt in dropped.items()
        )
        texts.append(f"ONNX attributes not carried to the Hip op: {detail}.")

    if codes:
        return "partial", codes, texts

    hip_ops = (attr_row or {}).get("hip_ops") or []
    if hip_ops and not any(h.startswith(_RUNTIME_DIALECT_PREFIX) for h in hip_ops):
        return "full", ["COMPILE_TIME_TENSOR_OP"], ["Handled at compile time."]
    # No runtime op came out of it, and the analysis proved none went missing.
    if not hip_ops and (
        (attr_row or {}).get("folded_instances")
        or (attr_row or {}).get("compile_time_instances")
    ):
        return "full", ["COMPILE_TIME_TENSOR_OP"], ["Handled at compile time."]
    return "full", [], []


def runtime_entry(hip_op, runtime_map):
    """Runtime function and backend for a converted op, when known."""
    row = (runtime_map or {}).get(hip_op or "") or {}
    return row.get("runtime_func"), row.get("backend")


def primary_hip_op(attr_row):
    hip_ops = (attr_row or {}).get("hip_ops") or []
    for hip_op in hip_ops:
        if hip_op.startswith(_RUNTIME_DIALECT_PREFIX):
            return hip_op
    return hip_ops[0] if hip_ops else None


def main():
    if len(sys.argv) != 5:
        raise SystemExit(
            "Usage: build_report_input.py <analyzed_graph> <step1_json> "
            "<analysis_dir> <repo_root>"
        )
    analyzed_graph = Path(sys.argv[1])
    step1_json = Path(sys.argv[2])
    analysis_dir = Path(sys.argv[3])
    repo_root = Path(sys.argv[4])

    step1 = read_json(step1_json)
    leftovers = load_optional(analysis_dir / "leftover_onnx.json")
    attrs = load_optional(analysis_dir / "attr_transfer.json")
    reasons = load_optional(analysis_dir / "leftover_reasons.json")
    runtime_map = (load_optional(analysis_dir / "hip_runtime_map.json") or {}).get(
        "ops"
    )
    probed = leftovers is not None

    leftover_by_key = index_by_key((leftovers or {}).get("unconverted"))
    attr_by_key = index_by_key((attrs or {}).get("rows"))
    reason_by_key = index_by_key((reasons or {}).get("rows"))

    op_dist, comp_rows, mapping_chain = [], [], []
    full_types = partial_types = unsupported_types = 0
    supported_instances = unsupported_instances = 0
    total_instances = 0

    op_entries = [
        (op, info)
        for op, info in step1.items()
        if not op.startswith("_") and isinstance(info, dict)
    ]
    op_entries.sort(key=lambda item: (-int(item[1].get("count", 0)), item[0]))

    for op, info in op_entries:
        count = int(info.get("count", 0))
        domains = info.get("domain") or ["ai.onnx"]
        domain = norm_domain(domains[0])
        key = (op, domain)
        leftover = leftover_by_key.get(key)
        attr_row = attr_by_key.get(key)

        status, reason_codes, reason_texts = classify(
            op, domain, count, leftover, attr_row, reason_by_key.get(key), probed
        )
        leftover_count = int((leftover or {}).get("count", 0))

        total_instances += count
        if probed:
            unsupported_instances += leftover_count
            supported_instances += count - leftover_count
        # Without the probe nothing is known, so no instance is counted as
        # supported: a 100% headline on an unverified run would be read as a
        # result rather than as an absence of one.

        if status == "full":
            full_types += 1
        elif status == "partial":
            partial_types += 1
        else:
            unsupported_types += 1

        hip_op = primary_hip_op(attr_row)
        runtime_func, backend = runtime_entry(hip_op, runtime_map)
        op_dist.append(
            {
                "onnx_op": op,
                "domain": domain,
                "count": count,
                "data_types": [str(x) for x in (info.get("data_types") or []) if x],
                "status": status,
                "hip_op": hip_op,
                "runtime_func": runtime_func,
                "backend": backend,
                "op_description": onnx_op_description(op, domain),
            }
        )

        evidence_file = "leftover_onnx.json" if leftover else "attr_transfer.json"
        comp_rows.append(
            {
                "onnx_op": op,
                "domain": domain,
                "status": status,
                "reason_codes": reason_codes,
                "reason_texts": reason_texts,
                "evidence": [
                    {
                        "source_file": evidence_file,
                        "json_pointer": f"/{'unconverted' if leftover else 'rows'}",
                    }
                ],
            }
        )

        mapping_chain.append(
            {
                "onnx_op": op,
                "domain": domain,
                "hip_op": hip_op or "—",
                "runtime_func": runtime_func,
                "backend": backend,
                "instances": count,
                "status": status,
            }
        )

    out = {
        "meta": {
            "model_path": str(analyzed_graph),
            "generated_at_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "repo_root": str(repo_root),
            "tool_versions": {
                "pipeline": "convert-onnx-to-hip oracle",
                "conversion_probed": str(probed).lower(),
            },
        },
        "summary": {
            "total_node_instances": total_instances,
            "supported_instances": supported_instances,
            "unsupported_instances": unsupported_instances,
            "total_operator_types": len(op_entries),
            "fully_compatible_operator_types": full_types,
            "partially_compatible_operator_types": partial_types,
            "unsupported_operator_types": unsupported_types,
        },
        "operator_distribution": op_dist,
        "mapping_chain": mapping_chain,
        "compatibility": comp_rows,
        "reason_catalog": [
            {"code": code, "default_text": text} for code, text in REASON_CATALOG
        ],
    }

    out_path = analysis_dir / "report_input.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    validate_against_schema(out)
    print(f"Wrote {out_path}")
    print(
        f"  {supported_instances}/{total_instances} instances supported, "
        f"{unsupported_types} unsupported operator type(s)"
    )


if __name__ == "__main__":
    main()
