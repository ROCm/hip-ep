#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys

from onnx_graph_walk import iter_model_nodes

FALLBACK_OP_DESCRIPTIONS = {
    "MatMulNBits": "Quantized N-bit matrix multiplication (com.microsoft)",
    "RotaryEmbedding": "Rotary position embedding (RoPE)",
    "SkipSimplifiedLayerNormalization": "Skip connection + RMS normalization",
    "GroupQueryAttention": "Group Query Attention mechanism",
    "QMoE": "Mixture-of-Experts routing and fused expert computation",
}


def norm_domain(domain: str) -> str:
    if not domain or domain == "ai.onnx":
        return "onnx"
    return domain


def unsupported_rec(op: str):
    # Fallback table for ops that have NO Conversion.cpp at all but are known
    # to be eliminated by MLIR's standard passes (constant folding /
    # canonicalization / dead-code elimination). Ops whose conversion EXISTS
    # but lowers to tensor.* / arith.* / memref.* are auto-detected by step2_1
    # (which emits a synthetic `tensor.<X>` mapping) and never reach this
    # branch -- they show up as supported / compile-time in the report.
    compile_time = {
        "Constant": "Constant folding / constant propagation - handled at compile time.",
        "CastLike": "Type canonicalization pattern - resolved at compile time.",
    }
    if op in compile_time:
        return "Compile Time Optimization", compile_time[op]
    return "Custom Hip Kernel", "No Hip Dialect implementation available."


def load_strict_attr_names(analysis_dir: Path, repo_root: Path):
    """
    Load strict-required attribute names from external config.
    Search order:
    1) <analysis_dir>/compatibility_attr_rules.json
    2) <repo_root>/compatibility_attr_rules.json
    """
    candidates = [
        analysis_dir / "compatibility_attr_rules.json",
        repo_root / "compatibility_attr_rules.json",
    ]
    for cfg in candidates:
        if not cfg.exists():
            continue
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
            names = data.get("strict_required_attrs") or []
            return {str(x) for x in names if str(x).strip()}
        except Exception:
            continue
    return set()


def onnx_op_description(op: str, domain: str) -> str:
    """
    Resolve ONNX operator description from schema docs.
    Falls back to curated descriptions for non-standard domains.
    """
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
    if op in FALLBACK_OP_DESCRIPTIONS:
        return FALLBACK_OP_DESCRIPTIONS[op]
    return "—"


def io_bounds(operands: dict, skip_ctx: bool):
    min_edges = 0
    max_edges = 0
    has_variadic = False
    for name, info in (operands or {}).items():
        if not isinstance(info, dict):
            continue
        if skip_ctx and name in {"ctx", "context"}:
            continue
        variadic = bool(info.get("variadic"))
        required = bool(info.get("required", True))
        if variadic:
            has_variadic = True
            if required:
                min_edges += 1
        else:
            max_edges += 1
            if required:
                min_edges += 1
    return min_edges, (None if has_variadic else max_edges)


def analyze_schema_for_key(nodes, hip_entry, strict_attr_names):
    ins = (hip_entry or {}).get("inputs") or {}
    outs = (hip_entry or {}).get("outputs") or {}
    attrs = (hip_entry or {}).get("attributes") or {}

    req_attrs = []
    for k, v in attrs.items():
        if not isinstance(v, dict):
            continue
        if (not v.get("optional", False)) or (k in strict_attr_names):
            req_attrs.append(k)
    declared_attrs = {k for k, v in attrs.items() if isinstance(v, dict)}
    req_attrs = sorted(req_attrs)

    min_in, max_in = io_bounds(ins, skip_ctx=True)
    min_out, max_out = io_bounds(outs, skip_ctx=False)
    total = len(nodes)

    miss_attr_counts = Counter()
    extra_attr_counts = Counter()
    in_low = in_high = out_low = out_high = 0
    nodes_with_extra = 0

    for n in nodes:
        attrs_set = {a.name for a in n.attribute}
        for a in req_attrs:
            if a not in attrs_set:
                miss_attr_counts[a] += 1
        extra = sorted(attrs_set - declared_attrs)
        if extra:
            nodes_with_extra += 1
            for x in extra:
                extra_attr_counts[x] += 1
        nin = len(n.input)
        nout = len(n.output)
        if nin < min_in:
            in_low += 1
        if max_in is not None and nin > max_in:
            in_high += 1
        if nout < min_out:
            out_low += 1
        if max_out is not None and nout > max_out:
            out_high += 1

    reason_codes = []
    reason_texts = []

    all_missing = [a for a in req_attrs if miss_attr_counts[a] == total and total > 0]
    some_missing = [a for a in req_attrs if 0 < miss_attr_counts[a] < total]
    if all_missing:
        reason_codes.append("MISSING_HIP_REQUIRED_ATTR")
        reason_texts.append(
            "Missing Hip-required attributes in all ONNX instances: "
            + ", ".join(sorted(all_missing))
        )
    if some_missing:
        if "MISSING_HIP_REQUIRED_ATTR" not in reason_codes:
            reason_codes.append("MISSING_HIP_REQUIRED_ATTR")
        reason_texts.append(
            "Missing Hip-required attributes in some ONNX instances: "
            + ", ".join(sorted(some_missing))
        )

    if in_low:
        reason_codes.append("ONNX_INPUT_BELOW_HIP_MIN")
        reason_texts.append(
            f"ONNX inputs below Hip minimum in {in_low}/{total} instance(s)."
        )
    if in_high:
        reason_codes.append("ONNX_INPUT_ABOVE_HIP_MAX")
        reason_texts.append(
            f"ONNX inputs above Hip maximum in {in_high}/{total} instance(s)."
        )
    if out_low:
        reason_codes.append("ONNX_OUTPUT_BELOW_HIP_MIN")
        reason_texts.append(
            f"ONNX outputs below Hip minimum in {out_low}/{total} instance(s)."
        )
    if out_high:
        reason_codes.append("ONNX_OUTPUT_ABOVE_HIP_MAX")
        reason_texts.append(
            f"ONNX outputs above Hip maximum in {out_high}/{total} instance(s)."
        )
    if nodes_with_extra:
        reason_codes.append("EXTRA_ONNX_ATTR_NOT_IN_HIP")
        top = [k for k, _ in extra_attr_counts.most_common(12)]
        reason_texts.append(
            "Extra attributes in ONNX not supported by Hip: " + ", ".join(top)
        )

    status = "partial" if reason_codes else "full"
    return status, sorted(set(reason_codes)), reason_texts


def main():
    if len(sys.argv) != 4:
        raise SystemExit(
            "Usage: build_report_input.py <model.onnx> <analysis_dir> <repo_root>"
        )
    model_path = Path(sys.argv[1])
    analysis_dir = Path(sys.argv[2])
    repo_root = Path(sys.argv[3])

    import onnx

    strict_attr_names = load_strict_attr_names(analysis_dir, repo_root)

    step1 = json.loads(
        (analysis_dir / "step1_onnx_ops.json").read_text(encoding="utf-8")
    )
    step21 = json.loads(
        (analysis_dir / "step2_1_onnx_to_hip_mappings.json").read_text(encoding="utf-8")
    )
    step23 = json.loads(
        (analysis_dir / "step2_3_backend_analysis.json").read_text(encoding="utf-8")
    )
    step2hip = json.loads(
        (analysis_dir / "step2_hip_ops.json").read_text(encoding="utf-8")
    )

    # Consolidate step2_1 mappings keyed on (op, domain). An ONNX op may
    # have MULTIPLE mappings -- e.g. Gather has hip.gather (the runtime path
    # in GatherConversion.cpp) AND tensor.from_elements (the shape-fold
    # variant in GatherShapeFold.cpp); ConstantOfShape has tensor.splat (the
    # MLIR-std fold) and no hip.* op. Prefer real hip.* mappings over
    # tensor.*/arith.*/memref.* compile-time fold variants when both exist
    # so the report column reflects the executed runtime path, not the
    # shape-fold fallback.
    def _mapping_priority(mapping):
        hop = mapping.get("hip_op", "") or ""
        if hop.startswith("hip."):
            return 0  # real runtime path -- highest priority
        if (
            hop.startswith("tensor.")
            or hop.startswith("arith.")
            or hop.startswith("memref.")
        ):
            return 1  # compile-time fold variant
        return 2

    support = {}
    for m in step21.get("mappings", []):
        key = (m.get("onnx_op", ""), norm_domain(m.get("onnx_domain", "onnx")))
        existing = support.get(key)
        if existing is None or _mapping_priority(m) < _mapping_priority(existing):
            support[key] = m
    backend_by_key = {}
    for m in step23.get("mappings", []):
        key = (m.get("onnx_op", ""), norm_domain(m.get("onnx_domain", "onnx")))
        existing = backend_by_key.get(key)
        if existing is None or _mapping_priority(m) < _mapping_priority(existing):
            backend_by_key[key] = m

    model = onnx.load(str(model_path), load_external_data=False)
    counts = Counter(
        (n.op_type, norm_domain(n.domain or ""))
        for n, _scope in iter_model_nodes(model)
    )
    nodes_by_key = defaultdict(list)
    for n, _scope in iter_model_nodes(model):
        nodes_by_key[(n.op_type, norm_domain(n.domain or ""))].append(n)

    op_dist = []
    comp_rows = []
    full_types = partial_types = unsupported_types = 0
    supported_instances = unsupported_instances = 0

    for idx, ((op, dom), cnt) in enumerate(
        sorted(counts.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
    ):
        s = support.get((op, dom))
        b = backend_by_key.get((op, dom), {})
        dtypes = []
        step1_op = step1.get(op)
        if isinstance(step1_op, dict) and not op.startswith("_"):
            dtypes = [str(x) for x in (step1_op.get("data_types") or []) if x]

        if s:
            hip_op = s.get("hip_op", "")
            if hip_op.startswith("tensor."):
                status = "full"
                reason_texts = ["Handled at compile time."]
                reason_codes = ["COMPILE_TIME_TENSOR_OP"]
            elif not isinstance(step2hip.get(hip_op), dict):
                status = "partial"
                reason_codes = ["NO_HIP_DIALECT_IMPL"]
                reason_texts = ["No Hip Dialect implementation available."]
            else:
                status, reason_codes, reason_texts = analyze_schema_for_key(
                    nodes_by_key[(op, dom)], step2hip.get(hip_op), strict_attr_names
                )
            supported_instances += cnt
        else:
            hip_op = None
            status = "unsupported"
            rec, why = unsupported_rec(op)
            reason_texts = [why]
            reason_codes = (
                ["COMPILE_TIME_TENSOR_OP"]
                if rec == "Compile Time Optimization"
                else ["NO_HIP_DIALECT_IMPL"]
            )
            unsupported_instances += cnt

        if status == "full":
            full_types += 1
        elif status == "partial":
            partial_types += 1
        else:
            unsupported_types += 1

        backend = b.get("backend") if b else None
        runtime = b.get("runtime_func") if b else None
        desc = "—"
        if hip_op and isinstance(step2hip.get(hip_op), dict):
            desc = str(step2hip[hip_op].get("summary") or "—")
        if not desc or desc == "—":
            desc = onnx_op_description(op, dom)

        op_dist.append(
            {
                "onnx_op": op,
                "domain": dom,
                "count": int(cnt),
                "data_types": dtypes,
                "status": status,
                "hip_op": hip_op,
                "runtime_func": runtime,
                "backend": backend if backend else "Unknown",
                "op_description": desc,
            }
        )

        comp_rows.append(
            {
                "onnx_op": op,
                "domain": dom,
                "status": status,
                "reason_codes": reason_codes,
                "reason_texts": reason_texts,
                "evidence": [
                    {
                        "source_file": "step2_1_onnx_to_hip_mappings.json",
                        "json_pointer": f"/mappings/{idx}",
                    }
                ],
            }
        )

    mapping_chain = []
    for m in step23.get("mappings", []):
        mapping_chain.append(
            {
                "onnx_op": m.get("onnx_op", ""),
                "domain": norm_domain(m.get("onnx_domain", "onnx")),
                "hip_op": m.get("hip_op", ""),
                "runtime_func": m.get("runtime_func"),
                "backend": m.get("backend", "Unknown")
                if m.get("backend")
                else "Unknown",
            }
        )

    out = {
        "meta": {
            "model_path": str(model_path),
            "generated_at_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "repo_root": str(repo_root),
            "tool_versions": {"pipeline": "step1-step2_3"},
        },
        "summary": {
            "total_node_instances": int(sum(counts.values())),
            "supported_instances": int(supported_instances),
            "unsupported_instances": int(unsupported_instances),
            "total_operator_types": int(len(counts)),
            "fully_compatible_operator_types": int(full_types),
            "partially_compatible_operator_types": int(partial_types),
            "unsupported_operator_types": int(unsupported_types),
        },
        "operator_distribution": op_dist,
        "mapping_chain": mapping_chain,
        "compatibility": comp_rows,
        "reason_catalog": [
            {
                "code": "NO_HIP_DIALECT_IMPL",
                "default_text": "No Hip Dialect implementation available.",
            },
            {
                "code": "COMPILE_TIME_TENSOR_OP",
                "default_text": "Handled at compile time.",
            },
            {
                "code": "MISSING_HIP_REQUIRED_ATTR",
                "default_text": "Missing Hip-required attributes.",
            },
            {
                "code": "EXTRA_ONNX_ATTR_NOT_IN_HIP",
                "default_text": "Extra ONNX attributes not in Hip op.",
            },
            {
                "code": "ONNX_INPUT_BELOW_HIP_MIN",
                "default_text": "ONNX inputs below Hip minimum.",
            },
            {
                "code": "ONNX_INPUT_ABOVE_HIP_MAX",
                "default_text": "ONNX inputs above Hip maximum.",
            },
            {
                "code": "ONNX_OUTPUT_BELOW_HIP_MIN",
                "default_text": "ONNX outputs below Hip minimum.",
            },
            {
                "code": "ONNX_OUTPUT_ABOVE_HIP_MAX",
                "default_text": "ONNX outputs above Hip maximum.",
            },
        ],
    }
    out_path = analysis_dir / "report_input.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
