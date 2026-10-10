#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Assemble the normalized input the report renders from.

Pure table lookup and arithmetic. Every verdict was already reached by the
probe (did it convert, does it lower, does the converter read its
attributes) or by the supported-operations doc (does an implementation
exist); nothing is inferred here.

The five statuses each name a different piece of work, which is the point of
separating them:

  supported        nothing to do
  partial          converts, but the converter ignores an attribute the
                   model sets -- teach it that attribute
  lowering-broken  converts, but some pass after that does not finish --
                   the error names which one
  blocked          no conversion, but an implementation exists -- relax a
                   dtype, shape or attribute restriction in it
  unsupported      no implementation -- write the operator

Usage:
  python build_report_input.py <ep_input_ops.json> <probe_result.json>
                               <repo_root> <out_dir> [options]
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from supported_ops_doc import default_doc_path, load_supported_ops  # noqa: E402

STATUSES = ("supported", "partial", "lowering-broken", "blocked", "unsupported")

# Two buckets are not statuses: "unverified" operators count as supported
# with half the check missing, and "import-blocked" ones the EP refused
# before any conversion ran -- work on the importer, not on a HIP operator.
WORKLIST_BUCKETS = STATUSES + ("unverified", "import-blocked")

# What to do about each bucket, carried into the report so the worklist does
# not have to restate it.
STATUS_ACTION = {
    "supported": "",
    "partial": "handle the ignored attribute",
    "lowering-broken": "fix the pass the error names",
    "blocked": "extend the existing operator",
    "unsupported": "implement the operator",
    "unverified": "check the lowering by hand",
    "import-blocked": "extend the MorphiZen ONNX importer",
}


def onnx_signature(inst: dict) -> str:
    """An ONNX node's operand and result types, read like an MLIR signature.

    Used when there is no EP input to quote a line from.
    """

    def one(tensor: dict) -> str:
        dtype = tensor.get("dtype") or "?"
        dims = tensor.get("shape") or []
        if not dims:
            return dtype
        shape = "x".join("?" if d in (None, "", 0) else str(d) for d in dims)
        return f"{dtype}[{shape}]"

    parts = [one(t) for t in inst.get("inputs", [])]
    results = [one(t) for t in inst.get("outputs", [])]
    if not any(p != "?" for p in parts + results):
        # Every type unknown, which is what a sequence operand looks like
        # here. "(?) -> ?" is worse than saying nothing.
        return ""
    ins = ", ".join(parts)
    return f"({ins}) -> {', '.join(results)}" if results else f"({ins})"


def _norm_domain(domain: str) -> str:
    return "onnx" if domain in ("", "ai.onnx") else domain


def _unverified_reason(s2: dict) -> str:
    """Why an operator's lowering went unchecked, including how it failed.

    Either no slice could be built, leaving only a reason, or one ran and
    failed on a guessed shape -- not enough to convict the lowering, but an
    observed failure the reader should still see.
    """
    if s2.get("status") != "not_sliceable":
        return ""
    reason = s2.get("reason", "")
    error = s2.get("error", "")
    if error and error not in reason:
        return f"{reason} — {error}" if reason else error
    return reason


def type_signature(line: str) -> str:
    """The `(operand types) -> result types` tail of an operation line.

    For a blocked operator this is usually where the reason lives:
    MatMulNBits is rejected for f32 zero_points, which is visible here and
    nowhere else in the report.
    """
    line = re.sub(r"\s*loc\(#loc\d+\)\s*$", "", line.rstrip())
    m = re.search(r":\s*(\([^:]*\)\s*->\s*.+)$", line)
    return " ".join(m.group(1).split()) if m else ""


def humanize_camel(name: str) -> str:
    out: list[str] = []
    for i, ch in enumerate(name):
        if i and ch.isupper() and not name[i - 1].isupper():
            out.append(" ")
        out.append(ch)
    return "".join(out)


def op_description(op: str, domain: str) -> str:
    """First sentence of the ONNX schema doc, else the name made readable.

    com.microsoft operators have no schema registered in the onnx package, so
    they take the fallback.
    """
    if domain in ("", "onnx", "ai.onnx"):
        try:
            from onnx import defs

            doc = " ".join((defs.get_schema(op, domain="").doc or "").split())
            if doc:
                return doc.split(". ")[0].strip().rstrip(".")
        except Exception:
            pass
    return humanize_camel(op)


def load_explicit_attrs(model_path: Path) -> dict[str, set[str]] | None:
    """Attribute names each node actually sets in the model file.

    ORT fills in ONNX schema defaults during Graph::Resolve, so an attribute
    present in the EP input MLIR does not mean the model author wrote it --
    Reshape ships with no attributes at all and arrives carrying allowzero.
    Only the file on disk distinguishes the two.
    """
    try:
        import onnx

        from onnx_graph_walk import iter_model_nodes

        model = onnx.load(str(model_path), load_external_data=False)
        out: dict[str, set[str]] = {}
        for node, _scope in iter_model_nodes(model):
            if node.name:
                out[node.name] = {a.name for a in node.attribute}
        return out
    except Exception:
        return None


def classify(
    s1: dict,
    s2: dict,
    in_doc: bool,
    instances: list[dict],
    explicit: dict[str, set[str]] | None,
) -> tuple[str, list[dict], list[dict]]:
    """Apply the status matrix.

    Returns (status, ignored attributes the model sets, all ignored ones).
    An ignored attribute is only a defect in *this* model if the model sets
    it; otherwise it is a gap that a different model would hit.
    """
    if not s1:
        # Nothing was compiled, so the documentation is all there is. It
        # cannot distinguish an implementation that exists from one that
        # accepts this model, which is why this path is evidence level D.
        return ("supported" if in_doc else "unsupported"), [], []

    if s1.get("unconverted"):
        # No converter ran. Whether an implementation exists decides between
        # extending one and writing one.
        return ("blocked" if in_doc else "unsupported"), [], []

    if s1.get("inconclusive") and not s1.get("converted"):
        # The operator was reached but nothing could be proved either way --
        # no usable slice could be built from it. The documentation is all
        # that is left; the worklist records that the check did not happen.
        return ("supported" if in_doc else "unsupported"), [], []

    if (s2 or {}).get("status") in ("lowering_broken", "timeout"):
        return "lowering-broken", [], []

    if (s2 or {}).get("status") == "not_sliceable":
        # stage1 converted it; stage2 could not build a standalone module to
        # check the rest of the chain. Trust what was observed and record
        # that the lowering is unverified rather than claiming either way.
        return "supported", [], []

    ignored = [
        f for f in (s2 or {}).get("attributes", []) if f.get("verdict") == "ignored"
    ]
    if not ignored:
        return "supported", [], []

    if explicit is None:
        # Without the original model there is no way to tell a default that
        # ORT supplied from a value the author chose. Assume the worse one.
        return "partial", ignored, ignored

    set_by_model = {
        f["attribute"]
        for f in ignored
        for inst in instances
        if f["attribute"] in explicit.get(inst.get("name", ""), set())
    }
    relevant = [f for f in ignored if f["attribute"] in set_by_model]
    return ("partial" if relevant else "supported"), relevant, ignored


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble report_input.json")
    ap.add_argument("ops_json", type=Path, help="ep_input_ops.json")
    ap.add_argument("probe_json", type=Path, help="probe_result.json")
    ap.add_argument("repo_root", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--model-path", default="")
    ap.add_argument("--ep-input-path", default="")
    ap.add_argument("--supported-ops-doc", type=Path, default=None)
    ap.add_argument(
        "--original-model",
        type=Path,
        default=None,
        help="Original .onnx, used to tell author-set attributes from ORT-filled "
        "defaults. Without it every ignored attribute counts as a defect.",
    )
    ap.add_argument(
        "--stages",
        type=Path,
        default=None,
        help="pipeline_stages.json from the orchestrator; the probe stages are "
        "appended from probe_result.json.",
    )
    args = ap.parse_args()

    ops = json.loads(args.ops_json.read_text(encoding="utf-8"))
    probe = json.loads(args.probe_json.read_text(encoding="utf-8"))
    doc_path = args.supported_ops_doc or default_doc_path(args.repo_root)
    doc = load_supported_ops(doc_path) if doc_path.exists() else {}

    meta_in = ops.get("_analysis_meta", {})
    s1_all = probe.get("stage1", {})
    s2_all = probe.get("stage2", {}).get("operators", {})

    original = args.original_model or (
        Path(args.model_path) if args.model_path else None
    )
    explicit = load_explicit_attrs(original) if original and original.exists() else None

    mlir_lines: list[str] = []
    if args.ep_input_path:
        ep_mlir = Path(args.ep_input_path)
        if ep_mlir.suffix == ".mlir" and ep_mlir.exists():
            mlir_lines = ep_mlir.read_text(encoding="utf-8").splitlines()

    # The note carries the specific cause only; what a degraded level means
    # in general is the renderer's fixed wording.
    probed = bool(s1_all.get("operators"))
    s1_meta = s1_all.get("meta", {})
    if not probed:
        evidence = "D"
        note = "Cause: " + (s1_meta.get("error") or "the probe did not run")
    elif s1_meta.get("mode") == "single-op":
        evidence = "C"
        rejected = sorted(
            k for k, v in s2_all.items() if v.get("status") == "import_failed"
        )
        note = "Cause: the model could not be imported as a whole" + (
            f" ({', '.join(rejected)} were rejected by the importer)"
            if rejected
            else ""
        )
    elif s1_meta.get("failed"):
        evidence = "B"
        note = "Cause: " + (s1_meta.get("error") or "unknown")
    elif explicit is None:
        evidence = "A"
        note = (
            "The original model was unavailable, so every ignored attribute is "
            "counted as a defect: ORT-filled defaults cannot be told from "
            "author-set values."
        )
    else:
        evidence, note = "A", ""

    rows: list[dict] = []
    worklist: dict[str, list] = {k: [] for k in WORKLIST_BUCKETS if k != "supported"}
    capability_gaps: list[dict] = []
    doc_stale: list[dict] = []

    for op, info in ops.items():
        if op.startswith("_"):
            continue
        # The two producers disagree: mlir_op_parser folds ai.onnx into onnx,
        # step1_onnx_parser reports it verbatim. Normalizing on the way in
        # keeps the fallback path from missing every standard-domain lookup.
        domain = _norm_domain((info.get("domain") or ["onnx"])[0])
        s1 = s1_all.get("operators", {}).get(op, {})
        s2 = s2_all.get(op, {})
        in_doc = (op, domain) in doc
        instances = info.get("instances") or []
        status, ignored, all_ignored = classify(s1, s2, in_doc, instances, explicit)

        targets = s1.get("targets", {})
        target = next(iter(targets), None)
        if target is None and s1.get("folded"):
            target = "(folded at compile time)"
        runtime = (s2.get("runtime_funcs") or [None])[0]
        doc_impl = doc.get((op, domain), {}).get("impl", "")
        # "Compile-time" is the probe's finding, not a guess from the target
        # name: stage2 lowered it fully and no runtime symbol appeared.
        if s2.get("status") == "ok" and s2.get("compile_time"):
            backend = "Compile-time"
        elif runtime:
            backend = f"{doc_impl} (`{runtime}`)" if doc_impl else f"`{runtime}`"
        else:
            backend = doc_impl

        count = info["count"]
        rows.append(
            {
                "onnx_op": op,
                "domain": domain,
                "count": count,
                "data_types": info.get("data_types", []),
                # Dynamic shapes are a common reason a converter that exists
                # still rejects a model, so they belong next to the dtypes.
                "shape_types": info.get("shape_types", []),
                "status": status,
                "target": target,
                "runtime_func": runtime,
                "backend": backend,
                "compile_time": bool(
                    s2.get("status") == "ok" and s2.get("compile_time")
                ),
                "lowering_unverified": s2.get("status") == "not_sliceable",
                "lowering_unverified_reason": _unverified_reason(s2),
                "op_description": op_description(op, domain),
                "ignored_attributes": [f["attribute"] for f in ignored],
            }
        )

        bucket = status
        if s2.get("status") == "import_failed":
            bucket = "import-blocked"
        elif status == "supported" and s2.get("status") == "not_sliceable":
            bucket = "unverified"
        if bucket != "supported":
            item = {
                "onnx_op": op,
                "domain": domain,
                "count": count,
                "action": STATUS_ACTION[bucket],
                "data_types": info.get("data_types", []),
                "shape_types": info.get("shape_types", []),
            }
            first = instances[0] if instances else {}
            item["attributes"] = first.get("attributes", {})
            item["source_line"] = first.get("line_no")
            if mlir_lines and first.get("line_no"):
                idx = first["line_no"] - 1
                if 0 <= idx < len(mlir_lines):
                    item["signature"] = type_signature(mlir_lines[idx])
            elif first.get("inputs") or first.get("outputs"):
                signature = onnx_signature(first)
                if signature:
                    item["signature"] = signature
            if status == "lowering-broken":
                item["error"] = s2.get("error", "")
            if bucket == "unverified":
                item["error"] = _unverified_reason(s2)
                item["implementation"] = doc_impl
            if bucket == "import-blocked":
                item["error"] = s2.get("error", "")
            if status == "partial":
                item["ignored_attributes"] = [
                    {"name": f["attribute"], "value": f.get("value")} for f in ignored
                ]
            if status == "blocked":
                item["implementation"] = doc_impl
                attribution = s2.get("attribution") or {}
                if attribution.get("found"):
                    item["blocking_operand"] = attribution["summary"]
                elif attribution.get("reason"):
                    item["blocking_operand"] = f"not isolated: {attribution['reason']}"
            worklist[bucket].append(item)

        # An ignored attribute is a real gap whether or not this model sets
        # it -- another model will. The status above says whether it bites
        # here; this list says the capability is missing either way.
        for f in all_ignored:
            capability_gaps.append(
                {
                    "onnx_op": op,
                    "domain": domain,
                    "target": target,
                    "attribute": f["attribute"],
                    "value": f.get("value"),
                    "set_by_model": f in ignored,
                }
            )

        if status == "supported" and not in_doc:
            doc_stale.append({"onnx_op": op, "domain": domain, "target": target})

    for key in worklist:
        worklist[key].sort(key=lambda x: -x["count"])
    rows.sort(key=lambda r: (-r["count"], r["onnx_op"]))

    # Derived, not accumulated: a running total kept alongside the data it
    # summarises is one more thing to forget to update. The last two sit
    # inside the status totals, so the report states them separately rather
    # than counting them twice.
    counts = {k: sum(r["count"] for r in rows if r["status"] == k) for k in STATUSES}
    type_counts = {k: sum(1 for r in rows if r["status"] == k) for k in STATUSES}

    def bucket_size(name: str) -> tuple[int, int]:
        items = worklist.get(name) or []
        return sum(i["count"] for i in items), len(items)

    unverified_instances, unverified_types = bucket_size("unverified")
    import_blocked_instances, import_blocked_types = bucket_size("import-blocked")

    stages: list[dict] = []
    if args.stages and args.stages.exists():
        try:
            loaded = json.loads(args.stages.read_text(encoding="utf-8"))
            stages = loaded if isinstance(loaded, list) else [loaded]
        except Exception:
            stages = []
    if probed:
        if s1_meta.get("mode") == "single-op":
            # No whole-graph pass to report on; what matters is how many
            # operators reached the converter at all.
            rejected = sorted(
                k for k, r in s2_all.items() if r.get("status") == "import_failed"
            )
            probed_types = s1_meta.get("operators_probed", len(s2_all))
            s1_result = "ok"
            s1_detail = (
                f"one-node models, {probed_types - len(rejected)}/{probed_types} "
                "operator type(s) imported"
            )
            if rejected:
                s1_detail += f"; the importer rejected {', '.join(rejected)}"
        elif s1_meta.get("failed"):
            s1_result = "failed"
            s1_detail = s1_meta.get("error") or "whole-graph conversion failed"
            if s1_meta.get("fallback"):
                s1_detail += f" — fell back to {s1_meta['fallback']}"
        else:
            # A pass that leaves operators unconverted still succeeded; that
            # is how it reports missing converters.
            unconv = s1_meta.get("diagnostic") or {}
            s1_result = "ok"
            s1_detail = (
                f"{len(unconv)} operator type(s) have no converter"
                if unconv
                else "every operator converted"
            )
        stages.append(
            {
                "stage": "convert-onnx-to-hip (stage1)",
                "result": s1_result,
                "detail": s1_detail,
            }
        )

        ok = sum(1 for r in s2_all.values() if r.get("status") == "ok")
        broken = [k for k, r in s2_all.items() if r.get("status") == "lowering_broken"]
        unsliceable = [
            k for k, r in s2_all.items() if r.get("status") == "not_sliceable"
        ]
        detail = f"{ok} operator(s) lowered"
        if broken:
            detail += f"; {len(broken)} broken ({', '.join(broken)})"
        if unsliceable:
            # Name how each one failed, not just that it was not verified.
            # Some of these crashed the compiler, which a bare count buries.
            named = ", ".join(
                f"{k}: {s2_all[k]['error']}" if s2_all[k].get("error") else k
                for k in unsliceable
            )
            detail += f"; {len(unsliceable)} not verified ({named})"
        stages.append(
            {
                "stage": "hip-to-llvm (stage2)",
                "result": "failed" if broken else "ok",
                "detail": detail,
            }
        )

    total = sum(counts.values())
    out = {
        "meta": {
            "model_path": args.model_path,
            # Empty when there was no dump. Naming the operator-count JSON
            # here instead would say the EP was given something it never saw.
            "ep_input_path": args.ep_input_path,
            "generated_at_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "evidence_level": evidence,
            "evidence_note": note,
            "supported_ops_doc": str(doc_path),
        },
        "summary": {
            "total_node_instances": total,
            "weight_constants_excluded": meta_in.get("weight_constants", 0),
            "total_operator_types": sum(type_counts.values()),
            "instances": {k: counts[k] for k in STATUSES},
            "operator_types": {k: type_counts[k] for k in STATUSES},
            "lowering_unverified_instances": unverified_instances,
            "lowering_unverified_types": unverified_types,
            "import_blocked_instances": import_blocked_instances,
            "import_blocked_types": import_blocked_types,
            "supported_pct": round(counts["supported"] / total * 100, 1)
            if total
            else 0.0,
        },
        "pipeline_stages": stages,
        "operator_distribution": rows,
        "worklist": worklist,
        "capability_gaps": capability_gaps,
        "doc_stale": doc_stale,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "report_input.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    s = out["summary"]
    print(f"evidence level: {evidence}" + (f" ({note})" if note else ""))
    print(
        f"total instances: {total}  (+{s['weight_constants_excluded']} weight constants excluded)"
    )
    for k in STATUSES:
        print(f"  {k:16s} {counts[k]:6d} instances  {type_counts[k]:3d} types")
    print(f"supported: {s['supported_pct']}%")
    print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
