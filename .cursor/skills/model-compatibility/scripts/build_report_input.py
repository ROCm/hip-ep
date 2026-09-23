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
  lowering-broken  front end converts, back end does not follow -- add the
                   HipToLLVM lowering or the runtime function
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

# Worklist buckets are the statuses plus two kinds of entry that are not
# statuses: "unverified", where the operator counts as supported but half
# the check is missing, and "import-blocked", where the EP rejected the
# operator before any conversion ran. The latter is work on the importer,
# not on a HIP operator, so it cannot share the unsupported wording.
WORKLIST_BUCKETS = STATUSES + ("unverified", "import-blocked")

# What to do about each bucket, carried into the report so the worklist does
# not have to restate it.
STATUS_ACTION = {
    "supported": "",
    "partial": "handle the ignored attribute",
    "lowering-broken": "add the HipToLLVM lowering or runtime function",
    "blocked": "extend the existing operator",
    "unsupported": "implement the operator",
    "unverified": "check the lowering by hand",
    "import-blocked": "extend the MorphiZen ONNX importer",
}


def _norm_domain(domain: str) -> str:
    return "onnx" if domain in ("", "ai.onnx") else domain


def _unverified_reason(s2: dict) -> str:
    """Why an operator's lowering went unchecked, including how it failed.

    Two different things end up here. Either no slice could be built, and
    there is nothing but a reason, or a slice ran and failed on a shape we
    guessed, which is not enough to convict the lowering but is still an
    observed failure. Reporting only the reason in the second case hides
    that the compiler crashed.
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

    # The note says what the report cannot say generically -- the specific
    # cause. What a degraded level means is the renderer's fixed wording, so
    # do not repeat it here.
    probed = bool(s1_all.get("operators"))
    if not probed:
        evidence = "D"
        note = "Cause: " + (
            s1_all.get("meta", {}).get("error") or "the probe did not run"
        )
    elif s1_all.get("meta", {}).get("mode") == "single-op":
        evidence = "C"
        blocked_at_import = sorted(
            k for k, v in s2_all.items() if v.get("status") == "import_failed"
        )
        note = "Cause: the model could not be imported as a whole" + (
            f" ({', '.join(blocked_at_import)} were rejected by the importer)"
            if blocked_at_import
            else ""
        )
    elif s1_all.get("meta", {}).get("failed"):
        evidence = "B"
        note = "Cause: " + (s1_all["meta"].get("error") or "unknown")
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
    counts = {k: 0 for k in STATUSES}
    type_counts = {k: 0 for k in STATUSES}
    # Counted apart from the statuses: these are inside the supported totals,
    # and saying so keeps the headline figure from overstating what was
    # checked.
    unverified_instances = 0
    unverified_types = 0
    # Likewise counted apart: these are spread across blocked and unsupported
    # depending on the documentation, but the work is the same and it is not
    # in this tree.
    import_blocked_instances = 0
    import_blocked_types = 0

    for op, info in ops.items():
        if op.startswith("_"):
            continue
        # Normalize here rather than trusting the producer: mlir_op_parser
        # already folds ai.onnx into onnx, step1_onnx_parser reports the
        # domain verbatim, and step1 only becomes the primary input on the
        # fallback path -- where the mismatch would silently miss every
        # standard-domain operator in the lookup.
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
        # name: stage2 lowered the operator all the way and no runtime symbol
        # appeared. Operators stage1 never converted have no stage2 result,
        # so they keep the documented implementation instead.
        if s2.get("status") == "ok" and s2.get("compile_time"):
            backend = "Compile-time"
        elif runtime:
            backend = f"{doc_impl} (`{runtime}`)" if doc_impl else f"`{runtime}`"
        else:
            backend = doc_impl

        count = info["count"]
        counts[status] += count
        type_counts[status] += 1

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

        # An unverified lowering is not a defect, so it is not one of the
        # statuses, but it is still something a person has to decide about.
        # It earns a worklist row on its own.
        bucket = status
        if s2.get("status") == "import_failed":
            # The EP never saw the operator: its own front end turned the
            # model away. Naming this "implement the operator" would send
            # someone to the wrong repository.
            bucket = "import-blocked"
            import_blocked_instances += count
            import_blocked_types += 1
        elif status == "supported" and s2.get("status") == "not_sliceable":
            bucket = "unverified"
            unverified_instances += count
            unverified_types += 1
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

    stages: list[dict] = []
    if args.stages and args.stages.exists():
        try:
            loaded = json.loads(args.stages.read_text(encoding="utf-8"))
            stages = loaded if isinstance(loaded, list) else [loaded]
        except Exception:
            stages = []
    if probed:
        s1_meta = s1_all.get("meta", {})
        unconv = s1_meta.get("diagnostic") or {}
        if s1_meta.get("mode") == "single-op":
            # There was no whole-graph pass to report on. What matters here
            # is how many operators got as far as the converter at all.
            rejected = [
                k for k, r in s2_all.items() if r.get("status") == "import_failed"
            ]
            total = s1_meta.get("operators_probed", len(s2_all))
            s1_result = "ok"
            s1_detail = (
                f"one-node models, {total - len(rejected)}/{total} operator type(s) "
                "imported"
            )
            if rejected:
                s1_detail += f"; the importer rejected {', '.join(sorted(rejected))}"
        elif s1_meta.get("failed"):
            s1_result = "failed"
            s1_detail = s1_meta.get("error") or "whole-graph conversion failed"
            if s1_meta.get("fallback"):
                s1_detail += f" — fell back to {s1_meta['fallback']}"
        else:
            # A pass that leaves operators unconverted still succeeded; that
            # is how it reports missing converters.
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
            # Some of these ran and crashed the compiler, which the bare
            # count would bury.
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
            "ep_input_path": args.ep_input_path or str(args.ops_json),
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
