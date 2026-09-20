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
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from hip_source import HipSource

FALLBACK_OP_DESCRIPTIONS = {
    "MatMulNBits": "Quantized N-bit matrix multiplication (com.microsoft)",
    "RotaryEmbedding": "Rotary position embedding (RoPE)",
    "SkipSimplifiedLayerNormalization": "Skip connection + RMS normalization",
    "GroupQueryAttention": "Group Query Attention mechanism",
    "QMoE": "Mixture-of-Experts routing and fused expert computation",
}

# What the log says when a stage failed. An MLIR pass reports the op it choked
# on; a crash reports a frame, and the first one inside the repository names
# the code that has the bug.
_MLIR_ERROR_RE = re.compile(r"^.*?:\d+:\d+: error: (?P<message>.+)$", re.M)
_CRASH_CODE_RE = re.compile(r"Exception Code: (0x[0-9A-Fa-f]+)")
_CRASH_FRAME_RE = re.compile(
    r"#\d+\s+0x[0-9a-f]+\s+(?P<symbol>[^\n]*?)\s+"
    r"(?P<file>[A-Za-z]:\\[^\s]*?(?:Conversion|Dialect|Runtime)[^\s]*?):(?P<line>\d+)"
)
_EXIT_CODE_RE = re.compile(r"exit code (-?\d+)")
_ORT_ERROR_RE = re.compile(r"Error in ORT API: \d+, message: (?P<message>.+)")

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
    ("PIPELINE_FAILED", "A pipeline stage failed; the model does not compile."),
    (
        "SLICE_VERIFIED",
        "Converted on its own after the whole-graph conversion failed.",
    ),
    (
        "SLICE_UNSUPPORTED",
        "Did not convert on its own; no working conversion for this form.",
    ),
    (
        "SLICE_PROBE_FAILED",
        "The conversion failed on a single-operator module.",
    ),
    (
        "SLICE_NOT_BUILDABLE",
        "No single-operator module could be built; support is unknown.",
    ),
]

# Slice verdicts that mean the operator's own conversion works, and those that
# mean it does not. A slice says the operator converts in isolation, which is
# weaker than the whole-graph run: it does not say this model compiles, only
# that this operator is not what stops it.
_SLICE_SUPPORTED = ("converts", "compile_time")
_SLICE_BLOCKED = ("unsupported", "probe_failed")

# Anything outside the hip dialect is a compile-time fold rather than a
# runtime kernel, whichever dialect the conversion happened to pick.
_RUNTIME_DIALECT_PREFIX = "hip."

# What a slice produces no matter which operator it holds: a destination for a
# DPS result, a dimension read for a dynamic shape, a materialized constant
# operand, a host read of a shape scalar. They sit beside everything and so
# identify nothing. arith.* is excluded wholesale for the same reason -- scalar
# arithmetic is the index math around an operator, never its result.
_SLICE_MACHINERY_OPS = frozenset(
    {
        "hip.constant",
        "hip.get_constant",
        "hip.readback_scalar",
        "tensor.empty",
        "tensor.dim",
    }
)
_SLICE_MACHINERY_PREFIX = "arith."

# `<file>:3:12: error: ` in front of an MLIR diagnostic. The file is a slice in
# the run's own output directory, so quoting its path in the report adds a line
# of text and no information.
_MLIR_DIAG_RE = re.compile(r"^.*?:\d+:\d+:\s*(?:error|warning):\s*")


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
    # utf-8-sig: PowerShell writes a BOM, and pipeline_failure.json comes from
    # the orchestrator.
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def describe_failure(failure: dict) -> dict:
    """Turn a failed stage into the sentence a reader needs.

    Which step and why: an MLIR diagnostic names the operator it could not
    handle, a crash names the source line that faulted, and an importer error
    names the construct it does not implement. Everything is quoted from the
    log rather than summarized, because this is the finding.
    """
    log_path = Path(failure.get("log") or "")
    text = ""
    if log_path.is_file():
        text = log_path.read_text(encoding="utf-8", errors="replace")

    details = []
    headline = failure.get("message") or ""

    ort_error = _ORT_ERROR_RE.search(text)
    if ort_error:
        headline = ort_error.group("message").strip()
        details.append(headline)

    errors = [m.group("message").strip() for m in _MLIR_ERROR_RE.finditer(text)]
    if errors:
        headline = errors[0]
        details.extend(errors[:3])

    crash = _CRASH_CODE_RE.search(text)
    if crash:
        # The faulting frame is often a shared helper, so keep the few frames
        # below it too: the conversion pattern that called the helper is the
        # code to fix.
        frames = [
            f"{m.group('symbol').strip()} ({Path(m.group('file')).name}:{m.group('line')})"
            for m in _CRASH_FRAME_RE.finditer(text)
        ][:3]
        pattern = next(
            (
                m
                for m in _CRASH_FRAME_RE.finditer(text)
                if "Conversion.cpp" in m.group("file")
            ),
            None,
        )
        site = pattern or _CRASH_FRAME_RE.search(text)
        where = ""
        if site:
            where = f" in {Path(site.group('file')).name}:{site.group('line')}"
        details.extend(frames)
        headline = f"the compiler crashed{where} (exception {crash.group(1)})"

    exit_code = failure.get("exit_code")
    reported = _EXIT_CODE_RE.search(failure.get("message") or "")
    if reported:
        exit_code = int(reported.group(1))

    stage = failure.get("stage", "pipeline")
    return {
        "stage": stage,
        "command": failure.get("command", ""),
        "exit_code": exit_code,
        "headline": (headline or f"{stage} failed").rstrip("."),
        "details": details,
        "log": str(log_path) if log_path else "",
    }


def index_by_key(entries, key_fields=("op_type", "domain")):
    """Index rows on (op_type, normalized domain)."""
    indexed = {}
    for entry in entries or []:
        key = (entry.get(key_fields[0], ""), norm_domain(entry.get(key_fields[1], "")))
        indexed[key] = entry
    return indexed


def _prefer_prose(messages):
    """Drop pattern tags when the same operator also refused in words.

    A refusal is written either for a human ("dynamic spatial output dim
    requires an explicit kernel_shape") or as a marker of which pattern
    declined ("conv.rank_mismatch", "causal_conv.not_rank3"). An operator with
    a family of patterns collects one marker per sibling, and they say the same
    thing as a name check: this pattern is not the one. Keep them only when
    they are all there is.
    """
    prose = [m for m in messages if " " in m]
    return prose or messages


def index_slice_verdicts(payload):
    """Instances per verdict for each operator, from the slice probe.

    One operator can have several signatures with different verdicts, so the
    rows are summed rather than indexed: 72 of 96 SkipLayerNormalization
    instances converting and 24 not is the finding, not either number alone.
    """
    per_key = {}

    def entry_for(row):
        key = (row.get("op_type", ""), norm_domain(row.get("domain", "")))
        return per_key.setdefault(
            key,
            {
                "verdicts": {},
                "produced": set(),
                "refusals": [],
                "probe_errors": [],
                "unbuildable_reasons": [],
            },
        )

    for case in (payload or {}).get("cases") or []:
        entry = entry_for(case)
        verdict = case.get("verdict", "")
        entry["verdicts"][verdict] = entry["verdicts"].get(verdict, 0) + int(
            case.get("instances", 0)
        )
        # What the operator became, for the report's mapping table. Without a
        # whole-graph result attr_transfer.json does not exist, and a row
        # reading "supported" with no operation beside it says less than the
        # run knows.
        entry["produced"].update(case.get("produced_ops") or [])
        if verdict == "unsupported":
            for reason in case.get("refusals") or []:
                if reason not in entry["refusals"]:
                    entry["refusals"].append(reason)
        elif verdict == "probe_failed":
            # An error is the proximate cause and outranks the refusals logged
            # on the way to it. A pass can also fail without emitting one, and
            # then the refusals are all there is: the Conv conversion asks for
            # an explicit kernel_shape on a dynamic spatial dim and returns
            # failure silently, which would otherwise reach the report as an
            # operator that did not convert for no stated reason.
            error = _MLIR_DIAG_RE.sub("", case.get("reason", "")).strip()
            for text in [error] if error else case.get("refusals") or []:
                if text and text not in entry["probe_errors"]:
                    entry["probe_errors"].append(text)
    for entry in per_key.values():
        entry["refusals"] = _prefer_prose(entry["refusals"])
        entry["probe_errors"] = _prefer_prose(entry["probe_errors"])

    for row in (payload or {}).get("unbuildable") or []:
        entry = entry_for(row)
        entry["verdicts"]["unbuildable"] = entry["verdicts"].get(
            "unbuildable", 0
        ) + int(row.get("count", 0))
        # Why no slice could be built varies -- a region, an operand whose rank
        # the graph leaves open -- and the probe recorded which, so the report
        # does not have to assume the first one.
        reason = row.get("reason", "")
        if reason and reason not in entry["unbuildable_reasons"]:
            entry["unbuildable_reasons"].append(reason)
    return per_key


def classify_from_slices(
    count, verdicts, failure, refusals, probe_errors, unbuildable_reasons
):
    """Status for one operator from the single-operator conversions.

    Reached when the whole-graph conversion failed. Reporting every operator
    as unverified would read as "none of this works", when the run does know
    which operators convert and which of them is the one that stopped it.
    """
    supported = sum(verdicts.get(name, 0) for name in _SLICE_SUPPORTED)
    blocked = sum(verdicts.get(name, 0) for name in _SLICE_BLOCKED)
    unknown = verdicts.get("unbuildable", 0)
    stage = failure["stage"]

    codes, texts = [], []
    if verdicts.get("probe_failed"):
        codes.append("SLICE_PROBE_FAILED")
        said = " It reported: " + "; ".join(probe_errors) + "." if probe_errors else ""
        texts.append(
            f"The conversion failed on a module holding only this operator, "
            f"for {verdicts['probe_failed']} of {count} instance(s).{said}"
        )
    if verdicts.get("unsupported"):
        codes.append("SLICE_UNSUPPORTED")
        said = " It reported: " + "; ".join(refusals) + "." if refusals else ""
        texts.append(
            f"{verdicts['unsupported']} of {count} instance(s) did not convert "
            f"on their own, so no working conversion exists for that form.{said}"
        )
    if unknown:
        codes.append("SLICE_NOT_BUILDABLE")
        why = "; ".join(unbuildable_reasons) or "no slice could be built for them"
        texts.append(f"{unknown} of {count} instance(s) were not verified: {why}.")

    if blocked and supported:
        return (
            "partial",
            ["PARTIAL_INSTANCE_CONVERSION", *codes],
            [
                f"{supported} of {count} instance(s) converted on their own "
                f"and {blocked} did not.",
                *texts,
            ],
        )
    if blocked:
        return "unsupported", codes, texts
    if unknown and not supported:
        return "partial", codes, texts

    folded = verdicts.get("compile_time", 0)
    code = (
        "COMPILE_TIME_TENSOR_OP"
        if folded and not verdicts.get("converts")
        else ("SLICE_VERIFIED")
    )
    verified = (
        "Handled at compile time on a module holding only this operator."
        if code == "COMPILE_TIME_TENSOR_OP"
        else (
            "Converted on a module holding only this operator. The "
            f"{stage} step failed on the whole graph, so this says the "
            "operator is not what stopped it, not that the model compiles."
        )
    )
    return "full", [code, *codes], [verified, *texts]


def leftover_reason(reason_row, refusals):
    """Reason code and text for an operator no instance of which converted.

    A missing converter and a converter that refused every instance are
    different findings: the first needs a new implementation, the second needs
    an existing one widened, so they must not share a reason text.

    `refusals` is what the conversion printed when the operator went through
    on its own, so the constraint is quoted rather than inferred. A converter
    that refuses without a message leaves the finding without one, which is
    still the right code: the implementation exists.
    """
    if not reason_row or not reason_row.get("converter_found"):
        return "NO_HIP_DIALECT_IMPL", ["No Hip Dialect implementation available."]

    files = (
        ", ".join(reason_row.get("converter_files") or []) or "lib/Conversion/OnnxToHip"
    )
    text = f"A conversion exists ({files}) but refused every instance."
    if refusals:
        text += " It reported: " + "; ".join(refusals) + "."
    return "CONVERSION_REJECTED_INSTANCES", [text]


def classify(
    op, domain, count, leftover, attr_row, reason_row, probed, failure, slice_entry
):
    """Status, reason codes and texts for one operator type."""
    slice_verdicts = slice_entry.get("verdicts") or {}
    refusals = slice_entry.get("refusals") or []
    if failure:
        if slice_verdicts:
            return classify_from_slices(
                count,
                slice_verdicts,
                failure,
                refusals,
                slice_entry.get("probe_errors") or [],
                slice_entry.get("unbuildable_reasons") or [],
            )
        return (
            "partial",
            ["PIPELINE_FAILED"],
            [
                f"The {failure['stage']} step failed, so no operator was "
                f"verified: {failure['headline']}."
            ],
        )
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
        code, texts = leftover_reason(reason_row, refusals)
        return "unsupported", [code], texts

    codes, texts = [], []
    if leftover_count:
        codes.append("PARTIAL_INSTANCE_CONVERSION")
        detail = (
            " It reported: " + "; ".join(refusals) + "."
            if refusals
            else " The conversion pattern bailed out on them."
        )
        texts.append(
            f"{leftover_count} of {count} instance(s) were not converted.{detail}"
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


def primary_produced_op(produced):
    """What a single-operator conversion turned the operator into.

    A slice holds one operator, so everything it produced is that operator's,
    and a hip op among them is the one that will run. Falling back to the rest
    is what primary_hip_op does with a whole-graph result, and for the same
    reason: an operator that produced no hip op was handled at compile time,
    and naming the tensor op it became says so. Reporting nothing would read
    as "unknown", which is a weaker claim than the slice supports.
    """
    candidates = [
        name
        for name in sorted(produced or [])
        if name not in _SLICE_MACHINERY_OPS
        and not name.startswith(_SLICE_MACHINERY_PREFIX)
    ]
    for name in candidates:
        if name.startswith(_RUNTIME_DIALECT_PREFIX):
            return name
    return candidates[0] if candidates else None


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
    failure_record = load_optional(analysis_dir / "pipeline_failure.json")
    failure = describe_failure(failure_record) if failure_record else None
    probed = leftovers is not None
    # Read on both paths, for different things. Without a whole-graph result
    # the slices are the only per-operator verdict there is; with one they only
    # cover its leftovers, and what they contribute is the message the
    # conversion printed when it refused.
    slice_by_key = index_slice_verdicts(
        load_optional(analysis_dir / "slice_probe.json")
    )
    if slice_by_key and not runtime_map:
        # Which runtime function stands behind a hip op is written in its
        # lowering, so the lookup needs the op names and nothing from the whole
        # graph. Taking them from the slices keeps the recommendation naming
        # the wrapper that will run the operator instead of falling back to the
        # dialect op, which tells a reader nothing they cannot already see.
        # analyze_conversion, which normally rejects an incomplete tree, does
        # not run on this path, so an unusable one has to stay non-fatal: the
        # report is still worth producing without the wrapper names.
        source = HipSource(repo_root)
        if not source.missing_paths():
            produced = set()
            for entry in slice_by_key.values():
                produced.update(entry["produced"])
            runtime_map = source.runtime_map(keep_ops=produced)

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

        slice_entry = slice_by_key.get(key) or {}
        slice_verdicts = slice_entry.get("verdicts") or {}
        status, reason_codes, reason_texts = classify(
            op,
            domain,
            count,
            leftover,
            attr_row,
            reason_by_key.get(key),
            probed,
            failure,
            slice_entry,
        )
        leftover_count = int((leftover or {}).get("count", 0))

        total_instances += count
        if probed and not failure:
            unsupported_instances += leftover_count
            supported_instances += count - leftover_count
        elif slice_verdicts:
            # The whole-graph conversion failed, so these come from converting
            # the operator on its own. An instance no slice could be built for
            # stays in neither count.
            supported_instances += sum(
                slice_verdicts.get(name, 0) for name in _SLICE_SUPPORTED
            )
            unsupported_instances += sum(
                slice_verdicts.get(name, 0) for name in _SLICE_BLOCKED
            )
        # With neither nothing is known, so no instance is counted as
        # supported: a 100% headline on an unverified run would be read as a
        # result rather than as an absence of one.

        if status == "full":
            full_types += 1
        elif status == "partial":
            partial_types += 1
        else:
            unsupported_types += 1

        hip_op = primary_hip_op(attr_row) or primary_produced_op(
            slice_entry.get("produced")
        )
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
                "conversion_probed": str(bool(probed) and not failure).lower(),
                "support_evidence": (
                    "whole graph"
                    if probed and not failure
                    else "single-operator conversions"
                    if slice_by_key
                    else "none"
                ),
            },
            **({"failure": failure} if failure else {}),
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
