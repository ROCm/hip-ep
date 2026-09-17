#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Compatibility oracle: what convert-onnx-to-hip actually did (steps 2 to 4).

Compares the EP-input MLIR with the same module after the conversion pipeline
and writes what the report is built from:

  leftover_onnx.json     onnx ops the conversion did not replace. The pass DCEs
                         unused onnx ops before it finishes, so anything left
                         is live: the operator is unsupported, or its pattern
                         bailed out on these instances.

  leftover_reasons.json  for each of those, whether a converter exists at all
                         and which of its refusals the observed types
                         contradict. Read from the source through hip_source,
                         because a release build compiles the refusal messages
                         out.

  attr_transfer.json     for converted ops, the ONNX attributes that did not
                         reach the HIP op, and which hip op each became.

  hip_runtime_map.json   the runtime function and backend behind each hip op
                         that appeared, so the report can name what executes
                         the operator.

An attribute equal to its schema default is recorded separately, because
dropping a default changes no behaviour; defaults come from the ONNX schema,
or from ONNX Runtime's contrib operator registry for com.microsoft.

Both dumps must be produced with --mlir-print-debuginfo so the locations line
up; the input dump is re-printed from the same file the conversion read.
"""

import argparse
import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from hip_source import HipSource
from mlir_text import parse_mlir_file, strip_quotes

# Anything that is no longer an onnx op is a conversion result: hip.* is the
# runtime path, and the rest are compile-time folds (a Reshape that became a
# static tensor.expand_shape, index arithmetic, and so on). Listing the
# expected dialects instead would silently drop a pairing the day a conversion
# starts emitting one that is not on the list.
_ONNX_DIALECT = "onnx"

_SCALAR_RE = re.compile(r"^\s*(-?[\d.eE+]+|\"[^\"]*\")")


def _primary_op(ops):
    """The op that best represents what an ONNX op became."""
    for op in ops:
        if op.dialect == "hip":
            return op
    return ops[0] if ops else None


def _attr_scalar(value: str):
    """Leading scalar of an MLIR attribute value (`0 : si64` -> 0)."""
    match = _SCALAR_RE.match(value or "")
    if not match:
        return None
    token = match.group(1)
    if token.startswith('"'):
        return strip_quotes(token)
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return None


def _attribute_value(proto_bytes):
    """Decode a serialized AttributeProto default."""
    from onnx import AttributeProto, helper

    proto = AttributeProto()
    proto.ParseFromString(proto_bytes)
    value = helper.get_attribute_value(proto)
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else value


@lru_cache(maxsize=1)
def _contrib_schemas():
    """Contrib operator schemas, keyed by (domain, op type).

    ONNX Runtime registers the contrib operators it implements, defaults
    included, so their defaults come from the same place the model's producer
    took them from. A hand-kept table would cover only the operators seen so
    far, and would be wrong wherever a default was guessed.
    """
    try:
        from onnxruntime.capi._pybind_state import get_all_operator_schema
    except ImportError:
        return {}
    try:
        return {(s.domain, s.name): s for s in get_all_operator_schema()}
    except Exception:
        return {}


def _contrib_defaults(op_type: str, domain: str):
    schema = _contrib_schemas().get((domain, op_type))
    if schema is None:
        return {}
    defaults = {}
    for name, attr in (schema.attributes or {}).items():
        raw = getattr(attr, "_default_value", None)
        # No recorded default means the operator requires a value, so dropping
        # one is a real finding rather than a no-op.
        if not raw:
            continue
        try:
            defaults[name] = _attribute_value(raw)
        except Exception:
            continue
    return defaults


def _schema_defaults(op_type: str, domain: str):
    """Attribute defaults from the ONNX schema, or the contrib registry."""
    try:
        from onnx import defs, helper
    except ImportError:
        return _contrib_defaults(op_type, domain)
    try:
        schema = defs.get_schema(
            op_type, domain="" if domain in {"", "onnx", "ai.onnx"} else domain
        )
    except Exception:
        return _contrib_defaults(op_type, domain)
    defaults = {}
    for name, attr in (schema.attributes or {}).items():
        proto = getattr(attr, "default_value", None)
        if proto is None or not proto.name and proto.type == 0:
            continue
        try:
            value = helper.get_attribute_value(proto)
        except Exception:
            continue
        if isinstance(value, bytes):
            value = value.decode("utf-8", "replace")
        defaults[name] = value
    return defaults


def _is_default(value: str, default) -> bool:
    if default is None:
        return False
    scalar = _attr_scalar(value)
    if scalar is None:
        return False
    if isinstance(default, float) or isinstance(scalar, float):
        try:
            return abs(float(scalar) - float(default)) < 1e-9
        except (TypeError, ValueError):
            return False
    return scalar == default


def collect_leftovers(post_ops):
    groups = defaultdict(lambda: {"count": 0, "samples": []})
    for op in post_ops:
        op_type, domain = op.onnx_key()
        key = f"{domain}.{op_type}"
        entry = groups[key]
        entry["count"] += 1
        if len(entry["samples"]) < 5:
            entry["samples"].append(
                {
                    "node_name": strip_quotes(op.attrs.get("onnx_node_name", "")),
                    "mlir_op": op.name,
                    "type_signature": op.type_signature,
                }
            )
        entry["op_type"] = op_type
        entry["domain"] = domain

    return [
        {
            "key": key,
            "op_type": entry["op_type"],
            "domain": entry["domain"],
            "count": entry["count"],
            "samples": entry["samples"],
        }
        for key, entry in sorted(groups.items(), key=lambda kv: -kv[1]["count"])
    ]


def orphan_hip_ops(pre_module, post_module):
    """Hip ops whose location matches no pre-conversion onnx op.

    Carriers count as pre ops: hip.constant comes from onnx.Constant, so
    excluding those would make every carrier look unexplained.

    When this is empty, every runtime op in the module is accounted for, which
    turns "this onnx op has no pair" into proof that it produced no runtime op
    rather than a suspicion that the pairing missed one.
    """
    accounted = {
        pre_module.resolve_loc(op)
        for op in pre_module.onnx_ops(include_non_compute=True)
    }
    accounted.discard("")

    orphans = defaultdict(int)
    for op in post_module.ops:
        if not op.name.startswith("hip."):
            continue
        if post_module.resolve_loc(op) not in accounted:
            orphans[op.name] += 1
    return dict(sorted(orphans.items()))


def build_attr_transfer(
    pre_module, post_module, leftover_locs, pairing_is_complete, hip_op_attributes
):
    post_by_loc = defaultdict(list)
    for op in post_module.ops:
        if op.dialect == _ONNX_DIALECT:
            continue
        loc = post_module.resolve_loc(op)
        if loc:
            post_by_loc[loc].append(op)

    defaults_cache = {}
    per_key = defaultdict(
        lambda: {
            "paired": 0,
            "folded": 0,
            "compile_time": 0,
            "hip_ops": set(),
            "dropped_attrs": defaultdict(int),
            "dropped_attr_values": defaultdict(set),
            "dropped_default_attrs": defaultdict(int),
            "samples": [],
        }
    )
    unpaired = defaultdict(int)

    for op in pre_module.onnx_ops():
        loc = pre_module.resolve_loc(op)
        if not loc or loc in leftover_locs:
            continue
        key = op.onnx_key()

        candidates = post_by_loc.get(loc)
        if not candidates:
            # A conversion that rewrites an op into a chain can hand the new
            # ops a neighbour's location, so a missing pair is only proof of a
            # fold when nothing else in the module is unaccounted for.
            if pairing_is_complete:
                per_key[key]["folded"] += 1
            else:
                unpaired[f"{key[1]}.{key[0]}"] += 1
            continue

        entry = per_key[key]
        entry["paired"] += 1
        primary = _primary_op(candidates)
        if primary:
            entry["hip_ops"].add(primary.name)

        # A conversion that produced no hip op folded this operator into
        # structure: Split's axis becomes extract_slice offsets, Concat's
        # becomes the destination offsets. There is no attribute dictionary
        # left to check, and its absence says nothing about correctness.
        if primary is None or not primary.name.startswith("hip."):
            entry["compile_time"] += 1
            continue

        landed = set(hip_op_attributes.get(primary.name) or {})
        for candidate in candidates:
            landed.update(candidate.attrs.keys())

        if key not in defaults_cache:
            defaults_cache[key] = _schema_defaults(*key)
        defaults = defaults_cache[key] or {}

        dropped = []
        for name in op.onnx_attr_names():
            if name in landed:
                continue
            if _is_default(op.attrs[name], defaults.get(name)):
                entry["dropped_default_attrs"][name] += 1
            else:
                entry["dropped_attrs"][name] += 1
                # The value matters to the reader: a schema that declares no
                # default leaves "harmless" unprovable, so the report has to
                # show what was actually set.
                scalar = _attr_scalar(op.attrs[name])
                entry["dropped_attr_values"][name].add(
                    str(scalar if scalar is not None else op.attrs[name])
                )
                dropped.append(name)

        if dropped and len(entry["samples"]) < 3:
            entry["samples"].append(
                {
                    "node_name": strip_quotes(op.attrs.get("onnx_node_name", "")),
                    "hip_op": primary.name if primary else "",
                    "dropped": {name: op.attrs[name] for name in dropped},
                }
            )

    rows = []
    for (op_type, domain), entry in sorted(per_key.items()):
        rows.append(
            {
                "key": f"{domain}.{op_type}",
                "op_type": op_type,
                "domain": domain,
                "paired_instances": entry["paired"],
                "folded_instances": entry["folded"],
                "compile_time_instances": entry["compile_time"],
                "hip_ops": sorted(entry["hip_ops"]),
                "dropped_attrs": dict(sorted(entry["dropped_attrs"].items())),
                "dropped_attr_values": {
                    name: sorted(values)
                    for name, values in sorted(entry["dropped_attr_values"].items())
                },
                "dropped_default_attrs": dict(
                    sorted(entry["dropped_default_attrs"].items())
                ),
                "status": "partial" if entry["dropped_attrs"] else "full",
                "samples": entry["samples"],
            }
        )
    return rows, dict(sorted(unpaired.items()))


def observed_element_types(entry):
    """Element types on the leftover instances' operands and results."""
    found = []
    for sample in entry.get("samples") or []:
        for body in re.findall(r"tensor<([^<>]*)>", sample.get("type_signature", "")):
            elem = body.rsplit("x", 1)[-1].strip() if "x" in body else body.strip()
            if elem and elem not in found:
                found.append(elem)
    return found


def explain_leftovers(source, leftovers, max_refusals):
    """Why each leftover did not convert: no converter, or a refused one."""
    rows = []
    for entry in leftovers:
        observed = observed_element_types(entry)
        explanation = source.explain_leftover(
            entry.get("op_type", ""), observed, max_refusals
        )
        rows.append(
            {
                "key": entry.get("key"),
                "op_type": entry.get("op_type", ""),
                "domain": entry.get("domain", ""),
                "observed_element_types": observed,
                **explanation,
            }
        )
    return rows


def write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_mlir", help="EP-input MLIR printed with locations")
    ap.add_argument(
        "converted_mlir", help="post-conversion MLIR printed with locations"
    )
    ap.add_argument("output_dir", help="Directory for the analysis JSON files")
    ap.add_argument(
        "repo_root",
        help="hip-ep repository root: which attributes a hip op declares, what "
        "runs it, and what its converter can refuse all come from there",
    )
    ap.add_argument(
        "--max-refusals",
        type=int,
        default=6,
        help="Candidate constraints to keep per leftover operator (default 6)",
    )
    args = ap.parse_args()

    source = HipSource(args.repo_root)
    missing = source.missing_paths()
    if missing:
        raise SystemExit("Not found in the repository: " + ", ".join(missing))
    hip_op_attributes = source.op_attributes()

    pre_module = parse_mlir_file(Path(args.input_mlir))
    post_module = parse_mlir_file(Path(args.converted_mlir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    leftover_ops = post_module.onnx_ops()
    leftover_locs = {post_module.resolve_loc(op) for op in leftover_ops}
    leftover_locs.discard("")
    leftovers = collect_leftovers(leftover_ops)

    orphans = orphan_hip_ops(pre_module, post_module)
    attr_rows, unpaired = build_attr_transfer(
        pre_module,
        post_module,
        leftover_locs,
        pairing_is_complete=not orphans,
        hip_op_attributes=hip_op_attributes,
    )

    reasons = explain_leftovers(source, leftovers, args.max_refusals)
    observed_hip_ops = {op for row in attr_rows for op in row["hip_ops"]}
    # The dialect is ~90 ops and a model uses a dozen; the rest would be noise
    # for whoever opens the file.
    runtime_map = source.runtime_map(keep_ops=observed_hip_ops)

    write(
        output_dir / "leftover_onnx.json",
        {
            "input_mlir": str(args.input_mlir),
            "converted_mlir": str(args.converted_mlir),
            "unconverted": leftovers,
            "unconverted_instances": sum(e["count"] for e in leftovers),
        },
    )
    write(
        output_dir / "leftover_reasons.json",
        {"conversion_dir": str(source.conversion_dir), "rows": reasons},
    )
    write(
        output_dir / "attr_transfer.json",
        {
            "rows": attr_rows,
            "unpaired_instances": unpaired,
            "orphan_hip_ops": orphans,
        },
    )
    write(output_dir / "hip_runtime_map.json", {"ops": runtime_map})

    for entry in leftovers:
        print(f"  unconverted {entry['key']} x{entry['count']}")
    for row in reasons:
        if not row["converter_found"]:
            print(f"  {row['key']}: no converter in {source.conversion_dir.name}")
            continue
        likely = [r for r in row["refusals"] if r.get("likely")] or row["refusals"]
        head = likely[0] if likely else None
        detail = f"{head['message']} ({head['location']})" if head else "unknown"
        print(f"  {row['key']}: converter exists, rejected all; likely: {detail}")
    for row in attr_rows:
        if row["status"] == "partial":
            print(f"  partial {row['key']} dropped {list(row['dropped_attrs'])}")
        elif row["folded_instances"] and not row["paired_instances"]:
            print(f"  folded (no runtime op) {row['key']} x{row['folded_instances']}")
    if orphans:
        print(f"  hip ops with no pre-conversion match: {orphans}")
    if unpaired:
        print(f"  unpaired (no location match): {unpaired}")
    print(f"  {len(runtime_map)} hip op(s) mapped to a runtime function")


if __name__ == "__main__":
    main()
