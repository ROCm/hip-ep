#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Compatibility oracle: what convert-onnx-to-hip actually did (steps 3 and 4).

Compares the compiler-input MLIR with the same module after the conversion
pipeline and writes two files:

  leftover_onnx.json   onnx ops the conversion did not replace. The pass DCEs
                       unused onnx ops before it finishes, so anything left is
                       live: the operator is unsupported, or its pattern bailed
                       out on this instance.

  attr_transfer.json   For converted ops, the ONNX attributes that did not
                       reach the HIP op. Pairing uses the location both dumps
                       carry (see mlir_text). An attribute whose value equals
                       the ONNX schema default is recorded separately, because
                       dropping a default changes no behaviour.

Both dumps must be produced with --mlir-print-debuginfo so the locations line
up; the input dump is re-printed from the same file the conversion read.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from mlir_text import parse_mlir_file, strip_quotes

# Dialects a converted ONNX op can legitimately land in. hip.* is the runtime
# path; the others are compile-time folds (a Reshape that became a static
# tensor.expand_shape, index arithmetic, and so on).
_CONVERTED_DIALECTS = ("hip", "tensor", "arith", "memref", "scf", "bufferization")

_SCALAR_RE = re.compile(r"^\s*(-?[\d.eE+]+|\"[^\"]*\")")

_CONTRIB_DEFAULTS_PATH = Path(__file__).with_name("contrib_attr_defaults.json")


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


def _contrib_defaults(op_type: str, domain: str):
    """Curated defaults for operators the onnx package has no schema for."""
    try:
        table = json.loads(_CONTRIB_DEFAULTS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return (table.get(domain) or {}).get(op_type) or {}


def _schema_defaults(op_type: str, domain: str):
    """ONNX attribute defaults, falling back to the curated contrib table."""
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


def build_attr_transfer(pre_module, post_module, leftover_locs):
    post_by_loc = defaultdict(list)
    for op in post_module.ops:
        if op.dialect not in _CONVERTED_DIALECTS:
            continue
        loc = post_module.resolve_loc(op)
        if loc:
            post_by_loc[loc].append(op)

    defaults_cache = {}
    per_key = defaultdict(
        lambda: {
            "paired": 0,
            "hip_ops": set(),
            "dropped_attrs": defaultdict(int),
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
            unpaired[f"{key[1]}.{key[0]}"] += 1
            continue

        entry = per_key[key]
        entry["paired"] += 1
        primary = _primary_op(candidates)
        if primary:
            entry["hip_ops"].add(primary.name)

        landed = set()
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
                "hip_ops": sorted(entry["hip_ops"]),
                "dropped_attrs": dict(sorted(entry["dropped_attrs"].items())),
                "dropped_default_attrs": dict(
                    sorted(entry["dropped_default_attrs"].items())
                ),
                "status": "partial" if entry["dropped_attrs"] else "full",
                "samples": entry["samples"],
            }
        )
    return rows, dict(sorted(unpaired.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_mlir", help="compiler-input MLIR printed with locations")
    ap.add_argument(
        "converted_mlir", help="post-conversion MLIR printed with locations"
    )
    ap.add_argument("output_dir", help="Directory for the two JSON files")
    args = ap.parse_args()

    pre_module = parse_mlir_file(Path(args.input_mlir))
    post_module = parse_mlir_file(Path(args.converted_mlir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    leftover_ops = post_module.onnx_ops()
    leftover_locs = {post_module.resolve_loc(op) for op in leftover_ops}
    leftover_locs.discard("")
    leftovers = collect_leftovers(leftover_ops)

    attr_rows, unpaired = build_attr_transfer(pre_module, post_module, leftover_locs)

    leftover_path = output_dir / "leftover_onnx.json"
    leftover_path.write_text(
        json.dumps(
            {
                "input_mlir": str(args.input_mlir),
                "converted_mlir": str(args.converted_mlir),
                "unconverted": leftovers,
                "unconverted_instances": sum(e["count"] for e in leftovers),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    attr_path = output_dir / "attr_transfer.json"
    attr_path.write_text(
        json.dumps(
            {
                "rows": attr_rows,
                "unpaired_instances": unpaired,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"[OK] {leftover_path}")
    for entry in leftovers:
        print(f"  unconverted {entry['key']} x{entry['count']}")
    print(f"[OK] {attr_path}")
    for row in attr_rows:
        if row["status"] == "partial":
            print(f"  partial {row['key']} dropped {list(row['dropped_attrs'])}")
    if unpaired:
        print(f"  unpaired (no location match): {unpaired}")


if __name__ == "__main__":
    main()
