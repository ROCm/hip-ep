#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Single-operator MLIR modules, built from the operators a model actually
contains, for running the conversion over one operator at a time.

The whole-graph probe answers support for every operator in one run, so this
exists for the two questions it cannot answer:

  - A conversion that fails outright leaves the run with no per-operator
    result at all. The slices convert independently, so a graph the pass
    refuses as a whole still yields an answer for each operator in it.
  - Why an operator was refused. Varying one thing about a slice -- an
    operand's element type, an attribute's value -- and converting again
    says which one the conversion objected to. Release builds compile the
    refusal messages out, so this is the only way to get a proven cause
    rather than a ranked guess.

Types come from the model's own operator line, so nothing is synthesized
from a schema. What the slice must reproduce rather than abstract away:

  - An operand defined by onnx.Constant stays an onnx.Constant. The
    shape-fold conversions require their shape operand to be constant and
    refuse a block argument, which would read as the operator being
    unsupported. The data becomes a splat so external weight files are not
    needed.
  - An operand defined by onnx.NoValue stays an onnx.NoValue. It carries
    type `none`, marking an omitted optional operand, which no block
    argument can stand in for.
  - The function is @main_graph. The conversion's metadata step requires
    that name and fails the run otherwise.
  - The terminator is onnx.Return, which the importer emits and the
    conversion rewrites; `none` results are left out of it because a
    function cannot return them.

Usage:
  slice_probe.py <ep_input.mlir> <output_dir> --hip-ep-package-root <dir>
                 [--leftovers <leftover_onnx.json>]
"""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from mlir_text import (
    MlirModule,
    MlirOp,
    element_type,
    parse_mlir,
    split_signature,
    strip_quotes,
)

# Operands these define have to be rebuilt as themselves; see the module
# docstring for why a block argument is not a substitute.
MATERIALIZED_DEFS = frozenset({"onnx.Constant", "onnx.NoValue"})

# Importer bookkeeping, per instance. Leaving it out is what makes two
# instances that differ only by node name produce the same slice, so the slice
# text is itself the signature: an equivalence class by construction, rather
# than a key that could omit a property the conversion happens to read.
BOOKKEEPING_ATTRS = frozenset({"node.outputs", "onnx_node_name"})

# Same prefix of buildOnnxToHipPipeline that run_convert_probe.ps1 runs over
# the whole graph. A slice has to go through the same passes: simplify-onnx
# rewrites operators (CastLike becomes Cast) and hip-add-context-arg injects
# the !hip.context operand that many conversions match on, so skipping either
# would report a different answer than production compiles.
CONVERT_PASSES = (
    "--onnx-dialect=stub",
    "--simplify-onnx",
    "--hip-add-context-arg",
    "--onnx-loop-outline",
    "--onnx-if-outline",
    "--hip-infer-loop-body-shapes",
    "--convert-onnx-to-hip",
)

# The conversion reports why it refused an operator through
# notifyMatchFailure, which the greedy driver logs under its own debug
# category -- `dialect-conversion` is a different driver and says nothing
# here. Reading this is what turns "did not convert" into the constraint
# that stopped it, with no source scanning and no ranking.
#
# Asking for it costs a second run, so only a slice that did not convert
# gets one: on an operator that converted the log is large and says nothing
# a reader wants.
REFUSAL_DEBUG_FLAG = "-debug-only=greedy-rewriter"
_REFUSAL_RE = re.compile(r"\*\* Match Failure : (?P<reason>.+?)\s*$", re.M)

# Every pattern rooted on onnx.Custom checks the function name first, so each
# one refuses every custom operator that is not its own. Those refusals are
# about the pattern, not about this operator.
_NAME_CHECK_RE = re.compile(r"^not a[n]? .*?(operation|custom op|op)$", re.I)

_FLOAT_ELEMS = frozenset({"f16", "f32", "f64", "bf16", "f8E4M3FN", "f8E5M2", "tf32"})

# `tensor<*xf16>`: a tensor of unknown rank, as against `tensor<?x?xf16>`,
# whose rank is known and whose extents are not.
_UNRANKED_RE = re.compile(r"^tensor<\*x")


def splat_value(tensor_type: str) -> str:
    """A `dense<...>` literal of `tensor_type`, all elements zero.

    Zero keeps the literal short whatever the shape, and the conversion reads
    a constant's type and rank rather than its values -- except for a shape
    operand, which is why `constant_text` prefers the original literal.
    """
    elem = element_type(tensor_type)
    if elem in _FLOAT_ELEMS:
        literal = "0.000000e+00"
    elif elem == "i1":
        literal = "false"
    else:
        literal = "0"
    return f"dense<{literal}> : {tensor_type}"


def constant_text(name: str, defining: MlirOp, tensor_type: str) -> str:
    """An onnx.Constant line for the slice.

    A constant whose value was inline in the dump keeps it: the shape-fold
    conversions read the shape operand's elements, and a zero shape is not
    the shape this model uses. One backed by external data has only a
    location in the dump, so it becomes a splat.
    """
    value = defining.attrs.get("value", "")
    if not value or "location" in defining.attrs:
        value = splat_value(tensor_type)
    return f'{name} = "onnx.Constant"() {{value = {value}}} : () -> {tensor_type}'


def _attr_text(attrs: Dict[str, str]) -> str:
    # A unit attribute has no value; `onnx.NoValue` carries `{value}`.
    return ", ".join(
        f"{k} = {v}" if v else k for k, v in attrs.items() if k not in BOOKKEEPING_ATTRS
    )


def _op_text(
    op: MlirOp,
    result: str,
    operands: List[str],
    in_types: List[str],
    out_types: List[str],
) -> str:
    lhs = f"{result} = " if result else ""
    if result and len(out_types) > 1:
        lhs = f"{result}:{len(out_types)} = "
    attrs = _attr_text(op.attrs)
    attr_group = f" {{{attrs}}}" if attrs else ""
    results = out_types[0] if len(out_types) == 1 else f"({', '.join(out_types)})"
    return (
        f'{lhs}"{op.name}"({", ".join(operands)}){attr_group}'
        f" : ({', '.join(in_types)}) -> {results}"
    )


def build_slice(
    module: MlirModule,
    op: MlirOp,
    attr_overrides: Optional[Dict[str, str]] = None,
    type_overrides: Optional[Dict[int, str]] = None,
) -> Tuple[str, List[str]]:
    """The text of a module holding `op` alone, and the types it returns.

    `attr_overrides` replaces attribute values and `type_overrides` replaces
    an operand type by index, which is how a caller varies one thing to find
    what a refusal objected to. An override of an operand's type applies to
    the constant or block argument feeding it, so the slice stays consistent.
    """
    if op.opens_region:
        raise ValueError(
            f"{op.name} carries a region, whose body and type signature are on "
            "the lines around it rather than on its own, so a line-based "
            "reader cannot reproduce the operation"
        )
    in_types, out_types = split_signature(op.type_signature)
    if len(in_types) != len(op.operands) or not out_types:
        raise ValueError(
            f"{op.name} at line {op.line_no}: {len(op.operands)} operands "
            f"against {len(in_types)} operand types in {op.type_signature!r}"
        )
    if type_overrides:
        in_types = [type_overrides.get(i, t) for i, t in enumerate(in_types)]

    # An operand of unranked type has to become a function argument, and the
    # pipeline requires a ranked one. The graph did not know the rank here
    # either, so there is no rank to supply that would not be invented, and an
    # invented one would be reported as this operator's result.
    unranked = sorted({t for t in in_types if _UNRANKED_RE.match(t)})
    if unranked:
        raise ValueError(
            f"{op.name} reads {', '.join(unranked)}, whose rank the graph "
            "leaves open; the slice would have to invent one to give the "
            "operation an argument"
        )

    body: List[str] = []
    args: List[str] = []
    operand_names: List[str] = []
    for index, (ssa, tensor_type) in enumerate(zip(op.operands, in_types)):
        defining = module.defining_op(ssa, before_line=op.line_no)
        if defining is not None and defining.name in MATERIALIZED_DEFS:
            name = f"%v{index}"
            if defining.name == "onnx.NoValue":
                body.append(
                    f'{name} = "onnx.NoValue"() {{value}} : () -> {tensor_type}'
                )
            else:
                body.append(constant_text(name, defining, tensor_type))
        else:
            name = f"%arg{len(args)}"
            args.append(f"{name}: {tensor_type}")
        operand_names.append(name)

    sliced = op
    if attr_overrides:
        sliced = MlirOp(**{**op.__dict__, "attrs": {**op.attrs, **attr_overrides}})

    result = "%out" if out_types else ""
    body.append(_op_text(sliced, result, operand_names, in_types, out_types))

    # A function cannot return `none`, which an omitted optional result prints
    # as, so the slice returns the rest. The operation keeps all of them.
    returned = [
        (f"{result}#{i}" if len(out_types) > 1 else result, t)
        for i, t in enumerate(out_types)
        if t.strip() != "none"
    ]
    returns = ", ".join(name for name, _ in returned)
    return_types = ", ".join(t for _, t in returned)
    body.append(f'"onnx.Return"({returns}) : ({return_types}) -> ()')

    signature = ", ".join(args)
    result_clause = return_types if len(returned) == 1 else f"({return_types})"
    indented = "\n".join(f"    {line}" for line in body)
    text = (
        "module {\n"
        f"  func.func @main_graph({signature}) -> {result_clause} {{\n"
        f"{indented}\n"
        "  }\n"
        "}\n"
    )
    return text, [t for _, t in returned]


@dataclass
class SliceCase:
    """Instances a single slice stands for, and that slice."""

    op_type: str
    domain: str
    text: str
    instances: int = 0
    node_names: List[str] = field(default_factory=list)
    lines: List[int] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.domain}.{self.op_type}"


def group_cases(
    module: MlirModule, keys: Optional[Iterable[str]] = None
) -> Tuple[List[SliceCase], List[dict]]:
    """Slice cases for `module`, plus the operators no slice could be built for.

    `keys` restricts the work to `domain.op_type` entries, which is how a run
    that already has a whole-graph result probes only what was left over.
    """
    wanted = set(keys) if keys is not None else None
    cases: Dict[str, SliceCase] = {}
    unbuildable: Dict[str, dict] = {}

    for op in module.onnx_ops():
        op_type, domain = op.onnx_key()
        key = f"{domain}.{op_type}"
        if wanted is not None and key not in wanted:
            continue
        try:
            text, _ = build_slice(module, op)
        except ValueError as exc:
            entry = unbuildable.setdefault(
                key,
                {
                    "key": key,
                    "op_type": op_type,
                    "domain": domain,
                    "count": 0,
                    "reason": str(exc),
                },
            )
            entry["count"] += 1
            continue
        case = cases.setdefault(
            text, SliceCase(op_type=op_type, domain=domain, text=text)
        )
        case.instances += 1
        if len(case.node_names) < 5:
            name = strip_quotes(op.attrs.get("onnx_node_name", ""))
            if name:
                case.node_names.append(name)
            case.lines.append(op.line_no)

    ordered = sorted(cases.values(), key=lambda c: (-c.instances, c.key))
    return ordered, sorted(unbuildable.values(), key=lambda e: e["key"])


def collect_refusals(
    hip_mlir_opt: str, slice_path: Path, passes: Sequence[str] = CONVERT_PASSES
) -> List[str]:
    """What the conversion said when it refused this slice's operator.

    A slice holds one operator, so the log carries only its refusals. Over a
    whole graph the same request returns one per pattern per custom operator,
    which is why this is worth asking per slice and not per model.
    """
    proc = subprocess.run(
        [hip_mlir_opt, str(slice_path), *passes, REFUSAL_DEBUG_FLAG],
        capture_output=True,
        text=True,
    )
    reasons = []
    for match in _REFUSAL_RE.finditer(proc.stderr or ""):
        reason = match.group("reason").strip()
        if not reason or _NAME_CHECK_RE.match(reason):
            continue
        if reason not in reasons:
            reasons.append(reason)
    return reasons


def convert_case(
    hip_mlir_opt: str,
    case: SliceCase,
    slice_path: Path,
    passes: Sequence[str] = CONVERT_PASSES,
) -> dict:
    """Run the conversion over one slice and say what it did.

    A slice holds one operator, so whatever the conversion produced is that
    operator's result: no location pairing, and an operator that produced no
    hip op provably folded rather than went unpaired.
    """
    slice_path.write_text(case.text, encoding="utf-8")
    proc = subprocess.run(
        [hip_mlir_opt, str(slice_path), *passes],
        capture_output=True,
        text=True,
    )
    row = {
        "key": case.key,
        "op_type": case.op_type,
        "domain": case.domain,
        "instances": case.instances,
        "node_names": case.node_names,
        "lines": case.lines,
        "slice": slice_path.name,
    }
    if proc.returncode != 0:
        stderr = [ln.strip() for ln in (proc.stderr or "").splitlines() if ln.strip()]
        row["verdict"] = "probe_failed"
        row["exit_code"] = proc.returncode
        row["reason"] = stderr[0] if stderr else ""
        row["refusals"] = collect_refusals(hip_mlir_opt, slice_path, passes)
        return row

    result = parse_mlir(proc.stdout)
    produced = sorted({op.name for op in result.ops if op.dialect != "onnx"})
    hip_ops = [name for name in produced if name.startswith("hip.")]
    row["produced_ops"] = produced
    if result.onnx_ops():
        row["verdict"] = "unsupported"
        row["refusals"] = collect_refusals(hip_mlir_opt, slice_path, passes)
    elif hip_ops:
        row["verdict"] = "converts"
    else:
        # Nothing in the hip dialect: the operator became structure the later
        # passes need no runtime call for.
        row["verdict"] = "compile_time"
    return row


def _slice_name(key: str, seen: Dict[str, int]) -> str:
    """A file name per case; one operator can have several signatures."""
    base = re.sub(r"[^A-Za-z0-9._-]", "_", key)
    seen[base] = seen.get(base, 0) + 1
    index = seen[base]
    return f"{base}.mlir" if index == 1 else f"{base}.{index}.mlir"


def resolve_hip_mlir_opt(explicit: str, package_root: str) -> str:
    if explicit:
        return explicit
    if not package_root:
        return ""
    # Executables carry no suffix outside Windows.
    for name in ("hip-mlir-opt.exe", "hip-mlir-opt"):
        candidate = Path(package_root) / "bin" / name
        if candidate.is_file():
            return str(candidate)
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_mlir", help="EP-input MLIR")
    ap.add_argument("output_dir", help="directory for slice_probe.json and slices/")
    ap.add_argument("--hip-ep-package-root", default="", help="hip-ep package")
    ap.add_argument("--hip-mlir-opt", default="", help="hip-mlir-opt, found if unset")
    ap.add_argument(
        "--leftovers",
        help=(
            "leftover_onnx.json from the whole-graph run. Given, only the "
            "operators it lists are probed, which separates an operator with "
            "no working conversion from one the graph's shapes or a failed "
            "upstream conversion blocked. Omitted, every operator is probed, "
            "which is the only per-operator result a failed whole-graph "
            "conversion leaves available."
        ),
    )
    args = ap.parse_args()

    hip_mlir_opt = resolve_hip_mlir_opt(args.hip_mlir_opt, args.hip_ep_package_root)
    if not hip_mlir_opt:
        # Never fall back to reading the source: a missing tool has to be
        # visible rather than silently downgrade the oracle.
        ap.error(
            "hip-mlir-opt not found; pass --hip-mlir-opt or "
            "--hip-ep-package-root pointing at a package with bin/hip-mlir-opt"
        )

    output_dir = Path(args.output_dir)
    slices_dir = output_dir / "slices"
    slices_dir.mkdir(parents=True, exist_ok=True)
    for stale in slices_dir.glob("*.mlir"):
        stale.unlink()

    keys = None
    mode = "all operators"
    if args.leftovers:
        payload = json.loads(Path(args.leftovers).read_text(encoding="utf-8-sig"))
        keys = [entry["key"] for entry in payload.get("unconverted", [])]
        mode = "leftovers"
        if not keys:
            write_result(output_dir, args, mode, [], [])
            print("[OK] nothing left over; no slice was probed")
            return 0

    module = parse_mlir(Path(args.input_mlir).read_text(encoding="utf-8"))
    cases, unbuildable = group_cases(module, keys)
    print(f"{len(cases)} slice(s) to probe ({mode})")

    seen: Dict[str, int] = {}
    rows = []
    for case in cases:
        row = convert_case(hip_mlir_opt, case, slices_dir / _slice_name(case.key, seen))
        rows.append(row)
        print(f"  {case.key:<48} {row['verdict']}")
        for reason in row.get("refusals") or []:
            print(f"      {reason}")

    write_result(output_dir, args, mode, rows, unbuildable)
    return 0


def write_result(
    output_dir: Path, args, mode: str, rows: List[dict], unbuildable: List[dict]
) -> None:
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    path = output_dir / "slice_probe.json"
    path.write_text(
        json.dumps(
            {
                "input_mlir": str(args.input_mlir),
                "mode": mode,
                "passes": list(CONVERT_PASSES),
                "cases": rows,
                "unbuildable": unbuildable,
                "verdict_counts": dict(sorted(counts.items())),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[OK] {path}")


if __name__ == "__main__":
    sys.exit(main())
