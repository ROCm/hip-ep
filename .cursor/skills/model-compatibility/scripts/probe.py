#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Decide operator support by compiling, not by reading source.

Two stages, because the compiler front end and back end fail in different
ways and the distinction drives different work:

  stage1  whole graph, through convert-onnx-to-hip
          Does a converter exist at all? Operators still named onnx.* after
          the pass had no converter (or their matcher bailed out). An
          unconverted operator does not fail the pass -- applyPatternsGreedily
          simply leaves it alone -- so one probe covers the whole model.

  stage2  one operator at a time, through hip-to-llvm-pipeline
          Is the lowering chain complete, and which runtime symbol does it
          reach? Whole-graph lowering is not an option: constant
          externalization needs an injected FileSystem that hip-mlir-opt does
          not have.

Support is read from the IR diff rather than from the
"[convert-onnx-to-hip] N unconverted ..." line, so a change to that
diagnostic's wording cannot silently break the verdict. The line is still
captured as a cross-check.

Usage:
  python probe.py <ep_input.mlir> <ep_input_ops.json> <out_dir> --package <pkg>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

# --onnx-to-hip-pipeline's first six passes. The rest of that pipeline
# bufferizes, and would fail on the very operators this probe reports on.
# --mlir-print-debuginfo makes each produced operation carry its source line,
# which is how results pair back to inputs.
STAGE1_PASSES = [
    "--simplify-onnx",
    "--hip-add-context-arg",
    "--onnx-loop-outline",
    "--onnx-if-outline",
    "--hip-infer-loop-body-shapes",
    "--convert-onnx-to-hip",
    "--mlir-print-debuginfo",
]

# Usually DPS init construction and shape math rather than an operator's own
# lowering target, so they are reported separately -- but not always: Shape
# lowers to tensor.dim + tensor.from_elements. An operator with nothing else
# falls back to reporting these.
DPS_HELPER_OPS = frozenset(
    {
        "tensor.dim",
        "tensor.empty",
        "tensor.from_elements",
        "arith.index_cast",
        "arith.constant",
    }
)

STAGE2_PASSES = ["--onnx-to-hip-pipeline", "--hip-to-llvm-pipeline"]

# Runtime entry points every lowered function calls regardless of which
# operator it holds; not evidence of what this operator maps to.
INFRA_SYMBOLS = re.compile(
    r"^wrap_hip(Malloc|Free|Memcpy\w*|StreamSynchronize|Stream\w*|Device\w*)$"
)

_LOC_DEF = re.compile(r'^#(loc\d+)\s*=\s*loc\(".*":(\d+):\d+\)')
_LOC_USE = re.compile(r"loc\(#(loc\d+)\)")
_QUOTED_OP = re.compile(r'"([a-z][a-z0-9_]*\.[A-Za-z0-9_]+)"\s*\(')
_BARE_OP = re.compile(r"=\s*([a-z][a-z0-9_]*\.[a-z0-9_.]+)")
_UNCONVERTED_LINE = re.compile(r"^\s{2}(onnx\.\S+)\s+x(\d+)\s*$")
_SSA_DEF = re.compile(r"^\s*(%[\w$.-]+)(?::\d+)?\s*=")
_SSA_REF = re.compile(r"(%[\w$.-]+)(?:#(\d+))?")
_WRAP_SYM = re.compile(r"\bwrap_\w+")
# An onnx.Constant whose data lives at a runtime memory address rather than in
# a file. Inlining one makes constant externalization fail with "memory-address
# sources require production externalization with an injected FileSystem".
_MEM_ADDR_CONST = re.compile(r'location\s*=\s*"\*/_ORT_MEM_ADDR_/\*"')


class ProbeError(RuntimeError):
    pass


def _tool(package_root: Path, name: str) -> Path:
    exe = package_root / "bin" / name
    if not exe.exists():
        raise ProbeError(f"{name} not found under {package_root / 'bin'}")
    return exe


# A run near this bound is not slow, it is stuck: a perturbed attribute can
# leave the greedy rewrite without a fixed point. Real runs are two orders of
# magnitude below it.
OPT_TIMEOUT_SEC = 5

TIMEOUT_EXIT = -9


def run_opt(
    opt: Path,
    src: Path,
    passes: list[str],
    out: Path,
    debug: bool = True,
    timeout: int = OPT_TIMEOUT_SEC,
) -> tuple[int, str]:
    """Run hip-mlir-opt; return (exit code, combined output).

    A timeout reports TIMEOUT_EXIT rather than raising: one operator that
    will not converge should not abort the whole probe.
    """
    env = dict(os.environ)
    env["HIPDNN_EP_DEBUG"] = "1" if debug else "0"
    try:
        proc = subprocess.run(
            [str(opt), str(src), *passes, "-o", str(out)],
            capture_output=True,
            text=True,
            env=env,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return TIMEOUT_EXIT, f"timed out after {timeout}s"
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _failure_reason(code: int, log: str) -> str:
    """Say why a run failed, in whatever form the failure took.

    A diagnostic is only one of the ways hip-mlir-opt can fail: it also
    crashes, and then there is no `error:` line to find. Reporting "unknown"
    for a segfault hides the most actionable failure of the three.
    """
    if code == TIMEOUT_EXIT:
        return log.strip()
    for line in log.splitlines():
        if "error:" in line:
            return line.strip()
    crash = re.search(r"exception 0x[0-9a-fA-F]+", log)
    if crash:
        top = re.search(r"^#0\s+0x\S+ in\s+(.+)$", log, re.MULTILINE)
        where = (
            f" at {top.group(1).strip()}" if top and "??" not in top.group(1) else ""
        )
        return f"hip-mlir-opt crashed ({crash.group(0)}){where}"
    if code and code & 0xC0000000 == 0xC0000000:
        return f"hip-mlir-opt terminated abnormally (0x{code & 0xFFFFFFFF:08X})"
    return f"hip-mlir-opt exited with code {code}"


def parse_loc_table(lines: list[str]) -> dict[str, int]:
    """Map loc identifier -> source line number."""
    table: dict[str, int] = {}
    for line in lines:
        m = _LOC_DEF.match(line.strip())
        if m:
            table[m.group(1)] = int(m.group(2))
    return table


def ops_on_line(line: str) -> list[str]:
    """Operation names produced on this line, in both MLIR syntaxes."""
    names = _QUOTED_OP.findall(line)
    names += [n for n in _BARE_OP.findall(line) if "." in n]
    return names


def read_conversion_result(after_path: Path) -> dict[int, dict]:
    """Group the conversion output by the source line each result came from.

    Returns {source_line: {"targets": [...], "helpers": [...],
    "surviving_onnx": [...]}}.
    """
    lines = after_path.read_text(encoding="utf-8").splitlines()
    loc_table = parse_loc_table(lines)

    by_src: dict[int, dict] = {}
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#loc"):
            continue
        use = _LOC_USE.search(line)
        if not use:
            continue
        src_line = loc_table.get(use.group(1))
        if src_line is None:
            continue
        entry = by_src.setdefault(
            src_line,
            {"targets": [], "helpers": [], "surviving_onnx": [], "result_text": []},
        )
        entry["result_text"].append(line.strip())
        for name in ops_on_line(line):
            if name.startswith("onnx."):
                entry["surviving_onnx"].append(name)
            elif name in DPS_HELPER_OPS:
                entry["helpers"].append(name)
            else:
                entry["targets"].append(name)
    return by_src


def parse_unconverted_diagnostic(log: str) -> dict[str, int]:
    """Parse the compiler's own unconverted-operator summary, for cross-check."""
    found: dict[str, int] = {}
    seen_header = False
    for line in log.splitlines():
        if "unconverted onnx op type(s) still live" in line:
            seen_header = True
            continue
        if not seen_header:
            continue
        m = _UNCONVERTED_LINE.match(line.rstrip())
        if not m:
            if line.strip():
                seen_header = False
            continue
        found[m.group(1)] = int(m.group(2))
    return found


def _match_paren(text: str, start: int) -> int:
    """Index of the ')' closing the '(' at `start`, honouring nesting."""
    depth = 0
    in_str = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _split_types(text: str) -> list[str]:
    """Split a comma-separated type list, honouring <> and () nesting."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in text:
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf and "".join(buf).strip():
        parts.append("".join(buf).strip())
    return parts


def parse_op_line(line: str) -> dict | None:
    """Break an MLIR operation line into the pieces a slice needs."""
    m = re.search(r'"(onnx\.[A-Za-z0-9_]+)"\s*\(', line)
    if not m:
        return None
    open_paren = line.index("(", m.end() - 1)
    close_paren = _match_paren(line, open_paren)
    if close_paren < 0:
        return None

    operands = [
        o.strip() for o in line[open_paren + 1 : close_paren].split(",") if o.strip()
    ]
    rest = line[close_paren + 1 :]
    rest = re.sub(r"\s*loc\(#loc\d+\)\s*$", "", rest.rstrip())

    # The type signature is the last ": (...) -> ..." on the line.
    sig = re.search(r":\s*\((.*)\)\s*->\s*(.+)$", rest, re.DOTALL)
    if not sig:
        return None
    in_types = _split_types(sig.group(1))
    out_text = sig.group(2).strip()
    if out_text.startswith("(") and out_text.endswith(")"):
        out_types = _split_types(out_text[1:-1])
    else:
        out_types = [out_text]

    attr_text = rest[: sig.start()].strip()
    return {
        "op": m.group(1),
        "operands": operands,
        "in_types": in_types,
        "out_types": out_types,
        "attrs": attr_text,
    }


def convert_candidates(
    opt: Path, candidates: list[str], work: Path, op_type: str
) -> tuple[bool, dict[str, int]]:
    """Convert the first candidate that converts, and name what it became.

    Returns (converted, target op counts). Trying each in turn is what makes
    a guessed shape usable: only a candidate that succeeds is evidence.
    """
    out = work / f"{op_type}.hip.mlir"
    work.mkdir(parents=True, exist_ok=True)
    for module in candidates:
        if _converts_cleanly(opt, module, work / f"{op_type}.mlir", out):
            targets: dict[str, int] = {}
            if out.exists():
                for line in out.read_text(encoding="utf-8").splitlines():
                    for name in ops_on_line(line):
                        if not name.startswith("onnx.") and name not in DPS_HELPER_OPS:
                            targets[name] = targets.get(name, 0) + 1
            return True, targets
    return False, {}


def lower_candidates(
    opt: Path, candidates: list[str], work: Path, op_type: str, inferred: bool
) -> dict:
    """Lower the first candidate that lowers, and name the symbols it reached.

    Returns a stage2 entry. A slice built on a guessed shape cannot convict
    the lowering -- the shape is as likely a suspect -- so its failure is
    recorded as unverified rather than broken.
    """
    work.mkdir(parents=True, exist_ok=True)
    src = work / f"{op_type}.s2.mlir"
    out = work / f"{op_type}.llvm.mlir"
    code, log = 1, "no candidates"
    for module in candidates:
        src.write_text(module, encoding="utf-8")
        code, log = run_opt(opt, src, STAGE2_PASSES, out, debug=False)
        if code == 0 and out.exists():
            break

    if code == TIMEOUT_EXIT:
        return {"status": "timeout", "error": log}
    if code != 0 or not out.exists():
        entry = {
            "status": "not_sliceable" if inferred else "lowering_broken",
            "error": _failure_reason(code, log)[:300],
        }
        if inferred:
            entry["reason"] = (
                "unranked types; a slice with an inferred shape did not lower"
            )
        return entry

    symbols = sorted(
        {
            s
            for s in _WRAP_SYM.findall(out.read_text(encoding="utf-8"))
            if not INFRA_SYMBOLS.match(s)
        }
    )
    return {"status": "ok", "runtime_funcs": symbols, "compile_time": not symbols}


def _slices_for(
    src_lines: list[str], def_index: dict[str, int], instances: list[dict]
) -> tuple[list[str], int, str, bool]:
    """Modules worth probing for one operator, strongest evidence first.

    Returns (candidates, line_no, reason, inferred). Whether an operator can
    be isolated depends on the types at that particular use -- shape
    inference leaves some uses unranked and others not -- so every instance
    is tried before falling back to guessing a shape. Several guesses are
    offered, since a reduction needs a narrower result than its operand.
    """
    why = "no instances"
    for inst in instances:
        module, why = build_single_op_module(src_lines, def_index, inst["line_no"])
        if module is not None:
            return [module], inst["line_no"], why, False

    for inst in instances:
        candidates = []
        for rank_delta in range(_MAX_RANK_GUESSES):
            module, why = build_single_op_module(
                src_lines, def_index, inst["line_no"], infer_unranked=rank_delta
            )
            if module is not None:
                candidates.append(module)
        if candidates:
            return candidates, inst["line_no"], "inferred-shape", True

    return [], instances[0]["line_no"] if instances else 0, why, False


# How many ranks below the widest operand to try for an unranked type.
# Shape-preserving operators need the first; reductions need a lower one.
_MAX_RANK_GUESSES = 3


def _widest_shape(types: list[str], rank_delta: int = 0) -> str | None:
    """Guess the shape an unranked type would have, from its neighbours.

    The widest ranked operand fits shape-preserving operators; rank_delta
    drops trailing dimensions for the ones that shrink. Being a guess decides
    how the answer may be used: a slice built this way that converts proves
    the operator is handled, while one that fails proves nothing.
    """
    best: str | None = None
    best_rank = -1
    for ty in types:
        m = _ELEM_OF_TENSOR.match(ty.strip())
        if not m or "*" in m.group(1):
            continue
        shape = m.group(1)
        rank = shape.count("x")
        if rank > best_rank:
            best, best_rank = shape, rank
    if best is None:
        # Nothing ranked to copy from, which happens when an operator sits
        # downstream of something shape inference gave up on. Start from a
        # plain 2-D dynamic shape and let rank_delta walk it down.
        best = "?x?x"
    if rank_delta == 0:
        return best
    dims = best.split("x")[:-1]
    if rank_delta > len(dims):
        return None
    kept = dims[: len(dims) - rank_delta]
    return "".join(d + "x" for d in kept)


def _apply_shape(ty: str, shape: str) -> str:
    m = _ELEM_OF_TENSOR.match(ty.strip())
    if not m or m.group(1) != "*x":
        return ty
    return f"tensor<{shape}{m.group(2)}>"


def build_single_op_module(
    src_lines: list[str],
    def_index: dict[str, int],
    line_no: int,
    type_override: tuple[int, str] | None = None,
    infer_unranked: int | None = None,
) -> tuple[str | None, str]:
    """Wrap one operation in a self-contained module.

    Operands become function arguments, with two exceptions that change what
    the converter decides:

    - An inline onnx.Constant is carried over verbatim. Several converters
      read the value: Pow only decomposes for a constant scalar exponent, and
      turning that operand into an argument makes it stop matching. Weight
      constants are the exception's exception -- their value is irrelevant to
      the decision, and a memory-address source breaks externalization -- so
      those become arguments.
    - onnx.NoValue is rebuilt in place, since `none` cannot be a function
      argument.

    Returns (module, reason). The module is None when the operator cannot be
    isolated, and the reason says why -- that is a finding in itself, not
    just a failure to report. With infer_unranked the reason on success is
    "inferred-shape", and the caller must then read a conversion failure as
    inconclusive: see _widest_shape for why.
    """
    parsed = parse_op_line(src_lines[line_no - 1])
    if not parsed:
        return None, "could not parse the operation"

    # Unranked is fine mid-graph but not in @main_graph's signature. A result
    # always lands there; operands are checked below, since an inlined
    # constant never does.
    inferred = False
    if infer_unranked is not None and any(
        "<*x" in t for t in parsed["in_types"] + parsed["out_types"]
    ):
        shape = _widest_shape(parsed["in_types"], infer_unranked)
        if shape is None:
            return None, "unranked type with no ranked operand to borrow a shape from"
        parsed = dict(
            parsed,
            in_types=[_apply_shape(t, shape) for t in parsed["in_types"]],
            out_types=[_apply_shape(t, shape) for t in parsed["out_types"]],
        )
        inferred = True

    unranked_out = [t for t in parsed["out_types"] if "<*x" in t]
    if unranked_out:
        return None, f"unranked result type {unranked_out[0]}"

    if type_override is not None:
        idx, new_type = type_override
        if not 0 <= idx < len(parsed["in_types"]):
            return None, "operand index out of range"
        parsed = dict(parsed, in_types=list(parsed["in_types"]))
        parsed["in_types"][idx] = new_type
        # An inlined constant carries its own type, which would now disagree
        # with the operand's. Forcing the operand to an argument keeps the
        # module consistent; the attribution test only varies types anyway.
        parsed["_force_arg"] = idx

    preamble: list[str] = []
    args: list[str] = []
    remap: dict[str, str] = {}
    inlined: set[str] = set()
    none_ssa: str | None = None

    forced_arg = parsed.get("_force_arg")
    for pos, (operand, ty) in enumerate(zip(parsed["operands"], parsed["in_types"])):
        ref = _SSA_REF.match(operand)
        base = ref.group(1) if ref else operand
        def_line_no = def_index.get(base)
        def_line = src_lines[def_line_no - 1] if def_line_no else ""

        if pos == forced_arg:
            arg = f"%arg{len(args)}"
            args.append(f"{arg}: {ty}")
            remap[operand] = arg
            continue

        if ty.strip() == "none" or '"onnx.NoValue"' in def_line:
            if none_ssa is None:
                none_ssa = "%none"
                preamble.append('  %none = "onnx.NoValue"() {value} : () -> none')
            remap[operand] = none_ssa
            continue

        inline_const = (
            '"onnx.Constant"' in def_line
            and "location" not in def_line
            and not _MEM_ADDR_CONST.search(def_line)
        )
        if inline_const:
            # One constant can feed several operands -- Slice routinely passes
            # the same zero vector as starts, ends and axes. Copying it once
            # per use would redefine its SSA name and make the module invalid.
            if base not in inlined:
                preamble.append("  " + def_line.strip())
                inlined.add(base)
            remap[operand] = base
            continue

        if "<*x" in ty:
            # Reaches the signature as an argument, where unranked is not
            # allowed.
            return None, f"unranked operand type {ty} at position {pos}"
        arg = f"%arg{len(args)}"
        args.append(f"{arg}: {ty}")
        remap[operand] = arg

    new_operands = ", ".join(remap[o] for o in parsed["operands"])
    attrs = f" {parsed['attrs']}" if parsed["attrs"] else ""
    in_sig = ", ".join(parsed["in_types"])
    outs = parsed["out_types"]
    out_sig = f"({', '.join(outs)})" if len(outs) > 1 else outs[0]

    if len(outs) > 1:
        lhs = f"%r:{len(outs)}"
        result_refs = [f"%r#{i}" for i in range(len(outs))]
    else:
        lhs = "%r"
        result_refs = ["%r"]

    # The operation keeps its `none` placeholders for absent optional
    # outputs; the function cannot, since a non-tensor in @main_graph's
    # signature is rejected.
    returned = [(ref, ty) for ref, ty in zip(result_refs, outs) if ty.strip() != "none"]
    if not returned:
        return None, "every result is a `none` placeholder"
    ret_types = [ty for _ref, ty in returned]
    ret_sig = f"({', '.join(ret_types)})" if len(ret_types) > 1 else ret_types[0]
    ret_vals = ", ".join(ref for ref, _ty in returned)

    body = [
        "module {",
        f"  func.func @main_graph({', '.join(args)}) -> {ret_sig} {{",
        *preamble,
        f'    {lhs} = "{parsed["op"]}"({new_operands}){attrs} : '
        f"({in_sig}) -> {out_sig}",
        f'    "onnx.Return"({ret_vals}) : ({", ".join(ret_types)}) -> ()',
        "  }",
        "}",
        "",
    ]
    return "\n".join(body), "inferred-shape" if inferred else ""


# Element types worth trying when looking for the operand that blocks a
# conversion. Converters most often accept a narrower set than the model
# supplies -- MatMulNBits takes i8 or f16 zero_points but not f32.
_DTYPE_ALTERNATIVES = {
    "f32": ["f16", "i8"],
    "f16": ["f32"],
    "f64": ["f32"],
    "bf16": ["f16", "f32"],
    "i8": ["f16"],
    "ui8": ["i8", "f16"],
    "i64": ["i32"],
    "i32": ["i64"],
}
# The element type is the final `x`-separated field. Anchoring on the
# separator matters: a looser pattern lets backtracking pull it into the
# element name, reading tensor<512x16xf32> as elem "xf32".
_ELEM_OF_TENSOR = re.compile(r"^tensor<(|.*x)([A-Za-z]+\d*)>$")


def _retype(tensor_type: str, new_elem: str) -> str | None:
    m = _ELEM_OF_TENSOR.match(tensor_type.strip())
    if not m:
        return None
    return f"tensor<{m.group(1)}{new_elem}>"


def _converts_cleanly(opt: Path, module: str, path: Path, out: Path) -> bool:
    """True when nothing named onnx.* survives conversion of this module."""
    path.write_text(module, encoding="utf-8")
    code, _ = run_opt(opt, path, STAGE1_PASSES[:-1], out, debug=False)
    if code != 0 or not out.exists():
        return False
    return not re.search(r'"onnx\.[A-Za-z0-9_]+"\s*\(', out.read_text(encoding="utf-8"))


def attribution_probe(
    opt: Path,
    src_lines: list[str],
    def_index: dict[str, int],
    line_no: int,
    op_type: str,
    work_dir: Path,
) -> dict:
    """For an operator that did not convert, find which operand blocks it.

    Change one operand's element type at a time and retry. If the conversion
    then succeeds, that operand is the reason -- a single-variable result
    that names the fix, rather than just reporting that something failed.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    parsed = parse_op_line(src_lines[line_no - 1])
    if not parsed:
        return {"found": False, "reason": "could not parse the operation"}

    for idx, ty in enumerate(parsed["in_types"]):
        m = _ELEM_OF_TENSOR.match(ty.strip())
        if not m:
            continue
        elem = m.group(2)
        for alt in _DTYPE_ALTERNATIVES.get(elem, []):
            new_type = _retype(ty, alt)
            if not new_type:
                continue
            module, _why = build_single_op_module(
                src_lines, def_index, line_no, type_override=(idx, new_type)
            )
            if module is None:
                continue
            tag = f"{op_type}.operand{idx}_{alt}"
            if _converts_cleanly(
                opt, module, work_dir / f"{tag}.mlir", work_dir / f"{tag}.out.mlir"
            ):
                return {
                    "found": True,
                    "operand_index": idx,
                    "operand_type": ty,
                    "accepted_type": new_type,
                    "summary": (
                        f"operand {idx} is {ty}; converting it to {new_type} "
                        f"makes the conversion succeed"
                    ),
                }
    return {
        "found": False,
        "reason": "no single operand element type change made it convert",
    }


def build_def_index(src_lines: list[str]) -> dict[str, int]:
    """Map each SSA result name to the line that defines it."""
    index: dict[str, int] = {}
    for i, line in enumerate(src_lines, 1):
        m = _SSA_DEF.match(line)
        if m:
            index.setdefault(m.group(1), i)
    return index


def perturb_value(raw: str | None) -> str | None:
    """A different but still well-formed value, or None if we cannot make one.

    Only the value matters, not what it means: the test is whether changing it
    changes the conversion output at all.
    """
    if raw is None:
        return None
    s = raw.strip()
    if s.startswith('"') and s.endswith('"'):
        return '"hipep_probe_sentinel"'
    if s in ("true", "false"):
        return "false" if s == "true" else "true"
    if re.fullmatch(r"-?\d+", s):
        return str(int(s) + 1)
    if re.fullmatch(r"-?\d*\.?\d+([eE][-+]?\d+)?", s):
        try:
            return repr(float(s) * 2.0 + 1.0)
        except ValueError:
            return None
    # Arrays, dense<> blobs, type literals: no safe generic edit.
    return None


def _rewrite_attr(module: str, attr: str, new_value: str) -> str:
    """Replace an attribute's value, keeping its MLIR type annotation.

    `axis = 0 : si64` must become `axis = 1 : si64`, not `axis = 1`: bare
    integer literals are i64, and swapping the signedness makes the converter
    take paths it never takes on real input -- one such rewrite sent the
    greedy rewrite into a loop that never terminated.
    """

    def repl(m: re.Match) -> str:
        value = m.group(2)
        typed = re.match(r"^(.*?)(\s*:\s*[a-z]?[iuf]\d+)\s*$", value)
        suffix = typed.group(2) if typed else ""
        return m.group(1) + new_value + suffix

    return re.sub(rf"(\b{re.escape(attr)}\s*=\s*)([^,}}]+)", repl, module, count=1)


def _normalize_ir(text: str) -> str:
    """Drop source-location noise so two runs compare on content alone."""
    out = []
    for line in text.splitlines():
        if line.strip().startswith("#loc"):
            continue
        out.append(re.sub(r"\s*loc\(#loc\d+\)", "", line).rstrip())
    return "\n".join(out)


def _attr_present(ir_text: str, attr: str) -> bool:
    return re.search(rf"\b{re.escape(attr)}\s*=", ir_text) is not None


def instances_of(info: dict) -> dict | None:
    insts = info.get("instances") or []
    return insts[0] if insts else None


def attribute_probe(
    opt: Path,
    base_module: str,
    attrs: dict,
    candidates: list[str],
    op_type: str,
    work_dir: Path,
) -> list[dict]:
    """Did the converter look at each attribute missing from its output?

    The diff that produced `candidates` cannot answer that: a converter that
    read an attribute and omitted it because it held the default value looks
    exactly like one that never read it.

    Perturbation settles it. Change the value and convert again; the verdict
    is one of three:

      handled       the output changed, the converter rejected the new
                    value, or conversion stopped terminating. All three are
                    behaviour differences, and a converter that ignores an
                    attribute has none.
      ignored       output byte-for-byte identical. It never looked, so the
                    attribute is silently dropped -- the only case worth
                    reporting.
      inconclusive  no well-formed different value could be built for this
                    value form.
    """
    if not candidates:
        return []
    work_dir.mkdir(parents=True, exist_ok=True)
    base_path = work_dir / f"{op_type}.attr_base.mlir"
    base_out = work_dir / f"{op_type}.attr_base.out.mlir"
    base_path.write_text(base_module, encoding="utf-8")
    code, _ = run_opt(opt, base_path, STAGE1_PASSES[:-1], base_out, debug=False)
    if code != 0 or not base_out.exists():
        return [
            {
                "attribute": "*",
                "verdict": "inconclusive",
                "detail": "baseline conversion of the slice failed",
            }
        ]

    base_ir = _normalize_ir(base_out.read_text(encoding="utf-8"))

    findings: list[dict] = []
    for attr in candidates:
        print(f"    attr {op_type}.{attr} ...", flush=True)
        value = attrs[attr]
        new_value = perturb_value(value)
        if new_value is None:
            findings.append(
                {
                    "attribute": attr,
                    "value": value,
                    "verdict": "inconclusive",
                    "detail": "no safe perturbation for this value form",
                }
            )
            continue

        perturbed = _rewrite_attr(base_module, attr, new_value)
        if perturbed == base_module:
            findings.append(
                {
                    "attribute": attr,
                    "value": value,
                    "verdict": "inconclusive",
                    "detail": "could not rewrite the attribute",
                }
            )
            continue

        p_path = work_dir / f"{op_type}.attr_{attr}.mlir"
        p_out = work_dir / f"{op_type}.attr_{attr}.out.mlir"
        p_path.write_text(perturbed, encoding="utf-8")
        p_code, _ = run_opt(opt, p_path, STAGE1_PASSES[:-1], p_out, debug=False)

        # Only one question: did the converter look at this attribute? A
        # different output, a rejected value and a conversion that stops
        # terminating all answer yes -- one ignoring it behaves identically
        # whatever the value.
        if p_code == TIMEOUT_EXIT:
            verdict = "handled"
            detail = "conversion stops terminating on the changed value"
        elif p_code != 0:
            verdict = "handled"
            detail = "converter rejects the changed value"
        elif _normalize_ir(p_out.read_text(encoding="utf-8")) == base_ir:
            verdict = "ignored"
            detail = "output identical after changing the value"
        else:
            verdict = "handled"
            detail = "output changed with the value"

        findings.append(
            {
                "attribute": attr,
                "value": value,
                "perturbed_to": new_value,
                "verdict": verdict,
                "detail": detail,
            }
        )
    return findings


def stage1_per_operator(
    opt: Path, mlir_path: Path, ops: dict, out_dir: Path
) -> dict[str, dict]:
    """Fallback for when the whole-graph probe fails.

    One operator failing the pass takes the whole run down with it -- an
    onnx.If whose output shapes are not static, say -- and the other
    operators' verdicts go with it. Slicing each one out isolates them, at
    the cost of losing context: fusions that need neighbouring operators
    cannot fire, so support here is a lower bound.
    """
    src_lines = mlir_path.read_text(encoding="utf-8").splitlines()
    def_index = build_def_index(src_lines)
    work = out_dir / "stage1_slices"
    work.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for op_type, info in ops.items():
        if op_type.startswith("_"):
            continue
        instances = info.get("instances") or []
        if not instances:
            continue
        candidates, line_no, why, inferred = _slices_for(
            src_lines, def_index, instances
        )
        base = {
            "domain": (info.get("domain") or ["onnx"])[0],
            "count": info["count"],
            "targets": {},
            "helpers": {},
            "folded": 0,
            "dropped_attrs": [],
            "unconverted_lines": [],
        }
        if not candidates:
            results[op_type] = {
                **base,
                "converted": 0,
                "unconverted": 0,
                "inconclusive": info["count"],
                "reason": why,
            }
            continue

        print(f"  stage1-slice {op_type} ...", flush=True)
        converts, targets = convert_candidates(opt, candidates, work, op_type)
        if not converts and inferred:
            # The slice only exists because we invented a shape for it, so
            # its failure may be ours rather than the converter's.
            results[op_type] = {
                **base,
                "converted": 0,
                "unconverted": 0,
                "inconclusive": info["count"],
                "reason": "unranked types; a slice with an inferred shape did not convert",
            }
            continue
        results[op_type] = {
            **base,
            "converted": info["count"] if converts else 0,
            "unconverted": 0 if converts else info["count"],
            "targets": targets,
            "unconverted_lines": [] if converts else [line_no],
        }
    return results


def stage2(
    opt: Path, mlir_path: Path, ops: dict, stage1_results: dict, out_dir: Path
) -> dict[str, dict]:
    """Per-operator lowering probe: is the chain complete, and to what symbol?"""
    src_lines = mlir_path.read_text(encoding="utf-8").splitlines()
    def_index = build_def_index(src_lines)
    slice_dir = out_dir / "slices"
    slice_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for op_type, info in ops.items():
        if op_type.startswith("_"):
            continue
        s1 = stage1_results.get(op_type, {})
        instances = info["instances"]
        if not instances:
            results[op_type] = {"status": "skipped", "reason": "no instances"}
            continue

        if s1.get("unconverted"):
            # Nothing converted, so there is no lowering to test. Ask instead
            # which operand the converter objects to -- that is the finding
            # the report needs for a blocked operator.
            bad_line = (s1.get("unconverted_lines") or [instances[0]["line_no"]])[0]
            print(f"  attribution {op_type} ...", flush=True)
            results[op_type] = {
                "status": "skipped",
                "reason": "unconverted in stage1",
                "attribution": attribution_probe(
                    opt,
                    src_lines,
                    def_index,
                    bad_line,
                    op_type,
                    out_dir / "attribution",
                ),
            }
            continue

        candidates, line_no, why, inferred = _slices_for(
            src_lines, def_index, instances
        )
        if not candidates:
            # Not a lowering failure: stage1 showed it converts, and only the
            # rest of the chain goes unchecked.
            results[op_type] = {
                "status": "not_sliceable",
                "source_line": line_no,
                "reason": f"{why} (no usable instance among {len(instances)})",
            }
            continue

        print(f"  stage2 {op_type} ...", flush=True)
        entry = lower_candidates(opt, candidates, slice_dir, op_type, inferred)
        if entry["status"] != "ok":
            results[op_type] = {**entry, "source_line": line_no}
            continue

        # Only operators stage1 saw lose an attribute need this; the same
        # slice serves, so nothing extra is built.
        results[op_type] = {
            **entry,
            "source_line": line_no,
            "attributes": attribute_probe(
                opt,
                candidates[0],
                instances[0].get("attributes", {}),
                s1.get("dropped_attrs", []),
                op_type,
                out_dir / "attrs",
            ),
        }
    return results


def stage1(
    opt: Path, mlir_path: Path, ops: dict, out_dir: Path
) -> tuple[dict[str, dict], dict]:
    """Whole-graph probe. Returns (per-operator result, run metadata)."""
    after = out_dir / "stage1_after.mlir"
    code, log = run_opt(opt, mlir_path, STAGE1_PASSES, after)
    (out_dir / "stage1.log").write_text(log, encoding="utf-8")

    meta = {
        "exit_code": code,
        "passes": STAGE1_PASSES,
        "output": str(after),
        "diagnostic": parse_unconverted_diagnostic(log),
    }
    if code != 0 or not after.exists():
        meta["failed"] = True
        meta["error"] = _failure_reason(code, log)
        return {}, meta

    by_src = read_conversion_result(after)

    results: dict[str, dict] = {}
    for op_type, info in ops.items():
        if op_type.startswith("_"):
            continue
        converted = 0
        unconverted = 0
        folded = 0
        targets: dict[str, int] = {}
        helpers: dict[str, int] = {}
        unconverted_lines: list[int] = []
        for inst in info["instances"]:
            entry = by_src.get(inst["line_no"])
            if entry is None:
                # No output carries this source line: the operation was
                # folded, or fused into a neighbour that now owns the result.
                # Either way it converted, leaving nothing behind.
                converted += 1
                folded += 1
                continue
            if entry["surviving_onnx"]:
                unconverted += 1
                unconverted_lines.append(inst["line_no"])
            else:
                converted += 1
            for t in entry["targets"]:
                targets[t] = targets.get(t, 0) + 1
            for h in entry["helpers"]:
                helpers[h] = helpers.get(h, 0) + 1
        # Shape and ConstantOfShape lower entirely into what is normally DPS
        # scaffolding; with nothing else to report, report that.
        if not targets and helpers:
            targets = helpers
            helpers = {}

        # Attributes that did not survive conversion. Free to collect -- the
        # whole-graph output is already in hand -- and only these operators
        # go on to the perturbation test.
        dropped_attrs: list[str] = []
        first = instances_of(info)
        if first is not None:
            entry = by_src.get(first["line_no"])
            produced = " ".join(entry["result_text"]) if entry else ""
            if produced:
                dropped_attrs = [
                    a
                    for a in first.get("attributes", {})
                    if not _attr_present(produced, a)
                ]
        results[op_type] = {
            "domain": info["domain"][0] if info["domain"] else "onnx",
            "count": info["count"],
            "converted": converted,
            "unconverted": unconverted,
            "folded": folded,
            "targets": dict(sorted(targets.items(), key=lambda kv: -kv[1])),
            "helpers": dict(sorted(helpers.items(), key=lambda kv: -kv[1])),
            "unconverted_lines": unconverted_lines[:10],
            "dropped_attrs": dropped_attrs,
        }
    return results, meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Compile-based operator support probe")
    ap.add_argument("mlir_path", type=Path)
    ap.add_argument("ops_json", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--package", type=Path, required=True, help="gpu-test-package root")
    ap.add_argument(
        "--stage1-only",
        action="store_true",
        help="Skip the per-operator lowering probe",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    opt = _tool(args.package, "hip-mlir-opt.exe")
    ops = json.loads(args.ops_json.read_text(encoding="utf-8"))

    results, meta = stage1(opt, args.mlir_path, ops, args.out_dir)

    print(f"stage1 exit={meta['exit_code']}")
    if meta.get("failed"):
        print(f"  FAILED: {meta.get('error', '')}")
        print("  falling back to per-operator slices")
        results = stage1_per_operator(opt, args.mlir_path, ops, args.out_dir)
        meta["fallback"] = "per-operator slices"
        meta["fallback_note"] = (
            "whole-graph conversion failed; each operator was converted in "
            "isolation, so fusions needing surrounding context could not fire "
            "and support is a lower bound"
        )

    print(f"{'operator':36s} {'conv':>5s} {'unconv':>7s} {'folded':>7s}  targets")
    print("-" * 104)
    for op, r in sorted(results.items(), key=lambda kv: -kv[1]["count"]):
        tgt = ", ".join(f"{k}x{v}" for k, v in list(r["targets"].items())[:3])
        if not tgt and r["folded"]:
            tgt = "(folded at compile time)"
        print(
            f"{op:36s} {r['converted']:5d} {r['unconverted']:7d} "
            f"{r['folded']:7d}  {tgt}"
        )
    print()
    print(f"compiler diagnostic (cross-check): {meta['diagnostic']}")

    s2: dict[str, dict] = {}
    if not args.stage1_only:
        print()
        s2 = stage2(opt, args.mlir_path, ops, results, args.out_dir)
        print(f"{'operator':36s} {'stage2':16s}  runtime / detail")
        print("-" * 104)
        for op, r in sorted(s2.items(), key=lambda kv: -ops[kv[0]]["count"]):
            if r["status"] == "ok":
                detail = (
                    ", ".join(r["runtime_funcs"])
                    if r["runtime_funcs"]
                    else "(compile-time only)"
                )
            else:
                detail = r.get("error") or r.get("reason", "")
            print(f"{op:36s} {r['status']:16s}  {detail[:60]}")

        flagged = [
            (op, f)
            for op, r in s2.items()
            for f in r.get("attributes", [])
            if f["verdict"] != "handled"
        ]
        if flagged:
            print()
            print("attributes the converter does not read:")
            for op, f in flagged:
                note = "" if f["verdict"] == "ignored" else f"  ({f['detail']})"
                print(
                    f"  [{f['verdict']:12s}] {op}.{f['attribute']} = "
                    f"{f.get('value', '')}{note}"
                )

    out = {
        "stage1": {"meta": meta, "operators": results},
        "stage2": {"passes": STAGE2_PASSES, "operators": s2},
    }
    (args.out_dir / "probe_result.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print()
    print(f"[OK] {args.out_dir / 'probe_result.json'}")


if __name__ == "__main__":
    main()
