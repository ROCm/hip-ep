#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Turn one line of MLIR into a module the compiler will accept.

Both probes need this and neither owns it: probe.py slices an operator out
of the whole-graph dump, single_op_probe.py out of a one-node import. What
they share is everything between an operation's source line and a module
that can be handed to hip-mlir-opt.
"""

from __future__ import annotations

import re

_SSA_DEF = re.compile(r"^\s*(%[\w$.-]+)(?::\d+)?\s*=")


_SSA_REF = re.compile(r"(%[\w$.-]+)(?:#(\d+))?")


# An onnx.Constant whose data lives at a runtime memory address rather than in
# a file. Inlining one makes constant externalization fail with "memory-address
# sources require production externalization with an injected FileSystem".
_MEM_ADDR_CONST = re.compile(r'location\s*=\s*"\*/_ORT_MEM_ADDR_/\*"')


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


def slices_for(
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


# The element type is the final `x`-separated field. Anchoring on the
# separator matters: a looser pattern lets backtracking pull it into the
# element name, reading tensor<512x16xf32> as elem "xf32".
_ELEM_OF_TENSOR = re.compile(r"^tensor<(|.*x)([A-Za-z]+\d*)>$")


def element_type(tensor_type: str) -> str | None:
    """The element type of a tensor type, or None if it is not one."""
    m = _ELEM_OF_TENSOR.match(tensor_type.strip())
    return m.group(2) if m else None


def retype(tensor_type: str, new_elem: str) -> str | None:
    m = _ELEM_OF_TENSOR.match(tensor_type.strip())
    if not m:
        return None
    return f"tensor<{m.group(1)}{new_elem}>"


def build_def_index(src_lines: list[str]) -> dict[str, int]:
    """Map each SSA result name to the line that defines it."""
    index: dict[str, int] = {}
    for i, line in enumerate(src_lines, 1):
        m = _SSA_DEF.match(line)
        if m:
            index.setdefault(m.group(1), i)
    return index
