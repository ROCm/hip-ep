#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Minimal reader for the textual MLIR the compatibility pipeline works with.

Two forms appear in the dumps:

  generic (unregistered onnx dialect, as the MorphiZen importer writes it)
    %1 = "onnx.Add"(%arg0, %arg0) {onnx_node_name = "a"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>

  custom (registered hip dialect, after convert-onnx-to-hip)
    %9 = hip.rms_norm(%arg0) ins(%8, %2 : tensor<?xf32>, tensor<2048xf32>) outs(%7 : tensor<?xf32>) {axis = -1 : i64} : tensor<?xf32>

Both are single-line per operation, so a line scanner is enough; a real
parse would need MLIR itself, which is not available from Python here.
Ops that open a region (func.func, module) are recognized by name only.

Positions come from `--mlir-print-debuginfo`: re-parsing a text file gives
every op a FileLineCol location into that file, and the conversion
propagates it to the ops it creates. That makes the location string a
usable join key between the pre- and post-conversion dumps.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Carrier / terminator ops that exist in the MLIR form but are not graph
# compute nodes. Initializers in particular become one onnx.Constant each,
# which would swamp an operator distribution taken from the protobuf.
# onnx.Yield terminates an If or Loop region the way onnx.Return terminates a
# function, so it is structure the importer wrote rather than a node the model
# contains -- the protobuf has no Yield to compare it against.
NON_COMPUTE_ONNX_OPS = frozenset(
    {
        "onnx.Constant",
        "onnx.NoValue",
        "onnx.Return",
        "onnx.Yield",
        "onnx.EntryPoint",
    }
)

_LOC_ALIAS_RE = re.compile(r"^#(loc\w*)\s*=\s*(.+?)\s*$")
_TRAILING_LOC_RE = re.compile(r"\s+loc\((#?\S+?)\)\s*$")
_GENERIC_OP_RE = re.compile(r'^(?:%\S+(?::\s*\d+)?\s*=\s*)?"([A-Za-z_][\w.]*)"\s*\(')
_CUSTOM_OP_RE = re.compile(r"^(?:%\S+(?::\s*\d+)?\s*=\s*)?([a-z][\w]*\.[\w.]+)\b")
# `%3 = ` names one result, `%827:3 = ` names three reached as %827#0..#2.
_RESULT_RE = re.compile(r"^(%[\w$.]+)(?::\s*(\d+))?\s*=\s*")

_ELEM_TYPE_NAMES = {
    "f16": "float16",
    "f32": "float32",
    "f64": "float64",
    "bf16": "bfloat16",
    "i1": "bool",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "ui8": "uint8",
    "ui16": "uint16",
    "ui32": "uint32",
    "ui64": "uint64",
}


@dataclass
class MlirOp:
    """One operation line."""

    name: str
    attrs: Dict[str, str] = field(default_factory=dict)
    type_signature: str = ""
    loc: str = ""
    line_no: int = 0
    scope: str = "main"
    # SSA name this op defines, with how many results it carries: `%827:3`
    # defines %827#0 through %827#2. Empty for an op that yields nothing.
    result: str = ""
    result_count: int = 0
    # SSA names the op reads, in order, for an op printed in generic form.
    # The custom form spreads its operands over ins()/outs() groups and puts
    # only the context in the leading parentheses, so it is left empty there
    # rather than reported as if it were the whole operand list.
    operands: List[str] = field(default_factory=list)
    generic: bool = False
    # The line with its trailing location removed, so a probe can re-emit the
    # op instead of rebuilding it from the parsed pieces.
    text: str = ""
    # The op opens a region: its body is on the lines that follow and its type
    # signature is on the line that closes it, so neither is on this one. A
    # line scanner cannot reproduce such an op, only report that it is here.
    opens_region: bool = False

    @property
    def dialect(self) -> str:
        return self.name.split(".", 1)[0]

    def onnx_key(self) -> Tuple[str, str]:
        """(op_type, domain) for an onnx.* op, matching ONNX protobuf naming.

        onnx.Custom is a container whose real operator is in `function_name`,
        so report that instead of collapsing every custom op into "Custom".
        """
        bare = self.name.split(".", 1)[1] if "." in self.name else self.name
        if self.name != "onnx.Custom":
            return bare, "ai.onnx"
        op_type = strip_quotes(self.attrs.get("function_name", "")) or "Custom"
        domain = strip_quotes(self.attrs.get("domain_name", "")) or "com.microsoft"
        return op_type, domain

    def onnx_attr_names(self) -> List[str]:
        """Attribute names that came from the ONNX node.

        `function_name` / `domain_name` select the operator for onnx.Custom and
        `node.outputs` / `onnx_node_name` are importer bookkeeping, so none of
        them is an operator attribute that a HIP op could carry.
        """
        skip = {"function_name", "domain_name", "node.outputs", "onnx_node_name"}
        return sorted(k for k in self.attrs if k not in skip)


def strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1]
    return value


def _split_top_level(text: str, separator: str = ",") -> List[str]:
    """Split on `separator` outside quotes, brackets, braces and angle types."""
    parts, depth, start, in_string, escaped = [], 0, 0, False, False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        elif ch == separator and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


def _find_attr_dict(line: str) -> Tuple[Optional[str], int]:
    """Return the operation's attribute dictionary body and where it started.

    The dictionary is the last brace group that the type signature follows,
    which distinguishes it from braces inside an attribute value.
    """
    idx = len(line)
    while True:
        close = line.rfind("}", 0, idx)
        if close < 0:
            return None, -1
        after = line[close + 1 :].lstrip()
        if after.startswith(":"):
            open_idx = _match_backwards(line, close)
            if open_idx >= 0:
                return line[open_idx + 1 : close], open_idx
        idx = close


def _match_backwards(line: str, close_idx: int) -> int:
    depth = 0
    in_string = False
    for i in range(close_idx, -1, -1):
        ch = line[i]
        if in_string:
            if ch == '"' and (i == 0 or line[i - 1] != "\\"):
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "}":
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _match_forward(line: str, open_idx: int) -> int:
    depth = 0
    in_string = False
    for i in range(open_idx, len(line)):
        ch = line[i]
        if in_string:
            if ch == '"' and line[i - 1] != "\\":
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _operands(line: str, open_idx: int) -> List[str]:
    """SSA names in the parenthesis group starting at `open_idx`.

    Keeping only `%` entries drops the type list that follows the operands in
    an ins()/outs() group, so the result is operands either way.
    """
    close = _match_forward(line, open_idx)
    if close < 0:
        return []
    return [
        part
        for part in _split_top_level(line[open_idx + 1 : close])
        if part.startswith("%")
    ]


def _parse_attrs(body: Optional[str]) -> Dict[str, str]:
    if not body:
        return {}
    attrs = {}
    for entry in _split_top_level(body):
        key, sep, value = entry.partition("=")
        key = key.strip()
        if not key:
            continue
        # A bare key is a unit attribute (`{value}` on onnx.NoValue).
        attrs[key] = value.strip() if sep else ""
    return attrs


def _type_signature(line: str, attr_start: int) -> str:
    """Text after the operation's final top-level colon."""
    search_from = attr_start if attr_start >= 0 else 0
    depth = 0
    in_string = False
    for i in range(search_from, len(line)):
        ch = line[i]
        if in_string:
            if ch == '"' and line[i - 1] != "\\":
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        elif ch == ":" and depth == 0:
            return line[i + 1 :].strip()
    return ""


def tensor_types(type_signature: str) -> List[str]:
    return re.findall(r"tensor<([^<>]*)>", type_signature)


def element_type(tensor_type: str) -> str:
    """The MLIR element type of one tensor type, shape stripped.

    Accepts either the whole type (`tensor<2x?xf32>`) or the body a
    `tensor_types` match returns (`2x?xf32`).
    """
    body = tensor_type.strip()
    if body.startswith("tensor<") and body.endswith(">"):
        body = body[len("tensor<") : -1]
    return (body.rsplit("x", 1)[-1] if "x" in body else body).strip()


def element_type_names(type_signature: str) -> List[str]:
    """Readable element-type names for every tensor in a type signature."""
    names = []
    for body in tensor_types(type_signature):
        elem = element_type(body)
        names.append(_ELEM_TYPE_NAMES.get(elem, elem))
    return [n for n in names if n]


def split_signature(type_signature: str) -> Tuple[List[str], List[str]]:
    """Operand and result types of `(t, t) -> t` or `(t) -> (t, t)`.

    A single result prints without parentheses, so the right side is a group
    only when it has one.
    """
    depth = 0
    in_string = False
    arrow = -1
    for i, ch in enumerate(type_signature):
        if in_string:
            if ch == '"' and type_signature[i - 1] != "\\":
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        elif ch == "-" and depth == 0 and type_signature[i : i + 2] == "->":
            arrow = i
            break
    if arrow < 0:
        return [], []
    return (
        _type_group(type_signature[:arrow]),
        _type_group(type_signature[arrow + 2 :]),
    )


def _type_group(text: str) -> List[str]:
    text = text.strip()
    if text.startswith("(") and text.endswith(")"):
        return _split_top_level(text[1:-1])
    return [text] if text else []


class MlirModule:
    """Operations of one textual MLIR module, in source order."""

    def __init__(self, ops: List[MlirOp], loc_aliases: Dict[str, str]):
        self.ops = ops
        self.loc_aliases = loc_aliases
        self._by_result: Dict[str, List[MlirOp]] = {}
        for op in ops:
            if op.result:
                self._by_result.setdefault(op.result, []).append(op)

    def defining_op(self, ssa: str, before_line: int) -> Optional[MlirOp]:
        """The op defining `ssa` as read from `before_line`, or None for a
        block argument.

        An SSA name belongs to its region, so a name defined in one onnx.If or
        onnx.Loop body can be defined again in a sibling body, and a dump of a
        model with a few of them reuses names freely. The definition in force
        at a use is the nearest one above it. Taking any definition of the name
        crosses region boundaries and resolves to an unrelated value, which is
        worse than failing to resolve: it answers with a real operation of the
        wrong type.

        An operand may name one result of several (`%827#1`), which the
        defining op carries under its base name.
        """
        found = None
        for op in self._by_result.get(ssa.split("#", 1)[0]) or []:
            if op.line_no >= before_line:
                break
            found = op
        return found

    def resolve_loc(self, op: MlirOp) -> str:
        if not op.loc:
            return ""
        if op.loc.startswith("#"):
            return self.loc_aliases.get(op.loc[1:], op.loc)
        return op.loc

    def onnx_ops(self, include_non_compute: bool = False) -> List[MlirOp]:
        return [
            op
            for op in self.ops
            if op.dialect == "onnx"
            and (include_non_compute or op.name not in NON_COMPUTE_ONNX_OPS)
        ]


def parse_mlir(text: str) -> MlirModule:
    loc_aliases: Dict[str, str] = {}
    ops: List[MlirOp] = []
    scope = "main"

    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line in {"}", "})", "{"}:
            continue

        if line.startswith("#"):
            alias = _LOC_ALIAS_RE.match(line)
            if alias:
                loc_aliases[alias.group(1)] = alias.group(2)
            continue

        loc = ""
        trailing = _TRAILING_LOC_RE.search(line)
        if trailing:
            loc = trailing.group(1)
            line = line[: trailing.start()].rstrip()

        # Track the enclosing function so nested (outlined loop/if body) ops
        # are attributed to it rather than to the main graph.
        func = re.match(r"^func\.func\s+(?:private\s+)?@([\w$.]+)", line)
        if func:
            scope = func.group(1)
            continue
        if line.startswith("module"):
            continue

        generic = _GENERIC_OP_RE.match(line)
        match = generic or _CUSTOM_OP_RE.match(line)
        if not match:
            continue
        name = match.group(1)
        if "." not in name:
            continue

        result = _RESULT_RE.match(line)
        attr_body, attr_start = _find_attr_dict(line)
        ops.append(
            MlirOp(
                name=name,
                attrs=_parse_attrs(attr_body),
                type_signature=_type_signature(line, attr_start),
                loc=loc,
                line_no=line_no,
                scope=scope,
                result=result.group(1) if result else "",
                result_count=(int(result.group(2) or 1) if result else 0),
                # The regex ends at the opening parenthesis of the operand list.
                operands=_operands(line, generic.end() - 1) if generic else [],
                generic=generic is not None,
                text=line,
                opens_region=line.endswith("{"),
            )
        )

    return MlirModule(ops, loc_aliases)


def parse_mlir_file(path) -> MlirModule:
    from pathlib import Path

    return parse_mlir(Path(path).read_text(encoding="utf-8"))
