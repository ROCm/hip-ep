#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
What the HIP dialect declares for each operation.

Two consumers need this and both key on an op the conversion actually
produced: the runtime map turns an op class into a mnemonic, and the
attribute check needs to know which attributes an op declares.

That second question exists because MLIR omits an attribute equal to its ODS
default when printing. `hip.gqa` declares `softcap` and the conversion sets
it, yet a module where every instance uses the default prints no `softcap` at
all. Without the declaration, "absent from the printed op" reads as "the
conversion dropped it", which is the opposite of the truth.

TableGen could answer this exactly, but only through llvm-tblgen plus the
MLIR include tree, which a hip-ep package does not carry. The subset needed
here is one `let arguments` block per op, so it is read directly. A parse that
fails leaves the op without declarations, which brings back the over-report it
was meant to avoid rather than inventing support.
"""

import re
from pathlib import Path

_DEF_RE = re.compile(
    r'^def\s+Hip_(?P<cls>\w+?Op)\s*:\s*(?P<base>[\w_]+)\s*<\s*"(?P<mnemonic>[\w.]+)"',
    re.M,
)
_CLASS_RE = re.compile(r"^class\s+(?P<name>Hip_\w+)\s*<", re.M)
_ARGUMENTS_RE = re.compile(r"let\s+arguments\s*=\s*\(ins\b", re.S)
_DEFAULT_ATTR_RE = re.compile(
    r'DefaultValuedO?p?t?i?o?n?a?l?Attr<[^>]*,\s*"([^"]*)"\s*>'
)


def _balanced(text: str, start: int, opening: str, closing: str) -> str:
    """Substring from `start` (at `opening`) through its matching close."""
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def _split_entries(body: str):
    """Top-level comma-separated entries of an (ins ...) list."""
    entries, depth, start = [], 0, 0
    for index, char in enumerate(body):
        if char in "<([{":
            depth += 1
        elif char in ">)]}":
            depth -= 1
        elif char == "," and depth == 0:
            entries.append(body[start:index])
            start = index + 1
    entries.append(body[start:])
    return [e.strip() for e in entries if e.strip()]


def _attributes_of(block: str):
    """Attribute name -> declared default (None when it has none)."""
    match = _ARGUMENTS_RE.search(block)
    if not match:
        return {}
    args = _balanced(block, block.index("(", match.start()), "(", ")")
    attributes = {}
    for entry in _split_entries(args[1:-1].replace("ins", "", 1)):
        name = entry.rsplit(":$", 1)
        if len(name) != 2:
            continue
        type_text, attr_name = name[0], name[1].split()[0]
        # Everything else in the list is an operand.
        if "Attr" not in type_text:
            continue
        default = _DEFAULT_ATTR_RE.search(type_text)
        attributes[attr_name] = default.group(1) if default else None
    return attributes


def load_hip_ops(td_path: Path):
    """Mnemonic -> {"cls", "attributes"} for every op in HipOps.td."""
    text = Path(td_path).read_text(encoding="utf-8", errors="replace")

    class_bodies = {}
    for match in _CLASS_RE.finditer(text):
        brace = text.find("{", match.end())
        semicolon = text.find(";", match.end())
        if brace >= 0 and (semicolon < 0 or brace < semicolon):
            class_bodies[match.group("name")] = _balanced(text, brace, "{", "}")

    ops = {}
    for match in _DEF_RE.finditer(text):
        brace = text.find("{", match.end())
        semicolon = text.find(";", match.end())
        body = ""
        if brace >= 0 and (semicolon < 0 or brace < semicolon):
            body = _balanced(text, brace, "{", "}")
        attributes = _attributes_of(body)
        if not attributes:
            # Ops that declare nothing of their own inherit the base's list.
            attributes = _attributes_of(class_bodies.get(match.group("base"), ""))
        ops[f"hip.{match.group('mnemonic')}"] = {
            "cls": match.group("cls"),
            "attributes": attributes,
        }
    return ops


def hip_ops_td_path(repo_root: Path) -> Path:
    return Path(repo_root) / "include" / "hip" / "Dialect" / "IR" / "HipOps.td"


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: hip_op_defs.py <repo_root>")
    print(json.dumps(load_hip_ops(hip_ops_td_path(Path(sys.argv[1]))), indent=2))
