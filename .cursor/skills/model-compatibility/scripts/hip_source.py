#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Everything the report needs from hip-ep's own source.

None of it decides support -- that comes from running the conversion. These
are the three questions the run cannot answer about itself, each keyed on
something the run observed:

  op_attributes()   which attributes a hip op declares. MLIR omits an
                    attribute equal to its ODS default when printing, so
                    without the declaration a carried attribute looks dropped.

  runtime_map()     which runtime function a hip op lowers to, and which
                    library implements it. Written once per op in the
                    HIP-to-LLVM lowering, so this is a lookup.

  explain_leftover() why a converter refused an operator. Converters report
                    refusals through notifyMatchFailure, which a release build
                    compiles out, so the messages are read from the source and
                    ranked against the types the model actually used.

Everything here is regex over C++ and TableGen, which is why it is confined to
this file and to questions whose wrong answer degrades an explanation rather
than a verdict. A parse that finds nothing leaves the caller with less
information, never with a claim of support.
"""

import re
from functools import lru_cache
from pathlib import Path

# --- HipOps.td: op mnemonics and their declared attributes ------------------

_DEF_RE = re.compile(
    r'^def\s+Hip_(?P<cls>\w+?Op)\s*:\s*(?P<base>[\w_]+)\s*<\s*"(?P<mnemonic>[\w.]+)"',
    re.M,
)
_CLASS_RE = re.compile(r"^class\s+(?P<name>Hip_\w+)\s*<", re.M)
_ARGUMENTS_RE = re.compile(r"let\s+arguments\s*=\s*\(ins\b", re.S)
_DEFAULT_ATTR_RE = re.compile(
    r'DefaultValuedO?p?t?i?o?n?a?l?Attr<[^>]*,\s*"([^"]*)"\s*>'
)

# --- HipToLLVM: which runtime symbol each op's lowering calls ---------------

_SYMBOL_RE = re.compile(r'(kWrap\w+)\s*=\s*"(\w+)"')
_STRUCT_RE = re.compile(
    r"struct\s+(\w+)\s*(?::\s*public\s+)?[\s\S]{0,120}?ConvertOpToLLVMPattern<\s*(\w+)\s*>"
)
_REGISTRATION_RE = re.compile(
    r"patterns\s*\.\s*(?:add|insert)\s*<([\s\S]*?)>\s*\(([\s\S]*?)\)\s*;"
)
_INSTANTIATION_RE = re.compile(r"(\w+)\s*<\s*(\w+Op)\b")
_PLAIN_PATTERN_RE = re.compile(r"\b(\w+OpLowering)\b")
_KWRAP_USE_RE = re.compile(r"\b(kWrap\w+)\b")

# Helpers every lowering may call for staging buffers; they are not the op's
# own runtime entry point.
_PLUMBING_SYMBOLS = {
    "kWrapHipMemcpyAsync",
    "kWrapHipMemcpy2DAsync",
    "kWrapStridedCopy",
}

# Calls into a library, not mentions of it. Every runtime file includes
# hipdnn_ep_runtime.h, so a substring match on "hipdnn" would tag them all.
# A wrapper can match several: GQA and MatMulNBits drive hipBLASLt for their
# matmuls and custom kernels for everything around them.
_BACKEND_MARKERS = [
    ("hipBLASLt", re.compile(r"\bhipblasLt\w*\s*\(")),
    ("hipDNN", re.compile(r"\bhipdnn[A-Z]\w*\s*\(")),
    ("MIOpen", re.compile(r"\bmiopen[A-Z]\w*\s*\(")),
    (
        "Custom Hip Kernel",
        re.compile(r"hip_custom_kernels\.h|\blaunch_\w+\s*\(|hip_\w+_kernel"),
    ),
    # An embedded binary launched through the module API, as rocMLIR kernels are.
    ("Embedded GPU module", re.compile(r"\bhipModule\w*\s*\(")),
]

# No library and no kernel: the wrapper only moves or computes a value on the
# host, such as wrap_size depositing a scalar into a device buffer.
_HELPER_BACKEND = "Runtime helper"

# --- OnnxToHip: why a converter refused --------------------------------------

# MLIR scalar type spellings, as they appear both in type signatures and in
# converter messages ("Expected i8 (packed uint4) or f16"). Matching the shape
# rather than listing the types keeps new ones (f8E4M3FN, i4) working.
_TYPE_TOKEN_RE = re.compile(r"\b(?:[su]?i\d+|[fu]\d+|f8E\w+|f4E\w+|bf16|index)\b")
_STRING_LITERAL_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
_NOTIFY_RE = re.compile(r"notifyMatchFailure\s*\(")
# How far above a notifyMatchFailure call its message may be built. Messages
# assembled through raw_string_ostream span several lines.
_MESSAGE_LOOKBEHIND = 14


class HipSource:
    """The hip-ep tree, answering one question per method."""

    def __init__(self, repo_root):
        self.root = Path(repo_root)
        self.hip_ops_td = self.root / "include" / "hip" / "Dialect" / "IR" / "HipOps.td"
        self.lowering_dir = self.root / "lib" / "Conversion" / "HipToLLVM"
        self.conversion_dir = self.root / "lib" / "Conversion" / "OnnxToHip"
        self.runtime_dir = self.root / "lib" / "Runtime" / "real"

    def missing_paths(self):
        return [
            str(p)
            for p in (
                self.hip_ops_td,
                self.lowering_dir,
                self.conversion_dir,
                self.runtime_dir,
            )
            if not p.exists()
        ]

    # -- dialect ------------------------------------------------------------

    @lru_cache(maxsize=1)
    def _ops(self):
        """Mnemonic -> {"cls", "attributes"} for every op in HipOps.td."""
        text = self.hip_ops_td.read_text(encoding="utf-8", errors="replace")

        class_bodies = {}
        for match in _CLASS_RE.finditer(text):
            body = _body_after(text, match.end())
            if body:
                class_bodies[match.group("name")] = body

        ops = {}
        for match in _DEF_RE.finditer(text):
            body = _body_after(text, match.end()) or ""
            attributes = _attributes_of(body)
            if not attributes:
                # Ops that declare nothing of their own inherit the base's list.
                attributes = _attributes_of(class_bodies.get(match.group("base"), ""))
            ops[f"hip.{match.group('mnemonic')}"] = {
                "cls": match.group("cls"),
                "attributes": attributes,
            }
        return ops

    def op_attributes(self):
        """Hip op mnemonic -> {attribute name: declared default or None}."""
        return {name: info["attributes"] for name, info in self._ops().items()}

    # -- lowering and runtime ------------------------------------------------

    def runtime_map(self, keep_ops=None):
        """Hip op mnemonic -> runtime function, backend and where each was read."""
        by_class = {info["cls"]: name for name, info in self._ops().items()}
        symbols = dict(
            _SYMBOL_RE.findall(
                (self.lowering_dir / "HipToLLVMUtils.h").read_text(
                    encoding="utf-8", errors="replace"
                )
            )
        )

        rows = {}
        backends = {}
        for op_class, (symbol, lowering_file) in self._lowering_calls().items():
            mnemonic = by_class.get(op_class)
            runtime_func = symbols.get(symbol)
            if not mnemonic or not runtime_func:
                continue
            if keep_ops is not None and mnemonic not in keep_ops:
                continue
            if runtime_func not in backends:
                backends[runtime_func] = self._backend_of(runtime_func)
            backend, runtime_file = backends[runtime_func]
            rows[mnemonic] = {
                "hip_op": mnemonic,
                "runtime_func": runtime_func,
                "backend": backend,
                "lowering_file": lowering_file,
                "runtime_file": runtime_file,
            }
        return rows

    @lru_cache(maxsize=1)
    def _lowering_calls(self):
        """Op class -> (symbol constant, file name).

        A lowering states its op in one of two ways. A pattern written for one
        op names it in `ConvertOpToLLVMPattern<XOp>` and calls its symbol in
        the body. A pattern shared by several ops is a template, so the op
        comes from the registration and the symbol from either that call or
        the shared body.
        """
        calls = {}
        for path in sorted(self.lowering_dir.glob("*.cpp")):
            text = path.read_text(encoding="utf-8", errors="replace")

            structs = list(_STRUCT_RE.finditer(text))
            body_symbol = {}
            struct_op = {}
            for index, struct in enumerate(structs):
                end = (
                    structs[index + 1].start()
                    if index + 1 < len(structs)
                    else len(text)
                )
                symbol = _first_symbol(
                    dict.fromkeys(_KWRAP_USE_RE.findall(text[struct.end() : end]))
                )
                name, op_class = struct.group(1), struct.group(2)
                if symbol:
                    body_symbol[name] = symbol
                # "OpTy" and friends are template parameters, not an op.
                if op_class.endswith("Op") and op_class != "OpTy":
                    struct_op[name] = op_class

            for name, op_class in struct_op.items():
                if name in body_symbol:
                    calls.setdefault(op_class, (body_symbol[name], path.name))

            for registration in _REGISTRATION_RE.finditer(text):
                template_args, call_args = registration.group(1), registration.group(2)
                call_symbol = _first_symbol(_KWRAP_USE_RE.findall(call_args))
                for struct_name, op_class in _INSTANTIATION_RE.findall(template_args):
                    symbol = call_symbol or body_symbol.get(struct_name)
                    if symbol:
                        calls.setdefault(op_class, (symbol, path.name))
                if call_symbol:
                    for struct_name in _PLAIN_PATTERN_RE.findall(template_args):
                        op_class = struct_op.get(struct_name)
                        if op_class:
                            calls.setdefault(op_class, (call_symbol, path.name))
        return calls

    def _backend_of(self, runtime_func: str):
        """Which libraries implement this wrapper, from its own source file."""
        definition = re.compile(rf"\bint\s+{re.escape(runtime_func)}\s*\(")
        for path in sorted(self.runtime_dir.glob("*.cpp")):
            text = path.read_text(encoding="utf-8", errors="replace")
            if not definition.search(text):
                continue
            found = [name for name, marker in _BACKEND_MARKERS if marker.search(text)]
            return " + ".join(found) if found else _HELPER_BACKEND, path.name
        return "Unknown", ""

    # -- conversion refusals --------------------------------------------------

    def explain_leftover(self, op_type: str, observed_types, max_refusals: int = 6):
        """Whether a converter exists for this operator, and what it can refuse."""
        files = self._converter_files(op_type)
        if not files:
            return {"converter_found": False, "converter_files": [], "refusals": []}

        refusals = []
        for path, text in files:
            refusals.extend(_extract_refusals(path, text))
        refusals = _rank_refusals(refusals, observed_types)[:max_refusals]
        return {
            "converter_found": True,
            "converter_files": [path.name for path, _ in files],
            "refusals": refusals,
        }

    def _converter_files(self, op_type: str):
        """Source files that match this operator by name.

        Converters select an operator with a string literal, either the op name
        ("onnx.Cast") or, for onnx.Custom containers, the function name
        ("MatMulNBits"), so only quoted occurrences count. A bare substring
        would match unrelated identifiers.
        """
        wanted = {f'"{op_type}"', f'"onnx.{op_type}"'}
        hits = []
        for path in sorted(self.conversion_dir.rglob("*.cpp")):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if any(token in text for token in wanted):
                hits.append((path, text))
        return hits


# --- helpers -----------------------------------------------------------------


def _body_after(text: str, position: int):
    """The `{...}` body that follows `position`, when there is one."""
    brace = text.find("{", position)
    semicolon = text.find(";", position)
    if brace < 0 or (0 <= semicolon < brace):
        return ""
    depth = 0
    for index in range(brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[brace : index + 1]
    return text[brace:]


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
    open_paren = block.index("(", match.start())
    depth = 0
    for index in range(open_paren, len(block)):
        if block[index] == "(":
            depth += 1
        elif block[index] == ")":
            depth -= 1
            if depth == 0:
                args = block[open_paren + 1 : index]
                break
    else:
        return {}

    attributes = {}
    for entry in _split_entries(args.replace("ins", "", 1)):
        parts = entry.rsplit(":$", 1)
        if len(parts) != 2:
            continue
        type_text, attr_name = parts[0], parts[1].split()[0]
        # Everything else in the list is an operand.
        if "Attr" not in type_text:
            continue
        default = _DEFAULT_ATTR_RE.search(type_text)
        attributes[attr_name] = default.group(1) if default else None
    return attributes


def _first_symbol(symbols):
    return next((s for s in symbols if s not in _PLUMBING_SYMBOLS), None)


def _clean_message(message: str) -> str:
    """Tidy a message whose dynamic parts (a printed type) are not available."""
    message = re.sub(r"\s+", " ", message).strip()
    # "unsupported element type: " + <printed type> + ". Expected ..." leaves a
    # colon with nothing after it.
    message = re.sub(r"\s*:\s*(?=[.,])", "", message)
    return message.strip(" :")


def _extract_refusals(path: Path, text: str):
    """Messages the file can pass to notifyMatchFailure, with line numbers."""
    lines = text.splitlines()
    refusals = []
    previous_notify = -1
    for index, line in enumerate(lines):
        if not _NOTIFY_RE.search(line):
            continue
        # Stay inside this message: never reach back past the previous refusal,
        # whose literals belong to it.
        start = max(previous_notify + 1, index - _MESSAGE_LOOKBEHIND, 0)
        previous_notify = index
        pieces = []
        for window_line in lines[start : index + 2]:
            # Skip comments so explanatory prose is not read as a message.
            if window_line.strip().startswith("//"):
                continue
            pieces.extend(_STRING_LITERAL_RE.findall(window_line))
        message = _clean_message(" ".join(p.strip() for p in pieces if p.strip()))
        if message:
            refusals.append(
                {"message": message, "location": f"{path.name}:{index + 1}"}
            )
    return refusals


def _rank_refusals(refusals, observed):
    """Put refusals the observed types violate first.

    A message that lists accepted types is a likely cause when an observed type
    is missing from that list, and an unlikely one when every observed type is
    covered.
    """

    def score(refusal):
        mentioned = set(_TYPE_TOKEN_RE.findall(refusal["message"]))
        if not mentioned:
            return 0
        return 2 if [t for t in observed if t not in mentioned] else 1

    ordered = sorted(refusals, key=lambda r: -score(r))
    for refusal in ordered:
        refusal["likely"] = score(refusal) == 2
    return ordered
