#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Map each HIP dialect op to the runtime function that executes it.

This answers a different question from operator support. Whether an ONNX
operator converts is decided by running the conversion; which runtime call a
resulting `hip.*` op lowers to is written down in one place, the HIP-to-LLVM
lowering, as a symbol constant per pattern. Reading that is a lookup, not an
inference, and the key comes from an op the conversion actually produced.

Three files are read:

  HipOps.td              op class <-> mnemonic, so `MatmulOp` becomes hip.matmul
  HipToLLVM/*.cpp        which kWrap* symbol each op's lowering calls
  HipToLLVMUtils.h       what those symbols expand to ("wrap_hipblasLtMatmul")

The backend column comes from the runtime implementation of that function: a
file calling hipBLASLt is hipBLASLt, one calling hipDNN is hipDNN, anything
else is a custom HIP kernel. An op whose lowering cannot be found is reported
as unknown rather than guessed.
"""

import argparse
import json
import re
from pathlib import Path

_TD_OP_RE = re.compile(r'^def\s+Hip_(\w+?Op)\s*:\s*[\w_]+<\s*"([\w.]+)"', re.M)
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


def op_class_to_mnemonic(td_path: Path):
    text = td_path.read_text(encoding="utf-8", errors="replace")
    return {cls: f"hip.{mnemonic}" for cls, mnemonic in _TD_OP_RE.findall(text)}


def wrap_symbols(utils_path: Path):
    text = utils_path.read_text(encoding="utf-8", errors="replace")
    return dict(_SYMBOL_RE.findall(text))


def _first_symbol(symbols):
    return next((s for s in symbols if s not in _PLUMBING_SYMBOLS), None)


def lowering_calls(lowering_dir: Path):
    """Op class -> (symbol constant, file name).

    A lowering states its op in one of two ways. A pattern written for one op
    names it in `ConvertOpToLLVMPattern<XOp>` and calls its symbol in the body.
    A pattern shared by several ops is a template, so the op comes from the
    registration (`patterns.insert<ReduceOpLowering<ReduceSumOp>>(...)`) and the
    symbol from either that call or the shared body.
    """
    calls = {}
    for path in sorted(lowering_dir.glob("*.cpp")):
        text = path.read_text(encoding="utf-8", errors="replace")

        structs = list(_STRUCT_RE.finditer(text))
        body_symbol = {}
        struct_op = {}
        for index, struct in enumerate(structs):
            end = structs[index + 1].start() if index + 1 < len(structs) else len(text)
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


def backend_of(runtime_func: str, runtime_dir: Path):
    """Which libraries implement this wrapper, from its own source file."""
    definition = re.compile(rf"\bint\s+{re.escape(runtime_func)}\s*\(")
    for path in sorted(runtime_dir.glob("*.cpp")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if not definition.search(text):
            continue
        found = [name for name, marker in _BACKEND_MARKERS if marker.search(text)]
        return " + ".join(found) if found else _HELPER_BACKEND, path.name
    return "Unknown", ""


def build_map(repo_root: Path):
    td_path = repo_root / "include" / "hip" / "Dialect" / "IR" / "HipOps.td"
    lowering_dir = repo_root / "lib" / "Conversion" / "HipToLLVM"
    utils_path = lowering_dir / "HipToLLVMUtils.h"
    runtime_dir = repo_root / "lib" / "Runtime" / "real"

    for path in (td_path, utils_path):
        if not path.is_file():
            raise SystemExit(f"Not found: {path}")
    if not lowering_dir.is_dir() or not runtime_dir.is_dir():
        raise SystemExit(f"Not found: {lowering_dir} or {runtime_dir}")

    mnemonics = op_class_to_mnemonic(td_path)
    symbols = wrap_symbols(utils_path)
    calls = lowering_calls(lowering_dir)

    backend_cache = {}
    rows = {}
    for op_class, (symbol, lowering_file) in calls.items():
        mnemonic = mnemonics.get(op_class)
        if not mnemonic:
            continue
        runtime_func = symbols.get(symbol)
        if not runtime_func:
            continue
        if runtime_func not in backend_cache:
            backend_cache[runtime_func] = backend_of(runtime_func, runtime_dir)
        backend, runtime_file = backend_cache[runtime_func]
        rows[mnemonic] = {
            "hip_op": mnemonic,
            "runtime_func": runtime_func,
            "backend": backend,
            "lowering_file": lowering_file,
            "runtime_file": runtime_file,
        }
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo_root", help="hip-ep repository root")
    ap.add_argument("output_dir", help="Directory for hip_runtime_map.json")
    args = ap.parse_args()

    rows = build_map(Path(args.repo_root))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "hip_runtime_map.json"
    out_path.write_text(
        json.dumps({"ops": rows}, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[OK] {out_path}")
    print(f"  {len(rows)} hip op(s) mapped to a runtime function")


if __name__ == "__main__":
    main()
