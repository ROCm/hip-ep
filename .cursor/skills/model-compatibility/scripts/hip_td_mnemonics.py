"""
Shared TableGen helper: map mlir::hip C++ op class (e.g. ReduceSumOp) to the
string inside Hip_(Dps)Op<"…"> in HipOps.td (e.g. reduce_sum → hip.reduce_sum).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

_HIP_DEF_RE = re.compile(r"def\s+(Hip_\w+Op)\s*:\s*([^{]+)\s*\{", re.DOTALL)
_HIP_MNEMONIC_RE = re.compile(r'Hip_(?:Dps)?Op<\s*"([^"]+)"')


def load_cpp_op_class_to_mnemonic_from_td(td_path: Path) -> Dict[str, str]:
    """
    Returns e.g. {"ReduceSumOp": "reduce_sum", "MiopenSoftmaxOp": "miopen.softmax"}.
    Keys match ConvertOpToLLVMPattern<ReduceSumOp> template parameters in C++.
    """
    text = td_path.read_text(encoding="utf-8")
    out: Dict[str, str] = {}
    for m in _HIP_DEF_RE.finditer(text):
        tablegen_def = m.group(1).strip()
        base = m.group(2).strip()
        mm = _HIP_MNEMONIC_RE.search(base)
        if not mm:
            continue
        if not tablegen_def.startswith("Hip_"):
            continue
        cpp_class = tablegen_def[len("Hip_") :]
        out[cpp_class] = mm.group(1)
    return out


def default_hip_ops_td_path(conversion_dir: Path) -> Path:
    """lib/Conversion -> repo root -> include/hip/Dialect/IR/HipOps.td"""
    return (
        conversion_dir.parent.parent
        / "include"
        / "hip"
        / "Dialect"
        / "IR"
        / "HipOps.td"
    )
