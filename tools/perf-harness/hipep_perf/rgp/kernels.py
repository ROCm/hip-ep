#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Decode a demangled, simplified kernel name into family / dtype / tile.

Two shapes dominate a HIP LLM capture and both encode their tuning config in the
name:

* Tensile / hipBLASLt GEMMs, ``Cijk_Ailk_Bljk_HHS_BH_Bias_..._MT128x48x32_
  MI16x16x1_..._WG64_2_1_..._ISA1151`` -> family ``gemm``, dtype ``f16``, tile
  ``128x48x32``, label ``hgemm 128x48x32 wmma16x16 wg64x2 +bias``.
* First-party HIP kernels, ``rope_kernel<__half>`` -> family ``rope``, dtype
  ``f16``; numeric template args become the config.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_TENSILE_DTYPE = {
    "H": "f16",
    "S": "f32",
    "B": "bf16",
    "D": "f64",
    "I": "i8",
    "C": "c32",
    "Z": "c64",
}
_GEMM_PREFIX = {
    "f16": "hgemm",
    "f32": "sgemm",
    "f64": "dgemm",
    "bf16": "bf16 gemm",
    "i8": "i8 gemm",
}
_CPP_DTYPE = {
    "__half": "f16",
    "_Float16": "f16",
    "half": "f16",
    "__hip_bfloat16": "bf16",
    "__bf16": "bf16",
    "hip_bfloat16": "bf16",
    "float": "f32",
    "double": "f64",
    "int8_t": "i8",
    "signed char": "i8",
    "char": "i8",
    "uint8_t": "u8",
    "unsigned char": "u8",
    "short": "i16",
    "int": "i32",
    "unsigned int": "u32",
    "long": "i64",
    "long long": "i64",
    "unsigned long": "u64",
}


@dataclass
class KernelInfo:
    library: str = "other"  # tensile | hip | other | unknown
    family: str = ""
    dtype: str = ""
    tile: str = ""
    mi: str = ""
    workgroup: str = ""
    arch: str = ""
    gsu: str = ""
    flags: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    label: str = ""


def _tensile(name: str) -> KernelInfo:
    info = KernelInfo(library="tensile", family="gemm")
    p: dict = {}
    m = re.match(r"^Cijk_[A-Za-z]+_[A-Za-z]+_([A-Z0-9]{2,4})(?:_|$)", name)
    if m:
        p["type_code"] = m.group(1)
        info.dtype = _TENSILE_DTYPE.get(m.group(1)[0], "")
    if mm := re.search(r"_MT(\d+x\d+x\d+)", name):
        info.tile = p["mt"] = mm.group(1)
    if mm := re.search(r"_MI(\d+x\d+x\d+(?:x\d+)?)", name):
        info.mi = p["mi"] = mm.group(1)
    if mm := re.search(r"_WG(\d+)_(\d+)_(\d+)", name):
        info.workgroup = f"{mm.group(1)}x{mm.group(2)}"
        p["wg"] = f"{mm.group(1)}_{mm.group(2)}_{mm.group(3)}"
    if mm := re.search(r"_ISA(\d+)", name):
        info.arch, p["isa"] = "gfx" + mm.group(1), mm.group(1)
    sk = re.search(r"_SK(\d+)", name)
    if "GSU" in name and sk and sk.group(1) != "0":
        info.gsu = sk.group(1)
    if "_Bias" in name:
        info.flags.append("bias")
    info.params = p
    parts = [_GEMM_PREFIX.get(info.dtype, "gemm")]
    if info.tile:
        parts.append(info.tile)
    if info.mi:
        a, b = info.mi.split("x")[:2]
        parts.append(f"wmma{a}x{b}")
    if info.workgroup:
        parts.append("wg" + info.workgroup)
    label = " ".join(parts)
    if info.gsu:
        label += f" gsu{info.gsu}"
    if "bias" in info.flags:
        label += " +bias"
    info.label = label
    return info


def _split_args(targs: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in targs:
        if ch in "<(":
            depth += 1
        elif ch in ")>":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _hip(name: str) -> KernelInfo:
    info = KernelInfo(library="hip")
    m = re.match(r"^([A-Za-z_][\w:]*)(?:<(.*)>)?$", name.strip())
    if not m:
        info.family = info.label = name
        return info
    family = m.group(1).split("::")[-1]
    family = family.removesuffix("_kernel")
    family = family.removeprefix("hip_")
    info.family = family
    nums, dtype = [], ""
    for a in _split_args(m.group(2)) if m.group(2) else []:
        if a in _CPP_DTYPE:
            dtype = dtype or _CPP_DTYPE[a]
        elif re.fullmatch(r"-?\d+", a):
            nums.append(a)
        elif a.lstrip("(").rstrip(")").lstrip("Li").rstrip("E").isdigit():
            # The demangler can keep literal wrappers such as (int)1 or Li1E.
            nums.append(re.sub(r"\D", "", a))
    info.dtype = dtype
    if nums:
        info.params["config"] = ",".join(nums)
    label = (
        family
        + (f" {dtype}" if dtype else "")
        + (f" <{','.join(nums)}>" if nums else "")
    )
    info.label = label
    return info


def parse(name: str) -> KernelInfo:
    if not name:
        return KernelInfo(library="unknown", family="", label="")
    if name.startswith("0x"):
        return KernelInfo(library="unknown", family="unresolved", label=name)
    if name.startswith("Cijk"):
        return _tensile(name)
    if re.match(r"^[A-Za-z_]", name):
        return _hip(name)
    return KernelInfo(library="other", family=name, label=name)
