#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Decode a (already simplified/demangled) kernel name into structured fields.

Two symbol shapes dominate a HIP LLM capture and both encode the tuning config in
the name; :func:`parse_kernel_name` normalizes them (plus a generic fallback) into
a :class:`KernelInfo` so formatters can group by op *family* / dtype / tile instead
of the raw 200-char string:

- Tensile / hipBLASLt GEMMs, e.g.
  ``Cijk_Ailk_Bljk_HHS_BH_Bias_..._MT128x48x32_MI16x16x1_..._WG64_2_1_..._ISA1151``
  -> library ``tensile``, family ``gemm``, dtype ``f16``, tile ``128x48x32``,
  mi ``16x16x1``, workgroup ``64x2``, arch ``gfx1151``, label ``hgemm 128x48x32
  wmma16x16 wg64x2 +bias``.
- First-party HIP kernels (demangled), e.g. ``rope_kernel<__half>``,
  ``gqa_flash_prefill_v5_kernel<1, 32, 64>``, ``matmul_nbits_gemv_dp4a_kernel<64, 8>``
  -> library ``hip``, family from the base (``_kernel`` stripped), dtype from the
  type template arg, config from the numeric template args.

Pure string parsing (no capture / HIP needed), so it is unit-testable in isolation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class KernelInfo:
    library: str = "other"  # tensile | hip | other | unknown
    family: str = ""  # normalized op family, e.g. gemm, rope, gqa_flash_prefill
    dtype: str = ""  # f16 | bf16 | f32 | i8 | ... ("" if undetermined)
    tile: str = ""  # macro tile "MxNxK" (tensile) or ""
    mi: str = ""  # matrix-instruction "MxNxK" (tensile) or ""
    workgroup: str = ""  # workgroup dims "XxY" (tensile) or ""
    arch: str = ""  # e.g. gfx1151
    gsu: str = ""  # split-K / global-split-U hint ("" if none)
    flags: list = field(default_factory=list)  # e.g. ["bias"]
    params: dict = field(default_factory=dict)  # raw extracted fields
    label: str = ""  # short human-readable label


# Tensile precision code -> input dtype (first char is the A/B operand type).
_TENSILE_DTYPE = {
    "H": "f16",
    "S": "f32",
    "B": "bf16",
    "D": "f64",
    "I": "i8",
    "C": "c32",
    "Z": "c64",
}
# GEMM label prefix by dtype.
_GEMM_PREFIX = {
    "f16": "hgemm",
    "f32": "sgemm",
    "f64": "dgemm",
    "bf16": "bf16 gemm",
    "i8": "i8 gemm",
}
# C++ type spellings (as they appear in demangled template args) -> dtype tag.
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


def _parse_tensile(name: str) -> KernelInfo:
    info = KernelInfo(library="tensile", family="gemm")
    p: dict = {}

    # precision code: the token right after the two contraction-index groups,
    # e.g. Cijk_Ailk_Bljk_HHS_...  -> "HHS"
    m = re.match(r"^Cijk_[A-Za-z]+_[A-Za-z]+_([A-Z0-9]{2,4})(?:_|$)", name)
    code = m.group(1) if m else ""
    if code:
        p["type_code"] = code
        info.dtype = _TENSILE_DTYPE.get(code[0], "")

    def grab(pat, key):
        mm = re.search(pat, name)
        if mm:
            p[key] = mm.group(1)
        return mm

    if grab(r"_MT(\d+x\d+x\d+)", "mt"):
        info.tile = p["mt"]
    if grab(r"_MI(\d+x\d+x\d+(?:x\d+)?)", "mi"):
        info.mi = p["mi"]
    mwg = re.search(r"_WG(\d+)_(\d+)_(\d+)", name)
    if mwg:
        info.workgroup = f"{mwg.group(1)}x{mwg.group(2)}"
        p["wg"] = f"{mwg.group(1)}_{mwg.group(2)}_{mwg.group(3)}"
    misa = re.search(r"_ISA(\d+)", name)
    if misa:
        info.arch = "gfx" + misa.group(1)
        p["isa"] = misa.group(1)
    # split-K / global-split-U: GSU token present and an explicit _SK<n> with n>0.
    msk = re.search(r"_SK(\d+)", name)
    if "GSU" in name and msk and msk.group(1) != "0":
        info.gsu = msk.group(1)
    if "_Bias" in name:
        info.flags.append("bias")

    info.params = p
    # readable label
    prefix = _GEMM_PREFIX.get(info.dtype, "gemm")
    parts = [prefix]
    if info.tile:
        parts.append(info.tile)
    if info.mi:
        mm = info.mi.split("x")
        parts.append(f"wmma{mm[0]}x{mm[1]}")
    if info.workgroup:
        parts.append("wg" + info.workgroup)
    label = " ".join(parts)
    if info.gsu:
        label += f" gsu{info.gsu}"
    if "bias" in info.flags:
        label += " +bias"
    info.label = label
    return info


def _split_targs(targs: str) -> list:
    """Split top-level template args, respecting nested <>/()."""
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


def _parse_hip(name: str) -> KernelInfo:
    info = KernelInfo(library="hip")
    m = re.match(r"^([A-Za-z_][\w:]*)(?:<(.*)>)?$", name.strip())
    if not m:
        info.family = name
        info.label = name
        return info
    base = m.group(1).split("::")[-1]
    targs = m.group(2) or ""

    family = base
    if family.endswith("_kernel"):
        family = family[: -len("_kernel")]
    if family.startswith("hip_"):
        family = family[len("hip_") :]
    info.family = family

    args = _split_targs(targs) if targs else []
    nums, dtype = [], ""
    for a in args:
        if a in _CPP_DTYPE:
            dtype = dtype or _CPP_DTYPE[a]
        elif re.fullmatch(r"-?\d+", a):
            nums.append(a)
        elif a.lstrip("(").rstrip(")").lstrip("Li").rstrip("E").isdigit():
            # itanium-demangler may keep literal wrappers like (int)1 or Li1E
            nums.append(re.sub(r"\D", "", a))
    info.dtype = dtype
    if nums:
        info.params["config"] = ",".join(nums)

    label = family
    if dtype:
        label += f" {dtype}"
    if nums:
        label += f" <{','.join(nums)}>"
    info.label = label
    return info


def parse_kernel_name(name: str) -> KernelInfo:
    """Decode ``name`` (a simplified/demangled symbol) into a :class:`KernelInfo`."""
    if not name:
        return KernelInfo(library="unknown", family="", label="")
    if name.startswith("0x"):  # unresolved PC
        return KernelInfo(library="unknown", family="unresolved", label=name)
    if name.startswith("Cijk"):
        return _parse_tensile(name)
    # first-party HIP / generic C++ symbol
    if re.match(r"^[A-Za-z_]", name):
        return _parse_hip(name)
    return KernelInfo(library="other", family=name, label=name)
