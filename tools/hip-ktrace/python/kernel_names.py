#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Turn mangled GPU kernel symbols into readable, still-distinct labels.

Shared by the analysis and export tools. The one hard rule is that shortening
must never merge two different kernels: a collision would silently add unrelated
kernels together in every table downstream.
"""

from __future__ import annotations

import collections
import ctypes
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


class _ComgrData(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint64)]


def _load_comgr():
    """Load bundled AMD COMGR, which exports an Itanium C++ demangler."""
    bundled = Path(__file__).resolve().parent.parent / "bin" / "amd_comgr_3.dll"
    candidates = [str(bundled)]
    explicit = os.environ.get("AMD_COMGR")
    if explicit:
        candidates.append(explicit)
    candidates.append("amd_comgr_3.dll")

    for candidate in candidates:
        try:
            lib = ctypes.CDLL(candidate)
            lib.amd_comgr_create_data.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(_ComgrData),
            ]
            lib.amd_comgr_create_data.restype = ctypes.c_int
            lib.amd_comgr_release_data.argtypes = [_ComgrData]
            lib.amd_comgr_release_data.restype = ctypes.c_int
            lib.amd_comgr_set_data.argtypes = [
                _ComgrData,
                ctypes.c_size_t,
                ctypes.c_char_p,
            ]
            lib.amd_comgr_set_data.restype = ctypes.c_int
            lib.amd_comgr_get_data.argtypes = [
                _ComgrData,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_void_p,
            ]
            lib.amd_comgr_get_data.restype = ctypes.c_int
            lib.amd_comgr_demangle_symbol_name.argtypes = [
                _ComgrData,
                ctypes.POINTER(_ComgrData),
            ]
            lib.amd_comgr_demangle_symbol_name.restype = ctypes.c_int
            return lib
        except (AttributeError, OSError):
            continue
    return None


def _demangle_comgr(names: list[str]) -> dict[str, str] | None:
    lib = _load_comgr()
    if lib is None:
        return None

    result = {}
    for name in names:
        source = _ComgrData()
        output = _ComgrData()
        if lib.amd_comgr_create_data(0x9, ctypes.byref(source)) != 0:
            return None
        try:
            encoded = name.encode()
            if lib.amd_comgr_set_data(source, len(encoded), encoded) != 0:
                return None
            if lib.amd_comgr_demangle_symbol_name(source, ctypes.byref(output)) != 0:
                return None
            size = ctypes.c_size_t()
            if (
                lib.amd_comgr_get_data(output, ctypes.byref(size), None) != 0
                or not size.value
            ):
                return None
            buffer = ctypes.create_string_buffer(size.value)
            if (
                lib.amd_comgr_get_data(
                    output, ctypes.byref(size), ctypes.cast(buffer, ctypes.c_void_p)
                )
                != 0
            ):
                return None
            result[name] = _label(buffer.raw[: size.value].rstrip(b"\0").decode())
        finally:
            if output.handle:
                lib.amd_comgr_release_data(output)
            lib.amd_comgr_release_data(source)
    return result


def _find_cxxfilt() -> str | None:
    """Find llvm-cxxfilt from explicit or toolchain environment settings."""
    bundled = Path(__file__).resolve().parent.parent / "bin" / "llvm-cxxfilt.exe"
    if bundled.is_file():
        return str(bundled)
    explicit = os.environ.get("CXXFILT")
    if explicit and os.path.exists(explicit):
        return explicit
    found = shutil.which("llvm-cxxfilt") or shutil.which("c++filt")
    if found:
        return found
    roots = (
        (
            os.environ.get("LLVM_ROOT") or os.environ.get("LLVM_HOME"),
            (r"bin\llvm-cxxfilt.exe",),
        ),
        (
            os.environ.get("HIP"),
            (r"lib\llvm\bin\llvm-cxxfilt.exe", r"bin\llvm-cxxfilt.exe"),
        ),
    )
    for root, relatives in roots:
        if not root:
            continue
        for relative in relatives:
            candidate = os.path.join(root, relative)
            if os.path.isfile(candidate):
                return candidate
    return None


def _label(demangled: str) -> str:
    """`void (anonymous namespace)::foo<__half, 4>(int*, float)` -> `foo<__half,4>`.

    Template arguments are kept: they are what distinguishes one instantiation
    of a tiled kernel from another, and merging them hides exactly the
    differences worth looking at. The argument list is dropped -- it is long,
    and it never differs between instantiations that share a name.
    """
    s = demangled.replace("(anonymous namespace)::", "")
    depth, cut = 0, len(s)
    for i, ch in enumerate(s):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "(" and depth == 0:
            cut = i
            break
    s = s[:cut]
    # Strip a leading return type, taking care not to cut inside template args.
    depth = 0
    for i, ch in enumerate(s):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == " " and depth == 0:
            s = s[i + 1 :]
            break
    s = re.sub(r",\s+", ",", s).strip()

    # Tensile encodes its whole configuration in the symbol -- 400+ characters
    # of flags. The macro tile and MFMA shape are what identify the variant;
    # the rest is noise. Collisions from this are caught by resolve_names().
    tensile = re.match(r"^(Cijk_\w*?_MT\d+x\d+x\d+_MI\d+x\d+x\d+)_", s)
    return tensile.group(1) if tensile else s


def _fallback(mangled: str) -> str:
    """Used only when no demangler is installed. Reads the Itanium nested-name
    components and takes the last one, so an anonymous namespace is skipped
    rather than mistaken for the kernel."""
    m = re.match(r"^_ZN(.*)$", mangled)
    if not m:
        m = re.match(r"^_Z(\d+)(.*)$", mangled)
        return m.group(2)[: int(m.group(1))] if m else mangled
    rest, last = m.group(1), mangled
    while True:
        n = re.match(r"^(\d+)(.*)$", rest)
        if not n:
            break
        length = int(n.group(1))
        last, rest = n.group(2)[:length], n.group(2)[length:]
    return last


def _distinguish(label: str, raws: list[str]) -> dict[str, str]:
    """Re-label kernels whose shortened names collided.

    Names that shorten to the same thing are near-identical -- Tensile variants
    differing in a couple of tuning flags out of fifty. Appending only the
    tokens that actually differ says what the variants are, where a bare index
    would only say that they are not the same.
    """
    tokens = {raw: set(raw.split("_")) for raw in raws}
    common = set.intersection(*tokens.values())
    out = {}
    for i, raw in enumerate(raws, 1):
        varying = sorted(tokens[raw] - common)
        out[raw] = (
            f"{label}[{','.join(varying)}]" if 0 < len(varying) <= 6 else f"{label}#{i}"
        )
    return out


def resolve_names(mangled: list[str]) -> dict[str, str]:
    """Map every distinct mangled name to a display label, in one demangler
    call. Doing this per row would fork thousands of processes."""
    names = [n for n in dict.fromkeys(mangled) if n]
    if not names:
        return {}
    mapping = _demangle_comgr(names)
    tool = None if mapping is not None else _find_cxxfilt()
    if mapping is None and tool:
        try:
            out = subprocess.run(
                [tool],
                input="\n".join(names),
                capture_output=True,
                text=True,
                timeout=60,
            ).stdout.splitlines()
            if len(out) == len(names):
                mapping = {n: _label(d) for n, d in zip(names, out)}
        except (OSError, subprocess.SubprocessError):
            pass
    if mapping is None:
        print(
            "[warn] amd_comgr and llvm-cxxfilt unavailable or failed; "
            "kernel names only partly resolved",
            file=sys.stderr,
        )
        mapping = {n: _fallback(n) for n in names}

    # Shortening must never merge two kernels. Where it would, disambiguate --
    # a label collision would silently sum unrelated kernels into one row.
    owners = collections.defaultdict(list)
    for raw, label in mapping.items():
        owners[label].append(raw)
    for label, raws in owners.items():
        if len(raws) > 1:
            for raw, extra in _distinguish(label, sorted(raws)).items():
                mapping[raw] = extra
    return mapping
