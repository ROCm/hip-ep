#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Dispatch PC -> kernel name.

Exact path: every CodeObject chunk header carries (at +8) the same 16-byte hash
as its COLoadEvent record, which also gives the GPU-VA load base, so symbols
are placed with no guessing. Fallback for captures without hashes: assign each
code object the load base under which its symbols cover the most observed PCs
(weighted by dispatch duration), greedily one-to-one.

COLoadEvent records are 40 bytes: u64 id, u64 GPU-VA base (+8), 16-byte hash
(+16), u64 host loader base (+32). Not in the libamdrdf spec; verified on real
captures.
"""

from __future__ import annotations

import bisect
import re
import shutil
import struct
import subprocess

from . import elf
from .container import Container

_COLOAD_RECORD = 40


def _cload(c: Container):
    chunks = c.find("COLoadEvent")
    return (c.header(chunks[0]), c.data(chunks[0])) if chunks else (b"", b"")


def hash_bases(c: Container) -> dict[bytes, int]:
    _, data = _cload(c)
    out = {}
    for r in range(len(data) // _COLOAD_RECORD):
        o = r * _COLOAD_RECORD
        out[data[o + 16 : o + 32]] = struct.unpack_from("<Q", data, o + 8)[0]
    return out


def load_bases(c: Container) -> list[int]:
    header, data = _cload(c)
    n = (
        struct.unpack_from("<I", header, 0)[0]
        if len(header) >= 4
        else len(data) // _COLOAD_RECORD
    )
    n = min(n, len(data) // _COLOAD_RECORD)
    seen: dict[int, None] = {}
    for r in range(n):
        seen.setdefault(struct.unpack_from("<Q", data, r * _COLOAD_RECORD + 8)[0], None)
    return list(seen)


def code_objects(c: Container) -> list[tuple[bytes, list[tuple[int, int, str]]]]:
    out = []
    for ch in c.find("CodeObject"):
        h = c.header(ch)
        out.append((h[8:24] if len(h) >= 24 else b"", elf.functions(c.data(ch))))
    return out


class _Resolver:
    def __init__(self, syms: list[tuple[int, int, str]], entry: dict[int, str]):
        self.syms = sorted(syms)
        self.starts = [s[0] for s in self.syms]
        self.entry = entry

    def __call__(self, pc: int) -> str | None:
        if pc in self.entry:
            return self.entry[pc]
        i = bisect.bisect_right(self.starts, pc) - 1
        if 0 <= i < len(self.syms) and self.syms[i][0] <= pc < self.syms[i][1]:
            return self.syms[i][2]
        return None


def resolve_exact(cos, bases: dict[bytes, int], pcw: dict[int, float]):
    pcset = set(pcw)
    entry, syms, matched = {}, [], 0
    for h, fns in cos:
        base = bases.get(h)
        if base is None:
            continue  # loaded before the capture window
        matched += 1
        for v, sz, n in fns:
            a = base + v
            syms.append((a, a + sz, n))
            if a in pcset:
                entry.setdefault(a, n)
    res = _Resolver(syms, entry)
    resolved = {}
    for pc in sorted(pcw):
        n = res(pc)
        if n:
            resolved[pc] = n
    return resolved, matched, len(cos)


def _coverage(base, fns, pcset, pcs, pcw):
    exact = sum(1 for v, _, _ in fns if base + v in pcset)
    contain = 0.0
    if exact:
        starts = [base + v for v, _, _ in fns]
        for pc in pcs:
            i = bisect.bisect_right(starts, pc) - 1
            if 0 <= i < len(fns):
                v, sz, _ = fns[i]
                if base + v <= pc < base + v + sz:
                    contain += pcw[pc]
    return exact, contain


def resolve_coverage(
    funcs: list[list[tuple[int, int, str]]], bases: list[int], pcw: dict[int, float]
):
    pcs = sorted(pcw)
    pcset = set(pcs)
    pairs = []
    for ci, fns in enumerate(funcs):
        for base in bases:
            ex, co = _coverage(base, fns, pcset, pcs, pcw)
            if ex > 0:
                pairs.append((ex, co, ci, base))
    pairs.sort(reverse=True)
    assigned, used_co, used_base = {}, set(), set()
    for _, _, ci, base in pairs:
        if ci in used_co or base in used_base:
            continue
        assigned[ci] = base
        used_co.add(ci)
        used_base.add(base)
    entry, syms = {}, []
    for ci, base in assigned.items():
        for v, sz, n in funcs[ci]:
            syms.append((base + v, base + v + sz, n))
            if base + v in pcset:
                entry.setdefault(base + v, n)
    res = _Resolver(syms, entry)
    resolved = {pc: res(pc) for pc in pcs if res(pc)}
    baseset = set(bases)
    for pc in pcs:
        if pc in resolved:
            continue
        for fns in funcs:
            hit = next((n for v, _, n in fns if (pc - v) in baseset), None)
            if hit:
                resolved[pc] = hit
                break
    return resolved


def demangle(names: list[str]) -> dict[str, str]:
    """llvm-cxxfilt / c++filt if on PATH, else itanium_demangler, else unchanged."""
    names = list(names)
    tool = shutil.which("llvm-cxxfilt") or shutil.which("c++filt")
    if tool:
        out = subprocess.run(
            [tool], input="\n".join(names), capture_output=True, text=True, check=False
        )
        lines = out.stdout.splitlines()
        return {n: (lines[i] if i < len(lines) else n) for i, n in enumerate(names)}
    try:
        from itanium_demangler import parse
    except ImportError:
        return {n: n for n in names}

    def one(n: str) -> str:
        try:
            r = parse(n)
            return str(r) if r else n
        except Exception:  # noqa: BLE001 - the demangler raises on odd symbols
            return n

    return {n: one(n) for n in names}


def simplify(dem: str) -> str:
    """'void ns::foo<int, 2>(float*)' -> 'foo<int, 2>'."""
    s = dem.removeprefix("void ")
    s = s.replace("(anonymous namespace)::", "")
    depth = 0
    for i, ch in enumerate(s):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "(" and depth == 0:
            s = s[:i]
            break
    m = re.match(r"(.*?)(<.*>)?$", s)
    out = (m.group(1).split("::")[-1] + (m.group(2) or "")).strip()
    return out or dem


def name_map(c: Container, pcw: dict[int, float]) -> tuple[dict[int, str], str]:
    """({pc: simplified name}, note describing how names were resolved)."""
    if not pcw:
        return {}, ""
    bases = hash_bases(c)
    if bases:
        resolved, matched, n_co = resolve_exact(code_objects(c), bases, pcw)
        if resolved:
            dem = demangle(sorted(set(resolved.values())))
            names = {pc: simplify(dem[n]) for pc, n in resolved.items()}
            total = sum(pcw.values()) or 1.0
            pct = 100.0 * sum(pcw.get(pc, 0.0) for pc in resolved) / total
            return names, (
                f"exact CO-hash pairing: {matched}/{n_co} code objects, "
                f"{len(resolved)}/{len(pcw)} PCs, {pct:.0f}% of dispatch-duration"
            )
    funcs = [fns for _, fns in code_objects(c)]
    resolved = resolve_coverage(funcs, load_bases(c), pcw)
    dem = demangle(sorted(set(resolved.values())))
    return (
        {pc: simplify(dem[n]) for pc, n in resolved.items()},
        "coverage-based CO<->base assignment (no COLoadEvent hashes)",
    )
