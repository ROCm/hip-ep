#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Map dispatch PCs to demangled kernel names.

Coverage-based CodeObject<->load-base assignment: treat observed dispatch PCs as
ground truth and, for each code object, pick the load base under which its symbols
actually cover the observed PCs (weighted by dispatch duration). Then demangle and
simplify to ``name<template args>``.
"""

from __future__ import annotations

import bisect
import re
import shutil
import subprocess


def _coverage(base, fns, pcset, pcs, pcw):
    exact = sum(1 for v, _sz, _n in fns if base + v in pcset)
    contain = 0.0
    if exact:
        starts = [base + v for v, _sz, _n in fns]
        for pc in pcs:
            i = bisect.bisect_right(starts, pc) - 1
            if 0 <= i < len(fns):
                v, sz, _n = fns[i]
                if base + v <= pc < base + v + sz:
                    contain += pcw[pc]
    return exact, contain


def assign_bases(co_funcs, bases, pcw):
    """Greedy one-to-one (code_object -> base) assignment maximizing PC coverage."""
    pcs = sorted(pcw)
    pcset = set(pcs)
    pairs = []
    for ci, fns in enumerate(co_funcs):
        for base in bases:
            ex, co = _coverage(base, fns, pcset, pcs, pcw)
            if ex > 0:
                pairs.append((ex, co, ci, base))
    pairs.sort(reverse=True)
    assigned, used_co, used_base = {}, set(), set()
    for _ex, _co, ci, base in pairs:
        if ci in used_co or base in used_base:
            continue
        assigned[ci] = base
        used_co.add(ci)
        used_base.add(base)
    return assigned


def resolve_names(co_funcs, bases, pcw):
    """Return {pc: mangled_name} for as many observed PCs as possible."""
    pcs = sorted(pcw)
    pcset = set(pcs)
    assigned = assign_bases(co_funcs, bases, pcw)

    entry, syms = {}, []
    for ci, base in assigned.items():
        for v, sz, n in co_funcs[ci]:
            syms.append((base + v, base + v + sz, n))
            if base + v in pcset:
                entry.setdefault(base + v, n)
    syms.sort()
    sstarts = [s[0] for s in syms]

    def resolve(pc):
        if pc in entry:
            return entry[pc]
        i = bisect.bisect_right(sstarts, pc) - 1
        if 0 <= i < len(syms) and syms[i][0] <= pc < syms[i][1]:
            return syms[i][2]
        return None

    resolved = {pc: resolve(pc) for pc in pcs if resolve(pc)}
    # fallback: exact-entry across ALL (code_object, base) combinations
    baseset = set(bases)
    for pc in pcs:
        if pc in resolved:
            continue
        for fns in co_funcs:
            hit = next((n for v, _sz, n in fns if (pc - v) in baseset), None)
            if hit:
                resolved[pc] = hit
                break
    return resolved


def resolve_names_exact(co_hash_funcs, hash2base, pcw):
    """Deterministic {pc: mangled_name} via CodeObject-hash <-> COLoadEvent-base pairing.

    Each CodeObject chunk header carries a 16-byte hash that also keys its
    COLoadEvent load base, so no coverage heuristic is needed. Returns
    (resolved, matched, n_code_objects)."""
    pcs = sorted(pcw)
    pcset = set(pcs)
    entry, syms, matched = {}, [], 0
    for h, fns in co_hash_funcs:
        base = hash2base.get(h)
        if base is None:
            continue  # CO not in COLoadEvent (loaded pre-capture)
        matched += 1
        for v, sz, n in fns:
            a = base + v
            syms.append((a, a + sz, n))
            if a in pcset:
                entry.setdefault(a, n)
    syms.sort()
    sstarts = [s[0] for s in syms]

    def resolve(pc):
        if pc in entry:
            return entry[pc]
        i = bisect.bisect_right(sstarts, pc) - 1
        if 0 <= i < len(syms) and syms[i][0] <= pc < syms[i][1]:
            return syms[i][2]
        return None

    resolved = {pc: resolve(pc) for pc in pcs if resolve(pc)}
    return resolved, matched, len(co_hash_funcs)


def _cxxfilt():
    return shutil.which("llvm-cxxfilt") or shutil.which("c++filt")


def demangle_batch(names):
    names = list(names)
    tool = _cxxfilt()
    if tool:
        out = subprocess.run(
            [tool], input="\n".join(names), capture_output=True, text=True
        )
        lines = out.stdout.splitlines()
        return {n: (lines[i] if i < len(lines) else n) for i, n in enumerate(names)}
    try:
        from itanium_demangler import parse as _dm

        def one(n):
            try:
                r = _dm(n)
                return str(r) if r else n
            except Exception:
                return n

        return {n: one(n) for n in names}
    except ImportError:
        return {n: n for n in names}


def simplify(dem: str) -> str:
    s = dem
    if s.startswith("void "):
        s = s[5:]
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
    base = m.group(1).split("::")[-1]
    targs = m.group(2) or ""
    out = (base + targs).strip()
    return out or dem


def name_map(co_funcs, bases, pcw):
    """{pc: simplified demangled name} for all resolvable observed PCs."""
    resolved = resolve_names(co_funcs, bases, pcw)
    dem = demangle_batch(sorted(set(resolved.values())))
    return {pc: simplify(dem[n]) for pc, n in resolved.items()}


def name_map_exact(co_hash_funcs, hash2base, pcw):
    """{pc: simplified demangled name} via deterministic hash<->base pairing.

    Returns (name_map, stats) where stats = (resolved_pcs, total_pcs, co_matched,
    co_total, pct_duration)."""
    resolved, matched, n_co = resolve_names_exact(co_hash_funcs, hash2base, pcw)
    dem = demangle_batch(sorted(set(resolved.values())))
    names = {pc: simplify(dem[n]) for pc, n in resolved.items()}
    rtot = sum(pcw.values()) or 1.0
    rres = sum(pcw.get(pc, 0.0) for pc in resolved)
    stats = (len(resolved), len(pcw), matched, n_co, 100.0 * rres / rtot)
    return names, stats
