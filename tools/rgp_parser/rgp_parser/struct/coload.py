#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""COLoadEvent chunk: code-object load records (GPU-VA load bases).

Layout reverse-engineered and verified on real captures (NOT in the libamdrdf
spec, so this is the least stable parser): a chunk-header record count followed by
N x 40-byte records::

    [+0  u64 id = 0x00C40000]
    [+8  u64 GPU_VA_base]        <- SQTT dispatch PCs are GPU virtual addresses
    [+16 16B hash]
    [+32 u64 host_loader_base]

Only the GPU-VA base is needed to place a code object's symbols in the PC space.
"""

from __future__ import annotations

import struct

from .rdf import RdfFile

_RECORD_SIZE = 40


def parse_bases(header: bytes, data: bytes) -> list[int]:
    """Distinct GPU-VA load bases, in record order (dedup preserves first-seen)."""
    n = (
        struct.unpack_from("<I", header, 0)[0]
        if len(header) >= 4
        else len(data) // _RECORD_SIZE
    )
    n = min(n, len(data) // _RECORD_SIZE)
    bases = [struct.unpack_from("<Q", data, r * _RECORD_SIZE + 8)[0] for r in range(n)]
    seen, out = set(), []
    for b in bases:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def load_bases(rdf: RdfFile) -> list[int]:
    chunks = rdf.by_id("COLoadEvent")
    if not chunks:
        return []
    c = chunks[0]
    return parse_bases(rdf.raw_header(c), rdf.data(c))


def parse_hash_bases(data: bytes) -> dict[bytes, int]:
    """{16-byte code-object hash -> GPU-VA load base} from the COLoadEvent records.

    Enables deterministic CodeObject<->base pairing (each CodeObject chunk header
    carries the same 16-byte hash), avoiding the coverage/greedy heuristic."""
    out: dict[bytes, int] = {}
    for r in range(len(data) // _RECORD_SIZE):
        o = r * _RECORD_SIZE
        base = struct.unpack_from("<Q", data, o + 8)[0]
        out[data[o + 16 : o + 32]] = base
    return out


def load_hash_bases(rdf: RdfFile) -> dict[bytes, int]:
    chunks = rdf.by_id("COLoadEvent")
    if not chunks:
        return {}
    return parse_hash_bases(rdf.data(chunks[0]))
