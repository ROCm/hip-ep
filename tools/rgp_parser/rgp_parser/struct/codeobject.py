#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""CodeObject chunks: AMDGPU ELF code objects carrying kernel symbols.

Each ``CodeObject`` chunk's data is an AMDGPU ELF. Its ``.symtab`` STT_FUNC symbols
give kernel entry offsets; combined with a load base (see :mod:`coload`) they map a
dispatch PC to a kernel name.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from .rdf import RdfFile


@dataclass
class Symbol:
    value: int  # st_value (offset within the code object)
    size: int  # st_size (0x100 fallback when the ELF reports 0)
    name: str


def parse_functions(elf_bytes: bytes) -> list[tuple[int, int, str]]:
    """Return sorted (value, size, name) for STT_FUNC symbols in one CO ELF."""
    from elftools.elf.elffile import ELFFile

    fns: list[tuple[int, int, str]] = []
    try:
        ef = ELFFile(io.BytesIO(elf_bytes))
        st = ef.get_section_by_name(".symtab") or ef.get_section_by_name(".dynsym")
        if st is None:
            return fns
        for s in st.iter_symbols():
            if s["st_info"]["type"] != "STT_FUNC":
                continue
            name = s.name
            if not name or name.startswith("__hip_cuid"):
                continue
            fns.append((s["st_value"], s["st_size"] or 0x100, name))
    except Exception:
        return fns
    return sorted(fns)


def load_code_objects(rdf: RdfFile) -> list[list[tuple[int, int, str]]]:
    """Per-CodeObject list of STT_FUNC (value, size, name) tuples, in chunk order."""
    return [parse_functions(rdf.data(c)) for c in rdf.by_id("CodeObject")]


def load_code_objects_hashed(
    rdf: RdfFile,
) -> list[tuple[bytes, list[tuple[int, int, str]]]]:
    """Per-CodeObject (16-byte header hash, STT_FUNC tuples).

    The CodeObject chunk header carries the same 16-byte hash as its COLoadEvent
    record (at header offset +8), enabling deterministic hash->base pairing."""
    out = []
    for c in rdf.by_id("CodeObject"):
        hdr = rdf.raw_header(c)
        h = hdr[8:24] if len(hdr) >= 24 else b""
        out.append((h, parse_functions(rdf.data(c))))
    return out
