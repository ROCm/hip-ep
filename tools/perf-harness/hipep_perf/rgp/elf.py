#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Function symbols from an AMDGPU code object (ELF64, little endian)."""

from __future__ import annotations

import struct

SHT_SYMTAB = 2
SHT_DYNSYM = 11
STT_FUNC = 2
_DEFAULT_SIZE = 0x100  # some code objects report st_size 0 for kernels


def functions(elf: bytes) -> list[tuple[int, int, str]]:
    """Sorted (st_value, st_size, name) of STT_FUNC symbols; [] if unreadable.

    .symtab is preferred over .dynsym; HIP's __hip_cuid markers are dropped.
    """
    try:
        if elf[:4] != b"\x7fELF" or elf[4] != 2 or elf[5] != 1:
            return []
        (shoff,) = struct.unpack_from("<Q", elf, 0x28)
        shentsize, shnum, shstrndx = struct.unpack_from("<HHH", elf, 0x3A)
        sections = []
        for i in range(shnum):
            o = shoff + i * shentsize
            name, typ, _flags, _addr, off, size, link, _info, _align, entsize = (
                struct.unpack_from("<IIQQQQIIQQ", elf, o)
            )
            sections.append((name, typ, off, size, link, entsize))
        shstr_off = sections[shstrndx][2]

        def sname(off: int) -> str:
            end = elf.index(b"\x00", shstr_off + off)
            return elf[shstr_off + off : end].decode("utf-8", "replace")

        table = next(
            (s for s in sections if s[1] == SHT_SYMTAB and sname(s[0]) == ".symtab"),
            None,
        )
        if table is None:
            table = next(
                (
                    s
                    for s in sections
                    if s[1] == SHT_DYNSYM and sname(s[0]) == ".dynsym"
                ),
                None,
            )
        if table is None:
            return []
        _, _, off, size, link, entsize = table
        str_off = sections[link][2]
        out = []
        for k in range(size // (entsize or 24)):
            st_name, st_info, _other, _shndx, value, sym_size = struct.unpack_from(
                "<IBBHQQ", elf, off + k * (entsize or 24)
            )
            if st_info & 0xF != STT_FUNC:
                continue
            end = elf.index(b"\x00", str_off + st_name)
            name = elf[str_off + st_name : end].decode("utf-8", "replace")
            if not name or name.startswith("__hip_cuid"):
                continue
            out.append((value, sym_size or _DEFAULT_SIZE, name))
        return sorted(out)
    except (struct.error, IndexError, ValueError):
        return []
