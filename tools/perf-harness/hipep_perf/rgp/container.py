#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""RDF ("AMD_RDF ") chunk container.

A 32-byte file header (magic, version, reserved, index offset, index size), the
chunk blobs, and a 64-byte-per-entry index (GPUOpen-Drivers/libamdrdf,
docs/specification.md). An entry is identifier[16], compression (u8) + 3
reserved bytes, version (u32 at +20), then header offset/size, data
offset/size and uncompressed size (five i64 at +24).
"""

from __future__ import annotations

import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"AMD_RDF "
_HEADER = struct.Struct("<8sIIqq")
_ENTRY = 64


@dataclass(frozen=True)
class Chunk:
    identifier: str
    version: int
    compression: int  # 0 raw, 1 zstd
    header_off: int
    header_size: int
    data_off: int
    data_size: int
    uncompressed_size: int


def _zstd(raw: bytes, size: int) -> bytes:
    try:
        from compression import zstd  # Python 3.14+

        return zstd.decompress(raw)
    except ImportError:
        import zstandard

        return zstandard.ZstdDecompressor().decompress(raw, max_output_size=size or 0)


class Container:
    def __init__(self, blob: bytes):
        magic, self.version, _, index_off, index_size = _HEADER.unpack_from(blob, 0)
        if magic != MAGIC:
            raise ValueError(f"not an RDF capture (magic {magic!r})")
        self.blob = blob
        self.chunks: list[Chunk] = []
        for i in range(index_size // _ENTRY):
            b = index_off + i * _ENTRY
            ident = blob[b : b + 16].split(b"\x00", 1)[0].decode("utf-8", "replace")
            (version,) = struct.unpack_from("<I", blob, b + 20)
            ho, hs, do, ds, us = struct.unpack_from("<qqqqq", blob, b + 24)
            self.chunks.append(Chunk(ident, version, blob[b + 16], ho, hs, do, ds, us))

    @classmethod
    def open(cls, path: str | Path) -> Container:
        return cls(Path(path).read_bytes())

    def header(self, c: Chunk) -> bytes:
        return self.blob[c.header_off : c.header_off + c.header_size]

    def raw(self, c: Chunk) -> bytes:
        return self.blob[c.data_off : c.data_off + c.data_size]

    def data(self, c: Chunk) -> bytes:
        raw = self.raw(c)
        if c.compression == 0:
            return raw
        if c.compression == 1:
            return _zstd(raw, c.uncompressed_size)
        raise ValueError(f"unknown chunk compression {c.compression}")

    def find(self, identifier: str) -> list[Chunk]:
        return [c for c in self.chunks if c.identifier == identifier]

    def counts(self) -> dict[str, int]:
        return dict(Counter(c.identifier for c in self.chunks))
