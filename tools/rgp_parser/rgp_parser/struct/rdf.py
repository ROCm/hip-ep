#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""AMD_RDF (Radeon Data File) container parser.

A ``.rgp`` capture is an RDF container: a 32-byte header, chunk data blobs, and a
64-byte-per-entry chunk index. Layout verified against the libamdrdf spec
(GPUOpen-Drivers/libamdrdf, docs/specification.md).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

_HEADER = struct.Struct("<8sIIqq")  # magic, version, reserved, indexOff, indexSize
_MAGIC = b"AMD_RDF "
_ENTRY_SIZE = 64


@dataclass
class Chunk:
    identifier: str
    version: int
    compression: int  # 0 = raw, 1 = Zstd
    header_off: int
    header_size: int
    data_off: int
    data_size: int
    uncompressed_size: int


class RdfFile:
    """Parsed RDF container. Holds the whole blob in memory and lazily returns
    chunk header/data (inflating Zstd on demand)."""

    def __init__(self, blob: bytes):
        magic, self.version, _reserved, ioff, isize = _HEADER.unpack_from(blob, 0)
        if magic != _MAGIC:
            raise ValueError(f"not an RDF file: bad magic {magic!r}")
        self._blob = blob
        self.chunks: list[Chunk] = []
        for i in range(isize // _ENTRY_SIZE):
            base = ioff + i * _ENTRY_SIZE
            cid = struct.unpack_from("<16s", blob, base)[0].split(b"\x00", 1)[0]
            compression = blob[base + 16]
            cver = struct.unpack_from("<I", blob, base + 20)[0]
            hoff, hsize, doff, dsize, usize = struct.unpack_from(
                "<qqqqq", blob, base + 24
            )
            self.chunks.append(
                Chunk(
                    identifier=cid.decode("utf-8", "replace"),
                    version=cver,
                    compression=compression,
                    header_off=hoff,
                    header_size=hsize,
                    data_off=doff,
                    data_size=dsize,
                    uncompressed_size=usize,
                )
            )

    @classmethod
    def from_path(cls, path: str) -> "RdfFile":
        with open(path, "rb") as f:
            return cls(f.read())

    def raw_header(self, chunk: Chunk) -> bytes:
        return self._blob[chunk.header_off : chunk.header_off + chunk.header_size]

    def raw_data(self, chunk: Chunk) -> bytes:
        return self._blob[chunk.data_off : chunk.data_off + chunk.data_size]

    def data(self, chunk: Chunk) -> bytes:
        """Chunk data, inflated if Zstd-compressed."""
        raw = self.raw_data(chunk)
        if chunk.compression == 0:
            return raw
        if chunk.compression == 1:
            try:
                import zstandard as zstd
            except ImportError as e:
                raise RuntimeError(
                    "chunk is Zstd-compressed; `pip install zstandard`"
                ) from e
            return zstd.ZstdDecompressor().decompress(
                raw, max_output_size=chunk.uncompressed_size or 0
            )
        raise ValueError(f"unknown compression {chunk.compression}")

    def by_id(self, identifier: str) -> list[Chunk]:
        return [c for c in self.chunks if c.identifier == identifier]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.chunks:
            out[c.identifier] = out.get(c.identifier, 0) + 1
        return out
