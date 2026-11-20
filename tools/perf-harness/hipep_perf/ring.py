#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Shared-memory ring between the trace shim and the collector.

Mirrors the layout in lib/Runtime/trace_shim.cpp; both sides check
MAGIC/VERSION/RECORD_SIZE. The collector creates the named mapping before the
target starts, so the shim only attaches and committed records outlive the
target however it exits.

Producers reserve a slot by advancing ``head`` (CAS), fill it, then publish
``seq = position + 1``. The consumer reads slots in order while ``seq`` matches
and publishes ``tail``. x86-64 keeps stores in order and aligned 8-byte loads
atomic, which is what makes reading seq-then-payload from Python safe.
"""

from __future__ import annotations

import mmap
import struct

import numpy as np

MAGIC = 0x3152544550454950  # "PIEPETR1"
VERSION = 1
RECORD_SIZE = 64
HEADER_SIZE = 4096

OFF_MAGIC = 0
OFF_VERSION = 8
OFF_RECORD_SIZE = 12
OFF_CAPACITY = 16
OFF_QPC_FREQ = 24
OFF_PID = 32
OFF_FLAGS = 36
OFF_API_COMPARED = 40
OFF_API_MISMATCHES = 44
OFF_API_DETAIL = 48
OFF_HEAD = 256
OFF_TAIL = 320
OFF_DROPPED = 384

FLAG_ATTACHED = 1 << 0
FLAG_TRACER = 1 << 1
FLAG_API = 1 << 2
FLAG_DETACHED = 1 << 3

REC_OP = 1
REC_API = 2
REC_PUSH = 3
REC_POP = 4
REC_MARK = 5
REC_NAME = 6
REC_FENCE = 7
REC_DEVICE = 8

API_NAME_BASE = 0x80000000

RECORD_DTYPE = np.dtype(
    [
        ("seq", "<u8"),
        ("type", "<u2"),
        ("flags", "<u2"),
        ("tid", "<u4"),
        ("f", "<u8", (6,)),
    ]
)
assert RECORD_DTYPE.itemsize == RECORD_SIZE


def mapping_name(session: str) -> str:
    return f"Local\\hipep_trace_{session}"


class Ring:
    """Collector-side view of the ring (creates the named mapping)."""

    def __init__(self, session: str, capacity: int = 1 << 20):
        if capacity & (capacity - 1):
            raise ValueError("capacity must be a power of two")
        self.session = session
        self.capacity = capacity
        size = HEADER_SIZE + capacity * RECORD_SIZE
        self.mm = mmap.mmap(-1, size, tagname=mapping_name(session))
        if self._u64(OFF_MAGIC) not in (0, MAGIC):
            raise RuntimeError(f"mapping {mapping_name(session)} is already in use")
        self.mm[:HEADER_SIZE] = b"\x00" * HEADER_SIZE
        struct.pack_into("<QIIQ", self.mm, 0, MAGIC, VERSION, RECORD_SIZE, capacity)
        self.records = np.frombuffer(
            self.mm, dtype=RECORD_DTYPE, count=capacity, offset=HEADER_SIZE
        )
        # head and tail go through single aligned 8-byte accesses: struct
        # writes byte by byte, and a producer that reads a torn tail sees a
        # ring that looks full once positions pass the capacity.
        self._head = np.frombuffer(self.mm, dtype="<u8", count=1, offset=OFF_HEAD)
        self._tail = np.frombuffer(self.mm, dtype="<u8", count=1, offset=OFF_TAIL)
        self.tail = 0

    def _u64(self, off: int) -> int:
        return struct.unpack_from("<Q", self.mm, off)[0]

    def _u32(self, off: int) -> int:
        return struct.unpack_from("<I", self.mm, off)[0]

    @property
    def head(self) -> int:
        return int(self._head[0])

    @property
    def dropped(self) -> int:
        return self._u64(OFF_DROPPED)

    @property
    def flags(self) -> int:
        return self._u32(OFF_FLAGS)

    def header(self) -> dict:
        detail = bytes(self.mm[OFF_API_DETAIL : OFF_API_DETAIL + 128]).split(
            b"\x00", 1
        )[0]
        return {
            "qpc_frequency": self._u64(OFF_QPC_FREQ),
            "producer_pid": self._u32(OFF_PID),
            "flags": self.flags,
            "attached": bool(self.flags & FLAG_ATTACHED),
            "tracer": bool(self.flags & FLAG_TRACER),
            "api": bool(self.flags & FLAG_API),
            "detached": bool(self.flags & FLAG_DETACHED),
            "api_compared": self._u32(OFF_API_COMPARED),
            "api_mismatches": self._u32(OFF_API_MISMATCHES),
            "api_detail": detail.decode("utf-8", "replace"),
            "dropped": self.dropped,
            "head": self.head,
        }

    def drain(self, limit: int = 1 << 16) -> np.ndarray:
        """Copy out committed records in order and release their slots."""
        mask = self.capacity - 1
        out = []
        taken = 0
        while taken < limit:
            head = self.head
            if self.tail >= head:
                break
            n = min(head - self.tail, limit - taken)
            start = self.tail & mask
            n = min(n, self.capacity - start)
            block = self.records[start : start + n]
            want = np.arange(self.tail + 1, self.tail + n + 1, dtype=np.uint64)
            ready = block["seq"] == want
            k = n if ready.all() else int(np.argmin(ready))
            if k == 0:
                break
            out.append(block[:k].copy())
            self.tail += k
            taken += k
            self._tail[0] = self.tail
        if not out:
            return np.empty(0, dtype=RECORD_DTYPE)
        return np.concatenate(out)

    def close(self) -> None:
        self.records = self._head = self._tail = None
        self.mm.close()


def decode_names(records: np.ndarray) -> dict[int, str]:
    """Reassemble kRecName chunks: id -> string."""
    names: dict[int, dict[int, bytes]] = {}
    totals: dict[int, int] = {}
    sel = records[records["type"] == REC_NAME]
    for r in sel:
        f0 = int(r["f"][0])
        nid, chunk, total = f0 & 0xFFFFFFFF, (f0 >> 32) & 0xFFFF, (f0 >> 48) & 0xFFFF
        names.setdefault(nid, {})[chunk] = r["f"][1:].tobytes()
        totals[nid] = total
    out = {}
    for nid, parts in names.items():
        if len(parts) != totals[nid]:
            continue
        out[nid] = (
            b"".join(parts[i] for i in range(totals[nid]))
            .split(b"\x00", 1)[0]
            .decode("utf-8", "replace")
        )
    return out
