#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Ring round-trip and attribution, with a Python stand-in for the C++ producer.

The producer below follows trace_shim.cpp: reserve head, write the payload,
publish seq = pos + 1. Names are chunked 40 bytes per record.
"""

import mmap
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from hipep_perf import ring, trace
from hipep_perf.collector import Collector


class FakeShim:
    def __init__(self, session: str, capacity: int):
        size = ring.HEADER_SIZE + capacity * ring.RECORD_SIZE
        self.mm = mmap.mmap(-1, size, tagname=ring.mapping_name(session))
        magic, version, rsize, cap = struct.unpack_from("<QIIQ", self.mm, 0)
        assert (magic, version, rsize, cap) == (
            ring.MAGIC,
            ring.VERSION,
            ring.RECORD_SIZE,
            capacity,
        )
        self.cap = capacity
        self.names: dict[str, int] = {}
        struct.pack_into(
            "<I",
            self.mm,
            ring.OFF_FLAGS,
            ring.FLAG_ATTACHED | ring.FLAG_TRACER | ring.FLAG_API,
        )

    def push(self, typ: int, f: list[int], tid: int = 7) -> bool:
        head = struct.unpack_from("<Q", self.mm, ring.OFF_HEAD)[0]
        tail = struct.unpack_from("<Q", self.mm, ring.OFF_TAIL)[0]
        if head - tail >= self.cap:
            d = struct.unpack_from("<Q", self.mm, ring.OFF_DROPPED)[0]
            struct.pack_into("<Q", self.mm, ring.OFF_DROPPED, d + 1)
            return False
        struct.pack_into("<Q", self.mm, ring.OFF_HEAD, head + 1)
        off = ring.HEADER_SIZE + (head % self.cap) * ring.RECORD_SIZE
        struct.pack_into(
            "<HHI6Q", self.mm, off + 8, typ, 0, tid, *(f + [0] * (6 - len(f)))
        )
        struct.pack_into("<Q", self.mm, off, head + 1)
        return True

    def name(self, s: str) -> int:
        if s in self.names:
            return self.names[s]
        nid = len(self.names) + 1
        self._emit(nid, s)
        self.names[s] = nid
        return nid

    def name_api(self, api_id: int, s: str) -> None:
        self._emit(ring.API_NAME_BASE | api_id, s)

    def _emit(self, nid: int, s: str) -> None:
        b = s.encode()
        chunks = max(1, (len(b) + 39) // 40)
        for c in range(chunks):
            part = b[c * 40 : (c + 1) * 40].ljust(40, b"\x00")
            words = list(struct.unpack("<5Q", part))
            self.push(ring.REC_NAME, [nid | (c << 32) | (chunks << 48)] + words)


def test_round_trip_and_attribution(tmp_path):
    col = Collector(tmp_path, session=f"t{os.getpid()}", capacity=1 << 10, poll_ms=0.5)
    shim = FakeShim(col.session, 1 << 10)
    col.start()
    long_label = "hip_ep.op.matmul_nbits m=1,n=2560,k=10240 bits=4 block=32 " + "x" * 30
    g = shim.name("hip_ep.graph.decoder sub=0")
    op = shim.name(long_label)
    kern = shim.name("_Z15gemv_int4_kernelv")
    shim.push(ring.REC_PUSH, [1000, g, 0])
    shim.push(ring.REC_PUSH, [1100, op, 1])
    shim.push(ring.REC_API, [42, 1150, 1180, 300])  # hipLaunchKernel inside op
    shim.push(ring.REC_POP, [1200, 1])
    shim.push(ring.REC_API, [43, 1250, 1260, 300])  # inside graph only
    shim.push(ring.REC_POP, [1300, 0])
    # GPU work completes after the host ranges closed: attribution must use the call.
    shim.push(ring.REC_OP, [42, 5000, 9000, 0xAB, 0x11F0 | (0 << 32), kern])
    shim.push(ring.REC_OP, [43, 9100, 9200, 0xAB, 0x11F0 | (0 << 32), kern])
    shim.push(
        ring.REC_OP, [99, 9300, 9400, 0xAB, 0x11F5 | (1 << 32), 4096]
    )  # copy, no call
    meta = col.stop()
    assert meta["records"] > 0 and meta["dropped"] == 0 and meta["attached"]

    tr = trace.load(tmp_path)
    assert tr.names[op] == long_label
    trace.attribute(tr)
    v = trace.validate(tr)
    assert v["ops"] == 3 and v["begin_before_call"] == 0 and v["ops_attributed"] == 2
    s = trace.summarize(tr)
    assert s["families"]["matmul_nbits"]["gpu_ns"] == 4000
    assert s["graphs"]["hip_ep.graph.decoder"]["gpu_ns"] == 4100
    assert s["unattributed_gpu_ns"] == 100 + 100
    trace.write_report(tr, tmp_path)
    trace.write_chrome(tr, tmp_path / "t.json")
    pf = trace.write_pftrace(tr, tmp_path / "t.pftrace").read_bytes()
    assert (
        pf[:1] == b"\x0a" and len(pf) > 100
    )  # Trace.packet (field 1, length-delimited)


GOLDEN_KERNEL = (
    '"Kind","Agent_Id","Queue_Id","Stream_Id","Thread_Id","Dispatch_Id","Kernel_Id",'
    '"Kernel_Name","Correlation_Id","Start_Timestamp","End_Timestamp","LDS_Block_Size",'
    '"Scratch_Size","VGPR_Count","Accum_VGPR_Count","SGPR_Count","Workgroup_Size_X",'
    '"Workgroup_Size_Y","Workgroup_Size_Z","Grid_Size_X","Grid_Size_Y","Grid_Size_Z"'
)
GOLDEN_API = (
    '"Domain","Function","Process_Id","Thread_Id","Correlation_Id","Start_Timestamp",'
    '"End_Timestamp"'
)
GOLDEN_COPY = (
    '"Kind","Direction","Stream_Id","Source_Agent_Id","Destination_Agent_Id",'
    '"Correlation_Id","Start_Timestamp","End_Timestamp"'
)


def _export(tmp_path):
    col = Collector(tmp_path, session=f"r{os.getpid()}", capacity=1 << 10)
    shim = FakeShim(col.session, 1 << 10)
    kern = shim.name("_Z16synthetic_kernelPfi")
    dev, arch = shim.name("AMD Radeon(TM) 8060S Graphics"), shim.name("gfx1151")
    shim.push(ring.REC_DEVICE, [dev, arch, 20 | 32 << 32, 2048 | 65536 << 32, 2900000])
    mal, launch = 34, 311  # hipMalloc and hipLaunchKernel ids
    shim.name_api(mal, "hipMalloc")
    shim.name_api(launch, "hipLaunchKernel")
    rng = shim.name("hip_ep.op.synthetic")
    shim.push(ring.REC_PUSH, [150, rng, 0])
    shim.push(ring.REC_API, [1, 100, 140, mal])
    shim.push(ring.REC_API, [2, 200, 300, launch])
    shim.push(ring.REC_POP, [310, 0])
    shim.push(ring.REC_OP, [2, 400, 900, 77, 0x11F0, kern])
    shim.push(ring.REC_OP, [3, 1000, 1200, 77, 0x11F4 | (1 << 32), 4096])
    meta = col.stop()
    assert meta["producer_pid"] == 0
    tr = trace.load(tmp_path)
    tr.meta["producer_pid"] = 4242
    return {p.name: p for p in trace.write_rocprof(tr, tmp_path)}


def test_rocprof_export_follows_ktrace_conventions(tmp_path):
    """Headers are the rocprofv3 7.2.1 contract pinned by ktrace's golden
    fixtures; unknown fields are written as ktrace writes them without
    call arguments."""
    out = _export(tmp_path)
    kl = out["trace_kernel_trace.csv"].read_text().splitlines()
    al = out["trace_hip_api_trace.csv"].read_text().splitlines()
    cl = out["trace_memory_copy_trace.csv"].read_text().splitlines()
    ml = out["trace_memory_allocation_trace.csv"].read_text().splitlines()
    assert (kl[0], al[0], cl[0]) == (GOLDEN_KERNEL, GOLDEN_API, GOLDEN_COPY)
    assert kl[1] == (
        '"KERNEL_DISPATCH","Agent 1",77,0,7,1,1,"_Z16synthetic_kernelPfi",2,400,900,'
        + ",".join(["0"] * 11)
    )
    assert cl[1] == (
        '"MEMORY_COPY","MEMORY_COPY_HOST_TO_DEVICE",0,"Agent 0","Agent 1",3,1000,1200'
    )
    assert ml[1] == (
        '"MEMORY_ALLOCATION","MEMORY_ALLOCATION_ALLOCATE","Agent 1",0,'
        '"0x0000000000000000",1,100,140'
    )
    agents = out["trace_agent_info.csv"].read_text().splitlines()
    assert ',"GPU",0,80,0,0,16,64,' in agents[2]
    roctx = out["trace_roctx_trace.csv"].read_text().splitlines()
    assert roctx[1:] == [
        '"ROCTX","RANGE_PUSH",4242,7,0,150,"hip_ep.op.synthetic"',
        '"ROCTX","RANGE_POP",4242,7,0,310,""',
    ]


KTRACE_SRC = os.environ.get("KTRACE_SRC")


@pytest.mark.skipif(not KTRACE_SRC, reason="set KTRACE_SRC to a ktrace checkout")
def test_rocprof_export_passes_rocprofv3_validator(tmp_path):
    sys.path.insert(0, str(Path(KTRACE_SRC) / "tests"))
    import rocprofv3_schema

    _export(tmp_path)
    rocprofv3_schema.validate(
        rocprofv3_schema.OutputPaths.from_directory(tmp_path, "trace")
    )


def test_full_ring_drops_instead_of_blocking(tmp_path):
    col = Collector(tmp_path, session=f"d{os.getpid()}", capacity=16)
    shim = FakeShim(col.session, 16)
    pushed = sum(shim.push(ring.REC_MARK, [i, 0]) for i in range(40))
    assert pushed == 16
    meta = col.stop()
    assert meta["dropped"] == 24 and meta["records"] == 16
    recs = np.fromfile(tmp_path / "trace.records", dtype=ring.RECORD_DTYPE)
    assert list(recs["f"][:, 0]) == list(range(16))
