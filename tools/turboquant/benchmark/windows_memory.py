#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Exact process HIP allocation payload peaks, not WDDM residency or total RAM."""

import ctypes as c


class Snapshot(c.Structure):
    _fields_ = [
        (k, c.c_uint64)
        for k in (
            "device",
            "host",
            "peak",
            "allocations",
            "frees",
            "unknown_frees",
            "failures",
            "live_count",
        )
    ]


class Memory:
    def __init__(self, root, hip, ep):
        self.counter = c.CDLL(str(root / "build/windows_memory_counter.dll"))
        self.counter.install.argtypes = [c.c_void_p, c.c_void_p]
        self.counter.snapshot.argtypes = [c.POINTER(Snapshot), c.c_int]
        self.counter.list_allocations.argtypes = [c.POINTER(c.c_uint64), c.c_size_t]
        self.counter.list_allocations.restype = c.c_size_t
        self.hooked = self.counter.install(ep._handle, hip._handle)
        assert self.hooked == 4, (
            f"Expected all four allocation imports, found {self.hooked}"
        )
        self.counter.tracked_alloc.argtypes = [c.POINTER(c.c_void_p), c.c_size_t]
        self.counter.tracked_host.argtypes = [
            c.POINTER(c.c_void_p),
            c.c_size_t,
            c.c_uint,
        ]
        self.counter.tracked_free.argtypes = [c.c_void_p]
        self.counter.tracked_host_free.argtypes = [c.c_void_p]
        a, b = c.c_void_p(), c.c_void_p()
        assert self.counter.tracked_alloc(c.byref(a), 4096) == 0
        assert self.counter.tracked_host(c.byref(b), 8192, 3) == 0
        s = self.read()
        assert s["device"] == 4096 and s["host"] == 8192 and s["peak"] == 12288, s
        assert self.counter.tracked_free(a) == 0
        assert self.counter.tracked_host_free(b) == 0
        s = self.read(True)
        assert s["device"] == s["host"] == s["unknown_frees"] == 0, s

    def read(self, reset=False):
        s = Snapshot()
        self.counter.snapshot(c.byref(s), int(reset))
        return {k: int(getattr(s, k)) for k, _ in Snapshot._fields_}

    def allocations(self):
        count = self.counter.list_allocations(None, 0)
        out = (c.c_uint64 * (3 * count))()
        assert self.counter.list_allocations(out, count) == count
        return [
            {
                "address": int(out[3 * i]),
                "bytes": int(out[3 * i + 1]),
                "host": bool(out[3 * i + 2]),
            }
            for i in range(count)
        ]

    def cache(self, capacity, quantized):
        size = 4 * capacity * (132 if quantized else 512)
        # Reproduce the EP's fixed size classes for the contexts in this study.
        # Beyond 16 MiB, fresh shared-cache requests are allocated exactly.
        classes = {128, 256, 512, 1024, 2**20, 4 * 2**20, 16 * 2**20}
        for low, high, steps in ((10, 20, 4), (20, 22, 16), (22, 24, 32)):
            for exponent in range(low, high):
                base = 2**exponent
                classes.update(base + base * k // steps for k in range(steps))
        allocated = next((x for x in sorted(classes) if x >= size), size)
        selected = [
            a for a in self.allocations() if a["bytes"] == allocated and a["host"]
        ]
        key = (capacity, quantized)
        if not hasattr(self, "cache_allocations"):
            self.cache_allocations = {}
        if key not in self.cache_allocations:
            # Identify the 32 buffers before any prefill transient exists.
            assert len(selected) == 32, (size, allocated, len(selected))
            self.cache_allocations[key] = {a["address"] for a in selected}
        else:
            # Small transient buffers may share this size class after warmup.
            # Preserve the verified cache budget; charge extra buffers to Other.
            assert self.cache_allocations[key] <= {a["address"] for a in selected}
        return {
            "buffers": 32,
            "bytes_per_buffer": allocated,
            "total_bytes": 32 * allocated,
            "payload_bytes": 32 * size,
            "allocator_padding_bytes": 32 * (allocated - size),
            "verification": "32 mapped-host allocations verified before first prefill and retained; additional same-size transient allocations excluded from KV",
            "same_size_live_buffers": len(selected),
        }
