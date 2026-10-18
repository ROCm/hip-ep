#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Sample WDDM GPU Process Memory total commitment, dedicated and shared usage."""

import ctypes as c
import os
import threading
import time


class Value(c.Structure):
    _fields_ = [("status", c.c_uint32), ("value", c.c_double)]


class Item(c.Structure):
    _fields_ = [("name", c.c_wchar_p), ("value", Value)]


class GpuMemory:
    def __init__(self, pid=None, adapter=False):
        self.api = c.WinDLL("pdh.dll")
        self.query = c.c_void_p()
        self.counters = {}
        self.api.PdhOpenQueryW.argtypes = [
            c.c_wchar_p,
            c.c_size_t,
            c.POINTER(c.c_void_p),
        ]
        self.api.PdhAddEnglishCounterW.argtypes = [
            c.c_void_p,
            c.c_wchar_p,
            c.c_size_t,
            c.POINTER(c.c_void_p),
        ]
        self.api.PdhCollectQueryData.argtypes = [c.c_void_p]
        self.api.PdhGetFormattedCounterArrayW.argtypes = [
            c.c_void_p,
            c.c_uint32,
            c.POINTER(c.c_uint32),
            c.POINTER(c.c_uint32),
            c.c_void_p,
        ]
        for f in (
            "PdhOpenQueryW",
            "PdhAddEnglishCounterW",
            "PdhCollectQueryData",
            "PdhGetFormattedCounterArrayW",
        ):
            getattr(self.api, f).restype = c.c_uint32
        assert self.api.PdhOpenQueryW(None, 0, c.byref(self.query)) == 0
        for key, name in [
            ("committed", "Total Committed"),
            ("dedicated", "Dedicated Usage"),
            ("shared", "Shared Usage"),
        ]:
            h = c.c_void_p()
            prefix = (
                "GPU Adapter Memory(*)"
                if adapter
                else f"GPU Process Memory(pid_{pid or os.getpid()}_*)"
            )
            rc = self.api.PdhAddEnglishCounterW(
                self.query, rf"\{prefix}\{name}", 0, c.byref(h)
            )
            assert rc == 0, (name, hex(rc))
            self.counters[key] = h
        self.lock = threading.Lock()
        self.phase = "setup"
        self.samples = []
        self.error = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.worker, daemon=True)

    def collect(self):
        with self.lock:
            rc = self.api.PdhCollectQueryData(self.query)
            assert rc == 0, hex(rc)
            result = {"time": time.perf_counter(), "phase": self.phase}
            for key, h in self.counters.items():
                size, count = c.c_uint32(), c.c_uint32()
                rc = self.api.PdhGetFormattedCounterArrayW(
                    h, 0x200, c.byref(size), c.byref(count), None
                )
                assert rc in (0, 0x800007D2), (key, hex(rc))
                if not size.value:
                    result[key] = 0
                    continue
                buf = c.create_string_buffer(size.value)
                rc = self.api.PdhGetFormattedCounterArrayW(
                    h, 0x200, c.byref(size), c.byref(count), buf
                )
                assert rc == 0, hex(rc)
                items = c.cast(buf, c.POINTER(Item))
                assert all(items[i].value.status in (0, 1) for i in range(count.value))
                result[key] = int(sum(items[i].value.value for i in range(count.value)))
            self.samples.append(result)
            return result

    def worker(self):
        while not self.stop_event.is_set():
            try:
                self.collect()
            except BaseException as exc:
                self.error = repr(exc)
                return
            self.stop_event.wait(0.02)

    def start(self):
        self.thread.start()

    def set_phase(self, phase):
        self.collect()
        with self.lock:
            self.phase = phase
        self.collect()

    def peak(self, phase):
        assert self.error is None, self.error
        rows = [r for r in self.samples if r["phase"] == phase]
        assert rows, phase
        best = max(rows, key=lambda x: x["committed"])
        return dict(best, sample_count=len(rows), sampling_target_ms=20)

    def finish(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join()
        assert self.error is None, self.error

    def validate_mapped(self, hip):
        # Confirm that mapped-host payload is already included; never add it twice.
        hip.hipHostMalloc.argtypes = [c.POINTER(c.c_void_p), c.c_size_t, c.c_uint]
        hip.hipHostFree.argtypes = [c.c_void_p]
        before = self.collect()
        p = c.c_void_p()
        size = 64 * 2**20
        assert hip.hipHostMalloc(c.byref(p), size, 3) == 0
        try:
            time.sleep(0.1)
            after = self.collect()
            delta = after["committed"] - before["committed"]
            assert size <= delta <= size + 2**20, (before, after, delta)
        finally:
            assert hip.hipHostFree(p) == 0
        return {"mapped_bytes": size, "wddm_committed_delta": delta, "passed": True}
