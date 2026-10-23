#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Append/64K correctness and paired direct timing; not model throughput."""

from hip_test_utils import Buffer, ROOT, check, device, hip, lib, ptr
from reference import encode, unpack_rows
import ctypes as c
import json
import numpy as np
import os
from reference import pack_indices
import time

const = np.load(ROOT / "reference/codec/constants.npz")
pi, ck, cv = (const[k] for k in ("rotation", "cb_k", "cb_v"))
dp, dk, dv = (device(x) for x in (pi, ck, cv))
stream = ptr()
check(hip.hipStreamCreate(c.byref(stream)))
ring = c.CDLL(
    os.environ.get(
        "TQ_RING_LIBRARY",
        str(ROOT / "install/ring-baseline/lib/libcustom_kernels_gfx1151.so"),
    )
)
ring.hip_gqa_ring_workspace.argtypes = [c.c_int, c.c_int]
ring.hip_gqa_ring_workspace.restype = c.c_size_t
ring.hip_gqa_ring_attention.argtypes = [ptr] * 5 + [c.c_int] * 3 + [ptr, c.c_size_t]
hip.hipEventCreate.argtypes = [c.POINTER(ptr)]
hip.hipEventRecord.argtypes = [ptr, ptr]
hip.hipEventSynchronize.argtypes = [ptr]
hip.hipEventElapsedTime.argtypes = [c.POINTER(c.c_float), ptr, ptr]
hip.hipEventDestroy.argtypes = [ptr]


def timed(fn, reps=30):
    for _ in range(3):
        check(fn())
    check(hip.hipStreamSynchronize(stream))
    a, b = ptr(), ptr()
    check(hip.hipEventCreate(c.byref(a)))
    check(hip.hipEventCreate(c.byref(b)))
    started = time.perf_counter()
    check(hip.hipEventRecord(a, stream))
    for _ in range(reps):
        check(fn())
    check(hip.hipEventRecord(b, stream))
    check(hip.hipEventSynchronize(b))
    elapsed = c.c_float()
    check(hip.hipEventElapsedTime(c.byref(elapsed), a, b))
    check(hip.hipEventDestroy(a))
    check(hip.hipEventDestroy(b))
    host_ms = (time.perf_counter() - started) * 1000 / reps
    # This gfx1151/toolchain reports zero event elapsed time. Keep synchronized
    # wall time as the diagnostic measure and never publish zero as latency.
    assert host_ms > 0
    return host_ms


records = []
rng = np.random.default_rng(77823)
for length in (33, 129, 4096, 8192, 16384, 32768, 65536, 65664):
    # Arbitrary indices cover all codes and are independent of the GPU encoder.
    capacity = length
    kr = pack_indices(
        rng.integers(0, 16, (4, length, 256), dtype=np.uint8),
        rng.uniform(4, 25, (4, length)).astype(np.float32),
    )
    vr = pack_indices(
        rng.integers(0, 16, (4, length, 256), dtype=np.uint8),
        rng.uniform(4, 25, (4, length)).astype(np.float32),
    )
    q = rng.normal(size=(24, 256)).astype(np.float16)
    newk = rng.normal(size=(3, 4, 256)).astype(np.float16)
    newv = rng.normal(size=(3, 4, 256)).astype(np.float16)
    newk[0, 0].fill(0)
    newv[1, 1].fill(np.nextafter(np.float16(0), np.float16(1)))
    newk[2, 2].fill(65504)
    newv[2, 3].fill(-65504)
    bs = [
        device(kr),
        device(vr),
        device(q),
        device(newk),
        device(newv),
        Buffer(24 * 256 * 2),
        Buffer(lib.hip_tq4_decode_workspace(length)),
    ]
    pk, pv, sq, nk, nv, out, work = bs
    # Three separate decode appends finish at the final physically allocated row.
    for i in range(3):
        kp = ptr(nk.ptr.value + i * 4 * 256 * 2)
        vp = ptr(nv.ptr.value + i * 4 * 256 * 2)
        check(
            lib.hip_tq4_encode_append(
                stream,
                kp,
                vp,
                pk.ptr,
                pv.ptr,
                1,
                length - 3 + i,
                capacity,
                dp.ptr,
                dk.ptr,
                dv.ptr,
            )
        )
    check(hip.hipStreamSynchronize(stream))
    ka = pk.get(np.uint8, kr.shape)
    va = pv.get(np.uint8, vr.shape)
    np.testing.assert_array_equal(ka[:, :-3], kr[:, :-3])
    np.testing.assert_array_equal(va[:, :-3], vr[:, :-3])
    for actual, x, cb in ((ka, newk, ck), (va, newv, cv)):
        expected = encode(x.transpose(1, 0, 2).copy(), pi, cb)
        np.testing.assert_array_equal(actual[:, -3:, 4:], expected[..., 4:])
        np.testing.assert_allclose(
            unpack_rows(actual[:, -3:])[1],
            unpack_rows(expected)[1],
            rtol=3e-7,
            atol=2e-6,
        )
    # Use moderate value norms for the attention gate and speed comparison.
    # Extreme encoder values were checked above, independently of FP16 overflow.
    pk.put(kr)
    pv.put(vr)

    def tq():
        return lib.hip_tq4_decode(
            stream,
            sq.ptr,
            pk.ptr,
            pv.ptr,
            out.ptr,
            length,
            capacity,
            0.0625,
            dp.ptr,
            dk.ptr,
            dv.ptr,
            work.ptr,
            work.size,
        )

    check(tq())
    check(hip.hipStreamSynchronize(stream))
    actual = out.get(np.float16, (24, 256)).astype(np.float32)
    # FP64 dense oracle, without partition recurrence, evaluated one KV head
    # at a time to keep temporary CPU memory bounded at maximum context.
    ki, kn = unpack_rows(kr)
    vi, vn = unpack_rows(vr)
    expected = np.empty((24, 256), np.float64)
    densek = np.empty((4, length, 256), np.float16)
    densev = np.empty_like(densek)
    for g in range(4):
        kk = (ck[ki[g]].astype(np.float64) @ pi.astype(np.float64)) * kn[g, :, None]
        vv = (cv[vi[g]].astype(np.float64) @ pi.astype(np.float64)) * vn[g, :, None]
        densek[g] = kk
        densev[g] = vv
        scores = q[g * 6 : g * 6 + 6].astype(np.float64) @ kk.T * 0.0625
        weights = np.exp(scores - scores.max(axis=1, keepdims=True))
        expected[g * 6 : g * 6 + 6] = (weights @ vv) / weights.sum(
            axis=1, keepdims=True
        )
    expected = expected.astype(np.float16).astype(np.float32)
    np.testing.assert_allclose(actual, expected, atol=0.005, rtol=0.005)
    densebs = [
        device(densek),
        device(densev),
        Buffer(max(1, ring.hip_gqa_ring_workspace(1, length))),
    ]
    rk, rv, rw = densebs

    def rd():
        return ring.hip_gqa_ring_attention(
            stream,
            sq.ptr,
            rk.ptr,
            rv.ptr,
            out.ptr,
            1,
            length - 1,
            capacity,
            rw.ptr,
            rw.size,
        )

    pairs = []
    for pair in range(3):
        if pair % 2:
            r = timed(rd)
            t = timed(tq)
        else:
            t = timed(tq)
            r = timed(rd)
        pairs.append({"tq_ms": t, "ring_ms": r})
    record = {
        "length": length,
        "max_abs": float(np.max(np.abs(actual - expected))),
        "relative_l2": float(
            np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        ),
        "pairs": pairs,
        "timing_scope": "attention including rotations, excludes append; concurrent CPU build, diagnostic only",
    }
    records.append(record)
    print(json.dumps(record), flush=True)
    for b in bs + densebs:
        b.close()
for b in (dp, dk, dv):
    b.close()
check(hip.hipStreamDestroy(stream))
(ROOT / "results/kernel-extended.json").write_text(
    json.dumps({"status": "passed", "cases": records}, indent=2) + "\n"
)
