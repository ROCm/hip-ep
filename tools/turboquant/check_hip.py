#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Frozen direct-HIP encoder and decoder checks."""

from hip_test_utils import Buffer, ROOT, check, device, hip, lib, ptr
from reference import attention, encode, unpack_rows
import ctypes as c
import json
import numpy as np

const = np.load(ROOT / "reference/codec/constants.npz")
pi, ck, cv = (const[k] for k in ("rotation", "cb_k", "cb_v"))
dp, dk, dv = (device(x) for x in (pi, ck, cv))
stream = ptr()
check(hip.hipStreamCreate(c.byref(stream)))
records = []
for length in (1, 2, 15, 16, 17, 31, 32, 33, 127, 128, 129, 1023, 1024, 1025, 4096):
    rng = np.random.default_rng(length)
    k = rng.normal(size=(4, length, 256)).astype(np.float16)
    v = rng.normal(size=(4, length, 256)).astype(np.float16)
    q = rng.normal(size=(24, 256)).astype(np.float16)
    if length == 2:
        q.fill(0)
        v.fill(1)
    if length == 1:
        k.fill(0)
        v.fill(0)
    capacity = length + 7
    bs = [
        device(k.transpose(1, 0, 2).copy()),
        device(v.transpose(1, 0, 2).copy()),
        device(q),
        Buffer(4 * capacity * 132),
        Buffer(4 * capacity * 132),
        Buffer(24 * 256 * 2),
        Buffer(lib.hip_tq4_decode_workspace(length)),
    ]
    sk, sv, sq, pk, pv, out, work = bs
    check(
        lib.hip_tq4_encode_append(
            stream,
            sk.ptr,
            sv.ptr,
            pk.ptr,
            pv.ptr,
            length,
            0,
            capacity,
            dp.ptr,
            dk.ptr,
            dv.ptr,
        )
    )
    check(hip.hipStreamSynchronize(stream))
    kr = pk.get(np.uint8, (4, capacity, 132))
    vr = pv.get(np.uint8, (4, capacity, 132))
    np.testing.assert_array_equal(kr[:, length:], 0xA5)
    np.testing.assert_array_equal(vr[:, length:], 0xA5)
    expected_k = encode(k, pi, ck)
    expected_v = encode(v, pi, cv)
    mismatches = int(
        np.count_nonzero(kr[:, :length, 4:] != expected_k[..., 4:])
        + np.count_nonzero(vr[:, :length, 4:] != expected_v[..., 4:])
    )
    # The 4096-vector Windows case currently has two boundary code-byte differences.
    # Require exact code bytes and use GPU bytes for the independent oracle.
    assert mismatches == 0, (length, mismatches)
    for actual, expected in (
        (kr[:, :length], expected_k),
        (vr[:, :length], expected_v),
    ):
        np.testing.assert_allclose(
            unpack_rows(actual)[1], unpack_rows(expected)[1], rtol=3e-7, atol=2e-6
        )
    check(
        lib.hip_tq4_decode(
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
    )
    check(hip.hipStreamSynchronize(stream))
    actual = out.get(np.float16, (24, 256)).astype(np.float32)
    expected = (
        attention(q, kr[:, :length].copy(), vr[:, :length].copy(), pi, ck, cv, 0.0625)
        .astype(np.float16)
        .astype(np.float32)
    )
    np.testing.assert_allclose(actual, expected, atol=0.005, rtol=0.005)
    assert np.isfinite(actual).all()
    before = out.get(np.uint8, (out.size,))
    assert (
        lib.hip_tq4_decode(
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
            work.size - 1,
        )
        != 0
    )
    np.testing.assert_array_equal(out.get(np.uint8, (out.size,)), before)
    assert (
        lib.hip_tq4_encode_append(
            stream,
            sk.ptr,
            sv.ptr,
            pk.ptr,
            pv.ptr,
            length,
            capacity,
            capacity,
            dp.ptr,
            dk.ptr,
            dv.ptr,
        )
        != 0
    )
    np.testing.assert_array_equal(pk.get(np.uint8, (4, capacity, 132)), kr)
    records.append(
        {
            "length": length,
            "capacity": capacity,
            "max_abs": float(np.abs(actual - expected).max()),
            "code_byte_mismatches": mismatches,
        }
    )
    print(records[-1], flush=True)
    for b in bs:
        b.close()
for b in (dp, dk, dv):
    b.close()
check(hip.hipStreamDestroy(stream))
(ROOT / "results/kernel-checks.json").write_text(
    json.dumps({"status": "passed", "cases": records}, indent=2) + "\n"
)
