#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Append-prefill GPU gates against independent FP64 mixed-cache attention."""

import ctypes as c
import json
import os
from pathlib import Path
import time

import numpy as np
from hip_test_utils import Buffer, ROOT, check, device, hip, lib, ptr
from reference import encode, pack_indices, unpack_rows

lib.hip_tq4_prefill_workspace.argtypes = [c.c_int]
lib.hip_tq4_prefill_workspace.restype = c.c_size_t
lib.hip_tq4_prefill.argtypes = [ptr] * 7 + [c.c_int] * 3 + [ptr] * 4 + [c.c_size_t]


def mixed_attention(q, k, v, pk, pv, pi, ck, cv):
    """FP64 dense softmax; no GPU tile recurrence or rounded rotated queries."""
    ki, kn = unpack_rows(pk)
    vi, vn = unpack_rows(pv)
    rotation = pi.astype(np.float64)
    output = np.empty(q.shape, np.float64)
    for h in range(24):
        g = h // 6
        qr = q[:, h].astype(np.float64) @ rotation.T
        # Rotated-frame algebra avoids an O(history * D^2) reconstruction.
        history_scores = qr @ ck[ki[g]].astype(np.float64).T
        history_scores *= kn[g][None] * 0.0625
        tail_scores = q[:, h].astype(np.float64) @ k[:, g].astype(np.float64).T * 0.0625
        tail_scores[np.triu_indices(len(q), 1)] = -np.inf
        scores = np.concatenate((history_scores, tail_scores), axis=1)
        weights = np.exp(scores - scores.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        old = (weights[:, : pk.shape[1]] * vn[g]) @ cv[vi[g]].astype(np.float64)
        output[:, h] = old @ rotation + weights[:, pk.shape[1] :] @ v[:, g].astype(
            np.float64
        )
    return output.astype(np.float16).astype(np.float32)


def main():
    const = np.load(ROOT / "reference/codec/constants.npz")
    pi, ck, cv = (const[k] for k in ("rotation", "cb_k", "cb_v"))
    dp, dk, dv = (device(x) for x in (pi, ck, cv))
    stream = ptr()
    check(hip.hipStreamCreate(c.byref(stream)))
    rng = np.random.default_rng(18327)
    records = []
    # Keep the existing strict CPU encoder gate in check_hip.py. Here the
    # append regression must match the frozen PR encoder byte for byte.
    baseline_path = os.environ.get(
        "TQ_ENCODER_BASELINE_LIBRARY",
        str(
            ROOT
            / (
                "install/turboquant/bin/custom_kernels_gfx1151.dll"
                if os.name == "nt"
                else "install/turboquant/lib/libcustom_kernels_gfx1151.so"
            )
        ),
    )
    baseline = c.CDLL(baseline_path)
    baseline.hip_tq4_encode_append.argtypes = lib.hip_tq4_encode_append.argtypes
    cases = [
        (1, 2),
        (15, 17),
        (32, 33),
        (129, 129),
        (4096, 257),
        (131072, 2),
        (262142, 2),
    ]
    assert lib.hip_tq4_prefill_workspace(0) == 0
    assert lib.hip_tq4_prefill_workspace(131073) == 0
    assert (
        lib.hip_tq4_prefill_workspace(128)
        == lib.hip_tq4_prefill_workspace(131072)
        == 7888896
    )
    for past, count in cases:
        capacity = past + count
        kr = pack_indices(
            rng.integers(0, 16, (4, capacity, 256), dtype=np.uint8),
            rng.uniform(4, 25, (4, capacity)).astype(np.float32),
        )
        vr = pack_indices(
            rng.integers(0, 16, (4, capacity, 256), dtype=np.uint8),
            rng.uniform(4, 25, (4, capacity)).astype(np.float32),
        )
        # Zero-norm vectors must contribute exactly zero even for arbitrary codes.
        kr[:, 0, :4] = 0
        vr[:, 0, :4] = 0
        q = rng.normal(size=(count, 24, 256)).astype(np.float16)
        k = rng.normal(size=(count, 4, 256)).astype(np.float16)
        v = rng.normal(size=(count, 4, 256)).astype(np.float16)
        q[0] = 0  # uniform probabilities expose incorrect prefix/suffix weighting
        k[0, 0] = 0
        if past == 15:
            q[1] = 65504  # Rotation must not overflow FP16 staging.
        bs = [device(x) for x in (q, k, v, kr, vr)]
        bs += [Buffer(q.nbytes), Buffer(lib.hip_tq4_prefill_workspace(count))]
        dq, nk, nv, pk, pv, out, work = bs

        def launch(n=count, p=past, cap=capacity, size=work.size):
            return lib.hip_tq4_prefill(
                stream,
                dq.ptr,
                nk.ptr,
                nv.ptr,
                pk.ptr,
                pv.ptr,
                out.ptr,
                n,
                p,
                cap,
                dp.ptr,
                dk.ptr,
                dv.ptr,
                work.ptr,
                size,
            )

        for args in (
            {"size": work.size - 1},
            {"p": 0},
            {"cap": capacity - 1},
            {"n": 0},
        ):
            assert launch(**args) != 0
        check(launch())
        check(hip.hipStreamSynchronize(stream))
        actual = out.get(np.float16, q.shape).astype(np.float32)
        expected = mixed_attention(q, k, v, kr[:, :past], vr[:, :past], pi, ck, cv)
        np.testing.assert_allclose(actual, expected, atol=0.005, rtol=0.005)
        assert np.isfinite(actual).all()
        np.testing.assert_array_equal(pk.get(np.uint8, kr.shape), kr)
        np.testing.assert_array_equal(pv.get(np.uint8, vr.shape), vr)
        started = time.perf_counter()
        for _ in range(3):
            check(launch())
        check(hip.hipStreamSynchronize(stream))
        ms = (time.perf_counter() - started) * 1000 / 3
        check(
            lib.hip_tq4_encode_append(
                stream,
                nk.ptr,
                nv.ptr,
                pk.ptr,
                pv.ptr,
                count,
                past,
                capacity,
                dp.ptr,
                dk.ptr,
                dv.ptr,
            )
        )
        check(hip.hipStreamSynchronize(stream))
        fresh_k, fresh_v = Buffer(count * 4 * 132), Buffer(count * 4 * 132)
        check(
            baseline.hip_tq4_encode_append(
                stream,
                nk.ptr,
                nv.ptr,
                fresh_k.ptr,
                fresh_v.ptr,
                count,
                0,
                count,
                dp.ptr,
                dk.ptr,
                dv.ptr,
            )
        )
        check(hip.hipStreamSynchronize(stream))
        cpu_code_mismatches = 0
        for buf, original, src, cb, fresh in (
            (pk, kr, k, ck, fresh_k),
            (pv, vr, v, cv, fresh_v),
        ):
            packed = buf.get(np.uint8, original.shape)
            np.testing.assert_array_equal(packed[:, :past], original[:, :past])
            ref = encode(src.transpose(1, 0, 2).copy(), pi, cb)
            np.testing.assert_array_equal(
                packed[:, past:], fresh.get(np.uint8, (4, count, 132))
            )
            cpu_code_mismatches += int(
                np.count_nonzero(packed[:, past:, 4:] != ref[..., 4:])
            )
            np.testing.assert_allclose(
                unpack_rows(packed[:, past:])[1],
                unpack_rows(ref)[1],
                rtol=3e-7,
                atol=2e-6,
            )
        fresh_k.close()
        fresh_v.close()
        row = dict(
            frozen_encoder_byte_exact=True,
            cpu_encoder_code_mismatches=cpu_code_mismatches,
            past=past,
            count=count,
            max_abs=float(np.max(np.abs(actual - expected))),
            relative_l2=float(
                np.linalg.norm(actual - expected) / np.linalg.norm(expected)
            ),
            attention_ms=ms,
            scratch_bytes=work.size,
            old_cache_unchanged=True,
        )
        records.append(row)
        print(json.dumps(row), flush=True)
        for b in bs:
            b.close()
    for b in (dp, dk, dv):
        b.close()
    check(hip.hipStreamDestroy(stream))
    dest = Path(os.environ.get("TQ_APPEND_RESULTS", ROOT / "results/append"))
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "kernel.json").write_text(
        json.dumps(
            dict(
                status="passed",
                gate="append_attention_and_frozen_encoder_regression",
                strict_cpu_encoder_parity_passed=all(
                    x["cpu_encoder_code_mismatches"] == 0 for x in records
                ),
                encoder_baseline=baseline_path,
                cases=records,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
