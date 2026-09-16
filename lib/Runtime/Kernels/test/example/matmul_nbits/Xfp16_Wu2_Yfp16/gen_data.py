#!/usr/bin/env python3

#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""
MatMulNBits bits=2 vs bits=4 test data generator + NumPy reference

Generates a SHARED FP16 A, then two independently-quantized B matrices of
the SAME (M, K, N, group_size) shape:
  - uint2, packed 4-per-byte, LSB-first (this IS the ONNX MatMulNBits
    blockwise convention for bits=2 — blob_size = block_size/4 bytes — and,
    because 2 divides 8 evenly, simultaneously a plain continuous per-row
    2-bit stream; the two are byte-identical when block_size % 4 == 0)
  - uint4, nibble-packed (the existing ONNX MatMulNBits convention, same
    format ../Xfp16_Wu4_Yfp16/gen_data.py produces)

so the C++ test binary can benchmark hip_matmul_nbits(bits=2) and
hip_matmul_nbits(bits=4) back-to-back on identical shapes and report a fair
side-by-side comparison.

uint2 packing: value k occupies bits [2k, 2k+2) of row n's bitstream,
LSB-first. Row byte stride = K/4 bytes (K must be a multiple of 4; the GEMV
fast path in the kernel additionally requires K % 32 == 0).

zero_points, THREE layouts (all emitted when zeros are enabled):
  - *_zeros_u8      : uint8 [N, num_groups_k]  one byte per group (0..3).
                      This is what the u2 kernels index directly, and what
                      the runtime's zp-unpack kernel produces.
  - *_zeros_packed  : uint8 [N, ceil(num_groups_k/4)]  the SAME values,
                      packed 4-per-byte exactly like the ONNX bits=2
                      zero_points blob a real model ships. The test binary
                      runs hip_matmul_nbits_unpack_zp_u8_2bit() on this and
                      checks it reproduces *_zeros_u8, then feeds the result
                      through the kernel as pre_unpacked_zp_u8 — i.e. the
                      real runtime integration path.
  - *_zeros_fp16    : FP16 [N, num_groups_k]  *fractional* per-group zero-
                      points (values drawn from {0.5, 1.5, 2.5}). AMD Quark
                      2-bit models express the non-uniform levels
                      [-1,-1/3,1/3,1] as (index - 1.5) * scale, so they ship
                      an FP16 zero-point (zp_elem_size==2) that MUST NOT be
                      rounded to an integer. The test binary passes this
                      straight through as zero_points with zp_elem_size=2 and
                      checks the kernel's fp16-zp path against a dedicated
                      fractional reference. Uses the SAME B/scales as the
                      integer path — only the zero-point differs.

Data layout (all row-major):
  A               : FP16  [M, K]                    (shared by both kernels)
  u2_B            : uint8 [N, K/4]        packed 2-bit stream
  u2_scales       : FP16  [N, num_groups_k]
  u2_zeros_u8     : uint8 [N, num_groups_k]          (optional)
  u2_zeros_packed : uint8 [N, ceil(num_groups_k/4)]  (optional)
  u2_C_ref        : FP16  [M, N]
  u4_B_packed     : uint8 [N, K/2]        nibble-packed
  u4_scales       : FP16  [N, num_groups_k]
  u4_zeros_u8     : uint8 [N, num_groups_k]          (optional, per-element)
  u4_zeros_fp16   : FP16  [N, num_groups_k]          (optional, same values)
  u4_C_ref        : FP16  [M, N]

Usage:
    python3 gen_data.py [MxKxN] [--group-size GS] [--dir DIR]
    python3 gen_data.py --model ../../models/gpt_oss_20b.json --group-size GS [--out-dir DIR]
"""

import numpy as np
import argparse
import json
import os
import sys
import time


def pack_2bit(vals):
    """vals: uint8 [N, L] with values 0..3 -> uint8 [N, ceil(L/4)], packing
    4 values per byte LSB-first: byte = v0 | v1<<2 | v2<<4 | v3<<6.

    Used for both the weight stream (L=K, K%4==0) and the packed zero_points
    blob (L=num_groups_k, padded with zeros to a multiple of 4)."""
    N, L = vals.shape
    Lpad = (L + 3) // 4 * 4
    if Lpad != L:
        pad = np.zeros((N, Lpad - L), dtype=np.uint8)
        vals = np.concatenate([vals, pad], axis=1)
    v = vals.reshape(N, Lpad // 4, 4).astype(np.uint32)
    packed = v[:, :, 0] | (v[:, :, 1] << 2) | (v[:, :, 2] << 4) | (v[:, :, 3] << 6)
    return packed.astype(np.uint8)


def unpack_2bit(packed, L):
    """packed: uint8 [N, ceil(L/4)] -> uint8 [N, L] values 0..3 (inverse of
    pack_2bit; used only to self-check the round-trip)."""
    N = packed.shape[0]
    out = np.zeros((N, packed.shape[1] * 4), dtype=np.uint8)
    for j in range(4):
        out[:, j::4] = (packed >> (2 * j)) & 0x3
    return out[:, :L]


def parse_size(size_str):
    parts = size_str.split('x')
    if len(parts) != 3:
        raise ValueError(f"Size must be MxKxN (got '{size_str}')")
    return int(parts[0]), int(parts[1]), int(parts[2])


def generate_one(M, K, N, group_size, out_dir, no_zeros=False, no_ref=False, seed=42):
    num_groups_k = (K + group_size - 1) // group_size

    if K % 32 != 0:
        raise ValueError(f"K ({K}) must be a multiple of 32 (uint2 GEMV fast "
                          f"path + uint4 packing both require it)")

    os.makedirs(out_dir, exist_ok=True)

    zp_str = "no-zeros" if no_zeros else "with-zeros"
    print(f"Generating M={M} N={N} K={K} gs={group_size} groups={num_groups_k} "
          f"({zp_str}, seed={seed})")

    np.random.seed(seed)

    # ---- Shared A ----
    A = np.random.uniform(-0.5, 0.5, (M, K)).astype(np.float16)
    A.flatten(order='C').tofile(os.path.join(out_dir, "matmul_nbits_A.bin"))

    group_idx = np.arange(K) // group_size

    # ================= uint2 (2-bit packed, ONNX convention) =================
    B_u2 = np.random.randint(0, 4, (N, K), dtype=np.uint8)
    B_u2_packed = pack_2bit(B_u2)

    # Self-check: pack/unpack round-trip before trusting the reference.
    B_u2_roundtrip = unpack_2bit(B_u2_packed, K)
    if not np.array_equal(B_u2, B_u2_roundtrip):
        raise RuntimeError("uint2 pack/unpack round-trip mismatch -- "
                           "pack_2bit has a bug")

    scales_u2 = np.random.uniform(0.01, 0.05, (N, num_groups_k)).astype(np.float16)
    zeros_u2 = None
    zeros_u2_packed = None
    zeros_u2_fp16 = None
    if not no_zeros:
        # 2-bit zero_points are in 0..3; default (symmetric) zp is 2.
        zeros_u2 = np.random.randint(1, 3, (N, num_groups_k)).astype(np.uint8)
        zeros_u2_packed = pack_2bit(zeros_u2)
        # Fractional FP16 zero-points (AMD Quark 2-bit uses 1.5). Vary per group
        # among fractional values so the test exercises both fp16 reading AND
        # per-group indexing; all are non-integer to prove no rounding occurs.
        zeros_u2_fp16 = np.random.choice(
            np.array([0.5, 1.5, 2.5], dtype=np.float32),
            size=(N, num_groups_k)).astype(np.float16)

    B_u2_packed.flatten(order='C').tofile(
        os.path.join(out_dir, "matmul_nbits_u2_B.bin"))
    scales_u2.flatten(order='C').tofile(
        os.path.join(out_dir, "matmul_nbits_u2_scales.bin"))
    if zeros_u2 is not None:
        zeros_u2.flatten(order='C').tofile(
            os.path.join(out_dir, "matmul_nbits_u2_zeros_u8.bin"))
        zeros_u2_packed.flatten(order='C').tofile(
            os.path.join(out_dir, "matmul_nbits_u2_zeros_packed.bin"))
        zeros_u2_fp16.flatten(order='C').tofile(
            os.path.join(out_dir, "matmul_nbits_u2_zeros_fp16.bin"))

    # ================= uint4 (nibble-packed, ONNX convention) =================
    B_u4 = np.random.randint(0, 16, (N, K), dtype=np.uint8)
    B_u4_even = B_u4[:, 0::2]
    B_u4_odd = B_u4[:, 1::2]
    B_u4_packed_real = (B_u4_even | (B_u4_odd << 4)).astype(np.uint8)

    # ONNX MatMulNBits pads each row's blob to num_groups_k * (group_size/2)
    # bytes -- the last group is padded to a full group_size even when K is
    # not a multiple of group_size (mirrors ../Xfp16_Wu4_Yfp16/gen_data.py).
    u4_row_bytes = num_groups_k * (group_size // 2)
    B_u4_packed = np.zeros((N, u4_row_bytes), dtype=np.uint8)
    B_u4_packed[:, :B_u4_packed_real.shape[1]] = B_u4_packed_real

    scales_u4 = np.random.uniform(0.01, 0.05, (N, num_groups_k)).astype(np.float16)
    zeros_u4_u8 = None
    zeros_u4_fp16 = None
    if not no_zeros:
        zeros_u4_u8 = np.random.randint(7, 10, (N, num_groups_k), dtype=np.uint8)
        zeros_u4_fp16 = zeros_u4_u8.astype(np.float16)

    B_u4_packed.flatten(order='C').tofile(
        os.path.join(out_dir, "matmul_nbits_u4_B_packed.bin"))
    scales_u4.flatten(order='C').tofile(
        os.path.join(out_dir, "matmul_nbits_u4_scales.bin"))
    if zeros_u4_u8 is not None:
        zeros_u4_u8.flatten(order='C').tofile(
            os.path.join(out_dir, "matmul_nbits_u4_zeros_u8.bin"))
        zeros_u4_fp16.flatten(order='C').tofile(
            os.path.join(out_dir, "matmul_nbits_u4_zeros_fp16.bin"))

    # ================= References =================
    if not no_ref:
        print("Computing references...", end=" ", flush=True)
        t0 = time.time()

        scales_u2_f32 = scales_u2.astype(np.float32)
        if zeros_u2 is not None:
            zp_u2_f32 = zeros_u2.astype(np.float32)
            B_u2_dq = (B_u2.astype(np.float32) - zp_u2_f32[:, group_idx]) \
                * scales_u2_f32[:, group_idx]
        else:
            B_u2_dq = (B_u2.astype(np.float32) - 2.0) * scales_u2_f32[:, group_idx]
        C_ref_u2 = (A.astype(np.float32) @ B_u2_dq.T).astype(np.float16)
        C_ref_u2.flatten(order='C').tofile(
            os.path.join(out_dir, "matmul_nbits_u2_C_ref.bin"))

        # Fractional FP16 zero-point reference: SAME B/scales, fractional zp.
        if zeros_u2_fp16 is not None:
            zp_u2_fp16_f32 = zeros_u2_fp16.astype(np.float32)
            B_u2_dq_fp16 = (B_u2.astype(np.float32)
                            - zp_u2_fp16_f32[:, group_idx]) \
                * scales_u2_f32[:, group_idx]
            C_ref_u2_fp16 = (A.astype(np.float32) @ B_u2_dq_fp16.T) \
                .astype(np.float16)
            C_ref_u2_fp16.flatten(order='C').tofile(
                os.path.join(out_dir, "matmul_nbits_u2_C_ref_fp16zp.bin"))

        scales_u4_f32 = scales_u4.astype(np.float32)
        if zeros_u4_u8 is not None:
            zp_u4_f32 = zeros_u4_u8.astype(np.float32)
            B_u4_dq = (B_u4.astype(np.float32) - zp_u4_f32[:, group_idx]) \
                * scales_u4_f32[:, group_idx]
        else:
            B_u4_dq = (B_u4.astype(np.float32) - 8.0) * scales_u4_f32[:, group_idx]
        C_ref_u4 = (A.astype(np.float32) @ B_u4_dq.T).astype(np.float16)
        C_ref_u4.flatten(order='C').tofile(
            os.path.join(out_dir, "matmul_nbits_u4_C_ref.bin"))

        elapsed = time.time() - t0
        print(f"done ({elapsed:.2f}s)")
    else:
        print("Skipping reference (--no-ref)")

    row_bytes_u2 = K // 4
    row_bytes_u4 = K // 2
    print(f"B size: uint2={N * row_bytes_u2} bytes, uint4={N * row_bytes_u4} "
          f"bytes ({100.0 * row_bytes_u2 / row_bytes_u4:.1f}% of uint4)")

    meta_file = os.path.join(out_dir, "matmul_nbits_meta.txt")
    with open(meta_file, 'w') as f:
        f.write(f"M={M}\nN={N}\nK={K}\ngroup_size={group_size}\n")
        f.write(f"num_groups_k={num_groups_k}\nseed={seed}\n")
        f.write(f"use_zeros={'true' if not no_zeros else 'false'}\n")

    print(f"Data saved to {out_dir}/")


def run_model_sweep(model_json, group_size, out_dir, no_zeros, no_ref):
    """Generate bits=2 vs bits=4 comparison test data for every shape defined
    in a model config JSON."""
    with open(model_json) as f:
        cfg = json.load(f)

    M_array = cfg['M_array']
    K_array = cfg['KN_pairs']['K']
    N_array = cfg['KN_pairs']['N']

    if len(K_array) != len(N_array):
        print(f'ERROR: K ({len(K_array)}) and N ({len(N_array)}) arrays '
              f'must have the same length', file=sys.stderr)
        sys.exit(1)

    total = len(K_array) * len(M_array)
    print(f'Model: {model_json}')
    print(f'  {len(M_array)} M values x {len(K_array)} KN pairs = {total} shapes')
    print(f'  group_size={group_size}  '
          f'{"no-zeros" if no_zeros else "with-zeros"}')
    print()

    idx = 0
    skipped = []

    for k, n in zip(K_array, N_array):
        for m in M_array:
            idx += 1
            shape = f'{m}x{k}x{n}'
            if k % 32 != 0:
                print(f'[{idx}/{total}] Skipping {shape}: K={k} not a '
                      f'multiple of 32')
                skipped.append(shape)
                continue
            out = os.path.join(out_dir, shape)
            print(f'[{idx}/{total}] Generating {shape} -> {out}/')
            generate_one(m, k, n, group_size, out, no_zeros=no_zeros, no_ref=no_ref)

    print(f'\nDone: {total - len(skipped)} shapes generated in {out_dir}/'
          f'{f" ({len(skipped)} skipped: K not %32)" if skipped else ""}')


def main():
    parser = argparse.ArgumentParser(
        description='Generate MatMulNBits bits=2 vs bits=4 comparison test data')
    parser.add_argument('size', nargs='?', type=str, default='128x128x128',
                        help='Matrix size MxKxN (default: 128x128x128)')
    parser.add_argument('--group-size', type=int, default=128,
                        help='Quantization group size along K (default: 128)')
    parser.add_argument('--no-ref', action='store_true',
                        help='Skip computing reference C')
    parser.add_argument('--no-zeros', action='store_true',
                        help='Skip generating zeros (use default zero point)')
    parser.add_argument('--dir', type=str, default='data',
                        help='Output directory for single-shape mode (default: data/)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model config JSON: generate every '
                             'M_array x KN_pairs shape instead of a single shape')
    parser.add_argument('--out-dir', type=str, default='data_model',
                        help='Root output directory for --model mode (default: data_model/)')
    args = parser.parse_args()

    if args.model:
        run_model_sweep(args.model, args.group_size, args.out_dir,
                         args.no_zeros, args.no_ref)
        return

    try:
        M, K, N = parse_size(args.size)
    except ValueError as e:
        parser.error(str(e))

    try:
        generate_one(M, K, N, args.group_size, args.dir,
                     no_zeros=args.no_zeros, no_ref=args.no_ref, seed=args.seed)
    except ValueError as e:
        parser.error(str(e))


if __name__ == '__main__':
    main()
