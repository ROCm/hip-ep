#!/usr/bin/env python3

#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""
MatMulNBits test data generator + NumPy reference

Generates random FP16 A, UINT4-packed B, scales, zeros, and computes
the golden reference C = A @ dequant(B)^T using NumPy.

Data layout matches the hip_matmul_nbits public API (all row-major):
  A      : FP16 row-major [M, K]
  B      : uint4 packed [N, K/2], row-major
  scales : FP16 [N, num_groups_k], row-major
  zeros  : FP16 [N, num_groups_k], row-major  (optional)
  C_ref  : FP16 row-major [M, N]

Usage:
    python3 gen_data.py [MxKxN] [--group-size GS] [--dir DIR]
    python3 gen_data.py --model ../../models/gpt_oss_20b.json --group-size GS [--out-dir DIR]

Examples:
    python3 gen_data.py                          # 128x128x128 gs=128
    python3 gen_data.py 256x512x256 --group-size 128
"""

import numpy as np
import argparse
import json
import os
import sys
import time


def parse_size(size_str):
    parts = size_str.split('x')
    if len(parts) != 3:
        raise ValueError(f"Size must be MxKxN (got '{size_str}')")
    return int(parts[0]), int(parts[1]), int(parts[2])


def generate_one(M, K, N, group_size, out_dir, no_zeros=False, no_ref=False, seed=42):
    num_groups_k = (K + group_size - 1) // group_size

    if K % 2 != 0:
        raise ValueError(f"K ({K}) must be even for uint4 packing")

    os.makedirs(out_dir, exist_ok=True)

    zp_str = "no-zeros" if no_zeros else "with-zeros"
    print(f"Generating M={M} N={N} K={K} gs={group_size} groups={num_groups_k} "
          f"({zp_str}, seed={seed})")

    np.random.seed(seed)

    A = np.random.uniform(-0.5, 0.5, (M, K)).astype(np.float16)

    B_uint4 = np.random.randint(0, 16, (N, K), dtype=np.uint8)
    B_even = B_uint4[:, 0::2]
    B_odd  = B_uint4[:, 1::2]
    B_packed_real = (B_even | (B_odd << 4)).astype(np.uint8)

    # ONNX MatMulNBits pads each row's blob to num_groups_k * (group_size/2)
    # bytes -- the last group is padded to a full group_size even when K is not
    # a multiple of group_size. hip_matmul_nbits' row stride assumes this
    # padded layout (see matmul_nbits_kernel.hip row-stride comments), so an
    # unpadded K/2-byte row misaligns every row n>0 whenever K % group_size != 0.
    row_bytes = num_groups_k * (group_size // 2)
    B_packed = np.zeros((N, row_bytes), dtype=np.uint8)
    B_packed[:, :B_packed_real.shape[1]] = B_packed_real

    scales = np.random.uniform(0.01, 0.05, (N, num_groups_k)).astype(np.float16)

    zeros = None
    if not no_zeros:
        # ONNX MatMulNBits zero_points are packed at `bits` bits (4 bits here),
        # so they are always small integers (0..15), never fractional -- the
        # FP16 buffer this test passes to hip_matmul_nbits (zp_elem_size==2)
        # is a pre-converted cast of those integers, not an independently
        # continuous value. Matches the integer convention already used by
        # ../Xfp16_Wu3_Yfp16/gen_data.py and ../Xfp16_Wu2_Yfp16/gen_data.py's
        # zero_points.
        zeros = np.random.randint(7, 10, (N, num_groups_k)).astype(np.uint8) \
                    .astype(np.float16)

    # A stored row-major (C order)
    file_A = os.path.join(out_dir, "matmul_nbits_A.bin")
    A.flatten(order='C').tofile(file_A)

    # B_packed stored row-major (C order)
    file_B = os.path.join(out_dir, "matmul_nbits_B_packed.bin")
    B_packed.flatten(order='C').tofile(file_B)

    # scales stored row-major
    file_S = os.path.join(out_dir, "matmul_nbits_scales.bin")
    scales.flatten(order='C').tofile(file_S)

    if zeros is not None:
        # zeros stored row-major (FP16)
        file_Z = os.path.join(out_dir, "matmul_nbits_zeros.bin")
        zeros.flatten(order='C').tofile(file_Z)

    if not no_ref:
        print("Computing reference C...", end=" ", flush=True)
        t0 = time.time()

        group_idx = np.arange(K) // group_size
        scales_f32 = scales.astype(np.float32)

        if zeros is not None:
            zeros_f32 = zeros.astype(np.float32)
            B_dq = (B_uint4.astype(np.float32) - zeros_f32[:, group_idx]) \
                 * scales_f32[:, group_idx]
        else:
            B_dq = (B_uint4.astype(np.float32) - 8.0) * scales_f32[:, group_idx]

        C_ref_f32 = A.astype(np.float32) @ B_dq.T
        C_ref = C_ref_f32.astype(np.float16)

        elapsed = time.time() - t0
        gflops = (2.0 * M * N * K) / (max(elapsed, 1e-9) * 1e9)

        # C_ref stored row-major (C order)
        file_C = os.path.join(out_dir, "matmul_nbits_C_ref.bin")
        C_ref.flatten(order='C').tofile(file_C)
        print(f"done ({elapsed:.2f}s, {gflops:.1f} GFLOPS)")
    else:
        print("Skipping reference (--no-ref)")

    meta_file = os.path.join(out_dir, "matmul_nbits_meta.txt")
    with open(meta_file, 'w') as f:
        f.write(f"M={M}\nN={N}\nK={K}\ngroup_size={group_size}\n")
        f.write(f"num_groups_k={num_groups_k}\nseed={seed}\n")
        f.write(f"use_zeros={'true' if zeros is not None else 'false'}\n")

    print(f"Data saved to {out_dir}/")


def run_model_sweep(model_json, group_size, out_dir, no_zeros, no_ref):
    """Generate test data for every shape defined in a model config JSON."""
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
    for k, n in zip(K_array, N_array):
        for m in M_array:
            idx += 1
            shape = f'{m}x{k}x{n}'
            out = os.path.join(out_dir, shape)
            print(f'[{idx}/{total}] Generating {shape} -> {out}/')
            generate_one(m, k, n, group_size, out, no_zeros=no_zeros, no_ref=no_ref)

    print(f'\nDone: {total} shapes generated in {out_dir}/')


def main():
    parser = argparse.ArgumentParser(description='Generate MatMulNBits test data')
    parser.add_argument('size', nargs='?', type=str, default='128x128x128',
                        help='Matrix size MxKxN (default: 128x128x128)')
    parser.add_argument('--group-size', type=int, default=128,
                        help='Quantization group size along K (default: 128)')
    parser.add_argument('--no-ref', action='store_true',
                        help='Skip computing reference C')
    parser.add_argument('--no-zeros', action='store_true',
                        help='Skip generating zeros (zero points = 0)')
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
