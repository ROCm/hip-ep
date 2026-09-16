#!/usr/bin/env python3

#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""
MatMulNBits bits=8 (uint8, NOT bit-packed) test data generator + NumPy reference

Unlike bits=2/3/4, the bits=8 B matrix is one byte per weight -- no nibble
or sub-byte packing at all. This generates:
  A       : FP16  [M, K]
  B       : uint8 [N, K]                     (one byte per weight, 0..255)
  scales  : FP16  [N, num_groups_k]
  zeros   : uint8 [N, num_groups_k]          (optional; default zp=128 when
                                               absent, values drawn near 128
                                               when present)
  C_ref   : FP16  [M, N]                     (fp32-accumulated reference,
                                               rounded to fp16 at the end --
                                               used as ground truth for BOTH
                                               the fp16 and fp32 element_size
                                               instantiations; see README /
                                               Xfp16_Wu2_Yfp16's testFp32Shape
                                               for why the same fp16 ref is
                                               valid ground truth for the
                                               fp32 path)

Dequant: w_fp = (uint8_weight - zero_point) * scale
Reference matmul: C[m,n] = sum_k A[m,k] * w_fp[n,k]

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


def parse_size(size_str):
    parts = size_str.split('x')
    if len(parts) != 3:
        raise ValueError(f"Size must be MxKxN (got '{size_str}')")
    return int(parts[0]), int(parts[1]), int(parts[2])


def generate_one(M, K, N, group_size, out_dir, no_zeros=False, no_ref=False, seed=42):
    num_groups_k = (K + group_size - 1) // group_size

    os.makedirs(out_dir, exist_ok=True)

    zp_str = "no-zeros" if no_zeros else "with-zeros"
    print(f"Generating M={M} N={N} K={K} gs={group_size} groups={num_groups_k} "
          f"({zp_str}, seed={seed})")

    np.random.seed(seed)

    # ---- A (shared by fp16 and fp32 instantiations; fp32 test upcasts this
    # exact fp16 value host-side, matching the kernel's internal fp32->fp16
    # downcast at the load boundary) ----
    A = np.random.uniform(-0.5, 0.5, (M, K)).astype(np.float16)
    A.flatten(order='C').tofile(os.path.join(out_dir, "matmul_nbits_i8_A.bin"))

    group_idx = np.arange(K) // group_size

    # ---- B: one byte per weight, NOT packed ----
    B = np.random.randint(0, 256, (N, K), dtype=np.uint8)
    B.flatten(order='C').tofile(os.path.join(out_dir, "matmul_nbits_i8_B.bin"))

    scales = np.random.uniform(0.01, 0.05, (N, num_groups_k)).astype(np.float16)
    scales.flatten(order='C').tofile(
        os.path.join(out_dir, "matmul_nbits_i8_scales.bin"))

    zeros = None
    if not no_zeros:
        # Default (symmetric) zero point for 8-bit is 128 = 2^(8-1); vary
        # around it per group to exercise the asymmetric path.
        zeros = np.random.randint(120, 137, (N, num_groups_k), dtype=np.uint8)
        zeros.flatten(order='C').tofile(
            os.path.join(out_dir, "matmul_nbits_i8_zeros_u8.bin"))

    if not no_ref:
        print("Computing reference...", end=" ", flush=True)
        t0 = time.time()

        scales_f32 = scales.astype(np.float32)
        if zeros is not None:
            zp_f32 = zeros.astype(np.float32)
            B_dq = (B.astype(np.float32) - zp_f32[:, group_idx]) \
                * scales_f32[:, group_idx]
        else:
            B_dq = (B.astype(np.float32) - 128.0) * scales_f32[:, group_idx]
        C_ref = (A.astype(np.float32) @ B_dq.T).astype(np.float16)
        C_ref.flatten(order='C').tofile(
            os.path.join(out_dir, "matmul_nbits_i8_C_ref.bin"))

        elapsed = time.time() - t0
        print(f"done ({elapsed:.2f}s)")
    else:
        print("Skipping reference (--no-ref)")

    print(f"B size: {N * K} bytes (uint8, unpacked)")

    meta_file = os.path.join(out_dir, "matmul_nbits_meta.txt")
    with open(meta_file, 'w') as f:
        f.write(f"M={M}\nN={N}\nK={K}\ngroup_size={group_size}\n")
        f.write(f"num_groups_k={num_groups_k}\nseed={seed}\n")
        f.write(f"use_zeros={'true' if not no_zeros else 'false'}\n")

    print(f"Data saved to {out_dir}/")


def run_model_sweep(model_json, group_size, out_dir, no_zeros, no_ref):
    """Generate bits=8 (uint8, unpacked) test data for every shape defined
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
    for k, n in zip(K_array, N_array):
        for m in M_array:
            idx += 1
            shape = f'{m}x{k}x{n}'
            out = os.path.join(out_dir, shape)
            print(f'[{idx}/{total}] Generating {shape} -> {out}/')
            generate_one(m, k, n, group_size, out, no_zeros=no_zeros, no_ref=no_ref)

    print(f'\nDone: {total} shapes generated in {out_dir}/')


def main():
    parser = argparse.ArgumentParser(
        description='Generate MatMulNBits bits=8 (uint8, unpacked) test data')
    parser.add_argument('size', nargs='?', type=str, default='128x128x128',
                        help='Matrix size MxKxN (default: 128x128x128)')
    parser.add_argument('--group-size', type=int, default=128,
                        help='Quantization group size along K (default: 128)')
    parser.add_argument('--no-ref', action='store_true',
                        help='Skip computing reference C')
    parser.add_argument('--no-zeros', action='store_true',
                        help='Skip generating zero_points (use default zp=128)')
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
