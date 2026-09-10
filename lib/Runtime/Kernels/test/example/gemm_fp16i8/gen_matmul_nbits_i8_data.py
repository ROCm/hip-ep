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
                                               gemm_fp16u2's testFp32Shape for
                                               why the same fp16 ref is valid
                                               ground truth for the fp32 path)

Dequant: w_fp = (uint8_weight - zero_point) * scale
Reference matmul: C[m,n] = sum_k A[m,k] * w_fp[n,k]

Usage:
    python3 gen_matmul_nbits_i8_data.py [MxKxN] [--group-size GS] [--dir DIR]
"""

import numpy as np
import argparse
import os
import time


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
                        help='Output directory (default: data/)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    parts = args.size.split('x')
    if len(parts) != 3:
        parser.error(f"Size must be MxKxN (got '{args.size}')")
    M, K, N = int(parts[0]), int(parts[1]), int(parts[2])
    group_size = args.group_size
    num_groups_k = (K + group_size - 1) // group_size

    out_dir = args.dir
    os.makedirs(out_dir, exist_ok=True)

    zp_str = "no-zeros" if args.no_zeros else "with-zeros"
    print(f"Generating M={M} N={N} K={K} gs={group_size} groups={num_groups_k} "
          f"({zp_str}, seed={args.seed})")

    np.random.seed(args.seed)

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
    if not args.no_zeros:
        # Default (symmetric) zero point for 8-bit is 128 = 2^(8-1); vary
        # around it per group to exercise the asymmetric path.
        zeros = np.random.randint(120, 137, (N, num_groups_k), dtype=np.uint8)
        zeros.flatten(order='C').tofile(
            os.path.join(out_dir, "matmul_nbits_i8_zeros_u8.bin"))

    if not args.no_ref:
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
        f.write(f"num_groups_k={num_groups_k}\nseed={args.seed}\n")
        f.write(f"use_zeros={'true' if not args.no_zeros else 'false'}\n")

    print(f"Data saved to {out_dir}/")


if __name__ == '__main__':
    main()
