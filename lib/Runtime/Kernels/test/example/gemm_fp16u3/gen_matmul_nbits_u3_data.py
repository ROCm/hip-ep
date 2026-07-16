#!/usr/bin/env python3
"""
MatMulNBits bits=3 vs bits=4 test data generator + NumPy reference

Generates a SHARED FP16 A, then two independently-quantized B matrices of
the SAME (M, K, N, group_size) shape:
  - uint3, packed as a continuous per-row 3-bit bitstream (custom format,
    see matmul_nbits_kernel.hip Section 1c/1d -- NOT an ONNX MatMulNBits
    convention)
  - uint4, nibble-packed (the existing ONNX MatMulNBits convention, same
    format gen_matmul_nbits_data.py in ../gemm_fp16u4 produces)

so the C++ test binary can benchmark hip_matmul_nbits(bits=3) and
hip_matmul_nbits(bits=4) back-to-back on identical shapes and report a fair
side-by-side comparison.

uint3 packing: value k occupies bits [3k, 3k+3) of row n's bitstream,
LSB-first. Row byte stride = ceil(K*3/8) bytes (K must be a multiple of 8;
the GEMV fast path in the kernel additionally requires K % 32 == 0).

Data layout (all row-major):
  A               : FP16  [M, K]                    (shared by both kernels)
  u3_B            : uint8 [N, ceil(K*3/8)]  continuous 3-bit bitstream
  u3_scales       : FP16  [N, num_groups_k]
  u3_zeros        : uint8 [N, num_groups_k]          (optional)
  u3_C_ref        : FP16  [M, N]
  u4_B_packed     : uint8 [N, K/2]       nibble-packed
  u4_scales       : FP16  [N, num_groups_k]
  u4_zeros_u8     : uint8 [N, num_groups_k]          (optional, per-element,
                                                       NOT nibble-packed)
  u4_zeros_fp16   : FP16  [N, num_groups_k]          (optional, same values
                                                       as u4_zeros_u8, cast
                                                       to fp16)
  u4_C_ref        : FP16  [M, N]

hip_matmul_nbits() takes zero_points as either nibble-packed uint8 (ONNX
convention, zp_elem_size=1, requires a separately-unpacked pre_unpacked_zp_u8
/ pre_unpacked_zp_fp16 pair -- normally produced by the runtime's zp-unpack
cache) or plain fp16 (zp_elem_size=2, only valid on the WMMA / col-major-GEMV
paths). To exercise every dispatch path (WMMA, GEMV, naive) correctly for
bits=4 with a *single* API call convention, this generator instead emits an
already-unpacked per-element uint8/fp16 zero-point pair and the test binary
passes them directly as pre_unpacked_zp_u8 / pre_unpacked_zp_fp16 (as if the
runtime's unpack cache had already run), matching the real integration path.

Usage:
    python3 gen_matmul_nbits_u3_data.py [MxKxN] [--group-size GS] [--dir DIR]
"""

import numpy as np
import argparse
import os
import time


def pack_u3_continuous(qvals):
    """qvals: uint8 [N, K] with values 0..7 -> uint8 [N, ceil(K*3/8)].

    Continuous per-row bitstream: value k occupies bits [3k, 3k+3),
    LSB-first. Bit fields never overlap, so accumulating each byte's
    contributions with plain integer addition is equivalent to OR-ing them.
    """
    N, K = qvals.shape
    row_bytes = (K * 3 + 7) // 8
    bit_pos = np.arange(K, dtype=np.int64) * 3
    byte_pos = bit_pos // 8
    shift = (bit_pos % 8).astype(np.uint32)

    contrib = (qvals.astype(np.uint32) << shift[None, :])
    lo = (contrib & 0xFF).astype(np.uint16)
    hi = ((contrib >> 8) & 0xFF).astype(np.uint16)

    out = np.zeros((N, row_bytes + 1), dtype=np.uint16)
    np.add.at(out, (slice(None), byte_pos), lo)
    np.add.at(out, (slice(None), byte_pos + 1), hi)

    return out[:, :row_bytes].astype(np.uint8)


def unpack_u3_continuous(packed, K):
    """packed: uint8 [N, row_bytes] -> uint8 [N, K] values 0..7 (inverse of
    pack_u3_continuous; used only to build the NumPy reference output)."""
    N = packed.shape[0]
    bit_pos = np.arange(K, dtype=np.int64) * 3
    byte_pos = bit_pos // 8
    shift = (bit_pos % 8).astype(np.uint32)

    padded = np.concatenate([packed, np.zeros((N, 1), dtype=np.uint8)], axis=1)
    lo = padded[:, byte_pos].astype(np.uint32)
    hi = padded[:, byte_pos + 1].astype(np.uint32)
    combined = lo | (hi << 8)
    return ((combined >> shift) & 0x7).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description='Generate MatMulNBits bits=3 vs bits=4 comparison test data')
    parser.add_argument('size', nargs='?', type=str, default='128x128x128',
                        help='Matrix size MxKxN (default: 128x128x128)')
    parser.add_argument('--group-size', type=int, default=128,
                        help='Quantization group size along K (default: 128)')
    parser.add_argument('--no-ref', action='store_true',
                        help='Skip computing reference C')
    parser.add_argument('--no-zeros', action='store_true',
                        help='Skip generating zeros (use default zero point)')
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

    if K % 32 != 0:
        parser.error(f"K ({K}) must be a multiple of 32 (uint3 GEMV fast "
                     f"path + uint4 packing both require it)")

    out_dir = args.dir
    os.makedirs(out_dir, exist_ok=True)

    zp_str = "no-zeros" if args.no_zeros else "with-zeros"
    print(f"Generating M={M} N={N} K={K} gs={group_size} groups={num_groups_k} "
          f"({zp_str}, seed={args.seed})")

    np.random.seed(args.seed)

    # ---- Shared A ----
    A = np.random.uniform(-0.5, 0.5, (M, K)).astype(np.float16)
    A.flatten(order='C').tofile(os.path.join(out_dir, "matmul_nbits_A.bin"))

    group_idx = np.arange(K) // group_size

    # ================= uint3 (continuous bitstream) =================
    B_u3 = np.random.randint(0, 8, (N, K), dtype=np.uint8)
    B_u3_packed = pack_u3_continuous(B_u3)

    # Self-check: pack/unpack round-trip before trusting the reference.
    B_u3_roundtrip = unpack_u3_continuous(B_u3_packed, K)
    if not np.array_equal(B_u3, B_u3_roundtrip):
        raise RuntimeError("uint3 pack/unpack round-trip mismatch -- "
                           "pack_u3_continuous has a bug")

    scales_u3 = np.random.uniform(0.01, 0.05, (N, num_groups_k)).astype(np.float16)
    zeros_u3 = None
    zeros_u3_packed = None
    if not args.no_zeros:
        zeros_u3 = np.random.randint(3, 6, (N, num_groups_k)).astype(np.uint8)
        # ONNX bits=3 packs zero_points as the same continuous per-row 3-bit
        # stream as the weights, one group-zp per 3-bit field.
        zeros_u3_packed = pack_u3_continuous(zeros_u3)

    B_u3_packed.flatten(order='C').tofile(
        os.path.join(out_dir, "matmul_nbits_u3_B.bin"))
    scales_u3.flatten(order='C').tofile(
        os.path.join(out_dir, "matmul_nbits_u3_scales.bin"))
    if zeros_u3 is not None:
        zeros_u3.flatten(order='C').tofile(
            os.path.join(out_dir, "matmul_nbits_u3_zeros.bin"))
        zeros_u3_packed.flatten(order='C').tofile(
            os.path.join(out_dir, "matmul_nbits_u3_zeros_packed.bin"))

    # ================= uint4 (nibble-packed, ONNX convention) =================
    B_u4 = np.random.randint(0, 16, (N, K), dtype=np.uint8)
    B_u4_even = B_u4[:, 0::2]
    B_u4_odd = B_u4[:, 1::2]
    B_u4_packed = (B_u4_even | (B_u4_odd << 4)).astype(np.uint8)

    scales_u4 = np.random.uniform(0.01, 0.05, (N, num_groups_k)).astype(np.float16)
    zeros_u4_u8 = None
    zeros_u4_fp16 = None
    if not args.no_zeros:
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
    if not args.no_ref:
        print("Computing references...", end=" ", flush=True)
        t0 = time.time()

        scales_u3_f32 = scales_u3.astype(np.float32)
        if zeros_u3 is not None:
            zp_u3_f32 = zeros_u3.astype(np.float32)
            B_u3_dq = (B_u3.astype(np.float32) - zp_u3_f32[:, group_idx]) \
                * scales_u3_f32[:, group_idx]
        else:
            B_u3_dq = (B_u3.astype(np.float32) - 4.0) * scales_u3_f32[:, group_idx]
        C_ref_u3 = (A.astype(np.float32) @ B_u3_dq.T).astype(np.float16)
        C_ref_u3.flatten(order='C').tofile(
            os.path.join(out_dir, "matmul_nbits_u3_C_ref.bin"))

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

    row_bytes_u3 = (K * 3 + 7) // 8
    row_bytes_u4 = K // 2
    print(f"B size: uint3={N * row_bytes_u3} bytes, uint4={N * row_bytes_u4} "
          f"bytes ({100.0 * row_bytes_u3 / row_bytes_u4:.1f}% of uint4)")

    meta_file = os.path.join(out_dir, "matmul_nbits_meta.txt")
    with open(meta_file, 'w') as f:
        f.write(f"M={M}\nN={N}\nK={K}\ngroup_size={group_size}\n")
        f.write(f"num_groups_k={num_groups_k}\nseed={args.seed}\n")
        f.write(f"use_zeros={'true' if not args.no_zeros else 'false'}\n")

    print(f"Data saved to {out_dir}/")


if __name__ == '__main__':
    main()
