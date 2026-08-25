#!/usr/bin/env python3
"""
MatMulNBits test data generator — bits=2, 3, and 4.

Single-shape mode:
    python gen_data.py 128x4096x4096 --bits 4 [--group-size 128] [--dir data] [--no-zeros]
    python gen_data.py 128x4096x4096 --bits 3 [--group-size 128] [--dir data]
    python gen_data.py 128x4096x4096 --bits 2 [--group-size 128] [--dir data]

Model sweep mode (reads M_array / KN_pairs from a JSON):
    python gen_data.py --model gemm_fp16u4/test_model/gpt_oss_20b.json --bits 4
    python gen_data.py --model gemm_fp16u4/test_model/gpt_oss_20b.json --bits 3 --no-zeros
    python gen_data.py --model gemm_fp16u4/test_model/gpt_oss_20b.json --bits 2 --out-dir data_model

bits=3 and bits=2 require K % 32 == 0 (GEMV fast-path constraint).

Data layouts — all row-major
  bits=4:
    matmul_nbits_A.bin          FP16  [M, K]
    matmul_nbits_B_packed.bin   uint8 [N, K/2]   nibble-packed (ONNX convention)
    matmul_nbits_scales.bin     FP16  [N, num_groups_k]
    matmul_nbits_zeros.bin      FP16  [N, num_groups_k]   (optional)
    matmul_nbits_C_ref.bin      FP16  [M, N]

  bits=3 (also emits paired u4 for side-by-side benchmark):
    matmul_nbits_A.bin               FP16  [M, K]
    matmul_nbits_u3_B.bin            uint8 [N, ceil(K*3/8)]   continuous 3-bit stream
    matmul_nbits_u3_scales.bin       FP16  [N, num_groups_k]
    matmul_nbits_u3_zeros.bin        uint8 [N, num_groups_k]          (optional)
    matmul_nbits_u3_zeros_packed.bin uint8 [N, ceil(num_groups_k/4)]  (optional, ONNX-packed zp)
    matmul_nbits_u3_C_ref.bin        FP16  [M, N]
    matmul_nbits_u4_B_packed.bin     uint8 [N, K/2]
    matmul_nbits_u4_scales.bin       FP16  [N, num_groups_k]
    matmul_nbits_u4_zeros_u8.bin     uint8 [N, num_groups_k]   (optional)
    matmul_nbits_u4_zeros_fp16.bin   FP16  [N, num_groups_k]   (optional)
    matmul_nbits_u4_C_ref.bin        FP16  [M, N]

  bits=2 (also emits paired u4 for side-by-side benchmark):
    matmul_nbits_A.bin               FP16  [M, K]
    matmul_nbits_u2_B.bin            uint8 [N, K/4]                  packed 2-bit stream
    matmul_nbits_u2_scales.bin       FP16  [N, num_groups_k]
    matmul_nbits_u2_zeros_u8.bin     uint8 [N, num_groups_k]          (optional)
    matmul_nbits_u2_zeros_packed.bin uint8 [N, ceil(num_groups_k/4)]  (optional, ONNX-packed zp)
    matmul_nbits_u2_C_ref.bin        FP16  [M, N]
    matmul_nbits_u4_B_packed.bin     uint8 [N, K/2]
    matmul_nbits_u4_scales.bin       FP16  [N, num_groups_k]
    matmul_nbits_u4_zeros_u8.bin     uint8 [N, num_groups_k]   (optional)
    matmul_nbits_u4_zeros_fp16.bin   FP16  [N, num_groups_k]   (optional)
    matmul_nbits_u4_C_ref.bin        FP16  [M, N]
"""

import json
import sys
import argparse
import os
import time

import numpy as np


# ============================================================
# Packing helpers (bits=2 and bits=3)
# ============================================================

def _pack_2bit(vals):
    """uint8 [N, L] values 0..3 -> uint8 [N, ceil(L/4)], LSB-first.
    byte = v0 | v1<<2 | v2<<4 | v3<<6.
    Used for both the weight stream (L=K) and the zero_points blob (L=num_groups_k)."""
    N, L = vals.shape
    Lpad = (L + 3) // 4 * 4
    if Lpad != L:
        vals = np.concatenate([vals, np.zeros((N, Lpad - L), dtype=np.uint8)], axis=1)
    v = vals.reshape(N, Lpad // 4, 4).astype(np.uint32)
    packed = v[:, :, 0] | (v[:, :, 1] << 2) | (v[:, :, 2] << 4) | (v[:, :, 3] << 6)
    return packed.astype(np.uint8)


def _unpack_2bit(packed, L):
    """uint8 [N, ceil(L/4)] -> uint8 [N, L] values 0..3 (inverse of _pack_2bit)."""
    N = packed.shape[0]
    out = np.zeros((N, packed.shape[1] * 4), dtype=np.uint8)
    for j in range(4):
        out[:, j::4] = (packed >> (2 * j)) & 0x3
    return out[:, :L]


def _pack_u3_continuous(qvals):
    """uint8 [N, K] values 0..7 -> uint8 [N, ceil(K*3/8)].
    Value k occupies bits [3k, 3k+3), LSB-first."""
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


def _unpack_u3_continuous(packed, K):
    """uint8 [N, row_bytes] -> uint8 [N, K] values 0..7 (inverse of _pack_u3_continuous)."""
    N = packed.shape[0]
    bit_pos = np.arange(K, dtype=np.int64) * 3
    byte_pos = bit_pos // 8
    shift = (bit_pos % 8).astype(np.uint32)
    padded = np.concatenate([packed, np.zeros((N, 1), dtype=np.uint8)], axis=1)
    lo = padded[:, byte_pos].astype(np.uint32)
    hi = padded[:, byte_pos + 1].astype(np.uint32)
    return ((lo | (hi << 8)) >> shift & 0x7).astype(np.uint8)


# ============================================================
# Per-bits generators
# ============================================================

def _generate_u4(M, K, N, group_size, no_zeros, no_ref, seed, out_dir):
    num_groups_k = (K + group_size - 1) // group_size
    if K % 2 != 0:
        raise ValueError(f"K ({K}) must be even for uint4 packing")
    os.makedirs(out_dir, exist_ok=True)

    zp_str = "no-zeros" if no_zeros else "with-zeros"
    print(f"  bits=4  M={M} N={N} K={K} gs={group_size} groups={num_groups_k} "
          f"({zp_str}, seed={seed})")

    np.random.seed(seed)

    A = np.random.uniform(-0.5, 0.5, (M, K)).astype(np.float16)
    B_u4 = np.random.randint(0, 16, (N, K), dtype=np.uint8)
    B_packed = (B_u4[:, 0::2] | (B_u4[:, 1::2] << 4)).astype(np.uint8)
    scales = np.random.uniform(0.01, 0.05, (N, num_groups_k)).astype(np.float16)
    zeros = None
    if not no_zeros:
        zeros = np.random.uniform(7.0, 9.0, (N, num_groups_k)).astype(np.float16)

    A.flatten(order='C').tofile(os.path.join(out_dir, "matmul_nbits_A.bin"))
    B_packed.flatten(order='C').tofile(os.path.join(out_dir, "matmul_nbits_B_packed.bin"))
    scales.flatten(order='C').tofile(os.path.join(out_dir, "matmul_nbits_scales.bin"))
    if zeros is not None:
        zeros.flatten(order='C').tofile(os.path.join(out_dir, "matmul_nbits_zeros.bin"))

    if not no_ref:
        print("  Computing reference C...", end=" ", flush=True)
        t0 = time.time()
        group_idx = np.arange(K) // group_size
        scales_f32 = scales.astype(np.float32)
        if zeros is not None:
            B_dq = (B_u4.astype(np.float32) - zeros.astype(np.float32)[:, group_idx]) \
                   * scales_f32[:, group_idx]
        else:
            B_dq = (B_u4.astype(np.float32) - 8.0) * scales_f32[:, group_idx]
        C_ref = (A.astype(np.float32) @ B_dq.T).astype(np.float16)
        C_ref.flatten(order='C').tofile(os.path.join(out_dir, "matmul_nbits_C_ref.bin"))
        elapsed = time.time() - t0
        gflops = 2.0 * M * N * K / (max(elapsed, 1e-9) * 1e9)
        print(f"done ({elapsed:.2f}s, {gflops:.1f} GFLOPS)")
    else:
        print("  Skipping reference (--no-ref)")

    with open(os.path.join(out_dir, "matmul_nbits_meta.txt"), 'w') as f:
        f.write(f"M={M}\nN={N}\nK={K}\ngroup_size={group_size}\n"
                f"num_groups_k={num_groups_k}\nseed={seed}\n"
                f"use_zeros={'true' if zeros is not None else 'false'}\n")
    print(f"  Data saved to {out_dir}/")


def _generate_u3(M, K, N, group_size, no_zeros, no_ref, seed, out_dir):
    num_groups_k = (K + group_size - 1) // group_size
    if K % 32 != 0:
        raise ValueError(f"K ({K}) must be a multiple of 32 for bits=3")
    os.makedirs(out_dir, exist_ok=True)

    zp_str = "no-zeros" if no_zeros else "with-zeros"
    print(f"  bits=3  M={M} N={N} K={K} gs={group_size} groups={num_groups_k} "
          f"({zp_str}, seed={seed})")

    np.random.seed(seed)

    A = np.random.uniform(-0.5, 0.5, (M, K)).astype(np.float16)
    A.flatten(order='C').tofile(os.path.join(out_dir, "matmul_nbits_A.bin"))

    group_idx = np.arange(K) // group_size

    # --- uint3 ---
    B_u3 = np.random.randint(0, 8, (N, K), dtype=np.uint8)
    B_u3_packed = _pack_u3_continuous(B_u3)
    if not np.array_equal(B_u3, _unpack_u3_continuous(B_u3_packed, K)):
        raise RuntimeError("uint3 pack/unpack round-trip mismatch -- "
                           "_pack_u3_continuous has a bug")

    scales_u3 = np.random.uniform(0.01, 0.05, (N, num_groups_k)).astype(np.float16)
    zeros_u3 = zeros_u3_packed = None
    if not no_zeros:
        zeros_u3 = np.random.randint(3, 6, (N, num_groups_k)).astype(np.uint8)
        # ONNX bits=3 packs zero_points as the same continuous 3-bit stream as weights.
        zeros_u3_packed = _pack_u3_continuous(zeros_u3)

    B_u3_packed.flatten(order='C').tofile(os.path.join(out_dir, "matmul_nbits_u3_B.bin"))
    scales_u3.flatten(order='C').tofile(os.path.join(out_dir, "matmul_nbits_u3_scales.bin"))
    if zeros_u3 is not None:
        zeros_u3.flatten(order='C').tofile(os.path.join(out_dir, "matmul_nbits_u3_zeros.bin"))
        zeros_u3_packed.flatten(order='C').tofile(
            os.path.join(out_dir, "matmul_nbits_u3_zeros_packed.bin"))

    # --- uint4 (paired for benchmark) ---
    B_u4 = np.random.randint(0, 16, (N, K), dtype=np.uint8)
    B_u4_packed = (B_u4[:, 0::2] | (B_u4[:, 1::2] << 4)).astype(np.uint8)
    scales_u4 = np.random.uniform(0.01, 0.05, (N, num_groups_k)).astype(np.float16)
    zeros_u4_u8 = zeros_u4_fp16 = None
    if not no_zeros:
        zeros_u4_u8 = np.random.randint(7, 10, (N, num_groups_k), dtype=np.uint8)
        zeros_u4_fp16 = zeros_u4_u8.astype(np.float16)

    B_u4_packed.flatten(order='C').tofile(os.path.join(out_dir, "matmul_nbits_u4_B_packed.bin"))
    scales_u4.flatten(order='C').tofile(os.path.join(out_dir, "matmul_nbits_u4_scales.bin"))
    if zeros_u4_u8 is not None:
        zeros_u4_u8.flatten(order='C').tofile(
            os.path.join(out_dir, "matmul_nbits_u4_zeros_u8.bin"))
        zeros_u4_fp16.flatten(order='C').tofile(
            os.path.join(out_dir, "matmul_nbits_u4_zeros_fp16.bin"))

    # --- references ---
    if not no_ref:
        print("  Computing references...", end=" ", flush=True)
        t0 = time.time()

        sf32 = scales_u3.astype(np.float32)
        if zeros_u3 is not None:
            B_u3_dq = (B_u3.astype(np.float32) - zeros_u3.astype(np.float32)[:, group_idx]) \
                * sf32[:, group_idx]
        else:
            B_u3_dq = (B_u3.astype(np.float32) - 4.0) * sf32[:, group_idx]
        (A.astype(np.float32) @ B_u3_dq.T).astype(np.float16).flatten(order='C').tofile(
            os.path.join(out_dir, "matmul_nbits_u3_C_ref.bin"))

        sf32 = scales_u4.astype(np.float32)
        if zeros_u4_u8 is not None:
            B_u4_dq = (B_u4.astype(np.float32) - zeros_u4_u8.astype(np.float32)[:, group_idx]) \
                * sf32[:, group_idx]
        else:
            B_u4_dq = (B_u4.astype(np.float32) - 8.0) * sf32[:, group_idx]
        (A.astype(np.float32) @ B_u4_dq.T).astype(np.float16).flatten(order='C').tofile(
            os.path.join(out_dir, "matmul_nbits_u4_C_ref.bin"))

        elapsed = time.time() - t0
        print(f"done ({elapsed:.2f}s)")
    else:
        print("  Skipping reference (--no-ref)")

    row_bytes_u3 = (K * 3 + 7) // 8
    print(f"  B size: uint3={N * row_bytes_u3} bytes, uint4={N * K // 2} bytes "
          f"({100.0 * row_bytes_u3 / (K // 2):.1f}% of uint4)")

    with open(os.path.join(out_dir, "matmul_nbits_meta.txt"), 'w') as f:
        f.write(f"M={M}\nN={N}\nK={K}\ngroup_size={group_size}\n"
                f"num_groups_k={num_groups_k}\nseed={seed}\n"
                f"use_zeros={'true' if not no_zeros else 'false'}\n")
    print(f"  Data saved to {out_dir}/")


def _generate_u2(M, K, N, group_size, no_zeros, no_ref, seed, out_dir):
    num_groups_k = (K + group_size - 1) // group_size
    if K % 32 != 0:
        raise ValueError(f"K ({K}) must be a multiple of 32 for bits=2")
    os.makedirs(out_dir, exist_ok=True)

    zp_str = "no-zeros" if no_zeros else "with-zeros"
    print(f"  bits=2  M={M} N={N} K={K} gs={group_size} groups={num_groups_k} "
          f"({zp_str}, seed={seed})")

    np.random.seed(seed)

    A = np.random.uniform(-0.5, 0.5, (M, K)).astype(np.float16)
    A.flatten(order='C').tofile(os.path.join(out_dir, "matmul_nbits_A.bin"))

    group_idx = np.arange(K) // group_size

    # --- uint2 ---
    B_u2 = np.random.randint(0, 4, (N, K), dtype=np.uint8)
    B_u2_packed = _pack_2bit(B_u2)
    if not np.array_equal(B_u2, _unpack_2bit(B_u2_packed, K)):
        raise RuntimeError("uint2 pack/unpack round-trip mismatch -- "
                           "_pack_2bit has a bug")

    scales_u2 = np.random.uniform(0.01, 0.05, (N, num_groups_k)).astype(np.float16)
    zeros_u2 = zeros_u2_packed = None
    if not no_zeros:
        # 2-bit zero_points are in 0..3; default (symmetric) zp is 2.
        zeros_u2 = np.random.randint(1, 3, (N, num_groups_k)).astype(np.uint8)
        zeros_u2_packed = _pack_2bit(zeros_u2)

    B_u2_packed.flatten(order='C').tofile(os.path.join(out_dir, "matmul_nbits_u2_B.bin"))
    scales_u2.flatten(order='C').tofile(os.path.join(out_dir, "matmul_nbits_u2_scales.bin"))
    if zeros_u2 is not None:
        zeros_u2.flatten(order='C').tofile(
            os.path.join(out_dir, "matmul_nbits_u2_zeros_u8.bin"))
        zeros_u2_packed.flatten(order='C').tofile(
            os.path.join(out_dir, "matmul_nbits_u2_zeros_packed.bin"))

    # --- uint4 (paired for benchmark) ---
    B_u4 = np.random.randint(0, 16, (N, K), dtype=np.uint8)
    B_u4_packed = (B_u4[:, 0::2] | (B_u4[:, 1::2] << 4)).astype(np.uint8)
    scales_u4 = np.random.uniform(0.01, 0.05, (N, num_groups_k)).astype(np.float16)
    zeros_u4_u8 = zeros_u4_fp16 = None
    if not no_zeros:
        zeros_u4_u8 = np.random.randint(7, 10, (N, num_groups_k), dtype=np.uint8)
        zeros_u4_fp16 = zeros_u4_u8.astype(np.float16)

    B_u4_packed.flatten(order='C').tofile(os.path.join(out_dir, "matmul_nbits_u4_B_packed.bin"))
    scales_u4.flatten(order='C').tofile(os.path.join(out_dir, "matmul_nbits_u4_scales.bin"))
    if zeros_u4_u8 is not None:
        zeros_u4_u8.flatten(order='C').tofile(
            os.path.join(out_dir, "matmul_nbits_u4_zeros_u8.bin"))
        zeros_u4_fp16.flatten(order='C').tofile(
            os.path.join(out_dir, "matmul_nbits_u4_zeros_fp16.bin"))

    # --- references ---
    if not no_ref:
        print("  Computing references...", end=" ", flush=True)
        t0 = time.time()

        sf32 = scales_u2.astype(np.float32)
        if zeros_u2 is not None:
            B_u2_dq = (B_u2.astype(np.float32) - zeros_u2.astype(np.float32)[:, group_idx]) \
                * sf32[:, group_idx]
        else:
            B_u2_dq = (B_u2.astype(np.float32) - 2.0) * sf32[:, group_idx]
        (A.astype(np.float32) @ B_u2_dq.T).astype(np.float16).flatten(order='C').tofile(
            os.path.join(out_dir, "matmul_nbits_u2_C_ref.bin"))

        sf32 = scales_u4.astype(np.float32)
        if zeros_u4_u8 is not None:
            B_u4_dq = (B_u4.astype(np.float32) - zeros_u4_u8.astype(np.float32)[:, group_idx]) \
                * sf32[:, group_idx]
        else:
            B_u4_dq = (B_u4.astype(np.float32) - 8.0) * sf32[:, group_idx]
        (A.astype(np.float32) @ B_u4_dq.T).astype(np.float16).flatten(order='C').tofile(
            os.path.join(out_dir, "matmul_nbits_u4_C_ref.bin"))

        elapsed = time.time() - t0
        print(f"done ({elapsed:.2f}s)")
    else:
        print("  Skipping reference (--no-ref)")

    print(f"  B size: uint2={N * K // 4} bytes, uint4={N * K // 2} bytes "
          f"({100.0 * (K // 4) / (K // 2):.1f}% of uint4)")

    with open(os.path.join(out_dir, "matmul_nbits_meta.txt"), 'w') as f:
        f.write(f"M={M}\nN={N}\nK={K}\ngroup_size={group_size}\n"
                f"num_groups_k={num_groups_k}\nseed={seed}\n"
                f"use_zeros={'true' if not no_zeros else 'false'}\n")
    print(f"  Data saved to {out_dir}/")


_GENERATORS = {4: _generate_u4, 3: _generate_u3, 2: _generate_u2}
_REQUIRES_K_ALIGN = {4: False, 3: True, 2: True}


# ============================================================
# Model sweep
# ============================================================

def _run_model_sweep(model_json, bits, group_size, no_zeros, no_ref, out_dir):
    with open(model_json) as f:
        cfg = json.load(f)

    M_array = cfg['M_array']
    K_array = cfg['KN_pairs']['K']
    N_array = cfg['KN_pairs']['N']

    if len(K_array) != len(N_array):
        print(f"ERROR: K ({len(K_array)}) and N ({len(N_array)}) arrays must have "
              f"the same length", file=sys.stderr)
        sys.exit(1)

    k_align = _REQUIRES_K_ALIGN[bits]
    total = len(K_array) * len(M_array)
    print(f"Model: {model_json}  bits={bits}")
    print(f"  {len(M_array)} M values x {len(K_array)} KN pairs = {total} shapes")
    print(f"  group_size={group_size}  {'no-zeros' if no_zeros else 'with-zeros'}")
    print()

    idx = 0
    skipped = []
    gen = _GENERATORS[bits]

    for k, n in zip(K_array, N_array):
        for m in M_array:
            idx += 1
            shape = f"{m}x{k}x{n}"
            if k_align and k % 32 != 0:
                print(f"[{idx}/{total}] Skipping {shape}: K={k} not a multiple of 32")
                skipped.append(shape)
                continue
            out = os.path.join(out_dir, shape)
            print(f"[{idx}/{total}] Generating {shape} -> {out}/")
            gen(m, k, n, group_size, no_zeros, no_ref, seed=42, out_dir=out)

    done = total - len(skipped)
    skip_note = f" ({len(skipped)} skipped: K not %32)" if skipped else ""
    print(f"\nDone: {done} shapes generated in {out_dir}/{skip_note}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate MatMulNBits test data (bits=2/3/4, single shape or model sweep)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gen_data.py 128x4096x4096 --bits 4                                    # single shape
  python gen_data.py 128x4096x4096 --bits 3 --no-zeros
  python gen_data.py 128x4096x4096 --bits 2
  python gen_data.py --model gemm_fp16u4/test_model/gpt_oss_20b.json --bits 4  # model sweep
  python gen_data.py --model gemm_fp16u4/test_model/gpt_oss_20b.json --bits 3 --no-zeros
""")
    parser.add_argument('size', nargs='?', type=str,
                        help='Matrix size MxKxN for single-shape mode')
    parser.add_argument('--model', type=str, metavar='JSON',
                        help='Model config JSON for sweep mode (overrides size)')
    parser.add_argument('--bits', type=int, choices=[2, 3, 4], default=4,
                        help='Weight quantization bit-width (default: 4)')
    parser.add_argument('--group-size', type=int, default=128)
    parser.add_argument('--no-zeros', action='store_true')
    parser.add_argument('--no-ref', action='store_true',
                        help='Skip computing reference C (faster)')
    parser.add_argument('--dir', type=str, default='data',
                        help='Output directory for single-shape mode (default: data/)')
    parser.add_argument('--out-dir', type=str, default='data_model',
                        help='Root output directory for model sweep (default: data_model/)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.model:
        _run_model_sweep(args.model, args.bits, args.group_size, args.no_zeros,
                         args.no_ref, args.out_dir)
    elif args.size:
        parts = args.size.split('x')
        if len(parts) != 3:
            parser.error(f"size must be MxKxN (got '{args.size}')")
        M, K, N = int(parts[0]), int(parts[1]), int(parts[2])
        _GENERATORS[args.bits](M, K, N, args.group_size, args.no_zeros, args.no_ref,
                               args.seed, args.dir)
    else:
        parser.error("Provide a size (MxKxN) or --model JSON")


if __name__ == '__main__':
    main()
