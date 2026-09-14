#!/usr/bin/env python3

#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""
Verify true-data folder: reload inputs, compute C = A @ dequant(B)^T in NumPy,
compare against the provided reference output and GPU output.

Usage:
    python verify_true_data.py true_data/layer_0/o_proj
"""

import numpy as np
import json
import os
import sys
import time


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <true_data_folder>")
        sys.exit(1)

    folder = sys.argv[1]
    json_path = os.path.join(folder, "shape.json")
    with open(json_path, 'r') as f:
        cfg = json.load(f)

    M = cfg["M"]
    K = cfg["K"]
    N = cfg["N"]
    block_size = cfg["block_size"]
    no_zeros = cfg.get("no_zeros", True)
    batch_size = cfg.get("batch_size", 1)
    num_groups_k = (K + block_size - 1) // block_size

    print(f"Shape: M={M} K={K} N={N} block_size={block_size} "
          f"groups={num_groups_k} no_zeros={no_zeros} batch={batch_size}")

    # ---- Load A (ifm) ----
    ifm_file = cfg["ifm_disc"]["file_name"]
    A_path = os.path.join(folder, ifm_file)
    A_raw = np.fromfile(A_path, dtype=np.float16)
    print(f"  A ({ifm_file}): {A_raw.shape[0]} elements, "
          f"nan={np.isnan(A_raw).sum()}, inf={np.isinf(A_raw).sum()}, "
          f"range=[{np.nanmin(A_raw):.4f}, {np.nanmax(A_raw):.4f}]")
    A = A_raw.reshape(batch_size, M, K)

    # ---- Load B (wts, uint4 packed as uint8) ----
    wts_file = cfg["wts_disc"]["file_name"]
    B_path = os.path.join(folder, wts_file)
    B_packed_raw = np.fromfile(B_path, dtype=np.uint8)
    print(f"  B ({wts_file}): {B_packed_raw.shape[0]} bytes")

    # Unpack uint4: low nibble = even k, high nibble = odd k
    B_packed = B_packed_raw.reshape(N, K // 2)
    B_low  = (B_packed & 0x0F).astype(np.uint8)
    B_high = ((B_packed >> 4) & 0x0F).astype(np.uint8)
    B_uint4 = np.empty((N, K), dtype=np.uint8)
    B_uint4[:, 0::2] = B_low
    B_uint4[:, 1::2] = B_high
    print(f"  B unpacked: shape={B_uint4.shape}, "
          f"range=[{B_uint4.min()}, {B_uint4.max()}]")

    # ---- Load scales ----
    scales_file = cfg["scales_disc"]["file_name"]
    S_path = os.path.join(folder, scales_file)
    S_raw = np.fromfile(S_path, dtype=np.float16)
    scales = S_raw.reshape(N, num_groups_k)
    print(f"  scales ({scales_file}): shape={scales.shape}, "
          f"nan={np.isnan(S_raw).sum()}, "
          f"range=[{np.nanmin(S_raw):.6f}, {np.nanmax(S_raw):.6f}]")

    # ---- Dequantize B ----
    print("  Dequantizing B...", end=" ", flush=True)
    group_idx = np.arange(K) // block_size
    scales_f32 = scales.astype(np.float32)

    if no_zeros:
        B_dq = B_uint4.astype(np.float32) * scales_f32[:, group_idx]
    else:
        zeros_file = cfg.get("zeros_disc", {}).get("file_name", "")
        if zeros_file:
            Z_path = os.path.join(folder, zeros_file)
            Z_raw = np.fromfile(Z_path, dtype=np.float16)
            zeros = Z_raw.reshape(N, num_groups_k)
            zeros_f32 = zeros.astype(np.float32)
            print(f"zeros range=[{np.nanmin(Z_raw):.4f}, {np.nanmax(Z_raw):.4f}]", end=" ")
            B_dq = (B_uint4.astype(np.float32) - zeros_f32[:, group_idx]) * scales_f32[:, group_idx]
        else:
            B_dq = B_uint4.astype(np.float32) * scales_f32[:, group_idx]

    print(f"done, B_dq range=[{np.nanmin(B_dq):.4f}, {np.nanmax(B_dq):.4f}], "
          f"nan={np.isnan(B_dq).sum()}")

    # ---- Compute C = A @ B_dq^T ----
    print("  Computing C = A @ dequant(B)^T ...", end=" ", flush=True)
    t0 = time.time()
    A_f32 = A[0].astype(np.float32)   # [M, K]
    C_f32 = A_f32 @ B_dq.T            # [M, N]
    C_py  = C_f32.astype(np.float16)
    elapsed = time.time() - t0
    print(f"done ({elapsed:.3f}s)")

    print(f"  C_python: shape={C_py.shape}, "
          f"nan={np.isnan(C_py).sum()}, inf={np.isinf(C_py).sum()}, "
          f"range=[{np.nanmin(C_py):.4f}, {np.nanmax(C_py):.4f}]")

    # ---- Sample values ----
    print("\n  Sample C_python values:")
    total = min(10, M * N)
    for i in range(total):
        m, n = divmod(i, N)
        print(f"    [{i}] (m={m},n={n}) = {float(C_py[m, n]):.6f}")

    # ---- Compare with reference output ----
    out_file = cfg.get("output_disc", {}).get("file_name", "")
    if out_file:
        out_path = os.path.join(folder, out_file)
        if os.path.exists(out_path):
            C_ref_raw = np.fromfile(out_path, dtype=np.float16)
            C_ref = C_ref_raw.reshape(batch_size, M, N)[0]  # [M, N]
            print(f"\n  Reference ({out_file}): shape={C_ref.shape}, "
                  f"nan={np.isnan(C_ref).sum()}, inf={np.isinf(C_ref).sum()}, "
                  f"range=[{np.nanmin(C_ref):.4f}, {np.nanmax(C_ref):.4f}]")

            print("  Sample reference values:")
            for i in range(total):
                m, n = divmod(i, N)
                print(f"    [{i}] (m={m},n={n}) ref={float(C_ref[m, n]):.6f}  "
                      f"py={float(C_py[m, n]):.6f}  "
                      f"diff={abs(float(C_py[m, n]) - float(C_ref[m, n])):.6f}")

            # Error stats (skip NaN elements)
            valid = ~(np.isnan(C_py.flatten()) | np.isnan(C_ref.flatten()))
            if valid.sum() > 0:
                diff = np.abs(C_py.flatten()[valid].astype(np.float32)
                              - C_ref.flatten()[valid].astype(np.float32))
                print(f"\n  Valid elements: {valid.sum()}/{M*N}")
                print(f"  Max abs diff: {diff.max():.6f}, mean: {diff.mean():.6f}")
            else:
                print(f"\n  All {M*N} elements are NaN — cannot compute diff")
        else:
            print(f"\n  Reference file {out_path} not found, skipping comparison")

    print("\nDone.")


if __name__ == '__main__':
    main()
