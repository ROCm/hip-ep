#!/usr/bin/env python3
"""
RoPE test data generator + NumPy reference

Generates random FP16 input, position_ids, cos/sin cache, and computes
the golden reference output using NumPy.

Data layout matches the hip_rope_forward API:
  input        : FP16 [batch, seq_len, num_heads, head_dim]
  position_ids : int64 [batch, seq_len]
  cos_cache    : FP16 [max_seq_len, half_rot]
  sin_cache    : FP16 [max_seq_len, half_rot]
  output_ref   : FP16 [batch, seq_len, num_heads, head_dim]

Usage:
    python gen_rope_data.py
    python gen_rope_data.py --batch 2 --seq 128 --heads 32 --head-dim 128
    python gen_rope_data.py --interleaved
    python gen_rope_data.py --rotary-dim 64 --head-dim 128
"""

import numpy as np
import argparse
import os
import time


def rope_reference(input_data, position_ids, cos_cache, sin_cache,
                   batch, seq_len, num_heads, head_dim, rotary_dim, interleaved):
    half_rot = rotary_dim // 2
    output = input_data.copy()

    for b in range(batch):
        for s in range(seq_len):
            pos = int(position_ids[b, s])
            cos_vals = cos_cache[pos]  # [half_rot]
            sin_vals = sin_cache[pos]  # [half_rot]

            for h in range(num_heads):
                head = input_data[b, s, h, :]

                if interleaved:
                    for d in range(half_rot):
                        x0 = float(head[2 * d])
                        x1 = float(head[2 * d + 1])
                        c = float(cos_vals[d])
                        sv = float(sin_vals[d])
                        output[b, s, h, 2 * d]     = np.float16(x0 * c - x1 * sv)
                        output[b, s, h, 2 * d + 1] = np.float16(x0 * sv + x1 * c)
                else:
                    for d in range(half_rot):
                        x0 = float(head[d])
                        x1 = float(head[d + half_rot])
                        c = float(cos_vals[d])
                        sv = float(sin_vals[d])
                        output[b, s, h, d]            = np.float16(x0 * c - x1 * sv)
                        output[b, s, h, d + half_rot] = np.float16(x0 * sv + x1 * c)

    return output


def main():
    parser = argparse.ArgumentParser(description='Generate RoPE test data')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--seq', type=int, default=128)
    parser.add_argument('--heads', type=int, default=32)
    parser.add_argument('--head-dim', type=int, default=128)
    parser.add_argument('--rotary-dim', type=int, default=0,
                        help='Rotary dim (default: head_dim)')
    parser.add_argument('--max-seq', type=int, default=2048)
    parser.add_argument('--interleaved', action='store_true')
    parser.add_argument('--dir', type=str, default='data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-ref', action='store_true')
    args = parser.parse_args()

    B = args.batch
    S = args.seq
    H = args.heads
    D = args.head_dim
    RD = args.rotary_dim if args.rotary_dim > 0 else D
    HR = RD // 2
    MAX_S = args.max_seq
    interleaved = args.interleaved

    os.makedirs(args.dir, exist_ok=True)

    mode_str = "interleaved" if interleaved else "half-rotated"
    print(f"Generating RoPE data: B={B} S={S} H={H} D={D} RD={RD} "
          f"max_seq={MAX_S} mode={mode_str} seed={args.seed}")

    np.random.seed(args.seed)

    input_data = np.random.uniform(-1.0, 1.0, (B, S, H, D)).astype(np.float16)
    position_ids = np.zeros((B, S), dtype=np.int64)
    for b in range(B):
        position_ids[b, :] = np.arange(S, dtype=np.int64)

    theta = 10000.0
    freqs = 1.0 / (theta ** (np.arange(0, HR, dtype=np.float64) * 2.0 / RD))
    positions = np.arange(MAX_S, dtype=np.float64)
    angles = np.outer(positions, freqs)  # [MAX_S, HR]
    cos_cache = np.cos(angles).astype(np.float16)
    sin_cache = np.sin(angles).astype(np.float16)

    input_data.tofile(os.path.join(args.dir, "rope_input.bin"))
    position_ids.tofile(os.path.join(args.dir, "rope_position_ids.bin"))
    cos_cache.tofile(os.path.join(args.dir, "rope_cos_cache.bin"))
    sin_cache.tofile(os.path.join(args.dir, "rope_sin_cache.bin"))

    if not args.no_ref:
        print("Computing reference...", end=" ", flush=True)
        t0 = time.time()
        output_ref = rope_reference(input_data, position_ids,
                                    cos_cache, sin_cache,
                                    B, S, H, D, RD, interleaved)
        elapsed = time.time() - t0
        output_ref.tofile(os.path.join(args.dir, "rope_output_ref.bin"))
        print(f"done ({elapsed:.2f}s)")
    else:
        print("Skipping reference (--no-ref)")

    meta_file = os.path.join(args.dir, "rope_meta.txt")
    with open(meta_file, 'w') as f:
        f.write(f"batch={B}\nseq_len={S}\nnum_heads={H}\nhead_dim={D}\n")
        f.write(f"rotary_dim={RD}\nmax_seq_len={MAX_S}\n")
        f.write(f"interleaved={'1' if interleaved else '0'}\n")
        f.write(f"seed={args.seed}\n")

    print(f"Data saved to {args.dir}/")


if __name__ == '__main__':
    main()
