#!/usr/bin/env python3
"""Write Q/K/V + int8 KV-cache inputs for one hip_gqa_flash_decode (int8 KV) shape.

Matches the CLI shape flags of test_gqa_decode_i8.cpp (--b/--h/--g/--d
/--max-seq/--total). `make test` / `make test_custom` call this before running
the exe with --data-dir pointing at the output directory.

K_i8/V_i8 use the same per-channel symmetric int8 scheme the kernel expects:
scale[g,e] = max_abs(K|V)[:, g, :eff, e] / 127, one scale per (kv_head, head_dim).
"""
import argparse
import json
import os

import numpy as np


def quantize(x, eff, b, g, s_len, d):
    x = x.reshape(b, g, s_len, d)
    amax = np.abs(x[:, :, :eff, :]).max(axis=(0, 2))  # [g, d]
    scale = np.where(amax > 0, amax, 1.0) / 127.0
    q = np.round(x / scale[None, :, None, :])
    q = np.clip(q, -128, 127).astype(np.int8)
    return q.reshape(-1), scale.astype("<f4").reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=1)
    ap.add_argument("--h", type=int, default=8)
    ap.add_argument("--g", type=int, default=8)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--total", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dir", default="data")
    args = ap.parse_args()

    b, h, g, d, max_seq, total = args.b, args.h, args.g, args.d, args.max_seq, args.total
    eff = min(total, max_seq)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.dir, exist_ok=True)

    Q = rng.uniform(-1.0, 1.0, size=(b * h * d,)).astype("<f4")
    K = rng.uniform(-1.0, 1.0, size=(b * g * max_seq * d,)).astype("<f4")
    V = rng.uniform(-1.0, 1.0, size=(b * g * max_seq * d,)).astype("<f4")

    K_i8, kscale = quantize(K, eff, b, g, max_seq, d)
    V_i8, vscale = quantize(V, eff, b, g, max_seq, d)

    Q.tofile(os.path.join(args.dir, "Q.bin"))
    K.tofile(os.path.join(args.dir, "K.bin"))
    V.tofile(os.path.join(args.dir, "V.bin"))
    K_i8.tofile(os.path.join(args.dir, "K_i8.bin"))
    V_i8.tofile(os.path.join(args.dir, "V_i8.bin"))
    kscale.tofile(os.path.join(args.dir, "kscale.bin"))
    vscale.tofile(os.path.join(args.dir, "vscale.bin"))

    meta = {"B": b, "H": h, "G": g, "D": d, "max_seq": max_seq, "total": total, "seed": args.seed}
    with open(os.path.join(args.dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
