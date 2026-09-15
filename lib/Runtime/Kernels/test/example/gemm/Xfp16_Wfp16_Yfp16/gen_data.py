#!/usr/bin/env python3
"""Write A/B(/C) inputs for one hip_gemm shape.

Matches the positional CLI of test_gemm.cpp (M N K TA TB TYPE). `make test` /
`make test_custom` call this before running the exe with --data-dir pointing
at the output directory. C is only written when BETA != 0 (bias case).
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--ta", type=int, default=0)
    ap.add_argument("--tb", type=int, default=1)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--c0", type=int, default=1)
    ap.add_argument("--c1", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dir", default="data")
    args = ap.parse_args()

    m, n, k, ta, tb = args.m, args.n, args.k, args.ta, args.tb
    c1 = args.c1 if args.c1 else n
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.dir, exist_ok=True)

    a_n = k * m if ta else m * k
    b_n = n * k if tb else k * n
    A = rng.uniform(-1.0, 1.0, size=(a_n,)).astype("<f4")
    B = rng.uniform(-1.0, 1.0, size=(b_n,)).astype("<f4")
    A.tofile(os.path.join(args.dir, "A.bin"))
    B.tofile(os.path.join(args.dir, "B.bin"))

    if args.beta != 0.0:
        C = rng.uniform(-1.0, 1.0, size=(args.c0 * c1,)).astype("<f4")
        C.tofile(os.path.join(args.dir, "C.bin"))

    meta = {"M": m, "N": n, "K": k, "TA": ta, "TB": tb, "beta": args.beta, "seed": args.seed}
    with open(os.path.join(args.dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
