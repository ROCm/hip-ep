#!/usr/bin/env python3
"""Write Q/K/V inputs for one hip_gqa_flash_prefill (fp16 KV) shape.

Matches the CLI shape flags of test_gqa_prefill.cpp (--b/--h/--g/--d/--sq
/--past). `make test` / `make test_custom` call this before running the exe
with --data-dir pointing at the output directory. K/V cache is sized to
total_seq = past + sq, matching the cpp's `max_seq = past_len + sq`.
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=1)
    ap.add_argument("--h", type=int, default=8)
    ap.add_argument("--g", type=int, default=8)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--sq", type=int, default=512)
    ap.add_argument("--past", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dir", default="data")
    args = ap.parse_args()

    b, h, g, d, sq, past = args.b, args.h, args.g, args.d, args.sq, args.past
    max_seq = past + sq
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.dir, exist_ok=True)

    Q = rng.uniform(-1.0, 1.0, size=(b * sq * h * d,)).astype("<f4")
    K = rng.uniform(-1.0, 1.0, size=(b * g * max_seq * d,)).astype("<f4")
    V = rng.uniform(-1.0, 1.0, size=(b * g * max_seq * d,)).astype("<f4")

    Q.tofile(os.path.join(args.dir, "Q.bin"))
    K.tofile(os.path.join(args.dir, "K.bin"))
    V.tofile(os.path.join(args.dir, "V.bin"))

    meta = {"B": b, "H": h, "G": g, "D": d, "sq": sq, "past": past, "seed": args.seed}
    with open(os.path.join(args.dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
