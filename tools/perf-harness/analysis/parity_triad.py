#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Cross-compare the three dumps flag_parity.ps1 writes per model.

Exists to answer one question the pairwise verdict cannot: when off-vs-on and
off-vs-control come out to the same number, is that a coincidence or are the
control and on runs actually identical? If they are, the divergence belongs to
the first run, not to the flag, and the flag is inert rather than merely quiet.
"""

import argparse
import glob
import os

import numpy as np

STATES = ("off", "control", "on")


def worst(a, b):
    c = [float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
         for x, y in zip(a, b)]
    return min(c) if c else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--flag", required=True)
    args = ap.parse_args()

    names = sorted({os.path.basename(p).split("." + args.flag)[0]
                    for p in glob.glob(os.path.join(args.dir, "*.npz"))})
    hdr = (f'{"model":46} {"off-ctl":>10} {"ctl-on":>10} {"off-on":>10}  '
           f'{"ctl==on":>8}')
    print(hdr)
    print("-" * len(hdr))
    for n in names:
        paths = {s: os.path.join(args.dir, f"{n}.{args.flag}.{s}.npz")
                 for s in STATES}
        if not all(os.path.exists(p) for p in paths.values()):
            continue
        d = {s: np.load(p)["logits"] for s, p in paths.items()}
        print(f'{n[:46]:46} {worst(d["off"], d["control"]):10.8f} '
              f'{worst(d["control"], d["on"]):10.8f} '
              f'{worst(d["off"], d["on"]):10.8f}  '
              f'{str(np.array_equal(d["control"], d["on"])):>8}')


if __name__ == "__main__":
    main()
