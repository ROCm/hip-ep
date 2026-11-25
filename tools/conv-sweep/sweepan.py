# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Summarise a sweep csv: per shape and batch, the default's time against the best forced configs.
usage: sweepan.py <shapes.txt> <sweep.csv> [--top 3]"""
import csv
import sys
from collections import defaultdict

shapes = {}
for line in open(sys.argv[1]):
    if line.strip() and not line.startswith("#"):
        f = line.split()
        shapes[f[0]] = dict(zip(("cin", "h", "w", "cout", "g", "k0", "k1", "s0", "s1"), f[1:10]))
top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 3
by = defaultdict(dict)
FIELDS = ("shape", "dtype", "B", "cfg", "blocks", "us", "relerr", "status")
for row in csv.reader(open(sys.argv[2])):  # sweep.exe writes no header row
    r = dict(zip(FIELDS, row))
    if row and row[0] != "shape" and r["status"] == "OK":
        by[(r["shape"], int(r["B"]))][r["cfg"]] = (float(r["us"]), int(r["blocks"]))
for (sid, batch), d in sorted(by.items()):
    s = shapes.get(sid)
    desc = f'{s["cin"]}x{s["h"]}x{s["w"]}->{s["cout"]} g{s["g"]} k{s["k0"]} s{s["s0"]}' if s else ""
    dflt = d.get("default", (float("nan"), -1))[0]
    best = sorted((v[0], k, v[1]) for k, v in d.items() if k != "default")[:top]
    if not best:
        continue
    print(f"{sid} B={batch} {desc:34} default {dflt:8.1f}us | "
          + " ".join(f"{k}({b}) {t:.1f}" for t, k, b in best) + f" | {dflt / best[0][0]:.2f}x")
