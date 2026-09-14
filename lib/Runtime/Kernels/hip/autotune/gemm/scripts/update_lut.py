#!/usr/bin/env python3
"""Gemm autotune LUT pipeline: build (CSV -> json) and compile (json -> fb).

    python update_lut.py build    # sweep CSV -> lut/<arch>.json
    python update_lut.py compile  # lut/<arch>.json -> lut/<arch>.fb (needs flatc)

Mirrors ../matmul_nbits/scripts/update_lut.py, trimmed to the pieces the Gemm
table needs. The winners come from a cooled fair-bench sweep (NOT the thermally
corrupted online-autotune loop -- see the skill's timing notes): each row is the
tile that a trustworthy per-shape measurement chose. `build` reads that CSV;
`compile` turns the json into the .fb; CMake embeds the .fb's bytes into the
custom_kernels DLL at configure time (no generated .cpp is committed).

Distance weights are fixed at 1.0 here; refit later (as matmul_nbits does) once
enough measured points exist. The lookup metric lives in gemm_autotune.cpp and
must stay in step with these weights.

CSV columns expected by `build` (produced by the cooled LUT sweep):
    phase,type_bytes,trans_b,M,N,K,bm,bn,wt_m,wt_n,swizzle,split_k,bk,threads,tile_n
  phase in {Wmma,GemvNt}; type_bytes in {2,4,8}; trans_b in {0,1}. For a Wmma
  row the gemv columns are 0 and vice-versa.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # autotune/gemm
FBS = ROOT / "gemm_autotune.fbs"
LUT_DIR = ROOT / "lut"

SCHEMA_VERSION = 1
KERNEL_ABI = "gemm-wmma-v1"

TYPE_BYTES = {2: "B2", 4: "B4", 8: "B8"}
TRANS_B = {0: "NN", 1: "NT"}


def config_entry(row: dict) -> dict:
    """One CSV row -> a GemmTuneConfig dict (kind + payload)."""
    if row["phase"] == "Wmma":
        return {"kind": "Wmma",
                "bm16": int(row["bm"]) // 16, "bn16": int(row["bn"]) // 16,
                "swizzle": int(row["swizzle"]), "wt_m": int(row["wt_m"]),
                "wt_n": int(row["wt_n"]), "bk": int(row["bk"]),
                "split_k": int(row.get("split_k", 1) or 1),
                "threads": 0, "tile_n": 0}
    return {"kind": "Gemv", "bm16": 0, "bn16": 0, "swizzle": 0, "wt_m": 0,
            "wt_n": 0, "bk": 0, "split_k": 0,
            "threads": int(row["threads"]), "tile_n": int(row["tile_n"])}


def config_key(c: dict):
    return tuple(sorted(c.items()))


def cmd_build(args) -> int:
    src = Path(args.csv)
    if not src.exists():
        print(f"[build] no {src}", file=sys.stderr)
        return 1
    pool, index, points = [], {}, []

    def config_index(c: dict) -> int:
        k = config_key(c)
        if k not in index:
            index[k] = len(pool)
            pool.append(c)
        return index[k]

    with open(src, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            phase = row["phase"]
            if phase not in ("Wmma", "GemvNt"):
                continue
            tb = TYPE_BYTES.get(int(row["type_bytes"]))
            trb = TRANS_B.get(int(row["trans_b"]))
            if tb is None or trb is None:
                continue
            m, n, k = int(row["M"]), int(row["N"]), int(row["K"])
            if m <= 0 or n <= 0 or k <= 0:
                continue
            points.append({
                "phase": "Wmma" if phase == "Wmma" else "GemvNt",
                "type_bytes": tb, "trans_b": trb,
                "config": config_index(config_entry(row)),
                "m": m, "n": n, "k": k,
            })

    if len(pool) > 256:
        print(f"[build] {len(pool)} configs exceeds the ubyte index",
              file=sys.stderr)
        return 1

    doc = {
        "schema_version": SCHEMA_VERSION, "gpu_arch": args.arch,
        "rocm_version": args.rocm_version, "kernel_abi": KERNEL_ABI,
        "model_key": args.model_key,
        "weight_m": args.weight_m, "weight_n": args.weight_n,
        "weight_k": args.weight_k,
    }
    LUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else LUT_DIR / f"{args.arch}.json"
    with open(out, "w", encoding="utf-8") as f:
        f.write("{\n")
        for key in doc:
            f.write(f' "{key}": {json.dumps(doc[key])},\n')
        f.write(' "configs": [\n')
        f.write(",\n".join("  " + json.dumps(c, sort_keys=True) for c in pool))
        f.write("\n ],\n \"fallbacks\": [],\n \"points\": [\n")
        f.write(",\n".join("  " + json.dumps(p, sort_keys=True) for p in
                           sorted(points, key=lambda r: (r["phase"], r["type_bytes"],
                                                         r["trans_b"], r["n"], r["k"], r["m"]))))
        f.write("\n ]\n}\n")
    print(f"[build] {len(points)} points, {len(pool)} configs -> {out}")
    return 0


def cmd_compile(args) -> int:
    lut_json = LUT_DIR / f"{args.arch}.json"
    if not lut_json.exists():
        print(f"[compile] no {lut_json}; run build first", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [args.flatc, "--binary", "--strict-json", "-o", tmp, str(FBS),
               str(lut_json)]
        print("[compile] " + " ".join(cmd))
        if subprocess.run(cmd).returncode != 0:
            return 1
        produced = list(Path(tmp).glob("*.bin")) + list(Path(tmp).glob("*.fb"))
        if not produced:
            print("[compile] flatc produced nothing", file=sys.stderr)
            return 1
        dst = LUT_DIR / f"{args.arch}.fb"
        shutil.copy(produced[0], dst)
    # Only the .fb is committed; the linkable kGemmLutData[] byte array is
    # generated from it at CMake configure time (file(READ ... HEX)), so no
    # generated .cpp is committed. Mirrors matmul_nbits / gqa.
    print(f"[compile] -> {dst} ({dst.stat().st_size} bytes)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build", "compile"])
    ap.add_argument("--arch", default="gfx1151")
    ap.add_argument("--csv", default=str(LUT_DIR / "gfx1151_points.csv"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--flatc", default="flatc")
    ap.add_argument("--rocm-version", type=int, default=70400)
    ap.add_argument("--model-key", default="oga_models fp16")
    ap.add_argument("--weight-m", type=float, default=1.0)
    ap.add_argument("--weight-n", type=float, default=1.0)
    ap.add_argument("--weight-k", type=float, default=1.0)
    args = ap.parse_args()
    if args.command == "build":
        return cmd_build(args)
    return cmd_compile(args)


if __name__ == "__main__":
    raise SystemExit(main())
