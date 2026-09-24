#!/usr/bin/env python3
"""CK GEMM autotune LUT pipeline: extract -> (sweep) -> build -> compile.

    python update_lut.py extract --logs <HIPDNN_EP_DEBUG logs...> --out shapes/<name>.csv
    ck_gemm_autotune_sweep.exe <shapes.csv> <winners.csv>
    python update_lut.py build --winners <winners.csv>... --rocm-version <n>  # -> lut/<arch>.json
    python update_lut.py compile --flatc <flatc>             # -> lut/<arch>.fb

extract turns the per-op debug lines into hip_ck_gemm_run's column-major
arguments using the same operand swap each call site applies, so a shape in the
table keys exactly as the runtime asks for it. The winners come from
ckSelectGemmInstance itself (the sweep driver runs it in online mode); nothing
here decides which instance is fastest.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FBS = ROOT / "ck_gemm_autotune.fbs"
LUT_DIR = ROOT / "lut"

SCHEMA_VERSION = 1
KERNEL_ABI = "ck_gemm-v1"
TIE_MARGIN = 1e-6  # squared log2 distance

# hip_dtype_t
F32, F16 = 0, 1
DTYPE_NAME = {F16: "F16", F32: "F32"}

# gemm.cpp typeCode
TYPE_F16, TYPE_F32 = 0, 1

KEY = ("ab", "d", "trans_a", "bias", "m", "n", "k", "batch")
FIELDS = list(KEY) + ["lda", "ldb", "ldd", "stride_a", "stride_b", "stride_d",
                      "alpha"]

RE_GEMM = re.compile(
    r"wrap_gemm: M=(\d+), N=(\d+), K=(\d+), transA=(\d+), transB=(\d+), "
    r"alpha=([-\d.]+), beta=([-\d.]+), typeCode=(\d+), C=(\S+), "
    r"cDim0=(\d+), cDim1=(\d+)")
RE_MATMUL = re.compile(
    r"wrap_hipblasLtMatmul: M=(\d+), N=(\d+), K=(\d+), batch=(\d+), "
    r"b_batch_stride=(-?\d+), transA=(\d+), transB=(\d+), elem_size=(\d+)")
# inFp32 is optional; a line without it has fp16 operands.
RE_GQA = re.compile(
    r"\[GQA\] CK instance -?\d+ for m=(\d+) n=(\d+) k=(\d+) batch=(\d+) "
    r"transA=(\d)(?: inFp32=(\d))? outFp32=(\d)")
RE_MHA = re.compile(
    r"\[MHA\] CK instance -?\d+ for m=(\d+) n=(\d+) k=(\d+) batch=(\d+) "
    r"transA=(\d) outFp32=(\d)")


def is_null(ptr: str) -> bool:
    return ptr.strip("0x").strip("0") == "" or ptr in ("(nil)", "(null)")


def from_gemm(g) -> dict | None:
    M, N, K, ta, tb = (int(x) for x in g[:5])
    alpha, beta, type_code = float(g[5]), float(g[6]), int(g[7])
    has_c = not is_null(g[8])
    c0, c1 = int(g[9]), int(g[10])
    if type_code not in (TYPE_F16, TYPE_F32):
        return None
    bias = (has_c and beta == 1.0 and c0 == 1 and c1 == N
            and type_code == TYPE_F16)
    eligible = alpha == 1.0 and ta == 0 and (
        (type_code == TYPE_F16 and (not has_c or bias))
        or type_code == TYPE_F32)
    if not eligible:
        return None
    dt = F16 if type_code == TYPE_F16 else F32
    return dict(m=N, n=M, k=K, batch=1, trans_a=tb, ab=dt, d=dt,
                bias=int(bias), lda=K if tb else N, ldb=K, ldd=N,
                stride_a=0, stride_b=0, stride_d=0, alpha=1.0)


def from_matmul(g) -> dict | None:
    M, N, K, batch, b_stride, ta, tb, es = (int(x) for x in g)
    if ta != 0 or es not in (2, 4):
        return None
    dt = F16 if es == 2 else F32
    return dict(m=N, n=M, k=K, batch=batch, trans_a=tb, ab=dt, d=dt, bias=0,
                lda=K if tb else N, ldb=K, ldd=N, stride_a=b_stride,
                stride_b=M * K, stride_d=M * N, alpha=1.0)


def attention_row(m, n, k, batch, ta, ab, d) -> dict:
    # The debug line omits the no-expand batch strides; dense ones stand in.
    return dict(m=m, n=n, k=k, batch=batch, trans_a=ta, ab=ab, d=d, bias=0,
                lda=k if ta else m, ldb=k, ldd=m, stride_a=m * k,
                stride_b=n * k, stride_d=n * m, alpha=1.0)


def from_gqa(g) -> dict | None:
    m, n, k, batch, ta, in32, out32 = (int(x or 0) for x in g)
    ab = F32 if in32 else F16
    d = F32 if out32 else F16
    if in32 and not out32:
        return None
    return attention_row(m, n, k, batch, ta, ab, d)


def from_mha(g) -> dict | None:
    m, n, k, batch, ta, out32 = (int(x) for x in g)
    return attention_row(m, n, k, batch, ta, F16, F32 if out32 else F16)


PARSERS = [("wrap_gemm:", RE_GEMM, from_gemm),
           ("wrap_hipblasLtMatmul:", RE_MATMUL, from_matmul),
           ("[GQA] CK instance", RE_GQA, from_gqa),
           ("[MHA] CK instance", RE_MHA, from_mha)]


def cmd_extract(args) -> int:
    rows: dict[tuple, dict] = {}
    for log in args.logs:
        seen = 0
        with open(log, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                for tag, rx, conv in PARSERS:
                    if tag not in line:
                        continue
                    mt = rx.search(line)
                    row = conv(mt.groups()) if mt else None
                    if row is None:
                        break
                    key = tuple(row[c] for c in KEY)
                    if key in rows:
                        rows[key]["calls"] += 1
                    else:
                        row["calls"] = 1
                        rows[key] = row
                        seen += 1
                    break
        print(f"[extract] {log}: {seen} new shapes")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows.values(), key=lambda r: tuple(r[c] for c in KEY))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + ["calls"])
        w.writeheader()
        w.writerows(ordered)
    print(f"[extract] {len(ordered)} shapes -> {out}")
    return 0


def align_class(x: int) -> int:
    return 3 if x % 8 == 0 else 2 if x % 4 == 0 else 1 if x % 2 == 0 else 0


def prune(points: dict[tuple, str]) -> dict[tuple, str]:
    """Drops every point whose removal leaves the nearest-point answer of every
    measured shape unchanged, so the pruned table resolves each of them exactly
    as the full one does. Mirrors the resolver: a measured shape is answered
    from its (partition, m/n/k alignment) group, which is never emptied, by
    squared log2 distance over (m, n, k, batch), ties to the earlier point in
    key order."""
    groups: dict[tuple, list[tuple]] = {}
    for key in sorted(points):
        m, n, k = key[4:7]
        group = key[:4] + (align_class(m), align_class(n), align_class(k))
        groups.setdefault(group, []).append(key)
    kept_all: dict[tuple, str] = {}
    for keys in groups.values():
        logs = [[math.log2(v) for v in key[4:]] for key in keys]
        name = [points[key] for key in keys]
        kept = list(range(len(keys)))
        nearest = list(range(len(keys)))

        def dist2(i: int, j: int) -> float:
            return sum((a - b) ** 2 for a, b in zip(logs[i], logs[j]))

        def closest(i: int, pool: list[int]) -> int:
            return min(pool, key=lambda j: (dist2(i, j), j))

        def settled(i: int, j: int, pool: list[int]) -> bool:
            # The resolver ranks in float32; a rival with another answer must
            # sit clearly farther so rounding cannot reorder the two.
            if name[j] != name[i]:
                return False
            limit = dist2(i, j) + TIE_MARGIN
            return all(name[r] == name[i] or dist2(i, r) > limit for r in pool)

        changed = True
        while changed:
            changed = False
            for p in list(kept):
                if len(kept) == 1:
                    break
                pool = [j for j in kept if j != p]
                affected = [i for i in range(len(keys)) if nearest[i] == p]
                moved = {i: closest(i, pool) for i in affected}
                if all(settled(i, j, pool) for i, j in moved.items()):
                    kept = pool
                    for i, j in moved.items():
                        nearest[i] = j
                    changed = True
        for j in kept:
            kept_all[keys[j]] = name[j]
    return kept_all


def cmd_build(args) -> int:
    points: dict[tuple, str] = {}
    for path in args.winners:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if not r["instance"] or r["instance"] == "-":
                    continue
                key = tuple(int(r[c]) for c in KEY)
                points.setdefault(key, r["instance"])
    measured = len(points)
    if not args.keep_all:
        points = prune(points)
    print(f"[build] {measured} measured shapes, {len(points)} kept")
    names = sorted(set(points.values()))
    index = {n: i for i, n in enumerate(names)}
    rows = []
    for key in sorted(points):
        ab, d, ta, bias, m, n, k, batch = key
        rows.append({"ab_dtype": DTYPE_NAME[ab], "d_dtype": DTYPE_NAME[d],
                     "trans_a": ta, "bias": bias,
                     "instance": index[points[key]],
                     "m": m, "n": n, "k": k, "batch": batch})
    lut = {"schema_version": SCHEMA_VERSION, "gpu_arch": args.arch,
           "rocm_version": args.rocm_version, "kernel_abi": KERNEL_ABI,
           "model_key": args.model_key, "weight_m": 1.0, "weight_n": 1.0,
           "weight_k": 1.0, "weight_batch": 1.0, "instances": names,
           "points": rows}
    out = LUT_DIR / f"{args.arch}.json"
    LUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="\n") as f:
        json.dump(lut, f, indent=1)
        f.write("\n")
    print(f"[build] {len(rows)} points, {len(names)} instances -> {out}")
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
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            return proc.returncode
        produced = list(Path(tmp).glob("*.bin")) + list(Path(tmp).glob("*.fb"))
        if not produced:
            print("[compile] flatc produced nothing", file=sys.stderr)
            return 1
        dst = LUT_DIR / f"{args.arch}.fb"
        shutil.copy(produced[0], dst)
    print(f"[compile] -> {dst} ({dst.stat().st_size} bytes)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["extract", "build", "compile"])
    ap.add_argument("--logs", nargs="+", default=[])
    ap.add_argument("--out", default=None)
    ap.add_argument("--winners", nargs="+", default=[])
    ap.add_argument("--arch", default="gfx1151")
    ap.add_argument("--flatc", default="flatc")
    ap.add_argument("--rocm-version", type=int)
    ap.add_argument("--model-key", default="")
    ap.add_argument("--keep-all", action="store_true",
                    help="build: emit every measured point, no pruning")
    args = ap.parse_args()
    if args.command == "extract":
        if not args.logs or not args.out:
            ap.error("extract needs --logs and --out")
        return cmd_extract(args)
    if args.command == "build":
        if not args.winners or args.rocm_version is None:
            ap.error("build needs --winners and --rocm-version")
        return cmd_build(args)
    return cmd_compile(args)


if __name__ == "__main__":
    raise SystemExit(main())
