#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Time hip_conv directly, one shape at a time, without running a model.

The convolution tile heuristic is a pure function of shape and CU count, so
changing it can be evaluated against the shapes the models contain without
paying for twenty model sweeps. A full sweep is minutes per arm and mixes the
convolution's time with everything else in the graph; this is seconds and
measures only the kernel under test.

Loads the deployed custom_kernels DLL through ctypes and calls its extern "C"
hip_conv, so it measures exactly the binary the harness deploys -- no separate
build, and no risk of testing a different compile of the same source.

Usage:
    python conv_shapes.py resnet50=...\\model.onnx --out shapes.json
    python conv_microbench.py shapes.json --bin C:\\...\\bin --out base.json
    # ... change the heuristic, rebuild, redeploy ...
    python conv_microbench.py shapes.json --bin C:\\...\\bin --out cand.json
    python conv_microbench.py --compare base.json cand.json

Numbers are per single convolution call. `weighted_ms` multiplies by the node
count, which is what ranks a shape by its contribution to a model's runtime.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import sys

# hip_conv's dtype enum (HIP_DTYPE_* in hip_custom_kernels.h).
_HIP_DTYPE = {"float32": 0, "float16": 1, "bfloat16": 2}
_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}

_HIP_SUCCESS = 0


class Hip:
    """The slice of the HIP runtime this benchmark needs."""

    def __init__(self, bindir: str):
        self.lib = self._load_runtime(bindir)
        c = self.lib

        c.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        c.hipFree.argtypes = [ctypes.c_void_p]
        c.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        c.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_size_t, ctypes.c_int]
        c.hipDeviceSynchronize.argtypes = []
        c.hipGetLastError.argtypes = []
        c.hipEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        c.hipEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        c.hipEventSynchronize.argtypes = [ctypes.c_void_p]
        c.hipEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float),
                                          ctypes.c_void_p, ctypes.c_void_p]
        c.hipEventDestroy.argtypes = [ctypes.c_void_p]

    @staticmethod
    def _load_runtime(bindir: str):
        # The kernels DLL is built against a specific HIP major version; load
        # the copy sitting next to it in preference to whatever is on PATH, so
        # the benchmark cannot silently bind a different runtime than the EP.
        names = []
        if bindir and os.path.isdir(bindir):
            names += [os.path.join(bindir, n) for n in os.listdir(bindir)
                      if n.lower().startswith("amdhip64")]
        names += ["amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll",
                  "libamdhip64.so"]
        last = None
        for n in names:
            try:
                return ctypes.CDLL(n)
            except OSError as e:  # noqa: PERF203
                last = e
        raise RuntimeError(f"could not load the HIP runtime: {last}")

    def check(self, rc: int, what: str):
        if rc != _HIP_SUCCESS:
            raise RuntimeError(f"{what} failed with HIP error {rc}")

    def malloc(self, nbytes: int) -> ctypes.c_void_p:
        p = ctypes.c_void_p()
        self.check(self.lib.hipMalloc(ctypes.byref(p), nbytes), "hipMalloc")
        # Non-zero so a kernel reading uninitialised memory is visible as a
        # timing anomaly rather than reading a page of convenient zeros.
        self.check(self.lib.hipMemset(p, 1, nbytes), "hipMemset")
        return p

    def free(self, p):
        if p:
            self.lib.hipFree(p)

    def sync(self):
        self.check(self.lib.hipDeviceSynchronize(), "hipDeviceSynchronize")

    def event(self):
        e = ctypes.c_void_p()
        self.check(self.lib.hipEventCreate(ctypes.byref(e)), "hipEventCreate")
        return e

    def elapsed_ms(self, start, stop) -> float:
        ms = ctypes.c_float()
        self.check(self.lib.hipEventElapsedTime(ctypes.byref(ms), start, stop),
                   "hipEventElapsedTime")
        return ms.value


def _load_kernels(bindir: str):
    cands = []
    if bindir and os.path.isdir(bindir):
        cands += [os.path.join(bindir, n) for n in os.listdir(bindir)
                  if n.lower().startswith("custom_kernels")
                  and n.lower().endswith((".dll", ".so"))]
    if not cands:
        raise RuntimeError(f"no custom_kernels DLL under {bindir!r}")
    lib = ctypes.CDLL(cands[0])
    fn = lib.hip_conv
    fn.restype = ctypes.c_int
    fn.argtypes = (
        [ctypes.c_void_p] * 5          # stream, x, w, bias, y
        + [ctypes.c_int, ctypes.c_int]  # dtype, spatial_rank
        + [ctypes.c_int64] * 22         # N..group
    )
    return cands[0], fn


def _prod(xs):
    n = 1
    for x in xs:
        n *= x
    return n


def bench_shape(hip: Hip, conv, rec: dict, iters: int, warmup: int):
    """Time one convolution shape. Returns a result dict, or raises."""
    dt = rec["dtype"]
    esz = _BYTES[dt]
    rank = rec["rank"]
    N, Cin, Cout, g = rec["N"], rec["Cin"], rec["Cout"], rec["group"]

    n_in = N * Cin * _prod(rec["in"])
    n_out = N * Cout * _prod(rec["out"])
    n_w = Cout * (Cin // g) * _prod(rec["kernel"])

    x = w = b = y = None
    try:
        x = hip.malloc(n_in * esz)
        w = hip.malloc(n_w * esz)
        b = hip.malloc(Cout * esz) if rec.get("bias", True) else None
        y = hip.malloc(n_out * esz)

        args = ([None, x, w, b, y, _HIP_DTYPE[dt], rank, N, Cin, Cout]
                + list(rec["in"]) + list(rec["out"]) + list(rec["kernel"])
                + list(rec["stride"]) + list(rec["pad_begin"])
                + list(rec["dilation"]) + [g])

        for _ in range(warmup):
            rc = conv(*args)
            if rc != 0:
                raise RuntimeError(f"hip_conv returned {rc}")
        hip.sync()

        # Time each iteration separately: a single event pair around a loop
        # would hide a bimodal distribution, and the tile ladder is exactly the
        # kind of thing that produces one.
        start, stop = hip.event(), hip.event()
        samples = []
        for _ in range(iters):
            hip.lib.hipEventRecord(start, None)
            conv(*args)
            hip.lib.hipEventRecord(stop, None)
            hip.check(hip.lib.hipEventSynchronize(stop), "hipEventSynchronize")
            samples.append(hip.elapsed_ms(start, stop))
        hip.lib.hipEventDestroy(start)
        hip.lib.hipEventDestroy(stop)

        ms = statistics.median(samples)
        # 2 flops per MAC, over the full im2col contraction.
        flops = 2.0 * N * Cout * _prod(rec["out"]) * (Cin // g) * _prod(rec["kernel"])
        # Compulsory traffic only -- what a perfect kernel would move. The
        # ratio of this to achieved bandwidth is the gather amplification.
        bytes_min = (n_in + n_w + n_out) * esz
        return {
            "ms": ms,
            "ms_min": min(samples),
            "ms_max": max(samples),
            "count": rec.get("count", 1),
            "weighted_ms": ms * rec.get("count", 1),
            "tflops": flops / (ms * 1e-3) / 1e12,
            "min_gbps": bytes_min / (ms * 1e-3) / 1e9,
        }
    finally:
        for p in (x, w, b, y):
            hip.free(p)


def cmd_run(args):
    with open(args.shapes) as f:
        shapes = json.load(f)

    dll, conv = _load_kernels(args.bin)
    hip = Hip(args.bin)
    print(f"kernels: {dll}", file=sys.stderr)

    out = {}
    for model, recs in shapes.items():
        if args.model and model not in args.model:
            continue
        out[model] = []
        for rec in recs:
            if rec["op"] != "Conv":
                continue  # hip_conv only; ConvTranspose has its own entry point
            label = ("Cin=%d Cout=%d out=%s k=%s s=%s g=%d %s"
                     % (rec["Cin"], rec["Cout"], rec["out"][:rec["rank"]],
                        rec["kernel"][:rec["rank"]], rec["stride"][:rec["rank"]],
                        rec["group"], rec["dtype"]))
            try:
                r = bench_shape(hip, conv, rec, args.iters, args.warmup)
            except Exception as e:  # noqa: BLE001
                print(f"  SKIP {label}: {e}", file=sys.stderr)
                continue
            r["label"] = label
            r["shape"] = rec
            out[model].append(r)
            print("  %-72s %8.3f ms x%-4d %9.3f ms tot %6.2f TF"
                  % (label, r["ms"], r["count"], r["weighted_ms"], r["tflops"]),
                  file=sys.stderr)
        tot = sum(r["weighted_ms"] for r in out[model])
        print(f"{model:<34} total {tot:9.3f} ms of convolution", file=sys.stderr)

    text = json.dumps(out, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)


def cmd_compare(args):
    with open(args.compare[0]) as f:
        base = json.load(f)
    with open(args.compare[1]) as f:
        cand = json.load(f)

    print(f"{'model / shape':<70}{'base ms':>10}{'cand ms':>10}{'delta':>9}")
    print("-" * 99)
    gb = gc = 0.0
    for model in base:
        if model not in cand:
            continue
        bl = {r["label"]: r for r in base[model]}
        cl = {r["label"]: r for r in cand[model]}
        mb = sum(r["weighted_ms"] for r in base[model])
        mc = sum(r["weighted_ms"] for r in cand[model] if r["label"] in bl)
        gb += mb
        gc += mc
        d = (mc - mb) / mb * 100 if mb else 0.0
        print(f"{model:<70}{mb:>10.3f}{mc:>10.3f}{d:>8.1f}%")
        rows = []
        for label, rb in bl.items():
            rc = cl.get(label)
            if rc is None:
                continue
            rows.append((rc["weighted_ms"] - rb["weighted_ms"], label, rb, rc))
        rows.sort(key=lambda t: t[0])
        for delta, label, rb, rc in rows:
            if abs(delta) < args.threshold:
                continue
            pct = (rc["ms"] - rb["ms"]) / rb["ms"] * 100 if rb["ms"] else 0.0
            mark = "  FASTER" if delta < 0 else "  SLOWER"
            print(f"    {label:<66}{rb['weighted_ms']:>10.3f}"
                  f"{rc['weighted_ms']:>10.3f}{pct:>8.1f}%{mark}")
    print("-" * 99)
    d = (gc - gb) / gb * 100 if gb else 0.0
    print(f"{'TOTAL':<70}{gb:>10.3f}{gc:>10.3f}{d:>8.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shapes", nargs="?", help="JSON from conv_shapes.py")
    ap.add_argument("--bin", default=r"C:\Users\zyq\gpu-test-package\bin",
                    help="Directory holding custom_kernels_<arch>.dll")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--model", action="append",
                    help="Only bench this model (repeatable).")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("BASE", "CAND"),
                    help="Diff two result files instead of benchmarking.")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="Hide per-shape rows moving less than this many ms.")
    args = ap.parse_args()

    if args.compare:
        cmd_compare(args)
    elif args.shapes:
        cmd_run(args)
    else:
        ap.error("need a shapes file, or --compare")


if __name__ == "__main__":
    main()
