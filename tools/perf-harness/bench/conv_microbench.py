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
import subprocess
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


def _load_winograd(bindir: str):
    """The Winograd entry points, or None on a build that predates them.

    Separate from _load_kernels because the Winograd path is not reachable
    through hip_conv -- the EP dispatches it in wrap_conv, above the kernel
    ABI -- so timing it means calling it directly.
    """
    cands = [os.path.join(bindir, n) for n in os.listdir(bindir)
             if n.lower().startswith("custom_kernels")
             and n.lower().endswith((".dll", ".so"))]
    lib = ctypes.CDLL(cands[0])
    try:
        elig = lib.hip_conv_winograd_eligible
        prof = lib.hip_conv_winograd_profitable
        felems = lib.hip_conv_winograd_filter_elems
        xform = lib.hip_conv_winograd_filter
        fused = lib.hip_conv_winograd_fused
    except AttributeError:
        return None

    elig.restype = ctypes.c_int
    elig.argtypes = [ctypes.c_int] + [ctypes.c_int64] * 10
    prof.restype = ctypes.c_int
    prof.argtypes = [ctypes.c_int64] * 5
    felems.restype = ctypes.c_int64
    felems.argtypes = [ctypes.c_int64] * 2
    xform.restype = ctypes.c_int
    xform.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 2
    fused.restype = ctypes.c_int
    fused.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int64] * 9
    return {"eligible": elig, "profitable": prof, "filter_elems": felems,
            "filter": xform, "fused": fused}


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


def bench_winograd(hip: Hip, wg, rec: dict, iters: int, warmup: int):
    """Time the fused Winograd path on one shape, or return None if it does
    not apply. The filter transform is excluded, as it is in the EP: it is
    hoisted out of the call and cached per weight tensor."""
    if rec["dtype"] != "float16" or rec["rank"] != 2:
        return None
    k = rec["kernel"][:2]
    s = rec["stride"][:2]
    d = rec["dilation"][:2]
    N, Cin, Cout, g = rec["N"], rec["Cin"], rec["Cout"], rec["group"]
    if not wg["eligible"](_HIP_DTYPE["float16"], 2, k[0], k[1], s[0], s[1],
                          d[0], d[1], g, Cin, Cout):
        return None

    inp, outp, pad = rec["in"][:2], rec["out"][:2], rec["pad_begin"][:2]
    esz = 2
    x = w = u = b = y = None
    try:
        x = hip.malloc(N * Cin * inp[0] * inp[1] * esz)
        w = hip.malloc(Cout * Cin * 9 * esz)
        u = hip.malloc(wg["filter_elems"](Cout, Cin) * esz)
        b = hip.malloc(Cout * esz)
        y = hip.malloc(N * Cout * outp[0] * outp[1] * esz)

        rc = wg["filter"](None, w, u, Cout, Cin)
        if rc != 0:
            raise RuntimeError(f"winograd_filter returned {rc}")

        args_ = [None, x, u, b, y, N, Cin, Cout, inp[0], inp[1],
                 outp[0], outp[1], pad[0], pad[1]]
        for _ in range(warmup):
            rc = wg["fused"](*args_)
            if rc != 0:
                raise RuntimeError(f"winograd_fused returned {rc}")
        hip.sync()

        start, stop = hip.event(), hip.event()
        samples = []
        for _ in range(iters):
            hip.lib.hipEventRecord(start, None)
            wg["fused"](*args_)
            hip.lib.hipEventRecord(stop, None)
            hip.check(hip.lib.hipEventSynchronize(stop), "hipEventSynchronize")
            samples.append(hip.elapsed_ms(start, stop))
        hip.lib.hipEventDestroy(start)
        hip.lib.hipEventDestroy(stop)
        return statistics.median(samples)
    finally:
        for p in (x, w, u, b, y):
            hip.free(p)


def cmd_winograd(args):
    """A/B the fused Winograd path against hip_conv, per shape."""
    with open(args.shapes) as f:
        shapes = json.load(f)

    dll, conv = _load_kernels(args.bin)
    wg = _load_winograd(args.bin)
    if not wg:
        raise SystemExit("this build has no hip_conv_winograd_fused")
    hip = Hip(args.bin)
    print(f"kernels: {dll}", file=sys.stderr)

    # Three totals, because the question is not whether Winograd is faster --
    # it is faster on some shapes and slower on others -- but whether the gate
    # picks the right one. `gated` is what actually ships; `oracle` is the best
    # possible per-shape choice and so the bound the gate is judged against.
    tot_d = tot_w = tot_g = tot_o = 0.0
    for model, recs in shapes.items():
        if args.model and model not in args.model:
            continue
        for rec in recs:
            if rec["op"] != "Conv":
                continue
            wms = bench_winograd(hip, wg, rec, args.iters, args.warmup)
            if wms is None:
                continue
            d = bench_shape(hip, conv, rec, args.iters, args.warmup)["ms"]
            cnt = rec.get("count", 1)
            gate = bool(wg["profitable"](rec["N"], rec["Cin"], rec["Cout"],
                                         rec["out"][0], rec["out"][1]))
            tot_d += d * cnt
            tot_w += wms * cnt
            tot_g += (wms if gate else d) * cnt
            tot_o += min(d, wms) * cnt
            label = ("Cin=%-4d Cout=%-4d out=%s"
                     % (rec["Cin"], rec["Cout"], rec["out"][:2]))
            flag = "USE" if gate else "   "
            miss = "" if gate == (wms < d) else "  <- gate wrong"
            print("  %-40s direct %7.3f  winograd %7.3f  %5.2fx %s x%-3d%s"
                  % (label, d, wms, d / wms, flag, cnt, miss),
                  file=sys.stderr)
    if tot_w > 0:
        print(file=sys.stderr)
        print("  always direct   %8.3f ms" % tot_d, file=sys.stderr)
        print("  always winograd %8.3f ms" % tot_w, file=sys.stderr)
        print("  gated (ships)   %8.3f ms  %.2fx vs direct, %.1f%% of oracle"
              % (tot_g, tot_d / tot_g, 100.0 * tot_o / tot_g), file=sys.stderr)
        print("  oracle          %8.3f ms" % tot_o, file=sys.stderr)


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


def cmd_tile_sweep(args):
    """Run every tile against every shape and report the best one for each.

    The cost model in conv_tile_select.h is a handful of ratios standing in for
    a memory system, and there is no way to know whether it ranks two tiles
    correctly except to run both. This produces the table it is answerable to.

    Each tile needs its own process: the kernel DLL reads the override once and
    caches it, which is the right thing for a DLL and the wrong thing for a
    sweep.
    """
    # Four-field names where a block shape appears at more than one register
    # tile, so the sweep row and the kernel launched agree. See
    # conv_tile_select.h for which entries share a block shape.
    tiles = args.tiles or ["128x256", "256x128", "128x128", "64x256", "32x256",
                           "16x256", "128x256x2x4", "128x128x2x2",
                           "64x256x2x2",
                           "128x128x8x8", "64x128x4x8", "128x64x8x4",
                           "64x64x4x4", "32x128x2x8", "64x32x4x2",
                           "32x64x2x4", "32x32x2x2", "direct"]
    results = {}
    for tile in tiles:
        out = os.path.join(args.workdir, f"tile_{tile}.json")
        os.makedirs(args.workdir, exist_ok=True)
        env = dict(os.environ)
        env["HIPDNN_EP_CONV_TILE"] = tile
        cmd = [sys.executable, os.path.abspath(__file__), args.shapes,
               "--bin", args.bin, "--iters", str(args.iters),
               "--warmup", str(args.warmup), "--out", out]
        for m in args.model or []:
            cmd += ["--model", m]
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if not os.path.exists(out):
            print(f"tile {tile}: no result\n{p.stderr[-800:]}", file=sys.stderr)
            continue
        with open(out) as f:
            results[tile] = json.load(f)
        tot = sum(r["weighted_ms"] for recs in results[tile].values()
                  for r in recs)
        print(f"tile {tile:<10} total {tot:9.3f} ms", file=sys.stderr)

    if not results:
        sys.exit("no tile produced results")

    # Per shape, which tile actually won.
    print()
    print(f"{'model':<20}{'shape':<62}{'best':>9}{'ms':>9}   runners-up")
    print("-" * 130)
    table = {}
    for model in next(iter(results.values())):
        for tile, data in results.items():
            for r in data.get(model, []):
                table.setdefault((model, r["label"]), {})[tile] = r

        rows = [(k, v) for k, v in table.items() if k[0] == model]
        rows.sort(key=lambda kv: -min(r["weighted_ms"] for r in kv[1].values()))
        for (m, label), byTile in rows:
            ranked = sorted(byTile.items(), key=lambda kv: kv[1]["ms"])
            best, br = ranked[0]
            rest = "  ".join(f"{t}:{r['ms'] / br['ms']:.2f}x"
                             for t, r in ranked[1:4])
            print(f"{m:<20}{label:<62}{best:>9}{br['ms']:>9.3f}   {rest}")

    with open(args.out or os.path.join(args.workdir, "tile_sweep.json"),
              "w") as f:
        json.dump({f"{m}|{label}": {t: r["ms"] for t, r in byTile.items()}
                   for (m, label), byTile in table.items()}, f, indent=2)


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
    ap.add_argument("--tile-sweep", action="store_true",
                    help="Force each tile in turn and report the best per shape.")
    ap.add_argument("--tiles", action="append",
                    help="Restrict --tile-sweep to these tiles (repeatable).")
    ap.add_argument("--workdir", default=r"C:\Users\zyq\scratch\conv-tiles",
                    help="Where --tile-sweep writes its per-tile results.")
    ap.add_argument("--winograd", action="store_true",
                    help="A/B the fused Winograd path against hip_conv.")
    args = ap.parse_args()

    if args.compare:
        cmd_compare(args)
    elif args.winograd:
        if not args.shapes:
            ap.error("--winograd needs a shapes file")
        cmd_winograd(args)
    elif args.tile_sweep:
        if not args.shapes:
            ap.error("--tile-sweep needs a shapes file")
        cmd_tile_sweep(args)
    elif args.shapes:
        cmd_run(args)
    else:
        ap.error("need a shapes file, or --compare")


if __name__ == "__main__":
    main()
