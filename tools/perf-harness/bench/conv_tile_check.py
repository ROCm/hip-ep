"""Check every conv tile against conv_direct on real data.

A new tile is a new kernel instantiation, and the staging arithmetic it changes
-- how many threads cover a contraction slice, how many rows each one carries --
is exactly where an off-by-one produces a plausible-looking wrong answer rather
than a crash. The perf sweep cannot see that: it fills its buffers with
hipMemset and only reads the clock.

So this runs each tile against conv_direct, which has no staging at all -- a
thread per output element, re-reading its own inputs -- on pseudorandom inputs,
and reports the worst relative error per tile. Direct is the reference because
it is the one path whose correctness is obvious by inspection, which is what the
comment in the ladder says it is for.

One process per tile, because the kernel DLL reads HIPDNN_EP_CONV_TILE once.

    py -3.11 conv_tile_check.py --bin <deploy dir>
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conv_microbench import Hip, _load_kernels, _HIP_DTYPE, _BYTES, _prod

# hipMemcpyKind
H2D, D2H = 1, 2

# Shapes with resolved extents covering both tables and every rung: wide and
# narrow output channel counts, few and many output positions, 1x1 and 3x3, a
# stride, a group, and a K long enough to trip the gather threshold. Kept small
# so the fp64 reference below stays cheap.
CASES = [
    # (dtype, N, Cin, Cout, in, out, kernel, stride, pad, dil, group)
    ("float16", 1, 64, 128, [28, 28, 1], [28, 28, 1], [3, 3, 1], [1, 1, 1], [1, 1, 0], [1, 1, 1], 1),
    ("float16", 1, 128, 256, [14, 14, 1], [14, 14, 1], [3, 3, 1], [1, 1, 1], [1, 1, 0], [1, 1, 1], 1),
    ("float16", 1, 256, 512, [7, 7, 1], [7, 7, 1], [1, 1, 1], [1, 1, 1], [0, 0, 0], [1, 1, 1], 1),
    ("float16", 1, 512, 512, [7, 7, 1], [7, 7, 1], [3, 3, 1], [1, 1, 1], [1, 1, 0], [1, 1, 1], 1),
    ("float16", 1, 32, 16, [30, 30, 1], [30, 30, 1], [3, 3, 1], [1, 1, 1], [1, 1, 0], [1, 1, 1], 1),
    ("float16", 1, 64, 3, [40, 40, 1], [40, 40, 1], [3, 3, 1], [1, 1, 1], [1, 1, 0], [1, 1, 1], 1),
    ("float16", 1, 96, 48, [20, 20, 1], [10, 10, 1], [3, 3, 1], [2, 2, 1], [1, 1, 0], [1, 1, 1], 1),
    ("float16", 2, 64, 64, [16, 16, 1], [16, 16, 1], [3, 3, 1], [1, 1, 1], [1, 1, 0], [1, 1, 1], 1),
    ("float16", 1, 128, 128, [16, 16, 1], [16, 16, 1], [3, 3, 1], [1, 1, 1], [1, 1, 0], [1, 1, 1], 2),
    # A K past the gather threshold (2048) with few output positions.
    ("float16", 1, 512, 256, [6, 6, 1], [6, 6, 1], [3, 3, 1], [1, 1, 1], [1, 1, 0], [1, 1, 1], 1),
    ("float32", 1, 64, 128, [28, 28, 1], [28, 28, 1], [3, 3, 1], [1, 1, 1], [1, 1, 0], [1, 1, 1], 1),
    ("float32", 1, 128, 128, [20, 20, 1], [20, 20, 1], [1, 1, 1], [1, 1, 1], [0, 0, 0], [1, 1, 1], 1),
    ("float32", 1, 96, 64, [24, 24, 1], [12, 12, 1], [3, 3, 1], [2, 2, 1], [1, 1, 0], [1, 1, 1], 1),
]

TILES = ["direct",
         "128x256", "256x128", "128x128", "64x256", "32x256", "16x256",
         "128x256x2x4", "128x128x2x2", "64x256x2x2",
         "128x128x8x8", "64x128x4x8", "128x64x8x4", "64x64x4x4",
         "32x128x2x8", "64x32x4x2", "32x64x2x4", "32x32x2x2"]


def rng(n, seed):
    """A small deterministic generator, so both sides see identical bytes.

    Values are centred and scaled to keep an fp16 accumulation over K taps
    well inside range -- the point is to compare two kernels, not to explore
    overflow, which the numeric suite covers.
    """
    s = seed & 0xFFFFFFFF
    out = []
    for _ in range(n):
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        out.append(((s >> 8) & 0x3FF) / 1023.0 - 0.5)
    return out


def to_bytes(vals, dtype):
    import struct
    if dtype == "float32":
        return struct.pack(f"<{len(vals)}f", *vals)
    # fp16 via the struct 'e' code, available since 3.6.
    return struct.pack(f"<{len(vals)}e", *vals)


def from_bytes(buf, dtype, n):
    import struct
    code = "f" if dtype == "float32" else "e"
    return list(struct.unpack_from(f"<{n}{code}", buf, 0))


def run_case(hip, conv, case, seed):
    (dt, N, Cin, Cout, ins, outs, ker, stride, pad, dil, g) = case
    esz = _BYTES[dt]
    n_in = N * Cin * _prod(ins)
    n_out = N * Cout * _prod(outs)
    n_w = Cout * (Cin // g) * _prod(ker)

    xv = rng(n_in, seed)
    wv = rng(n_w, seed + 7919)
    bv = rng(Cout, seed + 104729)

    x = w = b = y = None
    try:
        x, w, b, y = (hip.malloc(n_in * esz), hip.malloc(n_w * esz),
                      hip.malloc(Cout * esz), hip.malloc(n_out * esz))
        for ptr, vals in ((x, xv), (w, wv), (b, bv)):
            raw = to_bytes(vals, dt)
            hip.check(hip.lib.hipMemcpy(ptr, raw, len(raw), H2D), "hipMemcpy")
        args = ([None, x, w, b, y, _HIP_DTYPE[dt], 2, N, Cin, Cout]
                + list(ins) + list(outs) + list(ker) + list(stride)
                + list(pad) + list(dil) + [g])
        rc = conv(*args)
        if rc != 0:
            raise RuntimeError(f"hip_conv returned {rc}")
        hip.sync()
        buf = ctypes.create_string_buffer(n_out * esz)
        hip.check(hip.lib.hipMemcpy(buf, y, n_out * esz, D2H), "hipMemcpy")
        return from_bytes(buf, dt, n_out)
    finally:
        for p in (x, w, b, y):
            hip.free(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=r"C:\Users\zyq\gpu-test-package\bin")
    ap.add_argument("--out", help="write this tile's outputs here (internal)")
    ap.add_argument("--tiles", action="append")
    args = ap.parse_args()

    if args.out:  # child: run every case under whatever tile the env forces
        hip = Hip(args.bin)
        _, conv = _load_kernels(args.bin)
        res = []
        for i, case in enumerate(CASES):
            try:
                res.append(run_case(hip, conv, case, 1234 + i))
            except Exception as e:  # noqa: BLE001
                res.append({"error": str(e)})
        with open(args.out, "w") as f:
            json.dump(res, f)
        return 0

    os.environ.setdefault("HIPDNN_EP_DEBUG", "1")

    tiles = args.tiles or TILES
    workdir = os.path.join(os.environ.get("TEMP", "."), "conv_tile_check")
    os.makedirs(workdir, exist_ok=True)
    got = {}
    picks = {}
    for tile in tiles:
        out = os.path.join(workdir, f"{tile}.json")
        env = dict(os.environ)
        env["HIPDNN_EP_CONV_TILE"] = tile
        p = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--bin", args.bin,
             "--out", out], env=env, capture_output=True, text=True)
        if not os.path.exists(out):
            print(f"{tile}: no result\n{p.stderr[-600:]}", file=sys.stderr)
            continue
        with open(out) as f:
            got[tile] = json.load(f)
        # What the kernel says it launched, from the debug log. Without this the
        # comparison below passes for a tile that never ran: an unresolved name
        # leaves the ladder's own pick in place, which is correct code producing
        # a correct answer for the wrong tile.
        picks[tile] = set()
        for line in p.stderr.splitlines():
            if "hip_conv pick=" not in line:
                continue
            kind = line.split("pick=", 1)[1].split(None, 1)[0]
            dims = line.split("tile=", 1)[1].split(None, 1)[0]
            picks[tile].add(f"{kind} {dims}")

    if "direct" not in got:
        sys.exit("conv_direct produced no reference")
    ref = got["direct"]

    # fp16 carries ~3 decimal digits, and direct accumulates the contraction in
    # a different order than a tiled kernel does, so the two disagree in the
    # last bits by construction. The threshold is on the relative error against
    # the magnitude of the reference row, which is what distinguishes rounding
    # from a staging bug: a wrong tile is wrong by a whole term, not an epsilon.
    tol = {"float16": 3e-2, "float32": 1e-4}
    bad = 0
    # Split by dtype: a tile name only resolves against its own dtype's table,
    # so the scalar names leave the fp16 cases on the ladder's own pick and the
    # matrix names leave the fp32 ones. Reporting one worst error per tile hides
    # that, and would read as a pass for a tile that never ran.
    print(f"{'tile':<14}{'fp16':>12}{'fp32':>12}  worst case")
    print("-" * 56)
    for tile, res in got.items():
        if tile == "direct":
            continue
        worst = {"float16": 0.0, "float32": 0.0}
        worstCase = ""
        top = 0.0
        for i, (a, r) in enumerate(zip(res, ref)):
            dt = CASES[i][0]
            if isinstance(a, dict) or isinstance(r, dict):
                print(f"{tile:<14}{'ERROR':>24}  case {i}: "
                      f"{a if isinstance(a, dict) else r}")
                bad += 1
                continue
            scale = max(abs(v) for v in r) or 1.0
            err = max(abs(p - q) for p, q in zip(a, r)) / scale
            worst[dt] = max(worst[dt], err)
            if err > top:
                top, worstCase = err, f"case {i} ({dt})"
            if err > tol[dt]:
                bad += 1
        print(f"{tile:<14}{worst['float16']:>12.2e}"
              f"{worst['float32']:>12.2e}  {worstCase}")
    print()

    # Every tile under test must appear in some run's dispatch log, otherwise
    # its column above is measuring the ladder rather than the tile.
    print("dispatch (from HIPDNN_EP_DEBUG):")
    for tile in tiles:
        if tile in ("direct",) or tile not in picks:
            continue
        # A four-field name has to appear exactly; a two-field one only pins the
        # block shape, since it resolves to whichever register tile the table
        # lists first and this script does not read the table.
        want = tile if tile.count("x") == 3 else tile + "x"
        seen = sorted(picks[tile])
        ran = any(want in s for s in seen)
        if not ran:
            print(f"  {tile:<14} NOT DISPATCHED; saw {seen}")
            bad += 1
        else:
            print(f"  {tile:<14} {', '.join(seen)}")
    print()
    if bad:
        print(f"{bad} tile/case pair(s) disagree with conv_direct beyond "
              f"tolerance")
        return 1
    print("every tile agrees with conv_direct")
    return 0


if __name__ == "__main__":
    sys.exit(main())
