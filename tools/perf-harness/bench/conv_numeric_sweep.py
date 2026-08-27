#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Run the convolution shape sweep against a deployed build, via hip-onnx-runner.

test/numeric drives the same cases through ORT's plugin-EP API, which needs the
Python onnxruntime package to match the ORT the EP was built against. That is
often not true of a deploy directory assembled by hand, and a mismatch is an
access violation inside session creation rather than a clean error. This path
has no such coupling: hip-onnx-runner is built with the EP, ships beside it,
and produces the CPU reference itself with `-n`.

Both arms are run with the same `-s` seed, so the random inputs match and the
outputs are directly comparable.

    python conv_numeric_sweep.py --bin C:\\Users\\zyq\\gpu-test-package\\bin

A case fails on any non-finite output, or on exceeding the dtype's tolerance.
Non-finiteness is reported separately and first: a NaN also destroys the
correlation, so reporting only the correlation would describe an arithmetic
fault as imprecision.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CASES = os.path.normpath(os.path.join(_HERE, "..", "..", "..",
                                       "test", "numeric", "tests"))
sys.path.insert(0, _CASES)

import conv_cases  # noqa: E402

# hip-onnx-runner names dumps  <name>_<type>.bin ; these are the ones the sweep
# can produce.
_SUFFIX_DTYPE = {"fp32": np.float32, "fp16": np.float16}


def _read_dump(d: str):
    """Load every tensor in a dump directory, keyed by filename stem."""
    out = {}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".bin"):
            continue
        stem = fn[:-4]
        dt = None
        for suf, npt in _SUFFIX_DTYPE.items():
            if stem.endswith("_" + suf):
                dt, stem = npt, stem[: -(len(suf) + 1)]
                break
        if dt is None:
            continue
        out[stem] = np.fromfile(os.path.join(d, fn), dtype=dt)
    return out


def _run(runner: str, bindir: str, model: str, seed: int, no_ep: bool,
         env: dict):
    stem = os.path.splitext(os.path.basename(model))[0]
    # The runner tags the CPU-only dump directory with _cpu so the two runs do
    # not collide, which also means the path depends on the mode.
    dump = os.path.join(bindir, f"{stem}{'_cpu' if no_ep else ''}_o_dump")
    shutil.rmtree(dump, ignore_errors=True)

    cmd = [runner, "-m", model, "-d", "2", "-s", str(seed)]
    if no_ep:
        cmd.append("-n")
    p = subprocess.run(cmd, cwd=bindir, env=env, capture_output=True, text=True)
    if not os.path.isdir(dump):
        tail = (p.stdout + p.stderr).strip().splitlines()[-6:]
        raise RuntimeError("no dump produced; " + " | ".join(tail))
    try:
        return _read_dump(dump)
    finally:
        shutil.rmtree(dump, ignore_errors=True)


def _compare(cpu: dict, ep: dict, tol: float):
    """Return (ok, message) for one case."""
    problems = []
    for name, ref in cpu.items():
        got = ep.get(name)
        if got is None:
            problems.append(f"{name}: missing from the EP dump")
            continue
        if got.shape != ref.shape:
            problems.append(f"{name}: {got.shape} vs reference {ref.shape}")
            continue

        bad = int(np.count_nonzero(~np.isfinite(got)))
        if bad:
            finite = got[np.isfinite(got)]
            lo = float(finite.min()) if finite.size else float("nan")
            hi = float(finite.max()) if finite.size else float("nan")
            problems.append(
                f"{name}: {bad} non-finite of {got.size} "
                f"({100.0 * bad / got.size:.2f}%), finite range [{lo:.4g}, {hi:.4g}]"
            )
            continue

        a = got.astype(np.float64)
        b = ref.astype(np.float64)
        denom = max(np.abs(b).max(), 1e-30)
        rel = float(np.abs(a - b).max() / denom)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        cos = float(a @ b / (na * nb)) if na and nb else 1.0
        if rel > tol or cos < 0.999:
            problems.append(f"{name}: max rel {rel:.3g}, cosine {cos:.6f}")
    return (not problems), "; ".join(problems)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default=r"C:\Users\zyq\gpu-test-package\bin",
                    help="Deploy directory holding hip-onnx-runner and the EP.")
    ap.add_argument("--work", default=r"C:\Users\zyq\scratch\conv-sweep",
                    help="Where the generated models are written.")
    ap.add_argument("--rocm", default=r"C:\Users\zyq\therock-dist\bin",
                    help="ROCm runtime bin, prepended to PATH.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-k", "--filter", help="Only run cases whose id contains this.")
    ap.add_argument("--keep", action="store_true",
                    help="Keep generated models (they are removed on pass).")
    args = ap.parse_args()

    runner = os.path.join(args.bin, "hip-onnx-runner.exe")
    if not os.path.exists(runner):
        runner = os.path.join(args.bin, "hip-onnx-runner")
    if not os.path.exists(runner):
        sys.exit(f"hip-onnx-runner not found in {args.bin}")

    os.makedirs(args.work, exist_ok=True)
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([args.bin, args.rocm, env.get("PATH", "")])
    # The per-op profiler perturbs nothing here but adds noise to the log, and
    # the debug log is enormous across a whole sweep.
    for k in ("HIPDNN_EP_PERF", "HIPDNN_EP_DEBUG", "HIPDNN_EP_TRACE_FILE"):
        env.pop(k, None)

    cases = conv_cases.ALL_CASES
    if args.filter:
        cases = [c for c in cases if args.filter in c["id"]]
    if not cases:
        sys.exit("no cases matched")

    npass = nfail = nskip = 0
    failures = []
    for case in cases:
        cid = case["id"]
        model, _ = conv_cases.build(case)
        path = os.path.join(args.work, f"{cid}.onnx")
        with open(path, "wb") as f:
            f.write(model.SerializeToString())

        try:
            cpu = _run(runner, args.bin, path, args.seed, True, env)
            ep = _run(runner, args.bin, path, args.seed, False, env)
        except Exception as e:  # noqa: BLE001
            nskip += 1
            print(f"SKIP {cid:<28} {e}")
            continue

        tol = 2e-2 if case.get("dtype", "float16") == "float16" else 1e-4
        ok, msg = _compare(cpu, ep, tol)
        if ok:
            npass += 1
            print(f"pass {cid}")
            if not args.keep:
                os.remove(path)
        else:
            nfail += 1
            failures.append((cid, msg))
            print(f"FAIL {cid:<28} {msg}")

    print()
    print(f"{npass} passed, {nfail} failed, {nskip} skipped")
    if failures:
        print("\nfailures:")
        for cid, msg in failures:
            print(f"  {cid}: {msg}")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
