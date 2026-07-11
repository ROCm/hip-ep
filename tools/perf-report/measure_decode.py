#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Measurement harness for the HIP-graph decode-capture work (staged plan S0).

Two independent, noise-aware measurements, both driven by env flags so any
in-progress optimization (behind an env flag) can be A/B'd against baseline on
the SAME binary:

  readback : compile+run the decoder ONNX under HIPDNN_EP_PERF=1 via
             hip-onnx-runner and report the per-token `readback_scalar` count
             (the static-decode / capture-clean signal). Fast (~15 s), no
             autoregressive loop needed.

  tps      : interleaved paired A/B of decode throughput via OGA
             benchmark_multimodal (fresh process per sample cancels
             process-level variance; pairing cancels thermal drift). Reports
             mean/paired-delta/95% CI and a significance verdict.

Paths default to this StrixHalo box; override via the env vars in [brackets].

Usage:
  python measure_decode.py readback [--flag HIPDNN_EP_X=1]
  python measure_decode.py tps --flag HIPDNN_EP_X=1 [--pairs 6] [--gen 128] [--reps 5]
"""
import argparse
import math
import os
import re
import subprocess
import sys

ROOK = os.environ.get("THEROCK_BIN", r"C:\Users\Administrator\workspace\therock-keep\bin")
EP_BIN = os.environ.get("HIPDNN_EP_BIN", r"C:\Users\Administrator\workspace\build\onnx-hipdnn-ep\bin")
VENV_PY = os.environ.get("VENV_PY", r"C:\Users\Administrator\workspace\ci-fresh\venv314\Scripts\python.exe")
RUN_ONNX = os.environ.get("RUN_ONNX", r"C:\Users\Administrator\workspace\ci-fresh\hip-python-package\run_onnx.py")
BENCH = os.environ.get("BENCH_MM", r"C:\Users\Administrator\workspace\ci-fresh\gpu-test-package\bin\benchmark_multimodal.py")
MODEL_DIR = os.environ.get("MODEL_DIR", r"D:\Qwen3.5-35B-A3B-fp16-ve-fp16-int4-text-gs32-dml")
TEXT_ONNX = os.environ.get("TEXT_ONNX", os.path.join(MODEL_DIR, "text.onnx"))
IMAGE = os.environ.get("DECODE_IMAGE", r"C:\Users\Administrator\workspace\ci-fresh\dog_512.jpg")

TPS_RE = re.compile(r"Average Token Generation Throughput \(per token\):\s*([\d.]+)")
RB_RE = re.compile(r"readback_scalar\s+(\d+)")


def _split_flag(flag):
    if not flag:
        return None, None
    k, _, v = flag.partition("=")
    return k, (v or "1")


def measure_readback(flag):
    env = dict(os.environ)
    env["PATH"] = ROOK + os.pathsep + EP_BIN + os.pathsep + env.get("PATH", "")
    env["HIPDNN_EP_PERF"] = "1"
    k, v = _split_flag(flag)
    if k:
        env[k] = v
    p = subprocess.run([os.path.join(EP_BIN, "hip-onnx-runner.exe"), "-m", TEXT_ONNX, "-d", "0"],
                       env=env, capture_output=True, text=True, cwd=EP_BIN)
    counts = [int(m) for m in RB_RE.findall(p.stdout + p.stderr)]
    n = counts[0] if counts else -1
    print(f"[readback] flag={flag or 'baseline'}  readback_scalar={n}")
    return n


def _tps_sample(flag_on, flag):
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    k, v = _split_flag(flag)
    if k:
        env.pop(k, None)
        if flag_on:
            env[k] = v
    args = [VENV_PY, RUN_ONNX, "--benchmark", BENCH, "-i", MODEL_DIR, "-im", IMAGE,
            "-g", str(_tps_sample.gen), "-m", "2048", "-r", str(_tps_sample.reps),
            "-w", "1", "-k", "1", "-p", "1.0", "-v"]
    p = subprocess.run(args, env=env, capture_output=True, text=True,
                       cwd=os.path.dirname(RUN_ONNX))
    m = TPS_RE.search(p.stdout + p.stderr)
    return float(m.group(1)) if m else None


def measure_tps(flag, pairs, gen, reps):
    _tps_sample.gen, _tps_sample.reps = gen, reps
    diffs, offs, ons = [], [], []
    for i in range(pairs):
        off = _tps_sample(False, flag)
        on = _tps_sample(True, flag)
        if off is None or on is None:
            print(f"  pair {i+1}: FAILED (off={off} on={on})")
            continue
        diffs.append(on - off); offs.append(off); ons.append(on)
        print(f"  pair {i+1}: OFF={off:.2f} ON={on:.2f} d={on-off:+.2f}")
    if len(diffs) < 2:
        print("[tps] not enough samples"); return
    mo = sum(offs) / len(offs); mn = sum(ons) / len(ons); md = sum(diffs) / len(diffs)
    sd = math.sqrt(sum((d - md) ** 2 for d in diffs) / (len(diffs) - 1))
    ci = 1.96 * sd / math.sqrt(len(diffs))
    sig = abs(md) > ci and ci > 0
    print(f"\n[tps A/B] flag={flag}  OFF={mo:.2f}  ON={mn:.2f}  delta={md:+.2f} tps "
          f"({100*md/mo:+.1f}%)  95%CI=+/-{ci:.2f}")
    print(f"  VERDICT: {'SIGNIFICANT ' + ('GAIN' if md>0 else 'REGRESSION') if sig else 'within noise'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["readback", "tps"])
    ap.add_argument("--flag", default="", help="env flag to toggle, e.g. HIPDNN_EP_X=1")
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    if a.mode == "readback":
        measure_readback(a.flag)
    else:
        measure_tps(a.flag, a.pairs, a.gen, a.reps)


if __name__ == "__main__":
    main()
