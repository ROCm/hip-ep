#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Compile every custom-kernel source with the product flags and compare each
kernel's register / scratch / occupancy report and the shape of its ISA against
a checked-in baseline.

Run it whenever the HIP SDK (hipcc / clang) changes. A new compiler can need
more registers for unchanged source, and a kernel whose register budget is
close to its need then spills; or it can emit narrower memory operations for
unchanged source. Neither shows up in correctness tests. ROCm 7.14 -> 7.16 did
both: one MatMulNBits WMMA tile went from 3 to 138 spilled values (3.5-4x
slower), and every unaligned __shared__ array lost its 16-byte LDS loads
(QDQ Gemm decode 1.3x slower, fp32 Conv 1.4x).

  python check_kernel_resources.py --rocm <HIP SDK>                # compare
  python check_kernel_resources.py --rocm <HIP SDK> --update       # rewrite baseline
  python check_kernel_resources.py --rocm <HIP SDK> --squeeze 112  # headroom report

Exit status 1 when a kernel gained scratch (bytes per lane) or spilled VGPRs
over the baseline, a kernel not in the baseline uses scratch and is not
allowlisted, or a kernel's LDS loads got narrower (more ds_load instructions
for no more bytes). Occupancy drops and instruction-count growth are warnings
(--strict: errors). The baseline is one JSON per arch next to this script:
kernel_resources_<arch>.json = {"compiler": ..., "fields": [...],
"allow": {kernel: reason}, "kernels": {source: {kernel: [values]}}}.

The default archs are the two of the product build (gfx11-generic, gfx1170)
and gfx1151. Register budgets follow the compile target (HIPDNN_VGPRS_PER_SIMD
in include/hip_arch_compat.h): gfx1151 has a 1536-VGPR file where
gfx11-generic, the binary gfx1150 / 1151 / 1152 run, has 1024. A kernel can be
fine in the product binary and regress in a gfx1151-target build (the LUT
sweeps' historical target and the usual developer build), or the other way
round, so both are checked.

--squeeze N recompiles with -mllvm -amdgpu-stress-vgpr=N (clang 24 / ROCm 7.16
and later) and lists the spill-free kernels using between N and --squeeze-max
VGPRs (default: an eighth of the arch's register file, 128 or 192) that spill
once only N are available, i.e. have less than (budget - N) registers of
headroom. See README.md next to this script.
"""
import argparse
import collections
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNELS = HERE.parent  # lib/Runtime/Kernels
# Wave32 VGPRs per SIMD, as HIPDNN_VGPRS_PER_SIMD in include/hip_arch_compat.h;
# every other arch (and gfx11-generic) has 1024.
VGPRS_PER_SIMD = {"gfx1100": 1536, "gfx1101": 1536, "gfx1151": 1536}
REMARKS = {"VGPRs": 0, "ScratchSize [bytes/lane]": 1, "VGPRs Spill": 2, "Occupancy [waves/SIMD]": 3}
NAMES = ("vgpr", "scratch", "vgpr_spill", "occupancy", "insns", "lds_loads", "lds_load_bytes")
LABEL = re.compile(r"^(_Z\w+):")
INSN = re.compile(r"^\s+([a-z_][a-z0-9_]*)\b")
LDS_LOAD = re.compile(r"ds_(?:load|read)(?:_addtid)?_(2addr_(?:stride64_)?)?([biu])(\d+)")


def kernel_sources():
    text = (KERNELS / "CMakeLists.txt").read_text()
    block = text[text.index("set(_kernel_sources"):]
    block = block[:block.index(")")]
    return re.findall(r"(hip/[\w./-]+\.hip)", block)


def hipcc_env(rocm):
    rocm = Path(rocm)
    exe = rocm / "bin" / ("hipcc.exe" if os.name == "nt" else "hipcc")
    env = dict(os.environ, HIP_PATH=str(rocm))
    # Some SDK layouts ship clang in bin/ rather than lib/llvm/bin, where hipcc looks.
    clang = "clang.exe" if os.name == "nt" else "clang"
    if not (rocm / "lib" / "llvm" / "bin" / clang).exists() and (rocm / "bin" / clang).exists():
        env["HIP_CLANG_PATH"] = str(rocm / "bin")
    return exe, env


def product_flags(arch):
    # Mirrors cmake/hip_utils.cmake (Release) for the custom_kernels library.
    flags = ["-S", "--cuda-device-only", f"--offload-arch={arch}", "-O2", "-DNDEBUG",
             "-D_ITERATOR_DEBUG_LEVEL=0", "-DHIP_CUSTOM_KERNELS_EXPORTS", "-Wno-ignored-attributes"]
    if os.name == "nt":
        flags += ["-fms-extensions", "-fms-compatibility", "-fexceptions",
                  "-Xclang", "-fno-cuda-host-device-constexpr",
                  "-include", str(KERNELS / "include" / "msvc_hip_cmath_workaround.h")]
    for inc in ("include", "hip", "hip/autotune/gqa", "hip/autotune/matmul_nbits"):
        flags += ["-I", str(KERNELS / inc)]
    return flags


def parse_remarks(log):
    out, cur = {}, None
    for line in log.splitlines():
        m = re.search(r"remark: Function Name: (\S+)", line)
        if m:
            cur = m.group(1)
            out[cur] = [0, 0, 0, 0]
            continue
        m = re.search(r"remark:\s+(.+?): (\d+) \[-Rpass", line)
        if m and cur and m.group(1) in REMARKS:
            out[cur][REMARKS[m.group(1)]] = int(m.group(2))
    return out


def parse_isa(path):
    """Per kernel: [instructions, LDS load instructions, bytes they load]."""
    out, cur = {}, None
    with open(path, errors="replace") as fh:
        for line in fh:
            m = LABEL.match(line)
            if m:
                cur = out.setdefault(m.group(1), [0, 0, 0])
                continue
            if cur is None:
                continue
            if line.startswith(".Lfunc_end"):
                cur = None
                continue
            m = INSN.match(line)
            if not m:
                continue
            cur[0] += 1
            m = LDS_LOAD.match(m.group(1))
            if m:
                cur[1] += 1
                cur[2] += int(m.group(3)) // 8 * (2 if m.group(1) else 1)
    return out


def compile_one(exe, env, arch, rel, extra, tmp):
    asm = Path(tmp) / (re.sub(r"\W", "_", f"{arch}_{rel}_{len(extra)}") + ".s")
    cmd = [str(exe), *product_flags(arch), *extra, "-Rpass-analysis=kernel-resource-usage",
           str(KERNELS / rel), "-o", str(asm)]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, errors="replace")
    kern = {}
    if p.returncode == 0:
        res = parse_remarks(p.stdout + p.stderr)
        isa = parse_isa(asm)
        kern = {k: v + isa.get(k, [0, 0, 0]) for k, v in res.items()}
    try:
        asm.unlink()
    except FileNotFoundError:
        pass
    return arch, Path(rel).stem, p.returncode, kern, (p.stderr if p.returncode else "")


def run_all(rocm, archs, sources, jobs, extra=()):
    exe, env = hipcc_env(rocm)
    res, errors = {a: {} for a in archs}, []
    heavy = ("matmul_nbits", "gqa", "qmoe", "qgemm", "qmatmul", "gemm_wmma", "linear_attention", "conv")
    tasks = sorted(((a, s) for a in archs for s in sources), key=lambda t: not any(h in t[1] for h in heavy))
    with tempfile.TemporaryDirectory() as tmp, cf.ThreadPoolExecutor(jobs) as ex:
        futs = [ex.submit(compile_one, exe, env, a, s, list(extra), tmp) for a, s in tasks]
        for f in cf.as_completed(futs):
            arch, stem, rc, kern, err = f.result()
            if rc:
                errors.append(f"{arch} {stem}: hipcc rc={rc}\n{err[-2000:]}")
            res[arch][stem] = kern
    return res, errors


def compiler_version(rocm):
    exe, env = hipcc_env(rocm)
    p = subprocess.run([str(exe), "--version"], env=env, capture_output=True, text=True, errors="replace")
    v = [line.strip() for line in p.stdout.splitlines() if "version" in line.lower()]
    return "; ".join(v[:2])


def baseline_path(arch):
    return HERE / f"kernel_resources_{arch}.json"


def compare(cur, base, strict, insn_growth):
    errors, warns, infos = [], [], []
    allow = base.get("allow", {})
    bk = base.get("kernels", {})
    for src, kern in sorted(cur.items()):
        old = bk.get(src, {})
        for k, v in sorted(kern.items()):
            o = old.get(k)
            if o is None:
                if v[1] > 0 and k not in allow:
                    errors.append(f"NEW KERNEL WITH SCRATCH {src}: {k} vgpr={v[0]} scratch={v[1]} spill={v[2]}")
                else:
                    infos.append(f"new kernel {src}: {k} vgpr={v[0]} scratch={v[1]} occ={v[3]}")
                continue
            d, b = dict(zip(NAMES, v)), dict(zip(NAMES, o))
            if d["scratch"] > b["scratch"] or d["vgpr_spill"] > b["vgpr_spill"]:
                kind = "NEW SCRATCH" if b["scratch"] == 0 else "MORE SCRATCH"
                errors.append(f"{kind} {src}: {k} scratch {b['scratch']}->{d['scratch']} "
                              f"spill {b['vgpr_spill']}->{d['vgpr_spill']} vgpr {b['vgpr']}->{d['vgpr']}")
            # Narrower LDS loads: more instructions moving no more bytes.
            if d["lds_loads"] > b["lds_loads"] * 1.1 and d["lds_load_bytes"] <= b["lds_load_bytes"]:
                errors.append(f"NARROWER LDS LOADS {src}: {k} {b['lds_loads']} loads / {b['lds_load_bytes']} B "
                              f"-> {d['lds_loads']} / {d['lds_load_bytes']} B")
            if d["occupancy"] < b["occupancy"]:
                (errors if strict else warns).append(
                    f"OCCUPANCY DROP {src}: {k} occ {b['occupancy']}->{d['occupancy']} vgpr {b['vgpr']}->{d['vgpr']}")
            if b["insns"] and d["insns"] > b["insns"] * (1 + insn_growth):
                (errors if strict else warns).append(
                    f"MORE INSTRUCTIONS {src}: {k} {b['insns']}->{d['insns']} "
                    f"(+{100 * (d['insns'] / b['insns'] - 1):.0f}%)")
        for k in sorted(set(old) - set(kern)):
            infos.append(f"kernel gone {src}: {k}")
    return errors, warns, infos


def write_baseline(arch, cur, rocm):
    path = baseline_path(arch)
    old = json.loads(path.read_text()) if path.exists() else {}
    allow = old.get("allow", {})
    for src, kern in cur.items():
        for k, v in kern.items():
            if v[1] > 0 and k not in allow:
                allow[k] = f"scratch {v[1]} B/lane when the baseline was taken ({src})"
    allow = {k: r for k, r in allow.items() if any(k in kern for kern in cur.values())}
    data = {"compiler": compiler_version(rocm), "fields": list(NAMES), "allow": dict(sorted(allow.items())),
            "kernels": {s: dict(sorted(kern.items())) for s, kern in sorted(cur.items())}}
    # One kernel per line: small diffs when the baseline is updated.
    lines = ["{", f'"compiler":{json.dumps(data["compiler"])},', f'"fields":{json.dumps(data["fields"])},',
             '"allow":{']
    lines += [f"{json.dumps(k)}:{json.dumps(r)}," for k, r in data["allow"].items()]
    if data["allow"]:
        lines[-1] = lines[-1][:-1]
    lines.append('},"kernels":{')
    srcs = list(data["kernels"].items())
    for i, (s, kern) in enumerate(srcs):
        lines.append(f"{json.dumps(s)}:{{")
        items = list(kern.items())
        lines += [f"{json.dumps(k)}:{json.dumps(v, separators=(',', ':'))}" + ("," if j < len(items) - 1 else "")
                  for j, (k, v) in enumerate(items)]
        lines.append("}" + ("," if i < len(srcs) - 1 else ""))
    lines.append("}}")
    path.write_text("\n".join(lines) + "\n")
    return path, sum(len(v) for v in cur.values()), len(allow)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rocm", default=os.environ.get("HIP_PATH", ""), help="HIP SDK root (default: $HIP_PATH)")
    ap.add_argument("--archs", default="gfx11-generic,gfx1170,gfx1151",
                    help="offload archs: the product build's two and the 1536-VGPR gfx1151 target")
    ap.add_argument("--sources", default="", help="comma list of source stems (default: all custom kernels)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--update", action="store_true", help="write the current results as the baseline")
    ap.add_argument("--strict", action="store_true", help="occupancy drops and instruction growth are errors too")
    ap.add_argument("--insn-growth", type=float, default=0.10,
                    help="warn when a kernel has this many more instructions (default 0.10 = 10%%)")
    ap.add_argument("--squeeze", type=int, default=0, help="also list kernels without N-VGPR headroom")
    ap.add_argument("--squeeze-max", type=int, default=0,
                    help="with --squeeze: only kernels using more than N and at most this many VGPRs "
                         "(default: the budget of a 1024-thread block or waves_per_eu(8), i.e. 128 on a "
                         "1024-VGPR target and 192 on gfx1151; 256 for the architectural limit)")
    ap.add_argument("--json-out", default="", help="write the raw results here")
    a = ap.parse_args()
    if not a.rocm:
        sys.exit("--rocm (or HIP_PATH) is required")
    archs = [x for x in a.archs.split(",") if x]
    srcs = kernel_sources()
    if a.sources:
        want = set(a.sources.split(","))
        srcs = [s for s in srcs if Path(s).stem in want]
    print(f"compiler: {compiler_version(a.rocm)}; {len(srcs)} sources x {len(archs)} archs, jobs={a.jobs}", flush=True)
    cur, cerr = run_all(a.rocm, archs, srcs, a.jobs)
    for e in cerr:
        print("COMPILE ERROR", e)
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(cur))
    if cerr:
        return 2
    rc = 0
    for arch in archs:
        if a.update:
            path, n, nallow = write_baseline(arch, cur[arch], a.rocm)
            print(f"{arch}: baseline written to {path.name} ({n} kernels, {nallow} allowlisted with scratch)")
            continue
        path = baseline_path(arch)
        if not path.exists():
            print(f"{arch}: no baseline {path.name}; run with --update first")
            rc = 1
            continue
        base = json.loads(path.read_text())
        if base.get("fields") != list(NAMES):
            print(f"{arch}: baseline fields {base.get('fields')} != {list(NAMES)}; regenerate it with --update")
            rc = 1
            continue
        errors, warns, infos = compare(cur[arch], base, a.strict, a.insn_growth)
        print(f"== {arch}: baseline compiler {base.get('compiler', '?')}")
        for line in errors:
            print("  ERROR", line)
        for line in warns:
            print("  WARN ", line)
        for line in infos[:50]:
            print("  info ", line)
        if len(infos) > 50:
            print(f"  info  ... {len(infos) - 50} more")
        print(f"  {len(errors)} errors, {len(warns)} warnings, {len(infos)} notes")
        if errors:
            rc = 1
    if a.squeeze:
        sq, serr = run_all(a.rocm, archs, srcs, a.jobs, ["-mllvm", f"-amdgpu-stress-vgpr={a.squeeze}"])
        for e in serr:
            print("COMPILE ERROR (squeeze)", e)
        for arch in archs:
            band = a.squeeze_max or VGPRS_PER_SIMD.get(arch, 1024) // 8
            tight = collections.defaultdict(list)
            for src, kern in cur[arch].items():
                for k, v in kern.items():
                    s = sq[arch].get(src, {}).get(k)
                    # Spill-free now, between N and the band's budget, spills at N:
                    # less than (budget - N) registers of margin. The remark does
                    # not report the budget itself, so a kernel with a larger one
                    # that happens to sit in the band is listed too; read TIGHT
                    # as "review", not "fail".
                    if s and v[1] == 0 and a.squeeze < v[0] <= band and s[1] > 0:
                        tight[src].append(f"{k} vgpr={v[0]} -> scratch {s[1]} B at {a.squeeze} VGPRs")
            n = sum(len(v) for v in tight.values())
            print(f"== {arch}: {n} spill-free kernels with {a.squeeze}-{band} VGPRs "
                  f"that spill with {a.squeeze}")
            for src in sorted(tight):
                for line in sorted(tight[src]):
                    print(f"  TIGHT {src}: {line}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
