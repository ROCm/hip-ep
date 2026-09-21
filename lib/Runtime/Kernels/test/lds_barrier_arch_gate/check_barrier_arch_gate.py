#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Compile-time guard for the arch-gated LDS barrier in matmul_nbits_kernel.hip.

The kernel's ``HIPDNN_LDS_BARRIER`` macro must emit the raw ``s_barrier``
instruction only on architectures that provide it (gfx9 / gfx10 / gfx11) and
fall back to ``__syncthreads()`` everywhere else. A newer GPU family that
retired ``s_barrier`` but was wrongly routed to the raw asm fails to
device-link -- exactly the regression this guard prevents.

The test extracts the *actual* macro definition from the kernel source (so it
can never drift from the shipping code), compiles a minimal device translation
unit for a set of public architectures, and asserts the emitted device
assembly uses the expected barrier:

    gfx9 / gfx10 / gfx11 -> standalone ``s_barrier``
    gfx12                -> no standalone ``s_barrier`` (uses the
                            ``s_barrier_signal`` / ``s_barrier_wait`` pair)

gfx12 is a real, public arch that retired s_barrier, so the device-assembly
check pins the fallback on actual hardware. Narrowing the allowlist (dropping a
supported family) makes that family emit the wrong barrier there; reverting to a
denylist is instead caught by the simulated future-family check below.

Run directly, or via CTest. Set HIP_CLANG/CLANG to point at the compiler; else
amdclang++ / clang++ / hipcc from PATH is used (hipcc works too -- the flags
below pass through the wrapper unchanged).
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL = os.path.normpath(os.path.join(HERE, "..", "..", "hip", "matmul_nbits_kernel.hip"))

# arch -> True if the standalone s_barrier instruction is expected in device asm
ARCHS = {
    "gfx906": True,    # gfx9
    "gfx1030": True,   # gfx10 (RDNA2)
    "gfx1100": True,   # gfx11 (RDNA3)
    "gfx1200": False,  # gfx12 (RDNA4) -- retired standalone s_barrier
}


def find_clang():
    # amdclang++ or the hipcc wrapper both work; the flags below pass through
    # hipcc unchanged (hipcc is the compiler exposed on Windows CI).
    for cand in (os.environ.get("HIP_CLANG"), os.environ.get("CLANG"),
                 "amdclang++", "clang++", "clang", "hipcc"):
        if not cand:
            continue
        path = cand if os.path.isabs(cand) else shutil.which(cand)
        if path and os.path.exists(path):
            return path
    sys.exit("ERROR: no HIP compiler found (set HIP_CLANG, or put amdclang++/"
             "clang++/hipcc on PATH).")


def extract_barrier_block(src_path):
    """Return the '#if ... #define HIPDNN_LDS_BARRIER ... #endif' block verbatim."""
    lines = open(src_path).read().splitlines()
    define_idx = next(
        (i for i, ln in enumerate(lines)
         if ln.lstrip().startswith("#define") and "HIPDNN_LDS_BARRIER(" in ln),
        None,
    )
    if define_idx is None:
        sys.exit(f"ERROR: HIPDNN_LDS_BARRIER not found in {src_path}")
    start = define_idx
    while start >= 0 and not lines[start].lstrip().startswith("#if"):
        start -= 1
    depth, end = 0, start
    while end < len(lines):
        s = lines[end].lstrip()
        if s.startswith("#if"):
            depth += 1
        elif s.startswith("#endif"):
            depth -= 1
            if depth == 0:
                break
        end += 1
    return "\n".join(lines[start:end + 1])


def emit_device_asm(clang, arch, tu_path):
    out = subprocess.run(
        [clang, "-x", "hip", "--cuda-device-only", "-S", "-O2",
         f"--offload-arch={arch}", "-nogpulib", "-o", "-", tu_path],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        return None, out.stderr
    return out.stdout, None


def standalone_s_barrier_count(text):
    # count `s_barrier` that is NOT s_barrier_signal / s_barrier_wait / etc.
    return len(re.findall(r"\bs_barrier\b(?!_)", text))


# Hypothetical / simulated architectures for the preprocessor logic check. These
# are NOT real products -- they exercise the *gate logic* with only the family
# macros a given arch would define, so we can prove the allowlist behaviour
# (including that an unknown future family falls back) without a device target.
#   defines set    -> expect standalone s_barrier?
SIM_CASES = {
    "gfx9 (sim)":            (["__GFX9__"], True),
    "gfx10 (sim)":           (["__GFX10__"], True),
    "gfx11 (sim)":           (["__GFX11__"], True),
    "gfx12 (sim)":           (["__GFX12__"], False),
    "unknown future family": (["__GFX99__"], False),  # denylist would wrongly pick s_barrier
    "device, no family macro": ([], False),           # unrecognised target must fall back
}


def gate_branch_preprocess(clang, block, family_defines):
    """Preprocess the real macro block with simulated family macros; return
    True if HIPDNN_LDS_BARRIER expands to the raw s_barrier asm."""
    tu = block + "\nHIPDNN_LDS_BARRIER_PROBE HIPDNN_LDS_BARRIER()\n"
    with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as f:
        f.write(tu)
        path = f.name
    try:
        cmd = [clang, "-x", "c", "-E", "-P",
               "-D__HIP_DEVICE_COMPILE__", "-D__AMDGCN__"]
        cmd += [f"-D{d}" for d in family_defines]
        cmd += [path]
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            return None, out.stderr
        line = next((l for l in out.stdout.splitlines()
                     if "HIPDNN_LDS_BARRIER_PROBE" in l), "")
        # The two branches expand distinctly: the s_barrier branch emits the
        # inline-asm string (contains "s_barrier"); the fallback emits
        # "__syncthreads". Match on the fallback token first since it is
        # unambiguous, then on the asm.
        if "__syncthreads" in line:
            return False, None
        if "s_barrier" in line:
            return True, None
        return None, f"could not classify barrier expansion: {line!r}"
    finally:
        os.unlink(path)


def main():
    clang = find_clang()
    block = extract_barrier_block(KERNEL)
    failures = []
    print(f"clang : {clang}")
    print(f"kernel: {KERNEL}\n")

    # --- Part A: real device assembly (asm must actually assemble per arch) ---
    tu = (
        "#include <hip/hip_runtime.h>\n" + block + "\n"
        "__global__ void _probe(int* p) {\n"
        "    HIPDNN_LDS_BARRIER();\n"
        "    p[threadIdx.x] = 1;\n"
        "}\n"
    )
    print("[A] device assembly (real archs)")
    print("    arch      expect        emitted        result")
    with tempfile.TemporaryDirectory() as td:
        tu_path = os.path.join(td, "barrier_probe.hip")
        open(tu_path, "w").write(tu)
        for arch, expect_sb in ARCHS.items():
            asm, err = emit_device_asm(clang, arch, tu_path)
            if asm is None:
                failures.append(f"[A] {arch}: compile failed:\n{err}")
                print(f"    {arch:<8}  {'s_barrier' if expect_sb else 'fallback':<12}  COMPILE-FAIL   FAIL")
                continue
            has_sb = standalone_s_barrier_count(asm) > 0
            ok = has_sb == expect_sb
            print(f"    {arch:<8}  {'s_barrier' if expect_sb else 'fallback':<12}  "
                  f"{'s_barrier' if has_sb else 'signal/wait':<13}  {'ok' if ok else 'FAIL'}")
            if not ok:
                failures.append(f"[A] {arch}: expected {'s_barrier' if expect_sb else 'fallback'}")

    # --- Part B: gate logic, incl. a hypothetical future family (denylist trap) ---
    print("\n[B] gate logic (simulated families)")
    print("    case                     expect        selected      result")
    for name, (defs, expect_sb) in SIM_CASES.items():
        sel, err = gate_branch_preprocess(clang, block, defs)
        if sel is None:
            failures.append(f"[B] {name}: preprocess failed:\n{err}")
            print(f"    {name:<24}  {'s_barrier' if expect_sb else 'fallback':<12}  PREPROC-FAIL  FAIL")
            continue
        ok = sel == expect_sb
        print(f"    {name:<24}  {'s_barrier' if expect_sb else 'fallback':<12}  "
              f"{'s_barrier' if sel else 'fallback':<12}  {'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append(
                f"[B] {name}: expected {'s_barrier' if expect_sb else 'fallback'}, "
                f"gate selected {'s_barrier' if sel else 'fallback'}"
            )

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nOK: LDS barrier gate emits the correct instruction on every arch,")
    print("    and an unknown future family correctly falls back to __syncthreads().")


if __name__ == "__main__":
    main()
