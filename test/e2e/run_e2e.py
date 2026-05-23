"""E2E test runner: compile .mlir -> .dll, then execute and validate.

Invoked by Bazel py_test via sys.argv:
  argv[1]  path to hip-compiler executable
  argv[2]  path to .mlir input file
  argv[3]  path to hip-test-dll executable

Uses TEST_TMPDIR (set by Bazel test runner) for isolated output so each test
gets its own scratch directory and tests can run in parallel safely.

On Windows, hip-compiler invokes lld-link internally to produce the model DLL.
lld-link searches the LIB environment variable for system import libraries
(msvcrt.lib, ucrt.lib, kernel32.lib, etc.).  The Bazel toolchain stages these
via @bazel_cpp_toolchain//toolchain:clang_toolchain_linker_lib_files, so we
locate their directories and prepend them to LIB before calling hip-compiler.
"""
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

import os
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# Lib-dir discovery.
#
# The staged lib files live under known rlocation paths:
#   bazel_cpp_toolchain++local_tool_deps+msvc/lib/x64/
#   bazel_cpp_toolchain++local_tool_deps+windows_sdk/Lib/ucrt/x64/
#   bazel_cpp_toolchain++local_tool_deps+windows_sdk/Lib/um/x64/
#
# On RBE workers RUNFILES_MANIFEST_FILE is not set; the runfiles are staged
# as a physical directory tree under RUNFILES_DIR / TEST_SRCDIR.
# On local dev RUNFILES_MANIFEST_FILE is set; we scan it for sentinel files.
# ---------------------------------------------------------------------------
_LIB_SENTINELS = frozenset([
    "msvcrt.lib", "vcruntime.lib", "oldnames.lib", "libcpmt.lib", "libcmt.lib",
    "ucrt.lib",
    "kernel32.lib", "user32.lib",
])

_MSVC_REPO = "bazel_cpp_toolchain++local_tool_deps+msvc"
_SDK_REPO  = "bazel_cpp_toolchain++local_tool_deps+windows_sdk"


def _discover_lib_dirs():
    """Return list of directories containing Windows SDK / MSVC .lib files."""
    # Strategy 1: physical runfiles tree (RBE workers).
    # RUNFILES_MANIFEST_FILE is empty on the worker; RUNFILES_DIR points to
    # the staged tree. Evidence from worker:
    #   RUNFILES_DIR='D:\\b\\AA\\tmp\\Bazel.runfiles_...\\runfiles'
    #   TEST_SRCDIR='D:/b/AA/bazel-out/.../e2e_test_add_model.exe.runfiles'
    for env_var in ("RUNFILES_DIR", "TEST_SRCDIR"):
        rd = os.environ.get(env_var, "")
        if not (rd and os.path.isdir(rd)):
            continue
        candidates = [
            os.path.join(rd, _MSVC_REPO, "lib", "x64"),
            os.path.join(rd, _SDK_REPO,  "Lib", "ucrt", "x64"),
            os.path.join(rd, _SDK_REPO,  "Lib", "um",   "x64"),
        ]
        dirs = [d for d in candidates if os.path.isdir(d)]
        if dirs:
            return dirs

    # Strategy 2: manifest file (local dev / some CI setups).
    manifest = os.environ.get("RUNFILES_MANIFEST_FILE", "")
    if manifest and os.path.isfile(manifest):
        dirs = []
        seen = set()
        with open(manifest, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split(" ", 1)
                if len(parts) != 2:
                    continue
                key, path = parts
                if os.path.basename(key).lower() in _LIB_SENTINELS:
                    d = os.path.dirname(path)
                    if d and d not in seen and os.path.isdir(d):
                        dirs.append(d)
                        seen.add(d)
        if dirs:
            return dirs

    return []


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <hip-compiler> <mlir-file> <hip-test-dll>",
              file=sys.stderr)
        sys.exit(1)

    hip_compiler = os.path.abspath(sys.argv[1])
    mlir_file = os.path.abspath(sys.argv[2])
    hip_test_dll = os.path.abspath(sys.argv[3])

    work_dir = os.environ.get("TEST_TMPDIR") or tempfile.mkdtemp()
    model_name = os.path.splitext(os.path.basename(mlir_file))[0]
    dll_output = os.path.join(work_dir, model_name + ".dll")

    # Prepend staged lib dirs to LIB so lld-link (invoked inside hip-compiler)
    # can resolve msvcrt.lib, ucrt.lib, kernel32.lib, etc. on RBE workers.
    lib_dirs = _discover_lib_dirs()
    if lib_dirs:
        existing_lib = os.environ.get("LIB", "")
        os.environ["LIB"] = os.pathsep.join(
            lib_dirs + ([existing_lib] if existing_lib else [])
        )

    print(f"=== Compiling: {mlir_file} ===", flush=True)
    result = subprocess.run(
        [hip_compiler, mlir_file, "-o", dll_output],
        cwd=work_dir,
        check=False,
    )
    if result.returncode != 0:
        print(f"FAILED: hip-compiler exited with {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"=== Executing: {dll_output} ===", flush=True)
    result = subprocess.run(
        [hip_test_dll, dll_output, "--verbose", "--validate"],
        cwd=work_dir,
        check=False,
    )
    if result.returncode != 0:
        print(f"FAILED: hip-test-dll exited with {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"=== PASSED: {model_name} ===", flush=True)


if __name__ == "__main__":
    main()
