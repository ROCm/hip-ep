"""Smoke-test wrapper for benchmark_multimodal.py against the user's locally
built OGA (0.13.0.dev0 / ORT 1.26.0 / MorphiZenEP plugin).

Workarounds folded in (see docstring below for the four-version skew):

1. Pre-load locally-built ORT 1.26 onnxruntime.dll via ctypes BEFORE
   `import onnxruntime_genai`. OGA's `_dll_directory.py` checks
   `GetModuleHandleW("onnxruntime.dll")` first and skips its
   `find_spec("onnxruntime")` path (which would otherwise pick up the
   incompatible onnxruntime-vitisai 1.24.3 API-24 DLL).

2. Pre-load DirectML.dll from the OGA build tree (DML 1.15.4 NuGet drop-in
   bundled with the OGA build); the locally-built ORT 1.26 install dir
   does not bundle DirectML.dll.

3. chdir to prebuilt-local\\bin so OGA's plugin-EP loader resolves
   `onnxruntime_morphizen_ep.dll` via its cwd-relative LoadLibraryW.

4. Self-contained env: sets THEROCK_DIST + HIP_CUSTOM_KERNELS_DIR
   (required by the EP's JIT-link discoverLibraries) and HIPDNN_EP_PERF=1
   (enables PR #273's per-Compute + per-op profiling streams), all via
   setdefault so a shell-side override wins. Set HIPDNN_EP_PERF=0 in the
   shell to disable profiling entirely.

Usage:

    python oga_mm_smoketest.py [benchmark_multimodal.py args...]

Any args you pass are forwarded to benchmark_multimodal.py and override the
defaults below. Examples:

    # Default smoke test (whatever's hardcoded in DEFAULT_ARGV)
    python oga_mm_smoketest.py

    # Override generation length + warmup
    python oga_mm_smoketest.py -g 32 -m 768 -w 2 -r 3

    # Different model + image, same defaults otherwise
    python oga_mm_smoketest.py -i C:\\other\\model -im C:\\other\\img.jpg

Perf log inspection: with HIPDNN_EP_PERF=1 (default), the EP and runtime
emit `[PERF SUMMARY]` (process exit) and `[PERF] === ... ===` (per-Compute)
blocks. Capture stdout+stderr with PowerShell:

    python oga_mm_smoketest.py 2>&1 | Tee-Object -FilePath bench.log

Then:

    Select-String -Path bench.log -Pattern '\\[PERF SUMMARY\\]' -Context 0,20
    Select-String -Path bench.log -Pattern '\\[PERF\\] ===' -Context 0,40 | Select-Object -Last 50

(format_perf_report.py from PR #273 only handles model_benchmark.exe logs,
not benchmark_multimodal.py's output -- so manual grep is the path for
multimodal until/unless the formatter is extended.)
"""
import ctypes
import os
import sys

# ---------------------------------------------------------------------------
# Paths -- edit if you move any of these
# ---------------------------------------------------------------------------
ORT_126_DIR = r"C:\Users\Administrator\workspace\onnx-hipdnn-ep-qwen35\install\onnxruntime\lib"
PREBUILT_BIN = r"C:\Users\Administrator\workspace\prebuilt-local\bin"
PREBUILT_LIB = r"C:\Users\Administrator\workspace\prebuilt-local\lib"
THEROCK_DIST = r"C:\Users\Administrator\workspace\therock-dist"
THEROCK_BIN = os.path.join(THEROCK_DIST, "bin")
OGA_DIR = r"C:\Python3\Python310\lib\site-packages\onnxruntime_genai"
DML_DLL = r"C:\Users\Administrator\workspace\onnx-hipdnn-ep-qwen35\install\oga-build\RelWithDebInfo\_deps\dmllib-src\bin\x64-win\DirectML.dll"
BENCHMARK = r"C:\Users\Administrator\workspace\onnxruntime-genai\benchmark\python\benchmark_multimodal.py"

# ---------------------------------------------------------------------------
# Default benchmark args -- overridden by anything passed on the CLI
# ---------------------------------------------------------------------------
DEFAULT_ARGV = [
    "-i", r"C:\Users\Administrator\Downloads\Qwen3.5-9B-rtn-int4-int8-128gs-fp16-onnx-gpu",
    "--image_path", r"C:\Users\Administrator\workspace\onnx-hipdnn-ep\test\python\images\tower.jpg",
    "-g", "4",
    "-m", "512",
    "-r", "1",
    "-w", "1",
    "-o", r"C:\Users\Administrator\workspace\build\tower_mm_smoketest.csv",
    "-v",
]

# ---------------------------------------------------------------------------
# Self-contained env (setdefault so shell-side overrides win)
# ---------------------------------------------------------------------------
os.environ.setdefault("THEROCK_DIST", THEROCK_DIST)
os.environ.setdefault("HIP_CUSTOM_KERNELS_DIR", PREBUILT_LIB)
os.environ.setdefault("HIPDNN_EP_PERF", "1")
print(f"[wrap] HIPDNN_EP_PERF={os.environ['HIPDNN_EP_PERF']}  "
      f"THEROCK_DIST={os.environ['THEROCK_DIST']}", flush=True)

# ---------------------------------------------------------------------------
# DLL search-path setup (+ PATH for any indirect lookups)
# ---------------------------------------------------------------------------
for d in (THEROCK_BIN, PREBUILT_BIN, ORT_126_DIR, OGA_DIR, os.path.dirname(DML_DLL)):
    if os.path.isdir(d):
        os.add_dll_directory(d)
        os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]

# ---------------------------------------------------------------------------
# Pre-load workarounds (see module docstring)
# ---------------------------------------------------------------------------
ctypes.CDLL(DML_DLL)
ctypes.CDLL(os.path.join(ORT_126_DIR, "onnxruntime_providers_shared.dll"))
ort = ctypes.CDLL(os.path.join(ORT_126_DIR, "onnxruntime.dll"))
print(f"[wrap] Pre-loaded ORT 1.26 onnxruntime.dll handle: {ort._handle:#x}", flush=True)

os.chdir(PREBUILT_BIN)
print(f"[wrap] cwd -> {os.getcwd()}", flush=True)

# ---------------------------------------------------------------------------
# Assemble argv: DEFAULT_ARGV first, then anything the user passed
# (argparse takes the LAST occurrence of each flag, so user args win)
# ---------------------------------------------------------------------------
user_args = sys.argv[1:]
sys.argv = [BENCHMARK] + DEFAULT_ARGV + user_args
print(f"[wrap] argv: {sys.argv[1:]}", flush=True)

with open(BENCHMARK) as f:
    code = compile(f.read(), BENCHMARK, "exec")
exec(code, {"__name__": "__main__", "__file__": BENCHMARK})

print("", flush=True)
print("[wrap] DONE. With HIPDNN_EP_PERF=1, look in your captured log for:",
      flush=True)
print("[wrap]   [PERF SUMMARY]        -- per-Compute aggregates (process exit)",
      flush=True)
print("[wrap]   [PERF] === ... ===    -- per-op GPU/CPU table (one per Compute())",
      flush=True)
