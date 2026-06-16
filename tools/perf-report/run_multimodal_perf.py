#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Portable smoke-test runner for benchmark_multimodal.py on a locally-built
MorphiZen EP / OGA (ORT 1.26 / OGA 0.13 / MorphiZenEP plugin).

To run on a new machine: set the paths at the top of this file (or the matching
env vars). The install-tree paths derive from REPO_ROOT. Then:

    python run_multimodal_perf.py -i <model_dir> --image_path <img> --perf 0 --report

Our own flags (consumed here, NOT forwarded to the benchmark):
  --perf {0,1}    set HIPDNN_EP_PERF (default 1). Use 0 for true throughput
                  (PERF=1 adds profiling fences and is NOT a throughput number).
  --log PATH      tee ALL output (incl. native EP [PERF] lines) to a UTF-8 file.
                  Default: <OUT_DIR>/<image>_smoke.log
  --report        after the run, render perf_multimodal_report.py on the log + CSV.
  --keep-config   do not restore the model's genai_config.json after the run.
Any other args are forwarded to benchmark_multimodal.py (e.g. -g -m -r -w -k).

Workarounds folded in (4-way version skew): pre-load locally-built ORT 1.26
onnxruntime.dll + DirectML.dll before importing onnxruntime_genai, and chdir to
the EP bin dir so the plugin loader resolves the EP DLL.
"""

import atexit
import ctypes
import os
import shutil
import subprocess
import sys

# ===================== EDIT THESE PATHS PER MACHINE =====================
# Each may also be set via the env var in [brackets]; the env var wins.
REPO_ROOT = os.environ.get(
    "HIPDNN_EP_ROOT", r"C:\Users\Administrator\workspace\onnx-hipdnn-ep"
)  # onnx-hipdnn-ep checkout (contains install/)
THEROCK = os.environ.get(
    "THEROCK_DIST", r"C:\Users\Administrator\workspace\therock-7.11"
)  # TheRock ROCm SDK dir
OGA_DIR = os.environ.get(
    "HIPDNN_EP_OGA", r"C:\Python3\Python310\lib\site-packages\onnxruntime_genai"
)  # onnxruntime_genai package dir
DML_DLL = os.environ.get(
    "HIPDNN_EP_DML",
    r"C:\Users\Administrator\workspace\onnx-hipdnn-ep\install\oga-build\RelWithDebInfo\_deps\dmllib-src\bin\x64-win\DirectML.dll",
)  # x64 DirectML.dll
MODEL_DIR = os.environ.get("HIPDNN_EP_MODEL", r"")  # OGA model dir (or pass -i)
IMAGE_PATH = os.environ.get("HIPDNN_EP_IMAGE", r"")  # test image (or pass --image_path)
OUT_DIR = os.environ.get(
    "HIPDNN_EP_OUT", os.getcwd()
)  # where the CSV + log are written
# ========================================================================

# Install-tree paths (fixed layout under REPO_ROOT).
ORT_126_DIR = os.path.join(REPO_ROOT, "install", "onnxruntime", "lib")
PREBUILT_BIN = os.path.join(REPO_ROOT, "install", "dist", "bin")
PREBUILT_LIB = os.path.join(REPO_ROOT, "install", "dist", "lib")
BENCHMARK = os.path.join(
    REPO_ROOT, "install", "oga-source", "benchmark", "python", "benchmark_multimodal.py"
)
THEROCK_BIN = os.path.join(THEROCK, "bin")
REPORT_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "perf_multimodal_report.py"
)


# ---- Child mode: do the preloads + exec the benchmark in THIS process --------
# We re-exec ourselves as a child so the parent can capture the child's combined
# stdout/stderr (including native EP fd-1/2 output) over a pipe and write a UTF-8
# log. A child process flushes its C-runtime stdio buffers on exit *into the
# pipe the parent is still draining*, so nothing is lost (unlike redirecting our
# own fds, where the final flush would race the teardown).
def _run_child(bench_argv):
    for d in (
        THEROCK_BIN,
        PREBUILT_BIN,
        ORT_126_DIR,
        OGA_DIR,
        os.path.dirname(DML_DLL) if DML_DLL else "",
    ):
        if d and os.path.isdir(d):
            os.add_dll_directory(d)
            os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
    if DML_DLL and os.path.isfile(DML_DLL):
        ctypes.CDLL(DML_DLL)
    shared = os.path.join(ORT_126_DIR, "onnxruntime_providers_shared.dll")
    if os.path.isfile(shared):
        ctypes.CDLL(shared)
    ort = ctypes.CDLL(os.path.join(ORT_126_DIR, "onnxruntime.dll"))
    print(
        f"[wrap] Pre-loaded ORT 1.26 onnxruntime.dll handle: {ort._handle:#x}",
        flush=True,
    )
    os.chdir(PREBUILT_BIN)
    print(f"[wrap] cwd -> {os.getcwd()}", flush=True)
    sys.argv = [BENCHMARK] + list(bench_argv)
    with open(BENCHMARK) as f:
        code = compile(f.read(), BENCHMARK, "exec")
    exec(code, {"__name__": "__main__", "__file__": BENCHMARK})


if os.environ.get("_OGA_SMOKE_CHILD") == "1":
    _run_child(sys.argv[1:])
    sys.exit(0)


# ---- Parse OUR flags out of argv; the rest is forwarded to the benchmark -----
def _pop_flag(argv, name):
    if name in argv:
        argv.remove(name)
        return True
    return False


def _pop_value(argv, name, default=None):
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            val = argv[i + 1]
            del argv[i : i + 2]
            return val
        del argv[i : i + 1]
    return default


user_args = sys.argv[1:]
opt_perf = _pop_value(user_args, "--perf", None)
opt_log = _pop_value(user_args, "--log", None)
opt_report = _pop_flag(user_args, "--report")
opt_keep_config = _pop_flag(user_args, "--keep-config")


def _last_value(args, *flags):
    found = None
    for i, tok in enumerate(args):
        if tok in flags and i + 1 < len(args):
            found = args[i + 1]
    return found


# Effective model: CLI (-i) wins over the path block; used for the config swap.
eff_model = _last_value(user_args, "-i", "--input_folder", "-im_dir") or MODEL_DIR


# ---- Environment ------------------------------------------------------------
os.environ["THEROCK_DIST"] = THEROCK
os.environ["HIP_CUSTOM_KERNELS_DIR"] = PREBUILT_LIB
if opt_perf is not None:
    os.environ["HIPDNN_EP_PERF"] = opt_perf
else:
    os.environ.setdefault("HIPDNN_EP_PERF", "1")
os.environ.setdefault("PYTHONUTF8", "1")
perf_on = os.environ["HIPDNN_EP_PERF"] != "0"
print(
    f"[wrap] HIPDNN_EP_PERF={os.environ['HIPDNN_EP_PERF']}"
    + (
        "  (per-op profiling ON; tok/s are instrumentation-bound -- "
        "use --perf 0 for throughput)"
        if perf_on
        else "  (throughput mode)"
    ),
    flush=True,
)


# ---- Auto-select the MorphiZenEP genai_config variant -----------------------
def maybe_swap_config(model_dir, keep):
    cfg = os.path.join(model_dir, "genai_config.json")
    variant = os.path.join(model_dir, "genai_config_MorphiZenEP.json")
    if not os.path.isfile(cfg):
        return
    try:
        txt = open(cfg, encoding="utf-8", errors="replace").read()
    except Exception:
        return
    if "MorphiZenEP" in txt:
        return  # already MorphiZenEP
    if not os.path.isfile(variant):
        print(
            "[wrap] WARN: genai_config.json has no MorphiZenEP provider and "
            "no genai_config_MorphiZenEP.json variant found; running as-is.",
            flush=True,
        )
        return
    backup = cfg + ".orig"
    if not os.path.exists(backup):
        shutil.copyfile(cfg, backup)
    shutil.copyfile(variant, cfg)
    print(
        f"[wrap] swapped genai_config.json -> MorphiZenEP variant "
        f"(original backed up to {os.path.basename(backup)})",
        flush=True,
    )
    if not keep:

        def _restore():
            try:
                if os.path.exists(backup):
                    shutil.copyfile(backup, cfg)
                    print("[wrap] restored original genai_config.json", flush=True)
            except Exception as e:
                print(
                    f"[wrap] WARN: failed to restore genai_config.json: {e}", flush=True
                )

        atexit.register(_restore)


if eff_model:
    maybe_swap_config(eff_model, opt_keep_config)

# ---- Assemble the benchmark argv --------------------------------------------
DEFAULT_ARGV = [
    "-g",
    "64",
    "-m",
    "2048",
    "-r",
    "10",
    "-w",
    "1",
    "-k",
    "1",
    "-p",
    "1.0",
    "-v",
    "-mo",
]
argv = list(DEFAULT_ARGV) + list(user_args)
# Inject model/image from CONFIG only if the user didn't pass them.
if not _last_value(user_args, "-i", "--input_folder") and MODEL_DIR:
    argv = ["-i", MODEL_DIR] + argv
if not _last_value(user_args, "--image_path", "-im") and IMAGE_PATH:
    argv = ["--image_path", IMAGE_PATH] + argv

# CSV output (derived from image stem; into OUT_DIR) unless user gave -o.
csv_out = _last_value(argv, "-o", "--output")
if not csv_out:
    img = _last_value(argv, "--image_path", "-im") or "run"
    stem = os.path.splitext(os.path.basename(img))[0]
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_out = os.path.join(OUT_DIR, f"{stem}_mm.csv")
    argv += ["-o", csv_out]

# Log path (UTF-8, written by us): default into OUT_DIR.
if opt_log:
    log_path = opt_log
else:
    img = _last_value(argv, "--image_path", "-im") or "run"
    stem = os.path.splitext(os.path.basename(img))[0]
    os.makedirs(OUT_DIR, exist_ok=True)
    log_path = os.path.join(OUT_DIR, f"{stem}_smoke.log")

reps = int(_last_value(argv, "-r", "--repetitions") or "10")
print(f"[wrap] CSV  -> {csv_out}", flush=True)
print(f"[wrap] log  -> {log_path}  (UTF-8)", flush=True)
print(f"[wrap] argv -> {argv}", flush=True)


# ---- Spawn the child (this script in --child mode) and tee to a UTF-8 log ----
# Capturing a *child* process gives us its combined stdout+stderr (incl. the
# native EP fd-1/2 output) and, crucially, the child's exit-time C-runtime flush
# lands in the pipe we are still draining -- so the UTF-8 log is complete.
def run_and_tee(bench_argv, log_file_path):
    child_env = os.environ.copy()
    child_env["_OGA_SMOKE_CHILD"] = "1"
    child_env["THEROCK_DIST"] = THEROCK
    child_env["HIP_CUSTOM_KERNELS_DIR"] = PREBUILT_LIB
    child_env["HIPDNN_EP_PERF"] = os.environ["HIPDNN_EP_PERF"]
    child_env["PYTHONUTF8"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__)] + list(bench_argv),
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    out = sys.stdout.buffer
    with open(log_file_path, "wb") as logf:
        while True:
            chunk = proc.stdout.readline()
            if not chunk:
                if proc.poll() is not None:
                    break
                continue
            out.write(chunk)
            out.flush()
            logf.write(chunk)
            logf.flush()
    return proc.wait()


rc = run_and_tee(argv, log_path)

# atexit restore (config) runs at process exit.

# ---- Optional: render the perf report on the captured UTF-8 log + CSV --------
if opt_report:
    if not os.path.isfile(REPORT_SCRIPT):
        print(f"[wrap] --report: {REPORT_SCRIPT} not found; skipping.", flush=True)
    else:
        print("\n[wrap] ===== perf_multimodal_report =====", flush=True)
        cmd = [sys.executable, REPORT_SCRIPT, log_path]
        if os.path.isfile(csv_out):
            cmd.append(csv_out)
        cmd += ["--reps", str(reps)]
        subprocess.run(cmd)

sys.exit(rc)
