#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Serial launcher for the Windows long-context profiling campaign."""

from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "results/profile").mkdir(parents=True, exist_ok=True)
CORE = Path(sys.base_prefix) / "Lib/site-packages/_rocm_sdk_core"
DEV = Path(sys.base_prefix) / "Lib/site-packages/_rocm_sdk_devel"
install = ROOT / "install/turboquant"
env = dict(os.environ)
env.update(
    OMP_NUM_THREADS="2",
    OPENBLAS_NUM_THREADS="2",
    TQ_WORK_ROOT=str(ROOT),
    HIPDNN_EP_STRICT="1",
    HIPDNN_EP_DEBUG="0",
    TQ_HIP_RUNTIME=str(CORE / "bin/amdhip64_7.dll"),
    TQ_TEST_LIBRARY=str(install / "bin/custom_kernels_gfx1151.dll"),
    TQ_RING_LIBRARY=str(install / "bin/custom_kernels_gfx1151.dll"),
    THEROCK_DIST=str(DEV),
    AMDGPU_EP_PATH=str(install / "bin/hipgpu.dll"),
)
env["PATH"] = (
    os.pathsep.join(
        map(
            str,
            [
                install / "bin",
                ROOT / "venv/Scripts",
                DEV / "bin",
                CORE / "bin",
                DEV / "lib/llvm/bin",
            ],
        )
    )
    + os.pathsep
    + env["PATH"]
)
env["HIP_EP_PROFILE_ALLOCATOR_DLL"] = str(ROOT / "build/windows_memory_counter.dll")
p = argparse.ArgumentParser()
p.add_argument(
    "variant",
    choices=[
        "ring",
        "tq4",
        "existing",
        "ring_q",
        "kernel-long",
        "synthetic-ring",
        "synthetic-tq4",
        "synthetic-existing",
        "synthetic-ring_q",
    ],
)
p.add_argument("length", type=int, nargs="?", default=4096)
p.add_argument("--repetitions", type=int, default=3)
a = p.parse_args()
selection = a.variant.removeprefix("synthetic-")
if selection in ("existing", "ring_q"):
    install = (
        ROOT
        / "install"
        / ("existing" if selection == "existing" else "ring_q_resident")
    )
    assert (install / "bin/hipgpu.dll").is_file(), install
    env["PATH"] = str(install / "bin") + os.pathsep + env["PATH"]
    env["AMDGPU_EP_PATH"] = str(install / "bin/hipgpu.dll")
env.update(TQ_PROFILE_VARIANT=selection, TQ_PROFILE_INSTALL=str(install))
if not (ROOT / "results/profile/wddm-validation.json").exists():
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_windows_memory.py")],
        env=env,
        check=True,
    )
if not (ROOT / "results/profile/wddm-validation-v2.json").exists():
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_windows_memory_v2.py")],
        env=env,
        check=True,
    )
if a.variant == "tq4" and a.length > 4096:
    first = ROOT / "results/profile/tq4/4096/receipt.json"
    if not first.exists() or json.loads(first.read_text()).get("status") != "passed":
        # Complete the smallest full-model case before increasing TQ context.
        subprocess.run([sys.executable, __file__, "tq4", "4096"], check=True)
if a.variant == "kernel-long":
    command = [
        sys.executable,
        "-u",
        "-c",
        'import torch,runpy,sys;sys.path.insert(0,sys.argv[1]);runpy.run_path(sys.argv[2],run_name="__main__")',
        str(ROOT / "hip-ep/tools/turboquant"),
        str(ROOT / "scripts/check_long_windows.py"),
    ]
elif a.variant.startswith("synthetic-"):
    command = [sys.executable, "-u", str(ROOT / "scripts/windows_synthetic.py")] + (
        ["--turboquant"] if a.variant.endswith("tq4") else []
    )
else:
    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/windows_profile.py"),
        "--length",
        str(a.length),
        "--repetitions",
        str(a.repetitions),
    ] + (["--turboquant"] if a.variant == "tq4" else [])
name = f"{a.variant}-{a.length}"
dest = ROOT / "results/profile"
dest.mkdir(exist_ok=True)
status = {"status": "running", "command": command, "start": time.time()}
path = dest / (name + "-status.json")
path.write_text(json.dumps(status, indent=2))
with (dest / (name + ".log")).open("w") as log:
    proc = subprocess.Popen(
        command, env=env, cwd=ROOT / "hip-ep", stdout=log, stderr=subprocess.STDOUT
    )
    status["child_pid"] = proc.pid
    path.write_text(json.dumps(status, indent=2))
    low_memory_since = None
    while proc.poll() is None:
        now = time.time()
        available = psutil.virtual_memory().available
        if available < 2**30:
            if low_memory_since is None:
                low_memory_since = now
        else:
            low_memory_since = None
        reason = None
        if now - status["start"] > 3600:
            reason = "One-hour per-case watchdog"
        if low_memory_since is not None and now - low_memory_since > 20:
            reason = "System available memory stayed below 1 GiB for 20 seconds"
        if reason:
            status["watchdog_reason"] = reason
            # Stop only this launcher's owned benchmark tree, including the
            # base Python child created by the Windows venv redirector.
            try:
                parent = psutil.Process(proc.pid)
                children = parent.children(recursive=True)
                for child in reversed(children):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                parent.kill()
                psutil.wait_procs(children + [parent], timeout=10)
            except psutil.NoSuchProcess:
                pass
            proc.wait()
            break
        time.sleep(1)
    rc = 124 if status.get("watchdog_reason") else proc.returncode
if rc and a.variant != "kernel-long":
    folder = a.variant.removeprefix("synthetic-")
    receipt = (
        dest
        / folder
        / (
            str(a.length) + "-synthetic"
            if a.variant.startswith("synthetic-")
            else str(a.length)
        )
        / "receipt.json"
    )
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if record.get("status") not in ("passed", "failed"):
            record.update(
                status="failed",
                error=status.get(
                    "watchdog_reason", f"Native process exited with code {rc}"
                ),
            )
            receipt.write_text(json.dumps(record, indent=2) + "\n")
status.update(
    status="passed" if rc == 0 else "failed",
    returncode=rc,
    elapsed=time.time() - status["start"],
)
path.write_text(json.dumps(status, indent=2))
print(json.dumps(status), flush=True)
print((dest / (name + ".log")).read_text(errors="replace")[-6500:])
raise SystemExit(rc)
