#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Process control: the single-run lock and scoped process teardown.

Every timed run kills competing benchmark processes first, so two concurrent
invocations abort each other mid-measurement and append to the same CSV. The
lock refuses a second run instead. Python drivers are killed only if this
harness started them, matched by pid and creation time so a recycled pid is
never touched.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import time
from contextlib import contextmanager
from ctypes import wintypes
from pathlib import Path

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259

_k32 = ctypes.WinDLL("kernel32", use_last_error=True) if os.name == "nt" else None


def creation_time(pid: int) -> int | None:
    """Process creation time as a FILETIME integer, or None if not running."""
    if _k32 is None:
        return None
    h = _k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return None
    try:
        code = wintypes.DWORD()
        if (
            not _k32.GetExitCodeProcess(h, ctypes.byref(code))
            or code.value != _STILL_ACTIVE
        ):
            return None
        c, e, k, u = (wintypes.FILETIME() for _ in range(4))
        if not _k32.GetProcessTimes(h, *(ctypes.byref(x) for x in (c, e, k, u))):
            return None
        return (c.dwHighDateTime << 32) | c.dwLowDateTime
    finally:
        _k32.CloseHandle(h)


def alive(pid: int) -> bool:
    return creation_time(pid) is not None


def kill_tree(pid: int) -> None:
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def kill_by_name(*images: str) -> None:
    for image in images:
        subprocess.run(
            ["taskkill", "/IM", image, "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def pids_by_name(image: str) -> list[int]:
    out = subprocess.run(
        ["tasklist", "/FI", f"IMAGENAME eq {image}", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    pids = []
    for line in out.splitlines():
        cells = [c.strip('"') for c in line.split('","')]
        if len(cells) > 1 and cells[0].lower() == image.lower() and cells[1].isdigit():
            pids.append(int(cells[1]))
    return pids


class Harness:
    """Lock, driver registry and teardown rooted at one output directory."""

    def __init__(self, out_root: Path):
        self.out_root = out_root
        self.lock_path = out_root / "harness.lock"
        self.drivers_path = out_root / "drivers.txt"

    @contextmanager
    def lock(self):
        owner = os.environ.get("HIPEP_LOCK_OWNER")
        if owner and owner.isdigit() and alive(int(owner)):
            # An outer harness command already holds it: one logical run.
            yield
            return
        self.out_root.mkdir(parents=True, exist_ok=True)
        if self.lock_path.exists():
            holder = self.lock_path.read_text().strip().split()[:1]
            if holder and holder[0].isdigit() and alive(int(holder[0])):
                raise SystemExit(
                    f"another perf-harness run is in progress (pid {holder[0]}). "
                    "Concurrent runs abort each other and share one CSV. Wait for "
                    f"it, or delete {self.lock_path} if you are certain it is stale."
                )
            print(f"    clearing stale lock from pid {holder[0] if holder else '?'}")
        self.lock_path.write_text(str(os.getpid()))
        os.environ["HIPEP_LOCK_OWNER"] = str(os.getpid())
        try:
            yield
        finally:
            self.lock_path.unlink(missing_ok=True)
            os.environ.pop("HIPEP_LOCK_OWNER", None)

    def register_driver(self, pid: int) -> None:
        ct = creation_time(pid)
        if ct is None:
            return
        self.out_root.mkdir(parents=True, exist_ok=True)
        with self.drivers_path.open("a") as f:
            f.write(f"{pid} {ct}\n")

    def stop_competitors(self) -> None:
        kill_by_name(
            "model_benchmark.exe",
            "RadeonDeveloperPanel.exe",
            "RadeonDeveloperPanelCLI.exe",
            "RadeonDeveloperService.exe",
            "RadeonDeveloperServiceCLI.exe",
        )
        killed: list[tuple[int, str]] = []
        if self.drivers_path.exists():
            for line in self.drivers_path.read_text().splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[0].isdigit():
                    pid, ct = int(parts[0]), parts[1]
                    if str(creation_time(pid)) == ct:
                        kill_tree(pid)
                        killed.append((pid, ct))
            self.drivers_path.unlink(missing_ok=True)
        time.sleep(2)
        survivors: list[int] = []
        for _ in range(5):
            survivors = pids_by_name("model_benchmark.exe") + [
                pid for pid, ct in killed if str(creation_time(pid)) == ct
            ]
            if not survivors:
                return
            time.sleep(2)
        # A process stuck in GPU driver teardown ignores TerminateProcess and
        # keeps its device memory, so every later run would be measured against
        # a GPU that is partly taken.
        raise SystemExit(
            f"benchmark process(es) {survivors} survived a forced kill (stuck in "
            "the GPU driver) and still hold GPU memory; reboot before measuring"
        )
