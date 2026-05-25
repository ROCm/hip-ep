#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Backward-compat smoke for the dynamic-output-shape ABI in model.dll.

The dynamic-output-shape (DimSpec / RuntimeSlot) framework added five
new symbols to the model.dll ABI:

  * ``inference_get_metadata_json``     -- always emitted by the new
    compiler; legacy DLLs lack it (EP falls back to a static-shape-only
    path with ``metadata_ = nullptr``, ``dyn_dim_slots_count() = 0``).

  * ``inference_dyn_slot_get_dim``      -- ┐
  * ``inference_dyn_slot_get_buffer``   -- ├─ emitted by
  * ``inference_dyn_slot_reset``        -- ┘  ``GenerateInterface`` ONLY
    when ``dyn_dim_slots_count > 0``. All-static-output models skip all
    three; slot-using models MUST export all three.

  * ``hipdnn_ep_runtime_begin_compute`` -- emitted unconditionally by
    the new runtime to invalidate per-Compute() caches (currently the
    GQA seqlens_k cache). Legacy DLLs lack it; the EP detects that and
    LOG(WARNING)s only when ``HIPDNN_EP_GQA_CACHE_SEQLENS`` is not
    explicitly disabled.

The backward-compat contract is enforced at session creation time by
``InferenceState::InferenceState()`` (``backend-mlir-compiler/custom-op-
mlir/src/InferenceState.cpp``):

  1. **All-static-output legacy DLL on a new EP** -- works. The three
     ``inference_dyn_slot_*`` lookups return null, the metadata-JSON
     lookup returns null, ``dyn_dim_slots_count()`` returns 0, and
     none of the fatal checks fire.

  2. **Slot-using DLL freshly built** -- works. All five new exports
     are present; the metadata declares ``dyn_dim_slots_count > 0``;
     the fatal-check pre-condition (``dyn_slots > 0 && any_export
     missing``) is false because no export is missing.

  3. **Slot-using metadata + missing dyn_slot exports** -- the
     impossible-by-construction case that the LOG(FATAL) at
     ``InferenceState.cpp:85-96`` guards against. This combination can
     only arise if the compiler is rebuilt against a future ABI that
     introduces a new dyn_slot accessor while the runtime still emits
     stale exports, OR if a user manually patches a DLL. We don't
     reproduce it in-process (would require corrupting a built DLL),
     but we verify (a) the static-source FATAL message text remains
     actionable -- ``del %TEMP%\\morphizen_mlir_*`` -- and (b) the
     loaded DLL's actual exports match what the EP probes for, so a
     future ABI drift (renamed/removed export) is caught by the
     symbol-presence test below.

This test runs the EP under pytest-in-pytest (same pattern as
``test_debug_surface.py``) on (a) a static-only model and (b) a
slot-using model, captures the model.dll each compiles to
``%TEMP%/morphizen_mlir_*.dll`` BEFORE the EP destructor deletes it,
and verifies the exports of each via ``ctypes.WinDLL`` + ``GetProcAddress``
(the exact same Win32 mechanism ``morphizen::Plugin::get_method`` uses
under the hood).

Out of scope: actually triggering the LOG(FATAL). It calls
``abort()``; reproducing it requires patching a PE export table to
remove e.g. ``inference_dyn_slot_get_dim`` from an otherwise-valid
slot-using DLL, then loading it. The static-text-source check in
``test_fatal_message_text_documents_remediation`` is the supported
substitute -- it catches the only way the contract can regress
(message wording drift or export rename) without paying the cost of
PE patching.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Iterable

import pytest


# ---------------------------------------------------------------------------
# Symbol contracts
# ---------------------------------------------------------------------------

# Always present in any post-2026 DLL.
_ALWAYS_EXPORTS = (
    "inference_init",
    "inference_compute",
    "inference_cleanup",
)

# Emitted by the new compiler/runtime regardless of slot count.
_NEW_RUNTIME_EXPORTS = (
    "inference_get_metadata_json",
    "hipdnn_ep_runtime_begin_compute",
)

# Emitted ONLY when dyn_dim_slots_count > 0. Verified via
# ``ctypes.GetProcAddress`` against captured DLLs from each scenario.
_SLOT_ONLY_EXPORTS = (
    "inference_dyn_slot_get_dim",
    "inference_dyn_slot_get_buffer",
    "inference_dyn_slot_reset",
)


# ---------------------------------------------------------------------------
# pytest-in-pytest helper -- adapted from test_debug_surface.py
# ---------------------------------------------------------------------------


def _have_ep_dll(request) -> bool:
    try:
        v = request.config.getoption("--ep-dll", default="")
    except (KeyError, ValueError):
        return False
    return bool(v) and Path(v).exists()


def _resolve_arg_value(val: str) -> str:
    """Absolutify arg values that look like existing paths (subprocess cwd-safe)."""
    if not val:
        return val
    p = Path(val)
    if p.exists():
        return str(p.resolve())
    return val


def _build_subproc_cmd(request, target_test: str) -> list[str]:
    """Construct an ``python -m pytest target_test ...`` command line that
    forwards all EP backend flags from the parent invocation.
    """
    cfg = request.config
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        target_test,
        "-s",
        "-q",
        "-p",
        "no:cacheprovider",
        "--no-cache",   # framework's reference cache; orthogonal to model.dll cache
    ]
    for key in ("--backend", "--ep-name", "--ep-dll"):
        try:
            val = cfg.getoption(key, default="")
        except (KeyError, ValueError):
            val = ""
        if val:
            cmd += [key, _resolve_arg_value(str(val))]
    try:
        ep_opts = cfg.getoption("--ep-option", default=[]) or []
    except (KeyError, ValueError):
        ep_opts = []
    for kv in ep_opts:
        s = str(kv)
        if "=" in s:
            k, _, v = s.partition("=")
            cmd += ["--ep-option", f"{k}={_resolve_arg_value(v)}"]
        else:
            cmd += ["--ep-option", s]
    return cmd


# ---------------------------------------------------------------------------
# DLL-capture helper
# ---------------------------------------------------------------------------


class _TempDllSnapshot:
    """Run a subprocess pytest invocation while polling ``%TEMP%`` for
    ``morphizen_mlir_*.dll`` files; copy each new one to ``dest_dir`` BEFORE
    the EP destructor unlinks the original.

    Returns the list of captured DLL paths after the subprocess exits.
    Designed for the morphizen temp-DLL lifecycle (`InferenceState::create`
    writes the DLL synchronously, the destructor unlinks it many seconds
    later when the InferenceSession is destroyed); a 50 ms polling loop
    is sufficient to catch every DLL without missing any.

    Implementation note: we cannot use a true file-system watcher because
    Windows ``inotify`` equivalents (ReadDirectoryChangesW) don't see
    ``CreateFile`` events on ``%TEMP%`` reliably from Python without
    binding to ``win32api``; a 50 ms poll is portable, robust, and the
    extra latency is invisible against the multi-second EP compile time.
    """

    def __init__(self, dest_dir: Path, env: dict[str, str] | None = None):
        self.dest_dir = Path(dest_dir)
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        self.env = env or {}
        self.captured: list[Path] = []

    def run(self, cmd: list[str], cwd: str, timeout: int = 120) -> tuple[int, str]:
        temp_root = Path(tempfile.gettempdir())
        # Snapshot existing DLLs (so we don't claim a leftover from a
        # previous run as ours). The morphizen tempname includes the
        # PID, so collisions are impossible -- but a stale DLL from a
        # previously-crashed session would be missed without this
        # baseline.
        baseline = {p.name for p in temp_root.glob("morphizen_mlir_*.dll")}

        env = os.environ.copy()
        env.update(self.env)

        # CRITICAL: subprocess stdout MUST be drained continuously --
        # the Windows pipe buffer is ~64 KB, and the EP emits a large
        # multi-MB diagnostic stream per inference. If we only call
        # `proc.communicate()` after the polling loop, the subprocess
        # blocks on the first stdout write that fills the pipe and the
        # entire test serialises behind that block (we observed 119s
        # for a 4s sigmoid test). Use a daemon thread to drain the
        # pipe into an in-memory buffer while the polling loop watches
        # the file system. The thread joins naturally when the
        # subprocess exits and closes its stdout.
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered on the read side
        )

        out_chunks: list[str] = []

        def _drain() -> None:
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    out_chunks.append(line)
            except Exception:
                pass

        drain_thread = threading.Thread(target=_drain, daemon=True)
        drain_thread.start()

        deadline = time.time() + timeout
        seen: set[str] = set()
        try:
            while proc.poll() is None and time.time() < deadline:
                self._sweep(temp_root, baseline, seen)
                time.sleep(0.05)
            # One last sweep right after exit, in case the destructor
            # hadn't run by the time we polled (small race window).
            self._sweep(temp_root, baseline, seen)
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            drain_thread.join(timeout=5)
            return proc.returncode, "".join(out_chunks)
        except Exception:
            proc.kill()
            drain_thread.join(timeout=5)
            raise

    def _sweep(self, temp_root: Path, baseline: set[str], seen: set[str]) -> None:
        try:
            for p in temp_root.glob("morphizen_mlir_*.dll"):
                if p.name in baseline or p.name in seen:
                    continue
                seen.add(p.name)
                # Copy before the destructor can unlink. We use copy2 to
                # preserve mtime. Errors are silent -- a missed copy
                # just means one fewer DLL in self.captured; the test
                # asserts at least one was captured.
                try:
                    target = self.dest_dir / p.name
                    shutil.copy2(p, target)
                    self.captured.append(target)
                except (OSError, PermissionError):
                    pass
        except OSError:
            pass


# ---------------------------------------------------------------------------
# ctypes-based PE-export probe (mirrors morphizen::Plugin::get_method)
# ---------------------------------------------------------------------------


def _probe_exports(dll_path: Path, names: Iterable[str]) -> dict[str, bool]:
    """Return ``{name: True/False}`` for each ``name`` based on whether
    ``GetProcAddress`` resolves it on ``dll_path``.

    Uses the same Win32 mechanism the EP itself uses (via
    ``morphizen::Plugin::get_method`` -> ``LoadLibrary`` +
    ``GetProcAddress``). Loads the DLL with the default search order;
    requires ``%TEMP%`` and the TheRock bin dir to be on PATH for the
    DLL's own imports to resolve, which is true in the test's parent
    pytest invocation environment.
    """
    if sys.platform != "win32":
        pytest.skip("ctypes GetProcAddress probe is Windows-only")

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    LoadLibraryW = kernel32.LoadLibraryW
    LoadLibraryW.argtypes = [ctypes.c_wchar_p]
    LoadLibraryW.restype = ctypes.c_void_p
    GetProcAddress = kernel32.GetProcAddress
    GetProcAddress.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    GetProcAddress.restype = ctypes.c_void_p
    FreeLibrary = kernel32.FreeLibrary
    FreeLibrary.argtypes = [ctypes.c_void_p]
    FreeLibrary.restype = ctypes.c_int

    handle = LoadLibraryW(str(dll_path))
    if not handle:
        err = ctypes.get_last_error()
        raise RuntimeError(
            f"LoadLibraryW failed for {dll_path}: GetLastError={err}. "
            f"Ensure {Path('install/therock/bin').resolve()} is on PATH "
            f"so the DLL's own imports (amdhip64_7.dll etc) can resolve."
        )
    try:
        return {n: bool(GetProcAddress(handle, n.encode("ascii"))) for n in names}
    finally:
        FreeLibrary(handle)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


# Existing in-tree tests we reuse as scenario drivers. Picked because:
#   * test_sigmoid_small: smallest possible all-static EP test (Sigmoid on
#     [1, 10] fp16 -- no dynamic shapes, no Category-C ops, no slots).
#   * test_shape_of_nonzero_1d_i64: smallest possible slot-using EP test
#     (Shape(NonZero(X)) -- NonZero publishes RuntimeSlot 0, Shape reads
#     it). Already used as the canonical Category-C composition target by
#     test_debug_surface.py.
_STATIC_SCENARIO_TEST = (
    "tests/test_sigmoid.py::TestSigmoid::test_sigmoid_small"
)
_SLOT_SCENARIO_TEST = (
    "tests/test_nonzero_composition.py::TestNonZeroComposition::"
    "test_shape_of_nonzero_1d_i64"
)


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="The backcompat smoke depends on Win32 GetProcAddress and the "
    "Windows TEMP layout; the Linux EP shipping path is separate.",
)
class TestBackwardCompat:
    """ABI invariants between the EP and the compiled model.dll.

    All scenarios SKIP (not fail) when the EP backend wasn't supplied to
    the parent pytest run, so the suite stays usable on a tree where
    only the framework was built.
    """

    def test_static_only_model_does_not_export_slot_abi(
        self, request, tmp_path
    ):
        """All-static model -> DLL has the always-present exports + the
        new metadata-JSON + begin_compute exports, but does NOT export
        the three ``inference_dyn_slot_*`` symbols.

        Establishes the "legacy-compatible" half of the contract: an
        all-static graph compiles to a DLL whose surface is a strict
        subset of the slot-using surface, so a legacy EP (or one
        compiled without the dyn_slot probes) cannot accidentally try
        to call a missing symbol.
        """
        if not _have_ep_dll(request):
            pytest.skip("requires --ep-dll (EP backend)")

        cmd = _build_subproc_cmd(request, _STATIC_SCENARIO_TEST)
        subproc_cwd = Path(__file__).parent.parent
        snap = _TempDllSnapshot(dest_dir=tmp_path / "captured")
        rc, out = snap.run(cmd, cwd=str(subproc_cwd))
        assert rc == 0, f"static scenario test failed:\n{out}"
        assert snap.captured, (
            "no morphizen_mlir_*.dll captured during static scenario -- "
            "either the EP did not compile a DLL or the polling missed it.\n"
            f"Subprocess output:\n{out}"
        )

        for dll in snap.captured:
            present = _probe_exports(
                dll,
                _ALWAYS_EXPORTS + _NEW_RUNTIME_EXPORTS + _SLOT_ONLY_EXPORTS,
            )
            missing_always = [n for n in _ALWAYS_EXPORTS if not present[n]]
            assert not missing_always, (
                f"{dll.name}: always-required exports missing: "
                f"{missing_always}. ABI broken?"
            )
            missing_runtime = [n for n in _NEW_RUNTIME_EXPORTS if not present[n]]
            assert not missing_runtime, (
                f"{dll.name}: new runtime exports missing: {missing_runtime}. "
                "Build must export these from EVERY DLL (independent of "
                "slot count) -- the EP relies on them to detect legacy DLLs."
            )
            slot_present = [n for n in _SLOT_ONLY_EXPORTS if present[n]]
            assert not slot_present, (
                f"{dll.name}: slot-only exports present on a STATIC model: "
                f"{slot_present}. GenerateInterface should only emit these "
                "when dyn_dim_slots_count > 0; bloating every DLL with them "
                "wastes space and weakens the FATAL contract."
            )

    def test_slot_model_carries_full_new_abi(self, request, tmp_path):
        """Slot-using model -> DLL exports the always-present + new
        runtime + all three ``inference_dyn_slot_*`` symbols.

        Establishes the "fresh build is self-consistent" half of the
        contract: if a model needs RuntimeSlot dim resolution, every
        export the EP probes for is actually present. Regression here
        means the FATAL at ``InferenceState.cpp:85-96`` would fire on
        an EP+compiler combo from the SAME build, which is a bug.
        """
        if not _have_ep_dll(request):
            pytest.skip("requires --ep-dll (EP backend)")

        cmd = _build_subproc_cmd(request, _SLOT_SCENARIO_TEST)
        subproc_cwd = Path(__file__).parent.parent
        snap = _TempDllSnapshot(dest_dir=tmp_path / "captured")
        rc, out = snap.run(cmd, cwd=str(subproc_cwd))
        assert rc == 0, f"slot scenario test failed:\n{out}"
        assert snap.captured, (
            "no morphizen_mlir_*.dll captured during slot scenario.\n"
            f"Subprocess output:\n{out}"
        )

        all_required = _ALWAYS_EXPORTS + _NEW_RUNTIME_EXPORTS + _SLOT_ONLY_EXPORTS
        for dll in snap.captured:
            present = _probe_exports(dll, all_required)
            missing = [n for n in all_required if not present[n]]
            assert not missing, (
                f"{dll.name}: a slot-using DLL is missing one or more "
                f"required exports: {missing}. The EP will LOG(FATAL) at "
                f"session load with the actionable remediation message."
            )

    def test_fatal_message_text_documents_remediation(self):
        """The LOG(FATAL) text users see when the dyn_slot ABI mismatch
        triggers MUST stay actionable: it must name the exports that
        are missing AND tell the user how to clear the stale DLL.

        We assert this against the live source rather than trying to
        actually trigger the abort, because:
          * LOG(FATAL) calls abort() -> the only way to test it is in
            a child process that's expected to exit nonzero, which is
            doable but requires corrupting a DLL's export table to
            simulate "metadata says slots > 0 but exports are absent."
          * The message drift is the only failure mode of the FATAL
            itself; the EP-side dispatch logic is exercised by the
            two scenarios above.

        If the message text moves to a different file or changes
        wording, update both this test AND any user-facing docs
        (CLAUDE.md, docs/design/compiler-runtime-contract.md) in the
        same change.
        """
        src = (
            Path(__file__).parents[3]
            / "backend-mlir-compiler"
            / "custom-op-mlir"
            / "src"
            / "InferenceState.cpp"
        )
        text = src.read_text(encoding="utf-8", errors="replace")
        # The FATAL block is a single multi-line string-literal; check
        # for the actionable phrases that MUST stay together.
        anchors = (
            "dynamic dim slot",
            "inference_dyn_slot_",
            "del %TEMP%\\\\morphizen_mlir_*",
            "Rebuild",
        )
        for anchor in anchors:
            assert anchor in text, (
                f"Backward-compat FATAL message in {src} no longer "
                f"contains the actionable anchor {anchor!r}. The user "
                f"sees this string when their model.dll is stale; "
                f"the wording MUST stay actionable. Update this test "
                f"AND user-facing docs (CLAUDE.md, docs/design/...) "
                f"in the same change."
            )

    def test_begin_compute_warning_text_stays_actionable(self):
        """``InferenceState::create`` LOG(WARNING)s when the model.dll
        predates ``hipdnn_ep_runtime_begin_compute`` AND the GQA
        seqlens_k cache is enabled. The wording is the user's only
        signal before decode silently corrupts past token 1; protect
        it the same way as the FATAL above.
        """
        src = (
            Path(__file__).parents[3]
            / "backend-mlir-compiler"
            / "custom-op-mlir"
            / "src"
            / "InferenceState.cpp"
        )
        text = src.read_text(encoding="utf-8", errors="replace")
        anchors = (
            "GQA seqlens_k cache is enabled",
            "HIPDNN_EP_GQA_CACHE_SEQLENS",
            "hipdnn_ep_runtime_begin_compute",
            "incorrect from token 2 onward",
        )
        for anchor in anchors:
            assert anchor in text, (
                f"begin_compute legacy-DLL WARNING in {src} no longer "
                f"contains the actionable anchor {anchor!r}."
            )
