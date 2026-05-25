#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Smoke test for the dynamic-shape debug surface.

The debug surface gives users three env vars to introspect dynamic-output
resolution without having to instrument any code:

  * ``HIPDNN_EP_DEBUG_SHAPES=1`` -- per-output DimSpec dump at session
    open + per-Compute() pre/post-resolve traces, plus the resolver's
    per-leaf evaluation steps.

  * ``HIPDNN_EP_TRACE_SLOTS=1`` -- per-slot publish/read trace inside
    the generated model.dll.

  * ``HIPDNN_EP_VALIDATE_SHAPES=1`` -- gate exists, **intentionally not
    wired**. Designed-but-deferred CPU-reference shape-only validator;
    setting it today is a no-op. This test scrubs it from the env on
    every subprocess invocation so that, when the validator does get
    wired in the future, its log output won't accidentally pollute the
    DEBUG_SHAPES / TRACE_SLOTS marker checks below.
    See docs/design/dynamic-shape-debug-surface.md for the full
    rationale and the trigger conditions for reviving it.

This test (a) verifies the env-var gates compile in and behave as
zero-overhead when off (we don't try to measure that here; it's
guaranteed by the ``static const bool`` cache pattern in
``debug_log.h``), and (b) verifies that turning them on produces the
expected marker strings on stderr when a Category-C composition runs
end-to-end.

The test only runs when the EP DLL is built and reachable -- it
re-invokes pytest as a subprocess on a single, already-passing
composition test (``test_shape_of_nonzero_1d_i64``) and greps stderr
for the marker strings. If the EP CLI options were not supplied to the
parent pytest invocation, this whole module SKIPs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


COMPOSITION_TEST = (
    "tests/test_nonzero_composition.py::TestNonZeroComposition::"
    "test_shape_of_nonzero_1d_i64"
)


def _have_ep_dll(request) -> bool:
    """Cheap reflection: whether ``--ep-dll`` was passed to this pytest run."""
    try:
        v = request.config.getoption("--ep-dll", default="")
    except (KeyError, ValueError):
        return False
    return bool(v) and Path(v).exists()


def _resolve_arg_value(val: str) -> str:
    """Return ``val`` unchanged if it doesn't look like an existing
    path, otherwise resolve it to an absolute path from the parent
    process's cwd.

    The subprocess we spawn runs with cwd=test/numeric so any
    parent-relative ``..\\..\\install\\...`` argument would point at
    the wrong place. Absolutifying any value that resolves to an
    existing file from the parent cwd is the simplest, conservative
    fix -- non-path arguments (backend name, name=value option
    strings without an ``=`` -resolvable file) pass through untouched.
    """
    if not val:
        return val
    p = Path(val)
    if p.exists():
        return str(p.resolve())
    return val


def _rerun_with_env(request, extra_env: dict[str, str]) -> tuple[int, str]:
    """Spawn a pytest subprocess on the composition test with extra env.

    Returns ``(returncode, combined_stdout_stderr)``. The subprocess
    forwards every CLI option the parent received (so ``--ep-dll`` /
    ``--ep-option`` flow through automatically), plus -s so the EP
    diagnostic stderr is not swallowed by pytest's capture, plus
    -p no:cacheprovider to avoid pytest cache pollution.
    """
    cfg = request.config
    # Run the subprocess from test/numeric (where pytest.ini lives) so
    # rootdir auto-detection finds it. Any relative-path CLI args
    # received from the parent are absolutified against the parent's
    # cwd to survive the cwd switch.
    subproc_cwd = Path(__file__).parent.parent
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        COMPOSITION_TEST,
        "-s",
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    # Forward EP CLI args. They were registered on the parent's
    # pytest_addoption; we re-emit them so the subprocess has the
    # same backend wiring.
    forward_keys = (
        "--backend",
        "--ep-name",
        "--ep-dll",
    )
    for key in forward_keys:
        try:
            val = cfg.getoption(key, default="")
        except (KeyError, ValueError):
            val = ""
        if val:
            cmd += [key, _resolve_arg_value(str(val))]
    # --ep-option is a multi-value list option; emit one --ep-option
    # per entry. getoption() returns the list directly. Each value is
    # a ``key=value`` string -- if the value half is a path, absolutify
    # it; otherwise pass through verbatim.
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

    env = os.environ.copy()
    # Don't carry over any debug-surface env var from the parent so
    # toggling one knob at a time is genuinely isolated.
    for k in (
        "HIPDNN_EP_DEBUG_SHAPES",
        "HIPDNN_EP_TRACE_SHAPES",
        "HIPDNN_EP_TRACE_SLOTS",
        "HIPDNN_EP_VALIDATE_SHAPES",
    ):
        env.pop(k, None)
    env.update(extra_env)

    result = subprocess.run(
        cmd,
        cwd=str(subproc_cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return result.returncode, (result.stdout or "") + (result.stderr or "")


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="EP DLL is Windows/Linux only; macOS has no MorphiZen build.",
)
class TestDebugSurface:
    """End-to-end stderr-marker checks for the three debug env vars.

    Each test SKIPs (not fails) when the parent pytest run wasn't
    given the EP backend, so the suite stays usable on a tree where
    only the framework was built.
    """

    def test_debug_shapes_emits_per_output_trace(self, request):
        if not _have_ep_dll(request):
            pytest.skip("requires --ep-dll (EP backend)")
        rc, out = _rerun_with_env(
            request, {"HIPDNN_EP_DEBUG_SHAPES": "1"}
        )
        assert rc == 0, (
            f"composition test failed under HIPDNN_EP_DEBUG_SHAPES=1:\n{out}"
        )
        # The composition graph is `Shape(NonZero(X))`. Whether the
        # ONE output shape ends up Category-A (statically `[2]`) or
        # Category-C depends on the conversion path -- both are valid
        # outcomes and both go through MlirCustomOp::ctor. The session-
        # open `[CTor]` dump fires unconditionally for ANY output that
        # has dim_specs, so it's the most stable marker for "EP
        # observed the env var and printed shape diagnostics". A
        # negative-control test below verifies it does NOT fire with
        # the env var unset.
        assert "[CTor] Output[" in out, (
            "expected [CTor] Output[...] DimSpec dump under "
            "HIPDNN_EP_DEBUG_SHAPES=1:\n" + out
        )

    def test_trace_slots_emits_publish_and_read(self, request):
        if not _have_ep_dll(request):
            pytest.skip("requires --ep-dll (EP backend)")
        rc, out = _rerun_with_env(
            request, {"HIPDNN_EP_TRACE_SLOTS": "1"}
        )
        assert rc == 0, (
            f"composition test failed under HIPDNN_EP_TRACE_SLOTS=1:\n{out}"
        )
        # The marker emitted by hipdnn_ep_state_publish_dim and
        # _read_dim when called from inside the generated model.dll.
        # The Shape(NonZero(X)) graph publishes one dim slot from
        # wrap_nonzero and reads it from wrap_shape.
        assert "[Slots] publish_dim(" in out, (
            "expected publish_dim slot trace in stderr:\n" + out
        )
        assert "[Slots] read_dim(" in out, (
            "expected read_dim slot trace in stderr:\n" + out
        )

    def test_gates_default_off_no_markers(self, request):
        """Negative check: with no env vars set, none of the markers leak.

        Guarantees the debug surface is genuinely off by default and a
        production run never sees [Slots] / pre-compute resolved /
        post-compute resolved on stderr.
        """
        if not _have_ep_dll(request):
            pytest.skip("requires --ep-dll (EP backend)")
        rc, out = _rerun_with_env(request, {})
        assert rc == 0, f"composition test failed (no debug env):\n{out}"
        for marker in (
            "[Slots] publish_dim(",
            "[Slots] read_dim(",
            "[CTor] Output[",
            "pre-compute resolved shape",
            "post-compute resolved shape",
        ):
            assert marker not in out, (
                f"marker {marker!r} leaked with no debug env var set:\n"
                + out
            )
