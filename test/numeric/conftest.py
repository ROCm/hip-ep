#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Pytest configuration and shared fixtures for the numeric test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from framework.model_runner import ModelRunner
from framework.reference_cache import ReferenceCache

# ---------------------------------------------------------------------------
# Backend registry (only user-selectable backends live here)
#
# Each factory receives the pytest config so it can pull its own
# backend-specific CLI options (e.g. --ep-dll). Keeping per-backend
# argument plumbing inside the backend module rather than spreading it
# across conftest avoids "knob explosion" in this file as more backends
# arrive.
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, callable] = {
    "ort_ep": lambda config: __import__(
        "framework.ort_ep_backend", fromlist=["create"]
    ).create(config),
}


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        default="ort_ep",
        choices=list(_BACKENDS),
        help="Inference backend to test (default: ort_ep).",
    )
    parser.addoption(
        "--output-dir",
        default="",
        help="Root for all test output. Defaults to python_test_output/ "
        "next to this conftest. The framework writes into two "
        "subdirectories of this root: 'intermediate/' (per-test "
        "ephemeral artefacts -- model, inputs, EP outputs, "
        "reference outputs; deleted on pass) and 'cache/' "
        "(persistent reference-output cache, keyed by sanitised "
        "pytest node id). Use --work-dir / --cache-dir to override "
        "either subdirectory explicitly.",
    )
    parser.addoption(
        "--work-dir",
        default="",
        help="Per-test artifact directory. Defaults to "
        "<output-dir>/intermediate/. Overrides the default if set.",
    )
    parser.addoption(
        "--keep-artifacts",
        action="store_true",
        default=False,
        help="Keep all per-test work directories (model files, etc.) even "
        "for passing tests. By default, passing artifacts are deleted.",
    )
    parser.addoption(
        "--cache-dir",
        default="",
        help="Reference-output cache directory. Defaults to "
        "<output-dir>/cache/. Overrides the default if set.",
    )
    parser.addoption(
        "--no-cache",
        action="store_true",
        default=False,
        help="Disable the reference-output cache entirely (always run "
        "ORT CPU for the reference, never read or write the cache).",
    )
    parser.addoption(
        "--refresh-cache",
        action="store_true",
        default=False,
        help="Invalidate any matching cache entry before each test and "
        "re-run the CPU reference (writes the new value back).",
    )

    # --- ort_ep backend options (CLI flags only) ---------------------------
    # Only knobs Python itself consumes are exposed as CLI flags. Knobs
    # consumed by the EP DLL itself (e.g. PATH for dependent DLL lookup,
    # any debug/profile env vars the EP recognises) stay as environment
    # variables -- the shell already owns env setup, and a CLI flag that
    # just copied a value into os.environ for the DLL to read would be
    # useless indirection. See "Example: MorphiZen EP" in
    # `test/numeric/README.md` for a concrete shell-side recipe.
    parser.addoption(
        "--ep-dll",
        default=None,
        help="Path to the EP DLL. Required; tests SKIP if absent.",
    )
    parser.addoption(
        "--ep-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Pass-through entry for ORT's per-provider `provider_options` "
        "dict (the `Dict[str, str]` ORT hands to the EP's "
        "`OrtEpFactory::CreateEp`). Repeatable. Both KEY and VALUE are "
        "EP-specific -- consult your EP's docs for the keys it accepts. "
        "Examples: `--ep-option device_id=0` (CUDA / DirectML), "
        "`--ep-option config_file=/path/to/morphizen_config.json` "
        "(MorphiZen / VitisAI). Values containing '=' are preserved: "
        "only the first '=' is treated as the separator.",
    )
    parser.addoption(
        "--ep-name",
        default="ExecutionProvider",
        help="Name used both (a) as the alias to register the EP DLL "
        "under via `register_execution_provider_library` and "
        "(b) as the key to filter `get_ep_devices()` afterwards. "
        "ORT advertises the registered EP under this exact string, "
        "so any non-empty value 'works' -- pick a meaningful name "
        "matching the EP for log clarity. Default 'ExecutionProvider' "
        "is a generic placeholder.",
    )


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


@pytest.fixture(scope="session")
def _model_runner_session(request):
    """Session-scoped ModelRunner (internal -- use ``model_runner``).

    Resolution order for output paths:
      1. --work-dir / --cache-dir explicit overrides (any combination).
      2. <--output-dir>/intermediate and <--output-dir>/cache derived.
      3. <conftest.py dir>/python_test_output as the implicit root.
    """
    backend_name = request.config.getoption("--backend")
    output_dir = request.config.getoption("--output-dir")
    work_dir = request.config.getoption("--work-dir")
    cache_dir = request.config.getoption("--cache-dir")
    no_cache = request.config.getoption("--no-cache")
    refresh_cache = request.config.getoption("--refresh-cache")
    show_output = request.config.getoption("capture") == "no"

    backend = _BACKENDS[backend_name](request.config)

    here = Path(__file__).parent
    output_root = Path(output_dir) if output_dir else here / "python_test_output"
    if not work_dir:
        work_dir = output_root / "intermediate"
    if not cache_dir:
        cache_dir = output_root / "cache"

    cache = ReferenceCache(
        root=cache_dir,
        no_cache=no_cache,
        refresh=refresh_cache,
    )
    return ModelRunner(
        backend=backend,
        work_dir=str(work_dir),
        cache=cache,
        show_output=show_output,
    )


@pytest.fixture
def model_runner(request, _model_runner_session):
    """Per-test ModelRunner that tags work directories with the test name.

    After the test completes, artifacts for passing tests are deleted
    automatically to conserve disk space. Use ``--keep-artifacts`` to
    override this behaviour.
    """
    _model_runner_session.set_test_name(request.node.name)
    yield _model_runner_session

    keep = request.config.getoption("--keep-artifacts")
    rep = getattr(request.node, "rep_call", None)
    if not keep and rep is not None and rep.passed:
        _model_runner_session.cleanup_last()
