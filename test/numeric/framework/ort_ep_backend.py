#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""ORT execution-provider backend for the numeric test suite.

This module is wired up by `conftest.py` as the ``ort_ep`` backend and
loads a single, specific EP via ORT's CAPI. The EP's identity (the
name the C++ side registers as) is supplied at runtime via
``--ep-name``, and any per-provider configuration is forwarded
verbatim to ORT's ``provider_options`` dict via one or more
``--ep-option KEY=VALUE`` flags, so this module stays generic across
EPs. See the "Example: MorphiZen EP" section of
``test/numeric/README.md`` for a canonical end-to-end recipe (DLL
path, EP name, provider options, PATH-prep).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .backend import Backend

# Layering rule for this module:
#
#   * Knobs Python itself consumes (EP DLL path) are taken from pytest
#     CLI flags only -- no env-var fallback. Defaults belong to the
#     caller's wrapper, not to this module, so the framework stays
#     portable across CI tarballs, system installs, and arbitrary
#     artefact directories.
#
#   * Per-provider configuration is forwarded verbatim into ORT's
#     `provider_options` dict via repeating `--ep-option KEY=VALUE`
#     flags. The framework does NOT know about any specific provider
#     key (e.g. MorphiZen's "config_file", CUDA's "device_id"); it just
#     copies the strings the caller supplied into the dict ORT hands
#     to the EP's `OrtEpFactory::CreateEp`. Consult your EP's docs
#     for the keys it accepts.
#
#   * Knobs the EP DLL itself consumes via the *environment* (PATH for
#     dependent DLL lookup, any debug/profile env vars the EP
#     recognises) stay as environment variables only. We never expose
#     a CLI flag whose only job would be to copy a value into
#     os.environ for the DLL to read -- that surface already exists at
#     the shell layer.
#
# The single piece of implicit PATH manipulation this module does is
# prepending dirname(--ep-dll). That covers the common case where the
# EP DLL is packaged alongside small co-located dependencies (e.g. a
# compiler DLL) that the Windows loader needs to find when the EP
# registers itself. Larger external runtimes (ROCm, etc.) are the
# shell's responsibility -- see "Example: MorphiZen EP" in
# ``test/numeric/README.md``.


class OrtEpBackend(Backend):
    """Runs ONNX models through an ORT execution provider DLL.

    On construction the EP DLL is registered with ORT and the EP device
    is discovered. Each :meth:`run` call creates a new session with
    ``session.disable_cpu_ep_fallback=1`` so that session creation fails
    if any node would silently fall back to CPU -- the whole point of the
    numeric suite is to test EP kernels, not the CPU fallback.
    """

    def __init__(
        self,
        ep_name: str,
        ep_dll_path: str,
        provider_options: dict[str, str] | None = None,
    ) -> None:
        self._ep_name = ep_name
        self._ep_dll = Path(ep_dll_path)
        self._provider_options: dict[str, str] = dict(provider_options or {})
        self._ep_device = self._register_ep()

    @property
    def name(self) -> str:
        return self._ep_name

    def _register_ep(self):
        if not self._ep_dll.exists():
            raise FileNotFoundError(f"EP DLL not found: {self._ep_dll}")

        tag = f"[{self._ep_name}]"
        print(f"{tag} ORT version : {ort.__version__}")
        print(f"{tag} EP DLL      : {self._ep_dll}")
        if self._provider_options:
            print(f"{tag} provider_options ({len(self._provider_options)}):")
            for k, v in self._provider_options.items():
                print(f"{tag}   {k} = {v}")
        else:
            print(f"{tag} provider_options: <none>")

        # The first arg to register_execution_provider_library becomes
        # the EP's advertised ep_name in get_ep_devices(). Any non-empty
        # string works -- the post-registration filter below will find
        # exactly the device we just registered.
        capi = ort.capi._pybind_state
        capi.register_execution_provider_library(self._ep_name, str(self._ep_dll))
        print(f"{tag} register_execution_provider_library: OK")

        all_devices = capi.get_ep_devices()
        print(f"{tag} EP devices ({len(all_devices)}):")
        ep_device = None
        for d in all_devices:
            marker = ""
            if d.ep_name == self._ep_name:
                ep_device = d
                marker = "  <-- selected"
            print(f"{tag}   {d.ep_name} ({d.ep_vendor}){marker}")

        if ep_device is None:
            # Should be unreachable: register_execution_provider_library
            # didn't raise but the device list doesn't contain our name.
            # If it ever fires it's a real ORT issue, so RuntimeError is
            # the right signal -- not a silent skip.
            seen = sorted({d.ep_name for d in all_devices})
            raise RuntimeError(
                f"register_execution_provider_library succeeded but no "
                f"EP device named {self._ep_name!r} appeared. "
                f"ORT saw: {seen}."
            )
        return ep_device

    def make_session(self, model_path: str) -> ort.InferenceSession:
        """Build an ORT InferenceSession bound to this EP.

        Exposed (in addition to ``run``) so dynamic-shape tests can
        verify that a SINGLE session accepts multiple shape combinations
        across consecutive ``sess.run`` calls -- the canonical promise
        of leaving dim_params symbolic. Tests that don't care about
        session reuse should keep using ``run`` (which destroys the
        session per call, matching the surrounding numeric-test
        memory discipline).
        """
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        opts.add_provider_for_devices([self._ep_device], dict(self._provider_options))
        sess = ort.InferenceSession(model_path, sess_options=opts)
        # Belt-and-braces: even with `disable_cpu_ep_fallback=1` set above,
        # confirm the session actually picked up the EP and didn't quietly
        # route the subgraph elsewhere. See framework/ep_diagnostics.py for
        # the rationale (Phase 0 "strict fallback").
        from .ep_diagnostics import assert_subgraph_on_ep

        assert_subgraph_on_ep(sess, self._ep_name)
        return sess

    def run(
        self,
        model_path: str,
        inputs: list[np.ndarray],
    ) -> list[np.ndarray]:
        sess = self.make_session(model_path)
        input_dict = {sess.get_inputs()[i].name: inp for i, inp in enumerate(inputs)}
        outputs = sess.run(None, input_dict)
        del sess
        return outputs


def _prepend_path(directory: Path) -> None:
    """Prepend *directory* to PATH so the EP DLL can find its dependencies."""
    if not directory.exists():
        return
    current = os.environ.get("PATH", "")
    if str(directory) in current:
        return
    os.environ["PATH"] = str(directory) + os.pathsep + current


def _parse_ep_options(raw: list[str]) -> dict[str, str]:
    """Parse repeating ``--ep-option KEY=VALUE`` strings into a dict.

    Only the first ``=`` separates key from value, so values containing
    ``=`` (e.g. a path with an ``=`` in it, or a JSON-ish blob) are
    preserved verbatim. Empty keys, duplicate keys, and entries without
    ``=`` are rejected loudly via ``pytest.skip`` -- silently dropping
    them would hide config errors that the EP would otherwise notice
    at session-create time.
    """
    import pytest

    out: dict[str, str] = {}
    for entry in raw:
        if "=" not in entry:
            pytest.skip(
                f"--ep-option entry {entry!r} is malformed: expected KEY=VALUE."
            )
        key, value = entry.split("=", 1)
        key = key.strip()
        if not key:
            pytest.skip(f"--ep-option entry {entry!r} has an empty key.")
        if key in out:
            pytest.skip(
                f"--ep-option key {key!r} given more than once "
                f"(values {out[key]!r} and {value!r}); ORT's "
                "provider_options dict only stores one value per key."
            )
        out[key] = value
    return out


def create(pytest_config=None) -> OrtEpBackend:
    """Create an :class:`OrtEpBackend` from pytest CLI flags.

    Required:
      ``--ep-dll <path>``  Path to the EP DLL.

    Optional:
      ``--ep-name <name>``       Alias the EP DLL gets registered
      under; ORT then advertises the device with this name. Any
      non-empty string works -- pick something meaningful for log
      clarity (the default ``"ExecutionProvider"`` is a generic
      placeholder).
      ``--ep-option KEY=VALUE``  Repeatable. Each entry is forwarded
      verbatim into the ``provider_options`` dict ORT passes to the
      EP's ``OrtEpFactory::CreateEp``. Both KEY and VALUE are
      EP-specific; the framework has no knowledge of any particular
      provider key.

    Skips the suite cleanly (rather than failing) when ``--ep-dll`` is
    not supplied, so the framework stays runnable on machines without
    a local build.

    Runtime DLL search for any external dependencies the EP requires
    at session-create time (vendor runtimes, kernel libraries, ...) is
    the **shell's** responsibility, not this module's: ensure the
    relevant directories are on ``PATH`` before invoking pytest. See
    "Example: MorphiZen EP" in ``test/numeric/README.md`` for a
    worked-through recipe.
    """
    import pytest

    ep_dll = pytest_config.getoption("--ep-dll") if pytest_config is not None else None
    ep_name = (
        pytest_config.getoption("--ep-name")
        if pytest_config is not None
        else "ExecutionProvider"
    )
    raw_options: list[str] = (
        pytest_config.getoption("--ep-option") if pytest_config is not None else []
    )
    provider_options = _parse_ep_options(raw_options)

    if not ep_dll:
        pytest.skip(
            "EP DLL not supplied. Pass --ep-dll <path> "
            "(see test/numeric/README.md for a worked example)."
        )

    ep_dll_path = Path(ep_dll)
    if not ep_dll_path.exists():
        pytest.skip(f"EP DLL not found at {ep_dll_path}.")

    # Absolutise BEFORE handing the path to ORT. ORT's
    # `register_execution_provider_library` resolves a relative DLL
    # path against the onnxruntime *package* directory (e.g.
    # `<site-packages>/onnxruntime/capi/<ep_dll>`), NOT against the
    # process cwd. A user who copy-pastes the README recipe from
    # `test/numeric/` (where `--ep-dll ..\..\install\dist\bin\foo.dll`
    # is the natural form) would otherwise get a confusing
    # `Error 126: The specified module could not be found` from ORT
    # even though our own `.exists()` check above (which DOES go
    # through cwd) had just succeeded. `.resolve()` is non-strict so
    # this is safe even when the file is gone -- the explicit
    # skip-on-missing above is what catches that case.
    ep_dll_path = ep_dll_path.resolve()

    # Prepend the EP DLL's own directory to PATH so the Windows loader
    # finds any small co-located dependencies the EP needs at
    # registration time. This is an invariant of how an EP is
    # packaged, not a user choice. Use the resolved path so the PATH
    # entry is an absolute directory (relative PATH entries on Windows
    # are looked up against cwd at every call, which is fragile).
    _prepend_path(ep_dll_path.parent)

    return OrtEpBackend(
        ep_name=ep_name,
        ep_dll_path=str(ep_dll_path),
        provider_options=provider_options,
    )
