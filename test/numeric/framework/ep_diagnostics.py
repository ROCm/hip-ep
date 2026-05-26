#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Defense-in-depth helpers for asserting EP execution in numeric tests.

The MorphiZen EP's level-1 pass aborts the process when MLIR compilation
fails *unless* ``HIPDNN_EP_ALLOW_CPU_FALLBACK=1`` is set — see the
"Phase 0 — strict fallback" gotcha in ``CLAUDE.md``. That covers the
compile-time hole, but a session might still land its subgraph on CPU
for orthogonal reasons (the EP returned an empty supported-node list,
ORT rejected the fusion, the user dropped ``session.disable_cpu_ep_fallback``,
...). This module bundles the post-session-init checks the numeric tests
should perform to make those cases loud rather than silent.

The helpers here are intentionally cheap and read-only — no inference is
run, no kernels are launched. Call them immediately after constructing the
``InferenceSession`` so a regression that bypasses the EP is caught at
session-create time rather than masquerading as "matches CPU".
"""

from __future__ import annotations

from typing import Iterable

import onnxruntime as ort


def assert_subgraph_on_ep(
    session: ort.InferenceSession,
    ep_name: str,
    *,
    allowed_other_providers: Iterable[str] = ("CPUExecutionProvider",),
) -> None:
    """Assert that ``ep_name`` appears among the session's active providers.

    The check is two-pronged:

    1. ``session.get_providers()`` lists every provider ORT is willing to
       dispatch nodes to for this session. The target EP MUST be in that
       list — its absence means ORT either rejected the EP at session-create
       time or the test harness never registered it.

    2. Any provider that appears in addition to ``ep_name`` MUST be in
       ``allowed_other_providers``. The default tolerates ``CPUExecutionProvider``
       (always present as the fallback registry entry even when no nodes land
       on it) but rejects anything else. Combined with
       ``session.disable_cpu_ep_fallback=1`` (set by ``OrtEpBackend.run``),
       a CPU entry in the provider list is a tombstone rather than an active
       dispatch target.

    Raises ``AssertionError`` on either failure, with a message that names
    the EP and the unexpected providers — small enough to read in a CI log,
    actionable enough to diagnose the regression class without re-running.

    Parameters
    ----------
    session
        The ``InferenceSession`` to inspect. Must already be initialized.
    ep_name
        The EP name registered via
        ``register_execution_provider_library`` (e.g.
        ``"MorphiZenExecutionProvider"``).
    allowed_other_providers
        Providers that may co-exist with ``ep_name`` without triggering
        the assertion. ``CPUExecutionProvider`` is included by default.
    """
    providers = list(session.get_providers())
    allowed = set(allowed_other_providers) | {ep_name}

    if ep_name not in providers:
        raise AssertionError(
            f"Expected EP {ep_name!r} in session.get_providers(), "
            f"but the session is using {providers!r}. The EP likely "
            f"failed to register, was not selected by ORT's device "
            f"discovery, or the session was created without "
            f"add_provider_for_devices."
        )

    unexpected = [p for p in providers if p not in allowed]
    if unexpected:
        raise AssertionError(
            f"Session has unexpected providers {unexpected!r} alongside "
            f"{ep_name!r}. Allowed providers: {sorted(allowed)!r}. "
            f"This usually means a node fell back to a non-CPU EP or "
            f"the test forgot to pass session.disable_cpu_ep_fallback=1."
        )
