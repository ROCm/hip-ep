#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Formatters: pure functions turning a normalized Trace into an output artifact.

Each module exposes ``write(trace, path)``. ``WRITERS`` maps a format name to its
writer and the filename suffix appended to the output base (e.g. base ``sample``
-> ``sample_summary.json``). Every format is always emitted at once.
"""

from __future__ import annotations

from . import chrome, dispatches, operators, summary

WRITERS = {
    "summary": (summary.write, "_summary.json"),  # also writes _summary.md
    "operators": (operators.write, "_operators.csv"),
    "families": (operators.write_families, "_families.csv"),
    "dispatches": (dispatches.write, "_dispatches.csv"),
    "trace": (chrome.write, "_trace.json"),
}

__all__ = ["WRITERS", "chrome", "dispatches", "operators", "summary"]
