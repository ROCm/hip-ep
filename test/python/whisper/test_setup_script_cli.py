#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Offline CLI-parse test for scripts/setup_whisper_model.py (no setup run)."""

import importlib.util
import pathlib

_SCRIPTS = pathlib.Path(__file__).resolve().parents[3] / "scripts"
_spec = importlib.util.spec_from_file_location(
    "setup_whisper_model", _SCRIPTS / "setup_whisper_model.py"
)
swm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(swm)


def test_resolve_target_default_large_v3():
    name, precision, list_only = swm._resolve_args([])
    assert name == "large-v3" and precision == "fp16"
    assert list_only is False


def test_resolve_target_variant_fp32():
    name, precision, list_only = swm._resolve_args(["--variant", "tiny", "--fp32"])
    assert name == "tiny" and precision == "fp32"
    assert list_only is False


def test_resolve_list_flag():
    name, precision, list_only = swm._resolve_args(["--variant", "base", "--list"])
    assert (name, precision, list_only) == ("base", "fp16", True)
