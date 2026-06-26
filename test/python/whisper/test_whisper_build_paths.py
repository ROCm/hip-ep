#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Offline tests for build_whisper_models path/variant resolution (no OGA run)."""

import importlib.util
import pathlib

_SCRIPTS = pathlib.Path(__file__).resolve().parents[3] / "scripts"
_spec = importlib.util.spec_from_file_location(
    "build_whisper_models", _SCRIPTS / "build_whisper_models.py"
)
bwm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bwm)


def test_output_dirs_convention():
    fp32, fp16 = bwm.whisper_output_dirs("tiny")
    assert fp32.name == "whisper-tiny-onnx"
    assert fp16.name == "whisper-tiny-onnx-fp16"
    assert fp32.parent.name == "models"


def test_default_variants_cover_scope():
    for v in ("large-v3-turbo", "tiny", "base", "small", "medium"):
        assert v in bwm.DEFAULT_VARIANTS
        assert v in bwm.VARIANT_SOURCES


def test_variant_sources_match_conftest():
    # The builder duplicates the (hf_id, revision) table for standalone-ness; it
    # MUST stay in sync with conftest.WHISPER_VARIANTS.
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from conftest import WHISPER_VARIANTS

    for name, src in bwm.VARIANT_SOURCES.items():
        assert WHISPER_VARIANTS[name] == src
    # reverse: every conftest variant must also be in the builder table
    for name in WHISPER_VARIANTS:
        assert name in bwm.VARIANT_SOURCES, (
            f"{name} missing from builder VARIANT_SOURCES"
        )
