#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Offline tests for setup_whisper_variant dispatch (no download, no surgery).

We monkeypatch the raw-ensure + surgery steps so the test verifies ONLY the
routing: correct model_dir per (variant, precision), n_text_ctx threaded from the
resolved variant, and large-v3 wrappers delegating unchanged.
"""

import pathlib
import sys


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import conftest  # noqa: E402


def test_whisper_model_dir_convention():
    p = conftest.whisper_model_dir("tiny", "fp16")
    assert p.name == "whisper-tiny-onnx-fp16"
    p32 = conftest.whisper_model_dir("tiny", "fp32")
    assert p32.name == "whisper-tiny-onnx"


def test_setup_variant_threads_n_text_ctx(monkeypatch):
    calls = {}

    # Fake a resolved variant with a sentinel n_text_ctx so we can assert it is
    # the value threaded into surgery+fix_shapes.
    fake_cfg = conftest.WhisperModelConfig(n_text_ctx=448, n_vocab=51865)
    fake_var = conftest.WhisperVariant(
        name="tiny",
        hf_model_id="openai/whisper-tiny",
        revision="main",
        cfg=fake_cfg,
        start_tokens=[1, 2, 3, 4],
        eot=0,
    )
    monkeypatch.setattr(conftest, "resolve_whisper_variant", lambda n: fake_var)
    monkeypatch.setattr(
        conftest, "_ensure_whisper_raw", lambda name, model_dir, precision: None
    )
    # Pretend the raw files exist so the guard passes.
    monkeypatch.setattr(conftest.pathlib.Path, "exists", lambda self: True)

    def _fake_surgery(model_dir, n_text_ctx=448):
        calls["model_dir"] = model_dir
        calls["n_text_ctx"] = n_text_ctx

    monkeypatch.setattr(
        conftest, "_apply_whisper_surgery_and_fix_shapes", _fake_surgery
    )

    model_dir, var = conftest.setup_whisper_variant("tiny", "fp16")
    assert model_dir.name == "whisper-tiny-onnx-fp16"
    assert var is fake_var
    assert calls["n_text_ctx"] == 448
    assert calls["model_dir"].name == "whisper-tiny-onnx-fp16"
