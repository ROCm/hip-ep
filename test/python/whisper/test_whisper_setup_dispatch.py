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


def test_whisper_hf_repo_mapping():
    # Every variant has an fp16 AMD repo; only large-v3 ships fp32.
    for name in conftest.WHISPER_VARIANTS:
        assert conftest.whisper_hf_repo(name, "fp16") == f"amd/whisper-{name}-onnx-fp16"
    assert conftest.whisper_hf_repo("large-v3", "fp32") == "amd/whisper-large-v3-onnx-fp32"
    # The large-v3 back-compat aliases must agree with the resolver.
    assert conftest.WHISPER_HF_REPO_FP16 == conftest.whisper_hf_repo("large-v3", "fp16")
    assert conftest.WHISPER_HF_REPO_FP32 == conftest.whisper_hf_repo("large-v3", "fp32")


def test_ensure_whisper_raw_downloads_all_fp16_variants(monkeypatch):
    # fp16 of any variant must route to its AMD repo download (not the
    # local-build hint that was the pre-change behavior for non-large-v3).
    calls = {}
    monkeypatch.setattr(
        conftest,
        "_ensure_whisper_raw_downloaded",
        lambda model_dir, repo: calls.setdefault("repo", repo),
    )
    # Pretend the raw files are absent so the download branch is taken.
    monkeypatch.setattr(conftest.pathlib.Path, "exists", lambda self: False)
    for name in ("tiny", "base", "small", "medium", "large-v3-turbo"):
        calls.clear()
        conftest._ensure_whisper_raw(name, conftest.whisper_model_dir(name, "fp16"), "fp16")
        assert calls["repo"] == f"amd/whisper-{name}-onnx-fp16"


def test_ensure_whisper_raw_fp32_non_large_v3_uses_local_hint(monkeypatch):
    # fp32 has no AMD repo for non-large-v3 variants -> local-build hint.
    monkeypatch.setattr(conftest.pathlib.Path, "exists", lambda self: False)
    import pytest

    with pytest.raises(FileNotFoundError, match="No AMD HF fp32 repo"):
        conftest._ensure_whisper_raw("tiny", conftest.whisper_model_dir("tiny", "fp32"), "fp32")


def test_setup_variant_threads_n_text_ctx(monkeypatch):
    calls = {}

    # Fake a resolved variant with a distinguishable sentinel n_text_ctx (999,
    # not 448 which is the default) so the assertion proves the value was
    # threaded, not that the default was used.
    fake_cfg = conftest.WhisperModelConfig(n_text_ctx=999, n_vocab=51865)
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

    def _fake_surgery(model_dir, n_text_ctx=448):  # default stays 448; sentinel is 999
        calls["model_dir"] = model_dir
        calls["n_text_ctx"] = n_text_ctx

    monkeypatch.setattr(
        conftest, "_apply_whisper_surgery_and_fix_shapes", _fake_surgery
    )

    model_dir, var = conftest.setup_whisper_variant("tiny", "fp16")
    assert model_dir.name == "whisper-tiny-onnx-fp16"
    assert var is fake_var
    assert calls["n_text_ctx"] == 999
    assert calls["model_dir"].name == "whisper-tiny-onnx-fp16"
