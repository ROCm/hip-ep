#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Offline test: _runtime_params resolves large-v3 defaults and variant overrides."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import whisper_infer  # noqa: E402
from conftest import WhisperModelConfig, WhisperVariant  # noqa: E402


def test_runtime_params_default_is_large_v3():
    nl, nh, hd, slots, start, eot, vocab = whisper_infer._runtime_params(None)
    assert nl == 32 and nh == 20 and hd == 64 and slots == 448
    assert start == whisper_infer.START_TOKENS and eot == whisper_infer.EOT
    assert vocab == 51866


def test_runtime_params_variant_override():
    cfg = WhisperModelConfig(
        n_audio_state=384,
        n_audio_layer=4,
        n_audio_head=6,
        n_text_state=384,
        n_text_layer=4,
        n_text_head=6,
        n_text_ctx=448,
        n_mels=80,
        n_vocab=51865,
    )
    var = WhisperVariant(
        name="tiny",
        hf_model_id="openai/whisper-tiny",
        revision="main",
        cfg=cfg,
        start_tokens=[50258, 50259, 50359, 50363],
        eot=50257,
    )
    nl, nh, hd, slots, start, eot, vocab = whisper_infer._runtime_params(var)
    assert nl == 4 and nh == 6 and hd == 64 and slots == 448
    assert start == [50258, 50259, 50359, 50363] and eot == 50257
    assert vocab == 51865
