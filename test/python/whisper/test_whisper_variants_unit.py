#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Offline unit tests for the Whisper variant config/token derivation helpers.

No network, no GPU, no built models — these exercise the pure mapping functions
that turn a transformers WhisperConfig / tokenizer into our shape config and
forced-start tokens.
"""

import pathlib
import sys
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from conftest import (  # noqa: E402
    WHISPER_VARIANTS,
    WhisperModelConfig,
    whisper_model_config_from_hf_config,
    whisper_start_tokens,
)


class _FakeTok:
    def __init__(self, mapping):
        self._m = mapping

    def convert_tokens_to_ids(self, t):
        return self._m[t]


def test_registry_has_target_variants():
    for name in ("large-v3", "large-v3-turbo", "tiny", "base", "small", "medium"):
        assert name in WHISPER_VARIANTS
        hf_id, rev = WHISPER_VARIANTS[name]
        assert hf_id.startswith("openai/whisper")
        assert isinstance(rev, str) and rev


def test_model_config_from_hf_config_small():
    c = SimpleNamespace(
        d_model=384,
        encoder_layers=4,
        encoder_attention_heads=6,
        max_source_positions=1500,
        decoder_layers=4,
        decoder_attention_heads=6,
        max_target_positions=448,
        num_mel_bins=80,
        vocab_size=51865,
    )
    cfg = whisper_model_config_from_hf_config(c)
    assert isinstance(cfg, WhisperModelConfig)
    assert cfg.n_audio_state == 384 and cfg.n_text_state == 384
    assert cfg.n_text_layer == 4 and cfg.n_text_head == 6
    assert cfg.n_mels == 80 and cfg.n_vocab == 51865
    assert cfg.n_text_ctx == 448 and cfg.n_audio_ctx == 1500
    # head_dim invariant
    assert cfg.n_text_state // cfg.n_text_head == 64


def test_model_config_from_hf_config_turbo():
    c = SimpleNamespace(
        d_model=1280,
        encoder_layers=32,
        encoder_attention_heads=20,
        max_source_positions=1500,
        decoder_layers=4,
        decoder_attention_heads=20,
        max_target_positions=448,
        num_mel_bins=128,
        vocab_size=51866,
    )
    cfg = whisper_model_config_from_hf_config(c)
    assert cfg.n_audio_layer == 32 and cfg.n_text_layer == 4
    assert cfg.n_mels == 128 and cfg.n_vocab == 51866


def test_start_tokens_large_v3_layout():
    # large-v3 (51866 vocab): transcribe/notimestamps at the high IDs.
    m = {
        "<|startoftranscript|>": 50258,
        "<|en|>": 50259,
        "<|transcribe|>": 50360,
        "<|notimestamps|>": 50364,
        "<|endoftext|>": 50257,
    }
    start, eot = whisper_start_tokens(_FakeTok(m))
    assert start == [50258, 50259, 50360, 50364]
    assert eot == 50257


def test_start_tokens_multilingual_shift():
    # smaller variants (51865 vocab): one fewer language -> transcribe/
    # notimestamps shifted DOWN by one vs large-v3. The helper must read the
    # actual IDs, never hardcode the large-v3 ones.
    m = {
        "<|startoftranscript|>": 50258,
        "<|en|>": 50259,
        "<|transcribe|>": 50359,
        "<|notimestamps|>": 50363,
        "<|endoftext|>": 50257,
    }
    start, eot = whisper_start_tokens(_FakeTok(m))
    assert start == [50258, 50259, 50359, 50363]
    assert eot == 50257
