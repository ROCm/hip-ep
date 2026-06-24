#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Per-variant EP correctness smoke test for Whisper turbo + small sizes.

ORACLE: same-variant MorphiZen-GPU greedy tokens == ORT-CPU greedy tokens. NOT a
verbatim-text bar — small models (tiny/base) have low transcription accuracy, so
an absolute-text assertion would test model quality, not EP correctness. Comparing
EP vs CPU on the SAME variant isolates the EP. large-v3's full correctness +
verbatim-JFK matrix stays in test_whisper.py.

Each variant is built/setup on demand (fp16); a missing build SKIPS cleanly, so a
box without the OGA builder / GPU degrades gracefully. The GPU run is wrapped in a
bounded FD capture to assert the graph compiled on GPU (no silent CPU fallback —
otherwise EP==CPU would be trivially true).
"""

import gc
import os
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from conftest import (  # noqa: E402
    REPO_ROOT,
    make_whisper_inputs,
    setup_jfk_sample,
    setup_whisper_variant,
)
import whisper_infer  # noqa: E402

# turbo (128-mel, 4 decoder layers) + the 80-mel small sizes. large-v3 is covered
# by the full matrix in test_whisper.py, so it is intentionally absent here.
_SMOKE_VARIANTS = ["large-v3-turbo", "tiny", "base", "small", "medium"]

_WHISPER_DATA = REPO_ROOT / "test" / "python" / "data" / "whisper"
_AUDIO = _WHISPER_DATA / "jfk.wav"


@pytest.fixture(scope="module", autouse=True)
def _audio():
    therock = REPO_ROOT / "install" / "therock"
    if therock.exists():
        os.environ["THEROCK_DIST"] = str(therock)
    if not setup_jfk_sample(_WHISPER_DATA):
        pytest.skip("jfk.wav unavailable (network fetch failed, no local cache)")


@pytest.fixture(autouse=True)
def _gc():
    yield
    gc.collect()


@pytest.mark.parametrize("variant_name", _SMOKE_VARIANTS)
def test_variant_smoke(variant_name):
    # Build + surgery + fix_shapes on demand (fp16). Missing build -> skip.
    try:
        model_dir, var = setup_whisper_variant(variant_name, precision="fp16")
    except FileNotFoundError as e:
        pytest.skip(f"{variant_name} fp16 not built: {e}")

    dtype = np.float16
    audio = make_whisper_inputs(_AUDIO, var.cfg)["audio_features"].astype(dtype)

    # CPU reference (dynamic graph, same variant).
    cpu_factory = whisper_infer.make_cpu_session_factory(model_dir)
    cpu_tokens = whisper_infer.greedy_decode_cpu(
        cpu_factory, audio, variant=var, dtype=dtype
    )
    gc.collect()

    # GPU under test. make_morphizen_session_factory raises if the EP is absent.
    try:
        mz_factory = whisper_infer.make_morphizen_session_factory(REPO_ROOT, model_dir)
    except RuntimeError as e:
        pytest.skip(f"MorphiZen EP unavailable: {e}")

    with whisper_infer.CaptureFD() as cap:
        mz_tokens = whisper_infer.greedy_decode_morphizen(
            mz_factory, audio, variant=var, dtype=dtype
        )
    # No silent CPU fallback — else EP==CPU is meaningless (both CPU).
    whisper_infer.assert_compiled_on_gpu(cap.text)

    txt = whisper_infer.decode_text(
        mz_tokens, tokenizer_id=var.hf_model_id, eot=var.eot
    )
    print(f"\n[{variant_name}] GPU greedy: {txt!r}")
    assert mz_tokens == cpu_tokens, (
        f"{variant_name}: EP greedy tokens != CPU greedy tokens\n"
        f"  cpu_len={len(cpu_tokens)} gpu_len={len(mz_tokens)}"
    )
