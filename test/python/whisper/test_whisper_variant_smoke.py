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
    # Cache sessions so the compile/warmup run and the timed run reuse the SAME
    # sessions (each MorphiZen session compiles its ONNX at creation; without the
    # cache the timed run would recompile and the perf numbers would be cold).
    try:
        base_factory = whisper_infer.make_morphizen_session_factory(
            REPO_ROOT, model_dir
        )
    except RuntimeError as e:
        pytest.skip(f"MorphiZen EP unavailable: {e}")
    _session_cache = {}

    def mz_factory(name):
        if name not in _session_cache:
            _session_cache[name] = base_factory(name)
        return _session_cache[name]

    # First run: compiles the graphs (JIT) + proves GPU dispatch; also primes the
    # cached sessions so the timed run below is steady-state.
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

    # ── Perf metrics (steady-state) ──────────────────────────────────────────
    # Re-run on the cached (already-compiled) sessions with timing on, so CI logs
    # carry a per-variant performance line alongside the correctness check. This
    # is the reproducible in-repo source of the variant perf numbers (also
    # measurable standalone via `scripts/transcribe_whisper.py --variant <name>`).
    # decode tok/s = decode_steps / decode_loop_wall; RTF = total compute / audio.
    timings = {}
    whisper_infer.greedy_decode_morphizen(
        mz_factory, audio, variant=var, dtype=dtype, timings=timings
    )
    audio_s = whisper_infer.audio_duration_s(_AUDIO)
    enc, prefill = timings["enc_ms"], timings["prefill_ms"]
    decode, n = timings["decode_ms"], timings["n_decode_steps"]
    tps = n / (decode / 1e3) if decode > 0 else 0.0
    rtf = ((enc + prefill + decode) / 1e3) / audio_s if audio_s > 0 else 0.0
    print(
        f"[{variant_name}] PERF (fp16, gfx-GPU): encoder={enc:.0f}ms "
        f"prefill={prefill:.0f}ms decode={tps:.1f}tok/s ({n} steps) RTF={rtf:.3f}"
    )
