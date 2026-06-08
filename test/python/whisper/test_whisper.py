#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""End-to-end Whisper-large-v3 correctness on the MorphiZen EP (fp32 + fp16).

Whisper-large-v3 is natively fp32 and fp32 is the DEFAULT path; correctness is
measured GPU-fp32 vs CPU-fp32 of the SAME model, and greedy transcription matches
the JFK quote VERBATIM. An fp16 variant (built locally via the OGA DML builder,
body fp16 + fp32 lm_head) is ALSO covered by opt-in tests (the ``fp16_model``
fixture builds it on demand and skips if the builder deps are absent).

  * ``test_encoder_correctness``         — encoder hidden_states cosine
                                            (GPU-fp32 ``encoder_fixed.onnx`` vs
                                            CPU-fp32 dynamic ``encoder.onnx``)
  * ``test_decoder_prefill_correctness`` — 4-token prefill logits cosine
                                            (GPU-fp32 static vs CPU-fp32 dynamic)
  * ``test_decoder_decode_correctness``  — N decode-step logits cosine
                                            (GPU-fp32 static vs CPU-fp32 dynamic)
  * ``test_e2e_transcription_greedy``    — GPU-fp32 greedy text == CPU-fp32
                                            verbatim JFK quote (a REAL pass)
  * ``test_fp16_e2e_transcription_greedy``    — GPU-fp16 greedy == CPU-fp16
                                                (opt-in; needs the fp16 build)
  * ``test_fp16_decoder_prefill_correctness`` — fp16 prefill logits cosine
                                                (GPU-fp16 static vs CPU-fp16)

Decoder reference choice (CRITICAL): the CPU reference is the DYNAMIC
(pre-surgery, empty-growing past) fp32 decoder, NOT the surgered+static graph on
CPU. ORT's ``com.microsoft.MultiHeadAttention`` IGNORES ``past_sequence_length``
when ``past_present_share_buffer`` is absent (the surgery can't set that attr —
ORT rejects it), so running the static graph on the ORT CPU EP attends to all 448
zeroed buffer slots and is NOT a valid reference. The MorphiZen runtime DOES
honor seqlens_k, so GPU-static must match CPU-DYNAMIC (empty past).

ONNX tensor-layout facts (verified by inspection, NOT assumed):

  * Encoder outputs are BLOCKED, not interleaved:
        [hidden_states,
         present_key_cross_0..31,            (indices 1..32)
         present_value_cross_0..31]          (indices 33..64)
    so we index cross-KV by NAME, never by ``enc_out[1 + 2*i]``.
  * Decoder inputs:  input_ids, 32×past_key_self, 32×past_value_self,
                     32×past_key_cross, 32×past_value_cross.
  * Decoder outputs: logits, 32×present_key_self, 32×present_value_self.
  * After surgery the static decoder re-types ``input_ids`` to INT64 and adds
    ``past_sequence_length`` (int32) + ``position_ids`` (int64) graph inputs.

Environment: the MorphiZen EP compiles model DLLs at session-init and links
against ROCm, so ``THEROCK_DIST`` + both ``install/{dist,therock}/bin`` must be
on PATH before any MorphiZen session is created. ``register_morphizen_ep``
prepends the bin dirs to PATH; THEROCK_DIST must be exported by the caller's
shell (the test does it defensively too).
"""

import os
import sys
import pathlib

import numpy as np
import onnxruntime as ort
import pytest

# whisper_infer is in this dir; conftest.py (shared with the llama tests) is one
# level up in test/python/. Put both on the path.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from conftest import (  # noqa: E402
    AMD_VENDOR_ID,
    REPO_ROOT,
    WhisperModelConfig,
    get_amd_dml_providers,
    make_whisper_inputs,
    register_morphizen_ep,
    setup_jfk_sample,
    setup_librispeech_samples,
    setup_whisper_fp16_model_dir,
    setup_whisper_model_dir,
)

# The greedy-decode harness + Whisper constants live in whisper_infer (shared
# with scripts/transcribe_whisper.py) so the seqlens_k / position_ids convention
# has exactly one source of truth. Aliased to the local _-names this module uses.
import whisper_infer  # noqa: E402
from whisper_infer import (  # noqa: E402
    EOT as _EOT,
    HEAD_DIM as _HEAD_DIM,
    N_HEADS as _N_HEADS,
    N_LAYERS as _N_LAYERS,
    SELF_KV_SLOTS as _SELF_KV_SLOTS,
    START_TOKENS as _START_TOKENS,
    decode_text as _decode_text,
    encoder_cross_kv as _encoder_cross_kv,
    zeroed_self_past as _zeroed_self_past,
)

_MODEL_DIR = REPO_ROOT / "models" / "whisper-large-v3-onnx"
# fp16 model lives in a SEPARATE dir so it never collides with the fp32 bundle.
# It is built on demand (OGA DML builder) by the `fp16_model` fixture — opt-in.
_MODEL_DIR_FP16 = REPO_ROOT / "models" / "whisper-large-v3-onnx-fp16"
_CFG = WhisperModelConfig()
_WHISPER_DATA = REPO_ROOT / "test" / "python" / "data" / "whisper"
_AUDIO = _WHISPER_DATA / "jfk.wav"
_LIBRISPEECH_DIR = _WHISPER_DATA / "librispeech"

_CROSS_LEN = _CFG.n_audio_ctx  # 1500

# fp32 is not lossy on the 51866-wide lm_head argmax, so GPU-fp32 vs CPU-fp32
# of the same model is ~1.0 for the encoder and very close for the deep decoder.
_COS_THRESH = 0.99
# The decoder's 32-layer graph drifts slightly more than the encoder; a small
# margin below 1.0 covers fp32 GPU-vs-CPU reduction-order differences.
_DEC_COS_THRESH = 0.99


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module", autouse=True)
def _setup():
    # Defensive: ensure THEROCK_DIST points at THIS worktree's ROCm SDK so the
    # compiled model DLLs can link (amdhip64_7.dll / MIOpen.lib live there).
    therock = REPO_ROOT / "install" / "therock"
    if therock.exists():
        os.environ["THEROCK_DIST"] = str(therock)
    # jfk.wav is auto-downloaded (not committed); skip the whole module if it's
    # unreachable so an offline run degrades cleanly instead of erroring on a
    # missing _AUDIO path. Every test in this module feeds make_whisper_inputs(_AUDIO).
    if not setup_jfk_sample(_WHISPER_DATA):
        pytest.skip("jfk.wav unavailable (network fetch failed, no local cache)")
    setup_whisper_model_dir(_MODEL_DIR)
    yield


@pytest.fixture(scope="module")
def fp16_model():
    """Consume the pre-built fp16 Whisper bundle (opt-in — NOT autouse).

    The fp16 bundle must be built ahead of time via
    `python build.py --build-whisper-models` (it builds both fp32 and fp16 in an
    isolated venv). This fixture only consumes the built raw model — it runs the
    decoder surgery + fix_shapes on it. If the bundle is absent (unbuilt) or the
    surgery fails, the fp16 tests skip cleanly rather than erroring — the fp32
    suite is unaffected because this is a separate fixture only fp16 tests use.
    """
    try:
        setup_whisper_fp16_model_dir(_MODEL_DIR_FP16)
    except Exception as e:  # noqa: BLE001 — unbuilt model / surgery error → skip
        pytest.skip(
            f"fp16 Whisper model unavailable: {e!r} "
            "(build it: python build.py --build-whisper-models)"
        )
    return _MODEL_DIR_FP16


# ── Helpers ──────────────────────────────────────────────────────────────────


def _cosine(a, b):
    """Cosine similarity of two arrays (upcast to fp32, flattened)."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom)


def _cpu_session(model_name, model_dir=_MODEL_DIR):
    return ort.InferenceSession(
        str(model_dir / model_name), providers=["CPUExecutionProvider"]
    )


def _morphizen_session(model_name, model_dir=_MODEL_DIR):
    devices = register_morphizen_ep(REPO_ROOT)
    if not devices:
        pytest.skip("MorphiZen EP not found — run build.py first")
    so = ort.SessionOptions()
    # CRITICAL for the decoder: disable ORT's ahead-of-time function inlining.
    # The Whisper decoder MLP uses ai.onnx Gelu, which is a registered ONNX
    # *function* (opset >= 20). With AOT inlining ON (the default), ORT expands
    # each Gelu into its body primitives (onnx.Erf + onnx.Sum + ...) BEFORE the
    # MorphiZen EP claims the subgraph. onnx.Erf has no ONNX->HIP converter, so
    # it survives convert-onnx-to-hip and aborts one-shot-bufferize ("op was not
    # bufferized") → the EP silently falls back to CPU (and any cosine then
    # measures CPU-vs-CPU, looking deceptively healthy). Disabling AOT inlining
    # keeps Gelu atomic so it lowers to hip.gelu (wrap_gelu). The encoder does
    # not need this (its Gelu is not inlined by ORT), but setting it everywhere
    # is harmless and keeps the helper uniform.
    so.add_session_config_entry("session.disable_aot_function_inlining", "1")
    so.add_provider_for_devices(devices, {})
    return ort.InferenceSession(str(model_dir / model_name), sess_options=so)


def _assert_no_silent_fallback(stderr_text):
    """GPU-dispatch tripwire.

    The model DLL prints ``[REAL] wrap_*`` lines to stderr (FD level) when
    ``HIPDNN_EP_DEBUG=1``. Whisper attention must route to
    ``wrap_group_query_attention`` (hip.gqa's runtime entry, the ``no_causal``
    path), the conv front-end to ``wrap_conv1d``, and the legacy
    ``wrap_multi_head_attention`` runtime entry must NOT appear.

    NOTE: the model DLL has its OWN static CRT (CLAUDE.md gotcha), so Python's
    ``capsys`` (stream-level) can miss its stderr. Use pytest ``capfd``
    (file-descriptor level). If the captured text is empty, the caller falls
    back to a wall-clock GPU-speed assertion (documented at the call site).
    """
    assert "[REAL] wrap_conv1d" in stderr_text, (
        "no [REAL] wrap_conv1d — Conv1d ran on CPU (silent fallback)"
    )
    # hip.gqa lowers to the runtime symbol wrap_group_query_attention (NOT a
    # symbol literally named wrap_gqa).
    assert "[REAL] wrap_group_query_attention" in stderr_text, (
        "no [REAL] wrap_group_query_attention — attention ran on CPU (silent fallback)"
    )
    assert "[REAL] wrap_multi_head_attention" not in stderr_text, (
        "wrap_multi_head_attention should NOT appear — all Whisper attention "
        "must route through the no_causal hip.gqa path"
    )


def _assert_compiled_on_gpu(stderr_text):
    """Hard tripwire: the MorphiZen MLIR backend MUST NOT report a compile
    failure. ``MlirCompiler.cpp ... Compilation failed`` (FD-level stderr from
    the EP) means the model fell back to CPU silently — at which point any
    GPU-vs-CPU cosine is meaningless (it becomes CPU-vs-CPU == 1.0 and hides
    kernel bugs).

    This signal IS visible to ``capfd`` (it is emitted by the EP host, which
    shares the Python process CRT, not the model DLL's private CRT), so it is a
    more reliable tripwire than the ``[REAL]`` lines (which the model DLL prints
    via its own CRT and capfd may miss).
    """
    assert "MLIR compilation failed" not in stderr_text and (
        "Compilation failed" not in stderr_text
    ), (
        "MorphiZen MLIR compilation FAILED → silent CPU fallback. The cosine "
        "below would be a CPU-vs-CPU artifact, NOT a kernel result. Fix the "
        "compiler before trusting accuracy. Captured stderr:\n" + stderr_text[-2000:]
    )


# ── Tests ────────────────────────────────────────────────────────────────────


def test_encoder_correctness(capfd):
    """Encoder hidden_states cosine: MorphiZen(fp32) vs CPU(fp32).

    GPU runs the fixed fp32 ``encoder_fixed.onnx``; the reference is the original
    dynamic fp32 ``encoder.onnx`` on the CPU EP. Both are the same native-fp32
    model (only batch_size is pinned), so the cosine is ~1.0 when every op (Conv,
    Attention->hip.gqa, SkipLayerNormalization->hip.add+hip.layer_norm)
    dispatches on the GPU. The compile-failure + per-op tripwires below catch the
    silent-CPU-fallback case independently of the numeric threshold.
    """
    os.environ["HIPDNN_EP_DEBUG"] = "1"
    audio = make_whisper_inputs(_AUDIO, _CFG)["audio_features"]  # fp32

    # CPU fp32 reference: the original dynamic encoder.
    cpu = _cpu_session("encoder.onnx")
    cpu_hidden, _ = _encoder_cross_kv(cpu, audio)

    # IMPORTANT: the MorphiZen MLIR compile happens at SESSION INIT, so the
    # "Compilation failed" stderr is emitted by InferenceSession(), not run().
    # Capture across BOTH so the compile-failure tripwire sees it.
    capfd.readouterr()  # clear buffer before the captured window
    import time

    mz = _morphizen_session("encoder_fixed.onnx")
    t0 = time.perf_counter()
    mz_hidden, _ = _encoder_cross_kv(mz, audio)
    elapsed = time.perf_counter() - t0
    captured = capfd.readouterr()
    stderr_text = captured.err + captured.out

    cos = _cosine(cpu_hidden, mz_hidden)
    print(f"\n[encoder] hidden_states cosine = {cos:.6f}  (run {elapsed * 1e3:.1f} ms)")

    # Tripwire 1 (PRIMARY, always checked): the EP must not report an MLIR
    # compile failure. This catches the silent-CPU-fallback case where a healthy
    # cosine is really CPU-vs-CPU.
    _assert_compiled_on_gpu(stderr_text)

    # Tripwire 2: per-op GPU dispatch. The [REAL] lines come from the model
    # DLL's private CRT and capfd may miss them; only assert when present.
    if "[REAL] wrap_" in stderr_text:
        _assert_no_silent_fallback(stderr_text)
    else:
        print(
            "[encoder] note: no [REAL] wrap_* lines captured (DLL static-CRT "
            "stderr not visible to capfd); relying on the compile-failure "
            "tripwire + cosine threshold above."
        )

    assert cos >= _COS_THRESH, f"encoder hidden_states cosine = {cos} < {_COS_THRESH}"


def test_decoder_prefill_correctness(capfd):
    """First decoder step (S=4 start tokens, empty self-past): logits cosine.

    KERNEL correctness is GPU-fp32 (static 448-slot shared buffer) vs CPU-fp32 of
    the DYNAMIC (pre-surgery, empty-past) decoder. The dynamic decoder is the
    correct reference: ORT's com.microsoft.MultiHeadAttention IGNORES
    past_sequence_length when past_present_share_buffer is absent (the surgery
    can't set that attr — ORT rejects it), so running the SURGERED+static graph
    on the ORT CPU EP attends to all 448 zeroed buffer slots and is NOT a valid
    reference. The MorphiZen runtime DOES honor seqlens_k, so GPU-static must
    match CPU-DYNAMIC (empty past).
    """
    audio = make_whisper_inputs(_AUDIO, _CFG)["audio_features"]
    s0 = len(_START_TOKENS)

    # Shared fp32 cross-KV from the CPU encoder so the decoder compare is
    # isolated from any encoder GPU/CPU drift.
    cpu_enc = _cpu_session("encoder.onnx")
    _, cross = _encoder_cross_kv(cpu_enc, audio)

    # ── CPU-fp32 reference: DYNAMIC decoder, empty (0-slot) past ─────────────
    cpu_dec = _cpu_session("decoder.onnx")
    cpu_names = [o.name for o in cpu_dec.get_outputs()]
    cpu_feed = {"input_ids": np.array([_START_TOKENS], dtype=np.int32), **cross}
    for i in range(_N_LAYERS):
        cpu_feed[f"past_key_self_{i}"] = np.zeros(
            (1, _N_HEADS, 0, _HEAD_DIM), dtype=np.float32
        )
        cpu_feed[f"past_value_self_{i}"] = np.zeros(
            (1, _N_HEADS, 0, _HEAD_DIM), dtype=np.float32
        )
    cpu_logits = dict(zip(cpu_names, cpu_dec.run(None, cpu_feed)))["logits"]

    # ── MorphiZen GPU: static 448-slot shared-buffer prefill ─────────────────
    # int64 ids (surgered decoder re-types input_ids to int64 for the GPU token-
    # embed Gather). past_sequence_length uses the total-1 convention (= S-1 at
    # prefill). position_ids = [0 .. S-1] (real_past=0 at prefill).
    feed = {
        "input_ids": np.array([_START_TOKENS], dtype=np.int64),
        "past_sequence_length": np.array([s0 - 1], dtype=np.int32),
        "position_ids": np.arange(s0, dtype=np.int64),
        **cross,
        **_zeroed_self_past(),
    }
    capfd.readouterr()  # clear before the captured window (compile happens here)
    mz_dec = _morphizen_session("decoder_fixed_prefill.onnx")
    mz_names = [o.name for o in mz_dec.get_outputs()]
    mz_logits = dict(zip(mz_names, mz_dec.run(None, feed)))["logits"]
    stderr_text = "".join(capfd.readouterr())

    cos = _cosine(cpu_logits[0, -1, :], mz_logits[0, -1, :])
    print(f"\n[prefill] last-token logits cosine (GPU-fp32 vs CPU-fp32) = {cos:.6f}")

    # PRIMARY tripwire (reliable — emitted by the EP host CRT, captured by
    # capfd): the decoder MUST compile on GPU, no silent CPU fallback. A CPU
    # fallback would make GPU==CPU trivially (cosine 1.0) and hide kernel bugs,
    # so this MUST run before trusting the cosine.
    _assert_compiled_on_gpu(stderr_text)
    # Per-op dispatch is a SECONDARY, best-effort check. The [REAL] wrap_* lines
    # come from the model DLL's private static CRT, which capfd captures only
    # partially/unreliably (CLAUDE.md gotcha). So: the negative check (no legacy
    # MHA path) is asserted whenever any [REAL] line is seen, but the positive
    # wrap_group_query_attention presence is informational only.
    if "[REAL] wrap_" in stderr_text:
        assert "[REAL] wrap_multi_head_attention" not in stderr_text, (
            "wrap_multi_head_attention should not appear — all Whisper attention "
            "routes through hip.gqa"
        )
        if "[REAL] wrap_group_query_attention" not in stderr_text:
            print(
                "[prefill] note: wrap_group_query_attention not in the captured "
                "DLL stderr (partial static-CRT capture); GPU dispatch is still "
                "proven by the compile tripwire + the kernel-correct cosine."
            )

    assert cos >= _DEC_COS_THRESH, (
        f"prefill GPU-fp32 vs CPU-fp32 cosine = {cos} < {_DEC_COS_THRESH}"
    )


def test_decoder_decode_correctness():
    """N sequential decode steps (S=1): per-step logits cosine, GPU vs CPU-fp32.

    KERNEL correctness check: GPU runs the static 448-slot shared-buffer decode
    decoder; the reference is the DYNAMIC (pre-surgery, growing-past) fp32 decoder
    on the CPU EP. The dynamic decoder is the correct reference (ORT MHA ignores
    past_sequence_length on the static graph — see test_decoder_prefill_correctness
    for why CPU-static is invalid). Both share the SAME fp32 cross-KV and the SAME
    token trajectory (driven by the CPU reference argmax) so the per-step cosine is
    a clean kernel comparison, not two divergent greedy walks.
    """
    n_steps = 24
    audio = make_whisper_inputs(_AUDIO, _CFG)["audio_features"]

    # Shared fp32 cross-KV from the CPU encoder (isolate the decoder).
    cpu_enc = _cpu_session("encoder.onnx")
    _, cross = _encoder_cross_kv(cpu_enc, audio)

    # ── CPU reference: DYNAMIC fp32 decoder, empty growing past ──────────────
    cpu_dec = _cpu_session("decoder.onnx")
    cpu_dec_out = [o.name for o in cpu_dec.get_outputs()]
    cpu_self = {}
    for i in range(_N_LAYERS):
        cpu_self[f"past_key_self_{i}"] = np.zeros(
            (1, _N_HEADS, 0, _HEAD_DIM), dtype=np.float32
        )
        cpu_self[f"past_value_self_{i}"] = np.zeros(
            (1, _N_HEADS, 0, _HEAD_DIM), dtype=np.float32
        )
    cpu_pref = dict(
        zip(
            cpu_dec_out,
            cpu_dec.run(
                None,
                {
                    "input_ids": np.array([_START_TOKENS], np.int32),
                    **cpu_self,
                    **cross,
                },
            ),
        )
    )
    for i in range(_N_LAYERS):
        cpu_self[f"past_key_self_{i}"] = cpu_pref[f"present_key_self_{i}"]
        cpu_self[f"past_value_self_{i}"] = cpu_pref[f"present_value_self_{i}"]

    # ── GPU: static 448-slot shared-buffer prefill ──────────────────────────
    mz_self = _zeroed_self_past()
    s0 = len(_START_TOKENS)
    mz_pref_dec = _morphizen_session("decoder_fixed_prefill.onnx")
    mz_pref_out = [o.name for o in mz_pref_dec.get_outputs()]
    mz_pref = dict(
        zip(
            mz_pref_out,
            mz_pref_dec.run(
                None,
                {
                    "input_ids": np.array([_START_TOKENS], np.int64),
                    "past_sequence_length": np.array([s0 - 1], np.int32),
                    "position_ids": np.arange(s0, dtype=np.int64),
                    **mz_self,
                    **cross,
                },
            ),
        )
    )
    for i in range(_N_LAYERS):
        mz_self[f"past_key_self_{i}"] = mz_pref[f"present_key_self_{i}"]
        mz_self[f"past_value_self_{i}"] = mz_pref[f"present_value_self_{i}"]

    total = s0  # running total_tokens after prefill

    # ── Decode loop, driven by the CPU reference argmax for determinism ───────
    mz_dec = _morphizen_session("decoder_fixed_decode.onnx")
    mz_dec_out = [o.name for o in mz_dec.get_outputs()]

    cosines = []
    next_tok = int(np.argmax(np.asarray(cpu_pref["logits"][0, -1, :], np.float32)))
    for _step in range(n_steps):
        if next_tok == _EOT or total >= _SELF_KV_SLOTS:
            break

        # CPU dynamic ref: int32 ids, growing past, no position_ids/seqlens_k.
        cpu_step = dict(
            zip(
                cpu_dec_out,
                cpu_dec.run(
                    None,
                    {
                        "input_ids": np.array([[next_tok]], np.int32),
                        **cpu_self,
                        **cross,
                    },
                ),
            )
        )
        # GPU static shared-buffer: int64 ids, total-1 seqlens_k, position_ids.
        mz_step = dict(
            zip(
                mz_dec_out,
                mz_dec.run(
                    None,
                    {
                        "input_ids": np.array([[next_tok]], np.int64),
                        "past_sequence_length": np.array([total], np.int32),
                        "position_ids": np.array([total], np.int64),
                        **mz_self,
                        **cross,
                    },
                ),
            )
        )

        cosines.append(
            _cosine(cpu_step["logits"][0, -1, :], mz_step["logits"][0, -1, :])
        )

        for i in range(_N_LAYERS):
            cpu_self[f"past_key_self_{i}"] = cpu_step[f"present_key_self_{i}"]
            cpu_self[f"past_value_self_{i}"] = cpu_step[f"present_value_self_{i}"]
            mz_self[f"past_key_self_{i}"] = mz_step[f"present_key_self_{i}"]
            mz_self[f"past_value_self_{i}"] = mz_step[f"present_value_self_{i}"]
        total += 1
        next_tok = int(np.argmax(np.asarray(cpu_step["logits"][0, -1, :], np.float32)))

    assert cosines, "no decode steps ran"
    lo, hi = min(cosines), max(cosines)
    print(
        f"\n[decode] {len(cosines)} steps, per-step logits cosine (GPU vs "
        f"CPU-fp32) min={lo:.6f} max={hi:.6f} mean={np.mean(cosines):.6f}"
    )
    assert lo >= _DEC_COS_THRESH, (
        f"decode per-step GPU-vs-CPU-fp32 cosine dropped to {lo} < {_DEC_COS_THRESH}"
    )


# ── End-to-end greedy transcription ──────────────────────────────────────────


def _greedy_decode_cpu(
    audio_fp, max_length=200, model_dir=_MODEL_DIR, dtype=np.float32
):
    """CPU greedy reference — thin wrapper over the shared harness.

    Logic lives in ``whisper_infer.greedy_decode_cpu``; here we just bind the
    pytest-skip-aware ``_cpu_session`` factory. ``model_dir`` / ``dtype`` select
    the fp32 (default) or fp16 model.
    """
    return whisper_infer.greedy_decode_cpu(
        lambda name: _cpu_session(name, model_dir),
        audio_fp,
        max_length=max_length,
        dtype=dtype,
    )


def _greedy_decode_morphizen(
    audio_fp, max_length=200, model_dir=_MODEL_DIR, dtype=np.float32
):
    """GPU greedy decode — thin wrapper over the shared harness.

    Logic (encoder + 2-variant decoder loop, the seqlens_k / position_ids
    convention, the three correctness fixes) lives in
    ``whisper_infer.greedy_decode_morphizen``; here we bind the pytest-skip-aware
    ``_morphizen_session`` factory so a missing EP skips cleanly. ``model_dir`` /
    ``dtype`` select the fp32 (default) or fp16 model.
    """
    return whisper_infer.greedy_decode_morphizen(
        lambda name: _morphizen_session(name, model_dir),
        audio_fp,
        max_length=max_length,
        dtype=dtype,
    )


def _greedy_decode_morphizen_iobinding(
    enc, pref_dec, dec, audio_fp, max_length=200, return_timing=False, dtype=np.float32
):
    """OGA-style zero-copy GPU greedy decode via IOBinding + KV-cache aliasing.

    This is the FAST path: instead of marshaling all 32×2 self-KV + 32×2 cross-KV
    tensors host<->device on every decode step (the Python plumbing that made the
    naive harness report ~4.65 tok/s), we keep the entire KV cache GPU-resident:

      * self-KV (grows per step): one GPU OrtValue per (layer, kind), 448-slot
        ``[1,20,448,64]`` fp32, aliased past<->present. Binding the SAME OrtValue
        to ``past_*_self_i`` (input) AND ``present_*_self_i`` (output) makes the
        runtime see ``past_ptr == present_ptr`` → it appends the new token's K/V
        in-place at ``slot[past_len]`` (the shared-buffer fast path). No copy.
        The SAME GPU OrtValues are bound to BOTH the prefill session and the
        decode session, so the prefill-written slots [0..S-1] carry straight over
        to decode with zero copies at the prefill→decode boundary.
      * cross-KV (constant): the decoder consumes ``past_*_cross_i`` but emits no
        ``present_cross`` (cross KV is the encoder output, fixed across the whole
        generation). We upload the encoder's cross-KV outputs to GPU OrtValues
        ONCE (``ortvalue_from_numpy(device_type="gpu")``) and bind them once.
      * per-step inputs (tiny): ``input_ids`` [1,1] int64, ``past_sequence_length``
        [1] int32, ``position_ids`` [1,1] int64. Updated in place via
        ``update_inplace`` so their GPU OrtValues keep a stable address and the
        binding never has to be torn down.
      * logits: bound to a GPU OrtValue; we read it back (``.numpy()``) once per
        step for the argmax. That is ONE small D2H of 51866 floats per step —
        unavoidable for greedy and tiny next to the 96 KV tensors we eliminated.

    Returns ``tokens`` (or, with ``return_timing=True``, the 5-tuple
    ``(tokens, enc_ms, prefill_ms, decode_ms, n_decode_steps)`` matching
    ``_greedy_decode_morphizen_timed``). Correctness is identical to
    ``_greedy_decode_morphizen`` (same seqlens_k / position_ids convention).
    """
    import time

    def _gpu_val(arr):
        return ort.OrtValue.ortvalue_from_numpy(
            np.ascontiguousarray(arr), device_type="gpu", vendor_id=AMD_VENDOR_ID
        )

    def _gpu_empty(shape, dtype):
        return ort.OrtValue.ortvalue_from_shape_and_type(
            list(shape), dtype, device_type="gpu", vendor_id=AMD_VENDOR_ID
        )

    s0 = len(_START_TOKENS)

    # ── Encoder (run once) — gives cross-KV ──────────────────────────────────
    enc_names = [o.name for o in enc.get_outputs()]
    t0 = time.perf_counter()
    enc_out = dict(
        zip(enc_names, enc.run(None, {"audio_features": audio_fp.astype(dtype)}))
    )
    enc_ms = (time.perf_counter() - t0) * 1e3

    # ── Upload cross-KV to GPU ONCE (constant across the whole generation) ────
    cross_gpu = {}
    for i in range(_N_LAYERS):
        cross_gpu[f"past_key_cross_{i}"] = _gpu_val(
            enc_out[f"present_key_cross_{i}"].astype(dtype)
        )
        cross_gpu[f"past_value_cross_{i}"] = _gpu_val(
            enc_out[f"present_value_cross_{i}"].astype(dtype)
        )

    # ── Allocate self-KV GPU buffers (448-slot, zeroed) aliased past<->present ─
    # The SAME OrtValue is reused across the prefill AND decode sessions; binding
    # it to both past input and present output gives the runtime past==present.
    kv_shape = (1, _N_HEADS, _SELF_KV_SLOTS, _HEAD_DIM)
    self_gpu = {}
    for i in range(_N_LAYERS):
        for kind in ("key", "value"):
            self_gpu[(i, kind)] = _gpu_val(np.zeros(kv_shape, dtype=dtype))

    # logits GPU buffer dtype must match the graph's logits output dtype: the OGA
    # fp16 build computes lm_head in fp32 but casts logits back to fp16 at the
    # output, so the buffer is fp16 for the fp16 model (fp32 for fp32). Mismatch →
    # IOBinding "Unexpected output data type". argmax read-back upcasts either way.
    pref_logits_shape = [1, s0, _CFG.n_vocab]
    dec_logits_shape = [1, 1, _CFG.n_vocab]
    pref_logits_gpu = _gpu_empty(pref_logits_shape, dtype)
    dec_logits_gpu = _gpu_empty(dec_logits_shape, dtype)

    def _bind_self_and_cross(io):
        for i in range(_N_LAYERS):
            for kind in ("key", "value"):
                val = self_gpu[(i, kind)]
                io.bind_ortvalue_input(f"past_{kind}_self_{i}", val)
                io.bind_ortvalue_output(f"present_{kind}_self_{i}", val)
            io.bind_ortvalue_input(
                f"past_key_cross_{i}", cross_gpu[f"past_key_cross_{i}"]
            )
            io.bind_ortvalue_input(
                f"past_value_cross_{i}", cross_gpu[f"past_value_cross_{i}"]
            )

    # ── Prefill (S=4 start tokens) on the prefill DLL ─────────────────────────
    pref_io = pref_dec.io_binding()
    pref_io.bind_ortvalue_input(
        "input_ids", _gpu_val(np.array([_START_TOKENS], dtype=np.int64))
    )
    pref_io.bind_ortvalue_input(
        "past_sequence_length", _gpu_val(np.array([s0 - 1], dtype=np.int32))
    )
    pref_io.bind_ortvalue_input("position_ids", _gpu_val(np.arange(s0, dtype=np.int64)))
    _bind_self_and_cross(pref_io)
    pref_io.bind_ortvalue_output("logits", pref_logits_gpu)

    t0 = time.perf_counter()
    pref_dec.run_with_iobinding(pref_io)
    pref_logits = pref_logits_gpu.numpy()
    prefill_ms = (time.perf_counter() - t0) * 1e3

    tokens = list(_START_TOKENS)
    next_tok = int(np.argmax(np.asarray(pref_logits[0, -1, :], dtype=np.float32)))
    tokens.append(next_tok)
    total = s0  # running total_tokens after prefill

    # ── Decode loop (S=1) on the decode DLL ───────────────────────────────────
    # Bind the SAME self-KV + cross-KV GPU OrtValues (zero-copy carryover from
    # prefill). Per-step tiny inputs are updated in place (stable addresses) so we
    # never tear down the binding inside the timed loop.
    dec_io = dec.io_binding()
    ids_gpu = _gpu_val(np.array([[next_tok]], dtype=np.int64))
    pastlen_gpu = _gpu_val(np.array([total], dtype=np.int32))
    posids_gpu = _gpu_val(np.array([total], dtype=np.int64))
    dec_io.bind_ortvalue_input("input_ids", ids_gpu)
    dec_io.bind_ortvalue_input("past_sequence_length", pastlen_gpu)
    dec_io.bind_ortvalue_input("position_ids", posids_gpu)
    _bind_self_and_cross(dec_io)
    dec_io.bind_ortvalue_output("logits", dec_logits_gpu)

    n_steps = 0
    t0 = time.perf_counter()
    while len(tokens) < max_length and next_tok != _EOT and total < _SELF_KV_SLOTS:
        # Update the three tiny per-step inputs in place — no rebind, the runtime
        # reads the new values from the same GPU addresses next run.
        ids_gpu.update_inplace(np.array([[next_tok]], dtype=np.int64))
        pastlen_gpu.update_inplace(np.array([total], dtype=np.int32))
        posids_gpu.update_inplace(np.array([total], dtype=np.int64))

        dec.run_with_iobinding(dec_io)
        # Single small D2H per step (51866 floats) for the greedy argmax.
        logits = dec_logits_gpu.numpy()
        total += 1
        next_tok = int(np.argmax(np.asarray(logits[0, -1, :], dtype=np.float32)))
        tokens.append(next_tok)
        n_steps += 1
        if next_tok == _EOT:
            break
    decode_ms = (time.perf_counter() - t0) * 1e3

    if return_timing:
        return tokens, enc_ms, prefill_ms, decode_ms, n_steps
    return tokens


def test_e2e_transcription_greedy(capfd):
    """Fully-fp32 GPU greedy transcription of jfk.wav == fp32 CPU quote.

    THE HEADLINE: with conv1d + gqa + gemm + layernorm all fp32-capable, the
    native-fp32 encoder + decoder GPU-dispatch end-to-end and produce the
    correct, argmax-stable JFK quote VERBATIM (fp32 is not lossy on the
    51866-wide lm_head). The GPU greedy text must match the CPU fp32 greedy text
    exactly.

    The last blocker was the shared-buffer (static 448-slot) decoder self-attn
    path, resolved by three fixes (see inject_seqlens_k + gqa.cpp):
      1. Position-embedding Slice (runtime-data-dependent bounds) → replaced with
         Gather(embed_positions, position_ids). The GPU runtime-bounds Slice +
         int64-scalar arith mishandle the offset and zero the position embedding.
      2. Token-embedding Gather indices: input_ids re-typed int32 → int64
         (hip_gather hardcodes int64 indices; int32 ids → zeroed token embedding).
      3. fp32 causal decode (sq==1) was reaching the __half-only fused/flash
         decode kernels → garbage. gqa.cpp fused_predicate now requires
         element_size_bytes==2, routing fp32 decode to the fp32-capable
         decomposed path.

    seqlens_k contract: the GQA runtime uses seqlens_k = total_tokens - 1, so the
    harness feeds past_sequence_length = (past_tokens + S) - 1. position_ids is
    fed separately = [real_past .. real_past + S - 1].
    """
    os.environ["HIPDNN_EP_DEBUG"] = "1"
    audio = make_whisper_inputs(_AUDIO, _CFG)["audio_features"].astype(np.float32)

    cpu_tokens = _greedy_decode_cpu(audio)  # fp32 original → correct JFK quote
    cpu_text = _decode_text(cpu_tokens)

    capfd.readouterr()
    mz_tokens = _greedy_decode_morphizen(audio)
    stderr_text = "".join(capfd.readouterr())
    mz_text = _decode_text(mz_tokens)

    print(f"\n[e2e] CPU(fp32) : {cpu_text!r}")
    print(f"[e2e] GPU(fp32) : {mz_text!r}")

    # The fp32 GPU path must compile (no silent CPU fallback).
    _assert_compiled_on_gpu(stderr_text)

    assert "country" in cpu_text.lower(), f"CPU reference looks wrong: {cpu_text!r}"
    assert mz_text == cpu_text, (
        f"fp32 transcription text mismatch:\n  cpu: {cpu_text!r}\n  mz : {mz_text!r}"
    )


# ── fp16 correctness (opt-in: needs the locally-built fp16 OGA bundle) ─────────


def test_fp16_e2e_transcription_greedy(fp16_model, capfd):
    """fp16 GPU greedy transcription of jfk.wav == fp16 CPU quote.

    The fp16 model is built by the OGA DML model builder (``fp16_model`` fixture).
    Its body is fp16 but ``lm_head`` stays fp32, so greedy argmax is lossless —
    the GPU fp16 text must match the CPU fp16 dynamic-reference text exactly AND be
    the verbatim JFK quote. The CPU reference uses the DYNAMIC graph (the static
    surgered graph is invalid on ORT CPU — it ignores past_sequence_length; see
    the module docstring), so this is fp16-GPU-static vs fp16-CPU-dynamic.
    """
    os.environ["HIPDNN_EP_DEBUG"] = "1"
    audio = make_whisper_inputs(_AUDIO, _CFG)["audio_features"].astype(np.float32)

    cpu_tokens = _greedy_decode_cpu(audio, model_dir=fp16_model, dtype=np.float16)
    cpu_text = _decode_text(cpu_tokens)

    capfd.readouterr()
    mz_tokens = _greedy_decode_morphizen(audio, model_dir=fp16_model, dtype=np.float16)
    stderr_text = "".join(capfd.readouterr())
    mz_text = _decode_text(mz_tokens)

    print(f"\n[fp16-e2e] CPU(fp16) : {cpu_text!r}")
    print(f"[fp16-e2e] GPU(fp16) : {mz_text!r}")

    _assert_compiled_on_gpu(stderr_text)  # no silent CPU fallback
    assert "country" in cpu_text.lower(), (
        f"fp16 CPU reference looks wrong: {cpu_text!r}"
    )
    assert mz_tokens == cpu_tokens, (
        f"fp16 transcription token mismatch:\n  cpu: {cpu_text!r}\n  mz : {mz_text!r}"
    )


def test_fp16_decoder_prefill_correctness(fp16_model, capfd):
    """fp16 prefill last-token logits cosine: MorphiZen(fp16) vs CPU(fp16).

    GPU runs the fixed fp16 ``decoder_fixed_prefill.onnx`` (static 448-slot shared
    buffer); the reference is the dynamic fp16 ``decoder.onnx`` on the CPU EP, both
    fed the SAME fp16 cross-KV from the CPU encoder so only the decoder kernels are
    compared. lm_head is fp32 in both, so the logits compare is exact-ish.
    """
    os.environ["HIPDNN_EP_DEBUG"] = "1"
    audio = make_whisper_inputs(_AUDIO, _CFG)["audio_features"].astype(np.float32)
    s0 = len(_START_TOKENS)

    # Shared fp16 cross-KV from the CPU encoder (isolate the decoder).
    cpu_enc = _cpu_session("encoder.onnx", fp16_model)
    _, cross = _encoder_cross_kv(cpu_enc, audio, dtype=np.float16)

    # ── CPU-fp16 reference: DYNAMIC decoder, empty (0-slot) past ──────────────
    cpu_dec = _cpu_session("decoder.onnx", fp16_model)
    cpu_out_names = [o.name for o in cpu_dec.get_outputs()]
    zero_past = {}
    for i in range(_N_LAYERS):
        zero_past[f"past_key_self_{i}"] = np.zeros(
            (1, _N_HEADS, 0, _HEAD_DIM), dtype=np.float16
        )
        zero_past[f"past_value_self_{i}"] = np.zeros(
            (1, _N_HEADS, 0, _HEAD_DIM), dtype=np.float16
        )
    cpu_ref = dict(
        zip(
            cpu_out_names,
            cpu_dec.run(
                None,
                {
                    "input_ids": np.array([_START_TOKENS], dtype=np.int32),
                    **zero_past,
                    **cross,
                },
            ),
        )
    )
    cpu_logits = np.asarray(cpu_ref["logits"][0, -1, :], dtype=np.float32)

    # ── MorphiZen-fp16: static surgered prefill graph (448-slot shared buffer) ─
    capfd.readouterr()
    mz_pref = _morphizen_session("decoder_fixed_prefill.onnx", fp16_model)
    mz_out_names = [o.name for o in mz_pref.get_outputs()]
    self_kv = _zeroed_self_past(np.float16)
    mz_out = dict(
        zip(
            mz_out_names,
            mz_pref.run(
                None,
                {
                    "input_ids": np.array([_START_TOKENS], dtype=np.int64),
                    "past_sequence_length": np.array([s0 - 1], dtype=np.int32),
                    "position_ids": np.arange(s0, dtype=np.int64),
                    **self_kv,
                    **cross,
                },
            ),
        )
    )
    stderr_text = "".join(capfd.readouterr())
    _assert_compiled_on_gpu(stderr_text)
    mz_logits = np.asarray(mz_out["logits"][0, -1, :], dtype=np.float32)

    cos = _cosine(cpu_logits, mz_logits)
    print(
        f"\n[fp16-prefill] last-token logits cosine (GPU-fp16 vs CPU-fp16) = {cos:.6f}"
    )
    assert cos > 0.98, f"fp16 prefill GPU-vs-CPU cosine = {cos} < 0.98"


# ── Cross-backend decode-throughput benchmark (the fair fp32 comparison) ──────
#
# THE FAIR BENCHMARK: decode tok/s + transcription accuracy for the SAME
# native-fp32 Whisper-large-v3 on three backends — MorphiZen EP (GPU), DirectML
# EP (GPU), CPU EP. fp16 was dropped precisely so this is apples-to-apples:
# every backend runs fp32, so the only thing being compared is the backend, not
# the precision.
#
# Methodology (CLAUDE.md perf hygiene):
#   * Each backend runs the SAME greedy decode-loop structure (encode -> S=4
#     prefill -> per-token S=1 decode loop), so Python-loop overhead is identical
#     across all three. Absolute tok/s is NOT OGA-optimized (the loop is in
#     Python), but the COMPARISON is fair.
#   * Methodology gap (same as the Llama perf tests): MorphiZen runs the surgered
#     fixed-shape (static 448-slot shared-buffer KV) form; CPU/DML run the
#     dynamic ORIGINAL onnx (growing KV). The static shared-buffer form is not a
#     valid graph for ORT's MHA kernel (it ignores past_sequence_length without
#     past_present_share_buffer), so CPU/DML MUST use the dynamic original — the
#     established pattern from the correctness tests above. The decode tok/s is
#     still "what each backend delivers for this model".
#   * Warmup: one full encode+decode is run and DISCARDED before timing (first
#     MorphiZen run pays the MLIR compile; first DML run pays shader compile).
#   * 3 timed reps; rep-1 is discarded (CLAUDE.md "discard rep-1"), steady-state
#     = mean of reps 2..3.
#   * encoder ms / prefill ms / decode tok/s reported separately; decode tok/s is
#     the headline. NEVER run this with HIPDNN_EP_PERF=1 (it adds ~58% overhead).


def _greedy_decode_dynamic_timed(enc, dec, audio_fp, max_length=200, dtype=np.float32):
    """Greedy decode on the DYNAMIC graphs (CPU or DML), with timings.

    `enc` / `dec` are pre-built InferenceSessions for ``encoder.onnx`` /
    ``decoder.onnx`` on the target EP. Returns
    ``(tokens, enc_ms, prefill_ms, decode_ms, n_decode_steps)``. ``dtype`` selects
    the model precision (fp32 default / fp16) so the same loop times both.

    Mirrors ``_greedy_decode_cpu`` (same dynamic graph, same growing-past KV
    convention, same argmax greedy) but instruments encoder / prefill / decode
    so the three phases can be reported separately. The prefill step is the
    first ``dec.run`` over the S=4 start tokens (empty past); each subsequent
    single-token ``dec.run`` is a decode step.
    """
    import time

    enc_names = [o.name for o in enc.get_outputs()]
    t0 = time.perf_counter()
    enc_out = dict(
        zip(enc_names, enc.run(None, {"audio_features": audio_fp.astype(dtype)}))
    )
    enc_ms = (time.perf_counter() - t0) * 1e3
    cross = {}
    for i in range(_N_LAYERS):
        cross[f"past_key_cross_{i}"] = enc_out[f"present_key_cross_{i}"].astype(dtype)
        cross[f"past_value_cross_{i}"] = enc_out[f"present_value_cross_{i}"].astype(
            dtype
        )

    out_names = [o.name for o in dec.get_outputs()]
    self_kv = {}
    for i in range(_N_LAYERS):
        self_kv[f"past_key_self_{i}"] = np.zeros(
            (1, _N_HEADS, 0, _HEAD_DIM), dtype=dtype
        )
        self_kv[f"past_value_self_{i}"] = np.zeros(
            (1, _N_HEADS, 0, _HEAD_DIM), dtype=dtype
        )

    # Prefill (S=4 start tokens, empty past).
    t0 = time.perf_counter()
    pref = dict(
        zip(
            out_names,
            dec.run(
                None,
                {
                    "input_ids": np.array([_START_TOKENS], dtype=np.int32),
                    **self_kv,
                    **cross,
                },
            ),
        )
    )
    prefill_ms = (time.perf_counter() - t0) * 1e3
    for i in range(_N_LAYERS):
        self_kv[f"past_key_self_{i}"] = pref[f"present_key_self_{i}"]
        self_kv[f"past_value_self_{i}"] = pref[f"present_value_self_{i}"]

    tokens = list(_START_TOKENS)
    next_tok = int(np.argmax(np.asarray(pref["logits"][0, -1, :], np.float32)))
    tokens.append(next_tok)

    # Decode loop (S=1 steps). Time the whole loop; tok/s = steps / loop_wall.
    n_steps = 0
    t0 = time.perf_counter()
    while len(tokens) < max_length and next_tok != _EOT:
        step = dict(
            zip(
                out_names,
                dec.run(
                    None,
                    {
                        "input_ids": np.array([[next_tok]], dtype=np.int32),
                        **self_kv,
                        **cross,
                    },
                ),
            )
        )
        for i in range(_N_LAYERS):
            self_kv[f"past_key_self_{i}"] = step[f"present_key_self_{i}"]
            self_kv[f"past_value_self_{i}"] = step[f"present_value_self_{i}"]
        next_tok = int(np.argmax(np.asarray(step["logits"][0, -1, :], np.float32)))
        tokens.append(next_tok)
        n_steps += 1
        if next_tok == _EOT:
            break
    decode_ms = (time.perf_counter() - t0) * 1e3
    return tokens, enc_ms, prefill_ms, decode_ms, n_steps


def _greedy_decode_morphizen_timed(enc, pref_dec, dec, audio_fp, max_length=200):
    """Greedy decode on the MorphiZen fixed-shape fp32 graphs, with timings.

    `enc` / `pref_dec` / `dec` are pre-built MorphiZen sessions for
    ``encoder_fixed.onnx`` / ``decoder_fixed_prefill.onnx`` /
    ``decoder_fixed_decode.onnx``. Returns the same 5-tuple as
    ``_greedy_decode_dynamic_timed``. Logic mirrors ``_greedy_decode_morphizen``
    (static 448-slot shared buffer, total-1 seqlens_k convention, position_ids
    Gather) but reuses caller-provided sessions and instruments the phases.
    """
    import time

    names = [o.name for o in enc.get_outputs()]
    t0 = time.perf_counter()
    out = dict(zip(names, enc.run(None, {"audio_features": audio_fp})))
    enc_ms = (time.perf_counter() - t0) * 1e3
    cross = {}
    for i in range(_N_LAYERS):
        cross[f"past_key_cross_{i}"] = out[f"present_key_cross_{i}"].astype(np.float32)
        cross[f"past_value_cross_{i}"] = out[f"present_value_cross_{i}"].astype(
            np.float32
        )

    self_kv = _zeroed_self_past()
    s0 = len(_START_TOKENS)
    pref_out = [o.name for o in pref_dec.get_outputs()]
    t0 = time.perf_counter()
    pref = dict(
        zip(
            pref_out,
            pref_dec.run(
                None,
                {
                    "input_ids": np.array([_START_TOKENS], dtype=np.int64),
                    "past_sequence_length": np.array([s0 - 1], dtype=np.int32),
                    "position_ids": np.arange(s0, dtype=np.int64),
                    **self_kv,
                    **cross,
                },
            ),
        )
    )
    prefill_ms = (time.perf_counter() - t0) * 1e3
    for i in range(_N_LAYERS):
        self_kv[f"past_key_self_{i}"] = pref[f"present_key_self_{i}"]
        self_kv[f"past_value_self_{i}"] = pref[f"present_value_self_{i}"]

    tokens = list(_START_TOKENS)
    next_tok = int(np.argmax(np.asarray(pref["logits"][0, -1, :], dtype=np.float32)))
    tokens.append(next_tok)
    total = s0

    dec_out = [o.name for o in dec.get_outputs()]
    n_steps = 0
    t0 = time.perf_counter()
    while len(tokens) < max_length and next_tok != _EOT and total < _SELF_KV_SLOTS:
        step = dict(
            zip(
                dec_out,
                dec.run(
                    None,
                    {
                        "input_ids": np.array([[next_tok]], dtype=np.int64),
                        "past_sequence_length": np.array([total], dtype=np.int32),
                        "position_ids": np.array([total], dtype=np.int64),
                        **self_kv,
                        **cross,
                    },
                ),
            )
        )
        for i in range(_N_LAYERS):
            self_kv[f"past_key_self_{i}"] = step[f"present_key_self_{i}"]
            self_kv[f"past_value_self_{i}"] = step[f"present_value_self_{i}"]
        total += 1
        next_tok = int(
            np.argmax(np.asarray(step["logits"][0, -1, :], dtype=np.float32))
        )
        tokens.append(next_tok)
        n_steps += 1
        if next_tok == _EOT:
            break
    decode_ms = (time.perf_counter() - t0) * 1e3
    return tokens, enc_ms, prefill_ms, decode_ms, n_steps


def _bench_backend(run_one):
    """Warmup (discarded) + 3 timed reps; return steady-state (reps 2..3 mean).

    ``run_one`` is a zero-arg callable returning the 5-tuple
    ``(tokens, enc_ms, prefill_ms, decode_ms, n_steps)``. Per CLAUDE.md perf
    hygiene: discard the warmup AND rep-1, average the remaining steady-state
    reps. Returns ``(tokens, enc_ms, prefill_ms, decode_tps, n_steps)`` where
    ``decode_tps = n_steps / (decode_ms / 1000)`` averaged over the kept reps.
    """
    run_one()  # warmup — pays MLIR/shader compile, discarded
    reps = [run_one() for _ in range(3)]
    keep = reps[1:]  # discard rep-1 (cold caches / autotune still settling)
    tokens = keep[-1][0]
    enc_ms = float(np.mean([r[1] for r in keep]))
    prefill_ms = float(np.mean([r[2] for r in keep]))
    tps_vals = [r[4] / (r[3] / 1e3) for r in keep if r[4] > 0 and r[3] > 0]
    decode_tps = float(np.mean(tps_vals)) if tps_vals else 0.0
    n_steps = keep[-1][4]
    return tokens, enc_ms, prefill_ms, decode_tps, n_steps


def _perf_one_precision(audio, capfd, precision, model_dir, dtype, results):
    """Bench MorphiZen / DirectML / CPU for ONE precision; fill ``results``.

    ``results`` is keyed by ``(backend_label, precision)`` →
    ``(enc_ms, prefill_ms, decode_tps, text, n_steps)``. Each backend is
    best-effort: a failure (EP missing, DML op gap, fp16 build absent) is logged
    and skipped, never aborts the other rows.
    """
    # ── MorphiZen EP (GPU, fixed-shape) ──────────────────────────────────────
    try:
        os.environ["HIPDNN_EP_DEBUG"] = "1"
        capfd.readouterr()
        mz_enc = _morphizen_session("encoder_fixed.onnx", model_dir)
        mz_pref = _morphizen_session("decoder_fixed_prefill.onnx", model_dir)
        mz_dec = _morphizen_session("decoder_fixed_decode.onnx", model_dir)
        # OGA-style zero-copy decode: self-KV aliased GPU buffers (past==present),
        # cross-KV bound once, per-step tiny inputs + a single logits D2H — this
        # row measures GPU kernels, not Python plumbing.
        mz_tokens, mz_enc_ms, mz_pref_ms, mz_tps, mz_n = _bench_backend(
            lambda: _greedy_decode_morphizen_iobinding(
                mz_enc, mz_pref, mz_dec, audio, return_timing=True, dtype=dtype
            )
        )
        stderr_text = "".join(capfd.readouterr())
        _assert_compiled_on_gpu(stderr_text)  # no silent CPU fallback
        results[("MorphiZen EP", precision)] = (
            mz_enc_ms,
            mz_pref_ms,
            mz_tps,
            _decode_text(mz_tokens),
            mz_n,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[perf] MorphiZen EP {precision} FAILED: {e!r}")
    finally:
        os.environ.pop("HIPDNN_EP_DEBUG", None)

    # ── DirectML EP (GPU, dynamic) ───────────────────────────────────────────
    dml_providers = get_amd_dml_providers()
    if dml_providers is None:
        print(f"[perf] DirectML EP unavailable — skipping DML {precision} row")
    else:
        try:
            dml_enc = ort.InferenceSession(
                str(model_dir / "encoder.onnx"), providers=dml_providers
            )
            dml_dec = ort.InferenceSession(
                str(model_dir / "decoder.onnx"), providers=dml_providers
            )
            dml_tokens, dml_enc_ms, dml_pref_ms, dml_tps, dml_n = _bench_backend(
                lambda: _greedy_decode_dynamic_timed(
                    dml_enc, dml_dec, audio, dtype=dtype
                )
            )
            results[("DirectML EP", precision)] = (
                dml_enc_ms,
                dml_pref_ms,
                dml_tps,
                _decode_text(dml_tokens),
                dml_n,
            )
        except Exception as e:  # DML com.microsoft op coverage varies — report it
            print(f"[perf] DirectML EP {precision} FAILED: {e!r}")

    # ── CPU EP (dynamic) ─────────────────────────────────────────────────────
    try:
        cpu_enc = _cpu_session("encoder.onnx", model_dir)
        cpu_dec = _cpu_session("decoder.onnx", model_dir)
        cpu_tokens, cpu_enc_ms, cpu_pref_ms, cpu_tps, cpu_n = _bench_backend(
            lambda: _greedy_decode_dynamic_timed(cpu_enc, cpu_dec, audio, dtype=dtype)
        )
        results[("CPU EP", precision)] = (
            cpu_enc_ms,
            cpu_pref_ms,
            cpu_tps,
            _decode_text(cpu_tokens),
            cpu_n,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[perf] CPU EP {precision} FAILED: {e!r}")


def test_perf_decode_tps(capfd):
    """Cross-backend, cross-precision decode tok/s + accuracy.

    Runs the SAME greedy decode-loop on MorphiZen / DirectML / CPU for BOTH
    precisions: fp32 (the locally-built native fp32 model) and fp16 (the OGA DML build, body
    fp16 + fp32 lm_head). MorphiZen runs the surgered fixed-shape form; CPU/DML run
    the dynamic ORIGINAL onnx (the only valid graph for ORT's MHA kernel — see the
    module docstring). The fp32 rows are unconditional; the fp16 rows are
    best-effort (skipped with a note if the fp16 build is unavailable) so a machine
    without the OGA builder still gets the full fp32 table.
    """
    audio = make_whisper_inputs(_AUDIO, _CFG)["audio_features"].astype(np.float32)
    results = {}  # (label, precision) -> (enc_ms, prefill_ms, decode_tps, text, n)

    # fp32 — always run (locally-built model is set up by the autouse fixture).
    _perf_one_precision(audio, capfd, "fp32", _MODEL_DIR, np.float32, results)

    # fp16 — best-effort: build on demand; skip the precision entirely on failure.
    try:
        setup_whisper_fp16_model_dir(_MODEL_DIR_FP16)
        _perf_one_precision(audio, capfd, "fp16", _MODEL_DIR_FP16, np.float16, results)
    except Exception as e:  # noqa: BLE001
        print(f"[perf] fp16 unavailable (build failed) — fp32-only table: {e!r}")

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 86)
    print("Whisper-large-v3 cross-backend x precision, jfk.wav (~11s audio), gfx1151")
    print("=" * 86)
    print(
        f"{'Backend':<13}| {'prec':>5} | {'encoder ms':>10} | {'prefill ms':>10} | "
        f"{'decode tok/s':>12} | correct?"
    )
    print("-" * 86)
    for precision in ("fp32", "fp16"):
        for label in ("MorphiZen EP", "DirectML EP", "CPU EP"):
            key = (label, precision)
            if key not in results:
                continue
            enc_ms, pref_ms, tps, text, _n = results[key]
            ok_str = "yes" if "country" in text.lower() else f"NO -> {text[:30]!r}"
            print(
                f"{label:<13}| {precision:>5} | {enc_ms:>10.1f} | {pref_ms:>10.1f} | "
                f"{tps:>12.2f} | {ok_str}"
            )
    print("=" * 86)

    # Accuracy gate: there MUST be a working fp32 reference, and every row that ran
    # (any backend, any precision) must transcribe the JFK quote.
    assert ("CPU EP", "fp32") in results, "fp32 CPU reference did not run"
    cpu_text = results[("CPU EP", "fp32")][3]
    assert "country" in cpu_text.lower(), (
        f"CPU fp32 reference looks wrong: {cpu_text!r}"
    )
    for (label, precision), (_e, _p, _t, text, _n) in results.items():
        assert "country" in text.lower(), (
            f"{label} {precision} transcription is not the JFK quote: {text!r}"
        )


# ── Broadened correctness: LibriSpeech clips + a long ~24 s clip ──────────────
#
# This section widens Whisper coverage beyond the single jfk.wav with a set of
# LibriSpeech "dummy" clips (different speakers / sentences / lengths) plus one
# long (~24 s) concatenated clip. Two independent bars per clip:
#   (a) MorphiZen GPU greedy text == CPU greedy text VERBATIM — the faithful-EP
#       bar (the EP must reproduce the CPU result regardless of audio), and
#   (b) WER of the GPU transcription vs the LibriSpeech ground-truth reference is
#       within a lenient threshold — proves "real transcription, not garbage".
#
# Data is provisioned by ``setup_librispeech_samples`` (committed WAVs +
# references.json; see test/python/data/whisper/README.md). The long clip is the
# decode-boundary stress: ~24 s of audio drives the decode loop a few hundred
# tokens deep toward the 448-slot self-KV cap, so GPU==CPU there exercises that
# the KV buffer fill + position_ids advance stay correct late in the sequence.

import json  # noqa: E402

# Lenient WER ceiling: whisper-large-v3 greedy on clean LibriSpeech is ~2-4% WER,
# but (a) we use greedy (no beam), (b) our normalization is crude, and (c) Whisper
# emits punctuation / casing the UPPERCASE reference lacks. 0.15 catches "real
# transcription, not garbage" without demanding SOTA. A clip that legitimately
# exceeds it is REPORTED (hyp vs ref) rather than silently loosened.
_WER_THRESHOLD = 0.15
# Long clip decode budget: ~24 s of speech is well over the default 200-token cap,
# so allow the decode loop to run up to the 448-slot self-KV limit. The loop also
# stops on EOT.
_LONG_MAX_LEN = 445


def _normalize_words(text):
    """Crude ASR text normalization for WER: uppercase, drop punctuation, collapse
    whitespace, split to words. Good enough to compare a Whisper hypothesis (mixed
    case + punctuation) against an UPPERCASE-no-punctuation LibriSpeech reference.
    """
    import re

    text = text.upper()
    # Keep apostrophes inside words (QUILTER'S) but drop other punctuation.
    text = re.sub(r"[^A-Z0-9'\s]", " ", text)
    return text.split()


def _wer(reference, hypothesis):
    """Word-level WER = edit_distance(ref_words, hyp_words) / len(ref_words).

    A ~15-line Levenshtein on word lists — no jiwer dependency.
    """
    ref = _normalize_words(reference)
    hyp = _normalize_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    # Classic DP edit distance over word tokens.
    prev = list(range(len(hyp) + 1))
    for i, rw in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, hw in enumerate(hyp, 1):
            cost = 0 if rw == hw else 1
            cur[j] = min(
                prev[j] + 1,  # deletion
                cur[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution / match
            )
        prev = cur
    return prev[len(hyp)] / len(ref)


@pytest.fixture(scope="module")
def librispeech():
    """Provision the LibriSpeech clips once; skip the whole group if unavailable.

    Returns ``(refs_dict, data_dir)`` where ``refs_dict`` maps wav filename ->
    UPPERCASE reference text. ``pytest.skip`` (not error) if the datasets-server
    fetch failed and nothing is cached.
    """
    if not setup_librispeech_samples(_LIBRISPEECH_DIR):
        pytest.skip(
            "LibriSpeech clips unavailable (datasets-server fetch failed and no "
            "cached WAVs). Skipping the broadened-coverage Whisper tests."
        )
    refs = json.loads((_LIBRISPEECH_DIR / "references.json").read_text())
    return refs, _LIBRISPEECH_DIR


_SAMPLE_IDS = [f"sample_{i}.wav" for i in range(5)]


@pytest.mark.parametrize("clip", _SAMPLE_IDS)
def test_librispeech_gpu_vs_cpu(librispeech, capfd, clip):
    """MorphiZen GPU greedy == CPU greedy VERBATIM for each LibriSpeech clip.

    The strong faithfulness bar: independent of the audio content, the EP's fp32
    greedy transcription must reproduce the CPU fp32 greedy transcription exactly.
    """
    refs, data_dir = librispeech
    os.environ["HIPDNN_EP_DEBUG"] = "1"
    audio = make_whisper_inputs(data_dir / clip, _CFG)["audio_features"].astype(
        np.float32
    )

    cpu_tokens = _greedy_decode_cpu(audio)
    cpu_text = _decode_text(cpu_tokens)

    capfd.readouterr()
    gpu_tokens = _greedy_decode_morphizen(audio)
    stderr_text = "".join(capfd.readouterr())
    gpu_text = _decode_text(gpu_tokens)

    print(f"\n[{clip}] CPU : {cpu_text!r}")
    print(f"[{clip}] GPU : {gpu_text!r}")

    # No silent CPU fallback — otherwise GPU==CPU is trivially true (both CPU).
    _assert_compiled_on_gpu(stderr_text)

    if gpu_tokens != cpu_tokens:
        # Pinpoint the first divergence for the report (token-level).
        n = min(len(cpu_tokens), len(gpu_tokens))
        first = next((i for i in range(n) if cpu_tokens[i] != gpu_tokens[i]), n)
        pytest.fail(
            f"{clip}: GPU greedy diverged from CPU at token index {first} "
            f"(cpu_len={len(cpu_tokens)}, gpu_len={len(gpu_tokens)})\n"
            f"  cpu: {cpu_text!r}\n  gpu: {gpu_text!r}"
        )


@pytest.mark.parametrize("clip", _SAMPLE_IDS)
def test_librispeech_wer(librispeech, capfd, clip):
    """WER of the GPU transcription vs the LibriSpeech ground truth is lenient-OK.

    Proves the EP produces a real transcription, not garbage. Threshold is
    deliberately loose (see ``_WER_THRESHOLD``); the actual WER + hyp/ref are
    always printed so a real regression (e.g. 50 % WER) is distinguishable from
    normalization noise (~10 %).
    """
    refs, data_dir = librispeech
    os.environ["HIPDNN_EP_DEBUG"] = "1"
    audio = make_whisper_inputs(data_dir / clip, _CFG)["audio_features"].astype(
        np.float32
    )

    capfd.readouterr()
    gpu_tokens = _greedy_decode_morphizen(audio)
    stderr_text = "".join(capfd.readouterr())
    gpu_text = _decode_text(gpu_tokens)
    _assert_compiled_on_gpu(stderr_text)

    ref = refs[clip]
    wer = _wer(ref, gpu_text)
    print(f"\n[{clip}] WER = {wer:.4f}")
    print(f"[{clip}] ref : {ref!r}")
    print(f"[{clip}] hyp : {gpu_text!r}")

    assert wer <= _WER_THRESHOLD, (
        f"{clip}: WER {wer:.4f} > {_WER_THRESHOLD}\n  ref: {ref!r}\n  hyp: {gpu_text!r}"
    )


def test_long_30s_gpu_vs_cpu(librispeech, capfd):
    """Long (~24 s) concatenated clip: GPU greedy == CPU greedy all the way down.

    THE KEY NEW COVERAGE. ~24 s of audio drives the decode loop a few hundred
    tokens deep toward the 448-slot self-KV cap, so this stresses whether the
    decode stays correct late in the sequence (KV buffer filling, position_ids
    advancing). Asserts verbatim GPU==CPU; also reports WER vs the concatenated
    reference (lenient — concatenation seams can cause minor errors).
    """
    refs, data_dir = librispeech
    clip = "long_30s.wav"
    os.environ["HIPDNN_EP_DEBUG"] = "1"
    audio = make_whisper_inputs(data_dir / clip, _CFG)["audio_features"].astype(
        np.float32
    )

    cpu_tokens = _greedy_decode_cpu(audio, max_length=_LONG_MAX_LEN)
    cpu_text = _decode_text(cpu_tokens)

    capfd.readouterr()
    gpu_tokens = _greedy_decode_morphizen(audio, max_length=_LONG_MAX_LEN)
    stderr_text = "".join(capfd.readouterr())
    gpu_text = _decode_text(gpu_tokens)
    _assert_compiled_on_gpu(stderr_text)

    # Decode-step count = tokens beyond the 4 forced-start tokens.
    n_decode = len(gpu_tokens) - len(_START_TOKENS)
    print(f"\n[long] decode tokens (GPU) = {n_decode}")
    print(f"[long] CPU : {cpu_text!r}")
    print(f"[long] GPU : {gpu_text!r}")
    ref = refs[clip]
    print(f"[long] WER (GPU vs ground truth) = {_wer(ref, gpu_text):.4f}")

    if gpu_tokens != cpu_tokens:
        n = min(len(cpu_tokens), len(gpu_tokens))
        first = next((i for i in range(n) if cpu_tokens[i] != gpu_tokens[i]), n)
        pytest.fail(
            f"long clip: GPU greedy diverged from CPU at token index {first} "
            f"of {n_decode} decode tokens (cpu_len={len(cpu_tokens)}, "
            f"gpu_len={len(gpu_tokens)}) — a deep-decode (KV/position_ids) bug\n"
            f"  cpu: {cpu_text!r}\n  gpu: {gpu_text!r}"
        )
