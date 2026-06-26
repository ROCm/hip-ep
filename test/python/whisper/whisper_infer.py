#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Shared Whisper-large-v3 fp32 greedy-decode harness.

The single source of truth for the encoder + 2-variant decoder greedy loop, the
forced-start tokens, the static 448-slot self-KV layout, and the subtle seqlens_k
/ position_ids convention the surgered decoder requires. Both the pytest suite
(``test_whisper.py``) and the standalone CLI (``scripts/transcribe_whisper.py``)
call into here, so the decode logic — including its three hard-won correctness
fixes — lives in exactly one place.

The two greedy functions take a ``session_factory(model_name) -> InferenceSession``
callable instead of building sessions themselves. This keeps the module free of
any pytest dependency: the test passes a factory that ``pytest.skip``s when the
EP is missing; the CLI passes one that raises ``RuntimeError``. Default factory
builders (``make_cpu_session_factory`` / ``make_morphizen_session_factory``) are
provided for the CLI.

seqlens_k contract (CRITICAL): the GQA runtime uses ``seqlens_k = total_tokens - 1``
for the decoder self-attn, so the harness feeds ``past_sequence_length =
(past_tokens + S) - 1``. ``position_ids`` is fed separately as
``[real_past .. real_past + S - 1]`` (the position-embedding Slice was replaced
with a Gather(embed_positions, position_ids) during surgery to dodge the GPU
runtime-bounds-Slice limitation). ``input_ids`` is int64 (the surgered decoder
re-types it so the token-embed Gather gets int64 indices — the only dtype
hip_gather supports).
"""

import pathlib
import sys

import numpy as np
import onnxruntime as ort

# conftest.py (shared with the llama tests) lives one level up in test/python/;
# add it to the path so this module + the whisper tests can import its helpers.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from conftest import (  # noqa: E402
    AMD_VENDOR_ID,
    EP_PROVIDER_OPTIONS,
    WhisperModelConfig,
    apply_artifact_format,
    make_whisper_inputs,
    register_morphizen_ep,
)

# ── Model config + forced-start tokens ────────────────────────────────────────

CFG = WhisperModelConfig()

# Whisper-large-v3 forced-start tokens + EOT (from the openai tokenizer).
SOT = 50258  # <|startoftranscript|>
LANG_EN = 50259  # <|en|>
TASK_TRANSCRIBE = 50360  # <|transcribe|>
NOTIMESTAMPS = 50364  # <|notimestamps|>
EOT = 50257  # <|endoftext|>
START_TOKENS = [SOT, LANG_EN, TASK_TRANSCRIBE, NOTIMESTAMPS]

N_LAYERS = CFG.n_text_layer  # 32
N_HEADS = CFG.n_text_head  # 20
HEAD_DIM = 64
SELF_KV_SLOTS = CFG.n_text_ctx  # 448 (static buffer for shared-buffer mode)


# ── Session factories (CLI defaults; the test injects its own pytest-aware ones)─


def make_cpu_session_factory(model_dir):
    """Return ``factory(name) -> CPU InferenceSession`` for ``model_dir/name``."""

    def factory(model_name):
        return ort.InferenceSession(
            str(model_dir / model_name), providers=["CPUExecutionProvider"]
        )

    return factory


def make_morphizen_session_factory(repo_root, model_dir):
    """Return ``factory(name) -> AMDGPU-umbrella-EP InferenceSession``.

    Resolves the EP devices once and raises ``RuntimeError`` if the EP DLL isn't
    found (so a CLI run fails loudly rather than silently on CPU). Sets
    ``session.disable_aot_function_inlining=1`` — without it ORT expands ai.onnx
    Gelu into onnx.Erf primitives before the EP claims the subgraph, and onnx.Erf
    has no ONNX->HIP converter, so the decoder silently falls back to CPU.
    """
    devices = register_morphizen_ep(repo_root)
    if not devices:
        raise RuntimeError(
            "AMDGPU EP not found — run `python build.py` first (expected "
            "amdgpu-ep.dll in the EP bin dir)."
        )

    def factory(model_name):
        so = ort.SessionOptions()
        so.add_session_config_entry("session.disable_aot_function_inlining", "1")
        # bitcode by default; HIPEP_ARTIFACT_FORMAT=NATIVE is an opt-in escape
        # hatch (per-model DLL) — normally unneeded. See apply_artifact_format.
        apply_artifact_format(so)
        # profile=llm tells the AMDGPU umbrella to dispatch to the hipgpu backend.
        so.add_provider_for_devices(devices, dict(EP_PROVIDER_OPTIONS))
        return ort.InferenceSession(str(model_dir / model_name), sess_options=so)

    return factory


# ── KV helpers ────────────────────────────────────────────────────────────────


def encoder_cross_kv(enc_session, audio_fp, dtype=np.float32):
    """Run the encoder and return (hidden_states, cross_kv_dict).

    cross_kv keys are the DECODER input names (``past_key_cross_*`` /
    ``past_value_cross_*``); values are cast to ``dtype`` to match the decoder's
    KV precision. Indexed by NAME — the encoder output order is blocked (all keys
    then all values), not interleaved. ``dtype`` is ``np.float32`` for the fp32
    model and ``np.float16`` for the fp16 model (whose cross-KV outputs are
    already fp16, so the cast is a no-op there).
    """
    feats = audio_fp.astype(dtype)
    names = [o.name for o in enc_session.get_outputs()]
    out = enc_session.run(None, {"audio_features": feats})
    omap = dict(zip(names, out))
    cross = {}
    for i in range(N_LAYERS):
        cross[f"past_key_cross_{i}"] = omap[f"present_key_cross_{i}"].astype(dtype)
        cross[f"past_value_cross_{i}"] = omap[f"present_value_cross_{i}"].astype(dtype)
    return omap["hidden_states"], cross


def zeroed_self_past(dtype=np.float32):
    """32×(past_key_self, past_value_self) zeroed at the static 448-slot shape."""
    kv = {}
    for i in range(N_LAYERS):
        kv[f"past_key_self_{i}"] = np.zeros(
            (1, N_HEADS, SELF_KV_SLOTS, HEAD_DIM), dtype=dtype
        )
        kv[f"past_value_self_{i}"] = np.zeros(
            (1, N_HEADS, SELF_KV_SLOTS, HEAD_DIM), dtype=dtype
        )
    return kv


# ── Greedy decode ─────────────────────────────────────────────────────────────


def greedy_decode_cpu(
    session_factory, audio_fp, max_length=200, timings=None, dtype=np.float32
):
    """CPU greedy decode on the DYNAMIC (pre-surgery) graphs.

    Reference path: the dynamic decoder grows an empty past per step. This is the
    only valid CPU reference — ORT's MHA ignores ``past_sequence_length`` without
    ``past_present_share_buffer`` (which the surgery can't set), so the static
    graph on CPU would attend to all 448 zeroed slots. ``dtype`` selects model
    precision (fp32 default / fp16) for the fp16 reference compare.

    If ``timings`` (a dict) is passed it is populated with the same keys as the
    GPU path (``enc_ms`` / ``prefill_ms`` / ``decode_ms`` / ``n_decode_steps``)
    so the CLI can print one metrics table for either backend. Phase split
    mirrors the GPU path: the first ``dec.run`` over the S=4 start tokens is
    prefill (time-to-first-token); each subsequent single-token run is a decode
    step.
    """
    import time

    enc = session_factory("encoder.onnx")
    _t = time.perf_counter()
    _, cross = encoder_cross_kv(enc, audio_fp, dtype=dtype)
    enc_ms = (time.perf_counter() - _t) * 1e3
    dec = session_factory("decoder.onnx")
    out_names = [o.name for o in dec.get_outputs()]
    self_kv = {}
    for i in range(N_LAYERS):
        self_kv[f"past_key_self_{i}"] = np.zeros((1, N_HEADS, 0, HEAD_DIM), dtype=dtype)
        self_kv[f"past_value_self_{i}"] = np.zeros(
            (1, N_HEADS, 0, HEAD_DIM), dtype=dtype
        )
    tokens = list(START_TOKENS)
    ids = np.array([START_TOKENS], dtype=np.int32)
    prefill_ms = 0.0
    decode_ms = 0.0
    n_decode_steps = 0
    first = True
    while len(tokens) < max_length:
        _t = time.perf_counter()
        step = dict(
            zip(out_names, dec.run(None, {"input_ids": ids, **self_kv, **cross}))
        )
        dt = (time.perf_counter() - _t) * 1e3
        if first:
            prefill_ms = dt  # S=4 start tokens -> first generated token
            first = False
        else:
            decode_ms += dt
            n_decode_steps += 1
        nxt = int(np.argmax(step["logits"][0, -1, :]))
        tokens.append(nxt)
        if nxt == EOT:
            break
        ids = np.array([[nxt]], dtype=np.int32)
        for i in range(N_LAYERS):
            self_kv[f"past_key_self_{i}"] = step[f"present_key_self_{i}"]
            self_kv[f"past_value_self_{i}"] = step[f"present_value_self_{i}"]

    if timings is not None:
        timings["enc_ms"] = enc_ms
        timings["prefill_ms"] = prefill_ms
        timings["decode_ms"] = decode_ms
        timings["n_decode_steps"] = n_decode_steps
    return tokens


def greedy_decode_morphizen(
    session_factory,
    audio_fp,
    max_length=200,
    timings=None,
    use_iobinding=True,
    dtype=np.float32,
):
    """GPU greedy decode using the canonical fixed model variants.

    ``dtype`` selects the model precision: ``np.float32`` (default) for the fp32
    model, ``np.float16`` for the fp16 model. It drives the audio_features cast
    and all KV-cache buffer dtypes; the logits buffer stays fp32 either way
    because the OGA fp16 build keeps lm_head fp32 (argmax-lossless).

    Every op dispatches on the GPU path. The decoder self-attn uses the GQA
    seqlens_k convention (seqlens_k = total_tokens - 1), so past_sequence_length
    is fed as (past_tokens + S) - 1: S-1 at prefill, then the running total each
    decode step. position_ids is fed separately = [real_past .. real_past+S-1].

    ``use_iobinding`` (default True) selects the FAST, GPU-resident KV-cache path
    (zero-copy past<->present aliasing; see ``_greedy_decode_iobinding``). This is
    the default because it is BOTH correct (bit-identical tokens to the numpy
    path) AND fast — the naive numpy path re-marshals all 96 KV tensors
    host<->device every step and is decode-plumbing-bound, not a real GPU number.
    Set ``use_iobinding=False`` to fall back to the simple numpy path for
    debugging / per-token CPU-vs-GPU divergence bisection.

    If ``timings`` (a dict) is passed, it is populated in place with per-phase
    latency in milliseconds: ``enc_ms`` (encoder), ``prefill_ms`` (S=4 prefill,
    i.e. first-token compute), ``decode_ms`` (the whole S=1 decode loop), and
    ``n_decode_steps`` (decode-loop iterations). ORT run is synchronous (it
    materializes outputs on the host), so wall-clock around each call is a valid
    GPU latency. The FIRST call of each session also pays the one-time model-DLL
    load + kernel autotune, so on a cold run enc_ms / prefill_ms / the first
    decode step are inflated — warm up if you want steady-state numbers.
    """
    impl = _greedy_decode_iobinding if use_iobinding else _greedy_decode_numpy
    return impl(
        session_factory, audio_fp, max_length=max_length, timings=timings, dtype=dtype
    )


def _greedy_decode_iobinding(
    session_factory, audio_fp, max_length=200, timings=None, dtype=np.float32
):
    """FAST GPU-resident greedy decode: IOBinding + KV-cache aliasing.

    Keeps the whole KV cache on the GPU instead of marshaling 32×2 self-KV +
    32×2 cross-KV tensors host<->device every step:

      * self-KV (grows per step): one GPU OrtValue per (layer, kind), 448-slot
        ``[1,20,448,64]`` fp32, aliased past<->present. Binding the SAME OrtValue
        to ``past_*_self_i`` (input) AND ``present_*_self_i`` (output) makes the
        runtime see ``past_ptr == present_ptr`` → it appends the new token's K/V
        in-place at ``slot[past_len]`` (the shared-buffer fast path). No copy.
        The SAME GPU OrtValues are bound to BOTH the prefill and decode sessions,
        so the prefill-written slots [0..S-1] carry into decode with zero copies.
      * cross-KV (constant): uploaded to GPU OrtValues ONCE and bound once (the
        decoder consumes ``past_*_cross_i`` and emits no present_cross).
      * per-step inputs (tiny): ``input_ids`` [1,1] int64, ``past_sequence_length``
        [1] int32, ``position_ids`` [1,1] int64 — updated in place via
        ``update_inplace`` so the GPU addresses stay stable and the binding is
        never torn down inside the loop.
      * logits: bound to a GPU OrtValue, read back once per step for the argmax —
        one small D2H of n_vocab floats, tiny next to the 96 KV tensors removed.

    Tokens are bit-identical to ``_greedy_decode_numpy`` (same seqlens_k /
    position_ids convention); this path only changes WHERE the KV lives, not the
    math.
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

    s0 = len(START_TOKENS)

    # ── Encoder (run once) — gives cross-KV ──────────────────────────────────
    enc = session_factory("encoder_fixed.onnx")
    enc_names = [o.name for o in enc.get_outputs()]
    _t = time.perf_counter()
    enc_out = dict(
        zip(enc_names, enc.run(None, {"audio_features": audio_fp.astype(dtype)}))
    )
    enc_ms = (time.perf_counter() - _t) * 1e3

    # ── Upload cross-KV to GPU ONCE (constant across the whole generation) ────
    cross_gpu = {}
    for i in range(N_LAYERS):
        cross_gpu[f"past_key_cross_{i}"] = _gpu_val(
            enc_out[f"present_key_cross_{i}"].astype(dtype)
        )
        cross_gpu[f"past_value_cross_{i}"] = _gpu_val(
            enc_out[f"present_value_cross_{i}"].astype(dtype)
        )

    # ── self-KV GPU buffers (448-slot, zeroed) aliased past<->present ─────────
    kv_shape = (1, N_HEADS, SELF_KV_SLOTS, HEAD_DIM)
    self_gpu = {}
    for i in range(N_LAYERS):
        for kind in ("key", "value"):
            self_gpu[(i, kind)] = _gpu_val(np.zeros(kv_shape, dtype=dtype))

    # The logits GPU buffer dtype must MATCH the graph's logits output dtype, or
    # IOBinding raises "Unexpected output data type". The OGA fp16 build computes
    # the lm_head in fp32 internally (so argmax is lossless) but CASTS logits back
    # to fp16 at the graph output — so the bound buffer is fp16 for the fp16 model,
    # fp32 for the fp32 model. The argmax read-back upcasts either way.
    pref_logits_gpu = _gpu_empty([1, s0, CFG.n_vocab], dtype)
    dec_logits_gpu = _gpu_empty([1, 1, CFG.n_vocab], dtype)

    def _bind_self_and_cross(io):
        for i in range(N_LAYERS):
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

    # ── Prefill (S=4 start tokens) ───────────────────────────────────────────
    pref_dec = session_factory("decoder_fixed_prefill.onnx")
    pref_io = pref_dec.io_binding()
    pref_io.bind_ortvalue_input(
        "input_ids", _gpu_val(np.array([START_TOKENS], dtype=np.int64))
    )
    pref_io.bind_ortvalue_input(
        "past_sequence_length", _gpu_val(np.array([s0 - 1], dtype=np.int32))
    )
    pref_io.bind_ortvalue_input("position_ids", _gpu_val(np.arange(s0, dtype=np.int64)))
    _bind_self_and_cross(pref_io)
    pref_io.bind_ortvalue_output("logits", pref_logits_gpu)

    _t = time.perf_counter()
    pref_dec.run_with_iobinding(pref_io)
    pref_logits = pref_logits_gpu.numpy()
    prefill_ms = (time.perf_counter() - _t) * 1e3

    tokens = list(START_TOKENS)
    next_tok = int(np.argmax(np.asarray(pref_logits[0, -1, :], dtype=np.float32)))
    tokens.append(next_tok)
    total = s0  # running total_tokens after prefill

    # ── Decode loop (S=1) — same GPU KV OrtValues, tiny inputs updated in place ─
    dec = session_factory("decoder_fixed_decode.onnx")
    dec_io = dec.io_binding()
    ids_gpu = _gpu_val(np.array([[next_tok]], dtype=np.int64))
    pastlen_gpu = _gpu_val(np.array([total], dtype=np.int32))
    posids_gpu = _gpu_val(np.array([total], dtype=np.int64))
    dec_io.bind_ortvalue_input("input_ids", ids_gpu)
    dec_io.bind_ortvalue_input("past_sequence_length", pastlen_gpu)
    dec_io.bind_ortvalue_input("position_ids", posids_gpu)
    _bind_self_and_cross(dec_io)
    dec_io.bind_ortvalue_output("logits", dec_logits_gpu)

    n_decode_steps = 0
    _t = time.perf_counter()
    while len(tokens) < max_length and next_tok != EOT and total < SELF_KV_SLOTS:
        ids_gpu.update_inplace(np.array([[next_tok]], dtype=np.int64))
        pastlen_gpu.update_inplace(np.array([total], dtype=np.int32))
        posids_gpu.update_inplace(np.array([total], dtype=np.int64))

        dec.run_with_iobinding(dec_io)
        logits = dec_logits_gpu.numpy()  # one small D2H for the greedy argmax
        total += 1
        n_decode_steps += 1
        next_tok = int(np.argmax(np.asarray(logits[0, -1, :], dtype=np.float32)))
        tokens.append(next_tok)
        if next_tok == EOT:
            break
    decode_ms = (time.perf_counter() - _t) * 1e3

    if timings is not None:
        timings["enc_ms"] = enc_ms
        timings["prefill_ms"] = prefill_ms
        timings["decode_ms"] = decode_ms
        timings["n_decode_steps"] = n_decode_steps
    return tokens


def _greedy_decode_numpy(
    session_factory, audio_fp, max_length=200, timings=None, dtype=np.float32
):
    """Simple numpy greedy decode (debug fallback for the IOBinding path).

    Re-marshals all 32×2 self-KV + 32×2 cross-KV tensors host<->device every
    step, so it is decode-plumbing-bound — NOT a representative GPU throughput.
    Kept because its structure mirrors ``greedy_decode_cpu`` exactly, which makes
    per-token CPU-vs-GPU divergence bisection trivial. Tokens are bit-identical
    to the IOBinding path. ``dtype`` selects model precision (fp32 / fp16).
    """
    import time

    enc = session_factory("encoder_fixed.onnx")
    names = [o.name for o in enc.get_outputs()]
    _t = time.perf_counter()
    out = dict(zip(names, enc.run(None, {"audio_features": audio_fp.astype(dtype)})))
    enc_ms = (time.perf_counter() - _t) * 1e3
    cross = {}
    for i in range(N_LAYERS):
        cross[f"past_key_cross_{i}"] = out[f"present_key_cross_{i}"].astype(dtype)
        cross[f"past_value_cross_{i}"] = out[f"present_value_cross_{i}"].astype(dtype)

    self_kv = zeroed_self_past(dtype)
    s0 = len(START_TOKENS)
    pref_dec = session_factory("decoder_fixed_prefill.onnx")
    pref_out = [o.name for o in pref_dec.get_outputs()]
    _t = time.perf_counter()
    pref = dict(
        zip(
            pref_out,
            pref_dec.run(
                None,
                {
                    # int64 ids: the surgered decoder re-types input_ids to
                    # int64 so the token-embed Gather gets int64 indices (the
                    # only dtype hip_gather supports — see inject_seqlens_k).
                    "input_ids": np.array([START_TOKENS], dtype=np.int64),
                    # total-1 convention: 0 past + s0 query -> s0-1.
                    "past_sequence_length": np.array([s0 - 1], dtype=np.int32),
                    # position_ids = [real_past .. real_past + S - 1]; real_past
                    # = total - S = 0 at prefill (see inject_seqlens_k: the
                    # position-embedding Slice was replaced with a
                    # Gather(embed_positions, position_ids) to dodge the
                    # GPU runtime-bounds-Slice limitation).
                    "position_ids": np.arange(s0, dtype=np.int64),
                    **self_kv,
                    **cross,
                },
            ),
        )
    )
    prefill_ms = (time.perf_counter() - _t) * 1e3
    for i in range(N_LAYERS):
        self_kv[f"past_key_self_{i}"] = pref[f"present_key_self_{i}"]
        self_kv[f"past_value_self_{i}"] = pref[f"present_value_self_{i}"]

    tokens = list(START_TOKENS)
    next_tok = int(np.argmax(np.asarray(pref["logits"][0, -1, :], dtype=np.float32)))
    tokens.append(next_tok)
    total = s0  # running total_tokens after prefill

    dec = session_factory("decoder_fixed_decode.onnx")
    dec_out = [o.name for o in dec.get_outputs()]
    n_decode_steps = 0
    _t = time.perf_counter()
    while len(tokens) < max_length and next_tok != EOT and total < SELF_KV_SLOTS:
        step = dict(
            zip(
                dec_out,
                dec.run(
                    None,
                    {
                        "input_ids": np.array([[next_tok]], dtype=np.int64),
                        # total-1: total_tokens grows by 1 each step; feed prev total.
                        "past_sequence_length": np.array([total], dtype=np.int32),
                        # decode S=1: real_past = total (tokens already written);
                        # the single position id is `total`.
                        "position_ids": np.array([total], dtype=np.int64),
                        **self_kv,
                        **cross,
                    },
                ),
            )
        )
        for i in range(N_LAYERS):
            self_kv[f"past_key_self_{i}"] = step[f"present_key_self_{i}"]
            self_kv[f"past_value_self_{i}"] = step[f"present_value_self_{i}"]
        total += 1
        n_decode_steps += 1
        next_tok = int(
            np.argmax(np.asarray(step["logits"][0, -1, :], dtype=np.float32))
        )
        tokens.append(next_tok)
        if next_tok == EOT:
            break
    decode_ms = (time.perf_counter() - _t) * 1e3

    if timings is not None:
        timings["enc_ms"] = enc_ms
        timings["prefill_ms"] = prefill_ms
        timings["decode_ms"] = decode_ms
        timings["n_decode_steps"] = n_decode_steps
    return tokens


# ── Token <-> text + audio ────────────────────────────────────────────────────


def decode_text(tokens):
    """Detokenize a greedy token list to text, dropping the forced-start specials."""
    from transformers import WhisperTokenizer

    tok = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
    body = [t for t in tokens if t < EOT]
    return tok.decode(body, skip_special_tokens=True).strip()


def load_audio_features(audio_path):
    """Load a 16 kHz wav -> fp32 log-mel ``audio_features`` (Whisper's 30 s window)."""
    return make_whisper_inputs(pathlib.Path(audio_path), CFG)["audio_features"].astype(
        np.float32
    )


def audio_duration_s(audio_path):
    """Real (un-padded) duration of the wav in seconds, for the RTF denominator.

    RTF (real-time factor) = processing_time / audio_duration; < 1.0 means faster
    than real time. Uses the SOURCE clip length, not Whisper's padded 30 s window,
    so RTF reflects the actual speech processed.
    """
    import soundfile as sf

    info = sf.info(str(audio_path))
    return info.frames / float(info.samplerate)
