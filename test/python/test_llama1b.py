#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Comprehensive 1B model test suite.

Covers:
- ORT-only: fixed/dynamic shapes, prefill/decode, latency + accuracy vs CPU
- OGA: generation latency, accuracy, shape switching, chunked prefill

Model: Llama-3.2-1B-Instruct (q4f16, 16 layers, 8 KV heads, head_dim=64).
All MorphiZen EP tests use IOBinding with device memory.

Test order matters: DML tests run first (before EP registration), then EP tests,
then OGA tests. This avoids OOM on iGPU from concurrent EP+DML sessions.
"""

import gc
import json
import time

import numpy as np
import pytest

from conftest import (
    REPO_ROOT,
    LlamaModelConfig,
    cleanup,
    compare_logits,
    compare_outputs,
    create_cpu_session,
    create_dml_session,
    create_ep_session,
    download,
    ensure_fixed_model,
    ensure_model,
    ensure_pipeline_sliding_oga_files,
    extract_kv_cache,
    get_next_token,
    make_decode_inputs,
    make_dim_map,
    make_llama_inputs,
    make_pipeline_sliding_genai_config,
    make_prefill_inputs,
    make_prompt_tokens,
    oga_generate,
    oga_generate_timed,
    patch_genai_config_for_morphizen,
    report,
    restore_genai_config,
    run_cpu_reference_generation,
    run_iobinding_once,
    run_oga_static_kv_pipeline,
    run_timed,
    run_timed_iobinding,
    setup_oga_ep,
)

# ruff: noqa: F811  # pytest fixtures shadow function names

# ── Model config ─────────────────────────────────────────────────────────────

_MODEL_DIR = REPO_ROOT / "models" / "Llama-3.2-1B-Instruct"
_ONNX_FILE = "model_q4f16.onnx"
_DATA_FILE = "model_q4f16.onnx_data"
_HF_BASE = (
    "https://huggingface.co/onnx-community/Llama-3.2-1B-Instruct-ONNX/resolve/main/onnx"
)
_HF_ROOT = (
    "https://huggingface.co/onnx-community/Llama-3.2-1B-Instruct-ONNX/resolve/main"
)

NUM_LAYERS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 64
VOCAB_SIZE = 128256
BOS_TOKEN = 128000
MAX_SEQ_LEN = 256

# "Hello, how are you?" in Llama 3 tokenizer
PROMPT_TOKENS = [128000, 9906, 11, 1268, 527, 499, 30]
FILLER_TOKENS = [9906, 11, 1268, 527, 499, 30]
NUM_GENERATE_TOKENS = 10

_OGA_TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]

_GENAI_CONFIG = {
    "model": {
        "bos_token_id": 128000,
        "context_length": 131072,
        "decoder": {
            "session_options": {
                "log_id": "onnxruntime-genai",
                "provider_options": [],
            },
            "filename": "model_q4f16.onnx",
            "head_size": 64,
            "hidden_size": 2048,
            "inputs": {
                "input_ids": "input_ids",
                "attention_mask": "attention_mask",
                "past_key_names": "past_key_values.%d.key",
                "past_value_names": "past_key_values.%d.value",
            },
            "outputs": {
                "logits": "logits",
                "present_key_names": "present.%d.key",
                "present_value_names": "present.%d.value",
            },
            "num_attention_heads": 32,
            "num_hidden_layers": 16,
            "num_key_value_heads": 8,
        },
        "eos_token_id": [128001, 128008, 128009],
        "pad_token_id": 128001,
        "type": "llama",
        "vocab_size": 128256,
    },
    "search": {
        "chunk_size": 1024,
        "diversity_penalty": 0.0,
        "do_sample": True,
        "early_stopping": True,
        "length_penalty": 1.0,
        "max_length": 131072,
        "min_length": 0,
        "no_repeat_ngram_size": 0,
        "num_beams": 1,
        "num_return_sequences": 1,
        "past_present_share_buffer": True,
        "repetition_penalty": 1.0,
        "temperature": 0.6,
        "top_k": 1,
        "top_p": 0.9,
    },
}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _ensure_model():
    return ensure_model(_MODEL_DIR, _ONNX_FILE, _DATA_FILE, _HF_BASE)


def _ensure_fixed_model(seq_len, kv_len):
    _ensure_model()
    return ensure_fixed_model(_MODEL_DIR, _ONNX_FILE, seq_len, kv_len)


def _make_cfg(max_seq_len):
    return LlamaModelConfig(
        num_kv_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        max_seq_len=max_seq_len,
        has_position_ids=False,
    )


def _make_prompt_tokens(length):
    return make_prompt_tokens(BOS_TOKEN, FILLER_TOKENS, length)


def _ensure_oga_files():
    """Download tokenizer files and generate genai_config.json for 1B model."""
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for fname in _OGA_TOKENIZER_FILES:
        dest = _MODEL_DIR / fname
        if not dest.exists():
            download(f"{_HF_ROOT}/{fname}", dest)
    config_path = _MODEL_DIR / "genai_config.json"
    if not config_path.exists():
        with open(config_path, "w") as f:
            json.dump(_GENAI_CONFIG, f, indent=4)
        print(f"  Generated {config_path.name}")
    else:
        # Migrate stale on-disk config: ensure chunk_size=1024 is set for the
        # dynamic-shape model (Pareto-optimal default for MorphiZenEP, see
        # CLAUDE.md "OGA chunked prefill" gotcha).
        with open(config_path) as f:
            existing = json.load(f)
        if existing.get("search", {}).get("chunk_size") != 1024:
            existing.setdefault("search", {})["chunk_size"] = 1024
            with open(config_path, "w") as f:
                json.dump(existing, f, indent=4)
            print(f"  Migrated {config_path.name}: search.chunk_size=1024")


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def dynamic_model_path():
    return _ensure_model()


@pytest.fixture(scope="session")
def fixed_decode_path(dynamic_model_path):
    return _ensure_fixed_model(seq_len=1, kv_len=MAX_SEQ_LEN)


@pytest.fixture(scope="session")
def fixed_prefill_128_path(dynamic_model_path):
    return _ensure_fixed_model(seq_len=128, kv_len=MAX_SEQ_LEN)


# ── DML baseline tests (run before EP registration) ─────────────────────────


class TestLlama1BDML:
    """DML latency baselines. Run first to avoid EP memory contention."""

    def test_dml_decode(self, fixed_decode_path):
        """DML decode latency baseline (sq=1, kv=128)."""
        cfg = _make_cfg(MAX_SEQ_LEN)
        dml_sess = create_dml_session(fixed_decode_path)
        inputs = make_llama_inputs(cfg, seq_len=1)
        times = run_timed(dml_sess, inputs)
        report("DML decode (sq=1 kv=128)", times)
        cleanup(dml_sess)


# ── EP ORT tests ─────────────────────────────────────────────────────────────


class TestLlama1BORT:
    """MorphiZen EP tests with IOBinding + device memory."""

    def test_ort_fixed_decode(self, fixed_decode_path, repo_root):
        """Fixed-shape decode (seq=1, kv=128): accuracy vs CPU + latency."""
        cfg = _make_cfg(MAX_SEQ_LEN)
        inputs = make_llama_inputs(cfg, seq_len=1)

        cpu_sess = create_cpu_session(fixed_decode_path)
        ref = cpu_sess.run(None, inputs)
        output_names = [o.name for o in cpu_sess.get_outputs()]
        cleanup(cpu_sess)

        ep_sess = create_ep_session(fixed_decode_path, REPO_ROOT)
        test = run_iobinding_once(ep_sess, inputs, cfg, use_device_memory=False)

        ok, _ = compare_outputs(ref, test, output_names, "EP fixed decode sq=1")
        assert ok, "Fixed decode accuracy check failed"

        times = run_timed_iobinding(ep_sess, inputs, cfg, use_device_memory=True)
        report("EP fixed decode (sq=1 kv=128)", times)
        cleanup(ep_sess)

    def test_ort_fixed_prefill_128(self, fixed_prefill_128_path, repo_root):
        """Fixed-shape prefill (seq=128, kv=128): accuracy vs CPU + latency."""
        self._run_fixed_prefill(fixed_prefill_128_path, seq_len=128)

    def _run_fixed_prefill(self, model_path, seq_len):
        cfg = _make_cfg(MAX_SEQ_LEN)
        inputs = make_llama_inputs(cfg, seq_len=seq_len)

        cpu_sess = create_cpu_session(model_path)
        ref = cpu_sess.run(None, inputs)
        output_names = [o.name for o in cpu_sess.get_outputs()]
        cleanup(cpu_sess)

        ep_sess = create_ep_session(model_path, REPO_ROOT)
        test = run_iobinding_once(ep_sess, inputs, cfg, use_device_memory=False)

        ok, _ = compare_outputs(
            ref, test, output_names, f"EP fixed prefill sq={seq_len}"
        )
        assert ok, f"Fixed prefill sq={seq_len} accuracy check failed"

        times = run_timed_iobinding(ep_sess, inputs, cfg, use_device_memory=True)
        report(f"EP fixed prefill (sq={seq_len} kv={MAX_SEQ_LEN})", times)
        cleanup(ep_sess)

    def test_ort_dynamic_prefill_128(self, dynamic_model_path, repo_root):
        """Dynamic-shape prefill at seq_len=128."""
        self._run_dynamic_prefill(dynamic_model_path, seq_len=128)

    def _run_dynamic_prefill(self, model_path, seq_len):
        cfg = _make_cfg(seq_len)
        inputs = make_llama_inputs(cfg, seq_len=seq_len)
        dim_map = make_dim_map(seq_len, seq_len)

        cpu_sess = create_cpu_session(model_path)
        ref = cpu_sess.run(None, inputs)
        output_names = [o.name for o in cpu_sess.get_outputs()]
        cleanup(cpu_sess)

        ep_sess = create_ep_session(model_path, REPO_ROOT)
        test = run_iobinding_once(
            ep_sess, inputs, cfg, use_device_memory=False, dim_map=dim_map
        )

        ok, _ = compare_outputs(
            ref, test, output_names, f"EP dynamic prefill sq={seq_len}"
        )
        assert ok, f"Dynamic prefill sq={seq_len} accuracy check failed"

        times = run_timed_iobinding(
            ep_sess, inputs, cfg, use_device_memory=True, dim_map=dim_map
        )
        report(f"EP dynamic prefill (sq={seq_len})", times)
        cleanup(ep_sess)

    def test_ort_dynamic_decode(self, dynamic_model_path, repo_root):
        """Dynamic-shape decode (seq=1, kv=128): accuracy vs CPU + latency."""
        cfg = _make_cfg(MAX_SEQ_LEN)
        inputs = make_llama_inputs(cfg, seq_len=1)
        dim_map = make_dim_map(1, MAX_SEQ_LEN)

        cpu_sess = create_cpu_session(dynamic_model_path)
        ref = cpu_sess.run(None, inputs)
        output_names = [o.name for o in cpu_sess.get_outputs()]
        cleanup(cpu_sess)

        ep_sess = create_ep_session(dynamic_model_path, REPO_ROOT)
        test = run_iobinding_once(
            ep_sess, inputs, cfg, use_device_memory=False, dim_map=dim_map
        )

        ok, _ = compare_outputs(ref, test, output_names, "EP dynamic decode sq=1")
        assert ok, "Dynamic decode accuracy check failed"

        times = run_timed_iobinding(
            ep_sess, inputs, cfg, use_device_memory=True, dim_map=dim_map
        )
        report("EP dynamic decode (sq=1 kv=128)", times)
        cleanup(ep_sess)

    def test_ort_dynamic_vs_fixed(
        self, dynamic_model_path, fixed_decode_path, repo_root
    ):
        """Dynamic and fixed shape models produce same EP output (decode)."""
        cfg = _make_cfg(MAX_SEQ_LEN)
        inputs = make_llama_inputs(cfg, seq_len=1)
        dim_map = make_dim_map(1, MAX_SEQ_LEN)

        fixed_sess = create_ep_session(fixed_decode_path, REPO_ROOT)
        fixed_out = run_iobinding_once(fixed_sess, inputs, cfg, use_device_memory=False)
        output_names = [o.name for o in fixed_sess.get_outputs()]
        cleanup(fixed_sess)

        dyn_sess = create_ep_session(dynamic_model_path, REPO_ROOT)
        dyn_out = run_iobinding_once(
            dyn_sess, inputs, cfg, use_device_memory=False, dim_map=dim_map
        )
        cleanup(dyn_sess)

        ok, _ = compare_outputs(
            fixed_out, dyn_out, output_names, "dynamic vs fixed decode"
        )
        assert ok, "Dynamic vs fixed shape comparison failed"

    def test_ort_per_step_logits(self, dynamic_model_path, repo_root):
        """Prefill->decode generation loop with per-step logits vs CPU.

        Both sessions use sess.run() (not IOBinding) to extract and feed
        KV cache at each step.
        """
        cpu_sess = create_cpu_session(dynamic_model_path)
        ep_sess = create_ep_session(dynamic_model_path, REPO_ROOT)
        output_names = [o.name for o in cpu_sess.get_outputs()]
        logits_idx = output_names.index("logits")

        print(f"\n{'=' * 60}")
        print(
            f"EP vs CPU per-step logits: prompt={len(PROMPT_TOKENS)}, "
            f"generate={NUM_GENERATE_TOKENS}"
        )
        print(f"{'=' * 60}")

        all_ok = True

        # Prefill
        cfg = _make_cfg(MAX_SEQ_LEN)
        inputs = make_prefill_inputs(cfg, PROMPT_TOKENS, MAX_SEQ_LEN)
        cpu_out = cpu_sess.run(None, inputs)
        ep_out = ep_sess.run(None, inputs)

        print(f"\n  Prefill (seq_len={len(PROMPT_TOKENS)}):")
        ok = compare_logits(cpu_out[logits_idx], ep_out[logits_idx], "prefill")
        all_ok = all_ok and ok

        cpu_token = get_next_token(cpu_out[logits_idx])
        ep_token = get_next_token(ep_out[logits_idx])
        cpu_kv = extract_kv_cache(cpu_out, output_names)
        ep_kv = extract_kv_cache(ep_out, output_names)

        match = "MATCH" if cpu_token == ep_token else "DIFFER"
        print(f"  CPU: {cpu_token}, EP: {ep_token}  [{match}]")

        cpu_generated = [cpu_token]
        ep_generated = [ep_token]

        # Decode loop
        for step in range(NUM_GENERATE_TOKENS - 1):
            position = len(PROMPT_TOKENS) + step
            cpu_inputs = make_decode_inputs(
                cfg, cpu_token, position, cpu_kv, MAX_SEQ_LEN
            )
            ep_inputs = make_decode_inputs(cfg, ep_token, position, ep_kv, MAX_SEQ_LEN)

            cpu_out = cpu_sess.run(None, cpu_inputs)
            ep_out = ep_sess.run(None, ep_inputs)

            ok = compare_logits(
                cpu_out[logits_idx], ep_out[logits_idx], f"decode[{step + 1}]"
            )
            all_ok = all_ok and ok

            cpu_token = get_next_token(cpu_out[logits_idx])
            ep_token = get_next_token(ep_out[logits_idx])
            cpu_kv = extract_kv_cache(cpu_out, output_names)
            ep_kv = extract_kv_cache(ep_out, output_names)

            cpu_generated.append(cpu_token)
            ep_generated.append(ep_token)

            match = "MATCH" if cpu_token == ep_token else "DIFFER"
            print(f"  CPU: {cpu_token}, EP: {ep_token}  [{match}]")

        token_match_rate = sum(
            1 for a, b in zip(cpu_generated, ep_generated) if a == b
        ) / len(cpu_generated)
        print(f"\n  CPU  generated: {cpu_generated}")
        print(f"  EP   generated: {ep_generated}")
        print(f"  Token match rate: {token_match_rate:.0%}")
        print(f"{'=' * 60}")

        cleanup(cpu_sess, ep_sess)
        assert all_ok, "Per-step logits accuracy check failed"


# ── OGA EP tests ─────────────────────────────────────────────────────────────


class TestLlama1BOGA:
    """OGA integration tests. All use past_present_share_buffer=true unless noted."""

    def test_oga_ep_generation(self):
        """OGA+MorphiZen EP latency at prompt_len=128."""
        og, ep_dll = setup_oga_ep(REPO_ROOT)
        _ensure_model()
        _ensure_oga_files()
        patch_genai_config_for_morphizen(_MODEL_DIR, ep_dll)

        try:
            try:
                model = og.Model(str(_MODEL_DIR))
            except RuntimeError as e:
                if "Unknown provider name" in str(e):
                    pytest.skip("OGA does not recognize MorphiZen EP")
                raise

            tokenizer = og.Tokenizer(model)

            tokens = _make_prompt_tokens(128)
            generated, ttft_ms, tps = oga_generate_timed(
                og, model, tokenizer, tokens, max_new=128
            )
            print(
                f"\n  OGA EP prompt=128: prefill={ttft_ms:7.1f}ms  "
                f"tps={tps:5.1f}  generated={generated}"
            )
            assert generated > 0

            del model
            gc.collect()
        finally:
            restore_genai_config(_MODEL_DIR)

    def test_oga_ep_no_share_buffer(self):
        """Verify generation works with past_present_share_buffer=false."""
        og, ep_dll = setup_oga_ep(REPO_ROOT)
        _ensure_model()
        _ensure_oga_files()
        patch_genai_config_for_morphizen(_MODEL_DIR, ep_dll)

        try:
            config = og.Config(str(_MODEL_DIR))
            config.overlay(json.dumps({"search": {"past_present_share_buffer": False}}))

            try:
                model = og.Model(config)
            except RuntimeError as e:
                if "Unknown provider name" in str(e):
                    pytest.skip("OGA does not recognize MorphiZen EP")
                raise

            tokenizer = og.Tokenizer(model)
            tokens = _make_prompt_tokens(128)
            generated, ttft_ms, tps = oga_generate(
                og, model, tokenizer, tokens, max_new=10
            )
            print(
                f"\n  OGA EP no_share_buffer: generated={len(generated)}"
                f" ttft={ttft_ms:.1f}ms tps={tps:.1f}"
            )
            assert len(generated) > 0, (
                "No tokens generated with past_present_share_buffer=false"
            )

            del model
            gc.collect()
        finally:
            restore_genai_config(_MODEL_DIR)

    def test_oga_ep_shape_switching(self):
        """Different prompt lengths via rewind_to on a single Generator.

        Reuses one Generator (via rewind_to(0) between prompts) so the
        EP's constants and GEMM algorithm cache stay warm.  First use of
        each sequence length is slow (hipBLASLt algorithm finding),
        repeats are fast.
        """
        og, ep_dll = setup_oga_ep(REPO_ROOT)
        _ensure_model()
        _ensure_oga_files()
        patch_genai_config_for_morphizen(_MODEL_DIR, ep_dll)

        try:
            try:
                model = og.Model(str(_MODEL_DIR))
            except RuntimeError as e:
                if "Unknown provider name" in str(e):
                    pytest.skip("OGA does not recognize MorphiZen EP")
                raise

            max_prompt = 128
            max_new = 5
            params = og.GeneratorParams(model)
            params.set_search_options(max_length=max_prompt + max_new, do_sample=False)
            generator = og.Generator(model, params)

            print(f"\n{'=' * 60}")
            print("OGA EP shape switching test")
            print(f"{'=' * 60}")

            for prompt_len in [7, 128, 64, 128, 7]:
                generator.rewind_to(0)
                tokens = _make_prompt_tokens(prompt_len)

                t0 = time.perf_counter()
                generator.append_tokens(np.array(tokens, dtype=np.int32))
                generator.generate_next_token()
                ttft_ms = (time.perf_counter() - t0) * 1000
                generated = [int(generator.get_next_tokens()[0])]
                token_times_ms = []

                while not generator.is_done() and len(generated) < max_new:
                    t_tok = time.perf_counter()
                    generator.generate_next_token()
                    token_times_ms.append((time.perf_counter() - t_tok) * 1000)
                    generated.append(int(generator.get_next_tokens()[0]))

                tok_str = "  ".join(f"{t:.1f}" for t in token_times_ms)
                print(
                    f"  prompt={prompt_len:>3d}: "
                    f"prefill={ttft_ms:7.1f}ms  "
                    f"decode=[{tok_str}] ms"
                )
                assert len(generated) > 0

            print(f"{'=' * 60}")

            del generator, model
            gc.collect()
        finally:
            restore_genai_config(_MODEL_DIR)

    def test_oga_ep_chunked_prefill(self, dynamic_model_path):
        """Chunked prefill accuracy: OGA+EP with chunk_size=128 vs CPU.

        Runs a 200-token prompt through:
        1. CPU sess.run() prefill+decode loop (reference, no chunking)
        2. OGA+MorphiZen EP with chunk_size=128 (prompt split into 128+72)

        Compares generated token sequences.  Run with
        MORPHIZEN_DEBUG_MLIR_BACKEND=3 to see per-call input shapes.
        """
        og, ep_dll = setup_oga_ep(REPO_ROOT)
        _ensure_model()
        _ensure_oga_files()

        chunk_size = 128
        prompt_len = 200
        max_new = 10
        prompt_tokens = _make_prompt_tokens(prompt_len)
        max_seq_len = prompt_len + max_new

        cpu_generated = run_cpu_reference_generation(
            dynamic_model_path, _make_cfg(max_seq_len), prompt_tokens, max_new
        )

        # ── OGA+EP with chunked prefill ──
        patch_genai_config_for_morphizen(_MODEL_DIR, ep_dll)
        try:
            config = og.Config(str(_MODEL_DIR))
            config.overlay(json.dumps({"search": {"chunk_size": chunk_size}}))

            try:
                model = og.Model(config)
            except RuntimeError as e:
                if "Unknown provider name" in str(e):
                    pytest.skip("OGA does not recognize MorphiZen EP")
                raise

            tokenizer = og.Tokenizer(model)
            oga_generated, _, _ = oga_generate(
                og, model, tokenizer, prompt_tokens, max_new=max_new
            )

            del model
            gc.collect()
        finally:
            restore_genai_config(_MODEL_DIR)

        # ── Compare ──
        n = min(len(cpu_generated), len(oga_generated))
        cpu_trimmed = cpu_generated[:n]
        oga_trimmed = oga_generated[:n]
        matches = sum(1 for a, b in zip(cpu_trimmed, oga_trimmed) if a == b)
        match_rate = matches / n if n > 0 else 0.0

        print(f"\n{'=' * 60}")
        print(
            f"Chunked prefill accuracy (chunk_size={chunk_size}, "
            f"prompt={prompt_len}, generate={max_new})"
        )
        print(f"  CPU  tokens: {cpu_trimmed}")
        print(f"  OGA  tokens: {oga_trimmed}")
        print(f"  Match rate:  {match_rate:.0%} ({matches}/{n})")
        print(f"{'=' * 60}")

        assert len(oga_generated) > 0, "OGA generated no tokens"
        assert match_rate >= 0.5, (
            f"Token match rate {match_rate:.0%} too low — chunked prefill "
            f"may be producing incorrect results"
        )

    def test_oga_pipeline_sliding_window_160tok(self, dynamic_model_path):
        """OGA decoder-pipeline with sliding_window: chunked prefill via fixed [1,128].

        Builds a static-shape pipeline (prefill=[1,128], decode=[1,1], KV=4096)
        and uses OGA's `decoder.sliding_window` block (window_size=128) to chunk
        a 160-token prompt. OGA's decoder_only_pipeline.cpp:382-394 dispatches
        the prefill sub-model `ceil(prompt_len / window_size)` times within a
        SINGLE Generator::Run(), so a 160-token prompt produces exactly 2
        prefill executions (chunk 0: tokens 0..127, chunk 1: tokens 128..159
        right-padded to 128 with pad_token_id by WindowedInputIDs).

        WindowedInputIDs requires `p_device_inputs_->GetType()` ∈ {QNN, CPU}.
        MorphiZenEP is not in OGA's GPU-input-device list (model.cpp:592-597),
        so it falls through to CPU input device — sliding_window works.

        `slide_key_value_cache: false` keeps the KV buffer at the static
        `total_sequence_length=4096`, preserving the past_present_share_buffer
        fast path.

        Verification:
          1. Token validity — generated tokens compared to a CPU reference of
             the same 160-token prompt (≥50% match, mirroring
             test_oga_ep_chunked_prefill).
          2. Implicit prefill-call-count — the test only succeeds if OGA pushes
             all 160 prompt tokens through the static [1,128] prefill, which
             can ONLY happen via the 2-chunk sliding-window path. To see the
             individual MlirCustomOp::Compute() calls on stderr, run with
             `MORPHIZEN_DEBUG_MLIR_BACKEND=2 GLOG_logtostderr=1`.
        """
        og, _ = setup_oga_ep(REPO_ROOT)

        window_size = 128
        kv_len = _PIPELINE_KV_LEN
        prompt_len = 160
        max_new = 10
        num_chunks_expected = (prompt_len + window_size - 1) // window_size
        assert num_chunks_expected == 2, (
            "test premise: 160 tokens / window 128 = 2 chunks"
        )

        prompt_tokens = _make_prompt_tokens(prompt_len)
        max_seq_len = prompt_len + max_new

        cpu_generated = run_cpu_reference_generation(
            dynamic_model_path, _make_cfg(max_seq_len), prompt_tokens, max_new
        )

        pipeline_dir = _ensure_pipeline_sliding_oga_files(window_size, kv_len)
        oga_generated, ttft_ms = run_oga_static_kv_pipeline(
            og, pipeline_dir, prompt_tokens, kv_len, max_new
        )

        # ── Compare ──
        n = min(len(cpu_generated), len(oga_generated))
        cpu_trimmed = cpu_generated[:n]
        oga_trimmed = oga_generated[:n]
        matches = sum(1 for a, b in zip(cpu_trimmed, oga_trimmed) if a == b)
        match_rate = matches / n if n > 0 else 0.0

        print(f"\n{'=' * 60}")
        print(
            f"OGA pipeline sliding_window (window={window_size}, kv={kv_len}, "
            f"prompt={prompt_len}, generate={max_new})"
        )
        print(
            f"  expected prefill chunks: {num_chunks_expected} "
            f"(ceil({prompt_len}/{window_size}))"
        )
        print(f"  ttft (incl. {num_chunks_expected} prefill chunks): {ttft_ms:7.1f}ms")
        print(f"  CPU  tokens: {cpu_trimmed}")
        print(f"  OGA  tokens: {oga_trimmed}")
        print(f"  Match rate:  {match_rate:.0%} ({matches}/{n})")
        print(f"{'=' * 60}")

        assert len(oga_generated) > 0, "OGA generated no tokens"
        assert match_rate >= 0.5, (
            f"Token match rate {match_rate:.0%} too low — sliding_window "
            f"chunked prefill may be producing incorrect results"
        )


# ── OGA bi-model sliding-window pipeline config ────────────────────────────
# 1B-specific wiring around `make_pipeline_sliding_genai_config` /
# `ensure_pipeline_sliding_oga_files` from conftest.py — see those for the
# decoder-pipeline + sliding_window design notes.
_PIPELINE_KV_LEN = 4096


def _ensure_pipeline_sliding_oga_files(window_size, kv_len):
    """Materialize a 1B sliding-window pipeline directory for (window_size, kv_len)."""
    _ensure_model()
    _ensure_oga_files()
    pipeline_dir = (
        REPO_ROOT / "models" / f"Llama-3.2-1B-Instruct-Pipeline-p{window_size}m{kv_len}"
    )
    config_dict = make_pipeline_sliding_genai_config(
        window_size=window_size,
        kv_len=kv_len,
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        num_attention_heads=32,
        head_dim=HEAD_DIM,
        hidden_size=2048,
        vocab_size=VOCAB_SIZE,
        bos_token_id=BOS_TOKEN,
        eos_token_ids=[128001, 128008, 128009],
        pad_token_id=128001,
        has_position_ids=False,
    )
    return ensure_pipeline_sliding_oga_files(
        parent_dir=_MODEL_DIR,
        pipeline_dir=pipeline_dir,
        onnx_file=_ONNX_FILE,
        data_file=_DATA_FILE,
        tokenizer_files=_OGA_TOKENIZER_FILES,
        window_size=window_size,
        kv_len=kv_len,
        config_dict=config_dict,
    )
