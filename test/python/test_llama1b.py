#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Compare single-token decode latency: MorphiZen EP vs DirectML vs CPU.

Model: Llama-3.2-1B-Instruct (q4f16) — 16 layers, 8 KV heads, head_dim 64.
Uses IOBinding with shared KV cache buffers (past==present) to match OGA behavior.
"""

import onnxruntime as ort
import pytest

from conftest import (
    REPO_ROOT,
    LlamaModelConfig,
    compare_outputs,
    download,
    fix_shapes,
    get_amd_dml_providers,
    make_llama_inputs,
    register_morphizen_ep,
    report,
    run_timed,
    run_timed_iobinding,
)

# ── Model config ─────────────────────────────────────────────────────────────

_MODEL_DIR = REPO_ROOT / "models" / "Llama-3.2-1B-Instruct"
_ONNX_FILE = "model_q4f16.onnx"
_DATA_FILE = "model_q4f16.onnx_data"
_FIXED_FILE = "model_q4f16_fixed_kv128.onnx"
_HF_BASE = (
    "https://huggingface.co/onnx-community/Llama-3.2-1B-Instruct-ONNX/resolve/main/onnx"
)

_CFG = LlamaModelConfig(
    num_kv_layers=16,
    num_kv_heads=8,
    head_dim=64,
    max_seq_len=128,
)
_DIM_MAP = {
    "batch_size": 1,
    "sequence_length": 1,
    "past_sequence_length": _CFG.max_seq_len,
    "total_sequence_length": _CFG.max_seq_len,
}


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def fixed_model_path():
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = _MODEL_DIR / _ONNX_FILE
    data_path = _MODEL_DIR / _DATA_FILE
    fixed_path = _MODEL_DIR / _FIXED_FILE

    if not onnx_path.exists():
        download(f"{_HF_BASE}/{_ONNX_FILE}", onnx_path)
    if not data_path.exists():
        download(f"{_HF_BASE}/{_DATA_FILE}", data_path)
    if not fixed_path.exists():
        fix_shapes(onnx_path, fixed_path, _DIM_MAP)

    return str(fixed_path)


# ── Tests ────────────────────────────────────────────────────────────────────


class TestLlama1BPerformance:
    def test_dml_inference(self, fixed_model_path):
        """Single-token decode latency — DirectML EP (AMD GPU)."""
        providers = get_amd_dml_providers()
        if providers is None:
            pytest.skip("DmlExecutionProvider not available")

        sess = ort.InferenceSession(fixed_model_path, providers=providers)
        times = run_timed(sess, make_llama_inputs(_CFG))
        report("DirectML", times)

    def test_morphizen_ep_inference(self, fixed_model_path, repo_root):
        """Single-token decode latency — MorphiZen (HIP DNN) EP."""
        devices = register_morphizen_ep(repo_root)
        if not devices:
            pytest.skip("MorphiZen EP not found — run build.py first")

        so = ort.SessionOptions()
        so.add_provider_for_devices(devices, {})
        sess = ort.InferenceSession(fixed_model_path, sess_options=so)

        times = run_timed_iobinding(sess, make_llama_inputs(_CFG), _CFG)
        report("MorphiZen EP", times)

    def test_morphizen_ep_accuracy(self, fixed_model_path, repo_root):
        """MorphiZen EP output accuracy vs CPU reference."""
        devices = register_morphizen_ep(repo_root)
        if not devices:
            pytest.skip("MorphiZen EP not found — run build.py first")

        inputs = make_llama_inputs(_CFG)

        cpu_sess = ort.InferenceSession(
            fixed_model_path, providers=["CPUExecutionProvider"]
        )
        ref = cpu_sess.run(None, inputs)
        output_names = [o.name for o in cpu_sess.get_outputs()]

        so = ort.SessionOptions()
        so.add_provider_for_devices(devices, {})
        ep_sess = ort.InferenceSession(fixed_model_path, sess_options=so)
        test = ep_sess.run(None, inputs)

        ok, _ = compare_outputs(ref, test, output_names, "MorphiZen EP (1B)")
        assert ok, "MorphiZen EP accuracy check failed — see report above"

    def test_cpu_inference(self, fixed_model_path):
        """Single-token decode latency — CPU baseline."""
        sess = ort.InferenceSession(
            fixed_model_path, providers=["CPUExecutionProvider"]
        )
        times = run_timed(sess, make_llama_inputs(_CFG))
        report("CPU (baseline)", times)
