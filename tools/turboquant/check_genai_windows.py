#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Windows GenAI packed-cache regression; TQ_WORK_ROOT names the test workspace."""

from pathlib import Path
import os
import json
import sys
import subprocess

ROOT = Path(os.environ["TQ_WORK_ROOT"])
BASE = ROOT / "install/turboquant"
SDK = Path(sys.base_prefix) / "Lib/site-packages/_rocm_sdk_devel"
CORE = Path(sys.base_prefix) / "Lib/site-packages/_rocm_sdk_core"
paths = [BASE / "bin", SDK / "bin", CORE / "bin"]
if os.environ.get("TQ_INTEGRATION_CHILD") != "1":
    work = ROOT / "results/integration"
    work.mkdir(parents=True, exist_ok=True)
    assert (BASE / "bin/hipgpu.dll").is_file(), "Build and install hip-ep first"
    env = dict(
        os.environ,
        TQ_INTEGRATION_CHILD="1",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        HIPDNN_EP_STRICT="1",
        HIPDNN_EP_DEBUG="1",
        THEROCK_DIST=str(SDK),
        AMDGPU_EP_PATH=str(BASE / "bin/hipgpu.dll"),
        PATH=os.pathsep.join(map(str, paths)) + os.pathsep + os.environ["PATH"],
        LIB=os.pathsep.join(
            [str(SDK / "lib"), str(BASE / "lib"), os.environ.get("LIB", "")]
        ),
    )
    with (work / "run.log").open("w") as f:
        p = subprocess.run(
            [sys.executable, __file__],
            cwd=work,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    print((work / "run.log").read_text(errors="replace")[-8000:])
    sys.exit(p.returncode)
_dll_directories = [os.add_dll_directory(str(p)) for p in paths if p.is_dir()]
import numpy as np  # noqa: E402 - initialize Windows DLL search paths first
import onnx  # noqa: E402 - initialize Windows DLL search paths first
from onnx import helper as h, TensorProto as T, numpy_helper as nh  # noqa: E402 - initialize Windows DLL search paths first
import onnxruntime_genai as og  # noqa: E402 - initialize Windows DLL search paths first

sys.path.insert(0, str(ROOT / "hip-ep/tools/turboquant"))
from reference import attention, encode, unpack_rows  # noqa: E402 - initialize Windows DLL search paths first

rng = np.random.default_rng(227)
qtab = rng.normal(size=(16, 6144)).astype(np.float16)
ktab = rng.normal(size=(16, 1024)).astype(np.float16)
vtab = rng.normal(size=(16, 1024)).astype(np.float16)
const = np.load(ROOT / "reference/codec/constants.npz")
records = []
for codec, prompt_length in (
    ("tq4_mse_qr_v1", 1),
    ("tq4_mse_qr_v1", 4),
    ("invalid", 4),
):
    path = ROOT / "models" / f"integration-{codec}-{prompt_length}"
    path.mkdir(parents=True, exist_ok=True)
    inputs = [
        h.make_tensor_value_info(
            "input_ids", T.INT32, ["batch_size", "sequence_length"]
        ),
        h.make_tensor_value_info(
            "attention_mask", T.INT32, ["batch_size", "total_sequence_length"]
        ),
        h.make_tensor_value_info(
            "past_key_values.3.key",
            T.FLOAT16,
            ["batch_size", 4, "past_sequence_length", "kv_cache_dim"],
        ),
        h.make_tensor_value_info(
            "past_key_values.3.value",
            T.FLOAT16,
            ["batch_size", 4, "past_sequence_length", "kv_cache_dim"],
        ),
    ]
    outputs = [
        h.make_tensor_value_info(
            "logits", T.FLOAT, ["batch_size", "sequence_length", 16]
        ),
        h.make_tensor_value_info(
            "present.3.key",
            T.FLOAT16,
            ["batch_size", 4, "past_sequence_length", "kv_cache_dim"],
        ),
        h.make_tensor_value_info(
            "present.3.value",
            T.FLOAT16,
            ["batch_size", 4, "past_sequence_length", "kv_cache_dim"],
        ),
    ]
    init = [
        nh.from_array(x, n)
        for n, x in [
            ("qtab", qtab),
            ("ktab", ktab),
            ("vtab", vtab),
            ("axis1", np.array([1], np.int64)),
            ("one", np.array(1, np.int32)),
            ("start", np.array([0], np.int64)),
            ("end", np.array([16], np.int64)),
            ("axis2", np.array([2], np.int64)),
        ]
    ]
    nodes = [
        h.make_node("Gather", [n + "tab", "input_ids"], [n], axis=0)
        for n in ("q", "k", "v")
    ]
    nodes += [
        h.make_node("ReduceSum", ["attention_mask", "axis1"], ["lengths"], keepdims=0),
        h.make_node("Sub", ["lengths", "one"], ["seqlens"]),
        h.make_node("ReduceMax", ["lengths"], ["total"], keepdims=0),
        h.make_node(
            "GroupQueryAttention",
            [
                "q",
                "k",
                "v",
                "past_key_values.3.key",
                "past_key_values.3.value",
                "seqlens",
                "total",
            ],
            ["o", "present.3.key", "present.3.value"],
            domain="com.microsoft",
            num_heads=24,
            kv_num_heads=4,
            scale=0.0625,
            do_rotary=0,
        ),
        h.make_node("Slice", ["o", "start", "end", "axis2"], ["logits16"]),
        h.make_node("Cast", ["logits16"], ["logits"], to=T.FLOAT),
    ]
    graph = h.make_graph(nodes, "packed_gqa_integration", inputs, outputs, init)
    model = h.make_model(
        graph,
        opset_imports=[h.make_opsetid("", 18), h.make_opsetid("com.microsoft", 1)],
        ir_version=10,
    )
    onnx.checker.check_model(model)
    onnx.save(model, path / "decoder.onnx")
    provider = {
        "profile": "hip",
        "config_file": str(BASE / "bin/morphizen_config.json"),
        "kvCacheQuantizationBits": "4",
        "kv_cache_codec": codec,
        "kv_cache_codec_file": str(ROOT / "reference/codec/codec.bin"),
    }
    cfg = {
        "model": {
            "type": "llama",
            "context_length": 16,
            "vocab_size": 16,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "decoder": {
                "filename": "decoder.onnx",
                "head_size": 256,
                "hidden_size": 6144,
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
                "num_hidden_layers": 4,
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
                "session_options": {
                    "provider_options": [{"AMDGPU": provider}],
                    "session.disable_cpu_ep_fallback": "1",
                },
            },
        },
        "search": {
            "max_length": 16,
            "past_present_share_buffer": True,
            "num_beams": 1,
            "do_sample": False,
        },
    }
    (path / "genai_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    m = og.Model(str(path))
    p = og.GeneratorParams(m)
    p.set_search_options(max_length=16, min_length=16, do_sample=False)
    g = og.Generator(m, p)
    try:
        g.append_tokens(np.arange(3, 3 + prompt_length, dtype=np.int32)[None])
    except RuntimeError as error:
        if codec != "invalid":
            raise
        assert "inference_compute (allocator) failed with code: -1" in str(error), str(
            error
        )
        records.append({"codec": codec, "rejected": True, "message": str(error)})
        print("EXPECTED_REJECTION", str(error), flush=True)
        g = p = m = None
        continue
    if codec == "invalid":
        raise AssertionError("invalid codec accepted")
    prompt = g.get_sequence(0).copy()
    q = qtab[prompt[-1]].reshape(24, 256)[0].astype(np.float64)
    k = ktab[prompt].reshape(-1, 4, 256)[:, 0].astype(np.float64)
    v = vtab[prompt].reshape(-1, 4, 256)[:, 0].astype(np.float64)
    scores = k @ q * 0.0625
    weights = np.exp(scores - scores.max())
    dense_expected = (weights @ v / weights.sum()).astype(np.float16)[:16]
    np.testing.assert_allclose(
        np.asarray(g.get_logits()).reshape(-1, 16)[-1],
        dense_expected,
        rtol=0.005,
        atol=0.005,
    )
    decode_errors = []
    for _ in range(4):
        g.generate_next_token()
        logits = np.asarray(g.get_logits()).reshape(-1, 16)[-1].copy()
        processed = g.get_sequence(0).copy()
        packed = [
            g.get_output("present.3." + kind)
            .view(np.uint8)
            .reshape(4, 16, 132)[:, : len(processed)]
            .copy()
            for kind in ("key", "value")
        ]
        expected = (
            attention(
                qtab[processed[-1]].reshape(24, 256),
                *packed,
                const["rotation"],
                const["cb_k"],
                const["cb_v"],
                0.0625,
            )
            .astype(np.float16)
            .reshape(-1)[:16]
        )
        np.testing.assert_allclose(logits, expected, rtol=0.005, atol=0.005)
        decode_errors.append(float(np.max(np.abs(logits - expected))))
    processed = g.get_sequence(0)
    for kind, table, cb in [
        ("key", ktab, const["cb_k"]),
        ("value", vtab, const["cb_v"]),
    ]:
        cache = g.get_output("present.3." + kind)
        assert cache.shape == (1, 4, 16, 66), cache.shape
        actual = cache.view(np.uint8).reshape(4, 16, 132)[:, : len(processed)]
        vectors = table[processed].reshape(-1, 4, 256).transpose(1, 0, 2).copy()
        expected = encode(vectors, const["rotation"], cb)
        np.testing.assert_array_equal(actual[..., 4:], expected[..., 4:])
        np.testing.assert_allclose(
            unpack_rows(actual)[1], unpack_rows(expected)[1], rtol=3e-7, atol=2e-6
        )
    records.append(
        {
            "codec": codec,
            "prompt_length": prompt_length,
            "processed_tokens": processed.tolist(),
            "cache_shape": [1, 4, 16, 66],
            "prefill_dense_reference": "passed",
            "decode_reference_max_abs": decode_errors,
            "checked_logit_dimensions": 16,
            "status": "passed",
        }
    )
    print("PASS", records[-1], flush=True)
    del g, p, m
(ROOT / "results/integration/receipt.json").write_text(
    json.dumps({"status": "passed", "records": records}, indent=2) + "\n"
)
