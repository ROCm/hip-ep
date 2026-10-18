#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Qwen 27B same-generator 4K prefill, decode, append, decode smoke test."""

from pathlib import Path
import ctypes
import hashlib
import json
import os
import sys
import time

root = Path(os.environ["TQ_WORK_ROOT"])
base = Path(os.environ["TQ_INTEGRATION_INSTALL"])
dense = os.environ.get("TQ_MULTITURN_DENSE_CONTROL") == "1"
results = Path(os.environ["TQ_APPEND_RESULTS"])
out = results / ("qwen-ring" if dense else "qwen")
carrier_width, row_bytes = (256, 512) if dense else (66, 132)
control = json.loads((results / "qwen/receipt.json").read_text()) if dense else None
out.mkdir(parents=True, exist_ok=True)
sdk = Path(sys.base_prefix) / "Lib/site-packages/_rocm_sdk_devel"
core = Path(sys.base_prefix) / "Lib/site-packages/_rocm_sdk_core"
handles = [
    os.add_dll_directory(str(p)) for p in (base / "bin", sdk / "bin", core / "bin")
]
import numpy as np  # noqa: E402 - configure Windows DLL paths first
import onnxruntime_genai as og  # noqa: E402 - configure Windows DLL paths first
import psutil  # noqa: E402 - configure Windows DLL paths first

source = root / "models/qwen-original"
model_dir = out / "model"
model_dir.mkdir(exist_ok=True)
download = json.loads((root / "results/windows-qwen-download.json").read_text())
assert download["status"] == "passed"
for row in download["verified"]:
    p = source / row["file"]
    if p.name == "genai_config.json":
        continue
    target = model_dir / p.name
    if not target.exists():
        os.link(p, target)
    assert os.path.samefile(p, target)
cfg = json.loads((source / "genai_config.json").read_text())
capacity = 4608
cfg["model"]["context_length"] = capacity
cfg["search"].update(
    max_length=capacity, min_length=capacity, past_present_share_buffer=True
)
cfg["search"].pop("chunk_size", None)
for component in ("decoder", "embedding"):
    cfg["model"][component]["session_options"]["provider_options"] = [
        {
            "AMDGPU": {
                "profile": "hip",
                "config_file": str(base / "bin/morphizen_config.json"),
            }
        }
    ]
cfg["model"]["decoder"]["session_options"]["session.disable_cpu_ep_fallback"] = "1"
if not dense:
    cfg["model"]["decoder"]["session_options"]["provider_options"][0]["AMDGPU"].update(
        kvCacheQuantizationBits="4",
        kv_cache_codec="tq4_mse_qr_v1",
        kv_cache_codec_file=str(root / "reference/codec/codec.bin"),
    )
(model_dir / "genai_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
hip = ctypes.CDLL(str(core / "bin/amdhip64_7.dll"))


def sync():
    assert hip.hipDeviceSynchronize() == 0


report = dict(
    status="loading",
    scope="Same-generator multi-turn execution and matching-prefix logits; diagnostic timings, not a task-quality benchmark",
    kv_format="FP16" if dense else "TurboQuant 4-bit",
    comparisons=[],
    genai_module=og.__file__,
    initial_tokens=4096,
    append_lengths=[33, 129],
    capacity=capacity,
    turns=[],
)


def save():
    (out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")


save()
try:
    model = og.Model(str(model_dir))
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=capacity, min_length=capacity, do_sample=False)
    gen = og.Generator(model, params)
    paths = {
        Path(m.path)
        for m in psutil.Process().memory_maps()
        if Path(m.path).name.lower() in ("hipgpu.dll", "custom_kernels_gfx1151.dll")
    }
    assert {p.name.lower() for p in paths} == {
        "hipgpu.dll",
        "custom_kernels_gfx1151.dll",
    }
    assert all(p.parent.resolve() == (base / "bin").resolve() for p in paths)
    report["libraries"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
    }
    rng = np.random.default_rng(20261014)
    initial = rng.integers(100, 20000, size=65536, dtype=np.int32)[:4096].copy()
    for turn, tokens in enumerate(
        (
            initial,
            rng.integers(100, 20000, size=33, dtype=np.int32),
            rng.integers(100, 20000, size=129, dtype=np.int32),
        )
    ):
        old_len = 0 if turn == 0 else len(gen.get_sequence(0))
        old = {}
        if turn:
            for layer in range(3, 64, 4):
                for kind in ("key", "value"):
                    name = f"present.{layer}.{kind}"
                    x = gen.get_output(name)
                    assert x.shape == (1, 4, capacity, carrier_width), (name, x.shape)
                    old[name] = (
                        x.view(np.uint8)
                        .reshape(4, capacity, row_bytes)[:, :old_len]
                        .copy()
                    )
        sync()
        start = time.perf_counter()
        gen.append_tokens(tokens)
        logits = np.asarray(gen.get_logits()).copy()
        sync()
        prefill_seconds = time.perf_counter() - start
        assert np.isfinite(logits).all()
        assert len(gen.get_sequence(0)) == old_len + len(tokens)
        np.testing.assert_array_equal(gen.get_sequence(0)[old_len:], tokens)
        for name, before in old.items():
            after = (
                gen.get_output(name)
                .view(np.uint8)
                .reshape(4, capacity, row_bytes)[:, :old_len]
            )
            np.testing.assert_array_equal(after, before)
        np.save(out / f"turn-{turn}-logits.npy", logits)
        last_prefix = gen.get_sequence(0).copy()
        if dense:
            reference = (
                np.load(results / "qwen" / f"turn-{turn}-logits.npy")
                .astype(np.float64)
                .ravel()
            )
            actual = logits.astype(np.float64).ravel()
            cosine = float(
                np.dot(actual, reference)
                / (np.linalg.norm(actual) * np.linalg.norm(reference))
            )
            relative_l2 = float(
                np.linalg.norm(actual - reference) / np.linalg.norm(actual)
            )
            report["comparisons"].append(
                dict(
                    turn=turn,
                    cosine=cosine,
                    relative_l2=relative_l2,
                    passed=cosine >= 0.999 and relative_l2 <= 0.05,
                )
            )
        start = time.perf_counter()
        for step in range(8):
            if dense:
                # Feed the TQ run's choices so both models see exactly the same history.
                gen.append_tokens(
                    np.array(
                        [control["turns"][turn]["generated_tokens"][step]], np.int32
                    )
                )
            else:
                gen.generate_next_token()
            assert np.isfinite(np.asarray(gen.get_logits())).all()
        sync()
        row = dict(
            turn=turn,
            past=old_len,
            prompt_tokens=len(tokens),
            prefill_seconds=prefill_seconds,
            decode_seconds=time.perf_counter() - start,
            generated_tokens=gen.get_sequence(0)[-8:].tolist(),
            old_packed_rows_unchanged=True,
            finite_logits=True,
        )
        report["turns"].append(row)
        report["status"] = "running"
        save()
        print(json.dumps(row), flush=True)
    if dense:
        fresh = og.Generator(model, params)
        fresh.append_tokens(last_prefix)
        fresh_logits = np.asarray(fresh.get_logits()).astype(np.float64).ravel()
        actual = np.load(out / "turn-2-logits.npy").astype(np.float64).ravel()
        cosine = float(
            np.dot(actual, fresh_logits)
            / (np.linalg.norm(actual) * np.linalg.norm(fresh_logits))
        )
        relative_l2 = float(
            np.linalg.norm(actual - fresh_logits) / np.linalg.norm(fresh_logits)
        )
        report["dense_append_vs_fresh"] = dict(
            cosine=cosine,
            relative_l2=relative_l2,
            passed=cosine >= 0.999 and relative_l2 <= 0.05,
        )
        del fresh
        report["quality_gates_passed"] = (
            all(x["passed"] for x in report["comparisons"])
            and report["dense_append_vs_fresh"]["passed"]
        )
    report["status"] = "passed"
    save()
    if dense:
        assert report["quality_gates_passed"], report
except BaseException as e:
    report.update(status="failed", error=repr(e))
    save()
    raise
