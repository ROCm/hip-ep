#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Qwen Ring/TQ full-request profile with exact user HIP allocation accounting."""

from pathlib import Path
import argparse
import ctypes
import gc
import hashlib
import importlib.metadata
import json
import os
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--length", type=int, required=True)
p.add_argument("--turboquant", action="store_true")
p.add_argument("--repetitions", type=int, default=3)
a = p.parse_args()
n = a.length
capacity = n + 128
ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get("TQ_PROFILE_INSTALL", ROOT / "install/turboquant"))
SDK = Path(sys.base_prefix) / "Lib/site-packages/_rocm_sdk_devel"
CORE = Path(sys.base_prefix) / "Lib/site-packages/_rocm_sdk_core"
handles = [
    os.add_dll_directory(str(d)) for d in (BASE / "bin", SDK / "bin", CORE / "bin")
]
import numpy as np  # noqa: E402 - initialize Windows DLL search paths first
import onnxruntime_genai as og  # noqa: E402 - initialize Windows DLL search paths first
import psutil  # noqa: E402 - initialize Windows DLL search paths first
from windows_memory import Memory  # noqa: E402 - initialize Windows DLL search paths first

hip = ctypes.CDLL(str(CORE / "bin/amdhip64_7.dll"))
ep = ctypes.CDLL(str(BASE / "bin/hipgpu.dll"))
memory = Memory(ROOT, hip, ep)


def sync():
    rc = hip.hipDeviceSynchronize()
    if rc:
        raise RuntimeError(f"HIP synchronization: {rc}")


variant = os.environ.get("TQ_PROFILE_VARIANT", "tq4" if a.turboquant else "ring")
work = ROOT / "results/profile" / variant / str(n)
work.mkdir(parents=True, exist_ok=True)
model_dir = ROOT / "models" / f"profile-{variant}-{n}"
model_dir.mkdir(parents=True, exist_ok=True)
source = ROOT / "models/qwen-original"
download = json.loads((ROOT / "results/windows-qwen-download.json").read_text())
assert download["status"] == "passed"
for record in download["verified"]:
    src = source / record["file"]
    dst = model_dir / src.name
    if src.name == "genai_config.json":
        continue
    if not dst.exists():
        os.link(src, dst)
    assert os.path.samefile(src, dst)
cfg = json.loads((source / "genai_config.json").read_text())
cfg["model"]["context_length"] = capacity
cfg["search"].update(max_length=capacity, past_present_share_buffer=True)
cfg["search"].pop("chunk_size", None)
for component in ("decoder", "embedding"):
    cfg["model"][component]["session_options"]["provider_options"] = [
        {
            "AMDGPU": {
                "profile": "hip",
                "config_file": str(BASE / "bin/morphizen_config.json"),
            }
        }
    ]
cfg["model"]["decoder"]["session_options"]["session.disable_cpu_ep_fallback"] = "1"
if a.turboquant:
    cfg["model"]["decoder"]["session_options"]["provider_options"][0]["AMDGPU"].update(
        kvCacheQuantizationBits="4",
        kv_cache_codec="tq4_mse_qr_v1",
        kv_cache_codec_file=str(ROOT / "reference/codec/codec.bin"),
    )
(model_dir / "genai_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
report = {
    "status": "loading",
    "variant": variant,
    "input_tokens": n,
    "capacity": capacity,
    "generated_tokens": 128,
    "pid": os.getpid(),
    "weights_format": "INT4 group128",
    "cache_format": "tq4_mse_qr_v1" if a.turboquant else "FP16",
    "model_revision": "9b37cf56a001720ba7ce98ff80ea651b02e5931b",
    "records": [],
    "memory_scope": "Live successful hipMalloc + hipHostMalloc requested payload in hipgpu.dll, including retained pools; exact allocation-event phase peaks. Excludes driver-internal allocations, page rounding, CPU interpreter, and WDDM residency.",
    "memory_counter_imports": memory.hooked,
    "quality": [],
    "versions": {
        x: importlib.metadata.version(x)
        for x in ("onnxruntime", "onnxruntime-genai", "numpy")
    },
    "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "config_sha256": hashlib.sha256(
        (model_dir / "genai_config.json").read_bytes()
    ).hexdigest(),
}


def save():
    (work / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")


def capture(generator, name, prefix):
    x = np.asarray(generator.get_logits(), np.float32).copy()
    sync()
    assert np.isfinite(x).all(), name
    np.save(work / (name + ".npy"), x)
    if not a.turboquant:
        return
    refdir = ROOT / "results/profile/ring" / str(n)
    ref = json.loads((refdir / "receipt.json").read_text())
    reference_platform = "Windows current dense control"
    if ref["status"] != "passed" and n == 131072:
        refdir = Path(os.environ["TQ_ARCHIVED_RING_128K"])
        ref = json.loads((refdir / "probe.json").read_text())
        reference_platform = (
            "Archived lx01 Ring-KV diagnostic, current dense control did not complete"
        )
        assert ref["input_sha256"] == report["prompt_sha256"]
    else:
        assert (
            ref["status"] == "passed"
            and ref["prompt_sha256"] == report["prompt_sha256"]
        )
    assert prefix == ref["records"][0]["generated_token_ids"][: len(prefix)], (
        "Unaligned first-decode prefix"
    )
    y = np.load(refdir / (name + ".npy")).astype(np.float64).ravel()
    x = x.astype(np.float64).ravel()
    cosine = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
    l2 = float(np.linalg.norm(x - y) / np.linalg.norm(y))
    report["quality"].append(
        {
            "phase": name,
            "prefix": prefix,
            "cosine": cosine,
            "relative_l2": l2,
            "reference_platform": reference_platform,
            "reference_sha256": hashlib.sha256(
                (refdir / (name + ".npy")).read_bytes()
            ).hexdigest(),
            "passed": cosine >= 0.999 and l2 <= 0.05,
        }
    )
    save()


save()
try:
    tick = time.perf_counter()
    model = og.Model(str(model_dir))
    sync()
    report["model_load_seconds"] = time.perf_counter() - tick
    loaded = memory.allocations()
    # Runtime allocates each compiled graph's constants in one device blob.
    # Verify the two known model blobs, rather than calling all load memory weights.
    blobs = [x for x in loaded if not x["host"] and x["bytes"] > 2_000_000_000]
    report["parameter_allocations"] = blobs
    report["weights_bytes"] = sum(x["bytes"] for x in blobs)
    assert sorted(x["bytes"] for x in blobs) == [2542796872, 13384823360], blobs
    report["after_model_load"] = memory.read()
    paths = {
        Path(m.path)
        for m in psutil.Process().memory_maps()
        if Path(m.path).name.lower() in ("hipgpu.dll", "custom_kernels_gfx1151.dll")
    }
    assert len(paths) == 2 and all(
        p.parent.resolve() == (BASE / "bin").resolve() for p in paths
    ), paths
    report["loaded_libraries"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
    }
    report["loaded_dlls"] = [
        m.path
        for m in psutil.Process().memory_maps()
        if m.path.lower().endswith(".dll")
    ]
    tokens = (
        np.random.default_rng(20261014)
        .integers(100, 20000, size=max(65536, n), dtype=np.int32)[:n]
        .copy()
    )
    np.save(work / "prompt-tokens.npy", tokens)
    report["prompt_sha256"] = hashlib.sha256(tokens.tobytes()).hexdigest()
    print(
        "MODEL_LOADED",
        variant,
        n,
        report["model_load_seconds"],
        report["after_model_load"],
        flush=True,
    )
    expected = None
    for repetition in range(a.repetitions + 1):
        phase = "probe" if repetition == 0 else "measurement"
        params = og.GeneratorParams(model)
        params.set_search_options(
            max_length=capacity,
            min_length=capacity,
            batch_size=1,
            num_beams=1,
            do_sample=False,
            top_k=1,
            top_p=1.0,
        )
        generator = og.Generator(model, params)
        sync()
        cache = memory.cache(capacity, a.turboquant)
        report.update(
            status="running",
            active_phase="prefill",
            active_repetition=repetition,
            kv_allocation=cache,
        )
        save()
        print(
            "REQUEST_START",
            phase,
            repetition,
            "available_system_GiB",
            psutil.virtual_memory().available / 2**30,
            flush=True,
        )
        before = memory.read(True)
        start = time.perf_counter()
        generator.append_tokens(tokens)
        sync()
        prefill_end = time.perf_counter()
        prefill_memory = memory.read(True)
        if repetition == 0:
            capture(generator, "prefill-logits", [])
        generator.generate_next_token()
        sync()
        first_end = time.perf_counter()
        report["active_phase"] = "decode"
        save()
        if repetition == 0:
            capture(
                generator,
                "first-decode-logits",
                np.asarray(generator.get_sequence(0))[n:].tolist(),
            )
        for index in range(127):
            assert not generator.is_done()
            if repetition == 0 and index == 126:
                assert np.isfinite(np.asarray(generator.get_logits())).all()
            generator.generate_next_token()
        sync()
        end = time.perf_counter()
        decode_memory = memory.read(True)
        sequence = np.asarray(generator.get_sequence(0)).copy()
        assert len(sequence) == capacity and np.array_equal(sequence[:n], tokens)
        generated = sequence[n:].tolist()
        if expected is None:
            expected = generated
        else:
            assert expected == generated, "Request not repeatable"
        assert decode_memory["failures"] == decode_memory["unknown_frees"] == 0, (
            decode_memory
        )
        row = {
            "phase": phase,
            "repetition": repetition,
            "prefill_seconds": prefill_end - start,
            "ttft_seconds": first_end - start,
            "decode_seconds": end - first_end,
            "decode_tokens_per_second": 127 / (end - first_end),
            "total_seconds": end - start,
            "generated_token_ids": generated,
            "memory_before": before,
            "prefill_memory": prefill_memory,
            "decode_memory": decode_memory,
        }
        report["records"].append(row)
        save()
        print(
            "REQUEST_DONE",
            json.dumps({k: v for k, v in row.items() if k != "generated_token_ids"}),
            flush=True,
        )
        del generator, params
        gc.collect()
        sync()
    report.update(
        status="passed",
        quality_passed=all(x["passed"] for x in report["quality"]),
        active_phase=None,
    )
    save()
except BaseException as exc:
    report.update(status="failed", error=repr(exc), failure_memory=memory.read())
    save()
    raise
