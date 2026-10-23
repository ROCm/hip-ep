#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""256K full-model decode from seeded random KV/recurrent history, no prefill."""

from pathlib import Path
import argparse
import ctypes as c
import hashlib
import json
import os
import sys
import time
import gc

p = argparse.ArgumentParser()
p.add_argument("--turboquant", action="store_true")
args = p.parse_args()
ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get("TQ_PROFILE_INSTALL", ROOT / "install/turboquant"))
CORE = Path(sys.base_prefix) / "Lib/site-packages/_rocm_sdk_core"
DEV = Path(sys.base_prefix) / "Lib/site-packages/_rocm_sdk_devel"
handles = [
    os.add_dll_directory(str(d)) for d in (BASE / "bin", CORE / "bin", DEV / "bin")
]
import numpy as np  # noqa: E402 - initialize Windows DLL search paths first
import onnxruntime as ort  # noqa: E402 - initialize Windows DLL search paths first
import psutil  # noqa: E402 - initialize Windows DLL search paths first
from windows_memory import Memory  # noqa: E402 - initialize Windows DLL search paths first

hip = c.CDLL(str(CORE / "bin/amdhip64_7.dll"))
ep = c.CDLL(str(BASE / "bin/hipgpu.dll"))
memory = Memory(ROOT, hip, ep)
ptr = c.c_void_p
hip.hipMemcpy.argtypes = [ptr, ptr, c.c_size_t, c.c_int]
hip.hipMalloc.argtypes = [c.POINTER(ptr), c.c_size_t]
hip.hipFree.argtypes = [ptr]


def check(rc):
    assert rc == 0, rc


def sync():
    check(hip.hipDeviceSynchronize())


def empty(shape, dtype):
    return ort.OrtValue.ortvalue_from_shape_and_type(
        list(shape), dtype, device_type="gpu", vendor_id=0x1002
    )


def gpu(a):
    return ort.OrtValue.ortvalue_from_numpy(
        np.ascontiguousarray(a), device_type="gpu", vendor_id=0x1002
    )


capacity = 262144
initial = 262008
warmup = 8
steps = 128
variant = os.environ.get("TQ_PROFILE_VARIANT", "tq4" if args.turboquant else "ring")
work = ROOT / "results/profile" / variant / "262144-synthetic"
work.mkdir(parents=True, exist_ok=True)
report = {
    "status": "loading",
    "variant": variant,
    "protocol": "Full embedding/decoder, synthetic cache, no prefill",
    "capacity": capacity,
    "initial_history": initial,
    "warmup_steps": warmup,
    "measured_steps": steps,
    "seed": 20261017,
    "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "memory_scope": "Exact live user HIP device plus mapped-host allocation payload, excluding driver-internal memory and CPU interpreter",
    "pid": os.getpid(),
}


def save():
    (work / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")


save()
try:
    ort.register_execution_provider_library(
        "AMDGPUExecutionProvider", str(BASE / "bin/hipgpu.dll")
    )
    devices = [
        x for x in ort.get_ep_devices() if x.ep_name == "AMDGPUExecutionProvider"
    ]
    assert devices

    def session(name):
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        providers = {
            "profile": "hip",
            "config_file": str(BASE / "bin/morphizen_config.json"),
        }
        if args.turboquant and name == "text.onnx":
            providers.update(
                kvCacheQuantizationBits="4",
                kv_cache_codec="tq4_mse_qr_v1",
                kv_cache_codec_file=str(ROOT / "reference/codec/codec.bin"),
            )
        options.add_provider_for_devices(devices, providers)
        return ort.InferenceSession(
            str(ROOT / "models/qwen-original" / name), sess_options=options
        )

    emb = session("embedding.onnx")
    dec = session("text.onnx")
    sync()
    blobs = [
        x for x in memory.allocations() if not x["host"] and x["bytes"] > 2_000_000_000
    ]
    assert sorted(x["bytes"] for x in blobs) == [2542796872, 13384823360], blobs
    report["weights_bytes"] = sum(x["bytes"] for x in blobs)
    paths = {
        Path(m.path)
        for m in psutil.Process().memory_maps()
        if Path(m.path).name.lower() in ("hipgpu.dll", "custom_kernels_gfx1151.dll")
    }
    assert len(paths) == 2 and all(
        x.parent.resolve() == (BASE / "bin").resolve() for x in paths
    )
    report["loaded_libraries"] = {
        str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in paths
    }
    print("SESSIONS_LOADED", variant, flush=True)
    lib = c.CDLL(str(BASE / "bin/custom_kernels_gfx1151.dll"))
    lib.hip_tq4_encode_append.argtypes = [ptr] * 5 + [c.c_int] * 3 + [ptr] * 3
    temporary = []

    def device(a):
        v = ptr()
        check(hip.hipMalloc(c.byref(v), a.nbytes))
        temporary.append(v)
        check(hip.hipMemcpy(v, a.ctypes.data, a.nbytes, 1))
        return v

    if args.turboquant:
        report["codec_sha256"] = hashlib.sha256(
            (ROOT / "reference/codec/codec.bin").read_bytes()
        ).hexdigest()
        const = np.load(ROOT / "reference/codec/constants.npz")
        rotation, cbk, cbv = [
            device(np.ascontiguousarray(const[key]))
            for key in ("rotation", "cb_k", "cb_v")
        ]
        # Bound compression staging, including any HIP runtime retention after
        # free, to two 8 MiB buffers rather than two full 512 MiB caches.
        chunk_tokens = 4096
        staging_bytes = chunk_tokens * 4 * 256 * 2
        kd, vd = ptr(), ptr()
        for v in (kd, vd):
            check(hip.hipMalloc(c.byref(v), staging_bytes))
            temporary.append(v)
        report["initialization_device_staging_bytes"] = 2 * staging_bytes
    io = dec.io_binding()
    eio = emb.io_binding()
    states = {}
    kv = {}
    rng = np.random.default_rng(20261017)
    digest = hashlib.sha256()

    def seeded(shape, scale):
        count = int(np.prod(shape))
        a = np.empty(count, np.float16)
        for offset in range(0, count, 2**22):
            chunk = (
                rng.standard_normal(min(2**22, count - offset), dtype=np.float32)
                * scale
            ).astype(np.float16)
            a[offset : offset + len(chunk)] = chunk
        digest.update(a.tobytes())
        return a.reshape(shape)

    pending_keys = {}
    for spec in dec.get_inputs():
        name = spec.name
        if not name.startswith("past_key_values."):
            continue
        if name.endswith((".key", ".value")):
            val = empty((1, 4, capacity, 66 if args.turboquant else 256), np.float16)
            dense = seeded((4, capacity, 256), 0.05)
            if args.turboquant:
                if name.endswith(".key"):
                    # ONNX lists all 16 K inputs before all 16 V inputs.
                    # Spill seeded K to task-owned files instead of retaining
                    # eight GiB of extra host RAM or changing RNG input order.
                    layer = int(name.split(".")[1])
                    path = work / f"tmp-key-{layer}.bin"
                    dense.tofile(path)
                    pending_keys[name] = path
                else:
                    key_name = name.rsplit(".", 1)[0] + ".key"
                    path = pending_keys.pop(key_name)
                    kdense = np.fromfile(path, dtype=np.float16).reshape(
                        4, capacity, 256
                    )
                    for offset in range(0, capacity, chunk_tokens):
                        count = min(chunk_tokens, capacity - offset)
                        kt = np.ascontiguousarray(
                            kdense[:, offset : offset + count].transpose(1, 0, 2)
                        )
                        vt = np.ascontiguousarray(
                            dense[:, offset : offset + count].transpose(1, 0, 2)
                        )
                        check(hip.hipMemcpy(kd, kt.ctypes.data, kt.nbytes, 1))
                        check(hip.hipMemcpy(vd, vt.ctypes.data, vt.nbytes, 1))
                        check(
                            lib.hip_tq4_encode_append(
                                None,
                                kd,
                                vd,
                                kv[key_name].data_ptr(),
                                val.data_ptr(),
                                count,
                                offset,
                                capacity,
                                rotation,
                                cbk,
                                cbv,
                            )
                        )
                        sync()
                    del kdense
                    path.unlink()
                    del kt, vt
            else:
                check(hip.hipMemcpy(val.data_ptr(), dense.ctypes.data, dense.nbytes, 1))
            kv[name] = val
            io.bind_ortvalue_input(name, val)
            io.bind_ortvalue_output(name.replace("past_key_values.", "present."), val)
            del dense
            print("CACHE", name, flush=True)
        else:
            shape = [1 if isinstance(x, str) else x for x in spec.shape]
            a = seeded(shape, 0.01)
            states[name] = [gpu(a), empty(shape, np.float16)]
    for v in temporary:
        check(hip.hipFree(v))
    temporary.clear()
    gc.collect()
    sync()
    assert not pending_keys and len(kv) == 32 and len(states) == 96
    # Freeze the Windows-generated source, and require both variants to agree.
    # Same seed/distribution does not establish cross-platform byte identity.
    archived_hash = "63df732c563724ce0ef182fa3ce1dfb8c8d79f0fe68e7f7a8049ec4fca3a40c0"
    windows_hash = "f6b388c5a5d40cc7c24c670ebbb0e1bc906eaa039f2de8b98dafc23209ccc2d4"
    assert digest.hexdigest() == windows_hash, digest.hexdigest()
    report.update(archived_source_sha256=archived_hash, source_matches_archived=False)
    if args.turboquant:
        control = json.loads(
            (ROOT / "results/profile/ring/262144-synthetic/receipt.json").read_text()
        )
        assert (
            control["status"] == "passed"
            and control["source_history_sha256"] == digest.hexdigest()
        )
    report.update(
        source_history_sha256=digest.hexdigest(),
        kv_allocation=memory.cache(capacity, args.turboquant),
        recurrent_state_bytes=int(
            sum(np.prod(x[0].shape()) * 4 for x in states.values())
        ),
    )
    embedded = empty((1, 1, 5120), np.float16)
    logits = empty((1, 1, 248320), np.float16)
    ids = gpu(np.array([[317]], np.int64))
    images = gpu(np.zeros((0, 5120), np.float16))
    pos = gpu(np.full((3, 1, 1), initial, np.int64))
    eio.bind_ortvalue_input("input_ids", ids)
    eio.bind_ortvalue_input("image_features", images)
    eio.bind_ortvalue_output("inputs_embeds", embedded)
    io.bind_ortvalue_input("inputs_embeds", embedded)
    io.bind_ortvalue_input("position_ids", pos)
    io.bind_ortvalue_output("logits", logits)
    mask = np.ones((1, capacity), np.int64)
    token = 317
    history = initial

    def step(i, capture=False):
        global token, history
        ids.update_inplace(np.array([[token]], np.int64))
        pos.update_inplace(np.full((3, 1, 1), history, np.int64))
        io.bind_cpu_input("attention_mask", mask[:, : history + 1])
        for name, pair in states.items():
            io.bind_ortvalue_input(name, pair[i % 2])
            io.bind_ortvalue_output(
                name.replace("past_key_values.", "present."), pair[1 - i % 2]
            )
        emb.run_with_iobinding(eio)
        dec.run_with_iobinding(io)
        sync()
        result = logits.numpy()
        assert np.isfinite(result).all()
        if capture:
            np.save(work / "first-logits.npy", result)
            if args.turboquant:
                ref = (
                    np.load(
                        ROOT / "results/profile/ring/262144-synthetic/first-logits.npy"
                    )
                    .astype(np.float64)
                    .ravel()
                )
                x = result.astype(np.float64).ravel()
                cosine = float(x @ ref / (np.linalg.norm(x) * np.linalg.norm(ref)))
                l2 = float(np.linalg.norm(x - ref) / np.linalg.norm(ref))
                report["first_step_quality"] = {
                    "cosine": cosine,
                    "relative_l2": l2,
                    "passed": cosine >= 0.999 and l2 <= 0.05,
                    "input_token": 317,
                }
        token = int(np.argmax(result))
        history += 1
        return token

    report.update(status="warming_up")
    save()
    print("CACHE_READY", report["source_history_sha256"], flush=True)
    for i in range(warmup):
        step(i, capture=i == 0)
    report.update(status="measuring", measured_history_start=history)
    save()
    sync()
    memory.read(True)
    times = []
    generated = []
    start = time.perf_counter()
    for j in range(steps):
        tick = time.perf_counter()
        generated.append(step(warmup + j))
        times.append(time.perf_counter() - tick)
    end = time.perf_counter()
    stats = memory.read()
    assert history == capacity and stats["unknown_frees"] == stats["failures"] == 0
    report.update(
        status="passed",
        history_end=history,
        decode_seconds=end - start,
        decode_tokens_per_second=steps / (end - start),
        step_seconds=times,
        generated_token_ids=generated,
        all_logits_finite=True,
        decode_memory=stats,
    )
    save()
    print("DONE", variant, report["decode_tokens_per_second"], stats, flush=True)
except BaseException as exc:
    report.update(status="failed", error=repr(exc), failure_memory=memory.read())
    save()
    raise
