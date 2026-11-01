#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Serial current-build performance campaign. No accuracy artifacts or warmups."""

import argparse
import ctypes as c
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = REPO.parent
INSTALL = ROOT / "install/turboquant"
OUT = ROOT / "results/turboquant-performance"
PYTHON = Path(sys.executable)
MODEL = ROOT / "models/qwen-original"
CODEC = ROOT / "reference/codec/codec.bin"
CONSTANTS = ROOT / "reference/codec/constants.npz"
GENAI_PATH = None
MEMORY_COUNTER = ROOT / "build/windows_memory_counter.dll"
DEV = Path(sys.base_prefix) / "Lib/site-packages/_rocm_sdk_devel"
CORE = Path(sys.base_prefix) / "Lib/site-packages/_rocm_sdk_core"
MODES = ["no-ring", "ring", "turboquant"]
LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]


def save(path, data):
    path.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def power():
    class Power(c.Structure):
        _fields_ = [
            ("ac", c.c_ubyte),
            ("flag", c.c_ubyte),
            ("percent", c.c_ubyte),
            ("reserved", c.c_ubyte),
            ("life", c.c_uint32),
            ("full", c.c_uint32),
        ]

    p = Power()
    assert c.windll.kernel32.GetSystemPowerStatus(c.byref(p)) and p.ac == 1, (
        "AC required"
    )


def environment(mode):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(
            (
                "HIP_GQA_",
                "HIP_EP_GQA_",
                "HIPDNN_GQA_",
                "HIPDNN_EP_GQA_",
                "TQ_",
                "HIPDNN_PREFILL_",
            )
        )
    }
    env.update(
        AMDGPU_EP_PATH=str(INSTALL / "bin/hipgpu.dll"),
        HIPDNN_EP_STRICT="1",
        HIPDNN_EP_DEBUG="0",
        HIPDNN_EP_PERF="0",
        HIPDNN_EP_USE_RING_ATTENTION="0" if mode == "no-ring" else "1",
        THEROCK_DIST=str(DEV),
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
    )
    if GENAI_PATH is not None:
        env["PYTHONPATH"] = str(GENAI_PATH)
    env.pop("KMP_DUPLICATE_LIB_OK", None)
    env["PATH"] = (
        os.pathsep.join(
            map(str, [INSTALL / "bin", DEV / "bin", CORE / "bin", DEV / "lib/llvm/bin"])
        )
        + os.pathsep
        + env["PATH"]
    )
    env["LIB"] = (
        os.pathsep.join(map(str, [DEV / "lib", INSTALL / "lib"]))
        + os.pathsep
        + env.get("LIB", "")
    )
    return env


def libraries():
    handles = [
        os.add_dll_directory(str(d))
        for d in [INSTALL / "bin", DEV / "bin", CORE / "bin"]
    ]
    hip = c.CDLL(str(CORE / "bin/amdhip64_7.dll"))
    lib = c.CDLL(str(INSTALL / "bin/custom_kernels_gfx1151.dll"))
    return handles, hip, lib


def provenance():
    import numpy as np
    import onnxruntime as ort
    import onnxruntime_genai as og

    # Versions are recorded; use the documented pinned stack for comparable results.
    assert os.environ["HIPDNN_EP_DEBUG"] == "0"
    return dict(
        head=subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        python=sys.executable,
        numpy=np.__version__,
        ort=ort.__version__,
        genai=og.__version__,
        binaries={
            p.name: sha(p)
            for p in [
                INSTALL / "bin/hipgpu.dll",
                INSTALL / "bin/custom_kernels_gfx1151.dll",
            ]
        },
        harness_sha256=sha(Path(__file__)),
        warmup_calls=0,
        thread_settings={
            k: os.environ[k] for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"]
        },
    )


def model_worker(args):
    os.environ["LIB"] = (
        os.pathsep.join(map(str, [DEV / "lib", INSTALL / "lib"]))
        + os.pathsep
        + os.environ.get("LIB", "")
    )
    os.environ["HIP_EP_PROFILE_ALLOCATOR_DLL"] = str(MEMORY_COUNTER)
    import numpy as np

    handles, hip, lib = libraries()
    sys.path.insert(0, str(REPO / "tools/turboquant/benchmark"))
    from windows_memory import Memory

    ep = c.CDLL(str(INSTALL / "bin/hipgpu.dll"))
    assert MEMORY_COUNTER.name == "windows_memory_counter.dll"
    memory = Memory(MEMORY_COUNTER.parent.parent, hip, ep)
    import onnxruntime_genai as og
    import psutil

    def sync():
        assert hip.hipDeviceSynchronize() == 0

    out = (
        OUT
        / args.kind
        / (str(args.length) if args.kind == "model" else "8x4096")
        / args.mode
    )
    out.mkdir(parents=True, exist_ok=False)
    turns = 1 if args.kind == "model" else 8
    n = args.length if turns == 1 else 4096
    capacity = turns * (n + 128) + 128
    modeldir = out / "model"
    modeldir.mkdir()
    source = MODEL
    for p in source.iterdir():
        if p.is_file() and p.name != "genai_config.json" and p.suffix != ".part":
            os.link(p, modeldir / p.name)
    cfg = json.loads((source / "genai_config.json").read_text())
    cfg["model"]["context_length"] = capacity
    cfg["search"].update(
        max_length=capacity, min_length=0, past_present_share_buffer=True
    )
    cfg["search"].pop("chunk_size", None)
    for name in ["decoder", "embedding"]:
        sess = cfg["model"][name]["session_options"]
        sess["provider_options"] = [
            {
                "AMDGPU": {
                    "profile": "hip",
                    "config_file": str(INSTALL / "bin/morphizen_config.json"),
                }
            }
        ]
        sess["session.disable_cpu_ep_fallback"] = "1"
    if args.mode == "turboquant":
        cfg["model"]["decoder"]["session_options"]["provider_options"][0][
            "AMDGPU"
        ].update(
            kvCacheQuantizationBits="4",
            kv_cache_codec="tq4_mse_qr_v1",
            kv_cache_codec_file=str(CODEC),
        )
    save(modeldir / "genai_config.json", cfg)
    report = dict(
        status="loading",
        kind=args.kind,
        mode=args.mode,
        provenance=provenance(),
        capacity=capacity,
        prompt_tokens_per_turn=n,
        output_tokens_per_turn=128,
        repetitions=2,
        records=[],
        timing_scope="Synchronized wall time; no tracing, logits readback, cache hashing, warmups or accuracy probes",
    )
    report["memory_scope"] = (
        "Peak live HIP device plus mapped-host requested allocation bytes, including retained EP pools; excludes driver-internal allocations, page rounding, CPU interpreter, and WDDM residency. Same counter as PR1366."
    )
    report["memory_counter_sha256"] = sha(MEMORY_COUNTER)
    report["memory_counter_hooks"] = memory.hooked
    save(out / "receipt.json", report)
    start = time.perf_counter()
    model = og.Model(str(modeldir))
    sync()
    report["model_load_seconds"] = time.perf_counter() - start
    blobs = [
        x for x in memory.allocations() if not x["host"] and x["bytes"] > 2_000_000_000
    ]
    assert sorted(x["bytes"] for x in blobs) == [2542796872, 13384823360], blobs
    report["weights_bytes"] = sum(x["bytes"] for x in blobs)
    report["after_model_load_memory"] = memory.read(True)
    report["loaded_libraries"] = sorted(
        {
            m.path
            for m in psutil.Process().memory_maps()
            if Path(m.path).name.lower()
            in [
                "hipgpu.dll",
                "custom_kernels_gfx1151.dll",
                "onnxruntime.dll",
                "onnxruntime-genai.dll",
                "libomp.dll",
            ]
        }
    )
    rng = np.random.default_rng(20261014)
    prompts = [
        rng.integers(100, 20000, size=max(65536, n), dtype=np.int32)[:n].copy()
        for _ in range(turns)
    ]
    report["prompt_hashes"] = [hashlib.sha256(p.tobytes()).hexdigest() for p in prompts]
    report["status"] = "running"
    save(out / "receipt.json", report)
    for repetition in range(2):
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
        gen = og.Generator(model, params)
        sync()
        cache = memory.cache(capacity, args.mode == "turboquant")
        report["kv_allocation"] = cache
        for turn, tokens in enumerate(prompts, 1):
            power()
            past = (turn - 1) * (n + 128)
            sync()
            memory.read(True)
            start = time.perf_counter()
            gen.append_tokens(tokens)
            sync()
            prefill_end = time.perf_counter()
            prefill_memory = memory.read(True)
            selection_start = time.perf_counter()
            gen.generate_next_token()
            sync()
            first_end = time.perf_counter()
            for _ in range(127):
                gen.generate_next_token()
            sync()
            end = time.perf_counter()
            decode_memory = memory.read(True)
            assert decode_memory["failures"] == decode_memory["unknown_frees"] == 0, (
                decode_memory
            )
            seq = np.asarray(gen.get_sequence(0)).copy()
            assert len(seq) == past + n + 128 and np.array_equal(
                seq[past : past + n], tokens
            )
            row = dict(
                repetition=repetition + 1,
                turn=turn,
                past_tokens=past,
                context_after_prefill=past + n,
                prefill_seconds=prefill_end - start,
                ttft_seconds=(prefill_end - start) + (first_end - selection_start),
                decode_ms=(end - first_end) * 1000 / 127,
                decode_tokens_per_second=127 / (end - first_end),
                generated_tokens=128,
            )
            peak = max(prefill_memory["peak"], decode_memory["peak"])
            row.update(
                prefill_peak_bytes=prefill_memory["peak"],
                decode_peak_bytes=decode_memory["peak"],
                peak_total_bytes=peak,
                weights_bytes=report["weights_bytes"],
                kv_bytes=cache["total_bytes"],
                kv_payload_bytes=cache["payload_bytes"],
                other_bytes=peak - report["weights_bytes"] - cache["total_bytes"],
            )
            assert row["other_bytes"] >= 0
            report["records"].append(row)
            save(out / "receipt.json", report)
            print(json.dumps(row), flush=True)
        del gen, params
        gc.collect()
        sync()
    report["status"] = "measured"
    save(out / "receipt.json", report)


def kernel_worker(args):
    import numpy as np

    handles, hip, lib = libraries()
    P, Int, F, S = c.c_void_p, c.c_int, c.c_float, c.c_size_t
    hip.hipMalloc.argtypes = [c.POINTER(P), S]
    hip.hipFree.argtypes = [P]
    hip.hipMemcpy.argtypes = [P, P, S, Int]
    hip.hipMemset.argtypes = [P, Int, S]

    def check(rc):
        assert rc == 0, rc

    def sync():
        check(hip.hipDeviceSynchronize())

    allocations = []

    def alloc(size, x=None):
        p = P()
        check(hip.hipMalloc(c.byref(p), size))
        allocations.append(p)
        if x is None:
            check(hip.hipMemset(p, 0, size))
        else:
            check(hip.hipMemcpy(p, P(x.ctypes.data), size, 1))
        return p

    def upload(x):
        return alloc(x.nbytes, x)

    lib.hip_gqa_autotune_create.argtypes = [c.c_char_p]
    lib.hip_gqa_autotune_create.restype = P

    class PrefReq(c.Structure):
        _fields_ = [("variant", c.c_uint8)] + [
            (n, Int) for n in ["batch", "heads", "groups", "dim", "sq", "skv", "window"]
        ]

    class PrefConfig(c.Structure):
        _fields_ = [(n, Int) for n in ["hb", "qw", "mt", "bkv", "nd"]]

    class PrefResult(c.Structure):
        _fields_ = [("config", PrefConfig), ("source", c.c_uint8), ("distance", F)]

    class DecReq(c.Structure):
        _fields_ = [
            (n, Int)
            for n in [
                "dtype",
                "batch",
                "heads",
                "groups",
                "dim",
                "length",
                "splits",
                "window",
            ]
        ]

    class DecConfig(c.Structure):
        _fields_ = [("wmma", c.c_bool), ("splits", Int), ("bkv", Int)]

    class DecResult(c.Structure):
        _fields_ = [("config", DecConfig), ("source", c.c_uint8), ("distance", F)]

    lib.hip_gqa_autotune_resolve_prefill.argtypes = [
        P,
        c.POINTER(PrefReq),
        c.POINTER(PrefResult),
    ]
    lib.hip_gqa_autotune_resolve_decode.argtypes = [
        P,
        c.POINTER(DecReq),
        c.POINTER(DecResult),
    ]
    lib.hip_gqa_flash_prefill_v3_configured.argtypes = (
        [P] * 5 + [Int] * 8 + [F, Int, P, Int, Int] + [Int] * 5
    )
    lib.hip_gqa_flash_decode_configured.argtypes = (
        [P] * 6 + [Int] * 7 + [F, P, Int, P, Int, Int, P, P, Int, Int, Int]
    )
    lib.hip_gqa_ring_workspace.argtypes = [Int, Int]
    lib.hip_gqa_ring_workspace.restype = S
    lib.hip_gqa_ring_attention.argtypes = [P] * 5 + [Int] * 3 + [P, S]
    lib.hip_tq4_encode_append.argtypes = [P] * 5 + [Int] * 3 + [P] * 3
    lib.hip_tq4_decode_workspace.argtypes = [Int]
    lib.hip_tq4_decode_workspace.restype = S
    lib.hip_tq4_decode.argtypes = [P] * 5 + [Int, Int, F] + [P] * 4 + [S]
    out = OUT / "kernel" / str(args.length)
    out.mkdir(parents=True, exist_ok=False)
    n = args.length
    policy = lib.hip_gqa_autotune_create(None)
    pr, dr = PrefResult(), DecResult()
    lib.hip_gqa_autotune_resolve_prefill(
        policy, c.byref(PrefReq(1, 1, 24, 4, 256, n, n, 0)), c.byref(pr)
    )
    lib.hip_gqa_autotune_resolve_decode(
        policy, c.byref(DecReq(0, 1, 24, 4, 256, n, 64, 0)), c.byref(dr)
    )
    rng = np.random.default_rng(773)
    # Shared random operands for all arms. No GPU attention or encoder probes.
    q = upload(
        np.tile(rng.normal(0, 0.2, (128, 24, 256)).astype(np.float16), (n // 128, 1, 1))
    )
    kd = rng.normal(0, 0.2, (4, n, 256)).astype(np.float16)
    vd = rng.normal(0, 0.2, (4, n, 256)).astype(np.float16)
    k, v = upload(kd), upload(vd)
    ks, vs = upload(kd.transpose(1, 0, 2).copy()), upload(vd.transpose(1, 0, 2).copy())
    del kd, vd
    o = alloc(n * 24 * 256 * 2)
    pk, pv = alloc(4 * n * 132), alloc(4 * n * 132)
    const = np.load(CONSTANTS)
    pi, ck, cv = [upload(const[key]) for key in ["rotation", "cb_k", "cb_v"]]
    dw = alloc(24 * 64 * 258 * 4)
    rwsize = lib.hip_gqa_ring_workspace(1, n)
    rw = alloc(max(1, rwsize))
    twsize = lib.hip_tq4_decode_workspace(n)
    tw = alloc(twsize)

    def dense_prefill():
        return lib.hip_gqa_flash_prefill_v3_configured(
            None,
            q,
            k,
            v,
            o,
            1,
            24,
            4,
            n,
            n,
            256,
            n,
            0,
            0.0625,
            0,
            None,
            24,
            0,
            pr.config.hb,
            pr.config.qw,
            pr.config.mt,
            pr.config.bkv,
            pr.config.nd,
        )

    def ring_prefill():
        return lib.hip_gqa_ring_attention(None, q, k, v, o, n, 0, n, None, 0)

    def encode():
        return lib.hip_tq4_encode_append(None, ks, vs, pk, pv, n, 0, n, pi, ck, cv)

    def dense_decode():
        return lib.hip_gqa_flash_decode_configured(
            None,
            q,
            k,
            v,
            o,
            dw,
            1,
            24,
            4,
            256,
            n,
            n,
            64,
            0.0625,
            None,
            0,
            None,
            0,
            0,
            None,
            None,
            dr.config.wmma,
            dr.config.splits,
            dr.config.bkv,
        )

    def ring_decode():
        return lib.hip_gqa_ring_attention(None, q, k, v, o, 1, n - 1, n, rw, rwsize)

    def tq_decode():
        return lib.hip_tq4_decode(
            None, q, pk, pv, o, n, n, 0.0625, pi, ck, cv, tw, twsize
        )

    def timed(fn, reps=1):
        sync()
        start = time.perf_counter()
        for _ in range(reps):
            check(fn())
        sync()
        return (time.perf_counter() - start) * 1000 / reps

    report = dict(
        status="running",
        length=n,
        provenance=provenance(),
        records=[],
        prefill_config={
            name: getattr(pr.config, name) for name, _ in PrefConfig._fields_
        },
        decode_config={
            name: getattr(dr.config, name) for name, _ in DecConfig._fields_
        },
        lookup_sources=[pr.source, dr.source],
        timing_scope="Synchronized host wall time; prefill 3 single calls; decode 3 batches of 30 calls; all first calls included; no graph replay or cache displacement; decode attention includes TQ rotations but excludes KV append; TQ prefill includes separate full-cache encode",
    )
    save(out / "receipt.json", report)
    for round in range(3):
        for mode in MODES[round:] + MODES[:round]:
            power()
            pf = timed(dense_prefill if mode == "no-ring" else ring_prefill)
            enc = timed(encode) if mode == "turboquant" else 0.0
            row = dict(
                round=round + 1,
                mode=mode,
                prefill_attention_ms=pf,
                encode_ms=enc,
                prefill_total_ms=pf + enc,
            )
            report["records"].append(row)
            save(out / "receipt.json", report)
            print(json.dumps(row), flush=True)
    for round in range(3):
        for mode in MODES[round:] + MODES[:round]:
            fn = {
                "no-ring": dense_decode,
                "ring": ring_decode,
                "turboquant": tq_decode,
            }[mode]
            ms = timed(fn, 30)
            row = dict(round=round + 1, mode=mode, decode_ms=ms)
            report["records"].append(row)
            save(out / "receipt.json", report)
            print(json.dumps(row), flush=True)
    report["status"] = "measured"
    save(out / "receipt.json", report)
    for p in allocations:
        check(hip.hipFree(p))


def campaign(args):
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = []
    if args.kind in ["all", "kernel"]:
        tasks += [("kernel", n, "no-ring") for n in LENGTHS]
    if args.kind in ["all", "model"]:
        tasks += [
            ("model", n, m)
            for i, n in enumerate(LENGTHS[:10])
            for m in MODES[i % 3 :] + MODES[: i % 3]
        ]
    if args.kind in ["all", "multiturn"]:
        tasks += [("multiturn", 4096, m) for m in MODES]
    for kind, n, mode in tasks:
        label = f"{kind}-{n}-{mode}"
        path = OUT / kind / (str(n) if kind != "multiturn" else "8x4096")
        if kind != "kernel":
            path /= mode
        receipt = path / "receipt.json"
        if receipt.exists() and json.loads(receipt.read_text())["status"] == "measured":
            print("REUSE", label, flush=True)
            continue
        power()
        cmd = [
            str(PYTHON),
            str(Path(__file__)),
            kind,
            "--worker",
            "--length",
            str(n),
            "--mode",
            mode,
            "--install",
            str(INSTALL),
            "--output",
            str(OUT),
            "--model",
            str(MODEL),
            "--codec",
            str(CODEC),
            "--constants",
            str(CONSTANTS),
            "--sdk-devel",
            str(DEV),
            "--sdk-core",
            str(CORE),
            "--memory-counter",
            str(MEMORY_COUNTER),
        ]
        if GENAI_PATH is not None:
            cmd += ["--genai-pythonpath", str(GENAI_PATH)]
        start = time.time()
        save(OUT / "progress.json", dict(status="running", task=label, started=start))
        with (OUT / (label + ".log")).open("w", encoding="utf-8") as log:
            rc = subprocess.call(
                cmd, env=environment(mode), stdout=log, stderr=subprocess.STDOUT
            )
        print(label, rc, round(time.time() - start, 2), flush=True)
        if rc:
            save(
                OUT / "progress.json", dict(status="failed", task=label, returncode=rc)
            )
            print((OUT / (label + ".log")).read_text(errors="replace")[-5000:])
            raise SystemExit(rc)
    save(OUT / "progress.json", dict(status="measured", scope=args.kind))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["all", "kernel", "model", "multiturn"])
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", choices=MODES, default="no-ring")
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--install", type=Path, default=INSTALL)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--codec", type=Path, default=CODEC)
    parser.add_argument("--constants", type=Path, default=CONSTANTS)
    parser.add_argument("--sdk-devel", type=Path, default=DEV)
    parser.add_argument("--sdk-core", type=Path, default=CORE)
    parser.add_argument("--genai-pythonpath", type=Path)
    parser.add_argument("--memory-counter", type=Path, default=MEMORY_COUNTER)
    args = parser.parse_args()
    MEMORY_COUNTER = args.memory_counter.resolve()
    INSTALL, OUT, MODEL, CODEC, CONSTANTS, DEV, CORE, GENAI_PATH = (
        args.install.resolve(),
        args.output.resolve(),
        args.model.resolve(),
        args.codec.resolve(),
        args.constants.resolve(),
        args.sdk_devel.resolve(),
        args.sdk_core.resolve(),
        args.genai_pythonpath,
    )
    if os.name != "nt":
        raise SystemExit("This harness reproduces the Windows HIP campaign.")
    power()
    if not args.worker:
        campaign(args)
    elif args.kind == "kernel":
        kernel_worker(args)
    else:
        model_worker(args)
