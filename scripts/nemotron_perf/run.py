#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Run Nemotron with an isolated config, no CPU fallback, and optional IR."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path("/scratch/fhanuman/RCOmEP")
GPU_LOCK = WORKSPACE / "hrxWork/.cursor/artifacts/hrx-backend/gpu.lock"


@contextmanager
def gpu_telemetry(output):
    device = Path("/sys/class/drm/renderD128/device")
    counters = {
        name: device / name
        for name in ("mem_info_vram_used", "mem_info_gtt_used", "gpu_busy_percent")
    }
    for name in ("temp1_input", "power1_average"):
        paths = list(device.glob(f"hwmon/hwmon*/{name}"))
        if paths:
            counters[name] = paths[0]
    stop = threading.Event()

    def sample():
        with (output / "gpu-device-samples.jsonl").open("x") as log:
            while True:
                record = {"monotonic_seconds": time.monotonic()}
                for name, path in counters.items():
                    try:
                        record[name] = int(path.read_text().strip())
                    except (OSError, ValueError) as error:
                        record[name] = {"unavailable": str(error)}
                try:
                    for line in Path("/proc/meminfo").read_text().splitlines():
                        key, value = line.split(":", 1)
                        if key in ("MemAvailable", "SwapFree"):
                            record["host_" + key + "_bytes"] = (
                                int(value.split()[0]) * 1024
                            )
                except (OSError, ValueError) as error:
                    record["host_memory_unavailable"] = str(error)
                log.write(json.dumps(record) + "\n")
                log.flush()
                if stop.wait(0.25):
                    break

    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join()


@contextmanager
def gpu_lease(path, cooldown):
    # Use the same existing file as the other GPU runners. Do not replace it.
    with path.open("r") as lock:
        print(f"Waiting for exclusive GPU lock: {path}", flush=True)
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(1)
        try:
            print(f"GPU lock acquired; cooldown: {cooldown} seconds", flush=True)
            time.sleep(cooldown)
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            print("GPU lock released", flush=True)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-dir",
        type=Path,
        default=WORKSPACE / "oga_models/Nemotron-3.5-Lightning-30B-A3B-fp16-int4-onnx",
    )
    p.add_argument(
        "--runtime-dir", type=Path, default=ROOT / ".nemotron-perf/ci-main-04f1c0365"
    )
    p.add_argument(
        "--sdk", type=Path, default=WORKSPACE / "build/onnx-hipdnn-ep/_therock"
    )
    p.add_argument(
        "--prompt", default="What is the capital of France? Answer in one sentence."
    )
    p.add_argument(
        "--prompt-file", type=Path, help="Use this exact prompt without a chat template"
    )
    p.add_argument("--raw-prompt", action="store_true")
    p.add_argument("--gen", type=int, default=32, help="Number of NEW tokens")
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--docker-image", default="nemotron-main-runtime:20261020")
    p.add_argument(
        "--sdk-zstd-library",
        type=Path,
        help="Use a container-only SDK zstd replacement",
    )
    p.add_argument("--ir-dump-tree", action="store_true")
    p.add_argument("--ir-dump-after-only", action="store_true")
    p.add_argument(
        "--ir-dump-dir", type=Path, help="New output directory; implies IR tree mode"
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        help="New evidence directory, without enabling IR dumping",
    )
    p.add_argument(
        "--perf",
        action="store_true",
        help="Diagnostic kernel timing, not a performance baseline",
    )
    p.add_argument(
        "--kernel-selection-log",
        action="store_true",
        help="Log MatMulNBits/GQA lookup and tuning decisions; diagnostic only",
    )
    p.add_argument(
        "--blas-tuning-override",
        type=Path,
        help="Read a version-specific hipBLASLt tuning file for controlled comparisons",
    )
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument(
        "--sparse-experts",
        action="store_true",
        help="Enable sparse expert fusion; inactive expert results must be finite",
    )
    p.add_argument(
        "--logits-binary",
        type=Path,
        help="Run this diagnostic logits capture executable instead of the benchmark",
    )
    p.add_argument(
        "--reference-logits",
        type=Path,
        help="Use tokens from this logits capture as the exact history",
    )
    p.add_argument(
        "--no-matmul-autotune",
        action="store_true",
        help="Use the first hipBLASLt heuristic for a controlled numerical check",
    )
    p.add_argument(
        "--nbits-autotune-mode",
        choices=("lookup", "online"),
        help="Select the native MatMulNBits tuner; does not affect BLAS",
    )
    p.add_argument(
        "--router-sequential-f32",
        action="store_true",
        help="Enable the experimental sequential-FMA router with register prefetch",
    )
    p.add_argument(
        "--head-f16-matvec",
        action="store_true",
        help="Enable the separate, non-bit-exact FP16 vocabulary matvec experiment",
    )
    p.add_argument(
        "--la-f16-state",
        action="store_true",
        help="Fuse FP16 recurrent-state boundary casts into FP32 attention",
    )
    p.add_argument(
        "--la-compact-heads",
        action="store_true",
        help="Use attention head broadcasting instead of explicit Q/K repeats",
    )
    p.add_argument(
        "--la-fixed-geometry",
        action="store_true",
        help="Specialize gated linear-attention decode for Hq8/Hkv64/Dk128/Dv64",
    )
    p.add_argument(
        "--swiglu-f32",
        action="store_true",
        help="Fuse the FP16 activation chain with FP32 intermediate rounding",
    )
    p.add_argument(
        "--gpu-lock",
        type=Path,
        default=GPU_LOCK,
        help="Existing lock file shared by all GPU users",
    )
    p.add_argument(
        "--cooldown-seconds",
        type=int,
        default=5,
        help="Idle delay under the GPU lock before startup (1-60 seconds)",
    )
    args = p.parse_args()
    if args.output_dir and args.ir_dump_dir:
        p.error("Use either --output-dir or --ir-dump-dir")
    if args.reference_logits and not args.logits_binary:
        p.error("--reference-logits requires --logits-binary")
    if min(args.gen, args.reps, args.chunk_size) < 1 or args.warmup < 0:
        p.error("Token count, repetitions and chunk size must be positive")
    if not 1 <= args.cooldown_seconds <= 60:
        p.error("Cooldown must be between 1 and 60 seconds")
    source = args.model_dir.resolve(strict=True)
    runtime = args.runtime_dir.resolve(strict=True)
    sdk = args.sdk.resolve(strict=True)
    benchmark = runtime / "bin/model_benchmark"
    compiler = runtime / "lib/libamdgpu-ep.so"
    for path in (benchmark, compiler, runtime / "lib/libonnxruntime-genai.so"):
        if not path.is_file():
            raise FileNotFoundError(path)
    original_hash = digest(source / "genai_config.json")
    config = json.loads((source / "genai_config.json").read_text())
    opts = config["model"]["decoder"].setdefault("session_options", {})
    opts["provider_options"] = [{"AMDGPU": {"profile": "hip"}}]
    opts["session.disable_cpu_ep_fallback"] = "1"
    config["search"]["chunk_size"] = args.chunk_size
    # This matches the package template for a simple chat with thinking off.
    prompt = args.prompt
    if args.prompt_file:
        prompt = args.prompt_file.read_text()
    elif not args.raw_prompt:
        prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n" + prompt + "<|im_end|>\n"
            "<|im_start|>assistant\n<think></think>"
        )
    root = ROOT / ".nemotron-perf/runs"
    root.mkdir(parents=True, exist_ok=True)
    if args.ir_dump_dir or args.output_dir:
        out = (args.ir_dump_dir or args.output_dir).resolve()
        out.mkdir(parents=True, exist_ok=False)
    else:
        out = Path(
            tempfile.mkdtemp(
                prefix=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_"), dir=root
            )
        )
    for directory in ("model", "tmp", "home", "cache"):
        (out / directory).mkdir()
    for path in source.iterdir():
        if path.is_file() and path.name != "genai_config.json":
            (out / "model" / path.name).symlink_to(path)
    (out / "model/genai_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "prompt.txt").write_text(prompt)
    env = {
        "HOME": str(out / "home"),
        "TMPDIR": str(out / "tmp"),
        "XDG_CACHE_HOME": str(out / "cache"),
        "LD_LIBRARY_PATH": f"{runtime}/lib:{sdk}/lib",
        "PATH": f"{runtime}/bin:{sdk}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "THEROCK_DIST": str(sdk),
        "AMDGPU_EP_PATH": str(compiler),
        "HIPDNN_EP_STRICT": "1",
        "MORPHIZEN_EP_ENABLE_CPU_DEVICE": "0",
    }
    dump = args.ir_dump_tree or args.ir_dump_after_only or args.ir_dump_dir
    if dump:
        env.update(HIPDNN_EP_IR_DUMP_TREE="1", HIPDNN_EP_IR_DUMP_PATH=str(out / "ir"))
    if args.ir_dump_after_only:
        env["HIPDNN_EP_IR_DUMP_AFTER_ONLY"] = "1"
    if args.perf:
        env["HIPDNN_EP_PERF"] = "1"
    if args.kernel_selection_log:
        env["HIPDNN_MATMUL_AUTOTUNE_LOG"] = "1"
        env["HIPDNN_GQA_AUTOTUNE_LOG"] = "1"
        env["HIPDNN_GQA_LUT_LOG"] = "1"
        env["HIPBLASLT_LOG_MASK"] = "32"
    if args.blas_tuning_override:
        blas_override = args.blas_tuning_override.resolve(strict=True)
        env["HIPBLASLT_TUNING_OVERRIDE_FILE"] = str(blas_override)
    if args.sparse_experts:
        env["HIPDNN_EP_EXPERIMENTAL_SPARSE_EXPERTS"] = "1"
    if args.no_matmul_autotune:
        env["HIPDNN_EP_AUTOTUNE"] = "0"
    if args.nbits_autotune_mode:
        env["HIPDNN_MATMUL_AUTOTUNE_MODE"] = args.nbits_autotune_mode
    if args.router_sequential_f32:
        env["HIPDNN_EP_EXPERIMENTAL_SEQUENTIAL_ROUTER_F32"] = "1"
    if args.head_f16_matvec:
        env["HIPDNN_EP_EXPERIMENTAL_HEAD_F16_MATVEC"] = "1"
    if args.la_f16_state:
        env["HIPDNN_EP_EXPERIMENTAL_LA_F16_STATE"] = "1"
    if args.la_compact_heads:
        env["HIPDNN_EP_EXPERIMENTAL_LA_COMPACT_HEADS"] = "1"
    if args.la_fixed_geometry:
        env["HIPDNN_EP_EXPERIMENTAL_LA_FIXED_GEOMETRY"] = "1"
    if args.swiglu_f32:
        env["HIPDNN_EP_EXPERIMENTAL_SWIGLU_F32"] = "1"
    name = "nemotron-smoke-" + uuid.uuid4().hex[:12]
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        "none",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--ulimit",
        "core=0",
        "--mount",
        f"type=bind,source={WORKSPACE},target={WORKSPACE},readonly",
        "--mount",
        f"type=bind,source={out},target={out}",
        "--workdir",
        str(out),
    ]
    for device in ("/dev/kfd", "/dev/dri/renderD128"):
        command += ["--device", device, "--group-add", str(os.stat(device).st_gid)]
    if args.sdk_zstd_library:
        zstd = args.sdk_zstd_library.resolve(strict=True)
        target = sdk / "lib/rocm_sysdeps/lib/librocm_sysdeps_zstd.so.1.5.7"
        if not target.is_file():
            raise FileNotFoundError(target)
        command += ["--mount", f"type=bind,source={zstd},target={target},readonly"]
    if args.blas_tuning_override:
        command += [
            "--mount",
            f"type=bind,source={blas_override},target={blas_override},readonly",
        ]
    for key, value in env.items():
        command += ["-e", f"{key}={value}"]
    if args.logits_binary:
        command += [
            "--entrypoint",
            str(args.logits_binary.resolve(strict=True)),
            args.docker_image,
            str(out / "model"),
            str(out / "prompt.txt"),
            str(out),
            str(args.gen),
        ]
        if args.reference_logits:
            command.append(str(args.reference_logits.resolve(strict=True)))
    else:
        command += [
            "--entrypoint",
            str(benchmark),
            args.docker_image,
            "-i",
            str(out / "model"),
            "--prompt_file",
            str(out / "prompt.txt"),
            "-g",
            str(args.gen),
            "-r",
            str(args.reps),
            "-w",
            str(args.warmup),
            "-b",
            "1",
            "-ml",
            "0",
            "-v",
        ]
    manifest = {
        "source": str(source),
        "source_config_sha256": original_hash,
        "graph_sha256": digest(source / "model.onnx"),
        "compiler_sha256": digest(compiler),
        "command": command,
        "kernel_sha256": {
            p.name: digest(p) for p in (runtime / "lib").glob("libcustom_kernels_*.so")
        },
        "sparse_experts": args.sparse_experts,
        "router_sequential_f32": args.router_sequential_f32,
        "head_f16_matvec": args.head_f16_matvec,
        "la_f16_state": args.la_f16_state,
        "la_compact_heads": args.la_compact_heads,
        "la_fixed_geometry": args.la_fixed_geometry,
        "swiglu_f32": args.swiglu_f32,
        "nbits_autotune_mode": args.nbits_autotune_mode or "lookup",
        "gpu_memory_sampling": "Whole-device VRAM/GTT counters at 0.25 s; not exact model allocations",
        "cpu_fallback_disabled": True,
        "diagnostic": bool(
            dump or args.perf or args.logits_binary or args.kernel_selection_log
        ),
        "gpu_coordination": {
            "lock_file": str(args.gpu_lock.resolve()),
            "cooldown_seconds": args.cooldown_seconds,
        },
        "weights_size": (source / "model.onnx.data").stat().st_size,
    }
    if args.logits_binary:
        manifest["logits_binary_sha256"] = digest(args.logits_binary)
    if args.blas_tuning_override:
        blas_library = next(
            p
            for p in (runtime / "lib/libhipblaslt.so", sdk / "lib/libhipblaslt.so")
            if p.is_file()
        )
        manifest["blas_tuning_override"] = {
            "path": str(blas_override),
            "sha256": digest(blas_override),
            "library": str(blas_library.resolve()),
            "library_sha256": digest(blas_library),
            "scope": "Explicit comparison control; solution IDs are not portable",
        }
    if args.sdk_zstd_library:
        manifest["sdk_zstd_override"] = {
            "path": str(zstd),
            "sha256": digest(zstd),
            "original_sha256": digest(target),
            "container_only": True,
        }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Nemotron: AMDGPU/profile=hip; CPU fallback disabled\nEvidence: {out}",
        flush=True,
    )
    if args.prepare_only:
        return 0
    with gpu_lease(args.gpu_lock, args.cooldown_seconds):
        devices = json.loads(
            subprocess.check_output(
                [str(WORKSPACE / "therock-dist/bin/amd-smi"), "process", "--json"],
                text=True,
            )
        )
        if not devices or any(
            "no running processes" not in str(item["process_info"]).lower()
            for device in devices
            for item in device["process_list"]
        ):
            raise RuntimeError("GPU availability not established; no model run started")

        def stop(signum, frame):
            subprocess.run(
                ["docker", "kill", "--signal=TERM", name],
                capture_output=True,
                timeout=15,
            )

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        with gpu_telemetry(out), (out / "run.log").open("x") as log:
            proc = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = proc.wait()
    (out / "exit_status").write_text(str(code) + "\n")
    if digest(source / "genai_config.json") != original_hash:
        raise RuntimeError("Source config changed during the run")
    print(f"Exit: {code}; source config unchanged; evidence: {out}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
