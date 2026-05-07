#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Shared fixtures and helpers for Python performance tests."""

import gc
import itertools
import json
import os
import pathlib
import shutil
import time
import urllib.request
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

NUM_WARMUP = 1
NUM_RUNS = 3

EP_DLL_NAME = "onnxruntime_morphizen_ep.dll"
EP_REGISTRATION_NAME = "MorphiZenExecutionProvider"

_morphizen_registered = False


# ── Model download / shape fixing ────────────────────────────────────────────


def download(url: str, dest: pathlib.Path) -> None:
    print(f"  Downloading {url}")
    urllib.request.urlretrieve(url, str(dest))
    mb = dest.stat().st_size / (1024 * 1024)
    print(f"  Saved {dest.name} ({mb:.1f} MB)")


def fix_shapes(src: pathlib.Path, dst: pathlib.Path, dim_map: dict) -> None:
    import onnx

    m = onnx.load(str(src), load_external_data=False)
    for tensor in list(m.graph.input) + list(m.graph.output) + list(m.graph.value_info):
        if not tensor.type.tensor_type.HasField("shape"):
            continue
        for d in tensor.type.tensor_type.shape.dim:
            if d.dim_param in dim_map:
                v = dim_map[d.dim_param]
                d.Clear()
                d.dim_value = v
    onnx.save(m, str(dst), save_as_external_data=False)
    print(f"  Fixed-shape model -> {dst.name}")


# ── Perf helpers ─────────────────────────────────────────────────────────────


def run_timed(session, inputs, warmup=NUM_WARMUP, runs=NUM_RUNS):
    for _ in range(warmup):
        session.run(None, inputs)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, inputs)
        times.append(time.perf_counter() - t0)
    return times


def report(label, times_sec):
    ms = [t * 1000 for t in times_sec]
    avg, std = np.mean(ms), np.std(ms)
    print(f"\n{'=' * 60}")
    print(f"{label}  ({len(ms)} runs)")
    print(f"  Avg: {avg:8.2f} ms  (std: {std:.2f})")
    print(f"  Min: {min(ms):8.2f} ms")
    print(f"  Max: {max(ms):8.2f} ms")
    print(f"  Tokens/sec: {1000 / avg:.1f}")
    print(f"{'=' * 60}")
    return avg


def compare_outputs(ref_outputs, test_outputs, output_names, label):
    """Compare test EP outputs against CPU reference. Returns (passed, report_str)."""
    lines = []
    all_ok = True
    for i, name in enumerate(output_names):
        ref = ref_outputs[i].astype(np.float32).flatten()
        test = test_outputs[i].astype(np.float32).flatten()

        if ref.shape != test.shape:
            lines.append(f"  {name}: SHAPE MISMATCH {ref.shape} vs {test.shape}")
            all_ok = False
            continue

        finite = np.isfinite(ref) & np.isfinite(test)
        ref_f, test_f = ref[finite], test[finite]
        n_skipped = int(ref.size - np.count_nonzero(finite))

        if ref_f.size == 0:
            lines.append(f"  {name}: no finite elements to compare  [SKIP]")
            continue

        max_abs = float(np.max(np.abs(ref_f - test_f)))

        ref_norm = np.linalg.norm(ref_f)
        diff_norm = np.linalg.norm(ref_f - test_f)
        rel_l2 = float(diff_norm / ref_norm) if ref_norm > 0 else 0.0

        dot = np.dot(ref_f, test_f)
        norms = np.linalg.norm(ref_f) * np.linalg.norm(test_f)
        cosine = float(dot / norms) if norms > 0 else 1.0

        ok = cosine > 0.95
        if not ok:
            all_ok = False
        status = "OK" if ok else "FAIL"

        skip_note = f"  skipped={n_skipped}" if n_skipped else ""
        lines.append(
            f"  {name}: cosine={cosine:.6f}  rel_L2={rel_l2:.4e}"
            f"  max_abs={max_abs:.4e}{skip_note}  [{status}]"
        )

    header = f"\n{'=' * 60}\n{label} accuracy vs CPU\n"
    body = "\n".join(lines)
    footer = f"\n{'=' * 60}"
    report_str = header + body + footer
    print(report_str)
    return all_ok, report_str


def get_amd_dml_providers():
    """Return DML provider list targeting the AMD GPU, falling back to default."""
    if "DmlExecutionProvider" not in ort.get_available_providers():
        return None
    try:
        import subprocess

        r = subprocess.run(
            [
                "powershell",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for i, line in enumerate(r.stdout.strip().splitlines()):
            if "AMD" in line.upper() or "RADEON" in line.upper():
                print(f"  DirectML: using device {i} ({line.strip()})")
                return [
                    ("DmlExecutionProvider", {"device_id": str(i)}),
                    "CPUExecutionProvider",
                ]
    except Exception:
        pass
    return ["DmlExecutionProvider", "CPUExecutionProvider"]


def register_morphizen_ep(repo_root):
    """Register the MorphiZen EP library with ONNX Runtime (once per process)."""
    global _morphizen_registered

    dist_bin = repo_root / "install" / "dist" / "bin"
    therock_bin = repo_root / "install" / "therock" / "bin"
    ep_dll = dist_bin / EP_DLL_NAME

    if not ep_dll.exists():
        return None

    for d in [dist_bin, therock_bin]:
        if d.exists() and str(d) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")

    if not _morphizen_registered:
        ort.register_execution_provider_library(EP_REGISTRATION_NAME, str(ep_dll))
        _morphizen_registered = True

    from onnxruntime.capi._pybind_state import get_ep_devices

    devices = get_ep_devices()
    return [d for d in devices if d.ep_name == EP_REGISTRATION_NAME]


# ── IOBinding helpers ────────────────────────────────────────────────────────


@dataclass
class LlamaModelConfig:
    num_kv_layers: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    has_position_ids: bool = False


def make_llama_inputs(cfg: LlamaModelConfig, seq_len=1) -> dict:
    """Create inputs for a Llama model.

    seq_len=1 (default) creates single-token decode inputs.
    seq_len>1 creates prefill inputs with that many tokens.
    """
    inputs = {
        "input_ids": np.random.randint(1, 32000, (1, seq_len), dtype=np.int64),
        "attention_mask": np.ones((1, cfg.max_seq_len), dtype=np.int64),
    }
    if cfg.has_position_ids:
        inputs["position_ids"] = np.arange(
            cfg.max_seq_len - seq_len, cfg.max_seq_len, dtype=np.int64
        ).reshape(1, seq_len)
    for i in range(cfg.num_kv_layers):
        inputs[f"past_key_values.{i}.key"] = np.zeros(
            (1, cfg.num_kv_heads, cfg.max_seq_len, cfg.head_dim), dtype=np.float16
        )
        inputs[f"past_key_values.{i}.value"] = np.zeros(
            (1, cfg.num_kv_heads, cfg.max_seq_len, cfg.head_dim), dtype=np.float16
        )
    return inputs


AMD_VENDOR_ID = 0x1002


def run_timed_iobinding(
    sess,
    inputs,
    cfg: LlamaModelConfig,
    warmup=NUM_WARMUP,
    runs=NUM_RUNS,
    use_device_memory=False,
    dim_map=None,
):
    """Run inference using IOBinding with shared past/present KV cache buffers.

    Binds the same OrtValue to both past_key_values.N.{key,value} (input) and
    present.N.{key,value} (output) so the runtime sees past_ptr == present_ptr.

    When use_device_memory=True, KV cache OrtValues are allocated via the EP's
    GPU allocator (hipHostMalloc on AMD iGPU) so the runtime sees
    memory_type==TENSOR_MEMORY_GPU and aliases them directly — zero H2D/D2H.
    """
    io = sess.io_binding()

    kv_shape = (1, cfg.num_kv_heads, cfg.max_seq_len, cfg.head_dim)
    kv_cache = {}
    for i in range(cfg.num_kv_layers):
        for kind in ("key", "value"):
            if use_device_memory:
                kv_cache[(i, kind)] = ort.OrtValue.ortvalue_from_shape_and_type(
                    list(kv_shape),
                    np.float16,
                    device_type="gpu",
                    vendor_id=AMD_VENDOR_ID,
                )
            else:
                kv_cache[(i, kind)] = ort.OrtValue.ortvalue_from_numpy(
                    np.zeros(kv_shape, dtype=np.float16)
                )

    _ORT_TYPE_MAP = {
        "tensor(float16)": np.float16,
        "tensor(float)": np.float32,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
    }

    output_vals = {}
    for o in sess.get_outputs():
        if o.name.startswith("present."):
            continue
        dtype = _ORT_TYPE_MAP.get(o.type, np.float32)
        shape = o.shape
        if dim_map:
            shape = [dim_map.get(d, d) if isinstance(d, str) else d for d in shape]
        output_vals[o.name] = ort.OrtValue.ortvalue_from_shape_and_type(shape, dtype)

    def bind_all():
        io.clear_binding_inputs()
        io.clear_binding_outputs()
        for name, arr in inputs.items():
            if name.startswith("past_key_values"):
                continue
            io.bind_ortvalue_input(name, ort.OrtValue.ortvalue_from_numpy(arr))
        for i in range(cfg.num_kv_layers):
            for kind in ("key", "value"):
                val = kv_cache[(i, kind)]
                io.bind_ortvalue_input(f"past_key_values.{i}.{kind}", val)
                io.bind_ortvalue_output(f"present.{i}.{kind}", val)
        for name, val in output_vals.items():
            io.bind_ortvalue_output(name, val)

    for _ in range(warmup):
        bind_all()
        sess.run_with_iobinding(io)

    times = []
    for _ in range(runs):
        bind_all()
        t0 = time.perf_counter()
        sess.run_with_iobinding(io)
        times.append(time.perf_counter() - t0)
    return times


def run_iobinding_once(
    sess,
    inputs,
    cfg: LlamaModelConfig,
    use_device_memory=False,
    dim_map=None,
):
    """Run single inference with IOBinding, return outputs as list.

    Returns outputs in sess.get_outputs() order, compatible with
    compare_outputs(). Same IOBinding setup as run_timed_iobinding (shared
    past/present KV cache buffers).

    use_device_memory=False (default) zero-initializes KV cache from numpy,
    matching CPU sess.run() for fair accuracy comparison.
    """
    io = sess.io_binding()

    kv_shape = (1, cfg.num_kv_heads, cfg.max_seq_len, cfg.head_dim)
    kv_cache = {}
    for i in range(cfg.num_kv_layers):
        for kind in ("key", "value"):
            if use_device_memory:
                kv_cache[(i, kind)] = ort.OrtValue.ortvalue_from_shape_and_type(
                    list(kv_shape),
                    np.float16,
                    device_type="gpu",
                    vendor_id=AMD_VENDOR_ID,
                )
            else:
                past_name = f"past_key_values.{i}.{kind}"
                arr = inputs.get(past_name, np.zeros(kv_shape, dtype=np.float16))
                kv_cache[(i, kind)] = ort.OrtValue.ortvalue_from_numpy(arr)

    _ORT_TYPE_MAP = {
        "tensor(float16)": np.float16,
        "tensor(float)": np.float32,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
    }

    output_vals = {}
    for o in sess.get_outputs():
        if o.name.startswith("present."):
            continue
        dtype = _ORT_TYPE_MAP.get(o.type, np.float32)
        shape = o.shape
        if dim_map:
            shape = [dim_map.get(d, d) if isinstance(d, str) else d for d in shape]
        output_vals[o.name] = ort.OrtValue.ortvalue_from_shape_and_type(shape, dtype)

    for name, arr in inputs.items():
        if name.startswith("past_key_values"):
            continue
        io.bind_ortvalue_input(name, ort.OrtValue.ortvalue_from_numpy(arr))
    for i in range(cfg.num_kv_layers):
        for kind in ("key", "value"):
            val = kv_cache[(i, kind)]
            io.bind_ortvalue_input(f"past_key_values.{i}.{kind}", val)
            io.bind_ortvalue_output(f"present.{i}.{kind}", val)
    for name, val in output_vals.items():
        io.bind_ortvalue_output(name, val)

    sess.run_with_iobinding(io)

    results = []
    for o in sess.get_outputs():
        if o.name.startswith("present."):
            parts = o.name.split(".")
            idx, kind = int(parts[1]), parts[2]
            results.append(kv_cache[(idx, kind)].numpy())
        else:
            results.append(output_vals[o.name].numpy())
    return results


# ── Session helpers ──────────────────────────────────────────────────────────


def create_cpu_session(model_path):
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def create_ep_session(model_path, repo_root):
    devices = register_morphizen_ep(repo_root)
    if not devices:
        pytest.skip("MorphiZen EP not found — run build.py first")
    so = ort.SessionOptions()
    so.add_provider_for_devices(devices, {})
    return ort.InferenceSession(model_path, sess_options=so)


def create_dml_session(model_path):
    providers = get_amd_dml_providers()
    if providers is None:
        pytest.skip("DmlExecutionProvider not available")
    return ort.InferenceSession(model_path, providers=providers)


def cleanup(*args):
    for obj in args:
        del obj
    gc.collect()


# ── Dim map ─────────────────────────────────────────────────────────────────


def make_dim_map(seq_len, kv_len):
    return {
        "batch_size": 1,
        "sequence_length": seq_len,
        "past_sequence_length": kv_len,
        "total_sequence_length": kv_len,
    }


# ── Model download ─────────────────────────────────────────────────────────


def ensure_model(model_dir, onnx_file, data_file, hf_base):
    model_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = model_dir / onnx_file
    data_path = model_dir / data_file
    if not onnx_path.exists():
        download(f"{hf_base}/{onnx_file}", onnx_path)
    if not data_path.exists():
        download(f"{hf_base}/{data_file}", data_path)
    return str(onnx_path)


def ensure_fixed_model(model_dir, onnx_file, data_file, seq_len, kv_len):
    """Materialize a fixed-shape ONNX next to (not inside) the dynamic model dir.

    External weights are hardlinked (or copied as fallback) into the sibling
    `<model_dir>-fixed/` so the rewritten ONNX's external_data refs resolve.
    Keeping the dynamic model dir free of test-derived `model_fixed_kv*_sq*.onnx`
    leftovers — same hygiene as `ensure_pipeline_dir`.
    """
    fixed_dir = model_dir.parent / f"{model_dir.name}-fixed"
    fixed_dir.mkdir(parents=True, exist_ok=True)
    weights_dst = fixed_dir / data_file
    if not weights_dst.exists():
        try:
            os.link(model_dir / data_file, weights_dst)
        except OSError:
            shutil.copy2(model_dir / data_file, weights_dst)
    name = f"{onnx_file.rsplit('.', 1)[0]}_fixed_kv{kv_len}_sq{seq_len}.onnx"
    dst = fixed_dir / name
    if not dst.exists():
        dim_map = make_dim_map(seq_len, kv_len)
        fix_shapes(model_dir / onnx_file, dst, dim_map)
    return str(dst)


def ensure_pipeline_dir(
    parent_dir,
    pipeline_dir,
    onnx_file,
    data_file,
    prefill_seq_len,
    decode_seq_len,
    kv_len,
    tokenizer_files,
    prefill_basename="model_prefill.onnx",
    decode_basename="model_decode.onnx",
):
    """Materialize a self-contained OGA `decoder-pipeline` model dir.

    Produces `pipeline_dir/{prefill_basename, decode_basename}` from the dynamic
    parent ONNX via `fix_shapes`, hardlinks (or copies) external weights so the
    rewritten ONNX files' external_data refs resolve, and copies tokenizer files.
    Returns (prefill_path, decode_path).
    """
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    weights_dst = pipeline_dir / data_file
    if not weights_dst.exists():
        try:
            os.link(parent_dir / data_file, weights_dst)
        except OSError:
            shutil.copy2(parent_dir / data_file, weights_dst)
    for fname in tokenizer_files:
        dst = pipeline_dir / fname
        if not dst.exists():
            shutil.copy2(parent_dir / fname, dst)
    prefill = pipeline_dir / prefill_basename
    decode = pipeline_dir / decode_basename
    if not prefill.exists():
        fix_shapes(
            parent_dir / onnx_file, prefill, make_dim_map(prefill_seq_len, kv_len)
        )
    if not decode.exists():
        fix_shapes(parent_dir / onnx_file, decode, make_dim_map(decode_seq_len, kv_len))
    return str(prefill), str(decode)


# ── OGA decoder-pipeline (sliding_window) helpers ───────────────────────────
#
# Filename convention: prefill basename depends on (window_size, kv_len);
# decode basename depends only on kv_len (decode input is always [1, 1]).
# This lets multiple sliding-window configs reuse the same decode DLL.


def pipeline_sliding_prefill_filename(window_size, kv_len):
    return f"prefill_p{window_size}m{kv_len}.onnx"


def pipeline_sliding_decode_filename(kv_len):
    return f"decode_m{kv_len}.onnx"


def make_pipeline_sliding_genai_config(
    *,
    window_size,
    kv_len,
    num_layers,
    num_kv_heads,
    num_attention_heads,
    head_dim,
    hidden_size,
    vocab_size,
    bos_token_id,
    eos_token_ids,
    pad_token_id,
    has_position_ids,
):
    """Build a decoder-pipeline genai_config that chunks prompts via sliding_window.

    Object-style pipeline (keys → model_id), KV cache shared across sub-models
    via past_present_share_buffer=True, slide_key_value_cache=False to keep the
    static KV buffer. See CLAUDE.md "OGA decoder-pipeline" entry for details.
    """
    sub_inputs = [
        "input_ids",
        "attention_mask",
        *(("position_ids",) if has_position_ids else ()),
        *(
            name
            for i in range(num_layers)
            for name in (
                f"past_key_values.{i}.key",
                f"past_key_values.{i}.value",
            )
        ),
    ]
    sub_outputs = [
        "logits",
        *(
            name
            for i in range(num_layers)
            for name in (f"present.{i}.key", f"present.{i}.value")
        ),
    ]
    inputs_block = {
        "input_ids": "input_ids",
        "attention_mask": "attention_mask",
        "past_key_names": "past_key_values.%d.key",
        "past_value_names": "past_key_values.%d.value",
    }
    if has_position_ids:
        inputs_block["position_ids"] = "position_ids"
    return {
        "model": {
            "bos_token_id": bos_token_id,
            "eos_token_id": list(eos_token_ids),
            "pad_token_id": pad_token_id,
            "vocab_size": vocab_size,
            "context_length": kv_len,
            "type": "decoder-pipeline",
            "decoder": {
                "head_size": head_dim,
                "hidden_size": hidden_size,
                "num_attention_heads": num_attention_heads,
                "num_key_value_heads": num_kv_heads,
                "num_hidden_layers": num_layers,
                "sliding_window": {
                    "window_size": window_size,
                    "alignment": "left",
                    "slide_inputs": True,
                    "slide_key_value_cache": False,
                },
                "session_options": {
                    "session.disable_cpu_ep_fallback": "1",
                    "provider_options": [{"MorphiZenEP": {}}],
                },
                "inputs": inputs_block,
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "present.%d.key",
                    "present_value_names": "present.%d.value",
                },
                "pipeline": {
                    "prefill": {
                        "filename": pipeline_sliding_prefill_filename(
                            window_size, kv_len
                        ),
                        "run_on_prompt": True,
                        "run_on_token_gen": False,
                        "inputs": sub_inputs,
                        "outputs": sub_outputs,
                    },
                    "decode": {
                        "filename": pipeline_sliding_decode_filename(kv_len),
                        "run_on_prompt": False,
                        "run_on_token_gen": True,
                        "inputs": sub_inputs,
                        "outputs": sub_outputs,
                    },
                },
            },
        },
        "search": {
            "diversity_penalty": 0.0,
            "do_sample": True,
            "early_stopping": True,
            "length_penalty": 1.0,
            "max_length": kv_len,
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


def ensure_pipeline_sliding_oga_files(
    *,
    parent_dir,
    pipeline_dir,
    onnx_file,
    data_file,
    tokenizer_files,
    window_size,
    kv_len,
    config_dict,
):
    """Materialize a sliding-window pipeline dir + write its genai_config.json."""
    ensure_pipeline_dir(
        parent_dir=parent_dir,
        pipeline_dir=pipeline_dir,
        onnx_file=onnx_file,
        data_file=data_file,
        prefill_seq_len=window_size,
        decode_seq_len=1,
        kv_len=kv_len,
        tokenizer_files=tokenizer_files,
        prefill_basename=pipeline_sliding_prefill_filename(window_size, kv_len),
        decode_basename=pipeline_sliding_decode_filename(kv_len),
    )
    cfg_path = pipeline_dir / "genai_config.json"
    with open(cfg_path, "w") as f:
        json.dump(config_dict, f, indent=4)
    return pipeline_dir


# ── Generation loop helpers ─────────────────────────────────────────────────


def make_prefill_inputs(cfg, prompt_tokens, max_seq_len):
    seq_len = len(prompt_tokens)
    inputs = {
        "input_ids": np.array([prompt_tokens], dtype=np.int64),
        "attention_mask": np.zeros((1, max_seq_len), dtype=np.int64),
    }
    inputs["attention_mask"][0, :seq_len] = 1
    if cfg.has_position_ids:
        inputs["position_ids"] = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
    for i in range(cfg.num_kv_layers):
        inputs[f"past_key_values.{i}.key"] = np.zeros(
            (1, cfg.num_kv_heads, max_seq_len, cfg.head_dim), dtype=np.float16
        )
        inputs[f"past_key_values.{i}.value"] = np.zeros(
            (1, cfg.num_kv_heads, max_seq_len, cfg.head_dim), dtype=np.float16
        )
    return inputs


def make_decode_inputs(cfg, token_id, position, kv_cache, max_seq_len):
    inputs = {
        "input_ids": np.array([[token_id]], dtype=np.int64),
        "attention_mask": np.zeros((1, max_seq_len), dtype=np.int64),
    }
    inputs["attention_mask"][0, : position + 1] = 1
    if cfg.has_position_ids:
        inputs["position_ids"] = np.array([[position]], dtype=np.int64)
    for i in range(cfg.num_kv_layers):
        inputs[f"past_key_values.{i}.key"] = kv_cache[(i, "key")]
        inputs[f"past_key_values.{i}.value"] = kv_cache[(i, "value")]
    return inputs


def extract_kv_cache(outputs, output_names):
    kv = {}
    for idx, name in enumerate(output_names):
        if name.startswith("present."):
            parts = name.split(".")
            kv[(int(parts[1]), parts[2])] = outputs[idx]
    return kv


def get_next_token(logits):
    return int(np.argmax(logits[0, -1, :].astype(np.float32)))


def compare_logits(ref_logits, test_logits, step_name):
    ref = ref_logits.astype(np.float32).flatten()
    test = test_logits.astype(np.float32).flatten()
    finite = np.isfinite(ref) & np.isfinite(test)
    ref_f, test_f = ref[finite], test[finite]
    if ref_f.size == 0:
        print(f"  {step_name}: no finite elements [SKIP]")
        return True
    dot = np.dot(ref_f, test_f)
    norms = np.linalg.norm(ref_f) * np.linalg.norm(test_f)
    cosine = float(dot / norms) if norms > 0 else 1.0
    ref_norm = np.linalg.norm(ref_f)
    diff_norm = np.linalg.norm(ref_f - test_f)
    rel_l2 = float(diff_norm / ref_norm) if ref_norm > 0 else 0.0
    top1_match = np.argmax(ref_f) == np.argmax(test_f)
    ok = cosine > 0.95
    status = "OK" if ok else "FAIL"
    print(
        f"  {step_name}: cosine={cosine:.6f}  rel_L2={rel_l2:.4e}"
        f"  top1_match={top1_match}  [{status}]"
    )
    return ok


# ── OGA helpers ─────────────────────────────────────────────────────────────


def make_prompt_tokens(bos_token, filler_tokens, length):
    return [bos_token] + list(
        itertools.islice(itertools.cycle(filler_tokens), length - 1)
    )


_oga_ep_registered = False


def setup_oga_ep(repo_root):
    try:
        import onnxruntime_genai as og
    except ImportError:
        pytest.skip("onnxruntime-genai not installed")

    if not hasattr(og, "register_execution_provider_library"):
        pytest.skip("OGA version does not support custom EP registration")

    dist_bin = repo_root / "install" / "dist" / "bin"
    therock_bin = repo_root / "install" / "therock" / "bin"
    ep_dll = dist_bin / "onnxruntime_morphizen_ep.dll"
    if not ep_dll.exists():
        pytest.skip("MorphiZen EP DLL not found — run build.py first")

    for d in [dist_bin, therock_bin]:
        if d.exists() and str(d) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")

    global _oga_ep_registered
    if not _oga_ep_registered:
        og.register_execution_provider_library("MorphiZenEP", str(ep_dll))
        _oga_ep_registered = True

    return og, ep_dll


def patch_genai_config_for_morphizen(model_dir, ep_dll):
    config_path = model_dir / "genai_config.json"
    backup_path = model_dir / "genai_config.json.bak"
    if not config_path.exists():
        return
    shutil.copy2(config_path, backup_path)
    with open(config_path) as f:
        config = json.load(f)
    config["model"]["decoder"]["session_options"]["provider_options"] = [
        {"MorphiZenEP": {}}
    ]
    # Pareto-optimal chunk_size for dynamic-shape Llama on MorphiZenEP.
    # 8B sweep on gfx1151 (2026-05-06): chunk=1024 beats chunk=512 on long
    # prompts (-1.4% TTFT at L=2048, -3.6% at L=4096) at +0.16 GB peak WS at
    # the worst L. At L<=1024 it's a no-op (single chunk). Decode is invariant.
    # Skip for fixed-shape configs — decoder-pipeline / sliding_window /
    # fixed_prompt_length already chunk via their own mechanisms, and injecting
    # search.chunk_size on top would either be ignored or fight them.
    decoder = config["model"].get("decoder", {})
    is_fixed_shape = (
        config["model"].get("type") == "decoder-pipeline"
        or "sliding_window" in decoder
        or "fixed_prompt_length" in decoder
    )
    if not is_fixed_shape:
        config.setdefault("search", {})["chunk_size"] = 1024
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def restore_genai_config(model_dir):
    config_path = model_dir / "genai_config.json"
    backup_path = model_dir / "genai_config.json.bak"
    if backup_path.exists():
        shutil.move(backup_path, config_path)


def oga_generate(og, model, tokenizer, prompt_tokens, max_new=30):
    """Generate tokens with OGA, return (tokens, ttft_ms, tps)."""
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(prompt_tokens) + max_new, do_sample=False)
    generator = og.Generator(model, params)

    t0 = time.perf_counter()
    generator.append_tokens(np.array(prompt_tokens, dtype=np.int32))
    generator.generate_next_token()
    ttft_ms = (time.perf_counter() - t0) * 1000
    generated = [int(generator.get_next_tokens()[0])]

    if generator.is_done() or len(generated) >= max_new:
        return generated, ttft_ms, 0.0

    t_decode = time.perf_counter()
    while not generator.is_done() and len(generated) < max_new:
        generator.generate_next_token()
        generated.append(int(generator.get_next_tokens()[0]))
    decode_ms = (time.perf_counter() - t_decode) * 1000

    n_decode = len(generated) - 1
    tps = n_decode / (decode_ms / 1000) if decode_ms > 0 and n_decode > 0 else 0.0
    return generated, ttft_ms, tps


def oga_generate_timed(og, model, tokenizer, prompt_tokens, max_new=128):
    """Benchmark OGA generation, methodology aligned with model_benchmark.exe.

    Uses min_length to force exact token count and a clean decode loop
    (no get_next_tokens / list building inside timed section).
    """
    total_len = len(prompt_tokens) + max_new
    prompt_np = np.array(prompt_tokens, dtype=np.int32)

    params = og.GeneratorParams(model)
    params.set_search_options(
        max_length=total_len,
        min_length=total_len,
        do_sample=False,
    )
    generator = og.Generator(model, params)

    # Warmup: full prefill + all decode tokens (same generator object)
    generator.append_tokens(prompt_np)
    while not generator.is_done():
        generator.generate_next_token()
    generator.rewind_to(0)

    # Timed prefill (TTFT)
    t0 = time.perf_counter()
    generator.append_tokens(prompt_np)
    generator.generate_next_token()
    ttft_ms = (time.perf_counter() - t0) * 1000

    # Timed decode — only generate_next_token() + is_done() in hot loop
    n_decode = 0
    t_decode = time.perf_counter()
    while not generator.is_done():
        generator.generate_next_token()
        n_decode += 1
    decode_ms = (time.perf_counter() - t_decode) * 1000
    tps = n_decode / (decode_ms / 1000) if decode_ms > 0 and n_decode > 0 else 0.0

    generated = n_decode + 1  # +1 for the prefill token

    del generator
    return generated, ttft_ms, tps


def run_cpu_reference_generation(
    model_path, cfg, prompt_tokens, max_new, max_seq_len=None
):
    """CPU prefill + (max_new-1) decode steps via sess.run(). Returns token list.

    Used as accuracy reference for OGA-based generation tests. The session is
    created and torn down inside this helper to keep CPU memory transient.
    """
    if max_seq_len is None:
        max_seq_len = len(prompt_tokens) + max_new
    cpu_sess = create_cpu_session(model_path)
    output_names = [o.name for o in cpu_sess.get_outputs()]
    logits_idx = output_names.index("logits")

    cpu_inputs = make_prefill_inputs(cfg, prompt_tokens, max_seq_len)
    cpu_out = cpu_sess.run(None, cpu_inputs)
    cpu_token = get_next_token(cpu_out[logits_idx])
    cpu_kv = extract_kv_cache(cpu_out, output_names)
    generated = [cpu_token]

    for step in range(max_new - 1):
        position = len(prompt_tokens) + step
        cpu_inputs = make_decode_inputs(cfg, cpu_token, position, cpu_kv, max_seq_len)
        cpu_out = cpu_sess.run(None, cpu_inputs)
        cpu_token = get_next_token(cpu_out[logits_idx])
        cpu_kv = extract_kv_cache(cpu_out, output_names)
        generated.append(cpu_token)

    cleanup(cpu_sess)
    return generated


def run_oga_static_kv_pipeline(og, pipeline_dir, prompt_tokens, kv_len, max_new):
    """Run an OGA decoder-pipeline model with a static KV buffer of `kv_len`.

    For fixed-shape pipelines, `max_length` MUST equal the model's static
    `total_sequence_length` (= kv_len) — see CLAUDE.md "OGA static-mask sizing".
    Generation length is bounded via `min_length`. Returns (tokens, ttft_ms).
    """
    try:
        model = og.Model(str(pipeline_dir))
    except RuntimeError as e:
        if "Unknown provider name" in str(e):
            pytest.skip("OGA does not recognize MorphiZen EP")
        raise

    try:
        params = og.GeneratorParams(model)
        params.set_search_options(
            max_length=kv_len,
            min_length=len(prompt_tokens) + max_new,
            do_sample=False,
        )
        generator = og.Generator(model, params)
        generator.append_tokens(np.array(prompt_tokens, dtype=np.int32))

        # First generate_next_token triggers all prefill chunk dispatches
        # inside one Generator::Run() (sliding_window path) plus emits the
        # first sampled token.
        t0 = time.perf_counter()
        generator.generate_next_token()
        ttft_ms = (time.perf_counter() - t0) * 1000

        generated = [int(generator.get_next_tokens()[0])]
        while not generator.is_done() and len(generated) < max_new:
            generator.generate_next_token()
            generated.append(int(generator.get_next_tokens()[0]))

        del generator
    finally:
        del model
        gc.collect()

    return generated, ttft_ms


# ── Shared fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT
