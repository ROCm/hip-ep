#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Shared fixtures and helpers for Python performance tests."""

import os
import pathlib
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


def make_llama_inputs(cfg: LlamaModelConfig) -> dict:
    """Create decode-step inputs for a Llama model with pre-allocated KV cache."""
    inputs = {
        "input_ids": np.array([[1]], dtype=np.int64),
        "attention_mask": np.ones((1, cfg.max_seq_len), dtype=np.int64),
    }
    if cfg.has_position_ids:
        inputs["position_ids"] = np.array([[cfg.max_seq_len - 1]], dtype=np.int64)
    for i in range(cfg.num_kv_layers):
        inputs[f"past_key_values.{i}.key"] = np.zeros(
            (1, cfg.num_kv_heads, cfg.max_seq_len, cfg.head_dim), dtype=np.float16
        )
        inputs[f"past_key_values.{i}.value"] = np.zeros(
            (1, cfg.num_kv_heads, cfg.max_seq_len, cfg.head_dim), dtype=np.float16
        )
    return inputs


def run_timed_iobinding(
    sess, inputs, cfg: LlamaModelConfig, warmup=NUM_WARMUP, runs=NUM_RUNS
):
    """Run inference using IOBinding with shared past/present KV cache buffers.

    Binds the same OrtValue to both past_key_values.N.{key,value} (input) and
    present.N.{key,value} (output) so the runtime sees past_ptr == present_ptr.
    """
    io = sess.io_binding()

    kv_shape = (1, cfg.num_kv_heads, cfg.max_seq_len, cfg.head_dim)
    kv_cache = {}
    for i in range(cfg.num_kv_layers):
        for kind in ("key", "value"):
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
        output_vals[o.name] = ort.OrtValue.ortvalue_from_shape_and_type(o.shape, dtype)

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


# ── Shared fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT
