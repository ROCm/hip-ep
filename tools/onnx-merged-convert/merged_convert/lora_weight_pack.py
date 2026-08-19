#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""LoRA adapter export helpers for merged convert.

MatMulNBits graphs: pack int8 ``[K,N]`` weights to uint8 ``[N,K]`` for
``weight_quantized`` inputs.

Folded Gemm graphs: dequantize int8 adapter weights to fp16 ``weight_fp16``
inputs at export time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
from onnx import TensorProto


def pack_lora_weight_mnbits_int8(raw: np.ndarray) -> np.ndarray:
    """Pack raw signed int8 ``[K, N]`` into MatMulNBits uint8 ``[N, K]``."""
    if raw.ndim != 2:
        raise ValueError(f"expected rank-2 [K,N] weight, got shape {raw.shape}")
    return (raw.astype(np.int16) + 128).astype(np.uint8).T


def dequant_lora_weight_int8_to_fp16(
    raw: np.ndarray, *, scale: float, zero_point: int
) -> np.ndarray:
    """Dequantize signed int8 LoRA weights to fp16 (matches runtime DQ formula)."""
    fp32 = (raw.astype(np.float32) - np.float32(zero_point)) * np.float32(scale)
    return fp32.astype(np.float16)


def export_dequant_adapter_safetensors(
    src: Path,
    dst: Path,
    port_meta: dict[str, dict],
) -> int:
    """Dequantize int8 adapter tensors to fp16, keyed by graph ``onnx_input`` names."""
    try:
        from safetensors.numpy import save_file
    except ImportError as exc:
        raise ImportError("safetensors is required; pip install safetensors") from exc

    if not port_meta:
        return 0

    raw = load_adapter_int8_tensors(src)
    missing = set(port_meta) - set(raw)
    if missing:
        raise KeyError(
            f"adapter missing {len(missing)} requested ports, e.g. {next(iter(missing))!r}"
        )

    dequantized: dict[str, np.ndarray] = {}
    for weight_quantized, meta in port_meta.items():
        onnx_input = meta["onnx_input"]
        dequantized[onnx_input] = dequant_lora_weight_int8_to_fp16(
            raw[weight_quantized],
            scale=float(meta["scale"]),
            zero_point=int(meta["zero_point"]),
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(dequantized, str(dst))
    return len(dequantized)


def write_dequant_lora_artifacts(
    adapter_src: Path,
    output_dir: Path,
    port_meta: dict[str, dict],
) -> list[str]:
    """Export fp16 ``adapter.safetensors`` for folded Gemm LoRA graphs."""
    if not port_meta:
        return []

    if not adapter_src.is_file():
        raise FileNotFoundError(
            f"merged Gemm LoRA graph has {len(port_meta)} adapter ports but adapter missing: "
            f"{adapter_src}"
        )

    packed_out = output_dir / "adapter.safetensors"
    count = export_dequant_adapter_safetensors(adapter_src, packed_out, port_meta)
    return [f"adapter.safetensors ({count} fp16 tensors)"]


def load_adapter_int8_tensors(adapter_path: Path) -> dict[str, np.ndarray]:
    """Load raw ``adapter.safetensors`` int8 LoRA weights keyed by ONNX input name."""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError(
            "safetensors is required for LoRA adapter export; pip install safetensors"
        ) from exc

    tensors: dict[str, np.ndarray] = {}
    with safe_open(str(adapter_path), framework="numpy") as f:
        for key in f.keys():
            arr = np.asarray(f.get_tensor(key))
            if arr.dtype != np.int8:
                raise ValueError(
                    f"adapter tensor {key!r}: expected int8 raw weight, got {arr.dtype}"
                )
            tensors[key] = arr
    if not tensors:
        raise ValueError(f"no tensors in adapter file: {adapter_path}")
    return tensors


def export_packed_adapter_safetensors(
    src: Path,
    dst: Path,
    *,
    port_names: Iterable[str] | None = None,
) -> int:
    """Pack raw int8 ``[K,N]`` adapter tensors into uint8 ``[N,K]``, same keys."""
    try:
        from safetensors.numpy import save_file
    except ImportError as exc:
        raise ImportError("safetensors is required; pip install safetensors") from exc

    raw = load_adapter_int8_tensors(src)
    if port_names is not None:
        allowed = set(port_names)
        raw = {k: v for k, v in raw.items() if k in allowed}
        missing = allowed - set(raw)
        if missing:
            raise KeyError(
                f"adapter missing {len(missing)} requested ports, e.g. {next(iter(missing))!r}"
            )

    packed = {name: pack_lora_weight_mnbits_int8(arr) for name, arr in raw.items()}
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(packed, str(dst))
    return len(packed)


def retarget_weight_quantized_graph_input(
    graph: onnx.GraphProto, port_name: str, k: int, n: int
) -> str:
    """Keep ``weight_quantized`` name; change type/shape to packed uint8 ``[N,K]``."""
    for vi in graph.input:
        if vi.name != port_name:
            continue
        elem = vi.type.tensor_type
        elem.elem_type = TensorProto.UINT8
        del elem.shape.dim[:]
        elem.shape.dim.add().dim_value = n
        elem.shape.dim.add().dim_value = k
        return port_name
    raise KeyError(f"graph input not found: {port_name!r}")


def collect_packed_weight_quantized_specs(
    graph: onnx.GraphProto,
) -> list[tuple[str, tuple[int, int]]]:
    """``weight_quantized`` ports as uint8 ``[N,K]``."""
    specs: list[tuple[str, tuple[int, int]]] = []
    for vi in graph.input:
        if "weight_quantized" not in vi.name:
            continue
        if vi.type.tensor_type.elem_type != TensorProto.UINT8:
            continue
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        if len(dims) == 2 and all(dims):
            specs.append((vi.name, (dims[0], dims[1])))
    return specs


def write_packed_lora_artifacts(
    merged_path: Path,
    adapter_src: Path,
    output_dir: Path,
) -> list[str]:
    """Export packed ``adapter.safetensors`` for a merged LoRA MatMulNBits graph."""
    model = onnx.load(str(merged_path), load_external_data=False)
    specs = collect_packed_weight_quantized_specs(model.graph)
    if not specs:
        return []

    if not adapter_src.is_file():
        raise FileNotFoundError(
            f"merged model has {len(specs)} LoRA weight_quantized ports but adapter missing: "
            f"{adapter_src}"
        )

    port_names = [name for name, _ in specs]
    packed_out = output_dir / "adapter.safetensors"
    count = export_packed_adapter_safetensors(
        adapter_src, packed_out, port_names=port_names
    )
    return [f"adapter.safetensors ({count} packed tensors)"]
