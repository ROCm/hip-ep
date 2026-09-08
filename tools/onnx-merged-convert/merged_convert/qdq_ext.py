#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.compose import add_prefix, merge_graphs

from .step2_fp16_cleanup import optimize_fp16_activations
from .step3_lm_head import rewrite_lm_head_gather_unsqueeze

FP32_q = TensorProto.FLOAT
FP64_q = TensorProto.DOUBLE
FP16_q = TensorProto.FLOAT16
DEFAULT_BATCH_q = 1
DEFAULT_SEQ_LEN_q = 128
DEFAULT_MAX_SEQ_LEN_q = 16384
_INT4_UNSIGNED_OFFSET_q = 8
DEFAULT_ACCURACY_LEVEL_q = 1
MATMUL_NBITS_ACCURACY_LEVEL_q = DEFAULT_ACCURACY_LEVEL_q
_WEIGHT_DQ_CONSUMER_INPUTS: dict[str, tuple[int, ...]] = {
    "Mul": (0, 1),
    "Conv": (1,),
    "Add": (0, 1),
}
DECODER_EXT_DATA = "decoder_4bit.data"


def _dim_param_bindings_q(batch: int, seq_len: int, max_seq_len: int) -> dict[str, int]:
    return {"batch": batch, "seq_len": seq_len, "max_seq_len": max_seq_len}


def _fix_value_info_shapes_q(vi: onnx.ValueInfoProto, bindings: dict[str, int]) -> None:
    for dim in vi.type.tensor_type.shape.dim:
        if dim.dim_param and dim.dim_param in bindings:
            dim.dim_value = bindings[dim.dim_param]
            dim.ClearField("dim_param")


@dataclass(frozen=True)
class ShapeFixConfig_q:
    """Optional static-shape binding for symbolic ONNX dimensions."""

    enabled: bool = False
    batch: int = DEFAULT_BATCH_q
    seq_len: int = DEFAULT_SEQ_LEN_q
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN_q


@dataclass(frozen=True)
class ConvertProfile:
    """Which conversion passes to run (``lite`` = QDQ + Conv/MNB + fp16 + merge only)."""

    strip_qdq: bool = True
    conv_mnbits: bool = True
    weight_dq_fold: bool = True
    fp16: bool = True
    fp32_activations: bool = False
    emb_bits4: bool = True
    lm_head_gather: bool = True
    gqa_fp16_no_quant: bool = True
    gqa_seqlens_rewrite: bool = True
    matmul_nbits_accuracy: bool = True
    lpnorm_fp32: bool = True
    gqa_pre_cast_fp32_remove: bool = True
    activation_qdq_mode: str = "emu"


CONVERT_PROFILE_FULL = ConvertProfile()
CONVERT_PROFILE_LITE = ConvertProfile(
    gqa_fp16_no_quant=False,
    matmul_nbits_accuracy=False,
    lpnorm_fp32=False,
    activation_qdq_mode="bypass",
)
CONVERT_PROFILE_LOW_BIT_INTERNAL = ConvertProfile(
    gqa_fp16_no_quant=False,
    matmul_nbits_accuracy=False,
    lpnorm_fp32=False,
    gqa_seqlens_rewrite=False,
    gqa_pre_cast_fp32_remove=False,
    fp16=True,
    fp32_activations=False,
    activation_qdq_mode="bypass",
)


def maybe_fix_static_shapes_q(
    model: onnx.ModelProto, shape_fix: ShapeFixConfig_q
) -> None:
    """Bind symbolic dims when ``shape_fix.enabled``."""
    if not shape_fix.enabled:
        return
    bindings = _dim_param_bindings_q(
        shape_fix.batch, shape_fix.seq_len, shape_fix.max_seq_len
    )
    graph = model.graph
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        _fix_value_info_shapes_q(vi, bindings)


def fix_static_shapes_q(
    model: onnx.ModelProto,
    *,
    batch: int = DEFAULT_BATCH_q,
    seq_len: int = DEFAULT_SEQ_LEN_q,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN_q,
) -> None:
    maybe_fix_static_shapes_q(
        model,
        ShapeFixConfig_q(
            enabled=True, batch=batch, seq_len=seq_len, max_seq_len=max_seq_len
        ),
    )


def _nibble_pack_int4_q(values: np.ndarray) -> np.ndarray:
    nibbles = values.astype(np.int16) + _INT4_UNSIGNED_OFFSET_q & 15
    return nibbles[0::2].astype(np.uint8) | nibbles[1::2].astype(np.uint8) << 4


def choose_block_size_q(k: int) -> int:
    """Pick a MatMulNBits-legal block_size (power of 2, >= 16) for a K dim.

    Prefer the largest such block_size that divides K (no padding needed);
    fall back to 32 with zero-padded trailing block when none divides.
    """
    for bs in (256, 128, 64, 32, 16):
        if k % bs == 0:
            return bs
    return 32


def pack_matmul_nbits_weight_q(
    w_kn: np.ndarray, block_size: int
) -> tuple[np.ndarray, int, int, int]:
    k, n = w_kn.shape
    if k % 2 != 0:
        raise ValueError(f"K must be even for 4-bit packing, got K={k}")
    n_blocks = (k + block_size - 1) // block_size
    blob_size = block_size // 2
    packed = np.zeros((n, n_blocks, blob_size), dtype=np.uint8)
    for col in range(n):
        col_bytes = _nibble_pack_int4_q(w_kn[:, col])
        flat = np.zeros(n_blocks * blob_size, dtype=np.uint8)
        flat[: col_bytes.size] = col_bytes
        packed[col] = flat.reshape(n_blocks, blob_size)
    return (packed, k, n, n_blocks)


def pack_zero_points_q(z_n: np.ndarray, n: int, n_blocks: int) -> np.ndarray:
    """Pack one (per-channel) 4-bit zero point, replicated across all blocks.

    Layout matches MatMulNBits: per column, n_blocks nibbles packed two-per-byte
    (even block -> low nibble, odd block -> high nibble).
    """
    zp_bytes = (n_blocks + 1) // 2
    out = np.zeros((n, zp_bytes), dtype=np.uint8)
    for col in range(n):
        v = int(z_n[col]) + _INT4_UNSIGNED_OFFSET_q & 15
        for b in range(n_blocks):
            if b % 2 == 0:
                out[col, b // 2] |= v
            else:
                out[col, b // 2] |= v << 4
    return out


_SIGNED_INT8_MNBITS_OFFSET = 128


def pack_matmul_nbits_weight_int8(
    w_kn: np.ndarray, block_size: int
) -> tuple[np.ndarray, int, int, int]:
    """Pack signed int8 weight [K, N] into MatMulNBits uint8 blocks [N, n_blocks, block_size]."""
    k, n = w_kn.shape
    n_blocks = (k + block_size - 1) // block_size
    padded_k = n_blocks * block_size
    packed = np.zeros((n, n_blocks, block_size), dtype=np.uint8)
    for col in range(n):
        col_w = w_kn[:, col].astype(np.int16)
        if col_w.size < padded_k:
            col_w = np.pad(col_w, (0, padded_k - col_w.size))
        packed[col] = (
            (col_w + _SIGNED_INT8_MNBITS_OFFSET)
            .astype(np.uint8)
            .reshape(n_blocks, block_size)
        )
    return (packed, k, n, n_blocks)


def _int8_mnbits_zero_points(n: int, n_blocks: int) -> np.ndarray:
    return np.full((n, n_blocks), _SIGNED_INT8_MNBITS_OFFSET, dtype=np.uint8)


def convert_matmul_int8_weight_dq(
    graph: onnx.GraphProto,
    inits: dict[str, onnx.TensorProto],
    new_inits: dict[str, onnx.TensorProto],
    remove_inits: set[str],
) -> int:
    """Replace ``MatMul(int8 DQ weight)`` with ``MatMulNBits`` (bits=8)."""
    dq_by_out = {d.output[0]: d for d in graph.node if d.op_type == "DequantizeLinear"}
    replacements: dict[str, onnx.NodeProto] = {}
    remove_nodes: set[str] = set()
    converted = 0
    for mm in graph.node:
        if mm.op_type != "MatMul" or len(mm.input) < 2:
            continue
        dq = dq_by_out.get(mm.input[1])
        if dq is None or dq.input[0] not in inits:
            continue
        w_init = inits[dq.input[0]]
        if w_init.data_type != TensorProto.INT8:
            continue
        w = numpy_helper.to_array(w_init)
        if w.ndim != 2:
            continue
        k, n = (int(w.shape[0]), int(w.shape[1]))
        scale, zp = _expand_scale_zp_q(
            numpy_helper.to_array(inits[dq.input[1]]),
            numpy_helper.to_array(inits[dq.input[2]]),
            n,
        )
        block_size = choose_block_size_q(k)
        packed, k, n, n_blocks = pack_matmul_nbits_weight_int8(w, block_size)
        base = w_init.name.removesuffix("_quantized")
        q_name = f"{base}_mnbits_q8"
        s_name = f"{base}_mnbits_scales"
        z_name = f"{base}_mnbits_zp"
        new_inits[q_name] = numpy_helper.from_array(packed, name=q_name)
        new_inits[s_name] = numpy_helper.from_array(
            np.repeat(scale.astype(np.float16).reshape(n, 1), n_blocks, axis=1),
            name=s_name,
        )
        new_inits[z_name] = numpy_helper.from_array(
            _int8_mnbits_zero_points(n, n_blocks), name=z_name
        )
        replacements[mm.name] = helper.make_node(
            "MatMulNBits",
            [mm.input[0], q_name, s_name, z_name],
            list(mm.output),
            name=f"{mm.name}_mnbits",
            domain="com.microsoft",
            K=k,
            N=n,
            bits=8,
            block_size=block_size,
            accuracy_level=MATMUL_NBITS_ACCURACY_LEVEL_q,
        )
        remove_nodes.update({mm.name, dq.name})
        remove_inits.update({w_init.name, dq.input[1], dq.input[2]})
        converted += 1
    if converted:
        kept: list[onnx.NodeProto] = []
        for node in graph.node:
            if node.name in replacements:
                kept.append(replacements[node.name])
            elif node.name not in remove_nodes:
                kept.append(node)
        del graph.node[:]
        graph.node.extend(kept)
    return converted


def dequantize_initializer_q(
    w: np.ndarray, scale: np.ndarray, zp: np.ndarray, *, out_dtype: type = np.float16
) -> np.ndarray:
    w_f = w.astype(np.float64)
    s_f = np.asarray(scale, dtype=np.float64)
    z_f = np.asarray(zp, dtype=np.float64)
    return ((w_f - z_f) * s_f).astype(out_dtype)


def _replace_tensor_name_q(
    users: dict[str, list[tuple[onnx.NodeProto, int]]], old: str, new: str
) -> None:
    for node, idx in users.get(old, []):
        if node.input[idx] == old:
            node.input[idx] = new


def _build_tensor_users_q(
    graph: onnx.GraphProto,
) -> dict[str, list[tuple[onnx.NodeProto, int]]]:
    users: dict[str, list[tuple[onnx.NodeProto, int]]] = {}
    for node in graph.node:
        for idx, name in enumerate(node.input):
            if name:
                users.setdefault(name, []).append((node, idx))
    return users


def _referenced_initializer_names_q(graph: onnx.GraphProto) -> set[str]:
    used: set[str] = set()
    for node in graph.node:
        for name in node.input:
            if name:
                used.add(name)
    return used


def _rename_value_tensor_q(
    graph: onnx.GraphProto,
    users: dict[str, list[tuple[onnx.NodeProto, int]]],
    old: str,
    new: str,
) -> None:
    if old == new:
        return
    for node in graph.node:
        for idx, out_name in enumerate(node.output):
            if out_name == old:
                node.output[idx] = new
    _replace_tensor_name_q(users, old, new)
    for vi in graph.value_info:
        if vi.name == old:
            vi.name = new


def _activation_qdq_source(graph: onnx.GraphProto, tensor: str) -> str:
    """Prefer fp16 ``*_pre_cast`` over an upstream Cast(fp32) fake-quant input."""
    producer: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for out in node.output:
            if out:
                producer[out] = node
    node = producer.get(tensor)
    if node is None or node.op_type != "Cast" or (not node.input):
        return tensor
    if "pre_cast_castfp32" not in node.name:
        return tensor
    return node.input[0]


def _cast_to_q(node: onnx.NodeProto) -> int | None:
    if node.op_type != "Cast":
        return None
    for attr in node.attribute:
        if attr.name == "to":
            return int(attr.i)
    return None


def unwrap_castfp32_before_quantize(graph: onnx.GraphProto) -> int:
    """Point QuantizeLinear at fp16 ``*_pre_cast`` instead of Cast(fp32) inputs."""
    producer: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for out in node.output:
            if out:
                producer[out] = node
    remove: set[str] = set()
    unwrapped = 0
    for q in graph.node:
        if q.op_type != "QuantizeLinear" or not q.input:
            continue
        cast = producer.get(q.input[0])
        if cast is None or cast.op_type != "Cast" or _cast_to_q(cast) != FP32_q:
            continue
        if "pre_cast_castfp32" not in cast.name or not cast.input:
            continue
        if "GroupQueryAttention_output_" in cast.name:
            continue
        q.input[0] = cast.input[0]
        remove.add(cast.name)
        unwrapped += 1
    if remove:
        kept = [n for n in graph.node if n.name not in remove]
        del graph.node[:]
        graph.node.extend(kept)
    return unwrapped


def _fp16_zp_power_terms(zp: float) -> list[float]:
    """Decompose integer zero-point into float16-exact power-of-two summands."""
    remaining = int(round(zp))
    if remaining == 0:
        return [0.0]
    terms: list[float] = []
    sign = 1 if remaining >= 0 else -1
    remaining = abs(remaining)
    for bit in range(20):
        mask = 1 << bit
        if remaining & mask:
            terms.append(float(np.float16(sign * mask)))
            remaining -= mask
    if remaining:
        terms.append(float(np.float16(sign * remaining)))
    return terms


def _fp16_scale(scale: float) -> float:
    """Return scale as float16 (Div/Mul emulation; avoid 1/scale which overflows fp16)."""
    return float(np.float16(np.float32(scale)))


def _fp16_zp(zp_init: onnx.TensorProto) -> float:
    """Fold integer zero-point into one fp16 constant (convert-time power-sum)."""
    zp = float(np.asarray(numpy_helper.to_array(zp_init)).reshape(-1)[0])
    terms = _fp16_zp_power_terms(zp)
    acc = np.float32(0.0)
    for term in terms:
        acc = np.float32(acc + np.float32(term))
    return float(np.float16(acc))


def _insert_pure_fp16_qdq_emulation(
    *,
    x: str,
    y: str,
    scale_init: onnx.TensorProto,
    zp_init: onnx.TensorProto,
    prefix: str,
    new_inits: dict[str, onnx.TensorProto],
) -> list[onnx.NodeProto]:
    """Fake-quant in pure fp16: no Q/DQ ops, no fp32 activation tensors."""
    scale = float(np.asarray(numpy_helper.to_array(scale_init)).reshape(-1)[0])
    scale_f16 = _fp16_scale(scale)
    zp_f16 = _fp16_zp(zp_init)
    scale_name = f"{prefix}_emu_scale"
    zp_name = f"{prefix}_emu_zp"
    new_inits[scale_name] = numpy_helper.from_array(
        np.array(scale_f16, dtype=np.float16), name=scale_name
    )
    new_inits[zp_name] = numpy_helper.from_array(
        np.array(zp_f16, dtype=np.float16), name=zp_name
    )
    t_div = f"{prefix}_emu_div"
    t_add = f"{prefix}_emu_add"
    t_round = f"{prefix}_emu_round"
    t_sub = f"{prefix}_emu_sub"
    return [
        helper.make_node("Div", [x, scale_name], [t_div], name=f"{prefix}_emu_div"),
        helper.make_node("Add", [t_div, zp_name], [t_add], name=f"{prefix}_emu_add"),
        helper.make_node("Round", [t_add], [t_round], name=f"{prefix}_emu_round"),
        helper.make_node("Sub", [t_round, zp_name], [t_sub], name=f"{prefix}_emu_sub"),
        helper.make_node("Mul", [t_sub, scale_name], [y], name=f"{prefix}_emu_mul_y"),
    ]


def _quantize_scale_zp(
    scale_init: onnx.TensorProto,
    zp_init: onnx.TensorProto,
    *,
    scale_name: str,
    zp_name: str,
) -> tuple[float, float, onnx.TensorProto, onnx.TensorProto]:
    """Return scalar scale/zp and fp32 initializers for emulation."""
    scale = float(np.asarray(numpy_helper.to_array(scale_init)).reshape(-1)[0])
    zp = float(np.asarray(numpy_helper.to_array(zp_init)).reshape(-1)[0])
    scale_t = numpy_helper.from_array(
        np.array(scale, dtype=np.float32), name=scale_name
    )
    zp_t = numpy_helper.from_array(np.array(zp, dtype=np.float32), name=zp_name)
    return (scale, zp, scale_t, zp_t)


def _insert_qdq_emulation(
    *,
    x: str,
    y: str,
    scale_init: onnx.TensorProto,
    zp_init: onnx.TensorProto,
    prefix: str,
    new_inits: dict[str, onnx.TensorProto],
) -> list[onnx.NodeProto]:
    """Match ORT QuantizeLinear+DequantizeLinear (fp32 math, fp16 activation output)."""
    scale_name = f"{prefix}_emu_scale"
    zp_name = f"{prefix}_emu_zp"
    _, _, scale_t, zp_t = _quantize_scale_zp(
        scale_init, zp_init, scale_name=scale_name, zp_name=zp_name
    )
    new_inits[scale_name] = scale_t
    new_inits[zp_name] = zp_t
    x_fp32 = f"{prefix}_emu_xf32"
    t_div = f"{prefix}_emu_div"
    t_add = f"{prefix}_emu_add"
    t_round = f"{prefix}_emu_round"
    t_sub = f"{prefix}_emu_sub"
    return [
        helper.make_node(
            "Cast", [x], [x_fp32], name=f"{prefix}_emu_cast_in", to=FP32_q
        ),
        helper.make_node(
            "Div", [x_fp32, scale_name], [t_div], name=f"{prefix}_emu_div"
        ),
        helper.make_node("Add", [t_div, zp_name], [t_add], name=f"{prefix}_emu_add"),
        helper.make_node("Round", [t_add], [t_round], name=f"{prefix}_emu_round"),
        helper.make_node("Sub", [t_round, zp_name], [t_sub], name=f"{prefix}_emu_sub"),
        helper.make_node("Mul", [t_sub, scale_name], [y], name=f"{prefix}_emu_mul"),
    ]


def _rewire_tensor_uses(graph: onnx.GraphProto, old: str, new: str) -> None:
    """Replace ``old`` tensor name with ``new`` on node inputs and graph outputs."""
    for node in graph.node:
        for idx, inp in enumerate(node.input):
            if inp == old:
                node.input[idx] = new
    for out in graph.output:
        if out.name == old:
            out.name = new
    for vi in graph.value_info:
        if vi.name == old:
            vi.name = new


def strip_activation_qdq_q(
    graph: onnx.GraphProto,
    inits: dict[str, onnx.TensorProto],
    new_inits: dict[str, onnx.TensorProto],
    *,
    pure_fp16: bool = True,
    mode: str = "emu",
) -> int:
    """Remove activation Q/DQ pairs; bypass or emulate fake-quant without Q/DQ ops."""
    init_names = set(inits)
    dq_by_q_out = {
        d.input[0]: d for d in graph.node if d.op_type == "DequantizeLinear" and d.input
    }
    remove_nodes: set[str] = set()
    insert_nodes: list[onnx.NodeProto] = []
    removed = 0
    for q in graph.node:
        if q.op_type != "QuantizeLinear" or not q.input or len(q.input) < 3:
            continue
        if q.input[0] in init_names:
            continue
        dq = dq_by_q_out.get(q.output[0])
        if dq is None or not dq.output:
            continue
        scale_name, zp_name = (q.input[1], q.input[2])
        if scale_name not in inits or zp_name not in inits:
            continue
        src = q.input[0]
        dst = dq.output[0]
        prefix = q.name.replace("/", "_").replace(".", "_")
        if mode == "bypass":
            if src != dst:
                graph_outputs = {o.name for o in graph.output}
                if dst in graph_outputs:
                    insert_nodes.append(
                        helper.make_node(
                            "Identity", [src], [dst], name=f"{prefix}_qdq_bypass"
                        )
                    )
                else:
                    _rewire_tensor_uses(graph, dst, src)
        elif pure_fp16:
            insert_nodes.extend(
                _insert_pure_fp16_qdq_emulation(
                    x=src,
                    y=dst,
                    scale_init=inits[scale_name],
                    zp_init=inits[zp_name],
                    prefix=prefix,
                    new_inits=new_inits,
                )
            )
        else:
            insert_nodes.extend(
                _insert_qdq_emulation(
                    x=src,
                    y=dst,
                    scale_init=inits[scale_name],
                    zp_init=inits[zp_name],
                    prefix=prefix,
                    new_inits=new_inits,
                )
            )
        remove_nodes.add(q.name)
        remove_nodes.add(dq.name)
        removed += 1
    if insert_nodes:
        graph.node.extend(insert_nodes)
    if remove_nodes:
        kept = [n for n in graph.node if n.name not in remove_nodes]
        del graph.node[:]
        graph.node.extend(kept)
    return removed


def _expand_scale_zp_q(
    scale: np.ndarray, zp: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    if scale.ndim == 0:
        scale = np.full(n, float(scale), dtype=np.float32)
    if zp.ndim == 0:
        zp = np.full(n, int(zp), dtype=np.int8)
    return (scale, zp)


def fold_weight_dq_q(
    graph: onnx.GraphProto,
    inits: dict[str, onnx.TensorProto],
    new_inits: dict[str, onnx.TensorProto],
    remove_inits: set[str],
    *,
    weight_dtype: type = np.float16,
) -> int:
    """Fold weight/bias DQ into fp16/fp32 constants for Mul, Conv, and Add consumers."""
    dq_by_out = {d.output[0]: d for d in graph.node if d.op_type == "DequantizeLinear"}
    tensor_replace: dict[str, str] = {}
    remove_dq: set[str] = set()
    folded = 0
    for node in graph.node:
        input_indices = _WEIGHT_DQ_CONSUMER_INPUTS.get(node.op_type)
        if input_indices is None:
            continue
        for idx in input_indices:
            if idx >= len(node.input):
                continue
            dq = dq_by_out.get(node.input[idx])
            if dq is None or dq.input[0] not in inits:
                continue
            w_name = dq.input[0]
            w = numpy_helper.to_array(inits[w_name])
            if w.dtype.name == "int4":
                continue
            scale = numpy_helper.to_array(inits[dq.input[1]])
            zp = numpy_helper.to_array(inits[dq.input[2]])
            const_name = f"{w_name}_fp{('32' if weight_dtype == np.float32 else '16')}"
            new_inits[const_name] = numpy_helper.from_array(
                dequantize_initializer_q(w, scale, zp, out_dtype=weight_dtype),
                name=const_name,
            )
            tensor_replace[node.input[idx]] = const_name
            remove_dq.add(dq.output[0])
            remove_inits.update({w_name, dq.input[1], dq.input[2]})
            folded += 1
    if tensor_replace:
        for node in graph.node:
            for idx, inp in enumerate(node.input):
                if inp in tensor_replace:
                    node.input[idx] = tensor_replace[inp]
        kept = [
            n
            for n in graph.node
            if not (n.op_type == "DequantizeLinear" and n.output[0] in remove_dq)
        ]
        del graph.node[:]
        graph.node.extend(kept)
    return folded


def fold_mul_weight_dq_q(
    graph: onnx.GraphProto,
    inits: dict[str, onnx.TensorProto],
    new_inits: dict[str, onnx.TensorProto],
    remove_inits: set[str],
) -> int:
    """Backward-compatible alias for norm-weight DQ folding (Mul only)."""
    return fold_weight_dq_q(graph, inits, new_inits, remove_inits)


def set_matmul_nbits_accuracy_level_q(
    model: onnx.ModelProto, level: int | None = None
) -> int:
    """Force ``accuracy_level`` on every MatMulNBits node (incl. pre-existing ones).

    The exported lm_head ships as MatMulNBits with ``accuracy_level=4`` (int8
    activation), which degrades logits enough to pick wrong-but-adjacent tokens.
    """
    if level is None:
        level = MATMUL_NBITS_ACCURACY_LEVEL_q
    changed = 0
    for node in model.graph.node:
        if node.op_type != "MatMulNBits":
            continue
        attr = next((a for a in node.attribute if a.name == "accuracy_level"), None)
        if attr is not None:
            if attr.i != level:
                attr.i = level
                changed += 1
        else:
            node.attribute.extend([helper.make_attribute("accuracy_level", level)])
            changed += 1
    return changed


def _promote_value_info_elem_type_q(vi: onnx.ValueInfoProto) -> None:
    elem_type = vi.type.tensor_type.elem_type
    if elem_type in (FP32_q, FP64_q):
        vi.type.tensor_type.elem_type = FP16_q


def _replace_node_inputs(
    graph: onnx.GraphProto,
    old: str,
    new: str,
    *,
    skip_node_names: set[str] | None = None,
) -> int:
    skip = skip_node_names or set()
    replaced = 0
    for node in graph.node:
        if node.name in skip:
            continue
        for idx, inp in enumerate(node.input):
            if inp == old:
                node.input[idx] = new
                replaced += 1
    return replaced


def wrap_fp16_io_fp32_activations(model: onnx.ModelProto) -> dict[str, int]:
    """Expose fp16 graph I/O while keeping the decoder body in fp32 (ORT QDQ parity)."""
    graph = model.graph
    stats = {"input_cast": 0, "output_cast": 0}
    hidden_in = "input_hidden_states"
    if any((vi.name == hidden_in for vi in graph.input)):
        fp32_in = "input_hidden_states_compute_fp32"
        cast_in_name = "input_hidden_states_cast_fp32"
        for vi in graph.input:
            if vi.name == hidden_in:
                vi.type.tensor_type.elem_type = FP16_q
        graph.node.insert(
            0,
            helper.make_node(
                "Cast", [hidden_in], [fp32_in], name=cast_in_name, to=FP32_q
            ),
        )
        stats["input_cast"] = _replace_node_inputs(
            graph, hidden_in, fp32_in, skip_node_names={cast_in_name}
        )
    hidden_out = "output_hidden_states"
    producer: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for out in node.output:
            if out:
                producer[out] = node
    prod = producer.get(hidden_out)
    if prod is not None and prod.output and (prod.output[0] == hidden_out):
        fp32_buf = "output_hidden_states_compute_fp32"
        prod.output[0] = fp32_buf
        graph.node.append(
            helper.make_node(
                "Cast",
                [fp32_buf],
                [hidden_out],
                name="output_hidden_states_cast_fp16",
                to=FP16_q,
            )
        )
        stats["output_cast"] = 1
        for vi in graph.output:
            if vi.name == hidden_out:
                vi.type.tensor_type.elem_type = FP16_q
    return stats


def promote_model_to_fp16_q(
    model: onnx.ModelProto,
    *,
    skip_initializer_substrings: tuple[str, ...] = ("_emu_scale", "_emu_zp"),
) -> None:
    """Promote float32/float64 graph I/O and float initializers to float16."""
    graph = model.graph
    for vi in list(graph.input) + list(graph.output):
        _promote_value_info_elem_type_q(vi)
    for init in graph.initializer:
        if any((part in init.name for part in skip_initializer_substrings)):
            continue
        if init.data_type == FP32_q:
            arr = numpy_helper.to_array(init).astype(np.float16)
            init.CopyFrom(numpy_helper.from_array(arr, name=init.name))
        elif init.data_type == FP64_q:
            arr = numpy_helper.to_array(init).astype(np.float16)
            init.CopyFrom(numpy_helper.from_array(arr, name=init.name))
    del graph.value_info[:]


def stabilize_lpnorm_fp16_q(model: onnx.ModelProto) -> int:
    """Force every LpNormalization to compute in fp32 via Cast wrappers.

    The DirectML EP computes LpNormalization's internal sum-of-squares in the
    tensor dtype (fp16); for large activations this overflows fp16 (>65504),
    yielding a wrong norm that amplifies layer-over-layer into NaNs (the CPU EP
    accumulates in fp32 and is unaffected). Wrapping each node with
    ``Cast(fp32) -> LpNormalization -> Cast(fp16)`` makes both EPs agree.
    """
    graph = model.graph
    if not any((n.op_type == "LpNormalization" for n in graph.node)):
        return 0
    new_nodes: list[onnx.NodeProto] = []
    count = 0
    for node in graph.node:
        if node.op_type != "LpNormalization":
            new_nodes.append(node)
            continue
        src = node.input[0]
        dst = node.output[0]
        pre = f"{node.name}_xf32"
        post = f"{node.name}_yf32"
        node.input[0] = pre
        node.output[0] = post
        new_nodes.append(
            helper.make_node(
                "Cast", [src], [pre], name=f"{node.name}_cast_in", to=FP32_q
            )
        )
        new_nodes.append(node)
        new_nodes.append(
            helper.make_node(
                "Cast", [post], [dst], name=f"{node.name}_cast_out", to=FP16_q
            )
        )
        count += 1
    del graph.node[:]
    graph.node.extend(new_nodes)
    return count


def apply_new_initializers_q(
    graph: onnx.GraphProto, new_inits: dict[str, onnx.TensorProto]
) -> None:
    existing = {i.name: idx for idx, i in enumerate(graph.initializer)}
    for name, tensor in new_inits.items():
        if name in existing:
            graph.initializer[existing[name]] = tensor
        else:
            graph.initializer.append(tensor)


def prune_initializers_q(graph: onnx.GraphProto, remove_inits: set[str]) -> None:
    used = _referenced_initializer_names_q(graph)
    kept = [
        i for i in graph.initializer if i.name in used and i.name not in remove_inits
    ]
    del graph.initializer[:]
    graph.initializer.extend(kept)


def rewrite_gqa_past_seq_len_to_seqlens_k_q(model: onnx.ModelProto) -> int:
    """Convert GQA input #5 from literal ``past_seq_len`` to ORT ``seqlens_k``.

    BUNDLE / runtime pass the count of tokens already in the KV cache on graph input
    ``past_seq_len``.  ORT GroupQueryAttention expects
    ``seqlens_k = past_len + seq_q - 1``.  Insert a small subgraph once and
    rewire every GQA node::

        seqlens_k = Reshape(past_seq_len, [1]) + Shape(query)[1] - 1
    """
    graph = model.graph
    gqa_nodes = [n for n in graph.node if n.op_type == "GroupQueryAttention"]
    if not gqa_nodes:
        return 0
    past_input = "past_seq_len"
    if not any((i.name == past_input for i in graph.input)):
        return 0
    if any((n.op_type == "Sub" and n.name == "gqa_seqlens_k_sub" for n in graph.node)):
        return 0
    if not all((len(n.input) > 5 and n.input[5] == past_input for n in gqa_nodes)):
        return 0
    query = gqa_nodes[0].input[0]
    one_name = "gqa_seqlens_k_const_one"
    past_reshape_shape = "gqa_past_reshape_shape"
    graph.initializer.extend(
        [
            numpy_helper.from_array(np.array([1], dtype=np.int32), name=one_name),
            numpy_helper.from_array(
                np.array([1], dtype=np.int64), name=past_reshape_shape
            ),
            numpy_helper.from_array(
                np.array(1, dtype=np.int64), name="gqa_gather_sq_idx"
            ),
        ]
    )
    new_nodes = [
        helper.make_node("Shape", [query], ["gqa_query_shape"], name="gqa_query_shape"),
        helper.make_node(
            "Gather",
            ["gqa_query_shape", "gqa_gather_sq_idx"],
            ["gqa_sq_i64"],
            name="gqa_gather_sq",
            axis=0,
        ),
        helper.make_node(
            "Cast",
            ["gqa_sq_i64"],
            ["gqa_sq_i32"],
            name="gqa_cast_sq",
            to=TensorProto.INT32,
        ),
        helper.make_node(
            "Reshape",
            [past_input, past_reshape_shape],
            ["gqa_past_1d"],
            name="gqa_past_reshape",
        ),
        helper.make_node(
            "Add",
            ["gqa_past_1d", "gqa_sq_i32"],
            ["gqa_past_plus_sq"],
            name="gqa_past_plus_sq",
        ),
        helper.make_node(
            "Sub",
            ["gqa_past_plus_sq", one_name],
            ["gqa_seqlens_k"],
            name="gqa_seqlens_k_sub",
        ),
    ]
    insert_at = min(
        (i for i, n in enumerate(graph.node) if n.op_type == "GroupQueryAttention")
    )
    for offset, node in enumerate(new_nodes):
        graph.node.insert(insert_at + offset, node)
    for node in gqa_nodes:
        node.input[5] = "gqa_seqlens_k"
    return len(gqa_nodes)


def _external_data_location_q(init: onnx.TensorProto) -> str | None:
    if init.data_location != TensorProto.EXTERNAL:
        return None
    for kv in init.external_data:
        if kv.key == "location":
            return kv.value
    return None


def save_model_with_external_data_q(
    model: onnx.ModelProto,
    dst: Path,
    *,
    rewrite_head_data: bool,
    head_data_dir: Path | None = None,
) -> None:
    """Save emb/lm_head; rewrite external head blobs when present."""
    from onnx.external_data_helper import set_external_data

    if not model.graph.name:
        model.graph.name = f"{dst.stem}_graph"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    data_dir = head_data_dir or dst.parent
    touched_locations: set[str] = set()
    for init in model.graph.initializer:
        location = _external_data_location_q(init)
        if location is None:
            continue
        arr = numpy_helper.to_array(init)
        out_path = data_dir / location
        if rewrite_head_data or location not in touched_locations:
            if rewrite_head_data and out_path.exists():
                out_path.unlink()
            if rewrite_head_data or not out_path.exists():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(arr.tobytes())
            touched_locations.add(location)
        fresh = numpy_helper.from_array(arr, init.name)
        set_external_data(fresh, location=location, offset=0, length=arr.nbytes)
        init.CopyFrom(fresh)
    onnx.save(model, str(dst))


def patch_emb_model_q(
    src: Path,
    dst: Path,
    *,
    shape_fix: ShapeFixConfig_q,
    rewrite_head_data: bool,
    head_data_dir: Path | None = None,
    profile: ConvertProfile = CONVERT_PROFILE_FULL,
) -> None:
    model = onnx.load(str(src), load_external_data=True)
    if profile.emb_bits4:
        for node in model.graph.node:
            if node.op_type != "GatherBlockQuantized":
                continue
            if not any((a.name == "bits" for a in node.attribute)):
                node.attribute.extend([helper.make_attribute("bits", 4)])
    if profile.fp16 or profile.fp32_activations:
        promote_model_to_fp16_q(model)
    if profile.lpnorm_fp32:
        stabilize_lpnorm_fp16_q(model)
    maybe_fix_static_shapes_q(model, shape_fix)
    save_model_with_external_data_q(
        model, dst, rewrite_head_data=rewrite_head_data, head_data_dir=head_data_dir
    )


def convert_lm_head_model_q(
    src: Path,
    dst: Path,
    *,
    shape_fix: ShapeFixConfig_q,
    rewrite_head_data: bool,
    head_data_dir: Path | None = None,
    gather_unsqueeze: bool = False,
    profile: ConvertProfile = CONVERT_PROFILE_FULL,
) -> None:
    model = onnx.load(str(src), load_external_data=True)
    if profile.fp16 or profile.fp32_activations:
        promote_model_to_fp16_q(model)
    if profile.matmul_nbits_accuracy:
        set_matmul_nbits_accuracy_level_q(model)
    if profile.lpnorm_fp32:
        stabilize_lpnorm_fp16_q(model)
    maybe_fix_static_shapes_q(model, shape_fix)
    if gather_unsqueeze and profile.lm_head_gather:
        model = rewrite_lm_head_gather_unsqueeze(model)
    save_model_with_external_data_q(
        model, dst, rewrite_head_data=rewrite_head_data, head_data_dir=head_data_dir
    )


def assert_pure_fp16_activations(model: onnx.ModelProto, *, context: str = "") -> None:
    """Fail if any Cast->fp32 nodes remain (hard fp16 activation requirement)."""
    fp32_casts = sum(
        (
            1
            for n in model.graph.node
            if n.op_type == "Cast"
            and any((a.name == "to" and a.i == FP32_q for a in n.attribute))
        )
    )
    if fp32_casts:
        prefix = f"{context}: " if context else ""
        raise RuntimeError(
            f"{prefix}graph has {fp32_casts} Cast->fp32 node(s); pure fp16 activations required"
        )


def assert_qdq_removed(graph: onnx.GraphProto, *, context: str = "") -> None:
    """Fail fast if any QuantizeLinear / DequantizeLinear nodes remain."""
    q = sum((1 for n in graph.node if n.op_type == "QuantizeLinear"))
    dq = sum((1 for n in graph.node if n.op_type == "DequantizeLinear"))
    if q or dq:
        prefix = f"{context}: " if context else ""
        raise RuntimeError(
            f"{prefix}QDQ not fully removed (QuantizeLinear={q}, DequantizeLinear={dq})"
        )


def prepare_gqa_fp16_no_quant(model: onnx.ModelProto) -> int:
    """runtime GQA: fp16 KV cache, no quantization, no scale inputs.

    - ``past_*`` / ``present_*`` KV graph I/O: int8 -> float16
    - ``k_quant_type`` / ``v_quant_type``: ``NONE``
    - ``kv_cache_bit_width``: keep 8 (``hip.gqa`` requires 4 or 8; do not set 0)
    - Drop optional ``k_scale`` / ``v_scale`` inputs (truncate to 12) and remove
      scale initializers (runtime errors on any non-null scale pointer).
    """
    graph = model.graph
    remove_scales: set[str] = set()
    changed = 0
    for node in graph.node:
        if node.op_type != "GroupQueryAttention":
            continue
        if len(node.input) > 12:
            for idx in range(12, len(node.input)):
                if node.input[idx]:
                    remove_scales.add(node.input[idx])
            del node.input[12:]
        has_k_quant = has_v_quant = has_bit_width = False
        bit_width = 8
        for attr in node.attribute:
            if attr.name == "kv_cache_bit_width" and attr.i in (4, 8):
                bit_width = attr.i
            elif attr.name == "k_quant_type":
                attr.s = b"NONE"
                has_k_quant = True
            elif attr.name == "v_quant_type":
                attr.s = b"NONE"
                has_v_quant = True
            elif attr.name == "kv_cache_bit_width":
                has_bit_width = True
        if not has_k_quant:
            node.attribute.extend([helper.make_attribute("k_quant_type", "NONE")])
        if not has_v_quant:
            node.attribute.extend([helper.make_attribute("v_quant_type", "NONE")])
        for attr in node.attribute:
            if attr.name == "kv_cache_bit_width":
                attr.i = bit_width
                has_bit_width = True
                break
        if not has_bit_width:
            node.attribute.extend(
                [helper.make_attribute("kv_cache_bit_width", bit_width)]
            )
        changed += 1
    kv_names = ("past_keys_", "past_values_", "present_keys_", "present_values_")
    for vi in list(graph.input) + list(graph.output):
        if not any((part in vi.name for part in kv_names)):
            continue
        if vi.type.tensor_type.elem_type == TensorProto.INT8:
            vi.type.tensor_type.elem_type = FP16_q
    if remove_scales:
        kept = [i for i in graph.initializer if i.name not in remove_scales]
        del graph.initializer[:]
        graph.initializer.extend(kept)
    return changed


def _inline_initializer_names(model: onnx.ModelProto) -> set[str]:
    return {
        i.name
        for i in model.graph.initializer
        if i.data_location != TensorProto.EXTERNAL
    }


def _reembed_initializers(graph: onnx.GraphProto, names: set[str]) -> None:
    """Force listed initializers back into the ONNX protobuf (not external data)."""
    for init in graph.initializer:
        if init.name not in names:
            continue
        arr = numpy_helper.to_array(init)
        init.CopyFrom(numpy_helper.from_array(arr, name=init.name))


def save_decoder_model_q(
    model: onnx.ModelProto,
    dst: Path,
    *,
    external_data_name: str = DECODER_EXT_DATA,
    keep_inline: set[str] | None = None,
    reuse_external_data: bool = False,
    external_data_ref: Path | None = None,
) -> None:
    from .step1_qdq_fp16 import _save_decoder_graph_reusing_external_data

    if not model.graph.name:
        model.graph.name = f"{dst.stem}_graph"
    if keep_inline:
        _reembed_initializers(model.graph, keep_inline)
    dst.parent.mkdir(parents=True, exist_ok=True)
    data_path = dst.parent / external_data_name
    if dst.exists():
        dst.unlink()
    if reuse_external_data:
        if external_data_ref is None:
            raise ValueError(
                "external_data_ref is required when reuse_external_data=True"
            )
        if not data_path.is_file():
            raise RuntimeError(
                f"Expected shared external weights at {data_path} before reusing them"
            )
        _save_decoder_graph_reusing_external_data(
            model,
            dst,
            external_data_ref=external_data_ref,
            external_data_name=external_data_name,
        )
        return
    if data_path.exists():
        data_path.unlink()
    onnx.save_model(
        model,
        str(dst),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_name,
        size_threshold=65536,
    )


def infer_decoder_external_data_name_q(src: Path, bundle_root: Path) -> str:
    model = onnx.load(str(src), load_external_data=False)
    locations = {
        loc
        for init in model.graph.initializer
        if (loc := _external_data_location_q(init)) is not None
    }
    if len(locations) == 1:
        return locations.pop()
    data_files = sorted(bundle_root.glob("*.data"))
    if data_files:
        return data_files[0].name
    return DECODER_EXT_DATA


DEFAULT_EMB_SEQ_LENS = (1, 512)
DEFAULT_LM_HEAD_SEQ_LENS = (1, 512)
DECODER_PREFILL_NAME = "SPLIT_0.onnx"
DECODER_DECODE_NAME = "SPLIT_1.onnx"
EMB_SRC_NAME = "embeddings_quant.quant.onnx"
LM_HEAD_SRC_NAME = "lm_head_quant_w4a32.quant.onnx"
INT4_q = TensorProto.INT4
DECODER_EXTERNAL_DATA_q = DECODER_EXT_DATA
DEFAULT_RUNTIME_EP = True
MERGED_PIPELINE_DECODER_STEM = "decoder_lowbit"
MERGED_PIPELINE_STEM = "decoder_lowbit_merged"
MERGED_PIPELINE_OUTPUT_NAME = f"{MERGED_PIPELINE_STEM}.onnx"
MERGED_PIPELINE_EXTERNAL_DATA = f"{MERGED_PIPELINE_STEM}.data"
_PIPELINE_PREFIXES_q = ("emb_", "dec_", "head_")


def _conv_kernel_shape_q(conv: onnx.NodeProto) -> list[int] | None:
    for attr in conv.attribute:
        if attr.name == "kernel_shape":
            return list(attr.ints)
    return None


def _find_transpose_by_output_q(
    graph: onnx.GraphProto, tensor: str
) -> onnx.NodeProto | None:
    return next(
        (n for n in graph.node if n.op_type == "Transpose" and n.output[0] == tensor),
        None,
    )


def _find_transpose_by_input_q(
    graph: onnx.GraphProto, tensor: str
) -> onnx.NodeProto | None:
    return next(
        (n for n in graph.node if n.op_type == "Transpose" and n.input[0] == tensor),
        None,
    )


def fuse_conv_transpose_to_matmul_nbits_q(
    graph: onnx.GraphProto,
    inits: dict[str, onnx.TensorProto],
    new_inits: dict[str, onnx.TensorProto],
    remove_inits: set[str],
) -> int:
    """Replace Transpose -> Conv(int4/int8 DQ weight) -> Transpose with MatMulNBits."""
    dq_by_out = {d.output[0]: d for d in graph.node if d.op_type == "DequantizeLinear"}
    replacements: dict[str, list[onnx.NodeProto]] = {}
    remove_nodes: set[str] = set()
    converted = 0
    for conv in graph.node:
        if conv.op_type != "Conv" or len(conv.input) < 2:
            continue
        if _conv_kernel_shape_q(conv) != [1, 1]:
            continue
        w_dq = dq_by_out.get(conv.input[1])
        if w_dq is None or w_dq.input[0] not in inits:
            continue
        w_init = inits[w_dq.input[0]]
        if w_init.data_type == INT4_q:
            weight_bits = 4
        elif w_init.data_type == TensorProto.INT8:
            weight_bits = 8
        else:
            continue
        t0 = _find_transpose_by_output_q(graph, conv.input[0])
        t1 = _find_transpose_by_input_q(graph, conv.output[0])
        if t0 is None or t1 is None:
            continue
        block_input = t0.input[0]
        block_output = t1.output[0]
        w = numpy_helper.to_array(w_init)
        if w.ndim != 4 or w.shape[2:] != (1, 1):
            raise ValueError(
                f"Expected 1x1 Conv weight [N,K,1,1], got {w.shape} for {w_init.name}"
            )
        n_out, k_in = (int(w.shape[0]), int(w.shape[1]))
        w_kn = w[:, :, 0, 0].T
        scale, zp = _expand_scale_zp_q(
            numpy_helper.to_array(inits[w_dq.input[1]]),
            numpy_helper.to_array(inits[w_dq.input[2]]),
            n_out,
        )
        block_size = choose_block_size_q(k_in)
        if weight_bits == 4:
            packed, k, n, n_blocks = pack_matmul_nbits_weight_q(w_kn, block_size)
        else:
            packed, k, n, n_blocks = pack_matmul_nbits_weight_int8(w_kn, block_size)
        base = w_init.name.removesuffix("_quantized")
        q_name = f"{base}_mnbits_q{weight_bits}"
        s_name = f"{base}_mnbits_scales"
        z_name = f"{base}_mnbits_zp"
        pre_shape_name = f"{base}_mnbits_pre_shape"
        post_shape_name = f"{base}_mnbits_post_shape"
        new_inits[q_name] = numpy_helper.from_array(packed, name=q_name)
        new_inits[s_name] = numpy_helper.from_array(
            np.repeat(scale.astype(np.float16).reshape(n, 1), n_blocks, axis=1),
            name=s_name,
        )
        if weight_bits == 4:
            new_inits[z_name] = numpy_helper.from_array(
                pack_zero_points_q(zp, n, n_blocks), name=z_name
            )
        else:
            new_inits[z_name] = numpy_helper.from_array(
                _int8_mnbits_zero_points(n, n_blocks), name=z_name
            )
        new_inits[pre_shape_name] = numpy_helper.from_array(
            np.array([-1, k], dtype=np.int64), name=pre_shape_name
        )
        new_inits[post_shape_name] = numpy_helper.from_array(
            np.array([1, 1, -1, n], dtype=np.int64), name=post_shape_name
        )
        mm_in = f"{conv.name}_mnbits_in"
        mm_out = f"{conv.name}_mnbits_out"
        replacements[conv.name] = [
            helper.make_node(
                "Reshape",
                [block_input, pre_shape_name],
                [mm_in],
                name=f"{conv.name}_mnbits_reshape_in",
            ),
            helper.make_node(
                "MatMulNBits",
                [mm_in, q_name, s_name, z_name],
                [mm_out],
                name=f"{conv.name}_mnbits",
                domain="com.microsoft",
                K=k,
                N=n,
                bits=weight_bits,
                block_size=block_size,
                accuracy_level=MATMUL_NBITS_ACCURACY_LEVEL_q,
            ),
            helper.make_node(
                "Reshape",
                [mm_out, post_shape_name],
                [block_output],
                name=f"{conv.name}_mnbits_reshape_out",
            ),
        ]
        remove_nodes.update({t0.name, conv.name, t1.name, w_dq.name})
        remove_inits.update({w_init.name, w_dq.input[1], w_dq.input[2]})
        converted += 1
    if converted:
        kept: list[onnx.NodeProto] = []
        for node in graph.node:
            if node.name in replacements:
                kept.extend(replacements[node.name])
            elif node.name not in remove_nodes:
                kept.append(node)
        del graph.node[:]
        graph.node.extend(kept)
    return converted


def convert_decoder_model_q(
    src: Path,
    dst: Path,
    *,
    shape_fix: ShapeFixConfig_q,
    bundle_root: Path,
    external_data_name: str | None = None,
    reuse_external_data: bool = False,
    external_data_ref: Path | None = None,
    runtime_ep: bool = DEFAULT_RUNTIME_EP,
    profile: ConvertProfile = CONVERT_PROFILE_FULL,
) -> dict[str, int]:
    model = onnx.load(str(src), load_external_data=True)
    inline_at_load = _inline_initializer_names(model)
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    new_inits: dict[str, onnx.TensorProto] = {}
    remove_inits: set[str] = set()
    stats: dict[str, int] = {}
    stats["unwrap_castfp32"] = unwrap_castfp32_before_quantize(graph)
    stats["strip_qdq"] = (
        strip_activation_qdq_q(
            graph,
            inits,
            new_inits,
            pure_fp16=profile.fp16 and (not profile.fp32_activations),
            mode=profile.activation_qdq_mode,
        )
        if profile.strip_qdq
        else 0
    )
    stats["conv_mnbits"] = (
        fuse_conv_transpose_to_matmul_nbits_q(graph, inits, new_inits, remove_inits)
        if profile.conv_mnbits
        else 0
    )
    stats["weight_dq_fold"] = (
        fold_weight_dq_q(
            graph,
            inits,
            new_inits,
            remove_inits,
            weight_dtype=np.float32 if profile.fp32_activations else np.float16,
        )
        if profile.weight_dq_fold
        else 0
    )
    if new_inits or remove_inits:
        apply_new_initializers_q(graph, new_inits)
        prune_initializers_q(graph, remove_inits)
    if profile.fp32_activations:
        io_stats = wrap_fp16_io_fp32_activations(model)
        stats["fp32_io_input_cast"] = io_stats["input_cast"]
        stats["fp32_io_output_cast"] = io_stats["output_cast"]
        stats["gqa_pre_cast_fp32_removed"] = 0
        stats["mnbits_fp16_casts_folded"] = 0
        stats["fp32_casts_remaining"] = sum(
            (
                1
                for n in model.graph.node
                if n.op_type == "Cast"
                and any((a.name == "to" and a.i == FP32_q for a in n.attribute))
            )
        )
    elif profile.fp16:
        promote_model_to_fp16_q(model)
        fp16_stats = optimize_fp16_activations(
            model,
            remove_gqa_fp32_casts=profile.gqa_pre_cast_fp32_remove,
            rewrite_gqa_cast_fp32_to_fp16=not profile.gqa_pre_cast_fp32_remove,
        )
        stats["gqa_pre_cast_fp32_removed"] = fp16_stats["gqa_getitem_fp32_removed"]
        stats["gqa_cast_fp32_to_fp16"] = fp16_stats["gqa_cast_fp32_to_fp16"]
        stats["mnbits_fp16_casts_folded"] = fp16_stats["mnbits_fp16_casts_folded"]
        stats["fp32_casts_remaining"] = fp16_stats["fp32_casts_remaining"]
        if not profile.fp32_activations:
            assert_pure_fp16_activations(model, context=dst.name)
    if profile.matmul_nbits_accuracy:
        set_matmul_nbits_accuracy_level_q(model)
    if profile.gqa_fp16_no_quant and runtime_ep:
        stats["lpnorm_fp32"] = 0
        stats["gqa_fp16_no_quant"] = prepare_gqa_fp16_no_quant(model)
    elif profile.lpnorm_fp32 and (not runtime_ep):
        stats["lpnorm_fp32"] = stabilize_lpnorm_fp16_q(model)
        stats["gqa_fp16_no_quant"] = 0
    else:
        stats["lpnorm_fp32"] = 0
        stats["gqa_fp16_no_quant"] = 0
    maybe_fix_static_shapes_q(model, shape_fix)
    stats["gqa_seqlens_rewrite"] = (
        rewrite_gqa_past_seq_len_to_seqlens_k_q(model)
        if profile.gqa_seqlens_rewrite
        else 0
    )
    if profile.strip_qdq:
        assert_qdq_removed(graph, context=dst.name)
    from .step1_qdq_fp16 import DECODER_WORK_EXTERNAL_DATA

    ext_name = external_data_name or DECODER_WORK_EXTERNAL_DATA
    save_decoder_model_q(
        model,
        dst,
        external_data_name=ext_name,
        keep_inline=inline_at_load,
        reuse_external_data=reuse_external_data,
        external_data_ref=external_data_ref,
    )
    return stats


def _emb_shape_fix_q(shape_fix: ShapeFixConfig_q, seq_len: int) -> ShapeFixConfig_q:
    if not shape_fix.enabled:
        return shape_fix
    return ShapeFixConfig_q(
        enabled=True,
        batch=shape_fix.batch,
        seq_len=seq_len,
        max_seq_len=shape_fix.max_seq_len,
    )


def process_one_onnx_q(
    src: Path,
    dst: Path,
    *,
    kind: str,
    shape_fix: ShapeFixConfig_q,
    bundle_root: Path,
    head_data_dir: Path,
    rewrite_head_data: bool,
    external_data_name: str | None = None,
    reuse_external_data: bool = False,
    external_data_ref: Path | None = None,
    runtime_ep: bool = DEFAULT_RUNTIME_EP,
    gather_unsqueeze: bool = False,
    profile: ConvertProfile = CONVERT_PROFILE_FULL,
) -> dict[str, int] | None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if kind == "emb":
        patch_emb_model_q(
            src,
            dst,
            shape_fix=shape_fix,
            rewrite_head_data=rewrite_head_data,
            head_data_dir=head_data_dir,
            profile=profile,
        )
        return None
    if kind == "lm_head":
        convert_lm_head_model_q(
            src,
            dst,
            shape_fix=shape_fix,
            rewrite_head_data=rewrite_head_data,
            head_data_dir=head_data_dir,
            gather_unsqueeze=gather_unsqueeze,
            profile=profile,
        )
        return None
    return convert_decoder_model_q(
        src,
        dst,
        shape_fix=shape_fix,
        bundle_root=bundle_root,
        external_data_name=external_data_name,
        reuse_external_data=reuse_external_data,
        external_data_ref=external_data_ref,
        runtime_ep=runtime_ep,
        profile=profile,
    )


def _align_model_ir_version_q(model: onnx.ModelProto, ir_version: int) -> None:
    model.ir_version = ir_version


def _topological_sort_graph_q(graph: onnx.GraphProto) -> None:
    """Reorder ``graph.node`` so producers appear before consumers."""
    nodes = list(graph.node)
    if not nodes:
        return
    node_by_name = {n.name: n for n in nodes}
    producers: dict[str, str] = {}
    for node in nodes:
        for out in node.output:
            if out:
                producers[out] = node.name
    init_names = {init.name for init in graph.initializer}
    input_names = {inp.name for inp in graph.input}
    deps: dict[str, set[str]] = {}
    for node in nodes:
        dep: set[str] = set()
        for inp in node.input:
            if not inp or inp in input_names or inp in init_names:
                continue
            prod = producers.get(inp)
            if prod and prod != node.name:
                dep.add(prod)
        deps[node.name] = dep
    in_degree = {node.name: len(deps[node.name]) for node in nodes}
    children: dict[str, set[str]] = {node.name: set() for node in nodes}
    for node in nodes:
        for dep in deps[node.name]:
            children[dep].add(node.name)
    queue = deque((node for node in nodes if in_degree[node.name] == 0))
    sorted_nodes: list[onnx.NodeProto] = []
    while queue:
        node = queue.popleft()
        sorted_nodes.append(node)
        for child_name in children[node.name]:
            in_degree[child_name] -= 1
            if in_degree[child_name] == 0:
                sorted_nodes.append(node_by_name[child_name])
    if len(sorted_nodes) != len(nodes):
        seen = {node.name for node in sorted_nodes}
        sorted_nodes.extend((node for node in nodes if node.name not in seen))
    del graph.node[:]
    graph.node.extend(sorted_nodes)


def _rename_tensors_in_graph_q(graph: onnx.GraphProto, mapping: dict[str, str]) -> None:
    if not mapping:
        return
    for node in graph.node:
        for idx, name in enumerate(node.input):
            if name in mapping:
                node.input[idx] = mapping[name]
        for idx, name in enumerate(node.output):
            if name in mapping:
                node.output[idx] = mapping[name]
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        if vi.name in mapping:
            vi.name = mapping[vi.name]
    for init in graph.initializer:
        if init.name in mapping:
            init.name = mapping[init.name]


def _pipeline_boundary_renames_q(graph: onnx.GraphProto) -> dict[str, str]:
    """Map prefixed merge tensors back to the split-pipeline I/O names."""
    mapping: dict[str, str] = {
        "emb_input_ids": "input_ids",
        "dec_past_seq_len": "past_seq_len",
        "dec_total_seq_len": "total_seq_len",
        "head_logits": "logits",
    }
    for vi in graph.input:
        if vi.name.startswith("dec_past_keys_"):
            mapping[vi.name] = vi.name.removeprefix("dec_")
        elif vi.name.startswith("dec_past_values_"):
            mapping[vi.name] = vi.name.removeprefix("dec_")
    for vi in graph.output:
        if vi.name.startswith("dec_present_keys_"):
            mapping[vi.name] = vi.name.removeprefix("dec_")
        elif vi.name.startswith("dec_present_values_"):
            mapping[vi.name] = vi.name.removeprefix("dec_")
    return mapping


def merge_quantized_pipeline_models(
    emb_path: Path,
    decoder_path: Path,
    head_path: Path,
    dst_path: Path,
    *,
    external_data_name: str = MERGED_PIPELINE_EXTERNAL_DATA,
    gather_unsqueeze: bool = True,
) -> Path:
    """Fuse emb -> decoder -> lm_head into one ONNX (dynamic ``seq_len``)."""
    emb = onnx.load(str(emb_path), load_external_data=True)
    decoder = onnx.load(str(decoder_path), load_external_data=True)
    head = onnx.load(str(head_path), load_external_data=True)
    if gather_unsqueeze:
        head = rewrite_lm_head_gather_unsqueeze(head)
    target_ir = max(emb.ir_version, decoder.ir_version, head.ir_version)
    _align_model_ir_version_q(emb, target_ir)
    _align_model_ir_version_q(head, target_ir)
    emb_p = add_prefix(emb, prefix=_PIPELINE_PREFIXES_q[0])
    dec_p = add_prefix(decoder, prefix=_PIPELINE_PREFIXES_q[1])
    head_p = add_prefix(head, prefix=_PIPELINE_PREFIXES_q[2])
    graph = merge_graphs(
        emb_p.graph,
        dec_p.graph,
        [("emb_input_hidden_states", "dec_input_hidden_states")],
    )
    kv_outputs = [out.name for out in graph.output if "present_" in out.name]
    graph = merge_graphs(
        graph,
        head_p.graph,
        [("dec_output_hidden_states", "head_output_hidden_states")],
        outputs=["head_logits", *kv_outputs],
    )
    _topological_sort_graph_q(graph)
    _rename_tensors_in_graph_q(graph, _pipeline_boundary_renames_q(graph))
    logits_out = next((vi for vi in graph.output if vi.name == "logits"), None)
    if logits_out is not None:
        other_outs = [vi for vi in graph.output if vi.name != "logits"]
        del graph.output[:]
        graph.output.extend([logits_out, *other_outs])
    model = helper.make_model(graph, opset_imports=decoder.opset_import)
    model.ir_version = target_ir
    model.graph.name = f"{dst_path.stem}_graph"
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    data_path = dst_path.parent / external_data_name
    if dst_path.exists():
        dst_path.unlink()
    if data_path.exists():
        data_path.unlink()
    onnx.save_model(
        model,
        str(dst_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_name,
        size_threshold=65536,
    )
    return dst_path
