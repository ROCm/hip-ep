#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .step2_fp16_cleanup import optimize_fp16_activations
from .step3_lm_head import rewrite_lm_head_gather_unsqueeze

FP32__dup4 = TensorProto.FLOAT
FP64 = TensorProto.DOUBLE
FP16__dup4 = TensorProto.FLOAT16
INT4 = TensorProto.INT4
DECODER_EXTERNAL_DATA = "BUNDLE.data"
HEAD_QWEIGHT_DATA = "BUNDLE_head_qweight.data"
HEAD_SCALE_DATA = "BUNDLE_head_scale.data"
DEFAULT_BATCH = 1
DEFAULT_SEQ_LEN = 128
DEFAULT_MAX_SEQ_LEN = 16384


def _dim_param_bindings(batch: int, seq_len: int, max_seq_len: int) -> dict[str, int]:
    return {"batch": batch, "seq_len": seq_len, "max_seq_len": max_seq_len}


def _fix_value_info_shapes(vi: onnx.ValueInfoProto, bindings: dict[str, int]) -> None:
    for dim in vi.type.tensor_type.shape.dim:
        if dim.dim_param and dim.dim_param in bindings:
            dim.dim_value = bindings[dim.dim_param]
            dim.ClearField("dim_param")


@dataclass(frozen=True)
class ShapeFixConfig:
    """Optional static-shape binding for symbolic ONNX dimensions."""

    enabled: bool = False
    batch: int = DEFAULT_BATCH
    seq_len: int = DEFAULT_SEQ_LEN
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN


def maybe_fix_static_shapes(model: onnx.ModelProto, shape_fix: ShapeFixConfig) -> None:
    """Bind symbolic dims when ``shape_fix.enabled``."""
    if not shape_fix.enabled:
        return
    bindings = _dim_param_bindings(
        shape_fix.batch, shape_fix.seq_len, shape_fix.max_seq_len
    )
    graph = model.graph
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        _fix_value_info_shapes(vi, bindings)


def fix_static_shapes(
    model: onnx.ModelProto,
    *,
    batch: int = DEFAULT_BATCH,
    seq_len: int = DEFAULT_SEQ_LEN,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
) -> None:
    maybe_fix_static_shapes(
        model,
        ShapeFixConfig(
            enabled=True, batch=batch, seq_len=seq_len, max_seq_len=max_seq_len
        ),
    )


_INT4_UNSIGNED_OFFSET = 8
DEFAULT_ACCURACY_LEVEL = 1
MATMUL_NBITS_ACCURACY_LEVEL = DEFAULT_ACCURACY_LEVEL


def _nibble_pack_int4(values: np.ndarray) -> np.ndarray:
    nibbles = values.astype(np.int16) + _INT4_UNSIGNED_OFFSET & 15
    return nibbles[0::2].astype(np.uint8) | nibbles[1::2].astype(np.uint8) << 4


def choose_block_size(k: int) -> int:
    """Pick a MatMulNBits-legal block_size (power of 2, >= 16) for a K dim.

    Prefer the largest such block_size that divides K (no padding needed);
    fall back to 32 with zero-padded trailing block when none divides.
    """
    for bs in (256, 128, 64, 32, 16):
        if k % bs == 0:
            return bs
    return 32


def pack_matmul_nbits_weight(
    w_kn: np.ndarray, block_size: int
) -> tuple[np.ndarray, int, int, int]:
    k, n = w_kn.shape
    if k % 2 != 0:
        raise ValueError(f"K must be even for 4-bit packing, got K={k}")
    n_blocks = (k + block_size - 1) // block_size
    blob_size = block_size // 2
    packed = np.zeros((n, n_blocks, blob_size), dtype=np.uint8)
    for col in range(n):
        col_bytes = _nibble_pack_int4(w_kn[:, col])
        flat = np.zeros(n_blocks * blob_size, dtype=np.uint8)
        flat[: col_bytes.size] = col_bytes
        packed[col] = flat.reshape(n_blocks, blob_size)
    return (packed, k, n, n_blocks)


def pack_zero_points(z_n: np.ndarray, n: int, n_blocks: int) -> np.ndarray:
    """Pack one (per-channel) 4-bit zero point, replicated across all blocks.

    Layout matches MatMulNBits: per column, n_blocks nibbles packed two-per-byte
    (even block -> low nibble, odd block -> high nibble).
    """
    zp_bytes = (n_blocks + 1) // 2
    out = np.zeros((n, zp_bytes), dtype=np.uint8)
    for col in range(n):
        v = int(z_n[col]) + _INT4_UNSIGNED_OFFSET & 15
        for b in range(n_blocks):
            if b % 2 == 0:
                out[col, b // 2] |= v
            else:
                out[col, b // 2] |= v << 4
    return out


def dequantize_initializer(
    w: np.ndarray, scale: np.ndarray, zp: np.ndarray, *, out_dtype: type = np.float16
) -> np.ndarray:
    w_f = w.astype(np.float64)
    s_f = np.asarray(scale, dtype=np.float64)
    z_f = np.asarray(zp, dtype=np.float64)
    return ((w_f - z_f) * s_f).astype(out_dtype)


def dequantize_weight_array(
    w: np.ndarray, scale: np.ndarray, zp: np.ndarray, *, out_dtype: type = np.float16
) -> np.ndarray:
    """Dequantize int4/int8 (or float) weights for offline Gemm weight folding."""
    w_f = w.astype(np.float64)
    s_f = np.asarray(scale, dtype=np.float64)
    z_f = np.asarray(zp, dtype=np.float64)
    if w.dtype.name == "int4" and s_f.ndim == 1 and (s_f.shape[0] == w.shape[0]):
        return ((w_f - z_f.reshape(-1, 1)) * s_f.reshape(-1, 1)).astype(out_dtype)
    if s_f.ndim == 1 and s_f.size == w.shape[0] and (w.ndim >= 2):
        z_col = z_f.reshape(-1, 1) if z_f.ndim else z_f
        s_col = s_f.reshape(-1, 1)
        return ((w_f - z_col) * s_col).astype(out_dtype)
    return dequantize_initializer(w, scale, zp, out_dtype=out_dtype)


def _replace_tensor_name(
    users: dict[str, list[tuple[onnx.NodeProto, int]]], old: str, new: str
) -> None:
    for node, idx in users.get(old, []):
        if node.input[idx] == old:
            node.input[idx] = new


def _build_tensor_users(
    graph: onnx.GraphProto,
) -> dict[str, list[tuple[onnx.NodeProto, int]]]:
    users: dict[str, list[tuple[onnx.NodeProto, int]]] = {}
    for node in graph.node:
        for idx, name in enumerate(node.input):
            if name:
                users.setdefault(name, []).append((node, idx))
    return users


def _referenced_initializer_names(graph: onnx.GraphProto) -> set[str]:
    used: set[str] = set()
    for node in graph.node:
        for name in node.input:
            if name:
                used.add(name)
    return used


def _rename_value_tensor(
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
    _replace_tensor_name(users, old, new)
    for vi in graph.value_info:
        if vi.name == old:
            vi.name = new


def strip_activation_qdq(graph: onnx.GraphProto) -> int:
    users = _build_tensor_users(graph)
    dq_by_q_out = {
        d.input[0]: d for d in graph.node if d.op_type == "DequantizeLinear" and d.input
    }
    remove_nodes: set[str] = set()
    dq_to_src: dict[str, str] = {}
    removed = 0
    for q in graph.node:
        if q.op_type != "QuantizeLinear":
            continue
        dq = dq_by_q_out.get(q.output[0])
        if dq is None:
            continue
        src, dst = (q.input[0], dq.output[0])
        dq_to_src[dst] = src
        _replace_tensor_name(users, dst, src)
        remove_nodes.add(q.name)
        remove_nodes.add(dq.name)
        removed += 1
    graph_outputs = {out.name for out in graph.output}
    for dst, src in dq_to_src.items():
        if dst in graph_outputs:
            _rename_value_tensor(graph, users, src, dst)
    if remove_nodes:
        kept = [n for n in graph.node if n.name not in remove_nodes]
        del graph.node[:]
        graph.node.extend(kept)
    return removed


def _expand_scale_zp(
    scale: np.ndarray, zp: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    if scale.ndim == 0:
        scale = np.full(n, float(scale), dtype=np.float32)
    if zp.ndim == 0:
        zp = np.full(n, int(zp), dtype=np.int8)
    return (scale, zp)


def convert_matmul_weight_dq(
    graph: onnx.GraphProto,
    inits: dict[str, onnx.TensorProto],
    new_inits: dict[str, onnx.TensorProto],
    remove_inits: set[str],
) -> int:
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
        w_name = dq.input[0]
        w = numpy_helper.to_array(inits[w_name])
        if w.dtype.name != "int4":
            continue
        scale, zp = _expand_scale_zp(
            numpy_helper.to_array(inits[dq.input[1]]),
            numpy_helper.to_array(inits[dq.input[2]]),
            w.shape[1],
        )
        block_size = choose_block_size(w.shape[0])
        packed, k, n, n_blocks = pack_matmul_nbits_weight(w, block_size)
        base = w_name.removesuffix("_quantized")
        q_name = f"{base}_mnbits_q4"
        s_name = f"{base}_mnbits_scales"
        z_name = f"{base}_mnbits_zp"
        new_inits[q_name] = numpy_helper.from_array(packed, name=q_name)
        new_inits[s_name] = numpy_helper.from_array(
            np.repeat(scale.astype(np.float16).reshape(n, 1), n_blocks, axis=1),
            name=s_name,
        )
        new_inits[z_name] = numpy_helper.from_array(
            pack_zero_points(zp, n, n_blocks), name=z_name
        )
        replacements[mm.name] = helper.make_node(
            "MatMulNBits",
            [mm.input[0], q_name, s_name, z_name],
            list(mm.output),
            name=f"{mm.name}_mnbits",
            domain="com.microsoft",
            K=k,
            N=n,
            bits=4,
            block_size=block_size,
            accuracy_level=MATMUL_NBITS_ACCURACY_LEVEL,
        )
        remove_nodes.update({mm.name, dq.name})
        remove_inits.update({w_name, dq.input[1], dq.input[2]})
        converted += 1
    if converted:
        kept = []
        for node in graph.node:
            if node.name in replacements:
                kept.append(replacements[node.name])
            elif node.name in remove_nodes:
                continue
            else:
                kept.append(node)
        del graph.node[:]
        graph.node.extend(kept)
    return converted


def _expand_scale_zp_blocks(
    scale: np.ndarray, zp: np.ndarray, n: int, n_blocks: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """Expand per-tensor (or per-channel) scale/zp to MatMulNBits block layout."""
    scale_f = np.asarray(scale, dtype=np.float32)
    if scale_f.ndim == 0:
        scale_blocks = np.full((n, n_blocks), float(scale_f), dtype=np.float16)
    elif scale_f.size == 1:
        scale_blocks = np.full(
            (n, n_blocks), float(scale_f.reshape(())), dtype=np.float16
        )
    elif scale_f.ndim == 1 and scale_f.shape[0] == n:
        scale_blocks = np.repeat(
            scale_f.astype(np.float16).reshape(n, 1), n_blocks, axis=1
        )
    else:
        raise ValueError(f"unsupported scale shape {scale_f.shape} for N={n}")
    zp_arr = np.asarray(zp)
    if zp_arr.size == 1 and float(zp_arr.reshape(())) == 0.0:
        return (scale_blocks, None)
    if zp_arr.ndim == 0:
        zp_blocks = np.full((n, n_blocks), int(zp_arr), dtype=np.uint8)
    elif zp_arr.size == 1:
        zp_blocks = np.full((n, n_blocks), int(zp_arr.reshape(())), dtype=np.uint8)
    elif zp_arr.ndim == 1 and zp_arr.shape[0] == n:
        zp_blocks = np.repeat(zp_arr.astype(np.uint8).reshape(n, 1), n_blocks, axis=1)
    else:
        raise ValueError(f"unsupported zero-point shape {zp_arr.shape} for N={n}")
    return (scale_blocks, zp_blocks)


def _matmul_nbits_block_size(k: int) -> int:
    """Pick block_size for 8-bit MatMulNBits (power-of-2, divides K)."""
    for bs in (256, 128, 64, 32, 16):
        if k % bs == 0:
            return bs
    raise ValueError(f"K={k} has no power-of-2 block_size divisor >= 16")


def has_graph_input_weight_quantized(graph: onnx.GraphProto) -> bool:
    return any(("weight_quantized" in vi.name for vi in graph.input))


def _expand_scale_zp_blocks_signed_int8(
    scale: np.ndarray, zp: np.ndarray, n: int, n_blocks: int
) -> tuple[np.ndarray, np.ndarray]:
    """Block scale layout for signed int8 weights stored as uint8 with offset 128."""
    scale_blocks, _ = _expand_scale_zp_blocks(scale, zp, n, n_blocks)
    zp_blocks = np.full((n, n_blocks), 128, dtype=np.uint8)
    return (scale_blocks, zp_blocks)


def convert_matmul_graph_input_int8_dq(
    graph: onnx.GraphProto,
    inits: dict[str, onnx.TensorProto],
    new_inits: dict[str, onnx.TensorProto],
    remove_inits: set[str],
) -> int:
    """Fuse graph-input int8 ``weight_quantized -> DQ -> MatMul`` into MatMulNBits.

    Keeps each ``weight_quantized`` port as a graph input (shape ``[K, N]``).
    Inserts ``Transpose + Reshape`` so MatMulNBits receives ``[N, K/block, block]``.
    """
    graph_inputs = {vi.name: vi for vi in graph.input}
    dq_by_out = {d.output[0]: d for d in graph.node if d.op_type == "DequantizeLinear"}
    scale_cache: dict[tuple[str, str, int, int], str] = {}
    insert_before: dict[str, list[onnx.NodeProto]] = {}
    remove_nodes: set[str] = set()
    converted = 0
    for mm in graph.node:
        if mm.op_type != "MatMul" or len(mm.input) < 2:
            continue
        dq = dq_by_out.get(mm.input[1])
        if dq is None or len(dq.input) < 3:
            continue
        w_name = dq.input[0]
        if w_name not in graph_inputs or w_name not in {vi.name for vi in graph.input}:
            continue
        if "weight_quantized" not in w_name:
            continue
        if dq.input[1] not in inits or dq.input[2] not in inits:
            continue
        vi = graph_inputs[w_name]
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        if len(dims) != 2 or not all(dims):
            continue
        k, n = dims
        block_size = _matmul_nbits_block_size(k)
        n_blocks = k // block_size
        packed_shape_name = f"{mm.name}_mnbits_pack_shape"
        new_inits[packed_shape_name] = numpy_helper.from_array(
            np.array([n, n_blocks, block_size], dtype=np.int64), name=packed_shape_name
        )
        offset_name = f"{mm.name}_mnbits_signed_offset"
        new_inits[offset_name] = numpy_helper.from_array(
            np.array(128, dtype=np.int16), name=offset_name
        )
        trans_out = f"{mm.name}_mnbits_w_t"
        trans_i16 = f"{mm.name}_mnbits_w_t_i16"
        shifted_i16 = f"{mm.name}_mnbits_w_shifted_i16"
        trans_u8 = f"{mm.name}_mnbits_w_t_u8"
        packed_u8 = f"{mm.name}_mnbits_w_uint8"
        pack_nodes = [
            helper.make_node(
                "Transpose",
                [w_name],
                [trans_out],
                name=f"{mm.name}_mnbits_transpose",
                perm=[1, 0],
            ),
            helper.make_node(
                "Cast",
                [trans_out],
                [trans_i16],
                name=f"{mm.name}_mnbits_cast_i16",
                to=TensorProto.INT16,
            ),
            helper.make_node(
                "Add",
                [trans_i16, offset_name],
                [shifted_i16],
                name=f"{mm.name}_mnbits_add_offset",
            ),
            helper.make_node(
                "Cast",
                [shifted_i16],
                [trans_u8],
                name=f"{mm.name}_mnbits_cast_u8",
                to=TensorProto.UINT8,
            ),
            helper.make_node(
                "Reshape",
                [trans_u8, packed_shape_name],
                [packed_u8],
                name=f"{mm.name}_mnbits_pack",
            ),
        ]
        scale_key = (dq.input[1], dq.input[2], n, n_blocks)
        if scale_key not in scale_cache:
            scale_blocks, zp_blocks = _expand_scale_zp_blocks_signed_int8(
                numpy_helper.to_array(inits[dq.input[1]]),
                numpy_helper.to_array(inits[dq.input[2]]),
                n,
                n_blocks,
            )
            s_name = f"{dq.input[1]}_blocks_{n}x{n_blocks}"
            z_name = f"{dq.input[2]}_blocks_{n}x{n_blocks}_u8zp128"
            new_inits[s_name] = numpy_helper.from_array(scale_blocks, name=s_name)
            new_inits[z_name] = numpy_helper.from_array(zp_blocks, name=z_name)
            scale_cache[scale_key] = s_name
            scale_cache[dq.input[1], dq.input[2], n, n_blocks, "zp"] = z_name
        s_name = scale_cache[scale_key]
        z_name = scale_cache[dq.input[1], dq.input[2], n, n_blocks, "zp"]
        mnbits_inputs = [mm.input[0], packed_u8, s_name, z_name]
        mnbits = helper.make_node(
            "MatMulNBits",
            mnbits_inputs,
            list(mm.output),
            name=f"{mm.name}_mnbits",
            domain="com.microsoft",
            K=k,
            N=n,
            bits=8,
            block_size=block_size,
            accuracy_level=MATMUL_NBITS_ACCURACY_LEVEL,
        )
        insert_before[mm.name] = [*pack_nodes, mnbits]
        remove_nodes.update({mm.name, dq.name})
        converted += 1
    if not converted:
        return 0
    kept: list[onnx.NodeProto] = []
    for node in graph.node:
        if node.name in insert_before:
            kept.extend(insert_before[node.name])
        elif node.name in remove_nodes:
            continue
        else:
            kept.append(node)
    del graph.node[:]
    graph.node.extend(kept)
    return converted


def _conv_kernel_shape(conv: onnx.NodeProto) -> list[int] | None:
    for attr in conv.attribute:
        if attr.name == "kernel_shape":
            return list(attr.ints)
    return None


def _find_transpose_by_output(
    graph: onnx.GraphProto, tensor: str
) -> onnx.NodeProto | None:
    return next(
        (n for n in graph.node if n.op_type == "Transpose" and n.output[0] == tensor),
        None,
    )


def _find_transpose_by_input(
    graph: onnx.GraphProto, tensor: str
) -> onnx.NodeProto | None:
    return next(
        (n for n in graph.node if n.op_type == "Transpose" and n.input[0] == tensor),
        None,
    )


def fuse_conv_transpose_to_matmul_nbits(
    graph: onnx.GraphProto,
    inits: dict[str, onnx.TensorProto],
    new_inits: dict[str, onnx.TensorProto],
    remove_inits: set[str],
) -> int:
    """Replace Transpose -> Conv(int4 DQ weight) -> Transpose with MatMulNBits."""
    dq_by_out = {d.output[0]: d for d in graph.node if d.op_type == "DequantizeLinear"}
    replacements: dict[str, list[onnx.NodeProto]] = {}
    remove_nodes: set[str] = set()
    converted = 0
    for conv in graph.node:
        if conv.op_type != "Conv" or len(conv.input) < 2:
            continue
        if _conv_kernel_shape(conv) != [1, 1]:
            continue
        w_dq = dq_by_out.get(conv.input[1])
        if w_dq is None or w_dq.input[0] not in inits:
            continue
        w_init = inits[w_dq.input[0]]
        if w_init.data_type != INT4:
            continue
        t0 = _find_transpose_by_output(graph, conv.input[0])
        t1 = _find_transpose_by_input(graph, conv.output[0])
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
        scale, zp = _expand_scale_zp(
            numpy_helper.to_array(inits[w_dq.input[1]]),
            numpy_helper.to_array(inits[w_dq.input[2]]),
            n_out,
        )
        block_size = choose_block_size(k_in)
        packed, k, n, n_blocks = pack_matmul_nbits_weight(w_kn, block_size)
        base = w_init.name.removesuffix("_quantized")
        q_name = f"{base}_mnbits_q4"
        s_name = f"{base}_mnbits_scales"
        z_name = f"{base}_mnbits_zp"
        pre_shape_name = f"{base}_mnbits_pre_shape"
        post_shape_name = f"{base}_mnbits_post_shape"
        new_inits[q_name] = numpy_helper.from_array(packed, name=q_name)
        new_inits[s_name] = numpy_helper.from_array(
            np.repeat(scale.astype(np.float16).reshape(n, 1), n_blocks, axis=1),
            name=s_name,
        )
        new_inits[z_name] = numpy_helper.from_array(
            pack_zero_points(zp, n, n_blocks), name=z_name
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
                bits=4,
                block_size=block_size,
                accuracy_level=MATMUL_NBITS_ACCURACY_LEVEL,
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


_WEIGHT_DQ_CONSUMER_INPUTS: dict[str, tuple[int, ...]] = {
    "Mul": (0, 1),
    "Conv": (1,),
    "Gemm": (1,),
    "Add": (0, 1),
}


def fold_weight_dq(
    graph: onnx.GraphProto,
    inits: dict[str, onnx.TensorProto],
    new_inits: dict[str, onnx.TensorProto],
    remove_inits: set[str],
) -> int:
    """Fold weight/bias DQ into fp16 constants for Mul, Conv, and Add consumers."""
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
            const_name = f"{w_name}_fp16"
            new_inits[const_name] = numpy_helper.from_array(
                dequantize_initializer(w, scale, zp), name=const_name
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


def fold_mul_weight_dq(
    graph: onnx.GraphProto,
    inits: dict[str, onnx.TensorProto],
    new_inits: dict[str, onnx.TensorProto],
    remove_inits: set[str],
) -> int:
    """Backward-compatible alias for norm-weight DQ folding (Mul only)."""
    return fold_weight_dq(graph, inits, new_inits, remove_inits)


def fold_gemm_int4_backbone_dq(
    graph: onnx.GraphProto,
    inits: dict[str, onnx.TensorProto],
    new_inits: dict[str, onnx.TensorProto],
    remove_inits: set[str],
) -> int:
    """Fold int4 backbone ``DQ -> Gemm`` weights into fp16 initializers."""
    dq_by_out = {d.output[0]: d for d in graph.node if d.op_type == "DequantizeLinear"}
    tensor_replace: dict[str, str] = {}
    remove_dq: set[str] = set()
    folded = 0
    for node in graph.node:
        if node.op_type != "Gemm":
            continue
        for idx in (1,):
            if idx >= len(node.input):
                continue
            dq = dq_by_out.get(node.input[idx])
            if dq is None or dq.input[0] not in inits:
                continue
            w_name = dq.input[0]
            if dq.input[1] not in inits or dq.input[2] not in inits:
                continue
            w = numpy_helper.to_array(inits[w_name])
            if w.dtype.name != "int4":
                continue
            scale = numpy_helper.to_array(inits[dq.input[1]])
            zp = numpy_helper.to_array(inits[dq.input[2]])
            const_name = f"{w_name}_fp16"
            new_inits[const_name] = numpy_helper.from_array(
                dequantize_weight_array(w, scale, zp), name=const_name
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


def convert_gemm_lora_input_dq_to_fp16(
    graph: onnx.GraphProto,
    inits: dict[str, onnx.TensorProto],
    remove_inits: set[str],
    lora_meta: dict[str, dict],
) -> int:
    """Replace LoRA ``graph_input int8 -> DQ -> Gemm`` with fp16 graph inputs."""
    graph_inputs = {vi.name: vi for vi in graph.input}
    dq_by_out = {d.output[0]: d for d in graph.node if d.op_type == "DequantizeLinear"}
    remove_dq: set[str] = set()
    converted = 0
    for gemm in graph.node:
        if gemm.op_type != "Gemm" or len(gemm.input) < 2:
            continue
        dq = dq_by_out.get(gemm.input[1])
        if dq is None or len(dq.input) < 3:
            continue
        w_name = dq.input[0]
        if w_name not in graph_inputs or "weight_quantized" not in w_name:
            continue
        if dq.input[1] not in inits or dq.input[2] not in inits:
            continue
        fp16_name = w_name.replace("weight_quantized", "weight_fp16")
        scale = numpy_helper.to_array(inits[dq.input[1]])
        zp = numpy_helper.to_array(inits[dq.input[2]])
        lora_meta[w_name] = {
            "onnx_input": fp16_name,
            "scale": float(scale.reshape(())),
            "zero_point": int(zp.reshape(())),
        }
        vi = graph_inputs[w_name]
        vi.name = fp16_name
        vi.type.tensor_type.elem_type = FP16__dup4
        graph_inputs[fp16_name] = vi
        del graph_inputs[w_name]
        gemm.input[1] = fp16_name
        remove_dq.add(dq.name)
        remove_inits.update({dq.input[1], dq.input[2]})
        converted += 1
    if remove_dq:
        kept = [n for n in graph.node if n.name not in remove_dq]
        del graph.node[:]
        graph.node.extend(kept)
    return converted


def set_matmul_nbits_accuracy_level(
    model: onnx.ModelProto, level: int | None = None
) -> int:
    """Force ``accuracy_level`` on every MatMulNBits node (incl. pre-existing ones).

    The exported lm_head ships as MatMulNBits with ``accuracy_level=4`` (int8
    activation), which degrades logits enough to pick wrong-but-adjacent tokens.
    """
    if level is None:
        level = MATMUL_NBITS_ACCURACY_LEVEL
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


def _promote_value_info_elem_type(vi: onnx.ValueInfoProto) -> None:
    elem_type = vi.type.tensor_type.elem_type
    if elem_type in (FP32__dup4, FP64):
        vi.type.tensor_type.elem_type = FP16__dup4


def promote_model_to_fp16(model: onnx.ModelProto) -> None:
    """Promote float32/float64 graph I/O and float initializers to float16."""
    graph = model.graph
    for vi in list(graph.input) + list(graph.output):
        _promote_value_info_elem_type(vi)
    for init in graph.initializer:
        if init.data_type == FP32__dup4:
            arr = numpy_helper.to_array(init).astype(np.float16)
            init.CopyFrom(numpy_helper.from_array(arr, name=init.name))
        elif init.data_type == FP64:
            arr = numpy_helper.to_array(init).astype(np.float16)
            init.CopyFrom(numpy_helper.from_array(arr, name=init.name))
    del graph.value_info[:]


def ensure_matmul_nbits_fp16_inputs__dup4(model: onnx.ModelProto) -> int:
    """Cast MatMulNBits activations to fp16 so ORT does not mix float/float16 on T1."""
    graph = model.graph
    new_nodes: list[onnx.NodeProto] = []
    count = 0
    for node in graph.node:
        if node.op_type != "MatMulNBits":
            new_nodes.append(node)
            continue
        src = node.input[0]
        cast_out = f"{node.name}_act_fp16"
        node.input[0] = cast_out
        new_nodes.append(
            helper.make_node(
                "Cast", [src], [cast_out], name=f"{node.name}_act_cast", to=FP16__dup4
            )
        )
        new_nodes.append(node)
        count += 1
    del graph.node[:]
    graph.node.extend(new_nodes)
    return count


def stabilize_lpnorm_fp16(model: onnx.ModelProto) -> int:
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
                "Cast", [src], [pre], name=f"{node.name}_cast_in", to=FP32__dup4
            )
        )
        new_nodes.append(node)
        new_nodes.append(
            helper.make_node(
                "Cast", [post], [dst], name=f"{node.name}_cast_out", to=FP16__dup4
            )
        )
        count += 1
    del graph.node[:]
    graph.node.extend(new_nodes)
    return count


def apply_new_initializers(
    graph: onnx.GraphProto, new_inits: dict[str, onnx.TensorProto]
) -> None:
    existing = {i.name: idx for idx, i in enumerate(graph.initializer)}
    for name, tensor in new_inits.items():
        if name in existing:
            graph.initializer[existing[name]] = tensor
        else:
            graph.initializer.append(tensor)


def prune_initializers(graph: onnx.GraphProto, remove_inits: set[str]) -> None:
    used = _referenced_initializer_names(graph)
    kept = [
        i for i in graph.initializer if i.name in used and i.name not in remove_inits
    ]
    del graph.initializer[:]
    graph.initializer.extend(kept)


def rewrite_gqa_past_seq_len_to_seqlens_k(model: onnx.ModelProto) -> int:
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


def save_decoder_model(
    model: onnx.ModelProto,
    dst: Path,
    *,
    external_data_name: str = DECODER_EXTERNAL_DATA,
) -> None:
    if not model.graph.name:
        model.graph.name = f"{dst.stem}_graph"
    dst.parent.mkdir(parents=True, exist_ok=True)
    data_path = dst.parent / external_data_name
    if dst.exists():
        dst.unlink()
    if data_path.exists():
        data_path.unlink()
    onnx.save_model(
        model,
        str(dst),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_name,
        size_threshold=1024,
    )
    onnx_size = dst.stat().st_size
    if onnx_size > 256 * 1024 * 1024:
        raise RuntimeError(
            f"{dst.name}: ONNX protobuf is {onnx_size} bytes; weights were not externalized to {external_data_name}"
        )


def _external_data_location__dup4(init: onnx.TensorProto) -> str | None:
    if init.data_location != TensorProto.EXTERNAL:
        return None
    for kv in init.external_data:
        if kv.key == "location":
            return kv.value
    return None


def infer_decoder_external_data_name(src: Path, bundle_root: Path) -> str:
    model = onnx.load(str(src), load_external_data=False)
    locations = {
        loc
        for init in model.graph.initializer
        if (loc := _external_data_location__dup4(init)) is not None
    }
    if len(locations) == 1:
        return locations.pop()
    data_files = sorted(bundle_root.glob("*.data"))
    if data_files:
        return data_files[0].name
    return DECODER_EXTERNAL_DATA


def save_model_with_external_data(
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
        location = _external_data_location__dup4(init)
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


def convert_decoder_model(
    src: Path,
    dst: Path,
    *,
    shape_fix: ShapeFixConfig,
    bundle_root: Path,
    external_data_name: str | None = None,
    lpnorm_fp32: bool = False,
    gqa_seqlens_rewrite: bool = True,
    pure_gemm: bool = False,
    lora_dequant_meta: dict[str, dict] | None = None,
) -> dict[str, int]:
    model = onnx.load(str(src), load_external_data=True)
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    new_inits: dict[str, onnx.TensorProto] = {}
    remove_inits: set[str] = set()
    stats = {
        "strip_qdq": strip_activation_qdq(graph),
        "conv_mnbits": fuse_conv_transpose_to_matmul_nbits(
            graph, inits, new_inits, remove_inits
        ),
        "matmul_mnbits": convert_matmul_weight_dq(
            graph, inits, new_inits, remove_inits
        ),
        "matmul_mnbits_lora": 0,
        "gemm_backbone_fold": 0,
        "gemm_lora_fp16": 0,
        "weight_dq_fold": 0,
    }
    if pure_gemm:
        lora_meta = lora_dequant_meta if lora_dequant_meta is not None else {}
        stats["gemm_backbone_fold"] = fold_gemm_int4_backbone_dq(
            graph, inits, new_inits, remove_inits
        )
        stats["gemm_lora_fp16"] = convert_gemm_lora_input_dq_to_fp16(
            graph, inits, remove_inits, lora_meta
        )
    elif has_graph_input_weight_quantized(graph):
        stats["matmul_mnbits_lora"] = convert_matmul_graph_input_int8_dq(
            graph, inits, new_inits, remove_inits
        )
    stats["weight_dq_fold"] = fold_weight_dq(graph, inits, new_inits, remove_inits)
    apply_new_initializers(graph, new_inits)
    prune_initializers(graph, remove_inits)
    promote_model_to_fp16(model)
    stats["mnbits_fp16_inputs"] = ensure_matmul_nbits_fp16_inputs__dup4(model)
    set_matmul_nbits_accuracy_level(model)
    if lpnorm_fp32:
        stats["lpnorm_fp32"] = stabilize_lpnorm_fp16(model)
        stats["gqa_getitem_fp32_removed"] = 0
        stats["mnbits_fp16_casts_folded"] = 0
        stats["fp32_casts_remaining"] = 0
    else:
        pure = optimize_fp16_activations(model)
        stats["lpnorm_fp32"] = 0
        stats.update(pure)
    maybe_fix_static_shapes(model, shape_fix)
    stats["gqa_seqlens_rewrite"] = (
        rewrite_gqa_past_seq_len_to_seqlens_k(model) if gqa_seqlens_rewrite else 0
    )
    ext_name = external_data_name or infer_decoder_external_data_name(src, bundle_root)
    save_decoder_model(model, dst, external_data_name=ext_name)
    return stats


def patch_emb_model(
    src: Path,
    dst: Path,
    *,
    shape_fix: ShapeFixConfig,
    rewrite_head_data: bool,
    head_data_dir: Path | None = None,
    lpnorm_fp32: bool = False,
) -> None:
    model = onnx.load(str(src), load_external_data=True)
    for node in model.graph.node:
        if node.op_type != "GatherBlockQuantized":
            continue
        if not any((a.name == "bits" for a in node.attribute)):
            node.attribute.extend([helper.make_attribute("bits", 4)])
    promote_model_to_fp16(model)
    if lpnorm_fp32:
        stabilize_lpnorm_fp16(model)
    maybe_fix_static_shapes(model, shape_fix)
    save_model_with_external_data(
        model, dst, rewrite_head_data=rewrite_head_data, head_data_dir=head_data_dir
    )


def convert_lm_head_model(
    src: Path,
    dst: Path,
    *,
    shape_fix: ShapeFixConfig,
    rewrite_head_data: bool,
    head_data_dir: Path | None = None,
    gather_unsqueeze: bool = True,
    lpnorm_fp32: bool = False,
) -> None:
    model = onnx.load(str(src), load_external_data=True)
    promote_model_to_fp16(model)
    set_matmul_nbits_accuracy_level(model)
    if lpnorm_fp32:
        stabilize_lpnorm_fp16(model)
    maybe_fix_static_shapes(model, shape_fix)
    if gather_unsqueeze:
        model = rewrite_lm_head_gather_unsqueeze(model)
    save_model_with_external_data(
        model, dst, rewrite_head_data=rewrite_head_data, head_data_dir=head_data_dir
    )


def _emb_shape_fix(shape_fix: ShapeFixConfig, seq_len: int) -> ShapeFixConfig:
    if not shape_fix.enabled:
        return shape_fix
    return ShapeFixConfig(
        enabled=True,
        batch=shape_fix.batch,
        seq_len=seq_len,
        max_seq_len=shape_fix.max_seq_len,
    )


def process_one_onnx(
    src: Path,
    dst: Path,
    *,
    kind: str,
    shape_fix: ShapeFixConfig,
    bundle_root: Path,
    head_data_dir: Path,
    rewrite_head_data: bool,
    lm_head_gather_unsqueeze: bool = True,
    lpnorm_fp32: bool = False,
    gqa_seqlens_rewrite: bool = True,
) -> dict[str, int] | None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if kind == "emb":
        patch_emb_model(
            src,
            dst,
            shape_fix=shape_fix,
            rewrite_head_data=rewrite_head_data,
            head_data_dir=head_data_dir,
            lpnorm_fp32=lpnorm_fp32,
        )
        return None
    if kind == "lm_head":
        convert_lm_head_model(
            src,
            dst,
            shape_fix=shape_fix,
            rewrite_head_data=rewrite_head_data,
            head_data_dir=head_data_dir,
            gather_unsqueeze=lm_head_gather_unsqueeze,
            lpnorm_fp32=lpnorm_fp32,
        )
        return None
    stats = convert_decoder_model(
        src,
        dst,
        shape_fix=shape_fix,
        bundle_root=bundle_root,
        lpnorm_fp32=lpnorm_fp32,
        gqa_seqlens_rewrite=gqa_seqlens_rewrite,
    )
    return stats
