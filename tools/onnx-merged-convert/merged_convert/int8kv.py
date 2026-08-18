#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.compose import add_prefix, merge_graphs

from .qdq_ext import (
    CONVERT_PROFILE_LITE,
    _align_model_ir_version_q,
    _inline_initializer_names,
    _pipeline_boundary_renames_q,
    _rename_tensors_in_graph_q,
    _topological_sort_graph_q,
    apply_new_initializers_q,
    assert_qdq_removed,
    convert_matmul_int8_weight_dq,
    fold_weight_dq_q,
    fuse_conv_transpose_to_matmul_nbits_q,
    prune_initializers_q,
    promote_model_to_fp16_q,
    rewrite_gqa_past_seq_len_to_seqlens_k_q,
    save_decoder_model_q,
    strip_activation_qdq_q,
    unwrap_castfp32_before_quantize,
)
from .step2_fp16_cleanup import optimize_fp16_activations
from .step3_lm_head import rewrite_lm_head_gather_unsqueeze

FP16_q = TensorProto.FLOAT16
FP32_q = TensorProto.FLOAT
INT8 = TensorProto.INT8
PIPELINE_PREFIXES = ("emb_", "dec_", "head_")
KV_IO_PARTS = ("past_keys_", "past_values_", "present_keys_", "present_values_")
GQA_ROPE_INPUTS = (7, 8)  # cos_cache, sin_cache


def assert_int8_kv_io(model: onnx.ModelProto, *, context: str = "") -> None:
    """Ensure KV cache graph I/O remain int8."""
    prefix = f"{context}: " if context else ""
    for vi in list(model.graph.input) + list(model.graph.output):
        if not any((part in vi.name for part in KV_IO_PARTS)):
            continue
        if vi.type.tensor_type.elem_type != INT8:
            raise RuntimeError(
                f"{prefix}{vi.name} must stay int8, got {vi.type.tensor_type.elem_type}"
            )


def assert_gqa_int8_quant(model: onnx.ModelProto, *, context: str = "") -> None:
    prefix = f"{context}: " if context else ""
    for node in model.graph.node:
        if node.op_type != "GroupQueryAttention":
            continue
        attrs = {a.name: a for a in node.attribute}
        kq = attrs.get("k_quant_type")
        vq = attrs.get("v_quant_type")
        bw = attrs.get("kv_cache_bit_width")
        if kq is None or vq is None or bw is None:
            raise RuntimeError(f"{prefix}GQA missing quant attributes on {node.name}")
        if kq.s not in (b"PER_CHANNEL", b"PER_TENSOR"):
            raise RuntimeError(f"{prefix}unexpected k_quant_type={kq.s!r}")
        if vq.s not in (b"PER_CHANNEL", b"PER_TENSOR"):
            raise RuntimeError(f"{prefix}unexpected v_quant_type={vq.s!r}")
        if bw.i != 8:
            raise RuntimeError(f"{prefix}expected kv_cache_bit_width=8, got {bw.i}")


def _cast_to_q(node: onnx.NodeProto) -> int | None:
    if node.op_type != "Cast":
        return None
    for attr in node.attribute:
        if attr.name == "to":
            return int(attr.i)
    return None


def _graph_producers(graph: onnx.GraphProto) -> dict[str, onnx.NodeProto]:
    producer: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for out in node.output:
            if out:
                producer[out] = node
    return producer


def _is_int8_kv_gqa(node: onnx.NodeProto) -> bool:
    if node.op_type != "GroupQueryAttention":
        return False
    attrs = {a.name: a for a in node.attribute}
    bw = attrs.get("kv_cache_bit_width")
    kq = attrs.get("k_quant_type")
    return bw is not None and bw.i == 8 and kq is not None and kq.s != b"NONE"


def _trace_to_initializers(
    tensor: str,
    producer: dict[str, onnx.NodeProto],
    init_names: set[str],
    *,
    seen: set[str] | None = None,
) -> set[str]:
    """Follow RoPE tensor producers back to graph initializers."""
    if not tensor:
        return set()
    if seen is None:
        seen = set()
    if tensor in seen:
        return set()
    seen.add(tensor)
    if tensor in init_names:
        return {tensor}
    node = producer.get(tensor)
    if node is None:
        return set()
    if node.op_type in {"Cast", "Identity"} and node.input:
        return _trace_to_initializers(node.input[0], producer, init_names, seen=seen)
    if node.op_type in {"Gather", "Slice"} and node.input:
        return _trace_to_initializers(node.input[0], producer, init_names, seen=seen)
    return set()


def _promote_initializer_to_fp16(init: onnx.TensorProto) -> bool:
    if init.data_type != FP32_q:
        return False
    arr = numpy_helper.to_array(init).astype(np.float16)
    init.CopyFrom(numpy_helper.from_array(arr, name=init.name))
    return True


def _rewire_tensor_consumers(
    graph: onnx.GraphProto, old_name: str, new_name: str
) -> int:
    rewired = 0
    for consumer in graph.node:
        for i, inp in enumerate(consumer.input):
            if inp == old_name:
                consumer.input[i] = new_name
                rewired += 1
    return rewired


def ensure_gqa_rope_weights_fp16(model: onnx.ModelProto) -> dict[str, int]:
    """Promote RoPE cos/sin weights to fp16 for hip-ep fused int8-KV GQA.

    Discovers cos/sin tensors from GroupQueryAttention inputs 7 and 8, traces
    them back to fp32 initializers, and removes fp16->fp32 Cast wrappers so
    GQA consumes fp16 RoPE tables directly.
    """
    graph = model.graph
    init_by_name = {i.name: i for i in graph.initializer}
    init_names = set(init_by_name)
    producer = _graph_producers(graph)

    stats = {
        "rope_inits_promoted": 0,
        "rope_cast_removed": 0,
        "rope_consumers_rewired": 0,
    }
    promote: set[str] = set()
    remove: set[str] = set()

    for node in graph.node:
        if not _is_int8_kv_gqa(node):
            continue
        for idx in GQA_ROPE_INPUTS:
            if idx >= len(node.input) or not node.input[idx]:
                continue
            rope_tensor = node.input[idx]
            cast_node = producer.get(rope_tensor)
            if (
                cast_node is not None
                and cast_node.op_type == "Cast"
                and _cast_to_q(cast_node) == FP32_q
                and cast_node.input
            ):
                fp32_name = cast_node.output[0]
                fp16_name = cast_node.input[0]
                node.input[idx] = fp16_name
                stats["rope_cast_removed"] += 1
                stats["rope_consumers_rewired"] += _rewire_tensor_consumers(
                    graph, fp32_name, fp16_name
                )
                remove.add(cast_node.name)
                rope_tensor = fp16_name
            promote |= _trace_to_initializers(rope_tensor, producer, init_names)

    for name in promote:
        init = init_by_name.get(name)
        if init is not None and _promote_initializer_to_fp16(init):
            stats["rope_inits_promoted"] += 1

    if remove:
        kept = [n for n in graph.node if n.name not in remove]
        del graph.node[:]
        graph.node.extend(kept)
    return stats


def convert_decoder_int8kv(
    src: Path, dst: Path, *, bundle_root: Path, gqa_seqlens_rewrite: bool = True
) -> dict[str, int]:
    """Q/DQ strip + Conv/MNB + fp16 activations; preserve int8 KV GQA."""
    profile = CONVERT_PROFILE_LITE
    model = onnx.load(str(src), load_external_data=True)
    inline_at_load = _inline_initializer_names(model)
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    new_inits: dict[str, onnx.TensorProto] = {}
    remove_inits: set[str] = set()
    stats: dict[str, int] = {}
    stats["unwrap_castfp32"] = unwrap_castfp32_before_quantize(graph)
    stats["strip_qdq"] = strip_activation_qdq_q(
        graph, inits, new_inits, pure_fp16=True, mode=profile.activation_qdq_mode
    )
    stats["conv_mnbits"] = fuse_conv_transpose_to_matmul_nbits_q(
        graph, inits, new_inits, remove_inits
    )
    stats["matmul_int8_mnbits"] = convert_matmul_int8_weight_dq(
        graph, inits, new_inits, remove_inits
    )
    stats["weight_dq_fold"] = fold_weight_dq_q(
        graph, inits, new_inits, remove_inits, weight_dtype=np.float16
    )
    if new_inits or remove_inits:
        apply_new_initializers_q(graph, new_inits)
        prune_initializers_q(graph, remove_inits)
    promote_model_to_fp16_q(
        model,
        skip_initializer_substrings=("_emu_scale", "_emu_zp", "k_scale_", "v_scale_"),
    )
    fp16_stats = optimize_fp16_activations(model)
    stats["gqa_getitem_fp32_removed"] = fp16_stats["gqa_getitem_fp32_removed"]
    stats["mnbits_fp16_casts_folded"] = fp16_stats["mnbits_fp16_casts_folded"]
    rope_stats = ensure_gqa_rope_weights_fp16(model)
    for key, value in rope_stats.items():
        stats[f"rope_{key}"] = value
    # Re-count fp32 casts after RoPE rewiring.
    fp16_stats = optimize_fp16_activations(model)
    stats["fp32_casts_remaining"] = fp16_stats["fp32_casts_remaining"]
    if stats["fp32_casts_remaining"]:
        raise RuntimeError(
            f"{dst.name}: {stats['fp32_casts_remaining']} Cast->fp32 remain; pure fp16 activations required"
        )
    stats["gqa_seqlens_rewrite"] = (
        rewrite_gqa_past_seq_len_to_seqlens_k_q(model) if gqa_seqlens_rewrite else 0
    )
    assert_qdq_removed(graph, context=dst.name)
    assert_int8_kv_io(model, context=dst.name)
    assert_gqa_int8_quant(model, context=dst.name)
    ext_name = f"{dst.stem}.data"
    save_decoder_model_q(
        model, dst, external_data_name=ext_name, keep_inline=inline_at_load
    )
    return stats


def merge_pipeline(
    emb_path: Path,
    decoder_path: Path,
    head_path: Path,
    dst_path: Path,
    *,
    external_data_name: str = "merged.data",
) -> None:
    emb = onnx.load(str(emb_path), load_external_data=True)
    decoder = onnx.load(str(decoder_path), load_external_data=True)
    head = onnx.load(str(head_path), load_external_data=True)
    head = rewrite_lm_head_gather_unsqueeze(head)
    target_ir = max(emb.ir_version, decoder.ir_version, head.ir_version)
    _align_model_ir_version_q(emb, target_ir)
    _align_model_ir_version_q(head, target_ir)
    emb_p = add_prefix(emb, prefix=PIPELINE_PREFIXES[0])
    dec_p = add_prefix(decoder, prefix=PIPELINE_PREFIXES[1])
    head_p = add_prefix(head, prefix=PIPELINE_PREFIXES[2])
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
