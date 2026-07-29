#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper
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
    infer_decoder_external_data_name_q,
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


def _count_fp32_casts(model: onnx.ModelProto) -> int:
    return sum(
        (
            1
            for n in model.graph.node
            if n.op_type == "Cast"
            and any((a.name == "to" and a.i == FP32_q for a in n.attribute))
        )
    )


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


def wrap_gqa_fp32_query_for_int8_kv(model: onnx.ModelProto) -> int:
    """ORT CPU GQA with int8 KV requires fp32 Q and matching float aux inputs."""
    graph = model.graph
    float_input_indices = (0, 7, 8)
    new_nodes: list[onnx.NodeProto] = []
    wrapped = 0
    for node in graph.node:
        if node.op_type != "GroupQueryAttention":
            new_nodes.append(node)
            continue
        attrs = {a.name: a for a in node.attribute}
        bw = attrs.get("kv_cache_bit_width")
        kq = attrs.get("k_quant_type")
        if bw is None or bw.i != 8 or kq is None or (kq.s == b"NONE"):
            new_nodes.append(node)
            continue
        cast_nodes: list[onnx.NodeProto] = []
        for idx in float_input_indices:
            if idx >= len(node.input):
                continue
            src = node.input[idx]
            if not src:
                continue
            fp32_name = f"{node.name}_in{idx}_fp32"
            cast_nodes.append(
                helper.make_node(
                    "Cast",
                    [src],
                    [fp32_name],
                    name=f"{node.name}_in{idx}_cast_fp32",
                    to=FP32_q,
                )
            )
            node.input[idx] = fp32_name
        out_fp32 = f"{node.name}_out_fp32"
        out_orig = node.output[0]
        node.output[0] = out_fp32
        new_nodes.extend(cast_nodes)
        new_nodes.append(node)
        new_nodes.append(
            helper.make_node(
                "Cast",
                [out_fp32],
                [out_orig],
                name=f"{node.name}_out_cast_fp16",
                to=FP16_q,
            )
        )
        wrapped += 1
    del graph.node[:]
    graph.node.extend(new_nodes)
    return wrapped


def _cast_to_q(node: onnx.NodeProto) -> int | None:
    if node.op_type != "Cast":
        return None
    for attr in node.attribute:
        if attr.name == "to":
            return int(attr.i)
    return None


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
        graph, inits, new_inits, remove_inits, weight_dtype=__import__("numpy").float16
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
    stats["gqa_fp32_query_wrap"] = wrap_gqa_fp32_query_for_int8_kv(model)
    stats["fp32_casts_remaining"] = _count_fp32_casts(model)
    allowed_fp32_casts = 3 * stats["gqa_fp32_query_wrap"]
    if stats["fp32_casts_remaining"] > allowed_fp32_casts:
        raise RuntimeError(
            f"{dst.name}: {stats['fp32_casts_remaining']} Cast->fp32 remain (allowed {allowed_fp32_casts} at GQA boundaries)"
        )
    stats["gqa_seqlens_rewrite"] = (
        rewrite_gqa_past_seq_len_to_seqlens_k_q(model) if gqa_seqlens_rewrite else 0
    )
    assert_qdq_removed(graph, context=dst.name)
    assert_int8_kv_io(model, context=dst.name)
    assert_gqa_int8_quant(model, context=dst.name)
    ext_name = infer_decoder_external_data_name_q(src, bundle_root)
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
