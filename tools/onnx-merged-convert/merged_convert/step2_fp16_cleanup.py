#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

from __future__ import annotations

import re
from pathlib import Path

import onnx
from onnx import TensorProto, shape_inference

FP32 = TensorProto.FLOAT
FP16 = TensorProto.FLOAT16
_FLOAT_ELEM_TYPES = frozenset(
    {
        TensorProto.FLOAT,
        TensorProto.FLOAT16,
        TensorProto.DOUBLE,
        TensorProto.BFLOAT16,
    }
)
_PRE_CAST_CASTFP32_RE = re.compile(
    "(?:^getitem(?:_\\d+)?_pre_cast_castfp32$|GroupQueryAttention_output_\\d+_pre_cast_castfp32$)"
)


def _cast_to(node: onnx.NodeProto) -> int | None:
    if node.op_type != "Cast":
        return None
    for attr in node.attribute:
        if attr.name == "to":
            return int(attr.i)
    return None


def _replace_tensor_uses(graph: onnx.GraphProto, old: str, new: str) -> int:
    if old == new:
        return 0
    replacements = 0
    for node in graph.node:
        for idx, name in enumerate(node.input):
            if name == old:
                node.input[idx] = new
                replacements += 1
    for vi in list(graph.output) + list(graph.value_info):
        if vi.name == old:
            vi.name = new
            replacements += 1
    return replacements


def _build_tensor_users(
    graph: onnx.GraphProto,
) -> dict[str, list[tuple[onnx.NodeProto, int]]]:
    users: dict[str, list[tuple[onnx.NodeProto, int]]] = {}
    for node in graph.node:
        for idx, name in enumerate(node.input):
            if name:
                users.setdefault(name, []).append((node, idx))
    return users


def _infer_activation_elem_types(graph: onnx.GraphProto) -> dict[str, int]:
    """Best-effort elem_type map for activation tensors."""
    types: dict[str, int] = {}
    for init in graph.initializer:
        types[init.name] = init.data_type
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        elem_type = vi.type.tensor_type.elem_type
        if elem_type:
            types[vi.name] = elem_type

    changed = True
    while changed:
        changed = False
        for node in graph.node:
            if node.op_type == "Cast":
                to_type = _cast_to(node)
                if to_type is None or not node.output:
                    continue
                out = node.output[0]
                if types.get(out) != to_type:
                    types[out] = to_type
                    changed = True
                continue

            in_types = [
                types[inp]
                for inp in node.input
                if inp and inp in types and types[inp] in _FLOAT_ELEM_TYPES
            ]
            if not in_types or not node.output:
                continue
            if all(t == FP16 for t in in_types):
                out_type = FP16
            elif len(set(in_types)) == 1:
                out_type = in_types[0]
            else:
                continue
            for out in node.output:
                if not out or types.get(out) == out_type:
                    continue
                types[out] = out_type
                changed = True
    return types


def _is_fp16_consumer_op(node: onnx.NodeProto) -> bool:
    """True when a downstream op consumes fp16 activations on this edge."""
    if node.op_type == "Cast":
        return _cast_to(node) == FP16
    return True


def _keep_gqa_pre_cast_fp16_cast(src: str, dst: str) -> bool:
    return src.endswith("_pre_cast") and "GroupQueryAttention_output_" in dst


def _remove_nodes(graph: onnx.GraphProto, names: set[str]) -> int:
    kept = [n for n in graph.node if n.name not in names]
    removed = len(graph.node) - len(kept)
    del graph.node[:]
    graph.node.extend(kept)
    return removed


def unwrap_lpnorm_fp32_casts(model: onnx.ModelProto) -> int:
    """Remove Cast(fp32)->LpNormalization->Cast(fp16) wrappers."""
    graph = model.graph
    remove: set[str] = set()
    unwrapped = 0
    for node in graph.node:
        if node.op_type != "LpNormalization" or not node.input or (not node.output):
            continue
        lpnorm = node
        pre = lpnorm.input[0]
        post = lpnorm.output[0]
        cast_in = next(
            (
                n
                for n in graph.node
                if n.op_type == "Cast"
                and n.output
                and (n.output[0] == pre)
                and (_cast_to(n) == FP32)
            ),
            None,
        )
        cast_out = next(
            (
                n
                for n in graph.node
                if n.op_type == "Cast"
                and n.input
                and (n.input[0] == post)
                and (_cast_to(n) == FP16)
            ),
            None,
        )
        if (
            cast_in is None
            or cast_out is None
            or (not cast_in.input)
            or (not cast_out.output)
        ):
            continue
        fp16_src = cast_in.input[0]
        fp16_dst = cast_out.output[0]
        lpnorm.input[0] = fp16_src
        lpnorm.output[0] = fp16_dst
        remove.add(cast_in.name)
        remove.add(cast_out.name)
        unwrapped += 1
    _remove_nodes(graph, remove)
    return unwrapped


def rewrite_gqa_pre_cast_castfp32_to_fp16(model: onnx.ModelProto) -> int:
    """Runtime 3.2.0: keep ``pre_cast -> cast -> output_0`` wiring; use fp16 cast instead of fp32."""
    graph = model.graph
    changed = 0
    for node in graph.node:
        if node.op_type != "Cast" or _cast_to(node) != FP32:
            continue
        if not _PRE_CAST_CASTFP32_RE.search(node.name):
            continue
        for attr in node.attribute:
            if attr.name == "to":
                attr.i = int(FP16)
                changed += 1
    return changed


def remove_gqa_getitem_fp32_casts(model: onnx.ModelProto) -> int:
    """Bypass Cast(fp32) between GQA ``*_pre_cast`` and downstream users."""
    graph = model.graph
    remove: set[str] = set()
    removed = 0
    for node in graph.node:
        if node.op_type != "Cast" or _cast_to(node) != FP32:
            continue
        if not _PRE_CAST_CASTFP32_RE.search(node.name):
            continue
        if not node.input or not node.output:
            continue
        fp16_src = node.input[0]
        fp32_dst = node.output[0]
        _replace_tensor_uses(graph, fp32_dst, fp16_src)
        remove.add(node.name)
        removed += 1
    _remove_nodes(graph, remove)
    return removed


def count_fp32_activation_casts(model: onnx.ModelProto) -> int:
    return sum(
        (1 for n in model.graph.node if n.op_type == "Cast" and _cast_to(n) == FP32)
    )


def _tensor_consumer_count(graph: onnx.GraphProto, tensor: str) -> int:
    return sum((1 for node in graph.node for inp in node.input if inp == tensor))


def _emu_fp32_boundary_tensor(graph: onnx.GraphProto, tensor: str) -> bool:
    """True when *tensor* is an fp32 fake-quant intermediate (must keep Cast->fp16)."""
    if "_emu_" in tensor:
        return True
    producer: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for out in node.output:
            if out:
                producer[out] = node
    node = producer.get(tensor)
    if node is None:
        return False
    if node.op_type == "Cast" and _cast_to(node) == FP32:
        return True
    if node.op_type in {"Div", "Add", "Round", "Sub", "Mul"} and "_emu_" in node.name:
        return True
    return _emu_fp32_boundary_tensor(graph, node.input[0]) if node.input else False


def remove_redundant_mnbits_fp16_casts(model: onnx.ModelProto) -> int:
    """Drop identity Cast(fp16) nodes that only feed MatMulNBits T1."""
    graph = model.graph
    producer: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for out in node.output:
            if out:
                producer[out] = node
    remove: set[str] = set()
    folded = 0
    for mmb in graph.node:
        if mmb.op_type != "MatMulNBits" or not mmb.input:
            continue
        act = mmb.input[0]
        cast = producer.get(act)
        if cast is None or cast.op_type != "Cast" or _cast_to(cast) != FP16:
            continue
        if not cast.input or not cast.output:
            continue
        if _tensor_consumer_count(graph, act) != 1:
            continue
        if _emu_fp32_boundary_tensor(graph, cast.input[0]):
            continue
        src = cast.input[0]
        act_out = cast.output[0]
        if src.endswith("_pre_cast") and "GroupQueryAttention_output_" in act_out:
            continue
        mmb.input[0] = cast.input[0]
        remove.add(cast.name)
        folded += 1
    _remove_nodes(graph, remove)
    return folded


def remove_redundant_fp16_casts(model: onnx.ModelProto) -> int:
    """Drop Cast(fp16) when upstream and downstream ops already use fp16."""
    graph = model.graph
    total = 0
    while True:
        elem_types = _infer_activation_elem_types(graph)
        users = _build_tensor_users(graph)
        remove: set[str] = set()
        removed = 0
        for cast in graph.node:
            if cast.op_type != "Cast" or _cast_to(cast) != FP16:
                continue
            if not cast.input or not cast.output:
                continue
            src = cast.input[0]
            dst = cast.output[0]
            if elem_types.get(src) != FP16:
                continue
            if _emu_fp32_boundary_tensor(graph, src):
                continue
            if _keep_gqa_pre_cast_fp16_cast(src, dst):
                continue
            consumers = users.get(dst, [])
            if not consumers or any(
                not _is_fp16_consumer_op(node) for node, _idx in consumers
            ):
                continue
            _replace_tensor_uses(graph, dst, src)
            remove.add(cast.name)
            removed += 1
        _remove_nodes(graph, remove)
        total += removed
        if removed == 0:
            break
    return total


def ensure_matmul_nbits_fp16_inputs(model: onnx.ModelProto) -> int:
    """Cast MatMulNBits activations to fp16 so ORT does not mix float/float16 on T1."""
    graph = model.graph
    producer: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for out in node.output:
            if out:
                producer[out] = node
    new_nodes: list[onnx.NodeProto] = []
    count = 0
    for node in graph.node:
        if node.op_type != "MatMulNBits":
            new_nodes.append(node)
            continue
        src = node.input[0]
        cast_node = producer.get(src)
        if (
            cast_node is not None
            and cast_node.op_type == "Cast"
            and (_cast_to(cast_node) == FP16)
        ):
            new_nodes.append(node)
            continue
        cast_out = f"{node.name}_act_fp16"
        node.input[0] = cast_out
        new_nodes.append(
            onnx.helper.make_node(
                "Cast", [src], [cast_out], name=f"{node.name}_act_cast", to=FP16
            )
        )
        new_nodes.append(node)
        count += 1
    del graph.node[:]
    graph.node.extend(new_nodes)
    return count


def optimize_fp16_activations(
    model: onnx.ModelProto,
    *,
    clear_value_info: bool = True,
    remove_gqa_fp32_casts: bool = True,
    rewrite_gqa_cast_fp32_to_fp16: bool = False,
) -> dict[str, int]:
    """Apply all fp16 activation cleanup passes."""
    stats = {
        "lpnorm_unwrapped": unwrap_lpnorm_fp32_casts(model),
        "gqa_getitem_fp32_removed": remove_gqa_getitem_fp32_casts(model)
        if remove_gqa_fp32_casts
        else 0,
        "gqa_cast_fp32_to_fp16": rewrite_gqa_pre_cast_castfp32_to_fp16(model)
        if rewrite_gqa_cast_fp32_to_fp16
        else 0,
        "mnbits_fp16_casts_folded": remove_redundant_mnbits_fp16_casts(model),
        "mnbits_fp16_inputs": ensure_matmul_nbits_fp16_inputs(model),
        "fp16_casts_removed": remove_redundant_fp16_casts(model),
    }
    if clear_value_info:
        del model.graph.value_info[:]
    stats["fp32_casts_remaining"] = count_fp32_activation_casts(model)
    return stats


def promote_pure_fp16_activations(
    model: onnx.ModelProto, *, clear_value_info: bool = True
) -> dict[str, int]:
    """Strip fp32 activation casts; return rewrite stats."""
    return optimize_fp16_activations(model, clear_value_info=clear_value_info)


def _save_model_graph_only(model: onnx.ModelProto, dst: Path) -> None:
    """Write graph protobuf only; keep existing external weight blob references."""
    if not model.graph.name:
        model.graph.name = f"{dst.stem}_graph"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    onnx.save(model, str(dst))
    if dst.stat().st_size > 256 * 1024 * 1024:
        raise RuntimeError(
            f"{dst.name}: ONNX protobuf is {dst.stat().st_size} bytes after graph-only save; "
            "load with load_external_data=False to preserve external initializers"
        )


def patch_model_file(
    src: Path, dst: Path, *, reinfer_shapes: bool = False
) -> dict[str, int]:
    model = onnx.load(str(src), load_external_data=False)
    stats = optimize_fp16_activations(model)
    if stats["fp32_casts_remaining"]:
        raise RuntimeError(
            f"{src.name}: {stats['fp32_casts_remaining']} Cast->fp32 node(s) remain after rewrite"
        )
    if reinfer_shapes:
        model = shape_inference.infer_shapes(model)
    _save_model_graph_only(model, dst)
    return stats
