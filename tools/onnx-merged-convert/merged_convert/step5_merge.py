#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

from __future__ import annotations

from collections import deque
from pathlib import Path

import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.compose import add_prefix, merge_graphs


_PIPELINE_PREFIXES = ("emb_", "dec_", "head_")
DEFAULT_EXTERNAL_DATA = "merged.data"


def _align_model_ir_version(model: onnx.ModelProto, ir_version: int) -> None:
    model.ir_version = ir_version


def _topological_sort_graph(graph: onnx.GraphProto) -> None:
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


def _rename_tensors_in_graph(graph: onnx.GraphProto, mapping: dict[str, str]) -> None:
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


def _pipeline_boundary_renames(graph: onnx.GraphProto) -> dict[str, str]:
    """Map prefixed merge tensors back to split-pipeline I/O names."""
    mapping: dict[str, str] = {
        "emb_input_ids": "input_ids",
        "dec_past_seq_len": "past_seq_len",
        "dec_total_seq_len": "total_seq_len",
        "head_logits": "logits",
    }
    for vi in list(graph.input) + list(graph.output):
        name = vi.name
        if name.startswith("dec_past_keys_"):
            mapping[name] = name.removeprefix("dec_")
        elif name.startswith("dec_past_values_"):
            mapping[name] = name.removeprefix("dec_")
        elif name.startswith("dec_present_keys_"):
            mapping[name] = name.removeprefix("dec_")
        elif name.startswith("dec_present_values_"):
            mapping[name] = name.removeprefix("dec_")
        elif name.startswith("dec_model.") and name.endswith("weight_quantized"):
            mapping[name] = name.removeprefix("dec_")
        elif name.startswith("dec_model.") and name.endswith("weight_fp16"):
            mapping[name] = name.removeprefix("dec_")
    return mapping


def _merged_opset_imports(*models: onnx.ModelProto) -> list[onnx.OperatorSetIdProto]:
    versions: dict[str, int] = {}
    for model in models:
        for oi in model.opset_import:
            versions[oi.domain] = max(versions.get(oi.domain, 0), int(oi.version))
    return [
        helper.make_opsetid(domain, ver) for domain, ver in sorted(versions.items())
    ]


def _inline_small_external_initializers(
    model: onnx.ModelProto, *, max_bytes: int = 65536
) -> int:
    """ORT shape inference requires shape-like constants inline, not in .data."""
    inlined = 0
    for init in model.graph.initializer:
        if init.data_location != TensorProto.EXTERNAL:
            continue
        arr = numpy_helper.to_array(init)
        if arr.nbytes > max_bytes:
            continue
        init.CopyFrom(numpy_helper.from_array(arr, name=init.name))
        inlined += 1
    return inlined


def merge_split_pipeline_models(
    emb_path: Path,
    decoder_path: Path,
    head_path: Path,
    dst_path: Path,
    *,
    external_data_name: str = DEFAULT_EXTERNAL_DATA,
) -> Path:
    """Fuse emb -> decoder -> lm_head into one ONNX."""
    emb = onnx.load(str(emb_path), load_external_data=True)
    decoder = onnx.load(str(decoder_path), load_external_data=True)
    head = onnx.load(str(head_path), load_external_data=True)
    target_ir = max(emb.ir_version, decoder.ir_version, head.ir_version)
    _align_model_ir_version(emb, target_ir)
    _align_model_ir_version(head, target_ir)
    emb_p = add_prefix(emb, prefix=_PIPELINE_PREFIXES[0])
    dec_p = add_prefix(decoder, prefix=_PIPELINE_PREFIXES[1])
    head_p = add_prefix(head, prefix=_PIPELINE_PREFIXES[2])
    graph = merge_graphs(
        emb_p.graph,
        dec_p.graph,
        io_map=[("emb_input_hidden_states", "dec_input_hidden_states")],
    )
    kv_outputs = [out.name for out in graph.output if "present_" in out.name]
    graph = merge_graphs(
        graph,
        head_p.graph,
        io_map=[("dec_output_hidden_states", "head_output_hidden_states")],
        outputs=["head_logits", *kv_outputs],
    )
    _topological_sort_graph(graph)
    _rename_tensors_in_graph(graph, _pipeline_boundary_renames(graph))
    logits_out = next((vi for vi in graph.output if vi.name == "logits"), None)
    if logits_out is not None:
        other_outs = [vi for vi in graph.output if vi.name != "logits"]
        del graph.output[:]
        graph.output.extend([logits_out, *other_outs])
    model = helper.make_model(
        graph, opset_imports=_merged_opset_imports(emb, decoder, head)
    )
    model.ir_version = target_ir
    model.graph.name = f"{dst_path.stem}_graph"
    _inline_small_external_initializers(model)
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
