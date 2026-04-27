#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Universal ONNX submodel extractor for LLMs.

Auto-discovers graph topology — no hardcoded node names needed.
Works with any model following the standard LLM pattern:
  embed_tokens -> N transformer layers -> final_norm -> lm_head

Tested with: Qwen2.5 (FP16), DeepSeek-R1 (INT4), Mistral-7B (INT4).

Produces:
  single_op/<OpType>/<name>_<variant>.onnx   (8 variants each)
  single_layer/single_layer_<variant>.onnx   (8 variants, shared ext data)
  full_model/full_model_<variant>.onnx       (8 variants, shared ext data)

Variants: dynamic, seq1, seq128, seq256, seq512, seq1024, seq2048, seq3072

Rules applied to all extractions:
  - Delete branch: attention_mask -> Shape -> Gather -> Cast
  - Delete branch: input_ids -> Shape -> Gather -> Unsqueeze -> Concat -> Reshape
    (only if present; e.g. absent in DeepSeek)
  - Connect position_ids directly to RotaryEmbedding (when Reshape chain exists)
  - Replace GQA total_seq_len with constant = sequence_length
  - Orphaned Constant/initializer nodes are auto-removed
  - 8 variants: 1 dynamic + 7 fixed seq lengths; batch_size fixed to 1

Usage:
  python extract_submodels.py --model path/to/model.onnx
  python extract_submodels.py --model path/to/model.onnx --only single_op
  python extract_submodels.py --model path/to/model.onnx --output /out/dir
"""

import os
import re
import json
import shutil
import argparse
from typing import Optional

import numpy as np
from collections import defaultdict, Counter

import onnx
from onnx import helper, TensorProto, numpy_helper

try:
    import onnxoptimizer

    _HAS_OPTIMIZER = True
except ImportError:
    _HAS_OPTIMIZER = False

SEQ_VARIANTS = [
    ("dynamic", None),
    ("seq1", 1),
    ("seq128", 128),
    ("seq256", 256),
    ("seq512", 512),
    ("seq1024", 1024),
    ("seq2048", 2048),
    ("seq3072", 3072),
]

VISION_VARIANTS = [
    ("dynamic", None),
    ("np1024", {"num_patches": 1024, "num_logical_patches": 256}),
    ("np1200", {"num_patches": 1200, "num_logical_patches": 300}),
    ("np2520", {"num_patches": 2520, "num_logical_patches": 630}),
    ("np3600", {"num_patches": 3600, "num_logical_patches": 900}),
    ("np4096", {"num_patches": 4096, "num_logical_patches": 1024}),
    ("np8160", {"num_patches": 8160, "num_logical_patches": 2040}),
]

EMBEDDING_VARIANTS = [
    ("dynamic", None),
    ("seq1_if0", {"sequence_length": 1, "num_logical_patches": 0}),
    ("seq1_if2040", {"sequence_length": 1, "num_logical_patches": 2040}),
    ("seq128_if0", {"sequence_length": 128, "num_logical_patches": 0}),
    ("seq128_if2040", {"sequence_length": 128, "num_logical_patches": 2040}),
    ("seq512_if0", {"sequence_length": 512, "num_logical_patches": 0}),
    ("seq512_if2040", {"sequence_length": 512, "num_logical_patches": 2040}),
    ("seq1024_if0", {"sequence_length": 1024, "num_logical_patches": 0}),
    ("seq1024_if2040", {"sequence_length": 1024, "num_logical_patches": 2040}),
    ("seq2048_if0", {"sequence_length": 2048, "num_logical_patches": 0}),
    ("seq2048_if2040", {"sequence_length": 2048, "num_logical_patches": 2040}),
    ("seq3072_if0", {"sequence_length": 3072, "num_logical_patches": 0}),
    ("seq3072_if2040", {"sequence_length": 3072, "num_logical_patches": 2040}),
]

BATCH_SIZE = 1
TOTAL_SEQ_LEN_CONST_NAME = "total_seq_len_const"
# Llama exporters: ``Constant fold/total_seq_len`` → ``total_seq_len_constant`` (VI name).
FOLD_TOTAL_SEQ_NODE_NAME = "fold/total_seq_len"
TOTAL_SEQ_LEN_CONSTANT_OUT = "total_seq_len_constant"

PERF_SIGNIFICANT_OPS = {
    "Gemm",
    "MatMul",
    "MatMulNBits",
    "Conv",
    "ConvTranspose",
    "GroupQueryAttention",
    "LinearAttention",
    "QMoE",
    "CausalConvWithState",
    "RotaryEmbedding",
    "LayerNormalization",
    "SimplifiedLayerNormalization",
    "SkipSimplifiedLayerNormalization",
    "BatchNormalization",
    "LSTM",
    "GRU",
    "Loop",
}


# ─────────────────────── Graph Analyzer ───────────────────────


class GraphAnalyzer:
    """Auto-discover graph topology patterns for LLM ONNX models.

    Discovers:
      - Number of transformer layers
      - Preprocessing chains to delete (attn_mask, position_ids reshape)
      - Key structural nodes (embed_tokens, final_norm, lm_head)
      - Op type configuration (MatMul vs MatMulNBits, RotaryEmbedding)
      - Tensor rewiring map
      - Pre-layer dependency nodes (via backward traversal from layer 0)
    """

    LAYER_PATTERN = re.compile(r"/layers?[./](\d+)[/.]")

    def __init__(self, graph):
        self.graph = graph

        self.output_to_node = {}
        self.input_to_nodes = defaultdict(list)
        self.node_by_name = {}
        for node in graph.node:
            if node.name:
                self.node_by_name[node.name] = node
            for out in node.output:
                if out:
                    self.output_to_node[out] = node
            for inp in node.input:
                if inp:
                    self.input_to_nodes[inp].append(node)

        self.graph_input_names = {inp.name for inp in graph.input}
        self.init_names = {init.name for init in graph.initializer}

        self.constant_outputs = {}
        for node in graph.node:
            if node.op_type == "Constant":
                for attr in node.attribute:
                    if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                        for out in node.output:
                            if out:
                                self.constant_outputs[out] = attr.t

        self.num_layers = 0
        self.layer_indices = set()
        self.nodes_to_delete = set()
        self.deleted_outputs = set()
        self.rewire_map = {}
        self.attn_mask_chain_output = None
        self.reshape_chain_output = None
        self.embed_node = None
        self.final_norm_node = None
        self.lm_head_node = None
        self.shape_diff_ops = set()
        self.pre_layer_nodes = []
        self.post_layer_nodes = []

    @property
    def has_layers(self):
        return self.num_layers > 0

    def analyze(self):
        print("\n=== Auto-discovering graph topology ===")
        self._detect_layers()
        if not self.has_layers:
            print("  Flat model (no layer structure) — single_op only")
            self._print_summary()
            return
        self._detect_op_types()
        self._discover_attn_mask_chain()
        self._discover_position_ids_chain()
        self._find_orphaned_constants()
        self._compute_deleted_outputs()
        self._build_rewire_map()
        self._discover_embed_tokens()
        self._discover_lm_head()
        self._discover_final_norm()
        self._discover_post_layer_nodes()
        self._discover_pre_layer_nodes()
        self._print_summary()

    # ── Layer detection ──

    def _detect_layers(self):
        layer_counts = Counter()
        for node in self.graph.node:
            m = self.LAYER_PATTERN.search(node.name)
            if m:
                layer_counts[int(m.group(1))] += 1
        if not layer_counts:
            print("  No layer pattern found in node names")
            return

        typical_count = Counter(layer_counts.values()).most_common(1)[0][0]
        threshold = max(typical_count * 0.4, 5)
        real_layers = sorted(
            idx for idx, cnt in layer_counts.items() if cnt >= threshold
        )
        self.num_layers = len(real_layers)
        self.layer_indices = set(real_layers)

        layer_groups = defaultdict(list)
        for idx in real_layers:
            layer_groups[layer_counts[idx]].append(idx)
        desc = ", ".join(
            f"{len(idxs)}x{cnt}-node" for cnt, idxs in sorted(layer_groups.items())
        )
        print(
            f"  Layers: {self.num_layers} total "
            f"(index {real_layers[0]}..{real_layers[-1]}, {desc})"
        )

    # ── Op type detection ──

    def _detect_op_types(self):
        op_types = {n.op_type for n in self.graph.node}
        self.shape_diff_ops = set()

        if "MatMulNBits" in op_types:
            self.shape_diff_ops.add("MatMulNBits")
        else:
            for node in self.graph.node:
                if (
                    node.op_type == "MatMul"
                    and self.get_layer_index(node.name) is not None
                ):
                    if any(inp in self.init_names for inp in node.input):
                        self.shape_diff_ops.add("MatMul")
                        break

        if "RotaryEmbedding" in op_types:
            self.shape_diff_ops.add("RotaryEmbedding")

        print(f"  Shape-diff ops: {self.shape_diff_ops or 'none'}")

    # ── Chain discovery ──

    def _follow_chain(self, start_tensor, expected_ops):
        """Follow a chain of ops from a tensor. Returns list of matched nodes."""
        chain = []
        current_tensor = start_tensor
        for op in expected_ops:
            consumers = [
                n
                for n in self.input_to_nodes.get(current_tensor, [])
                if n.op_type == op
            ]
            if len(consumers) != 1:
                break
            node = consumers[0]
            chain.append(node)
            current_tensor = node.output[0] if node.output else None
            if current_tensor is None:
                break
        return chain

    def _discover_attn_mask_chain(self):
        """attention_mask -> Shape -> Gather -> Cast"""
        if "attention_mask" not in self.graph_input_names:
            return
        shape_nodes = [
            n
            for n in self.input_to_nodes.get("attention_mask", [])
            if n.op_type == "Shape"
        ]
        for sn in shape_nodes:
            out = sn.output[0] if sn.output else None
            if not out:
                continue
            tail = self._follow_chain(out, ["Gather", "Cast"])
            if len(tail) == 2:
                chain = [sn] + tail
                self.attn_mask_chain_output = chain[-1].output[0]
                for n in chain:
                    self.nodes_to_delete.add(n.name)
                print(f"  Attn-mask chain: {' -> '.join(n.op_type for n in chain)}")
                return

    def _discover_position_ids_chain(self):
        """input_ids -> Shape -> Gather -> Unsqueeze -> Concat -> Reshape"""
        if "input_ids" not in self.graph_input_names:
            return
        shape_nodes = [
            n for n in self.input_to_nodes.get("input_ids", []) if n.op_type == "Shape"
        ]
        for sn in shape_nodes:
            out = sn.output[0] if sn.output else None
            if not out:
                continue
            tail = self._follow_chain(out, ["Gather", "Unsqueeze", "Concat", "Reshape"])
            if len(tail) == 4:
                chain = [sn] + tail
                self.reshape_chain_output = chain[-1].output[0]
                for n in chain:
                    self.nodes_to_delete.add(n.name)
                print(f"  Pos-ids chain:   {' -> '.join(n.op_type for n in chain)}")
                return
        print("  Pos-ids chain:   (not present)")

    def _find_orphaned_constants(self):
        """Remove Constant/initializer nodes used only by deleted nodes."""
        if not self.nodes_to_delete:
            return
        deleted_inputs = set()
        for node in self.graph.node:
            if node.name in self.nodes_to_delete:
                for inp in node.input:
                    if inp:
                        deleted_inputs.add(inp)
        for name in deleted_inputs:
            if name in self.graph_input_names:
                continue
            consumers = self.input_to_nodes.get(name, [])
            non_deleted = [n for n in consumers if n.name not in self.nodes_to_delete]
            if not non_deleted and name in self.output_to_node:
                producer = self.output_to_node[name]
                if producer.op_type == "Constant":
                    self.nodes_to_delete.add(producer.name)
                    print(f"  Orphaned const:  {producer.name}")

    def _compute_deleted_outputs(self):
        for node in self.graph.node:
            if node.name in self.nodes_to_delete:
                for out in node.output:
                    if out:
                        self.deleted_outputs.add(out)

    def _build_rewire_map(self):
        if self.attn_mask_chain_output:
            self.rewire_map[self.attn_mask_chain_output] = TOTAL_SEQ_LEN_CONST_NAME
        if self.reshape_chain_output:
            self.rewire_map[self.reshape_chain_output] = "position_ids"

    # ── Key node discovery ──

    def _discover_embed_tokens(self):
        if "input_ids" not in self.graph_input_names:
            return
        for node in self.input_to_nodes.get("input_ids", []):
            if node.op_type == "Gather" and node.name not in self.nodes_to_delete:
                self.embed_node = node
                print(f"  Embed tokens:    {node.name}")
                return

    def _discover_lm_head(self):
        if "logits" in self.output_to_node:
            self.lm_head_node = self.output_to_node["logits"]
            print(
                f"  LM head:         {self.lm_head_node.name} "
                f"({self.lm_head_node.op_type})"
            )
            return
        for node in reversed(list(self.graph.node)):
            if node.op_type in ("MatMul", "MatMulNBits"):
                idx = self.get_layer_index(node.name)
                if idx is None or idx not in self.layer_indices:
                    self.lm_head_node = node
                    print(f"  LM head:         {node.name} ({node.op_type})")
                    return

    def _discover_final_norm(self):
        """Trace backward from lm_head to find the final normalization."""
        if self.lm_head_node is None:
            return
        norm_ops = {
            "SkipSimplifiedLayerNormalization",
            "SimplifiedLayerNormalization",
            "LayerNormalization",
        }
        for inp in self.lm_head_node.input:
            if inp in self.init_names:
                continue
            if inp not in self.output_to_node:
                continue
            p = self.output_to_node[inp]
            if p.op_type in norm_ops:
                self.final_norm_node = p
                print(f"  Final norm:      {p.name} ({p.op_type})")
                return
            for inp2 in p.input:
                if inp2 in self.output_to_node:
                    p2 = self.output_to_node[inp2]
                    if p2.op_type in norm_ops:
                        self.final_norm_node = p2
                        print(f"  Final norm:      {p2.name} ({p2.op_type})")
                        return

    # ── Pre-layer backward traversal ──

    def _discover_pre_layer_nodes(self):
        """Find all non-layer nodes required by layer 0 via backward BFS."""
        layer0_names = set()
        layer0_inputs = set()
        for node in self.graph.node:
            if self.get_layer_index(node.name) == 0:
                layer0_names.add(node.name)
                for inp in node.input:
                    if inp:
                        layer0_inputs.add(inp)

        visited = set()
        found_names = set()
        queue = list(layer0_inputs)

        while queue:
            t = queue.pop()
            if t in visited:
                continue
            visited.add(t)
            if t in self.graph_input_names or t in self.init_names:
                continue
            if t in self.deleted_outputs:
                continue
            if t not in self.output_to_node:
                continue
            producer = self.output_to_node[t]
            if producer.name in layer0_names or producer.name in self.nodes_to_delete:
                continue
            idx = self.get_layer_index(producer.name)
            if idx is not None and idx > 0:
                continue
            if producer.name in found_names:
                continue
            found_names.add(producer.name)
            for inp in producer.input:
                if inp:
                    queue.append(inp)

        self.pre_layer_nodes = [n for n in self.graph.node if n.name in found_names]
        print(f"  Pre-layer nodes: {len(self.pre_layer_nodes)}")
        for n in self.pre_layer_nodes:
            print(f"    {n.op_type:35s} {n.name}")

    # ── Post-layer node discovery ──

    def _discover_post_layer_nodes(self):
        """BFS backward from lm_head to find all nodes between
        the last real layer and the graph outputs (final_norm layer + lm_head)."""
        if not self.lm_head_node:
            self.post_layer_nodes = []
            return

        visited = set()
        found = []
        queue = [self.lm_head_node]

        while queue:
            node = queue.pop(0)
            if node.name in visited:
                continue
            visited.add(node.name)
            if node.name in self.nodes_to_delete:
                continue

            idx = self.get_layer_index(node.name)
            if idx is not None and idx in self.layer_indices:
                continue

            found.append(node)
            for inp in node.input:
                if inp and inp in self.output_to_node:
                    producer = self.output_to_node[inp]
                    if producer.name not in visited:
                        queue.append(producer)

        node_order = {n.name: i for i, n in enumerate(self.graph.node)}
        found.sort(key=lambda n: node_order.get(n.name, 0))
        self.post_layer_nodes = found
        print(f"  Post-layer nodes: {len(found)}")
        for n in found:
            print(f"    {n.op_type:35s} {n.name}")

    # ── Final-norm rewiring for single_layer ──

    def compute_final_norm_rewire(self):
        """Map post-layer inputs from last-layer tensors to layer-0 equivalents."""
        rewire = {}
        for node in self.post_layer_nodes:
            for inp in node.input:
                if not inp or inp in self.init_names:
                    continue
                if inp in rewire:
                    continue
                m = re.search(r"(layers?[./])(\d+)([/.])", inp)
                if (
                    m
                    and int(m.group(2)) not in (0,)
                    and int(m.group(2)) in self.layer_indices
                ):
                    new_name = inp[: m.start(2)] + "0" + inp[m.end(2) :]
                    rewire[inp] = new_name
        if rewire:
            print(f"  Final-norm rewire: {len(rewire)} tensor(s)")
            for old, new in rewire.items():
                print(f"    {old}  ->  {new}")
        else:
            print("  Final-norm rewire: none needed")
        return rewire

    # ── Utilities ──

    def get_layer_index(self, node_name):
        if not node_name:
            return None
        m = self.LAYER_PATTERN.search(node_name)
        return int(m.group(1)) if m else None

    def _print_summary(self):
        print("\n  === Discovery Summary ===")
        print(f"  Layers:         {self.num_layers}")
        print(f"  Nodes to delete:{len(self.nodes_to_delete):>3d}")
        print(f"  Rewire map:     {self.rewire_map}")
        print(f"  Shape-diff ops: {self.shape_diff_ops}")
        print()


# ─────────────────────── Helper functions ───────────────────────


def get_initializer_map(graph):
    return {init.name: init for init in graph.initializer}


def get_value_info_map(graph):
    vi = {}
    for v in graph.value_info:
        vi[v.name] = v
    for v in graph.input:
        vi[v.name] = v
    for v in graph.output:
        vi[v.name] = v
    return vi


def get_tensor_shape(vi_map, name):
    if name in vi_map:
        vi = vi_map[name]
        if vi.type.tensor_type.shape:
            return [
                d.dim_param if d.dim_param else d.dim_value
                for d in vi.type.tensor_type.shape.dim
            ]
    return None


def get_tensor_elem_type(vi_map, name):
    if name in vi_map:
        return vi_map[name].type.tensor_type.elem_type
    return TensorProto.FLOAT16


def _build_dim_subs(dim_map, extra=None):
    """Build substitution dict from a variant value.
    Accepts int (seq_len), dict (dim_map), or None (dynamic).
    extra: optional dict of additional constant dims (from config)."""
    subs = {"batch_size": BATCH_SIZE}
    if extra:
        subs.update(extra)
    if dim_map is None:
        return subs
    if isinstance(dim_map, int):
        seq_len = dim_map
        subs["sequence_length"] = seq_len
        subs["total_sequence_length"] = seq_len
        subs["past_sequence_length"] = seq_len
        return subs
    subs.update(dim_map)
    return subs


def resolve_dim_param(param, dim_map, extra=None):
    """Evaluate a symbolic dim like 'batch_size * sequence_length'.
    dim_map: int (seq_len), dict, or None (keep symbolic).
    extra: optional dict of additional constant dims (from config).
    Returns int if fully resolved, or the original string if not."""
    subs = _build_dim_subs(dim_map, extra=extra)

    has_op = any(c in param for c in ("*", "+", "-", "/", "(", ")"))
    if has_op:
        expr = param
        for sym, val in sorted(subs.items(), key=lambda x: -len(x[0])):
            expr = expr.replace(sym, str(val))
        try:
            return int(eval(expr))  # noqa: S307
        except Exception:
            return param

    if param in subs:
        return subs[param]

    if "batch" in param and "seq" not in param:
        return BATCH_SIZE

    if isinstance(dim_map, int):
        return dim_map

    return param


def total_seq_len_scalar_from_dim_map(dim_map) -> Optional[int]:
    """Scalar total sequence length for ``total_seq_len_const`` / fold Constant (LLM)."""
    if dim_map is None:
        return None
    if isinstance(dim_map, dict):
        if "total_sequence_length" in dim_map:
            return int(dim_map["total_sequence_length"])
        if "sequence_length" in dim_map:
            return int(dim_map["sequence_length"])
        return None
    if isinstance(dim_map, int):
        return int(dim_map) if dim_map else None
    return None


def upsert_total_seq_len_const_initializer(
    graph: onnx.GraphProto,
    value: int,
    *,
    name: str = TOTAL_SEQ_LEN_CONST_NAME,
) -> None:
    """Replace or append int32 scalar initializer ``total_seq_len_const`` = ``value``."""
    new_init = numpy_helper.from_array(
        np.array(int(value), dtype=np.int32),
        name=name,
    )
    for i, init in enumerate(graph.initializer):
        if init.name == name:
            graph.initializer[i].CopyFrom(new_init)
            return
    graph.initializer.append(new_init)


def patch_fold_total_seq_len_constant_value(model: onnx.ModelProto, value: int) -> bool:
    """Set ``Constant fold/total_seq_len`` tensor value to ``value`` (int32) if present."""
    for n in model.graph.node:
        if n.name != FOLD_TOTAL_SEQ_NODE_NAME or n.op_type != "Constant":
            continue
        for attr in n.attribute:
            if attr.name != "value" or attr.type != onnx.AttributeProto.TENSOR:
                continue
            new_t = numpy_helper.from_array(np.array(int(value), dtype=np.int32))
            attr.t.CopyFrom(new_t)
            return True
    return False


def shape_to_dynamic_fixed(shape, dim_map, extra=None):
    if shape is None:
        return None
    result = []
    for d in shape:
        if isinstance(d, str):
            result.append(resolve_dim_param(d, dim_map, extra=extra))
        else:
            result.append(d)
    return result


def make_tensor_vi(name, elem_type, shape):
    tp = onnx.TypeProto()
    tp.tensor_type.elem_type = elem_type
    for d in shape:
        dp = tp.tensor_type.shape.dim.add()
        if isinstance(d, str):
            dp.dim_param = d
        else:
            dp.dim_value = d
    vi = onnx.ValueInfoProto()
    vi.name = name
    vi.type.CopyFrom(tp)
    return vi


def build_single_op_model(
    node, inputs, outputs, initializers, opsets, ir_version=7, value_infos=None
):
    graph = helper.make_graph(
        [node],
        "single_op_graph",
        inputs,
        outputs,
        initializer=initializers,
        value_info=value_infos or [],
    )
    model = helper.make_model(graph)
    model.ir_version = ir_version
    del model.opset_import[:]
    for oi in opsets:
        new_oi = model.opset_import.add()
        new_oi.domain = oi.domain
        new_oi.version = oi.version
    return model


def build_multi_node_model(
    nodes, inputs, outputs, initializers, opsets, ir_version=7, value_infos=None
):
    graph = helper.make_graph(
        nodes,
        "subgraph",
        inputs,
        outputs,
        initializer=initializers,
        value_info=value_infos or [],
    )
    model = helper.make_model(graph)
    model.ir_version = ir_version
    del model.opset_import[:]
    for oi in opsets:
        new_oi = model.opset_import.add()
        new_oi.domain = oi.domain
        new_oi.version = oi.version
    return model


PROTOBUF_LIMIT = 1_800_000_000  # ~1.8 GB, safely below protobuf 2GB cap

DTYPE_BYTE_SIZE = {
    TensorProto.FLOAT: 4,
    TensorProto.UINT8: 1,
    TensorProto.INT8: 1,
    TensorProto.UINT16: 2,
    TensorProto.INT16: 2,
    TensorProto.INT32: 4,
    TensorProto.INT64: 8,
    TensorProto.BOOL: 1,
    TensorProto.FLOAT16: 2,
    TensorProto.DOUBLE: 8,
    TensorProto.UINT32: 4,
    TensorProto.UINT64: 8,
    TensorProto.BFLOAT16: 2,
}


def estimate_tensor_bytes(tensor_meta):
    """Estimate byte size from dims and data_type (works without raw_data)."""
    elem = DTYPE_BYTE_SIZE.get(tensor_meta.data_type, 4)
    n = 1
    for d in tensor_meta.dims:
        n *= d
    return n * elem


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    onnx.save(model, path)
    print(f"  Saved: {path}")


# ─────────────────── Weight / input shape signatures ───────────────────


def get_weight_signature(node, init_map):
    return tuple(tuple(init_map[inp].dims) for inp in node.input if inp in init_map)


def get_input_shape_signature(node, vi_map):
    sig = []
    for inp in node.input:
        if inp and inp in vi_map:
            s = get_tensor_shape(vi_map, inp)
            if s is not None:
                sig.append(tuple(d for d in s if isinstance(d, int) and d > 0))
    return tuple(sig)


# ─────────────────────── Extractor ───────────────────────


class SingleOpExtractor:
    def __init__(self, model_path, output_dir=None):
        self.model_path = model_path
        self.base_dir = output_dir or os.path.dirname(model_path)
        self.single_op_dir = os.path.join(self.base_dir, "single_op")
        self.single_layer_dir = os.path.join(self.base_dir, "single_layer")
        self.full_model_dir = os.path.join(self.base_dir, "full_model")

        print("Loading model graph (no weights)...")
        self.graph_model = onnx.load(model_path, load_external_data=False)
        self.ir_version = self.graph_model.ir_version

        print("Running shape inference...")
        try:
            self.graph_model = onnx.shape_inference.infer_shapes(self.graph_model)
            print("Shape inference succeeded.")
        except Exception as e:
            print(f"Shape inference warning: {e}")

        self.graph = self.graph_model.graph
        self._remove_orphan_inputs()
        self.init_map = get_initializer_map(self.graph)
        self.vi_map = get_value_info_map(self.graph)
        self.opset_imports = list(self.graph_model.opset_import)

        self.constant_outputs = {}
        for node in self.graph.node:
            if node.op_type == "Constant":
                for attr in node.attribute:
                    if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                        for out in node.output:
                            if out:
                                self.constant_outputs[out] = attr.t

        self.analyzer = GraphAnalyzer(self.graph)
        self.analyzer.analyze()

        self.full_model = None
        self.ext_data_name = self._detect_ext_data_name()
        self.model_type = self._detect_model_type()
        self.config_dims = self._load_config_dims()
        self.variants = self._get_variants()

    def _remove_orphan_inputs(self):
        """Remove graph inputs not consumed by any node (and not initializers).
        Also remove orphan nodes whose outputs are not consumed by any
        downstream node or graph output."""
        consumed_tensors = set()
        for node in self.graph.node:
            for inp in node.input:
                if inp:
                    consumed_tensors.add(inp)

        init_names = {init.name for init in self.graph.initializer}
        orphan_inputs = [
            inp
            for inp in self.graph.input
            if inp.name not in consumed_tensors and inp.name not in init_names
        ]
        if orphan_inputs:
            names = [inp.name for inp in orphan_inputs]
            print(f"Removing orphan inputs: {names}")
            for inp in orphan_inputs:
                self.graph.input.remove(inp)

        output_names = {o.name for o in self.graph.output}
        changed = True
        removed_total = 0
        while changed:
            changed = False
            all_consumed = set()
            for node in self.graph.node:
                for inp in node.input:
                    if inp:
                        all_consumed.add(inp)
            all_consumed |= output_names

            dead = [
                n
                for n in self.graph.node
                if all(o not in all_consumed for o in n.output if o)
            ]
            if dead:
                for n in dead:
                    self.graph.node.remove(n)
                removed_total += len(dead)
                changed = True

        if removed_total:
            print(f"Removing orphan nodes:  {removed_total}")

    def _detect_ext_data_name(self):
        """Detect the external data filename from the input model's
        initializers so we can reuse it in the output."""
        for init in self.graph.initializer:
            if init.data_location == TensorProto.EXTERNAL:
                for entry in init.external_data:
                    if entry.key == "location":
                        print(f"External data:     {entry.value}")
                        return entry.value
        default = os.path.basename(self.model_path) + ".data"
        print(f"External data:     {default} (default)")
        return default

    def _detect_model_type(self):
        input_names = {inp.name for inp in self.graph.input}
        if "pixel_values" in input_names:
            print("Model type: vision")
            return "vision"
        if "image_features" in input_names and "input_ids" in input_names:
            print("Model type: embedding")
            return "embedding"
        return "llm"

    _CONFIG_DIM_KEYS = [
        "num_attention_heads",
        "num_key_value_heads",
        "head_size",
        "hidden_size",
    ]

    def _load_config_dims(self):
        """Read genai_config.json next to the model and extract dimension
        constants (num_attention_heads, num_key_value_heads, etc.)."""
        model_dir = os.path.dirname(self.model_path)
        cfg_path = os.path.join(model_dir, "genai_config.json")
        if not os.path.isfile(cfg_path):
            return {}
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            return {}

        decoder = cfg.get("model", {}).get("decoder", {})
        dims = {}
        for key in self._CONFIG_DIM_KEYS:
            val = decoder.get(key)
            if isinstance(val, int):
                dims[key] = val
        if dims:
            print(f"Config dims:       {dims}")
        return dims

    def _get_variants(self):
        if self.model_type == "vision":
            return VISION_VARIANTS
        if self.model_type == "embedding":
            return EMBEDDING_VARIANTS
        return SEQ_VARIANTS

    # ── Full model loading ──

    def _load_full_model(self):
        if self.full_model is not None:
            return
        model_dir = os.path.dirname(self.model_path)
        total = os.path.getsize(self.model_path)
        for f in os.listdir(model_dir):
            if f.endswith((".data", ".bin")) and f != os.path.basename(self.model_path):
                total += os.path.getsize(os.path.join(model_dir, f))
        print(f"Loading full model with weights (~{total / (1024**3):.1f} GB)...")
        self.full_model = onnx.load(self.model_path)
        print("Full model loaded.")
        saved_graph = self.graph
        self.graph = self.full_model.graph
        self._remove_orphan_inputs()
        self.graph = saved_graph
        self.full_init_map = get_initializer_map(self.full_model.graph)

    def _get_initializer(self, name):
        self._load_full_model()
        return self.full_init_map.get(name)

    def _resolve_input_info(self, name):
        if not name:
            return None, None
        if name in self.init_map:
            init = self.init_map[name]
            return list(init.dims), init.data_type
        shape = get_tensor_shape(self.vi_map, name)
        elem_type = get_tensor_elem_type(self.vi_map, name)
        return shape, elem_type

    # ════════════════════ single_op ════════════════════

    def _pick_fp16_representative(self, candidates):
        """Among candidates for the same unique key, prefer fp16 inputs."""
        if len(candidates) == 1:
            return candidates[0]
        best = candidates[0]
        best_score = -1
        for node in candidates:
            fp16 = 0
            total = 0
            for inp in node.input:
                vi = self.vi_map.get(inp)
                if vi and vi.type.tensor_type.elem_type:
                    total += 1
                    if vi.type.tensor_type.elem_type == TensorProto.FLOAT16:
                        fp16 += 1
            score = fp16 / max(total, 1)
            if score > best_score:
                best_score = score
                best = node
        return best

    def _identify_unique_ops(self):
        a = self.analyzer
        layer_ops = defaultdict(list)
        non_layer_nodes = []

        flat_mode = not a.has_layers

        for node in self.graph.node:
            if node.name in a.nodes_to_delete or node.op_type == "Constant":
                continue

            if flat_mode:
                if node.op_type in PERF_SIGNIFICANT_OPS:
                    wsig = get_weight_signature(node, self.init_map)
                    isig = get_input_shape_signature(node, self.vi_map)
                    nout = len([o for o in node.output if o])
                    key = (node.op_type, wsig, isig, nout)
                else:
                    key = (node.op_type,)
                layer_ops[key].append(node)
                continue

            idx = a.get_layer_index(node.name)
            if idx is not None and idx in a.layer_indices:
                if node.op_type in a.shape_diff_ops:
                    wsig = get_weight_signature(node, self.init_map)
                    isig = get_input_shape_signature(node, self.vi_map)
                    nout = len([o for o in node.output if o])
                    key = (node.op_type, wsig, isig, nout)
                else:
                    key = (node.op_type,)
                layer_ops[key].append(node)
            else:
                non_layer_nodes.append(node)

        unique = {k: self._pick_fp16_representative(vs) for k, vs in layer_ops.items()}

        mode_label = "Unique ops (flat)" if flat_mode else "Unique layer ops"
        print(f"\n{mode_label}: {len(unique)}")
        for k, n in unique.items():
            print(f"  {k[0]:30s} {n.name}")
        if non_layer_nodes:
            print(f"Non-layer nodes:  {len(non_layer_nodes)}")
            for n in non_layer_nodes:
                print(f"  {n.op_type:30s} {n.name}")
        return unique, non_layer_nodes

    def _get_name_prefix(self, node, key=None):
        """Generate semantic name for the op folder."""
        op_type = node.op_type
        need_suffix = (
            op_type in self.analyzer.shape_diff_ops or not self.analyzer.has_layers
        )
        if not need_suffix:
            return op_type
        if not node.name:
            return op_type

        clean = re.sub(r"^/?model/", "", node.name)
        clean = re.sub(r"^/", "", clean)
        clean = re.sub(r"(layers?|blocks?)\.\d+/", "", clean)
        clean = re.sub(r"^node_", "", clean)
        parts = clean.split("/")

        trail = parts[-1] if parts else ""
        op_base = op_type.replace("NBits", "")
        if trail.startswith(op_base) or trail == op_type:
            parts = parts[:-1]

        if self.analyzer.get_layer_index(node.name) is not None:
            parts = [p for p in parts if p not in ("attn", "mlp", "self_attn")]

        if parts and parts[-1]:
            suffix = re.sub(r"[^a-zA-Z0-9_]", "_", parts[-1])
            return f"{op_type}_{suffix}"
        return op_type

    def _extract_single_op(self, node, name_prefix, is_gqa=False):
        op_dir = os.path.join(self.single_op_dir, name_prefix)

        clean_node = onnx.NodeProto()
        clean_node.CopyFrom(node)

        for i, inp in enumerate(clean_node.input):
            if inp in self.analyzer.rewire_map:
                clean_node.input[i] = self.analyzer.rewire_map[inp]

        init_names = set()
        const_names = set()
        large_init_names = set()
        for inp in clean_node.input:
            if not inp:
                continue
            if inp in self.init_map:
                init_names.add(inp)
                if estimate_tensor_bytes(self.init_map[inp]) > PROTOBUF_LIMIT:
                    large_init_names.add(inp)
            elif inp in self.constant_outputs:
                const_names.add(inp)

        output_names = [o for o in node.output if o]

        for variant_name, dim_map in self.variants:
            model_inputs = []
            initializers = []
            seen_inputs = set()

            for inp in clean_node.input:
                if not inp:
                    continue
                if inp in seen_inputs:
                    continue
                seen_inputs.add(inp)
                if inp in init_names:
                    if inp in large_init_names:
                        continue
                    it = self._get_initializer(inp)
                    if it:
                        initializers.append(it)
                    continue
                if inp in const_names:
                    t = self.constant_outputs[inp]
                    it = TensorProto()
                    it.CopyFrom(t)
                    it.name = inp
                    initializers.append(it)
                    continue
                if is_gqa and self._is_gqa_constant_input(clean_node, inp):
                    continue

                shape, elem_type = self._resolve_input_info(inp)
                if shape is None:
                    shape, elem_type = self._resolve_input_info(
                        node.input[list(clean_node.input).index(inp)]
                    )
                if shape is None:
                    shape = [1]
                adj = shape_to_dynamic_fixed(shape, dim_map, extra=self.config_dims)
                model_inputs.append(make_tensor_vi(inp, elem_type, adj))

            model_outputs = []
            for out in output_names:
                _, elem_type = self._resolve_input_info(out)
                model_outputs.append(make_tensor_vi(out, elem_type or 1, []))

            actual_node = onnx.NodeProto()
            actual_node.CopyFrom(clean_node)

            if is_gqa:
                initializers.extend(
                    self._make_gqa_constant_inputs(actual_node, dim_map)
                )

            model = build_single_op_model(
                actual_node,
                model_inputs,
                model_outputs,
                initializers,
                self.opset_imports,
                ir_version=self.ir_version,
            )

            try:
                model = onnx.shape_inference.infer_shapes(model)
            except Exception:
                pass

            for out_vi in model.graph.output:
                if not out_vi.type.tensor_type.shape.dim:
                    shape, etype = self._resolve_input_info(out_vi.name)
                    if shape:
                        adj = shape_to_dynamic_fixed(
                            shape, dim_map, extra=self.config_dims
                        )
                        for d in adj:
                            dp = out_vi.type.tensor_type.shape.dim.add()
                            if isinstance(d, str):
                                dp.dim_param = d
                            else:
                                dp.dim_value = d

            fname = f"{name_prefix}_{variant_name}.onnx"
            fpath = os.path.join(op_dir, fname)
            os.makedirs(op_dir, exist_ok=True)

            if large_init_names:
                self._save_with_external_weights(model, fpath, large_init_names)
            else:
                save_model(model, fpath)

    def _save_with_external_weights(self, model, path, large_weight_names):
        """Save model with oversized weights in a separate .data file."""
        data_file = os.path.basename(path).replace(".onnx", ".data")
        data_path = os.path.join(os.path.dirname(path), data_file)

        with open(data_path, "wb") as f:
            for wname in large_weight_names:
                tensor = self._get_initializer(wname)
                if tensor is None:
                    continue
                offset = f.tell()
                raw = tensor.raw_data
                if not raw:
                    raw = numpy_helper.to_array(tensor).tobytes()
                f.write(raw)
                length = len(raw)

                ext_init = TensorProto()
                ext_init.name = wname
                ext_init.data_type = tensor.data_type
                ext_init.dims.extend(tensor.dims)
                ext_init.data_location = TensorProto.EXTERNAL
                for k, v in [
                    ("location", data_file),
                    ("offset", str(offset)),
                    ("length", str(length)),
                ]:
                    kv = ext_init.external_data.add()
                    kv.key = k
                    kv.value = v
                model.graph.initializer.append(ext_init)

        with open(path, "wb") as f:
            f.write(model.SerializeToString())
        print(f"  Saved: {path} (+{data_file})")

    @staticmethod
    def _is_gqa_constant_input(node, input_name):
        inputs = list(node.input)
        idx = inputs.index(input_name) if input_name in inputs else -1
        return idx in (5, 6)

    @staticmethod
    def _make_gqa_constant_inputs(node, dim_map):
        inputs = list(node.input)
        inits = []
        eff = dim_map if isinstance(dim_map, int) and dim_map else 128

        if len(inputs) > 5 and inputs[5]:
            node.input[5] = "seqlens_k"
            inits.append(
                numpy_helper.from_array(
                    np.array([eff - 1], dtype=np.int32), name="seqlens_k"
                )
            )
        if len(inputs) > 6 and inputs[6]:
            node.input[6] = "total_seq_len"
            inits.append(
                numpy_helper.from_array(
                    np.array(eff, dtype=np.int32), name="total_seq_len"
                )
            )
        return inits

    def _unique_prefix(self, base, used):
        """Return a unique folder name, appending _2, _3, ... if needed."""
        if base not in used:
            return base
        n = 2
        while f"{base}_{n}" in used:
            n += 1
        return f"{base}_{n}"

    def _dedup_non_layer(self, non_layer_nodes):
        """Deduplicate non-layer nodes the same way layer ops are deduped:
        simple ops keep one per op_type, perf-significant ops keep one per
        full signature (weight shape + input shape)."""
        groups = defaultdict(list)
        for node in non_layer_nodes:
            if node.op_type in PERF_SIGNIFICANT_OPS:
                wsig = get_weight_signature(node, self.init_map)
                isig = get_input_shape_signature(node, self.vi_map)
                nout = len([o for o in node.output if o])
                key = (node.op_type, wsig, isig, nout)
            else:
                key = (node.op_type,)
            groups[key].append(node)
        return [self._pick_fp16_representative(vs) for vs in groups.values()]

    def extract_all_single_ops(self):
        unique, non_layer = self._identify_unique_ops()

        print("\n" + "=" * 60)
        label = (
            "Extracting ops..."
            if not self.analyzer.has_layers
            else "Extracting layer ops..."
        )
        print(label)
        print("=" * 60)
        used = set()
        for key, node in unique.items():
            prefix = self._get_name_prefix(node, key)
            prefix = self._unique_prefix(prefix, used)
            used.add(prefix)
            is_gqa = node.op_type == "GroupQueryAttention"
            print(f"\nExtracting {prefix}...")
            self._extract_single_op(node, prefix, is_gqa=is_gqa)

        if non_layer:
            deduped = self._dedup_non_layer(non_layer)
            print("\n" + "=" * 60)
            print(
                f"Extracting non-layer ops "
                f"({len(non_layer)} -> {len(deduped)} after dedup)..."
            )
            print("=" * 60)
            for node in deduped:
                prefix = self._get_name_prefix(node)
                prefix = self._unique_prefix(prefix, used)
                used.add(prefix)
                print(f"\nExtracting {prefix}...")
                self._extract_single_op(node, prefix, is_gqa=False)

    # ════════════════════ single_layer ════════════════════

    def extract_single_layer(self):
        print("\n" + "=" * 60)
        if not self.analyzer.has_layers:
            if self.model_type == "vision":
                self._extract_vision_single_block()
            else:
                print("Skipping single_layer: no layer structure detected")
                print("=" * 60)
            return
        print("Extracting single layer (embed + layer 0 + final_norm + lm_head)...")
        print("=" * 60)

        a = self.analyzer

        layer0_nodes = [n for n in self.graph.node if a.get_layer_index(n.name) == 0]

        post_nodes = list(a.post_layer_nodes)

        all_nodes = list(a.pre_layer_nodes) + layer0_nodes + post_nodes
        print(
            f"Total nodes: {len(all_nodes)} "
            f"(pre={len(a.pre_layer_nodes)}, "
            f"layer0={len(layer0_nodes)}, "
            f"post={len(post_nodes)})"
        )

        input_rewire = dict(a.rewire_map)
        fnr = a.compute_final_norm_rewire()
        input_rewire.update(fnr)

        all_outputs = set()
        all_inputs_ordered = []
        for node in all_nodes:
            for o in node.output:
                if o:
                    all_outputs.add(o)
            for i in node.input:
                if i:
                    all_inputs_ordered.append(input_rewire.get(i, i))

        init_names = set()
        const_names = set()
        graph_input_names = []
        for inp in all_inputs_ordered:
            if inp in all_outputs:
                continue
            if inp in self.init_map:
                init_names.add(inp)
            elif inp in self.constant_outputs:
                const_names.add(inp)
            elif inp == TOTAL_SEQ_LEN_CONST_NAME:
                continue
            elif inp not in graph_input_names:
                graph_input_names.append(inp)

        graph_output_names = self._detect_single_layer_outputs(all_nodes, all_outputs)

        print(f"Inputs:  {graph_input_names}")
        print(f"Outputs: {graph_output_names}")

        nodes_copy = []
        for n in all_nodes:
            nc = onnx.NodeProto()
            nc.CopyFrom(n)
            for i, inp in enumerate(nc.input):
                if inp in input_rewire:
                    nc.input[i] = input_rewire[inp]
            nodes_copy.append(nc)

        small_inits = []
        large_inits = []
        for iname in init_names:
            it = self._get_initializer(iname)
            if it:
                if estimate_tensor_bytes(self.init_map[iname]) > PROTOBUF_LIMIT:
                    large_inits.append(it)
                else:
                    small_inits.append(it)
        for cname in const_names:
            t = self.constant_outputs[cname]
            it = TensorProto()
            it.CopyFrom(t)
            it.name = cname
            small_inits.append(it)

        small_inits.append(
            numpy_helper.from_array(
                np.array(128, dtype=np.int32), name=TOTAL_SEQ_LEN_CONST_NAME
            )
        )

        model_inputs = []
        for inp in graph_input_names:
            shape, elem_type = self._resolve_input_info(inp)
            if shape is None:
                shape = [1]
            model_inputs.append(
                make_tensor_vi(
                    inp,
                    elem_type,
                    shape_to_dynamic_fixed(shape, None, extra=self.config_dims),
                )
            )

        model_outputs = []
        for out in graph_output_names:
            shape, elem_type = self._resolve_input_info(out)
            if shape is None:
                shape = [1]
            model_outputs.append(
                make_tensor_vi(
                    out,
                    elem_type,
                    shape_to_dynamic_fixed(shape, None, extra=self.config_dims),
                )
            )

        out_set = set(graph_output_names)
        in_set = set(graph_input_names)
        value_infos = []
        for node in all_nodes:
            for o in node.output:
                if not o or o in out_set or o in in_set or o not in all_outputs:
                    continue
                shape, elem_type = self._resolve_input_info(o)
                if shape:
                    value_infos.append(
                        make_tensor_vi(
                            o,
                            elem_type,
                            shape_to_dynamic_fixed(shape, None, extra=self.config_dims),
                        )
                    )

        model = build_multi_node_model(
            nodes_copy,
            model_inputs,
            model_outputs,
            small_inits,
            self.opset_imports,
            ir_version=self.ir_version,
            value_infos=value_infos,
        )

        if large_inits:
            print(
                f"  {len(large_inits)} oversized initializer(s), "
                f"adding via direct assignment"
            )
            for init in large_inits:
                new_init = model.graph.initializer.add()
                new_init.name = init.name
                new_init.data_type = init.data_type
                new_init.dims.extend(init.dims)
                new_init.raw_data = init.raw_data

        self._save_variants(model, self.single_layer_dir, "single_layer")

    def _detect_single_layer_outputs(self, all_nodes, all_outputs):
        """Auto-detect outputs: logits + all layer-0 graph outputs (present states)."""
        outputs = []
        a = self.analyzer
        graph_out_names = {o.name for o in self.graph.output}
        if a.lm_head_node:
            for out in a.lm_head_node.output:
                if out and out in all_outputs:
                    outputs.append(out)
        for node in all_nodes:
            if a.get_layer_index(node.name) == 0:
                for out in node.output:
                    if out and out in graph_out_names and out not in outputs:
                        outputs.append(out)
        if not outputs:
            fallback = [o.name for o in self.graph.output if o.name in all_outputs]
            outputs = fallback[:5]
        return outputs

    # ════════════════════ vision single block ════════════════════

    def _find_vision_blocks(self):
        """Find repeating ViT block boundaries via FFN residual pattern:
        LayerNorm -> Gemm -> Gelu -> Gemm -> Add."""
        nodes = self.graph.node
        block_ends = []
        for i in range(len(nodes) - 4):
            if (
                nodes[i].op_type == "LayerNormalization"
                and nodes[i + 1].op_type == "Gemm"
                and nodes[i + 2].op_type == "Gelu"
                and nodes[i + 3].op_type == "Gemm"
                and nodes[i + 4].op_type == "Add"
            ):
                block_ends.append(i + 4)
        if len(block_ends) < 2:
            return None

        gap = block_ends[1] - block_ends[0]
        first_start = None
        for i in range(max(0, block_ends[0] - gap - 10), block_ends[0]):
            if nodes[i].op_type == "LayerNormalization":
                first_start = i
                break
        if first_start is None:
            return None

        blocks = [(first_start, block_ends[0])]
        for k in range(1, len(block_ends)):
            blocks.append((block_ends[k - 1] + 1, block_ends[k]))
        return blocks

    def _collect_cast_deps(self, core_indices, forbidden_inputs=None):
        """Given a set of node indices, expand to include Cast nodes that
        the core nodes depend on.  Skip Cast nodes whose inputs come from
        forbidden_inputs (e.g. outputs of removed middle blocks)."""
        if forbidden_inputs is None:
            forbidden_inputs = frozenset()
        nodes = self.graph.node
        output_to_idx = {}
        for i, n in enumerate(nodes):
            for o in n.output:
                if o:
                    output_to_idx[o] = i

        init_names = set(self.init_map.keys())
        graph_in = {inp.name for inp in self.graph.input}
        const_names = set(self.constant_outputs.keys())

        required = set(core_indices)
        queue = list(core_indices)
        while queue:
            idx = queue.pop(0)
            n = nodes[idx]
            for inp in n.input:
                if (
                    not inp
                    or inp in init_names
                    or inp in graph_in
                    or inp in const_names
                ):
                    continue
                src = output_to_idx.get(inp)
                if src is not None and src not in required:
                    if nodes[src].op_type == "Cast":
                        cast_inputs = {ci for ci in nodes[src].input if ci}
                        if cast_inputs & forbidden_inputs:
                            continue
                        required.add(src)
                        queue.append(src)
        return sorted(required)

    def _extract_vision_single_block(self):
        """Extract pre-block + one ViT block + post-block as single_layer
        (equivalent to the full model with only 1 iteration of the
        repeated blocks)."""
        blocks = self._find_vision_blocks()
        if not blocks:
            print("Skipping single_layer: no repeating ViT blocks found")
            print("=" * 60)
            return

        n_blocks = len(blocks)
        b0_start, b0_end = blocks[0]
        last_end = blocks[-1][1]
        print(
            f"Extracting vision single block ({n_blocks} blocks detected, "
            f"block 0: nodes [{b0_start}..{b0_end}])..."
        )
        print("=" * 60)

        nodes = self.graph.node

        pre_block = list(range(0, b0_start))
        block0_indices = list(range(b0_start, b0_end + 1))

        post_block = [
            i for i in range(last_end + 1, len(nodes)) if nodes[i].op_type != "Cast"
        ]

        middle_outputs = set()
        for k in range(1, n_blocks - 1):
            for i in range(blocks[k][0], blocks[k][1] + 1):
                for o in nodes[i].output:
                    if o:
                        middle_outputs.add(o)

        core = pre_block + block0_indices + post_block
        all_indices = self._collect_cast_deps(core, forbidden_inputs=middle_outputs)

        block0_output_names = set()
        for i in block0_indices:
            for o in nodes[i].output:
                if o:
                    block0_output_names.add(o)

        last_block_outputs = set()
        for i in range(blocks[-1][0], blocks[-1][1] + 1):
            for o in nodes[i].output:
                if o:
                    last_block_outputs.add(o)

        b0_main_out = None
        for o in nodes[b0_end].output:
            if o:
                b0_main_out = o
                break

        input_rewire = {}
        for idx in all_indices:
            n = nodes[idx]
            for inp_name in n.input:
                if (
                    inp_name
                    and inp_name in last_block_outputs
                    and inp_name not in block0_output_names
                ):
                    input_rewire[inp_name] = b0_main_out

        self._load_full_model()
        fm_nodes = list(self.full_model.graph.node)
        fm_init_map = {init.name: init for init in self.full_model.graph.initializer}

        all_nodes_list = [fm_nodes[i] for i in all_indices if i < len(fm_nodes)]

        all_outputs_set = set()
        for n in all_nodes_list:
            for o in n.output:
                if o:
                    all_outputs_set.add(o)

        init_names = set()
        const_names = set()
        graph_input_names = []
        for n in all_nodes_list:
            for inp in n.input:
                actual = input_rewire.get(inp, inp)
                if not actual or actual in all_outputs_set:
                    continue
                if actual in fm_init_map:
                    init_names.add(actual)
                elif actual in self.constant_outputs:
                    const_names.add(actual)
                elif actual not in graph_input_names:
                    graph_input_names.append(actual)

        graph_output_names = [o.name for o in self.graph.output]

        n_cast = (
            len(all_indices) - len(pre_block) - len(block0_indices) - len(post_block)
        )
        print(
            f"  Pre-block: {len(pre_block)}, block 0: {len(block0_indices)}, "
            f"post-block: {len(post_block)}, Cast deps: {n_cast}"
        )
        print(f"  Inputs:  {graph_input_names}")
        print(f"  Outputs: {graph_output_names}")

        nodes_copy = []
        for n in all_nodes_list:
            nc = onnx.NodeProto()
            nc.CopyFrom(n)
            for i, inp in enumerate(nc.input):
                if inp in input_rewire:
                    nc.input[i] = input_rewire[inp]
            nodes_copy.append(nc)

        model_inputs = []
        for inp_name in graph_input_names:
            shape, elem_type = self._resolve_input_info(inp_name)
            if shape is None:
                shape = [1]
            adj = shape_to_dynamic_fixed(shape, None, extra=self.config_dims)
            model_inputs.append(make_tensor_vi(inp_name, elem_type, adj))

        model_outputs = []
        for out_name in graph_output_names:
            shape, elem_type = self._resolve_input_info(out_name)
            if shape is None:
                shape = [1]
            adj = shape_to_dynamic_fixed(shape, None, extra=self.config_dims)
            model_outputs.append(make_tensor_vi(out_name, elem_type, adj))

        small_inits = []
        large_inits = []
        for name in init_names:
            it = fm_init_map.get(name)
            if it is None:
                continue
            if estimate_tensor_bytes(it) > PROTOBUF_LIMIT:
                large_inits.append(it)
            else:
                small_inits.append(it)

        for name in const_names:
            t = self.constant_outputs[name]
            it = TensorProto()
            it.CopyFrom(t)
            it.name = name
            small_inits.append(it)

        value_infos = []
        out_set = set(graph_output_names)
        in_set = set(graph_input_names)
        for n in all_nodes_list:
            for o in n.output:
                if not o or o in out_set or o in in_set:
                    continue
                shape, elem_type = self._resolve_input_info(o)
                if shape:
                    value_infos.append(
                        make_tensor_vi(
                            o,
                            elem_type,
                            shape_to_dynamic_fixed(shape, None, extra=self.config_dims),
                        )
                    )

        model = build_multi_node_model(
            nodes_copy,
            model_inputs,
            model_outputs,
            small_inits,
            self.opset_imports,
            ir_version=self.ir_version,
            value_infos=value_infos,
        )

        if large_inits:
            print(f"  {len(large_inits)} oversized initializer(s)")
            for init in large_inits:
                new_init = model.graph.initializer.add()
                new_init.name = init.name
                new_init.data_type = init.data_type
                new_init.dims.extend(init.dims)
                new_init.raw_data = init.raw_data

        self._save_variants(model, self.single_layer_dir, "single_layer")

    # ════════════════════ full_model ════════════════════

    def extract_full_model(self):
        print("\n" + "=" * 60)
        if not self.analyzer.has_layers:
            print("Extracting fixed-shape variants (flat model)...")
            print("=" * 60)
            self._load_full_model()
            self._save_variants(self.full_model, self.full_model_dir, "full_model")
            return
        print("Extracting full model...")
        print("=" * 60)

        model = onnx.load(self.model_path, load_external_data=False)
        saved_graph = self.graph
        self.graph = model.graph
        self._remove_orphan_inputs()
        self.graph = saved_graph
        a = self.analyzer

        deleted_tensors = set()
        for node in model.graph.node:
            if node.name in a.nodes_to_delete:
                for o in node.output:
                    if o:
                        deleted_tensors.add(o)

        keep = [n for n in model.graph.node if n.name not in a.nodes_to_delete]
        del model.graph.node[:]
        model.graph.node.extend(keep)

        for node in model.graph.node:
            for i, inp in enumerate(node.input):
                if inp in a.rewire_map:
                    node.input[i] = a.rewire_map[inp]

        vis = [vi for vi in model.graph.value_info if vi.name not in deleted_tensors]
        del model.graph.value_info[:]
        model.graph.value_info.extend(vis)

        used_inputs = set()
        for node in model.graph.node:
            for inp in node.input:
                if inp:
                    used_inputs.add(inp)
        orphaned_idx = [
            i
            for i, init in enumerate(model.graph.initializer)
            if init.name not in used_inputs
        ]
        if orphaned_idx:
            print(f"  Removing {len(orphaned_idx)} orphaned initializer(s)")
            for idx in reversed(orphaned_idx):
                del model.graph.initializer[idx]

        self._save_full_model_variants(model)

    # ════════════════════ variant saving ════════════════════

    _OPT_PASSES = [
        "eliminate_shape_gather",
        "eliminate_shape_op",
        "extract_constant_to_initializer",
        "eliminate_deadend",
        "eliminate_unused_initializer",
    ]

    def _full_model_variant_onnx_basename(self, vname: str) -> str:
        """Basename under ``full_model/`` for the layered-graph export path."""
        return f"full_model_{vname}.onnx"

    def _save_full_model_variants(self, model):
        """Save full_model variants reusing the original external data file.

        The model must be loaded with load_external_data=False so that
        initializer tensors still carry their original external_data
        references (offset, length, location).  We copy the original
        .data file once and only serialise the lightweight .onnx protos.
        """
        out_dir = self.full_model_dir
        os.makedirs(out_dir, exist_ok=True)
        data_name = self.ext_data_name

        src_data = os.path.join(os.path.dirname(self.model_path), data_name)
        dst_data = os.path.join(out_dir, data_name)
        if os.path.isfile(src_data):
            if os.path.abspath(src_data) != os.path.abspath(dst_data):
                print(
                    f"  Copying external data: {data_name} "
                    f"({os.path.getsize(src_data) / (1024**3):.2f} GB)"
                )
                shutil.copy2(src_data, dst_data)
            else:
                print(f"  External data already in place: {data_name}")
        else:
            print(f"  WARNING: external data file not found: {src_data}")

        orig_in = {i.name: self._save_vi_shape(i) for i in model.graph.input}
        orig_out = {o.name: self._save_vi_shape(o) for o in model.graph.output}
        orig_vi = {v.name: self._save_vi_shape(v) for v in model.graph.value_info}

        has_tsl = (
            any(
                inp == TOTAL_SEQ_LEN_CONST_NAME
                for node in model.graph.node
                for inp in node.input
            )
            and self.model_type == "llm"
        )

        for idx, (vname, dim_map) in enumerate(self.variants):
            llm_total = (
                total_seq_len_scalar_from_dim_map(dim_map)
                if self.model_type == "llm"
                else None
            )

            if has_tsl:
                if dim_map is None:
                    for node in model.graph.node:
                        for j, inp in enumerate(node.input):
                            if inp == TOTAL_SEQ_LEN_CONST_NAME:
                                node.input[j] = "total_sequence_length"
                    tsl_vi = make_tensor_vi(
                        "total_sequence_length", TensorProto.INT32, [1]
                    )
                    model.graph.input.append(tsl_vi)
                else:
                    for node in model.graph.node:
                        for j, inp in enumerate(node.input):
                            if inp == "total_sequence_length":
                                node.input[j] = TOTAL_SEQ_LEN_CONST_NAME
                    to_rm = [
                        i
                        for i, vi in enumerate(model.graph.input)
                        if vi.name == "total_sequence_length"
                    ]
                    for i in reversed(to_rm):
                        del model.graph.input[i]

            if llm_total is not None and (not has_tsl or dim_map is not None):
                patch_fold_total_seq_len_constant_value(model, llm_total)
                upsert_total_seq_len_const_initializer(model.graph, llm_total)

            for vi in (
                list(model.graph.input)
                + list(model.graph.output)
                + list(model.graph.value_info)
            ):
                self._set_vi_shape(vi, dim_map, extra=self.config_dims)

            path = os.path.join(out_dir, self._full_model_variant_onnx_basename(vname))
            with open(path, "wb") as f:
                f.write(model.SerializeToString())
            print(f"  Saved: {path}")

            for vi in model.graph.input:
                self._restore_vi_shape(vi, orig_in.get(vi.name))
            for vi in model.graph.output:
                self._restore_vi_shape(vi, orig_out.get(vi.name))
            for vi in model.graph.value_info:
                self._restore_vi_shape(vi, orig_vi.get(vi.name))

    def _save_variants(self, model, out_dir, prefix):
        """Save variants with shared external data file."""
        os.makedirs(out_dir, exist_ok=True)
        data_name = self.ext_data_name
        stale = os.path.join(out_dir, data_name)
        if os.path.exists(stale):
            os.remove(stale)

        orig_in = {i.name: self._save_vi_shape(i) for i in model.graph.input}
        orig_out = {o.name: self._save_vi_shape(o) for o in model.graph.output}
        orig_vi = {v.name: self._save_vi_shape(v) for v in model.graph.value_info}

        for idx, (vname, dim_map) in enumerate(self.variants):
            if self.model_type == "llm":
                llm_total = total_seq_len_scalar_from_dim_map(dim_map)
                if llm_total is not None:
                    patch_fold_total_seq_len_constant_value(model, llm_total)
                    upsert_total_seq_len_const_initializer(model.graph, llm_total)

            for vi in (
                list(model.graph.input)
                + list(model.graph.output)
                + list(model.graph.value_info)
            ):
                self._set_vi_shape(vi, dim_map, extra=self.config_dims)

            save_model = model
            if dim_map is not None:
                save_model = self._reinfer_and_fold(model)

            path = os.path.join(out_dir, f"{prefix}_{vname}.onnx")
            if idx == 0:
                onnx.save(
                    save_model,
                    path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=data_name,
                    size_threshold=1024,
                )
            else:
                with open(path, "wb") as f:
                    f.write(save_model.SerializeToString())
            print(f"  Saved: {path}")

            for vi in model.graph.input:
                self._restore_vi_shape(vi, orig_in.get(vi.name))
            for vi in model.graph.output:
                self._restore_vi_shape(vi, orig_out.get(vi.name))
            for vi in model.graph.value_info:
                self._restore_vi_shape(vi, orig_vi.get(vi.name))

    @classmethod
    def _reinfer_and_fold(cls, model):
        """Re-run shape inference on a fixed-shape variant to resolve
        intermediate symbolic dims (u1, u2, ...), then optionally run
        onnxoptimizer to eliminate Shape/Gather nodes."""
        m = onnx.ModelProto()
        m.CopyFrom(model)

        try:
            m = onnx.shape_inference.infer_shapes(m)
        except Exception:
            del m.graph.value_info[:]
            try:
                m = onnx.shape_inference.infer_shapes(m)
            except Exception as e:
                print(f"    re-infer skipped: {e}")

        if not _HAS_OPTIMIZER:
            return m

        before = len(m.graph.node)
        try:
            opt = onnxoptimizer.optimize(m, cls._OPT_PASSES)
            after = len(opt.graph.node)
            if after < before:
                print(f"    shape-fold: {before} -> {after} nodes (-{before - after})")
            return opt
        except Exception as e:
            print(f"    shape-fold skipped: {e}")
            return m

    # ── Shape helpers ──

    @staticmethod
    def _save_vi_shape(vi):
        if not vi.type.tensor_type.shape or len(vi.type.tensor_type.shape.dim) == 0:
            return None
        return [
            ("param", d.dim_param) if d.dim_param else ("value", d.dim_value)
            for d in vi.type.tensor_type.shape.dim
        ]

    @staticmethod
    def _restore_vi_shape(vi, saved):
        if saved is None:
            return
        for dim, (kind, val) in zip(vi.type.tensor_type.shape.dim, saved):
            if kind == "param":
                dim.ClearField("dim_value")
                dim.dim_param = val
            else:
                dim.ClearField("dim_param")
                dim.dim_value = val

    @staticmethod
    def _set_vi_shape(vi, dim_map, extra=None):
        if not vi.type.tensor_type.shape:
            return
        for dim in vi.type.tensor_type.shape.dim:
            if dim.dim_param:
                val = resolve_dim_param(dim.dim_param, dim_map, extra=extra)
                if isinstance(val, int):
                    dim.ClearField("dim_param")
                    dim.dim_value = val

    # ── run ──

    def run(self):
        self.extract_all_single_ops()
        self.extract_single_layer()
        self.extract_full_model()
        print("\n" + "=" * 60)
        print("Done!")
        print("=" * 60)


# ─────────────────────── main ───────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Universal ONNX submodel extractor for LLMs "
        "(auto-discovers graph topology)"
    )
    parser.add_argument("--model", required=True, help="Path to model.onnx")
    parser.add_argument(
        "--output", default=None, help="Output directory (default: same as model)"
    )
    parser.add_argument(
        "--only",
        choices=["single_op", "single_layer", "full_model"],
        help="Run only a specific extraction step",
    )
    args = parser.parse_args()

    extractor = SingleOpExtractor(args.model, output_dir=args.output)
    if args.only == "single_op":
        extractor.extract_all_single_ops()
    elif args.only == "single_layer":
        extractor.extract_single_layer()
    elif args.only == "full_model":
        extractor.extract_full_model()
    else:
        extractor.run()


if __name__ == "__main__":
    main()
