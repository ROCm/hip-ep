"""Convert an ONNX model to HIP dialect MLIR.

Applies the following transformations:
  Phase A1: Op remapping (onnx.* -> hip.*, com.microsoft.* -> hip.miopen.*)
  Phase A2: Weight extraction (constants -> function arguments with memref types)
  Phase A3: Pattern fusion (LpNorm+Mul -> rms_norm, Sigmoid+Mul -> silu)
  Phase B:  Structural transforms (tensor->memref, layer loop, memory lifecycle)
"""

import argparse
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import onnx
from onnx import TensorProto

# ---------------------------------------------------------------------------
# Shared utilities (from onnx_to_mlir.py)
# ---------------------------------------------------------------------------

ELEM_TYPE_TO_MLIR = {
    TensorProto.FLOAT: "f32",
    TensorProto.DOUBLE: "f64",
    TensorProto.FLOAT16: "f16",
    TensorProto.BFLOAT16: "bf16",
    TensorProto.INT8: "i8",
    TensorProto.INT16: "i16",
    TensorProto.INT32: "i32",
    TensorProto.INT64: "i64",
    TensorProto.UINT8: "ui8",
    TensorProto.UINT16: "ui16",
    TensorProto.UINT32: "ui32",
    TensorProto.UINT64: "ui64",
    TensorProto.BOOL: "i1",
}


def mlir_elem_type(onnx_elem_type: int) -> str:
    return ELEM_TYPE_TO_MLIR.get(onnx_elem_type, "f32")


def sanitize_name(name: str) -> str:
    if not name:
        return ""
    s = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if s and s[0].isdigit():
        s = "v" + s
    return s


# ---------------------------------------------------------------------------
# Op mapping table: (domain, op_type) -> hip op name
# ---------------------------------------------------------------------------

OP_MAP: dict[tuple[str, str], str] = {
    # --- hipBLASLt (GEMM) ---
    ("", "MatMul"):              "hip.hipblaslt.matmul",
    ("", "Gemm"):                "hip.hipblaslt.matmul",
    # --- MIOpen: normalization (miopenLayerNormForward / miopenT5LayerNormForward / miopenAddLayerNormForward) ---
    ("", "LayerNormalization"):  "hip.miopen.layer_norm",
    ("", "SimplifiedLayerNormalization"): "hip.miopen.t5_layer_norm",
    ("com.microsoft", "SkipSimplifiedLayerNormalization"): "hip.miopen.skip_rms_norm",
    ("com.microsoft", "SkipLayerNormalization"): "hip.miopen.skip_layer_norm",
    ("com.microsoft", "SimplifiedLayerNormalization"): "hip.miopen.t5_layer_norm",
    # --- MIOpen: activation (miopenActivationForward) ---
    ("", "Sigmoid"):             "hip.miopen.sigmoid",
    ("", "Relu"):                "hip.miopen.relu",
    # --- MIOpen: softmax (miopenSoftmaxForward) ---
    ("", "Softmax"):             "hip.miopen.softmax",
    # --- MIOpen: element-wise tensor ops (miopenOpTensor) ---
    ("", "Add"):                 "hip.miopen.add",
    ("", "Mul"):                 "hip.miopen.mul",
    # --- MIOpen: reduction (miopenReduceTensor) ---
    ("", "ReduceMean"):          "hip.miopen.reduce_mean",
    # --- MIOpen: concat (miopenCatForward, experimental) ---
    ("", "Concat"):              "hip.miopen.cat",
    # --- Attention ---
    ("com.microsoft", "GroupQueryAttention"): "hip.gqa",
    # --- Quantization (blank impl placeholders) ---
    ("", "QuantizeLinear"):      "hip.quantize_linear",
    ("", "DequantizeLinear"):    "hip.dequantize_linear",
    # --- Zero-cost metadata ops (no kernel) ---
    ("", "Reshape"):             "hip.reshape",
    ("", "Unsqueeze"):           "hip.unsqueeze",
    ("", "Squeeze"):             "hip.squeeze",
    # --- Custom HIP kernels (no MIOpen equivalent) ---
    ("", "Transpose"):           "hip.transpose",
    ("", "Gather"):              "hip.gather",
    ("", "Cast"):                "hip.cast",
    ("", "Div"):                 "hip.div",
    ("", "Pow"):                 "hip.pow",
    ("", "Sqrt"):                "hip.sqrt",
    ("", "Constant"):            "hip.constant",
    ("", "LpNormalization"):     "hip.lp_normalization",
}


def map_op(domain: str, op_type: str) -> str:
    key = (domain, op_type)
    if key in OP_MAP:
        return OP_MAP[key]
    return f"hip.{op_type}"


# ---------------------------------------------------------------------------
# Graph region grouping
# ---------------------------------------------------------------------------

def _op_backend(hip_op: str) -> str | None:
    """Return the backend graph region for an op, or None if ungrouped."""
    if hip_op.startswith("hip.hipblaslt."):
        return "hip.hipblaslt.graph"
    if hip_op.startswith("hip.miopen."):
        return "hip.miopen.graph"
    return None


# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------

def _parse_dims(type_proto) -> tuple[list[int | str], str]:
    """Extract (dims, elem_type_str) from a TensorTypeProto."""
    tt = type_proto.tensor_type
    et = mlir_elem_type(tt.elem_type)
    dims: list[int | str] = []
    if tt.HasField("shape"):
        for d in tt.shape.dim:
            if d.HasField("dim_value"):
                dims.append(d.dim_value)
            elif d.HasField("dim_param"):
                dims.append(d.dim_param)
            else:
                dims.append("?")
    return dims, et


def mlir_type(dims: list[int | str], et: str, *, use_memref: bool = False) -> str:
    prefix = "memref" if use_memref else "tensor"
    suffix = ", 1" if use_memref else ""
    if not dims:
        if use_memref:
            return f"memref<{et}{suffix}>"
        return f"tensor<{et}>"
    dim_strs = []
    for d in dims:
        if isinstance(d, int) and d >= 0:
            dim_strs.append(str(d))
        else:
            dim_strs.append("?")
    return f"{prefix}<{'x'.join(dim_strs)}x{et}{suffix}>"


# ---------------------------------------------------------------------------
# Graph-level pattern fusion (Phase A3)
# ---------------------------------------------------------------------------

@dataclass
class FusedOp:
    """Represents a fused operation that replaces multiple ONNX nodes."""
    hip_op: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict
    consumed_nodes: set[int]  # indices of nodes consumed by this fusion


def _build_producer_map(nodes: list[onnx.NodeProto]) -> dict[str, int]:
    """Map output tensor name -> node index."""
    m: dict[str, int] = {}
    for i, n in enumerate(nodes):
        for out in n.output:
            if out:
                m[out] = i
    return m


def _build_consumer_map(nodes: list[onnx.NodeProto]) -> dict[str, list[int]]:
    """Map input tensor name -> list of node indices that consume it."""
    m: dict[str, list[int]] = defaultdict(list)
    for i, n in enumerate(nodes):
        for inp in n.input:
            if inp:
                m[inp].append(i)
    return m


def detect_fusions(nodes: list[onnx.NodeProto]) -> dict[int, FusedOp]:
    """Detect fusible patterns and return a map of anchor_node_index -> FusedOp.

    Patterns detected:
    - LpNormalization(p=2,axis=-1) -> [Q/DQ] -> Mul(_, weight) => hip.miopen.rms_norm
    - Sigmoid(x) -> [Q/DQ] -> Mul(x, sigmoid_out) => hip.silu
    """
    producer = _build_producer_map(nodes)
    consumer = _build_consumer_map(nodes)
    fusions: dict[int, FusedOp] = {}
    consumed: set[int] = set()

    for i, node in enumerate(nodes):
        if i in consumed:
            continue

        # --- RMSNorm: LpNormalization(p=2, axis=-1) ... Mul ---
        if node.op_type == "LpNormalization":
            p_val = next((a.i for a in node.attribute if a.name == "p"), 2)
            axis_val = next((a.i for a in node.attribute if a.name == "axis"), -1)
            if p_val == 2 and axis_val == -1:
                mul_node, chain = _find_mul_after(nodes, i, producer, consumer)
                if mul_node is not None:
                    mul_idx = next(
                        idx for idx, n in enumerate(nodes) if n is mul_node
                    )
                    all_consumed = {i, mul_idx} | chain
                    if not all_consumed & consumed:
                        fusions[i] = FusedOp(
                            hip_op="hip.miopen.rms_norm",
                            inputs=list(node.input) + [_get_weight_input(mul_node, nodes, i, chain)],
                            outputs=list(mul_node.output),
                            attrs={},
                            consumed_nodes=all_consumed,
                        )
                        consumed |= all_consumed

        # --- SiLU: Sigmoid(x) ... Mul(x, sigmoid_out) ---
        if node.op_type == "Sigmoid" and i not in consumed:
            mul_node, chain = _find_silu_mul_after(nodes, i, producer, consumer)
            if mul_node is not None:
                mul_idx = next(
                    idx for idx, n in enumerate(nodes) if n is mul_node
                )
                all_consumed = {i, mul_idx} | chain
                if not all_consumed & consumed:
                    fusions[i] = FusedOp(
                        hip_op="hip.silu",
                        inputs=list(node.input),
                        outputs=list(mul_node.output),
                        attrs={},
                        consumed_nodes=all_consumed,
                    )
                    consumed |= all_consumed

    return fusions


def _skip_qdq_chain(
    nodes: list[onnx.NodeProto],
    start_output: str,
    consumer: dict[str, list[int]],
) -> tuple[str, set[int]]:
    """Follow a QuantizeLinear -> DequantizeLinear chain starting from start_output.
    Returns (final_output_name, set_of_consumed_node_indices)."""
    chain: set[int] = set()
    current = start_output

    consumers = consumer.get(current, [])
    if len(consumers) != 1:
        return current, chain
    q_idx = consumers[0]
    q_node = nodes[q_idx]
    if q_node.op_type != "QuantizeLinear":
        return current, chain
    chain.add(q_idx)
    q_out = q_node.output[0]

    dq_consumers = consumer.get(q_out, [])
    if len(dq_consumers) != 1:
        return current, chain
    dq_idx = dq_consumers[0]
    dq_node = nodes[dq_idx]
    if dq_node.op_type != "DequantizeLinear":
        return current, chain
    chain.add(dq_idx)
    return dq_node.output[0], chain


def _find_mul_after(
    nodes: list[onnx.NodeProto],
    lpnorm_idx: int,
    producer: dict[str, int],
    consumer: dict[str, list[int]],
) -> tuple[onnx.NodeProto | None, set[int]]:
    """Find the Mul node that follows an LpNormalization, possibly through Q/DQ."""
    lpnorm = nodes[lpnorm_idx]
    out_name = lpnorm.output[0]
    final_name, chain = _skip_qdq_chain(nodes, out_name, consumer)

    consumers = consumer.get(final_name, [])
    for c_idx in consumers:
        c_node = nodes[c_idx]
        if c_node.op_type == "Mul":
            return c_node, chain
    return None, set()


def _get_weight_input(
    mul_node: onnx.NodeProto,
    nodes: list[onnx.NodeProto],
    lpnorm_idx: int,
    chain: set[int],
) -> str:
    """For a Mul node that's part of RMSNorm, return the weight input name.

    The weight is the Mul input that does NOT trace back to the LpNorm output
    (the other input comes through the Q/DQ chain).
    """
    chain_outputs = set()
    chain_outputs.add(nodes[lpnorm_idx].output[0])
    for ci in chain:
        for o in nodes[ci].output:
            chain_outputs.add(o)

    for inp in mul_node.input:
        if inp and inp not in chain_outputs:
            return inp
    return mul_node.input[1]


def _find_silu_mul_after(
    nodes: list[onnx.NodeProto],
    sigmoid_idx: int,
    producer: dict[str, int],
    consumer: dict[str, list[int]],
) -> tuple[onnx.NodeProto | None, set[int]]:
    """Find the Mul(x, sigmoid(x)) pattern after a Sigmoid node."""
    sigmoid = nodes[sigmoid_idx]
    sigmoid_input = sigmoid.input[0]
    sigmoid_output = sigmoid.output[0]

    final_name, chain = _skip_qdq_chain(nodes, sigmoid_output, consumer)

    consumers = consumer.get(final_name, [])
    for c_idx in consumers:
        c_node = nodes[c_idx]
        if c_node.op_type == "Mul":
            inputs = list(c_node.input)
            # The original input to sigmoid must also be an input to Mul
            # (possibly through its own Q/DQ chain)
            if _traces_back_to(inputs, sigmoid_input, nodes, producer, consumer):
                return c_node, chain
    return None, set()


def _traces_back_to(
    mul_inputs: list[str],
    target: str,
    nodes: list[onnx.NodeProto],
    producer: dict[str, int],
    consumer: dict[str, list[int]],
) -> bool:
    """Check if any of mul_inputs traces back to target, possibly through Q/DQ."""
    for inp in mul_inputs:
        if inp == target:
            return True
        if inp in producer:
            p_node = nodes[producer[inp]]
            if p_node.op_type == "DequantizeLinear":
                q_input = p_node.input[0]
                if q_input in producer:
                    q_node = nodes[producer[q_input]]
                    if q_node.op_type == "QuantizeLinear" and q_node.input[0] == target:
                        return True
    return False


# ---------------------------------------------------------------------------
# HIP MLIR Emitter
# ---------------------------------------------------------------------------

class HipEmitter:
    """Emits HIP dialect MLIR from an ONNX graph."""

    def __init__(
        self,
        graph: onnx.GraphProto,
        *,
        use_memref: bool = False,
        extract_weights: bool = False,
        fuse_patterns: bool = False,
        add_lifecycle: bool = False,
    ):
        self.graph = graph
        self.use_memref = use_memref
        self.extract_weights = extract_weights
        self.fuse_patterns = fuse_patterns
        self.add_lifecycle = add_lifecycle

        self.lines: list[str] = []
        self.indent = 0
        self.name_counter = Counter()
        self.ssa_map: dict[str, str] = {}
        self.type_map: dict[str, str] = {}
        self.init_names: set[str] = set()

        self.fusions: dict[int, FusedOp] = {}
        self.consumed_by_fusion: set[int] = set()

    def emit(self) -> str:
        self._build_type_map()
        self._build_init_set()
        if self.fuse_patterns:
            self.fusions = detect_fusions(list(self.graph.node))
            for f in self.fusions.values():
                self.consumed_by_fusion |= f.consumed_nodes
        self._emit_module()
        return "\n".join(self.lines) + "\n"

    # --- Type resolution ---

    def _build_type_map(self):
        for vi in list(self.graph.input) + list(self.graph.output) + list(self.graph.value_info):
            if vi.type.HasField("tensor_type"):
                dims, et = _parse_dims(vi.type)
                self.type_map[vi.name] = mlir_type(dims, et, use_memref=self.use_memref)
            elif not vi.type.HasField("tensor_type"):
                self.type_map[vi.name] = "none"

        for init in self.graph.initializer:
            if init.name not in self.type_map:
                et = mlir_elem_type(init.data_type)
                dims = list(init.dims)
                self.type_map[init.name] = mlir_type(dims, et, use_memref=self.use_memref)

    def _build_init_set(self):
        self.init_names = {init.name for init in self.graph.initializer}

    def _get_type(self, tensor_name: str) -> str:
        return self.type_map.get(tensor_name, "tensor<*xf32>")

    # --- SSA ---

    def _fresh_ssa(self, hint: str = "") -> str:
        base = sanitize_name(hint) if hint else "v"
        if not base:
            base = "v"
        self.name_counter[base] += 1
        count = self.name_counter[base]
        return base if count == 1 else f"{base}_{count}"

    def _def_ssa(self, onnx_name: str) -> str:
        ssa = self._fresh_ssa(onnx_name)
        self.ssa_map[onnx_name] = ssa
        return f"%{ssa}"

    def _use_ssa(self, onnx_name: str) -> str:
        if not onnx_name:
            return "%_none"
        ssa = self.ssa_map.get(onnx_name)
        if ssa is None:
            ssa = self._fresh_ssa(onnx_name)
            self.ssa_map[onnx_name] = ssa
        return f"%{ssa}"

    # --- Output ---

    def _line(self, text: str):
        self.lines.append("  " * self.indent + text)

    def _blank(self):
        self.lines.append("")

    @staticmethod
    def _format_attr(attr: onnx.AttributeProto) -> str:
        if attr.type == onnx.AttributeProto.INT:
            return f"{attr.i} : i64"
        if attr.type == onnx.AttributeProto.FLOAT:
            v = attr.f
            s = f"{v:.8g}"
            if "." not in s and "e" not in s.lower():
                s += ".0"
            return f"{s} : f32"
        if attr.type == onnx.AttributeProto.STRING:
            escaped = attr.s.decode("utf-8", errors="replace").replace('"', '\\"')
            return f'"{escaped}"'
        if attr.type == onnx.AttributeProto.INTS:
            vals = ", ".join(str(v) for v in attr.ints)
            return f"[{vals}]"
        if attr.type == onnx.AttributeProto.FLOATS:
            vals = ", ".join(f"{v:.8g}" for v in attr.floats)
            return f"[{vals}]"
        if attr.type == onnx.AttributeProto.TENSOR:
            t = attr.t
            et = mlir_elem_type(t.data_type)
            dims = list(t.dims)
            return f"dense<0> : {mlir_type(dims, et)}"
        return f'"<{onnx.AttributeProto.AttributeType.Name(attr.type)}>"'

    # --- Emission ---

    def _emit_module(self):
        self._line("module {")
        self.indent += 1
        self._emit_func()
        self.indent -= 1
        self._line("}")

    def _emit_func(self):
        real_inputs = [inp for inp in self.graph.input if inp.name not in self.init_names]

        arg_strs = []
        for inp in real_inputs:
            ssa = self._def_ssa(inp.name)
            ty = self._get_type(inp.name)
            arg_strs.append(f"{ssa}: {ty}")

        if self.extract_weights:
            for init in self.graph.initializer:
                ssa = self._def_ssa(init.name)
                ty = self._get_type(init.name)
                arg_strs.append(f"{ssa}: {ty}  /* weight: {init.name} */")

        ret_types = [self._get_type(out.name) for out in self.graph.output]

        graph_name = sanitize_name(self.graph.name) or "main_graph"
        args_str = ",\n    ".join(arg_strs)
        rets_str = ", ".join(ret_types) if ret_types else ""

        if rets_str:
            self._line(f"func.func @{graph_name}(")
            self._line(f"    {args_str}")
            self._line(f"  ) -> ({rets_str}) {{")
        else:
            self._line(f"func.func @{graph_name}(")
            self._line(f"    {args_str}")
            self._line(f"  ) {{")
        self.indent += 1

        if self.add_lifecycle:
            self._line('%handle = hip.create_handle() : !hip.handle')
            self._blank()

        if not self.extract_weights:
            self._emit_initializers()
            self._blank()

        self._emit_nodes()
        self._blank()

        if self.add_lifecycle:
            self._line('hip.destroy_handle(%handle) : !hip.handle')
            self._blank()

        self._emit_return()

        self.indent -= 1
        self._line("}")

    def _emit_initializers(self):
        for init in self.graph.initializer:
            ssa = self._def_ssa(init.name)
            ty = self._get_type(init.name)
            self._line(
                f'{ssa} = "hip.constant"() '
                f'{{value = dense<0> : {ty}}} : () -> {ty}'
                f'  // {init.name}'
            )

    def _emit_nodes(self):
        nodes = list(self.graph.node)

        # Build a list of (hip_op_name, emit_fn) for each visible node
        emit_items: list[tuple[str, callable]] = []
        for i, node in enumerate(nodes):
            if i in self.consumed_by_fusion and i not in self.fusions:
                continue
            if i in self.fusions:
                fop = self.fusions[i]
                emit_items.append((fop.hip_op, lambda f=fop: self._emit_fused_op(f)))
            else:
                domain = node.domain or ""
                hip_op = map_op(domain, node.op_type)
                emit_items.append((hip_op, lambda n=node: self._emit_node(n)))

        # Group consecutive ops by backend and emit with graph regions
        idx = 0
        while idx < len(emit_items):
            hip_op, emit_fn = emit_items[idx]
            backend = _op_backend(hip_op)

            if backend is None:
                emit_fn()
                idx += 1
            else:
                # Collect consecutive ops with the same backend
                group_start = idx
                while idx < len(emit_items) and _op_backend(emit_items[idx][0]) == backend:
                    idx += 1
                group = emit_items[group_start:idx]

                self._line(f"{backend} {{")
                self.indent += 1
                for _, fn in group:
                    fn()
                self.indent -= 1
                self._line("}")

    def _emit_fused_op(self, fop: FusedOp):
        input_ssas = [self._use_ssa(n) for n in fop.inputs if n]
        input_types = [self._get_type(n) for n in fop.inputs if n]
        output_ssas = [self._def_ssa(n) for n in fop.outputs]
        output_types = [self._get_type(n) for n in fop.outputs]

        lhs = ", ".join(output_ssas)
        rhs = ", ".join(input_ssas)
        in_t = ", ".join(input_types)
        out_t = ", ".join(output_types)

        attr_strs = []
        for k, v in fop.attrs.items():
            attr_strs.append(f"{k} = {v}")
        attrs_part = " {" + ", ".join(attr_strs) + "}" if attr_strs else ""

        func_type = f"({in_t}) -> {out_t}" if out_t else f"({in_t}) -> ()"

        if lhs:
            self._line(f'{lhs} = "{fop.hip_op}"({rhs}){attrs_part} : {func_type}')
        else:
            self._line(f'"{fop.hip_op}"({rhs}){attrs_part} : {func_type}')

    def _emit_node(self, node: onnx.NodeProto):
        input_ssas = []
        input_types = []
        for name in node.input:
            if name == "":
                input_ssas.append("%_none")
                input_types.append("none")
            else:
                input_ssas.append(self._use_ssa(name))
                input_types.append(self._get_type(name))

        output_ssas = []
        output_types = []
        for name in node.output:
            output_ssas.append(self._def_ssa(name))
            output_types.append(self._get_type(name))

        attr_strs = [f"{a.name} = {self._format_attr(a)}" for a in node.attribute]

        domain = node.domain or ""
        hip_op = map_op(domain, node.op_type)

        lhs = ", ".join(output_ssas)
        rhs_args = ", ".join(input_ssas)
        in_types = ", ".join(input_types)
        out_types = ", ".join(output_types)
        attrs_part = " {" + ", ".join(attr_strs) + "}" if attr_strs else ""
        func_type = f"({in_types}) -> {out_types}" if out_types else f"({in_types}) -> ()"
        name_comment = f"  // {node.name}" if node.name else ""

        if lhs:
            self._line(f'{lhs} = "{hip_op}"({rhs_args}){attrs_part} : {func_type}{name_comment}')
        else:
            self._line(f'"{hip_op}"({rhs_args}){attrs_part} : {func_type}{name_comment}')

    def _emit_return(self):
        if not self.graph.output:
            self._line("return")
            return
        ret_ssas = [self._use_ssa(out.name) for out in self.graph.output]
        ret_types = [self._get_type(out.name) for out in self.graph.output]
        self._line(f"return {', '.join(ret_ssas)} : {', '.join(ret_types)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert(
    model_path: str,
    *,
    use_memref: bool = False,
    extract_weights: bool = False,
    fuse_patterns: bool = False,
    add_lifecycle: bool = False,
) -> str:
    model = onnx.load(model_path, load_external_data=False)
    try:
        model = onnx.shape_inference.infer_shapes(model, data_prop=True)
    except Exception:
        pass

    emitter = HipEmitter(
        model.graph,
        use_memref=use_memref,
        extract_weights=extract_weights,
        fuse_patterns=fuse_patterns,
        add_lifecycle=add_lifecycle,
    )
    return emitter.emit()


def main():
    parser = argparse.ArgumentParser(
        description="Convert an ONNX model to HIP dialect MLIR"
    )
    parser.add_argument("model", help="Path to .onnx model file")
    parser.add_argument("-o", "--output", help="Output .mlir file (default: stdout)")
    parser.add_argument("--memref", action="store_true",
                        help="Use memref<..., 1> types instead of tensor<...>")
    parser.add_argument("--extract-weights", action="store_true",
                        help="Move weights from constants to function arguments")
    parser.add_argument("--fuse", action="store_true",
                        help="Fuse patterns (RMSNorm, SiLU) into single ops")
    parser.add_argument("--lifecycle", action="store_true",
                        help="Add hip.create_handle / hip.destroy_handle lifecycle")
    parser.add_argument("--all", action="store_true",
                        help="Enable all transformations (--memref --extract-weights --fuse --lifecycle)")
    args = parser.parse_args()

    if args.all:
        args.memref = True
        args.extract_weights = True
        args.fuse = True
        args.lifecycle = True

    mlir_text = convert(
        args.model,
        use_memref=args.memref,
        extract_weights=args.extract_weights,
        fuse_patterns=args.fuse,
        add_lifecycle=args.lifecycle,
    )

    if args.output:
        with open(args.output, "w") as f:
            f.write(mlir_text)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(mlir_text)


if __name__ == "__main__":
    main()
