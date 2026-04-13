#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
FX graph → Torch dialect MLIR emitter.

Refactored from the monolithic fx_to_mlir.py into:
  - EmitterContext: shared state and helper methods
  - Per-op dispatch via fx_emitter_ops.py
  - Clean constant folding and type inference

Usage:
    from hip_torch.fx_emitter import fx_graph_to_mlir
    ep = torch.export.export(model, example_inputs)
    mlir_text = fx_graph_to_mlir(ep)
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.export import ExportedProgram

from . import op_registry


# ──────────────────────────────────────────────────────────────────────
# MLIR Type Helpers
# ──────────────────────────────────────────────────────────────────────

_DTYPE_MAP = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
    torch.float64: "f64",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.uint8: "ui8",
    torch.bool: "i1",
}


def dtype_to_mlir(dtype: torch.dtype) -> str:
    """Convert a PyTorch dtype to an MLIR type string."""
    if dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return _DTYPE_MAP[dtype]


def tensor_type_str(shape: List[int], dtype: torch.dtype) -> str:
    """Build an MLIR tensor type string like 'tensor<1x4x64xf16>'."""
    if len(shape) == 0:
        return f"tensor<{dtype_to_mlir(dtype)}>"
    dims = "x".join(str(d) for d in shape)
    return f"tensor<{dims}x{dtype_to_mlir(dtype)}>"


def _is_weight_placeholder(node) -> bool:
    """Check if a placeholder node is a weight (parameter) vs input."""
    return node.target.startswith("p_") or node.target.startswith("b_")


# ──────────────────────────────────────────────────────────────────────
# Emitter Context
# ──────────────────────────────────────────────────────────────────────


class EmitterContext:
    """Shared state for MLIR emission.

    Provides helpers for:
    - SSA value naming and reference resolution
    - Constant emission (int, float, bool, None, list)
    - Type tracking across nodes
    - Argument padding for ops with optional trailing args
    """

    def __init__(self):
        self.lines: List[str] = []
        self.type_map: Dict[str, str] = {}
        self.arg_name_map: Dict[str, str] = {}
        self.indent = "    "
        self._const_counter = 0
        self._const_cache: Dict[Any, str] = {}

    def _next_name(self, prefix: str) -> str:
        self._const_counter += 1
        return f"%_k{self._const_counter}_{prefix}"

    def get_tensor_type(self, node, transposed: bool = False) -> Optional[str]:
        """Get MLIR tensor type from node metadata."""
        val = node.meta.get("val", None)
        if isinstance(val, torch.Tensor):
            shape = list(val.shape)
            if transposed and len(shape) == 2:
                shape = [shape[1], shape[0]]
            return tensor_type_str(shape, val.dtype)
        if isinstance(val, torch.SymInt):
            return "!torch.int"
        if isinstance(val, torch.SymBool):
            return "!torch.bool"
        if val is None:
            return "!torch.none"
        return None

    def resolve_arg(self, arg) -> Tuple[str, str]:
        """Resolve an FX arg to (ssa_ref, mlir_type)."""
        if hasattr(arg, "name"):
            ref = self.arg_name_map.get(arg.name, f"%{arg.name.replace('.', '_')}")
            atype = self.type_map.get(arg.name, "!torch.unknown")
            return ref, atype
        if isinstance(arg, bool):
            # Bool as int (most torch ops expect int for scalar bool args)
            name = self.emit_int_constant(1 if arg else 0)
            return name, "!torch.int"
        if isinstance(arg, int):
            name = self.emit_int_constant(arg)
            return name, "!torch.int"
        if isinstance(arg, float):
            name = self.emit_float_constant(arg)
            return name, "!torch.float"
        if arg is None:
            name = self.emit_none_constant()
            return name, "!torch.none"
        if isinstance(arg, (list, tuple)):
            return self.emit_list_constant(arg)
        return f"/*{type(arg).__name__}*/", "!torch.unknown"

    def emit_int_constant(self, val: int) -> str:
        """Emit torch.constant.int, returns SSA name."""
        cache_key = ("int", val)
        if cache_key in self._const_cache:
            return self._const_cache[cache_key]
        name = self._next_name(f"i{val}" if val >= 0 else f"in{abs(val)}")
        self.lines.append(
            f'{self.indent}{name} = "torch.constant.int"() '
            f"{{value = {val} : i64}} : () -> !torch.int"
        )
        self._const_cache[cache_key] = name
        return name

    def emit_float_constant(self, val: float) -> str:
        """Emit torch.constant.float, returns SSA name."""
        cache_key = ("float", val)
        if cache_key in self._const_cache:
            return self._const_cache[cache_key]
        name = self._next_name("f")
        if math.isinf(val):
            val_str = (
                "0x7FF0000000000000 : f64" if val > 0 else "0xFFF0000000000000 : f64"
            )
        elif math.isnan(val):
            val_str = "0x7FF8000000000000 : f64"
        else:
            val_str = f"{val:.6e} : f64"
        self.lines.append(
            f'{self.indent}{name} = "torch.constant.float"() '
            f"{{value = {val_str}}} : () -> !torch.float"
        )
        self._const_cache[cache_key] = name
        return name

    def emit_none_constant(self) -> str:
        """Emit torch.constant.none, returns SSA name."""
        cache_key = ("none", None)
        if cache_key in self._const_cache:
            return self._const_cache[cache_key]
        name = "%_none"
        self.lines.append(
            f'{self.indent}{name} = "torch.constant.none"() : () -> !torch.none'
        )
        self._const_cache[cache_key] = name
        return name

    def emit_list_constant(self, items) -> Tuple[str, str]:
        """Emit torch.prim.ListConstruct, returns (SSA name, type)."""
        has_nodes = any(hasattr(e, "name") for e in items)
        if has_nodes:
            elem_names, elem_types = [], []
            for elem in items:
                if hasattr(elem, "name"):
                    ref, t = self.resolve_arg(elem)
                    elem_names.append(ref)
                    elem_types.append(t)
                else:
                    ref, t = self.resolve_arg(elem)
                    elem_names.append(ref)
                    elem_types.append(t)
            list_name = self._next_name("list")
            self.lines.append(
                f'{self.indent}{list_name} = "torch.prim.ListConstruct"'
                f"({', '.join(elem_names)}) : "
                f"({', '.join(elem_types)}) -> !torch.list<tensor>"
            )
            return list_name, "!torch.list<tensor>"
        else:
            elem_names = []
            for elem in items:
                ref, _ = self.resolve_arg(elem)
                elem_names.append(ref)
            list_name = self._next_name("list")
            types_str = ", ".join(["!torch.int"] * len(elem_names))
            self.lines.append(
                f'{self.indent}{list_name} = "torch.prim.ListConstruct"'
                f"({', '.join(elem_names)}) : "
                f"({types_str}) -> !torch.list<int>"
            )
            return list_name, "!torch.list<int>"

    def pad_args(self, node, torch_op: str) -> list:
        """Pad trailing optional args with defaults from registry."""
        padded = list(node.args)
        info = op_registry.get_info(torch_op)
        if info and info.default_values:
            expected = len(padded) + len(info.default_values)
            while len(padded) < expected:
                idx = len(padded) - (expected - len(info.default_values))
                if 0 <= idx < len(info.default_values):
                    padded.append(info.default_values[idx][0])
                else:
                    padded.append(None)
        return padded

    def format_op(self, torch_op: str, node, args: list) -> str:
        """Format a torch op as MLIR generic form and append to lines."""
        mlir_args, mlir_types = [], []
        for arg in args:
            ref, t = self.resolve_arg(arg)
            mlir_args.append(ref)
            mlir_types.append(t)

        out_type = self.get_tensor_type(node) or "!torch.unknown"
        self.type_map[node.name] = out_type

        val_name = f"%{node.name.replace('.', '_')}"
        self.arg_name_map[node.name] = val_name

        self.lines.append(
            f'{self.indent}{val_name} = "{torch_op}"'
            f"({', '.join(mlir_args)}) : "
            f"({', '.join(mlir_types)}) -> {out_type}"
        )
        return val_name


# ──────────────────────────────────────────────────────────────────────
# Constant Folding
# ──────────────────────────────────────────────────────────────────────


def _find_runtime_dependent(graph, weight_names: set) -> set:
    """Find nodes that transitively depend on runtime inputs."""
    runtime_deps = set()
    for node in graph.nodes:
        if node.op == "placeholder" and node.name not in weight_names:
            runtime_deps.add(node.name)

    changed = True
    while changed:
        changed = False
        for node in graph.nodes:
            if node.name in runtime_deps or node.op != "call_function":
                continue
            for arg in node.args:
                if hasattr(arg, "name") and arg.name in runtime_deps:
                    runtime_deps.add(node.name)
                    changed = True
                    break
                if isinstance(arg, (list, tuple)):
                    for elem in arg:
                        if hasattr(elem, "name") and elem.name in runtime_deps:
                            runtime_deps.add(node.name)
                            changed = True
                            break
    return runtime_deps


# ──────────────────────────────────────────────────────────────────────
# Op Name Resolution
# ──────────────────────────────────────────────────────────────────────

# Ops that are no-ops (pass through input unchanged)
_PASSTHROUGH_OPS = {"alias", "clone", "_to_copy"}

# Ops to skip entirely (no compute, no output)
_SKIP_OPS = {"_assert_tensor_metadata"}


def _resolve_torch_op(target_str: str, full_name: str) -> Optional[str]:
    """Resolve FX target to torch.aten.* op name.

    Returns None for ops that should be skipped.
    """
    if any(s in target_str for s in _SKIP_OPS):
        return None

    if "aten." in target_str:
        parts = target_str.split(".")
        aten_idx = parts.index("aten")
        op_name = parts[aten_idx + 1]
        if aten_idx + 2 < len(parts) and parts[aten_idx + 2] != "default":
            op_name += "." + parts[aten_idx + 2]
        return "torch.aten." + op_name

    return f"torch.unknown.{full_name}"


# ──────────────────────────────────────────────────────────────────────
# Main Emitter
# ──────────────────────────────────────────────────────────────────────


def fx_graph_to_mlir(ep: ExportedProgram, decompose: bool = True) -> str:
    """Convert a torch.export ExportedProgram to Torch dialect MLIR text.

    Args:
        ep: The exported program from torch.export.export()
        decompose: If True, run decompositions to inline submodules
    """
    if decompose:
        ep = ep.run_decompositions()

    gm = ep.graph_module
    graph = gm.graph
    ctx = EmitterContext()

    # ── Classify placeholders ──────────────────────────────────────
    weight_nodes, input_nodes = [], []
    linear_weight_names = set()

    for node in graph.nodes:
        if node.op == "placeholder":
            if _is_weight_placeholder(node):
                weight_nodes.append(node)
            else:
                input_nodes.append(node)
        if node.op == "call_function" and "aten.linear" in str(node.target):
            if len(node.args) >= 2 and hasattr(node.args[1], "name"):
                linear_weight_names.add(node.args[1].name)

    all_params = weight_nodes + input_nodes
    weight_name_set = {n.name for n in weight_nodes}

    # ── Constant folding ──────────────────────────────────────────
    runtime_deps = _find_runtime_dependent(graph, weight_name_set)

    const_tensor_nodes = []
    for node in graph.nodes:
        if node.op == "call_function" and node.name not in runtime_deps:
            val = node.meta.get("val", None)
            if isinstance(val, torch.Tensor):
                const_tensor_nodes.append(node)

    # ── Build function signature ──────────────────────────────────
    arg_strs = []
    for i, node in enumerate(all_params):
        transposed = node.name in linear_weight_names
        ttype = ctx.get_tensor_type(node, transposed=transposed)
        if ttype is None:
            raise ValueError(f"No type info for placeholder: {node.name}")
        ctx.type_map[node.name] = ttype
        ctx.arg_name_map[node.name] = f"%arg{i}"
        arg_strs.append(f"%arg{i}: {ttype}")

    for j, node in enumerate(const_tensor_nodes):
        idx = len(all_params) + j
        ttype = ctx.get_tensor_type(node)
        if ttype:
            ctx.type_map[node.name] = ttype
            ctx.arg_name_map[node.name] = f"%arg{idx}"
            arg_strs.append(f"%arg{idx}: {ttype}")

    # ── Find output ──────────────────────────────────────────────
    output_node = next(n for n in graph.nodes if n.op == "output")
    return_vals = output_node.args[0]
    if isinstance(return_vals, tuple):
        return_vals = list(return_vals)
    if not isinstance(return_vals, list):
        return_vals = [return_vals]

    return_types = []
    for rv in return_vals:
        if hasattr(rv, "name"):
            rt = ctx.get_tensor_type(rv)
            if rt:
                return_types.append(rt)

    # ── Emit ops ─────────────────────────────────────────────────
    for node in graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        if node.op != "call_function":
            continue

        # Skip constant (non-runtime) nodes
        if node.name not in runtime_deps:
            continue

        target_str = str(node.target)
        full_name = getattr(node.target, "__name__", target_str)

        # Resolve op name
        torch_op = _resolve_torch_op(target_str, full_name)
        if torch_op is None:
            continue

        # Handle passthrough ops (alias, clone)
        short_name = full_name.split(".")[-1] if "." in full_name else full_name
        if short_name in _PASSTHROUGH_OPS:
            if node.args and hasattr(node.args[0], "name"):
                ref = ctx.arg_name_map.get(node.args[0].name, f"%{node.args[0].name}")
                ctx.arg_name_map[node.name] = ref
                ttype = ctx.get_tensor_type(node)
                if ttype:
                    ctx.type_map[node.name] = ttype
            continue

        # Pad optional args from registry defaults
        padded_args = ctx.pad_args(node, torch_op)

        # Emit the op
        ctx.format_op(torch_op, node, padded_args)

    # ── Assemble module ──────────────────────────────────────────
    sig_args = ", ".join(arg_strs)
    sig_returns = ", ".join(return_types)
    if len(return_types) > 1:
        sig_returns = f"({sig_returns})"

    result = [
        "module {",
        f"  func.func @main_graph({sig_args}) -> {sig_returns} {{",
    ]
    result.extend(ctx.lines)

    ret_refs = []
    for rv in return_vals:
        if hasattr(rv, "name"):
            ref = ctx.arg_name_map.get(rv.name, f"%{rv.name}")
            ret_refs.append(ref)

    result.append(f"    return {', '.join(ret_refs)} : {', '.join(return_types)}")
    result.append("  }")
    result.append("}")
    result.append("")

    return "\n".join(result)
