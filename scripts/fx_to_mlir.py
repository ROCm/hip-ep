#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""
Convert a torch.export ExportedProgram's FX graph to Torch dialect MLIR text.

The output MLIR uses generic-form torch.aten.* ops with standard MLIR tensor
types (post type-conversion), matching the format expected by the
convert-torch-to-hip pass in hip-mlir-opt / hip-compiler.

Usage:
    from fx_to_mlir import fx_graph_to_mlir
    ep = torch.export.export(model, example_inputs)
    mlir_text = fx_graph_to_mlir(ep)
"""

import torch
from torch.export import ExportedProgram
from typing import Any, Dict, List, Optional, Tuple


def _dtype_to_mlir(dtype: torch.dtype) -> str:
    """Convert a PyTorch dtype to an MLIR type string."""
    mapping = {
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
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def _tensor_type_str(shape: List[int], dtype: torch.dtype) -> str:
    """Build an MLIR tensor type string like 'tensor<1x4x64xf16>'."""
    if len(shape) == 0:
        return f"tensor<{_dtype_to_mlir(dtype)}>"  # scalar tensor
    dims = "x".join(str(d) for d in shape)
    return f"tensor<{dims}x{_dtype_to_mlir(dtype)}>"


def _value_name(node) -> str:
    """Get the SSA value name for a graph node."""
    return f"%{node.name.replace('.', '_')}"


def _format_attr_value(val: Any) -> str:
    """Format a Python value as an MLIR attribute."""
    if isinstance(val, bool):
        return f"{str(val).lower()}"
    if isinstance(val, int):
        return f"{val} : i64"
    if isinstance(val, float):
        # Use scientific notation for very small values
        if val == 0.0:
            return "0.000000e+00 : f64"
        return f"{val:.6e} : f64"
    if isinstance(val, str):
        return f'"{val}"'
    if val is None:
        return "none"
    return str(val)


def _is_weight_placeholder(node) -> bool:
    """Check if a placeholder node is a weight (parameter) vs input."""
    return node.target.startswith("p_") or node.target.startswith("b_")


class FxToMlirEmitter:
    """Converts an FX graph to Torch dialect MLIR text."""

    def __init__(self, ep: ExportedProgram):
        self.ep = ep
        self.gm = ep.graph_module
        self.graph = self.gm.graph
        self.lines: List[str] = []
        self.indent = "    "
        # Map from node name to its MLIR type string
        self.type_map: Dict[str, str] = {}
        # Counter for unique constant names
        self._const_counter = 0
        # Cache: avoid re-emitting identical constants
        self._const_cache: Dict[Any, str] = {}

    def _get_tensor_type(self, node) -> Optional[str]:
        """Get the MLIR tensor type for a node from its metadata."""
        val = node.meta.get("val", None)
        if isinstance(val, torch.Tensor):
            return _tensor_type_str(list(val.shape), val.dtype)
        if isinstance(val, torch.SymInt):
            return "!torch.int"
        if isinstance(val, torch.SymBool):
            return "!torch.bool"
        if val is None:
            return "!torch.none"
        return None

    def _get_value_ref(self, arg) -> str:
        """Get the MLIR value reference for a function argument."""
        if hasattr(arg, "name"):
            return _value_name(arg)
        if isinstance(arg, (int, float, bool)):
            return str(arg)
        if arg is None:
            return "none"
        if isinstance(arg, (list, tuple)):
            return str(list(arg))
        return repr(arg)

    def _next_const_name(self, prefix: str) -> str:
        self._const_counter += 1
        return f"%_k{self._const_counter}_{prefix}"

    def _emit_constant_for_arg(self, arg, arg_type: str) -> str:
        """Emit a torch.constant.* op for a literal argument and return its SSA name."""
        # Cache scalar constants to avoid duplicates
        cache_key = (type(arg).__name__, arg)
        if not isinstance(arg, (list, tuple)) and cache_key in self._const_cache:
            return self._const_cache[cache_key]

        if isinstance(arg, int):
            name = self._next_const_name(f"i{arg}" if arg >= 0 else f"in{abs(arg)}")
            self.lines.append(
                f'{self.indent}{name} = "torch.constant.int"() '
                f"{{value = {arg} : i64}} : () -> !torch.int"
            )
            self._const_cache[cache_key] = name
            return name
        if isinstance(arg, float):
            name = self._next_const_name("f")
            self.lines.append(
                f'{self.indent}{name} = "torch.constant.float"() '
                f"{{value = {arg:.6e} : f64}} : () -> !torch.float"
            )
            self._const_cache[cache_key] = name
            return name
        if isinstance(arg, bool):
            name = self._next_const_name(f"b{int(arg)}")
            val_str = "true" if arg else "false"
            self.lines.append(
                f'{self.indent}{name} = "torch.constant.bool"() '
                f"{{value = {val_str}}} : () -> !torch.bool"
            )
            self._const_cache[cache_key] = name
            return name
        if arg is None:
            name = "%_none"
            if ("NoneType", None) not in self._const_cache:
                self.lines.append(
                    f'{self.indent}{name} = "torch.constant.none"() : () -> !torch.none'
                )
                self._const_cache[("NoneType", None)] = name
            return name
        if isinstance(arg, (list, tuple)):
            # Check if elements are node references (tensor list) vs constants
            has_nodes = any(hasattr(e, "name") for e in arg)
            if has_nodes:
                # Tensor list: emit torch.prim.ListConstruct with tensor refs
                elem_names = []
                elem_types = []
                for elem in arg:
                    if hasattr(elem, "name"):
                        ref = self.arg_name_map.get(elem.name, _value_name(elem))
                        elem_names.append(ref)
                        t = self.type_map.get(elem.name, self._get_tensor_type(elem) or "tensor<*xf16>")
                        elem_types.append(t)
                    else:
                        c = self._emit_constant_for_arg(elem, "")
                        elem_names.append(c)
                        elem_types.append("!torch.int")
                list_name = f"%_list_{id(arg)}"
                elem_refs = ", ".join(elem_names)
                types_str = ", ".join(elem_types)
                self.lines.append(
                    f'{self.indent}{list_name} = "torch.prim.ListConstruct"({elem_refs}) '
                    f": ({types_str}) -> !torch.list<tensor>"
                )
                return list_name
            else:
                # Integer/constant list
                elem_names = []
                for elem in arg:
                    elem_names.append(self._emit_constant_for_arg(elem, "i64"))
                list_name = f"%_list_{id(arg)}"
                elem_refs = ", ".join(elem_names)
                elem_types = ", ".join(["!torch.int"] * len(elem_names))
                self.lines.append(
                    f'{self.indent}{list_name} = "torch.prim.ListConstruct"({elem_refs}) '
                    f": ({elem_types}) -> !torch.list<int>"
                )
                return list_name
        raise ValueError(f"Cannot emit constant for: {type(arg).__name__} = {arg}")

    def emit(self) -> str:
        """Generate MLIR text for the full module."""
        # Collect placeholders (weights then inputs)
        weight_nodes = []
        input_nodes = []
        for node in self.graph.nodes:
            if node.op == "placeholder":
                if _is_weight_placeholder(node):
                    weight_nodes.append(node)
                else:
                    input_nodes.append(node)

        # All placeholders become function args (weights first, then inputs)
        all_params = weight_nodes + input_nodes

        # Build function signature
        arg_strs = []
        for i, node in enumerate(all_params):
            ttype = self._get_tensor_type(node)
            if ttype is None:
                raise ValueError(f"No type info for placeholder: {node.name}")
            self.type_map[node.name] = ttype
            arg_strs.append(f"%arg{i}: {ttype}")

        # Find output node and its type
        output_node = None
        for node in self.graph.nodes:
            if node.op == "output":
                output_node = node
                break

        # Extract return values (handle nested tuples from output)
        return_vals = output_node.args[0]
        if isinstance(return_vals, tuple):
            return_vals = list(return_vals)
        if not isinstance(return_vals, list):
            return_vals = [return_vals]

        return_types = []
        for rv in return_vals:
            if hasattr(rv, "name"):
                rt = self._get_tensor_type(rv)
                if rt:
                    return_types.append(rt)

        # Emit module
        result = []
        result.append("module {")

        sig_args = ", ".join(arg_strs)
        sig_returns = ", ".join(return_types)
        if len(return_types) > 1:
            sig_returns = f"({sig_returns})"
        result.append(f"  func.func @main_graph({sig_args}) -> {sig_returns} {{")

        # Map placeholder names to arg references
        self.arg_name_map = {}
        for i, node in enumerate(all_params):
            self.arg_name_map[node.name] = f"%arg{i}"
        arg_name_map = self.arg_name_map

        # Emit operations
        self.lines = []
        emitted_constants = set()

        for node in self.graph.nodes:
            if node.op in ("placeholder", "output"):
                continue

            if node.op == "call_function":
                op_name = str(node.target).split(".")[-1]  # e.g. "default"
                # Get full aten op name: aten.linear.default
                full_name = str(node.target)
                if hasattr(node.target, "__name__"):
                    full_name = node.target.__name__

                # Resolve the torch.aten.* op name
                target_str = str(node.target)
                if "aten." in target_str:
                    # Extract: aten.rms_norm.default -> torch.aten.rms_norm
                    parts = target_str.split(".")
                    aten_idx = parts.index("aten")
                    torch_op = "torch.aten." + parts[aten_idx + 1]
                else:
                    torch_op = f"torch.unknown.{full_name}"

                # Known ops that need trailing None args when torch.export omits them
                _OPTIONAL_TRAILING_NONES = {
                    "torch.aten.linear": 3,       # input, weight, bias
                    "torch.aten.add": 3,           # self, other, alpha
                    "torch.aten.sub": 3,           # self, other, alpha
                }

                # Pad args with None for known ops
                padded_args = list(node.args)
                expected_count = _OPTIONAL_TRAILING_NONES.get(torch_op)
                if expected_count and len(padded_args) < expected_count:
                    padded_args.extend([None] * (expected_count - len(padded_args)))

                # Resolve argument references
                mlir_args = []
                mlir_arg_types = []

                for arg in padded_args:
                    if hasattr(arg, "name"):
                        # SSA reference to another node
                        ref = arg_name_map.get(arg.name, _value_name(arg))
                        atype = self.type_map.get(
                            arg.name, self._get_tensor_type(arg) or "!torch.unknown"
                        )
                        mlir_args.append(ref)
                        mlir_arg_types.append(atype)
                    elif isinstance(arg, (int, float, bool)) or arg is None:
                        # Literal -> emit torch.constant.* and reference it
                        const_name = self._emit_constant_for_arg(arg, "")
                        mlir_args.append(const_name)
                        if isinstance(arg, int):
                            mlir_arg_types.append("!torch.int")
                        elif isinstance(arg, float):
                            mlir_arg_types.append("!torch.float")
                        elif isinstance(arg, bool):
                            mlir_arg_types.append("!torch.bool")
                        elif arg is None:
                            mlir_arg_types.append("!torch.none")
                    elif isinstance(arg, (list, tuple)):
                        const_name = self._emit_constant_for_arg(arg, "")
                        mlir_args.append(const_name)
                        # Determine list type from contents
                        has_nodes = any(hasattr(e, "name") for e in arg)
                        mlir_arg_types.append(
                            "!torch.list<tensor>" if has_nodes else "!torch.list<int>"
                        )
                    else:
                        mlir_args.append(f"/*{type(arg).__name__}*/")
                        mlir_arg_types.append("!torch.unknown")

                # Output type
                out_type = self._get_tensor_type(node) or "!torch.unknown"
                self.type_map[node.name] = out_type

                # Emit the op
                val_name = _value_name(node)
                args_str = ", ".join(mlir_args)
                types_str = ", ".join(mlir_arg_types)

                self.lines.append(
                    f'{self.indent}{val_name} = "{torch_op}"({args_str}) : '
                    f"({types_str}) -> {out_type}"
                )

                # Register in arg_name_map for downstream references
                arg_name_map[node.name] = val_name

            elif node.op == "get_attr":
                # Constant attribute (rare in torch.export)
                pass

        # Add lines to result
        result.extend(self.lines)

        # Emit return
        ret_refs = []
        for rv in return_vals:
            if hasattr(rv, "name"):
                ref = arg_name_map.get(rv.name, _value_name(rv))
                ret_refs.append(ref)
        ret_str = ", ".join(ret_refs)
        ret_type_str = ", ".join(return_types)
        result.append(f"    return {ret_str} : {ret_type_str}")
        result.append("  }")
        result.append("}")
        result.append("")

        return "\n".join(result)


def fx_graph_to_mlir(ep: ExportedProgram) -> str:
    """Convert a torch.export ExportedProgram to Torch dialect MLIR text."""
    emitter = FxToMlirEmitter(ep)
    return emitter.emit()
