#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
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
from typing import Any, Dict, List, Optional


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

    def _get_tensor_type(self, node, transposed: bool = False) -> Optional[str]:
        """Get the MLIR tensor type for a node from its metadata."""
        val = node.meta.get("val", None)
        if isinstance(val, torch.Tensor):
            shape = list(val.shape)
            if transposed and len(shape) == 2:
                shape = [shape[1], shape[0]]
            return _tensor_type_str(shape, val.dtype)
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

        if isinstance(arg, bool):
            # Emit as torch.constant.int (0 or 1) rather than torch.constant.bool
            # because most torch ops expect !torch.int for scalar args.
            # torch.aten.full, where, etc. use int representation for bools.
            int_val = 1 if arg else 0
            name = self._next_const_name(f"i{int_val}")
            self.lines.append(
                f'{self.indent}{name} = "torch.constant.int"() '
                f"{{value = {int_val} : i64}} : () -> !torch.int"
            )
            self._const_cache[cache_key] = name
            return name
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
            import math

            if math.isinf(arg):
                val_str = (
                    "0x7FF0000000000000 : f64"
                    if arg > 0
                    else "0xFFF0000000000000 : f64"
                )
            elif math.isnan(arg):
                val_str = "0x7FF8000000000000 : f64"
            else:
                val_str = f"{arg:.6e} : f64"
            self.lines.append(
                f'{self.indent}{name} = "torch.constant.float"() '
                f"{{value = {val_str}}} : () -> !torch.float"
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
                        t = self.type_map.get(
                            elem.name, self._get_tensor_type(elem) or "tensor<*xf16>"
                        )
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

    def _find_constant_nodes(self) -> set:
        """Find nodes that can be folded at compile time.

        A node is constant if all its inputs are either:
        - Placeholders that are weights (parameters)
        - Literal constants (int, float, bool, None, list)
        - Other constant nodes

        These include position index computation, mask construction,
        and RoPE cos/sin cache — all of which depend only on shapes
        and model config, not runtime inputs.
        """
        constant_nodes = set()
        weight_names = set()

        for node in self.graph.nodes:
            if node.op == "placeholder" and _is_weight_placeholder(node):
                weight_names.add(node.name)

        # Iterate until fixpoint
        changed = True
        while changed:
            changed = False
            for node in self.graph.nodes:
                if node.name in constant_nodes:
                    continue
                if node.op != "call_function":
                    continue

                # Check if all args are constants or weight refs
                all_const = True
                for arg in node.args:
                    if hasattr(arg, "name"):
                        if (
                            arg.name not in constant_nodes
                            and arg.name not in weight_names
                            and arg.op != "placeholder"
                        ):
                            all_const = False
                            break
                    # Literals are always constant

                if all_const:
                    # Also check kwargs
                    for v in node.kwargs.values():
                        if hasattr(v, "name") and v.name not in constant_nodes:
                            all_const = False
                            break

                if all_const:
                    # Don't fold large compute ops even if inputs are constant
                    target = str(node.target)
                    if any(
                        k in target
                        for k in ["mm", "matmul", "bmm", "softmax", "embedding"]
                    ):
                        continue
                    constant_nodes.add(node.name)
                    changed = True

        return constant_nodes

    def _find_runtime_dependent(self) -> set:
        """Find nodes that depend (transitively) on runtime inputs."""
        runtime_deps = set()
        for node in self.graph.nodes:
            if node.op == "placeholder" and not _is_weight_placeholder(node):
                runtime_deps.add(node.name)

        # Forward propagation
        changed = True
        while changed:
            changed = False
            for node in self.graph.nodes:
                if node.name in runtime_deps:
                    continue
                if node.op != "call_function":
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

    def emit(self) -> str:
        """Generate MLIR text for the full module."""

        # Find which nodes depend on runtime inputs vs are pure constants
        runtime_deps = self._find_runtime_dependent()

        # Pre-scan: find weight placeholders that feed torch.aten.linear
        # (before decomposition). After run_decompositions(), linear is
        # decomposed to mm with the weight already transposed via permute,
        # so no pre-transposition is needed in decomposed mode.
        linear_weight_names = set()
        for node in self.graph.nodes:
            if node.op == "call_function":
                target_str = str(node.target)
                if "aten.linear" in target_str and len(node.args) >= 2:
                    weight_arg = node.args[1]
                    if hasattr(weight_arg, "name"):
                        linear_weight_names.add(weight_arg.name)

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

        # Pre-scan: find constant tensor nodes (not runtime-dependent)
        # These will become additional function args after the placeholders
        const_tensor_nodes = []
        for node in self.graph.nodes:
            if node.op == "call_function" and node.name not in runtime_deps:
                val = node.meta.get("val", None)
                if isinstance(val, torch.Tensor):
                    const_tensor_nodes.append(node)

        # Build function signature
        # Linear weights are emitted with transposed shape [K,N] instead of [N,K]
        arg_strs = []
        for i, node in enumerate(all_params):
            needs_transpose = node.name in linear_weight_names
            ttype = self._get_tensor_type(node, transposed=needs_transpose)
            if ttype is None:
                raise ValueError(f"No type info for placeholder: {node.name}")
            self.type_map[node.name] = ttype
            arg_strs.append(f"%arg{i}: {ttype}")

        # Add constant tensor nodes as extra function args
        for j, node in enumerate(const_tensor_nodes):
            idx = len(all_params) + j
            ttype = self._get_tensor_type(node)
            if ttype:
                self.type_map[node.name] = ttype
                arg_strs.append(f"%arg{idx}: {ttype}")

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
        # Map constant tensor nodes to their arg indices
        for j, node in enumerate(const_tensor_nodes):
            idx = len(all_params) + j
            self.arg_name_map[node.name] = f"%arg{idx}"
        arg_name_map = self.arg_name_map

        # Emit operations
        self.lines = []

        for node in self.graph.nodes:
            if node.op in ("placeholder", "output"):
                continue

            if node.op == "call_function":
                op_name = str(node.target).split(".")[-1]  # e.g. "default"
                full_name = str(node.target)
                if hasattr(node.target, "__name__"):
                    full_name = node.target.__name__

                target_str = str(node.target)

                # Skip constant nodes — they're already mapped to function args
                if node.name not in runtime_deps:
                    continue

                # Skip framework ops that have no compute semantics
                if "_assert_tensor_metadata" in target_str:
                    # Assertion op — skip entirely
                    continue
                if "alias" in full_name and "aten.alias" in target_str:
                    # alias is a no-op — forward the input
                    if node.args and hasattr(node.args[0], "name"):
                        ref = arg_name_map.get(
                            node.args[0].name, _value_name(node.args[0])
                        )
                        arg_name_map[node.name] = ref
                        ttype = self._get_tensor_type(node)
                        if ttype:
                            self.type_map[node.name] = ttype
                    continue
                if "clone" in full_name and "aten.clone" in target_str:
                    # clone is typically a no-op in inference
                    if node.args and hasattr(node.args[0], "name"):
                        ref = arg_name_map.get(
                            node.args[0].name, _value_name(node.args[0])
                        )
                        arg_name_map[node.name] = ref
                        ttype = self._get_tensor_type(node)
                        if ttype:
                            self.type_map[node.name] = ttype
                    continue

                # Resolve the torch.aten.* op name
                if "aten." in target_str:
                    # Extract: aten.mul.Tensor -> torch.aten.mul.Tensor
                    #          aten.rms_norm.default -> torch.aten.rms_norm
                    parts = target_str.split(".")
                    aten_idx = parts.index("aten")
                    op_name = parts[aten_idx + 1]
                    # Include overload if it's not "default"
                    if aten_idx + 2 < len(parts) and parts[aten_idx + 2] != "default":
                        op_name += "." + parts[aten_idx + 2]
                    torch_op = "torch.aten." + op_name
                else:
                    torch_op = f"torch.unknown.{full_name}"

                # Known ops that need trailing None args when torch.export omits them
                # Known ops that need specific trailing args when torch.export omits them
                _OPTIONAL_TRAILING_DEFAULTS = {
                    "torch.aten.linear": [(None,)],  # bias=None
                    "torch.aten.add.Tensor": [(1,)],  # alpha=1
                    "torch.aten.sub.Tensor": [(1,)],  # alpha=1
                }

                # Pad args with defaults for known ops
                padded_args = list(node.args)
                defaults = _OPTIONAL_TRAILING_DEFAULTS.get(torch_op, [])
                if defaults:
                    expected_count = len(padded_args) + len(defaults)
                    while len(padded_args) < expected_count:
                        idx = len(padded_args) - (expected_count - len(defaults))
                        if 0 <= idx < len(defaults):
                            padded_args.append(defaults[idx][0])
                        else:
                            padded_args.append(None)

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


def _evaluate_constant_subgraph(ep: ExportedProgram) -> Dict[str, torch.Tensor]:
    """Evaluate nodes that depend only on weights/constants, not runtime inputs.

    Returns a dict mapping node name -> evaluated tensor value.
    These nodes will be emitted as constant function args instead of ops.
    """
    gm = ep.graph_module

    # Identify runtime input placeholders (non-weight)
    runtime_inputs = set()
    for node in gm.graph.nodes:
        if node.op == "placeholder" and not _is_weight_placeholder(node):
            runtime_inputs.add(node.name)

    # Mark nodes that transitively depend on runtime inputs
    depends_on_runtime = set()
    for node in gm.graph.nodes:
        if node.name in runtime_inputs:
            depends_on_runtime.add(node.name)
            continue
        if node.op == "call_function":
            for arg in node.args:
                if hasattr(arg, "name") and arg.name in depends_on_runtime:
                    depends_on_runtime.add(node.name)
                    break
            else:
                for arg in node.args:
                    if isinstance(arg, (list, tuple)):
                        for elem in arg:
                            if (
                                hasattr(elem, "name")
                                and elem.name in depends_on_runtime
                            ):
                                depends_on_runtime.add(node.name)
                                break

    # Evaluate constant nodes by running the graph with real weights
    # and dummy runtime inputs
    constant_results = {}
    state_dict = ep.state_dict if hasattr(ep, "state_dict") else {}

    # Create input dict with real weights
    input_dict = {}
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            val = node.meta.get("val", None)
            if _is_weight_placeholder(node):
                # Use actual weight value
                # Map placeholder name to state dict key
                for key, param in state_dict.items():
                    if (
                        node.target.replace("p_", "").replace("_", ".")
                        in key.replace(".", "_")
                        or node.target == f"p_{key.replace('.', '_')}"
                    ):
                        input_dict[node.name] = param
                        break
                else:
                    # Fallback: use meta val (shape info only)
                    if isinstance(val, torch.Tensor):
                        input_dict[node.name] = torch.zeros_like(val)

    # For non-runtime-dependent nodes, try to evaluate
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.name not in depends_on_runtime:
            val = node.meta.get("val", None)
            if isinstance(val, torch.Tensor):
                constant_results[node.name] = val

    return constant_results


def fx_graph_to_mlir(ep: ExportedProgram, decompose: bool = True) -> str:
    """Convert a torch.export ExportedProgram to Torch dialect MLIR text.

    Args:
        ep: The exported program from torch.export.export()
        decompose: If True, run decompositions to inline submodules and
                   decompose higher-level ops (linear→mm, silu→sigmoid*x,
                   sdpa→bmm+softmax, rms_norm→pow+mean+rsqrt+mul).
                   This eliminates framework ops like wrap_with_set_grad_enabled.
    """
    if decompose:
        ep = ep.run_decompositions()
    emitter = FxToMlirEmitter(ep)
    return emitter.emit()
