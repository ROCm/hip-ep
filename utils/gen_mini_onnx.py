#!/usr/bin/env python3
"""
Generate MiniONNX dialect from ONNX schemas.
Supports operations with subgraph regions (If/Loop/Scan).
"""

import argparse
import sys
from collections import OrderedDict
from typing import Text, Optional, Tuple, List

try:
    from onnx import defs
    from onnx.defs import OpSchema
except ImportError:
    print("ERROR: onnx package not found. Install with: pip install onnx")
    sys.exit(1)


def inc_indent(indent=None):
    return "" if indent is None else indent + "  "


def onnx_attr_type_to_mlir_attr_type(t):
    onnx_attr_type = Text(t)
    onnx_attr_type = onnx_attr_type[onnx_attr_type.rfind(".") + 1:].lower()

    type_map = {
        "int": "SI64Attr",
        "float": "F32Attr",
        "ints": "I64ArrayAttr",
        "floats": "F32ArrayAttr",
        "string": "StrAttr",
        "strings": "StrArrayAttr",
        "type": "TypeAttr",
        "type_proto": "TypeAttr",
    }
    return type_map.get(onnx_attr_type, "AnyAttr")


def is_supported_type(type_str: str) -> bool:
    lower = type_str.lower()
    if "seq(" in lower:
        return False
    if "map(" in lower:
        return False
    if "optional(" in lower:
        return False
    if "string" in lower:
        return False
    if "complex" in lower:
        return False
    return True


def parse_type_str(allowed_type: str) -> Optional[str]:
    if not is_supported_type(allowed_type):
        return None

    onnx_to_mlir = {
        "(": "<[",
        ")": "]>",
        "tensor": "TensorOf",
        "bool": "I1",
        "uint8": "UI8",
        "uint16": "UI16",
        "uint32": "UI32",
        "uint64": "UI64",
        "uint4": "UI<4>",
        "int8": "I8",
        "int16": "I16",
        "int32": "I32",
        "int64": "I64",
        "int4": "I<4>",
        "float16": "F16",
        "float": "F32",
        "double": "F64",
        "bfloat16": "BF16",
        "float8e4m3fn": "F8E4M3FN",
        "float8e4m3fnuz": "F8E4M3FNUZ",
        "float8e5m2": "F8E5M2",
        "float8e5m2fnuz": "F8E5M2FNUZ",
    }

    result = allowed_type
    mapping = sorted(onnx_to_mlir.items(), key=lambda x: len(x[0]), reverse=True)
    for key, value in mapping:
        result = result.replace(key, value)
    return result


def can_generate_operation(schema) -> Tuple[bool, str]:
    # Check if operation has GRAPH attributes
    has_graph_attrs = any(attr.type == OpSchema.AttrType.GRAPH
                          for attr in schema.attributes.values())

    # For operations with GRAPH attributes, allow only If, Loop, Scan
    if has_graph_attrs:
        if schema.name not in ['If', 'Loop', 'Scan']:
            return False, f"has subgraph attributes (not If/Loop/Scan)"

    for inp in schema.inputs:
        if inp.type_str and not is_supported_type(inp.type_str):
            return False, f"unsupported input type: {inp.type_str}"
        for constraint in schema.type_constraints:
            if inp.type_str == constraint.type_param_str:
                if not any(is_supported_type(t) for t in constraint.allowed_type_strs):
                    return False, "all input types unsupported"

    for out in schema.outputs:
        if out.type_str and not is_supported_type(out.type_str):
            return False, f"unsupported output type: {out.type_str}"
        for constraint in schema.type_constraints:
            if out.type_str == constraint.type_param_str:
                if not any(is_supported_type(t) for t in constraint.allowed_type_strs):
                    return False, "all output types unsupported"

    return True, ""


def parse_type_constraints(schema):
    type_str_dict = {}
    for constraint in schema.type_constraints:
        mlir_types: List[str] = []
        for allowed_type in constraint.allowed_type_strs:
            mlir_type = parse_type_str(allowed_type)
            if mlir_type and mlir_type not in mlir_types:
                mlir_types.append(mlir_type)
        if mlir_types:
            type_str_dict[constraint.type_param_str] = mlir_types
    return type_str_dict


def get_onnx_mlir_types(schema, type_str_dict, io_value):
    if io_value.type_str:
        if io_value.type_str in type_str_dict:
            return type_str_dict[io_value.type_str]
        else:
            mlir_type = parse_type_str(io_value.type_str)
            if mlir_type:
                return [mlir_type]
    return []


def get_operands_or_results(schema, type_str_dict, is_input):
    value_list = schema.inputs if is_input else schema.outputs
    if not value_list:
        return OrderedDict()

    def any_type_of(types):
        if len(types) == 1:
            return types[0]
        return f"AnyTypeOf<[{', '.join(types)}]>"

    name_to_types = OrderedDict()
    for value in value_list:
        types = get_onnx_mlir_types(schema, type_str_dict, value)

        if not types:
            types = ["AnyTensor"]

        if OpSchema.FormalParameterOption.Optional == value.option:
            types.append("NoneType")

        if OpSchema.FormalParameterOption.Variadic == value.option:
            types = [f"Variadic<{any_type_of(types)}>"]

        if is_input:
            value_name = value.name
        else:
            value_name = value.name
            for inp in schema.inputs:
                if inp.name == value.name:
                    value_name = "out_" + value.name
                    break

        name_to_types[value_name] = any_type_of(types)

    return name_to_types


def get_attrs(schema):
    """Get non-GRAPH attributes"""
    if not schema.attributes:
        return OrderedDict()

    name_to_type = OrderedDict()
    for _, attr in sorted(schema.attributes.items()):
        # Skip GRAPH attributes - they become regions
        if attr.type == OpSchema.AttrType.GRAPH:
            continue

        mlir_type = onnx_attr_type_to_mlir_attr_type(attr.type)

        if attr.required:
            name_to_type[attr.name] = mlir_type
        else:
            # All non-required attributes use OptionalAttr
            name_to_type[attr.name] = f"OptionalAttr<{mlir_type}>"

    return name_to_type


def get_regions(schema):
    """Get GRAPH attributes as regions"""
    regions = OrderedDict()
    for attr_name, attr in sorted(schema.attributes.items()):
        if attr.type == OpSchema.AttrType.GRAPH:
            # GRAPH attributes become regions with size 1 (one block)
            regions[attr_name] = "AnyRegion"
    return regions


def gen_op_def(schema):
    indent = inc_indent()
    op_name = schema.name

    s = f'def MiniONNX_{op_name}Op : MiniONNX_Op<"{op_name}",\n'

    traits = ["Pure"]
    s += inc_indent(indent) + f"[{', '.join(traits)}]> {{\n"

    indent = inc_indent(indent)

    s += indent + f'let summary = "ONNX {schema.name} operation";\n'

    s += indent + "let description = [{\n"
    if schema.doc:
        for line in schema.doc.lstrip().splitlines()[:5]:
            escaped = line.replace('"', '\\"').replace("}]", "\\}\\]")
            s += indent + f"{escaped}\n"
    s += indent + "}];\n"

    type_str_dict = parse_type_constraints(schema)

    # Get inputs and non-GRAPH attributes
    ins = get_operands_or_results(schema, type_str_dict, is_input=True)
    ins.update(get_attrs(schema))

    if ins:
        ins_strs = [f"{ty}:${name}" for name, ty in ins.items()]
        s += indent + "let arguments = (ins\n"
        s += inc_indent(indent) + (",\n" + inc_indent(indent)).join(ins_strs)
        s += "\n" + indent + ");\n"
    else:
        s += indent + "let arguments = (ins);\n"

    # Get outputs
    outs = get_operands_or_results(schema, type_str_dict, is_input=False)

    if outs:
        outs_strs = [f"{ty}:${name}" for name, ty in outs.items()]
        s += indent + "let results = (outs\n"
        s += inc_indent(indent) + (",\n" + inc_indent(indent)).join(outs_strs)
        s += "\n" + indent + ");\n"
    else:
        s += indent + "let results = (outs);\n"

    # Get GRAPH attributes as regions
    regions = get_regions(schema)
    if regions:
        region_strs = [f"{ty}:${name}" for name, ty in regions.items()]
        s += indent + "let regions = (region\n"
        s += inc_indent(indent) + (",\n" + inc_indent(indent)).join(region_strs)
        s += "\n" + indent + ");\n"

    # Add assembly format
    # This enables TableGen to auto-generate all properties infrastructure
    if regions:
        # Operations with regions: include region names in assembly format
        # For multiple regions, list them explicitly
        region_names = list(regions.keys())
        if len(region_names) == 1:
            # Single region: $region_name
            s += indent + f'let assemblyFormat = "operands attr-dict-with-keyword ${region_names[0]} `:` functional-type(operands, results)";\n'
        else:
            # Multiple regions: list them all
            region_refs = ' '.join([f'${name}' for name in region_names])
            s += indent + f'let assemblyFormat = "operands attr-dict-with-keyword {region_refs} `:` functional-type(operands, results)";\n'
    else:
        # Standard format for operations without regions
        s += indent + 'let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";\n'

    s += "}\n\n"
    return s


def get_schema(op_name, domain=""):
    schemas = defs.get_all_schemas_with_history()
    for schema in reversed(schemas):
        if schema.name == op_name and schema.domain == domain:
            return schema
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate MiniONNX dialect from ONNX schemas"
    )
    parser.add_argument(
        "--ops",
        required=True,
        help="Comma-separated list of ONNX operations or 'all'"
    )
    parser.add_argument(
        "--skip",
        default="",
        help="Comma-separated list of operations to skip"
    )
    parser.add_argument(
        "--output",
        default="MiniONNXOps.td.inc",
        help="Output file"
    )

    args = parser.parse_args()

    skip_set = set(name.strip() for name in args.skip.split(",") if name.strip())

    if args.ops.lower() == "all":
        schemas = defs.get_all_schemas_with_history()
        all_ops = sorted(set(s.name for s in schemas if s.domain == ""))

        op_names: List[str] = []
        skipped_filter: List[Tuple[str, str]] = []
        skipped_manual: List[str] = []

        for op_name in all_ops:
            if op_name in skip_set:
                skipped_manual.append(op_name)
                continue

            schema = get_schema(op_name)
            can_gen, reason = can_generate_operation(schema)
            if can_gen:
                op_names.append(op_name)
            else:
                skipped_filter.append((op_name, reason))

        print(f"Generating {len(op_names)}/{len(all_ops)} operations")
        if skipped_manual:
            print(f"\nManually skipped ({len(skipped_manual)}):")
            for name in skipped_manual:
                print(f"  - {name}")
        if skipped_filter:
            print(f"\nFiltered out ({len(skipped_filter)}):")
            for name, reason in skipped_filter:
                print(f"  - {name}: {reason}")
    else:
        op_names = [name.strip() for name in args.ops.split(",") if name.strip() not in skip_set]

    header = """//********************************************************
//   MiniONNX Operation Definitions
//   Auto-generated from ONNX schemas
//
//   To regenerate: python3 gen_mini_onnx.py --ops all
//********************************************************

"""

    with open(args.output, "w") as f:
        f.write(header)

        for op_name in op_names:
            schema = get_schema(op_name)
            if not schema:
                print(f"WARNING: Operation {op_name} not found", file=sys.stderr)
                continue

            op_def = gen_op_def(schema)
            f.write(op_def)

    print(f"\n✅ Generated {len(op_names)} operations to {args.output}")


if __name__ == "__main__":
    main()
