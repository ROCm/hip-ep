#!/usr/bin/env python3
"""
Generate MiniONNX dialect from ONNX schemas - FIXED version.
"""

import argparse
import sys
from collections import OrderedDict
from typing import Text

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


def parse_type_str(allowed_type):
    """Convert ONNX type string to MLIR TableGen type."""
    # Skip unsupported types
    if "string" in allowed_type.lower():
        return None  # Skip string types
    if "complex" in allowed_type.lower():
        return None  # Skip complex types

    onnx_to_mlir = {
        "(": "<[",
        ")": "]>",
        "tensor": "TensorOf",
        "seq": "SeqOf",
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

    # Apply substitutions in decreasing order of key length
    mapping = sorted(onnx_to_mlir.items(), key=lambda x: len(x[0]), reverse=True)
    for key, value in mapping:
        allowed_type = allowed_type.replace(key, value)
    return allowed_type


def parse_type_constraints(schema):
    """Parse type constraints from ONNX schema."""
    type_str_dict = {}
    for constraint in schema.type_constraints:
        mlir_types = []
        for allowed_type in constraint.allowed_type_strs:
            mlir_type = parse_type_str(allowed_type)
            if mlir_type and mlir_type not in mlir_types:
                mlir_types.append(mlir_type)
        if mlir_types:  # Only add if we have valid types
            type_str_dict[constraint.type_param_str] = mlir_types
    return type_str_dict


def get_onnx_mlir_types(schema, type_str_dict, io_value):
    """Get MLIR types for an input or output."""
    if io_value.type_str:
        if io_value.type_str in type_str_dict:
            return type_str_dict[io_value.type_str]
        else:
            # Direct type specification
            mlir_type = parse_type_str(io_value.type_str)
            if mlir_type:
                return [mlir_type]
    return []


def get_operands_or_results(schema, type_str_dict, is_input):
    """Generate operands or results dictionary."""
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

        # Handle optional
        if OpSchema.FormalParameterOption.Optional == value.option:
            types.append("NoneType")

        # Handle variadic
        if OpSchema.FormalParameterOption.Variadic == value.option:
            types = [f"Variadic<{any_type_of(types)}>"]

        # Avoid name conflicts between inputs and outputs
        if is_input:
            value_name = value.name
        else:
            # Check if output name conflicts with input
            value_name = value.name
            for inp in schema.inputs:
                if inp.name == value.name:
                    value_name = "out_" + value.name
                    break

        name_to_types[value_name] = any_type_of(types)

    return name_to_types


def get_attrs(schema):
    """Generate attributes dictionary."""
    if not schema.attributes:
        return OrderedDict()

    name_to_type = OrderedDict()
    for _, attr in sorted(schema.attributes.items()):
        # Skip graph attributes (subgraphs not supported in MiniONNX)
        if attr.type == OpSchema.AttrType.GRAPH:
            continue

        mlir_type = onnx_attr_type_to_mlir_attr_type(attr.type)

        if attr.required:
            name_to_type[attr.name] = mlir_type
        elif attr.default_value.name:
            # Has default value - simplify to just optional for now
            name_to_type[attr.name] = f"OptionalAttr<{mlir_type}>"
        else:
            # Optional
            name_to_type[attr.name] = f"OptionalAttr<{mlir_type}>"

    return name_to_type


def gen_op_def(schema):
    """Generate TableGen operation definition."""
    indent = inc_indent()
    op_name = schema.name

    s = f'def MiniONNX_{op_name}Op : MiniONNX_Op<"{op_name}",\n'

    # Traits
    traits = ["Pure"]
    s += inc_indent(indent) + f"[{', '.join(traits)}]> {{\n"

    indent = inc_indent(indent)

    # Summary
    s += indent + f'let summary = "ONNX {schema.name} operation";\n'

    # Description
    s += indent + "let description = [{\n"
    if schema.doc:
        for line in schema.doc.lstrip().splitlines()[:10]:  # Limit description length
            escaped = line.replace('"', '\\"').replace("}]", "\\}\\]")
            s += indent + f"{escaped}\n"
    s += indent + "}];\n"

    # Parse type constraints
    type_str_dict = parse_type_constraints(schema)

    # Arguments (inputs + attributes)
    ins = get_operands_or_results(schema, type_str_dict, is_input=True)
    ins.update(get_attrs(schema))

    if ins:
        ins_strs = [f"{ty}:${name}" for name, ty in ins.items()]
        s += indent + "let arguments = (ins\n"
        s += inc_indent(indent) + (",\n" + inc_indent(indent)).join(ins_strs)
        s += "\n" + indent + ");\n"
    else:
        s += indent + "let arguments = (ins);\n"

    # Results (outputs)
    outs = get_operands_or_results(schema, type_str_dict, is_input=False)
    if outs:
        outs_strs = [f"{ty}:${name}" for name, ty in outs.items()]
        s += indent + "let results = (outs\n"
        s += inc_indent(indent) + (",\n" + inc_indent(indent)).join(outs_strs)
        s += "\n" + indent + ");\n"
    else:
        s += indent + "let results = (outs);\n"

    s += "}\n\n"
    return s


def get_schema(op_name, domain=""):
    """Get ONNX schema for an operation."""
    schemas = defs.get_all_schemas_with_history()
    # Find latest version of the operation
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
        help="Comma-separated list of ONNX operations (e.g., Cast,MatMul,Add)"
    )
    parser.add_argument(
        "--output",
        default="MiniONNXOps.td.inc",
        help="Output file (default: MiniONNXOps.td.inc)"
    )

    args = parser.parse_args()

    op_names = [name.strip() for name in args.ops.split(",")]

    header = """//********************************************************
//   MiniONNX Operation Definitions
//   Auto-generated from ONNX schemas
//
//   To regenerate: python3 gen_mini_onnx.py --ops <ops>
//********************************************************

"""

    with open(args.output, "w") as f:
        f.write(header)

        for op_name in op_names:
            schema = get_schema(op_name)
            if not schema:
                print(f"WARNING: Operation {op_name} not found in ONNX schemas", file=sys.stderr)
                continue

            print(f"Generating {op_name}...")
            op_def = gen_op_def(schema)
            f.write(op_def)

    print(f"\nGenerated {len(op_names)} operations to {args.output}")


if __name__ == "__main__":
    main()
