#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
import sys
import json
from pathlib import Path


def escape_string(s):
    """Escape a string for C++ code generation."""
    if not s:
        return '""'
    # Escape backslashes and quotes
    s = (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )
    return f'"{s}"'


def bool_to_string(b):
    return "true" if b else "false"


def enum_to_cxx_string(e, prefix):
    enum_name = e.rsplit(".", 1)[-1]
    return f"{prefix}::{enum_name}"


def generate_type_constraint(type_constraint):
    """Generate TypeConstraint code from JSON data."""
    type_param = escape_string(type_constraint.get("type_param_str", "T"))
    description = escape_string(type_constraint.get("description", ""))
    allowed_types = type_constraint.get("allowed_type_strs", [])

    if not allowed_types:
        return ""

    # Format allowed types as C++ initializer list
    types_list = "{" + ", ".join(escape_string(t) for t in allowed_types) + "}"

    return f".TypeConstraint({type_param}, {types_list}, {description})"


def generate_input_output(io_spec, method, index):
    """Generate Input or Output code from JSON data."""
    name = escape_string(io_spec.get("name", ""))
    description = escape_string(io_spec.get("description", ""))
    type_str = escape_string(io_spec.get("typeStr", "T"))
    option = enum_to_cxx_string(
        io_spec.get("option", "FormalParameterOption.Single"),
        "OpSchema::FormalParameterOption",
    )
    is_homogeneous = bool_to_string(io_spec.get("is_homogeneous", False))

    return f".{method}({index}, {name}, {description}, {type_str}, {option}, {is_homogeneous})"


def generate_attribute(attr):
    """Generate Attr code from JSON data."""
    name = escape_string(attr.get("name", ""))
    description = escape_string(attr.get("description", ""))
    attr_type = enum_to_cxx_string(attr.get("type", "AttrType.INT"), "AttributeProto")
    required = bool_to_string(attr.get("required", False))

    # patch for "Optional" op
    if attr_type == "AttributeProto::???":
        attr_type = "AttributeProto::TYPE_PROTO"

    return f".Attr({name}, {description}, {attr_type}, {required})"


def generate_schema_registration(schema, counter):
    """Generate a complete schema registration from JSON data."""
    name = escape_string(schema.get("name", ""))
    domain = escape_string(schema.get("domain", ""))
    since_version = schema.get("since_version", 1)
    doc = escape_string(schema.get("doc", ""))
    file = escape_string(schema.get("file", ""))
    line = schema.get("line", 0)

    if not name:
        return ""

    # Start the registration
    lines = []
    lines.append(
        f"  if (OpSchemaRegistry::Schema({name}, {since_version}, {domain}) == nullptr) "
        + "{"
    )
    lines.append(f"      RegisterSchema(OpSchema({name}, {file}, {line})")

    # Add domain and version
    if domain:
        lines.append(f"              .SetDomain({domain})")
    lines.append(f"              .SinceVersion({since_version})")

    # Add documentation
    if doc != '""':
        lines.append(f"              .SetDoc({doc})")

    # Add inputs
    inputs = schema.get("inputs", [])
    for i, input_spec in enumerate(inputs):
        input_code = generate_input_output(input_spec, "Input", i)
        if input_code:
            lines.append(f"              {input_code}")

    # Add outputs
    outputs = schema.get("outputs", [])
    for i, output_spec in enumerate(outputs):
        output_code = generate_input_output(output_spec, "Output", i)
        if output_code:
            lines.append(f"              {output_code}")

    # Add attributes
    attributes = schema.get("attributes", [])
    for attr in attributes:
        attr_code = generate_attribute(attr)
        if attr_code:
            lines.append(f"              {attr_code}")

    # Add type constraints
    type_constraints = schema.get("type_constraints", [])
    for tc in type_constraints:
        tc_code = generate_type_constraint(tc)
        if tc_code:
            lines.append(f"              {tc_code}")

    # Close the registration
    lines.append("      );")
    lines.append("  }")

    return "\n".join(lines)


def main():
    if len(sys.argv) != 2:
        output_file = "onnx_schema_json_binary.hpp"
    else:
        output_file = sys.argv[1]

    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    schemas_json_path = script_dir / "schemas.json"

    if not schemas_json_path.exists():
        raise FileNotFoundError(f"schemas.json not found at {schemas_json_path}")

    # Load and parse the JSON file
    with open(schemas_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    domains = data.get("domains", [])
    schemas = data.get("schemas", [])

    # Generate C++ header file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("/*\n")
        f.write(
            " * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.\n"
        )
        f.write(" * Licensed under the MIT License.\n")
        f.write(" */\n")
        f.write("\n")

        f.write("// clang-format off\n\n")
        f.write("// Auto-generated ONNX schema registrations\n")
        f.write(f"// Source: {schemas_json_path.name}\n")
        f.write("// DO NOT EDIT - This file is automatically generated\n\n")

        f.write("#pragma once\n\n")
        f.write("#include <onnx/defs/schema.h>\n\n")

        f.write("namespace morphizen {\n")
        f.write("namespace generated_schemas {\n\n")

        f.write("// Register all ONNX schemas from JSON data\n")
        f.write("using namespace ONNX_NAMESPACE;\n")
        f.write("inline void RegisterAllSchemas() {\n")

        # Generate domain registration if needed
        f.write("  // Register domains\n")
        f.write("  auto& domainToVersionRangeInstance =\n")
        f.write("      OpSchemaRegistry::DomainToVersionRange::Instance();\n")
        f.write("  auto& domain_version_map =\n")
        f.write("      domainToVersionRangeInstance.Map();\n")
        f.write("\n")

        # Register commonly used domains
        for domain in domains:
            domain_name = escape_string(domain.get("name", ""))
            min_version = domain.get("min_version", 1)
            max_version = domain.get("max_version", 1)

            f.write(f"  if (domain_version_map.count({domain_name}) == 0) " + "{\n")
            f.write(
                f"      domainToVersionRangeInstance.AddDomainToVersion({domain_name}, {min_version}, {max_version});\n"
            )
            f.write("  }\n\n")

        # Generate schema registrations
        f.write("  // Register schemas\n")
        counter = 0
        for schema in schemas:
            name = schema.get("name", "")
            domain = schema.get("domain", "")

            if not name:
                continue

            # Skip schemas with empty domain or ai.onnx domain
            if not domain or domain == "ai.onnx" or domain == "ai.onnx.ml":
                continue

            schema_code = generate_schema_registration(schema, counter)
            if schema_code:
                f.write(f"{schema_code}\n\n")
                counter += 1

        f.write("}\n\n")  # Close RegisterAllSchemas function
        f.write("} // namespace generated_schemas\n")
        f.write("} // namespace morphizen\n")
        f.write("\n// clang-format on\n")

    print(f"Generated {counter} schema registrations in {output_file}")


if __name__ == "__main__":
    main()
