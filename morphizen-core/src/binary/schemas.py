#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

# pylint: disable=missing-module-docstring
# pylint: disable=c-extension-no-member

import json
import onnxruntime as ort


if __name__ == "__main__":
    schemas = ort.capi.onnxruntime_pybind11_state.get_all_operator_schema()

    schema_dict = {}
    schema_domains = {}
    schema_list = []
    for s in schemas:
        schema_obj = {}
        schema_obj["domain"] = s.domain
        schema_obj["name"] = s.name
        # schema_obj["doc"] = s.doc
        schema_obj["deprecated"] = s.deprecated
        # schema_obj["file"] = s.file
        # schema_obj["line"] = s.line
        schema_obj["since_version"] = s.since_version
        schema_obj["support_level"] = str(s.support_level)
        schema_obj["min_input"] = s.min_input
        schema_obj["max_input"] = s.max_input
        schema_obj["min_output"] = s.min_output
        schema_obj["max_output"] = s.max_output

        if s.domain in schema_domains:
            if schema_domains[s.domain]["max_version"] < s.since_version:
                schema_domains[s.domain]["max_version"] = s.since_version
        else:
            schema_domains[s.domain] = {
                "min_version": 1,
                "max_version": s.since_version,
            }

        # inputs
        schema_inputs = []
        for i in s.inputs:
            schema_input_obj = {}
            schema_input_obj["name"] = i.name
            # schema_input_obj["description"] = i.description
            schema_input_obj["isHomogeneous"] = i.isHomogeneous
            schema_input_obj["option"] = str(i.option)
            schema_input_obj["typeStr"] = i.typeStr
            schema_inputs.append(schema_input_obj)
        schema_obj["inputs"] = schema_inputs

        # outputs
        schema_outputs = []
        for o in s.outputs:
            schema_output_obj = {}
            schema_output_obj["name"] = o.name
            # schema_output_obj["description"] = o.description
            schema_output_obj["isHomogeneous"] = o.isHomogeneous
            schema_output_obj["option"] = str(o.option)
            schema_output_obj["typeStr"] = o.typeStr
            schema_outputs.append(schema_output_obj)
        schema_obj["outputs"] = schema_outputs

        # type_constraints
        schema_type_constraints = []
        for c in s.type_constraints:
            schema_type_constraint_obj = {}
            schema_type_constraint_obj["type_param_str"] = c.type_param_str
            # schema_type_constraint_obj["description"] = c.description

            schema_type_constraint_allowed_type_strs = []
            for t in c.allowed_type_strs:
                schema_type_constraint_allowed_type_strs.append(str(t))
            schema_type_constraint_obj["allowed_type_strs"] = schema_type_constraint_allowed_type_strs

            schema_type_constraints.append(schema_type_constraint_obj)
        schema_obj["type_constraints"] = schema_type_constraints

        # attributes
        schema_attributes = []
        for _, a in s.attributes.items():
            schema_attribut_obj = {}
            schema_attribut_obj["name"] = a.name
            # schema_attribut_obj["description"] = a.description
            schema_attribut_obj["required"] = a.required
            schema_attribut_obj["type"] = str(a.type)
            schema_attributes.append(schema_attribut_obj)
        schema_obj["attributes"] = schema_attributes

        schema_list.append(schema_obj)

    schema_dict["domains"] = [
        {
            "name": name,
            "min_version": domain["min_version"],
            "max_version": domain["max_version"],
        }
        for name, domain in schema_domains.items()
    ]
    schema_dict["schemas"] = schema_list

    with open("schemas.json", "w", encoding="utf-8") as f:
        json.dump(schema_dict, f, indent=4)
        f.write("\n")
