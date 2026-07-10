/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include <algorithm>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <vector>
namespace morphizen {
namespace test {

/**
 * @brief Get the complete list of all API function names in OrtApiForMorphizen
 *
 * This function returns a comprehensive list of all ~108 API functions
 * defined in the OrtApiForMorphizen structure, organized by category.
 *
 * @return std::vector<std::string> Complete list of API function names
 */
inline std::vector<std::string> get_all_morphizen_ort_api_functions() {
  return {
    // Model APIs [0-6]
    "model_load", "model_delete", "model_clone", "model_main_graph",
        "model_set_meta_data", "model_get_meta_data", "model_has_meta_data",

        // Graph APIs [7-23]
        "graph_get_name", "graph_get_model", "graph_nodes_unsafe",
        "graph_get_inputs_unsafe", "graph_get_outputs_unsafe",
        "graph_set_outputs", "graph_get_node", "graph_producer_node",
        "graph_get_node_arg", "graph_get_all_initialized_tensors",
        "graph_remove_node", "graph_add_node", "graph_save", "graph_fuse",
        "graph_resolve", "graph_get_consumer_nodes_unsafe",
        "graph_reverse_dfs_from",

        // Node APIs [24-33]
        "node_get_name", "node_description", "node_get_index", "node_op_type",
        "node_op_domain", "node_get_inputs_unsafe",
        "node_get_output_node_args_unsafe", "node_get_attributes",
        "node_get_function_body", "node_type_is_fused",

        // NodeArg APIs [34-45]
        "node_arg_get_name_unsafe", "node_arg_is_exists",
        "node_arg_is_constant", "node_arg_clone", "node_arg_new",
        "node_arg_get_shape_i64_unsafe", "node_arg_get_denotation_unsafe",
        "node_arg_set_shape_i64", "node_arg_set_denotation",
        "node_arg_get_element_type", "node_arg_set_element_type",
        "node_arg_get_const_data_as_tensor",

        // NodeAttributes APIs [46-50]
        "node_attributes_new", "node_attributes_delete", "node_attributes_add",
        "node_attributes_get", "node_attributes_get_keys",

        // AttributeProto APIs [51-69]
        "attr_proto_delete", "attr_proto_clone", "attr_proto_get_name",
        "attr_proto_get_type", "attr_proto_set_name", "attr_proto_new_int",
        "attr_proto_new_float", "attr_proto_new_string",
        "attr_proto_new_tensor", "attr_proto_new_ints", "attr_proto_new_floats",
        "attr_proto_new_strings", "attr_proto_get_int", "attr_proto_get_float",
        "attr_proto_get_string", "attr_proto_get_tensor", "attr_proto_get_ints",
        "attr_proto_get_floats", "attr_proto_get_strings",

        // TensorProto APIs [70-89]
        "tensor_proto_delete", "tensor_proto_get_shape_unsafe",
        "tensor_proto_data_type", "tensor_proto_new_floats",
        "tensor_proto_new_i64", "tensor_proto_new_i32", "tensor_proto_new_i8",
        "tensor_proto_get_name", "tensor_proto_raw_data_size",
        "tensor_proto_as_raw",

        // Extended APIs [80-108]
        "get_lib_id", "get_lib_name", "graph_add_initialized_tensor",
        "tensor_proto_new_doubles", "tensor_proto_new_i16",
        "tensor_proto_new_u16", "tensor_proto_new_u32", "tensor_proto_new_u8",
        "tensor_proto_new_u64", "tensor_proto_new_fp16",
        "tensor_proto_new_bf16", "get_model_path", "create_empty_model",
        "graph_set_inputs", "node_arg_external_location",
        "session_option_configuration", "model_to_proto",
        "model_proto_serialize_as_string", "model_proto_delete",
        "attr_proto_release_string", "is_profiling_enabled",
        "tensor_proto_new_i4", "tensor_proto_new_u4",
        "graph_remove_initialized_tensor", "graph_reverse_dfs_from_preemp",
        "graph_set_name", "graph_infer_shapes_from_filepath",
        "graph_to_graph_proto", "graph_proto_delete",
#if MORPHIZEN_ORT_API_MAJOR >= 18
        "graph_save_string",
#endif
        "graph_infer_shapes"
  };
}

/**
 * @brief Get API functions organized by category
 *
 * @return std::map<std::string, std::vector<std::string>> APIs grouped by
 * category
 */
inline std::map<std::string, std::vector<std::string>>
get_morphizen_ort_api_by_category() {
  return {
      {"Model",
       {"model_load", "model_delete", "model_clone", "model_main_graph",
        "model_set_meta_data", "model_get_meta_data", "model_has_meta_data"}},
      {"Graph",
       {"graph_get_name",
        "graph_get_model",
        "graph_nodes_unsafe",
        "graph_get_inputs_unsafe",
        "graph_get_outputs_unsafe",
        "graph_set_outputs",
        "graph_get_node",
        "graph_producer_node",
        "graph_get_node_arg",
        "graph_get_all_initialized_tensors",
        "graph_remove_node",
        "graph_add_node",
        "graph_save",
#if MORPHIZEN_ORT_API_MAJOR >= 18
        "graph_save_string",
#endif // MORPHIZEN_ORT_API_MAJOR >= 18
        "graph_fuse",
        "graph_resolve",
        "graph_get_consumer_nodes_unsafe",
        "graph_reverse_dfs_from",
        "graph_add_initialized_tensor",
        "graph_set_inputs",
        "graph_remove_initialized_tensor",
        "graph_reverse_dfs_from_preemp",
        "graph_set_name",
        "graph_infer_shapes_from_filepath",
        "graph_to_graph_proto",
        "graph_proto_delete",
        "graph_infer_shapes"}},
      {"Node",
       {"node_get_name", "node_description", "node_get_index", "node_op_type",
        "node_op_domain", "node_get_inputs_unsafe",
        "node_get_output_node_args_unsafe", "node_get_attributes",
        "node_get_function_body", "node_type_is_fused"}},
      {"NodeArg",
       {"node_arg_get_name_unsafe", "node_arg_is_exists",
        "node_arg_is_constant", "node_arg_clone", "node_arg_new",
        "node_arg_get_shape_i64_unsafe", "node_arg_get_denotation_unsafe",
        "node_arg_set_shape_i64", "node_arg_set_denotation",
        "node_arg_get_element_type", "node_arg_set_element_type",
        "node_arg_get_const_data_as_tensor", "node_arg_external_location"}},
      {"NodeAttributes",
       {"node_attributes_new", "node_attributes_delete", "node_attributes_add",
        "node_attributes_get", "node_attributes_get_keys"}},
      {"AttributeProto",
       {"attr_proto_delete",      "attr_proto_clone",
        "attr_proto_get_name",    "attr_proto_get_type",
        "attr_proto_set_name",    "attr_proto_new_int",
        "attr_proto_new_float",   "attr_proto_new_string",
        "attr_proto_new_tensor",  "attr_proto_new_ints",
        "attr_proto_new_floats",  "attr_proto_new_strings",
        "attr_proto_get_int",     "attr_proto_get_float",
        "attr_proto_get_string",  "attr_proto_get_tensor",
        "attr_proto_get_ints",    "attr_proto_get_floats",
        "attr_proto_get_strings", "attr_proto_release_string"}},
      {"TensorProto",
       {"tensor_proto_delete",        "tensor_proto_get_shape_unsafe",
        "tensor_proto_data_type",     "tensor_proto_new_floats",
        "tensor_proto_new_i64",       "tensor_proto_new_i32",
        "tensor_proto_new_i8",        "tensor_proto_get_name",
        "tensor_proto_raw_data_size", "tensor_proto_as_raw",
        "tensor_proto_new_doubles",   "tensor_proto_new_i16",
        "tensor_proto_new_u16",       "tensor_proto_new_u32",
        "tensor_proto_new_u8",        "tensor_proto_new_u64",
        "tensor_proto_new_fp16",      "tensor_proto_new_bf16",
        "tensor_proto_new_i4",        "tensor_proto_new_u4"}},
      {"Extended",
       {"get_lib_id", "get_lib_name", "get_model_path", "create_empty_model",
        "session_option_configuration", "model_to_proto",
        "model_proto_serialize_as_string", "model_proto_delete",
        "is_profiling_enabled"}}};
}

/**
 * @brief Check API coverage completeness
 *
 * @param called_apis Set of API function names that were called
 * @return std::pair<size_t, std::set<std::string>> Coverage percentage and
 * missing APIs
 */
inline std::pair<double, std::set<std::string>>
check_api_coverage(const std::map<std::string, size_t> &call_stats) {
  auto all_apis = get_all_morphizen_ort_api_functions();
  std::set<std::string> all_api_set(all_apis.begin(), all_apis.end());
  std::set<std::string> called_api_set;

  for (const auto &[api_name, count] : call_stats) {
    if (count > 0) {
      called_api_set.insert(api_name);
    }
  }

  std::set<std::string> missing_apis;
  std::set_difference(all_api_set.begin(), all_api_set.end(),
                      called_api_set.begin(), called_api_set.end(),
                      std::inserter(missing_apis, missing_apis.begin()));

  double coverage =
      static_cast<double>(called_api_set.size()) / all_api_set.size() * 100.0;
  return {coverage, missing_apis};
}

} // namespace test
} // namespace morphizen
