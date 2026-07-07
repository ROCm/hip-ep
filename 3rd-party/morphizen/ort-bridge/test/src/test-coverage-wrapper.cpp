/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./test-coverage-wrapper.hpp"
#include <cstring>
#include <functional>
#include <map>
#include <memory>

namespace morphizen {
namespace test {

namespace {

// Global instance for the wrapped API
static std::unique_ptr<morphizen::OrtApiForMorphizen> g_wrapped_api;
static morphizen::OrtApiForMorphizen* g_original_api = nullptr;

// Counter for API call tracking
static std::map<std::string, size_t> g_api_call_counts;

#define LOG_API_CALL(name)                                                     \
  do {                                                                         \
    g_api_call_counts[#name]++;                                                \
    VLOG_IF(2, 0) << "MorphiZen API Call: " << #name                           \
                  << " (count: " << g_api_call_counts[#name] << ")";           \
  } while (0)

#define WRAP_API_CALL(name, ...)                                               \
  LOG_API_CALL(name);                                                          \
  return g_original_api->name(__VA_ARGS__)

// Model API wrappers [0-6]
morphizen::Model* wrapped_model_load(const std::string& file) {
  WRAP_API_CALL(model_load, file);
}

void wrapped_model_delete(morphizen::Model* model) {
  WRAP_API_CALL(model_delete, model);
}

morphizen::Model* wrapped_model_clone(const morphizen::Model& model,
                                      int64_t external_data_threshold) {
  WRAP_API_CALL(model_clone, model, external_data_threshold);
}

morphizen::Graph& wrapped_model_main_graph(morphizen::Model& model) {
  WRAP_API_CALL(model_main_graph, model);
}

void wrapped_model_set_meta_data(morphizen::Model& model,
                                 const std::string& key,
                                 const std::string& value) {
  WRAP_API_CALL(model_set_meta_data, model, key, value);
}

morphizen::DllSafe<std::string>
wrapped_model_get_meta_data(const morphizen::Model& model,
                            const std::string& key) {
  WRAP_API_CALL(model_get_meta_data, model, key);
}

int wrapped_model_has_meta_data(const morphizen::Model& model,
                                const std::string& key) {
  WRAP_API_CALL(model_has_meta_data, model, key);
}

// Graph API wrappers [7-23]
const std::string& wrapped_graph_get_name(const morphizen::Graph& graph) {
  WRAP_API_CALL(graph_get_name, graph);
}

const morphizen::Model& wrapped_graph_get_model(const morphizen::Graph& graph) {
  WRAP_API_CALL(graph_get_model, graph);
}

morphizen::DllSafe<std::vector<const morphizen::Node*>>
wrapped_graph_nodes_unsafe(const morphizen::Graph& graph) {
  WRAP_API_CALL(graph_nodes_unsafe, graph);
}

morphizen::DllSafe<std::vector<const morphizen::NodeArg*>>
wrapped_graph_get_inputs_unsafe(const morphizen::Graph& graph) {
  WRAP_API_CALL(graph_get_inputs_unsafe, graph);
}

morphizen::DllSafe<std::vector<const morphizen::NodeArg*>>
wrapped_graph_get_outputs_unsafe(const morphizen::Graph& graph) {
  WRAP_API_CALL(graph_get_outputs_unsafe, graph);
}

void wrapped_graph_set_outputs(
    morphizen::Graph& graph,
    gsl::span<const morphizen::NodeArg* const> outputs) {
  WRAP_API_CALL(graph_set_outputs, graph, outputs);
}

const morphizen::Node* wrapped_graph_get_node(const morphizen::Graph& graph,
                                              size_t index) {
  WRAP_API_CALL(graph_get_node, graph, index);
}

const morphizen::Node*
wrapped_graph_producer_node(const morphizen::Graph& graph,
                            const std::string& node_arg_name) {
  WRAP_API_CALL(graph_producer_node, graph, node_arg_name);
}

const morphizen::NodeArg*
wrapped_graph_get_node_arg(const morphizen::Graph& graph,
                           const std::string& name) {
  WRAP_API_CALL(graph_get_node_arg, graph, name);
}

const morphizen::InitializedTensorSet&
wrapped_graph_get_all_initialized_tensors(const morphizen::Graph& graph) {
  WRAP_API_CALL(graph_get_all_initialized_tensors, graph);
}

void wrapped_graph_remove_node(morphizen::Graph& graph,
                               const morphizen::NodeInput& node_input) {
  WRAP_API_CALL(graph_remove_node, graph, node_input);
}

morphizen::Node& wrapped_graph_add_node(
    morphizen::Graph& graph, const std::string& name,
    const std::string& op_type, const std::string& description,
    const std::vector<const morphizen::NodeArg*>& input_args,
    const std::vector<const morphizen::NodeArg*>& output_args,
    const morphizen::NodeAttributes& attributes, const std::string& domain) {
  WRAP_API_CALL(graph_add_node, graph, name, op_type, description, input_args,
                output_args, attributes, domain);
}

void wrapped_graph_save(const morphizen::Graph& graph,
                        const std::string& filename,
                        const std::string& dat_filename,
                        size_t external_data_threshold) {
  WRAP_API_CALL(graph_save, graph, filename, dat_filename,
                external_data_threshold);
}
#if MORPHIZEN_ORT_API_MAJOR >= 18
morphizen::DllSafe<std::string>
wrapped_graph_save_string(const morphizen::Graph& graph) {
  WRAP_API_CALL(graph_save_string, graph);
}
#endif // MORPHIZEN_ORT_API_MAJOR >= 18

morphizen::Node&
wrapped_graph_fuse(morphizen::Graph& graph, const std::string& name,
                   const std::string& op_type, const std::vector<size_t>& nodes,
                   const std::vector<std::string>& inputs,
                   const std::vector<std::string>& outputs,
                   const std::vector<std::string>& constant_initializers) {
  WRAP_API_CALL(graph_fuse, graph, name, op_type, nodes, inputs, outputs,
                constant_initializers);
}

int wrapped_graph_resolve(morphizen::Graph& graph, bool force) {
  WRAP_API_CALL(graph_resolve, graph, force);
}

morphizen::DllSafe<std::vector<const morphizen::Node*>>
wrapped_graph_get_consumer_nodes_unsafe(const morphizen::Graph& graph,
                                        const std::string& node_arg_name) {
  WRAP_API_CALL(graph_get_consumer_nodes_unsafe, graph, node_arg_name);
}

void wrapped_graph_reverse_dfs_from(
    const morphizen::Graph& graph, gsl::span<const morphizen::Node* const> from,
    const std::function<void(const morphizen::Node*)>& enter,
    const std::function<void(const morphizen::Node*)>& leave,
    const std::function<bool(const morphizen::Node* from,
                             const morphizen::Node* to)>& stop) {
  WRAP_API_CALL(graph_reverse_dfs_from, graph, from, enter, leave, stop);
}

// Node API wrappers [24-33]
const std::string& wrapped_node_get_name(const morphizen::Node& node) {
  WRAP_API_CALL(node_get_name, node);
}

const std::string& wrapped_node_description(const morphizen::Node& node) {
  WRAP_API_CALL(node_description, node);
}

size_t wrapped_node_get_index(const morphizen::Node& node) {
  WRAP_API_CALL(node_get_index, node);
}

const std::string& wrapped_node_op_type(const morphizen::Node& node) {
  WRAP_API_CALL(node_op_type, node);
}

const std::string& wrapped_node_op_domain(const morphizen::Node& node) {
  WRAP_API_CALL(node_op_domain, node);
}

morphizen::DllSafe<std::vector<morphizen::NodeInput>>
wrapped_node_get_inputs_unsafe(const morphizen::Node& node) {
  WRAP_API_CALL(node_get_inputs_unsafe, node);
}

morphizen::DllSafe<std::vector<const morphizen::NodeArg*>>
wrapped_node_get_output_node_args_unsafe(const morphizen::Node& node) {
  WRAP_API_CALL(node_get_output_node_args_unsafe, node);
}

morphizen::NodeAttributes& wrapped_node_get_attributes(morphizen::Node& node) {
  WRAP_API_CALL(node_get_attributes, node);
}

const morphizen::Graph&
wrapped_node_get_function_body(const morphizen::Node& node) {
  WRAP_API_CALL(node_get_function_body, node);
}

bool wrapped_node_type_is_fused(const morphizen::Node& node) {
  WRAP_API_CALL(node_type_is_fused, node);
}

// NodeArg API wrappers [34-45]
const std::string&
wrapped_node_arg_get_name_unsafe(const morphizen::NodeArg& node_arg) {
  WRAP_API_CALL(node_arg_get_name_unsafe, node_arg);
}

bool wrapped_node_arg_is_exists(const morphizen::NodeArg& node_arg) {
  WRAP_API_CALL(node_arg_is_exists, node_arg);
}

bool wrapped_node_arg_is_constant(const morphizen::Graph& graph,
                                  const morphizen::NodeArg& node_arg) {
  WRAP_API_CALL(node_arg_is_constant, graph, node_arg);
}

morphizen::NodeArg& wrapped_node_arg_clone(morphizen::Graph& graph,
                                           const morphizen::NodeArg& node_arg,
                                           const std::string& name) {
  WRAP_API_CALL(node_arg_clone, graph, node_arg, name);
}

morphizen::NodeArg& wrapped_node_arg_new(morphizen::Graph& graph,
                                         const std::string& name,
                                         const std::vector<int64_t>* shape,
                                         int element_type) {
  WRAP_API_CALL(node_arg_new, graph, name, shape, element_type);
}

morphizen::DllSafe<std::vector<int64_t>>
wrapped_node_arg_get_shape_i64_unsafe(const morphizen::NodeArg& node_arg) {
  WRAP_API_CALL(node_arg_get_shape_i64_unsafe, node_arg);
}

morphizen::DllSafe<std::vector<std::string>>
wrapped_node_arg_get_denotation_unsafe(const morphizen::NodeArg& node_arg) {
  WRAP_API_CALL(node_arg_get_denotation_unsafe, node_arg);
}

void wrapped_node_arg_set_shape_i64(const morphizen::NodeArg& node_arg,
                                    const std::vector<int64_t>& shape) {
  WRAP_API_CALL(node_arg_set_shape_i64, node_arg, shape);
}

void wrapped_node_arg_set_denotation(
    const morphizen::NodeArg& node_arg,
    const std::vector<std::string>& denotation) {
  WRAP_API_CALL(node_arg_set_denotation, node_arg, denotation);
}

int wrapped_node_arg_get_element_type(const morphizen::NodeArg& node_arg) {
  WRAP_API_CALL(node_arg_get_element_type, node_arg);
}

void wrapped_node_arg_set_element_type(morphizen::NodeArg& node_arg,
                                       int data_type) {
  WRAP_API_CALL(node_arg_set_element_type, node_arg, data_type);
}

const morphizen::TensorProto&
wrapped_node_arg_get_const_data_as_tensor(const morphizen::Graph& graph,
                                          const morphizen::NodeArg& node_arg) {
  WRAP_API_CALL(node_arg_get_const_data_as_tensor, graph, node_arg);
}

// NodeAttributes API wrappers [46-50]
morphizen::NodeAttributes* wrapped_node_attributes_new() {
  WRAP_API_CALL(node_attributes_new);
}

void wrapped_node_attributes_delete(morphizen::NodeAttributes* p) {
  WRAP_API_CALL(node_attributes_delete, p);
}

void wrapped_node_attributes_add(morphizen::NodeAttributes& p,
                                 morphizen::AttributeProto&& attr) {
  WRAP_API_CALL(node_attributes_add, p, std::move(attr));
}

const morphizen::AttributeProto*
wrapped_node_attributes_get(const morphizen::NodeAttributes& p,
                            const std::string& name) {
  WRAP_API_CALL(node_attributes_get, p, name);
}

morphizen::DllSafe<std::vector<std::string>>
wrapped_node_attributes_get_keys(morphizen::NodeAttributes& p) {
  WRAP_API_CALL(node_attributes_get_keys, p);
}

// AttributeProto API wrappers [51-69]
void wrapped_attr_proto_delete(morphizen::AttributeProto* attr) {
  WRAP_API_CALL(attr_proto_delete, attr);
}

morphizen::AttributeProto*
wrapped_attr_proto_clone(const morphizen::AttributeProto& attr) {
  WRAP_API_CALL(attr_proto_clone, attr);
}

const std::string&
wrapped_attr_proto_get_name(const morphizen::AttributeProto& attr) {
  WRAP_API_CALL(attr_proto_get_name, attr);
}

int wrapped_attr_proto_get_type(const morphizen::AttributeProto& attr) {
  WRAP_API_CALL(attr_proto_get_type, attr);
}

void wrapped_attr_proto_set_name(morphizen::AttributeProto* attr,
                                 const std::string& name) {
  WRAP_API_CALL(attr_proto_set_name, attr, name);
}

morphizen::AttributeProto* wrapped_attr_proto_new_int(const std::string& name,
                                                      int64_t value) {
  WRAP_API_CALL(attr_proto_new_int, name, value);
}

morphizen::AttributeProto* wrapped_attr_proto_new_float(const std::string& name,
                                                        float value) {
  WRAP_API_CALL(attr_proto_new_float, name, value);
}

morphizen::AttributeProto*
wrapped_attr_proto_new_string(const std::string& name,
                              const std::string& value) {
  WRAP_API_CALL(attr_proto_new_string, name, value);
}

morphizen::AttributeProto*
wrapped_attr_proto_new_tensor(const std::string& name,
                              const morphizen::TensorProto& value) {
  WRAP_API_CALL(attr_proto_new_tensor, name, value);
}

morphizen::AttributeProto*
wrapped_attr_proto_new_ints(const std::string& name,
                            const std::vector<int64_t>& value) {
  WRAP_API_CALL(attr_proto_new_ints, name, value);
}

morphizen::AttributeProto*
wrapped_attr_proto_new_floats(const std::string& name,
                              const std::vector<float>& value) {
  WRAP_API_CALL(attr_proto_new_floats, name, value);
}

morphizen::AttributeProto*
wrapped_attr_proto_new_strings(const std::string& name,
                               const std::vector<std::string>& value) {
  WRAP_API_CALL(attr_proto_new_strings, name, value);
}

int64_t wrapped_attr_proto_get_int(const morphizen::AttributeProto& attr) {
  WRAP_API_CALL(attr_proto_get_int, attr);
}

float wrapped_attr_proto_get_float(const morphizen::AttributeProto& attr) {
  WRAP_API_CALL(attr_proto_get_float, attr);
}

const std::string&
wrapped_attr_proto_get_string(const morphizen::AttributeProto& attr) {
  WRAP_API_CALL(attr_proto_get_string, attr);
}

const morphizen::TensorProto&
wrapped_attr_proto_get_tensor(const morphizen::AttributeProto& attr) {
  WRAP_API_CALL(attr_proto_get_tensor, attr);
}

gsl::span<const int64_t>
wrapped_attr_proto_get_ints(const morphizen::AttributeProto& attr) {
  WRAP_API_CALL(attr_proto_get_ints, attr);
}

gsl::span<const float>
wrapped_attr_proto_get_floats(const morphizen::AttributeProto& attr) {
  WRAP_API_CALL(attr_proto_get_floats, attr);
}

std::vector<std::string>
wrapped_attr_proto_get_strings(const morphizen::AttributeProto& attr) {
  WRAP_API_CALL(attr_proto_get_strings, attr);
}

// TensorProto API wrappers [70-89]
void wrapped_tensor_proto_delete(morphizen::TensorProto* tp) {
  WRAP_API_CALL(tensor_proto_delete, tp);
}

morphizen::DllSafe<std::vector<int64_t>> wrapped_tensor_proto_get_shape_unsafe(
    const morphizen::TensorProto& tensor_proto) {
  WRAP_API_CALL(tensor_proto_get_shape_unsafe, tensor_proto);
}

int wrapped_tensor_proto_data_type(const morphizen::TensorProto& tensor_proto) {
  WRAP_API_CALL(tensor_proto_data_type, tensor_proto);
}

morphizen::TensorProto*
wrapped_tensor_proto_new_floats(const std::string& name,
                                const std::vector<int64_t>& shape,
                                const std::vector<float>& data) {
  WRAP_API_CALL(tensor_proto_new_floats, name, shape, data);
}

morphizen::TensorProto*
wrapped_tensor_proto_new_i64(const std::string& name,
                             const std::vector<int64_t>& shape,
                             const std::vector<int64_t>& data) {
  WRAP_API_CALL(tensor_proto_new_i64, name, shape, data);
}

morphizen::TensorProto*
wrapped_tensor_proto_new_i32(const std::string& name,
                             const std::vector<int64_t>& shape,
                             const std::vector<int32_t>& data) {
  WRAP_API_CALL(tensor_proto_new_i32, name, shape, data);
}

morphizen::TensorProto*
wrapped_tensor_proto_new_i8(const std::string& name,
                            const std::vector<int64_t>& shape,
                            const std::vector<int8_t>& data) {
  WRAP_API_CALL(tensor_proto_new_i8, name, shape, data);
}

const std::string&
wrapped_tensor_proto_get_name(const morphizen::TensorProto& tensor_proto) {
  WRAP_API_CALL(tensor_proto_get_name, tensor_proto);
}

size_t
wrapped_tensor_proto_raw_data_size(const morphizen::TensorProto& tensor) {
  WRAP_API_CALL(tensor_proto_raw_data_size, tensor);
}

gsl::span<const char>
wrapped_tensor_proto_as_raw(const morphizen::Graph& graph,
                            const morphizen::TensorProto& tensor) {
  WRAP_API_CALL(tensor_proto_as_raw, graph, tensor);
}

// Extended API wrappers [80-108]
morphizen::DllSafe<std::string> wrapped_get_lib_id() {
  WRAP_API_CALL(get_lib_id);
}

morphizen::DllSafe<std::string> wrapped_get_lib_name() {
  WRAP_API_CALL(get_lib_name);
}

void wrapped_graph_add_initialized_tensor(
    morphizen::Graph& graph, const morphizen::TensorProto& tensor) {
  WRAP_API_CALL(graph_add_initialized_tensor, graph, tensor);
}

morphizen::TensorProto*
wrapped_tensor_proto_new_doubles(const std::string& name,
                                 const std::vector<int64_t>& shape,
                                 const std::vector<double>& data) {
  WRAP_API_CALL(tensor_proto_new_doubles, name, shape, data);
}

morphizen::TensorProto*
wrapped_tensor_proto_new_i16(const std::string& name,
                             const std::vector<int64_t>& shape,
                             const std::vector<int16_t>& data) {
  WRAP_API_CALL(tensor_proto_new_i16, name, shape, data);
}

morphizen::TensorProto*
wrapped_tensor_proto_new_u16(const std::string& name,
                             const std::vector<int64_t>& shape,
                             const std::vector<uint16_t>& data) {
  WRAP_API_CALL(tensor_proto_new_u16, name, shape, data);
}

morphizen::TensorProto*
wrapped_tensor_proto_new_u32(const std::string& name,
                             const std::vector<int64_t>& shape,
                             const std::vector<uint32_t>& data) {
  WRAP_API_CALL(tensor_proto_new_u32, name, shape, data);
}

morphizen::TensorProto*
wrapped_tensor_proto_new_u8(const std::string& name,
                            const std::vector<int64_t>& shape,
                            const std::vector<uint8_t>& data) {
  WRAP_API_CALL(tensor_proto_new_u8, name, shape, data);
}

morphizen::TensorProto*
wrapped_tensor_proto_new_u64(const std::string& name,
                             const std::vector<int64_t>& shape,
                             const std::vector<uint64_t>& data) {
  WRAP_API_CALL(tensor_proto_new_u64, name, shape, data);
}

morphizen::TensorProto*
wrapped_tensor_proto_new_fp16(const std::string& name,
                              const std::vector<int64_t>& shape,
                              const std::vector<int16_t>& data) {
  WRAP_API_CALL(tensor_proto_new_fp16, name, shape, data);
}

morphizen::TensorProto*
wrapped_tensor_proto_new_bf16(const std::string& name,
                              const std::vector<int64_t>& shape,
                              const std::vector<int16_t>& data) {
  WRAP_API_CALL(tensor_proto_new_bf16, name, shape, data);
}

const std::filesystem::path&
wrapped_get_model_path(const morphizen::Graph& graph) {
  WRAP_API_CALL(get_model_path, graph);
}

morphizen::Model* wrapped_create_empty_model(
    const std::filesystem::path& path,
    const std::vector<std::pair<std::string, int64_t>>& opset) {
  WRAP_API_CALL(create_empty_model, path, opset);
}

void wrapped_graph_set_inputs(
    morphizen::Graph& graph,
    gsl::span<const morphizen::NodeArg* const> inputs) {
  WRAP_API_CALL(graph_set_inputs, graph, inputs);
}

int wrapped_node_arg_external_location(const morphizen::Graph& graph,
                                       const morphizen::NodeArg& node_arg,
                                       std::string& file, size_t& offset,
                                       size_t& size, size_t& checksum) {
  WRAP_API_CALL(node_arg_external_location, graph, node_arg, file, offset, size,
                checksum);
}

void wrapped_session_option_configuration(void* mmap, void* session_option,
                                          void (*push)(void* mmap,
                                                       const char* name,
                                                       const char* value)) {
  WRAP_API_CALL(session_option_configuration, mmap, session_option, push);
}

morphizen::ModelProto* wrapped_model_to_proto(morphizen::Model& model) {
  WRAP_API_CALL(model_to_proto, model);
}

morphizen::DllSafe<std::string>
wrapped_model_proto_serialize_as_string(morphizen::ModelProto& model_proto) {
  WRAP_API_CALL(model_proto_serialize_as_string, model_proto);
}

void wrapped_model_proto_delete(morphizen::ModelProto* p) {
  WRAP_API_CALL(model_proto_delete, p);
}

morphizen::DllSafe<std::string>
wrapped_attr_proto_release_string(morphizen::AttributeProto* attr) {
  WRAP_API_CALL(attr_proto_release_string, attr);
}

bool wrapped_is_profiling_enabled(void* session_options) {
  WRAP_API_CALL(is_profiling_enabled, session_options);
}
#if MORPHIZEN_ORT_API_MAJOR >= 19
morphizen::TensorProto*
wrapped_tensor_proto_new_bool(const std::string& name,
                              const std::vector<int64_t>& shape,
                              const std::vector<uint8_t>& data) {
  WRAP_API_CALL(tensor_proto_new_bool, name, shape, data);
}
#endif // MORPHIZEN_ORT_API_MAJOR >= 19

morphizen::TensorProto*
wrapped_tensor_proto_new_i4(const std::string& name,
                            const std::vector<int64_t>& shape,
                            const std::vector<int8_t>& data) {
  WRAP_API_CALL(tensor_proto_new_i4, name, shape, data);
}

morphizen::TensorProto*
wrapped_tensor_proto_new_u4(const std::string& name,
                            const std::vector<int64_t>& shape,
                            const std::vector<uint8_t>& data) {
  WRAP_API_CALL(tensor_proto_new_u4, name, shape, data);
}

void wrapped_graph_remove_initialized_tensor(morphizen::Graph& graph,
                                             const std::string& tensor_name) {
  WRAP_API_CALL(graph_remove_initialized_tensor, graph, tensor_name);
}

void wrapped_graph_reverse_dfs_from_preemp(
    const morphizen::Graph& graph, gsl::span<const morphizen::Node* const> from,
    const std::function<bool(const morphizen::Node*)>& enter,
    const std::function<bool(const morphizen::Node*)>& leave,
    const std::function<bool(const morphizen::Node*, const morphizen::Node*)>&
        comp,
    const std::function<bool(const morphizen::Node* from,
                             const morphizen::Node* to)>& stop) {
  WRAP_API_CALL(graph_reverse_dfs_from_preemp, graph, from, enter, leave, comp,
                stop);
}

void wrapped_graph_set_name(morphizen::Graph& graph, const char* name) {
  WRAP_API_CALL(graph_set_name, graph, name);
}

void wrapped_graph_infer_shapes_from_filepath(const std::string& m,
                                              const std::string& save_path) {
  WRAP_API_CALL(graph_infer_shapes_from_filepath, m, save_path);
}

morphizen::GraphProto*
wrapped_graph_to_graph_proto(const morphizen::Graph& graph) {
  WRAP_API_CALL(graph_to_graph_proto, graph);
}

void wrapped_graph_proto_delete(morphizen::GraphProto* p) {
  WRAP_API_CALL(graph_proto_delete, p);
}

void wrapped_graph_infer_shapes(morphizen::ModelProto& m) {
  WRAP_API_CALL(graph_infer_shapes, m);
}

} // anonymous namespace

morphizen::OrtApiForMorphizen* get_morphizen_ort_api_for_coverage_test(
    morphizen::OrtApiForMorphizen* original_api) {
  if (!original_api) {
    LOG(ERROR) << "Original API is null, cannot create wrapper";
    return nullptr;
  }

  g_original_api = original_api;
  g_wrapped_api = std::make_unique<morphizen::OrtApiForMorphizen>();

  // Copy basic fields
  g_wrapped_api->magic = original_api->magic;
  g_wrapped_api->major = original_api->major;
  g_wrapped_api->minor = original_api->minor;
  g_wrapped_api->patch = original_api->patch;
  g_wrapped_api->host_ = original_api->host_;
  g_wrapped_api->ort_api_ = original_api->ort_api_;

  // Assign all wrapped function pointers
  g_wrapped_api->model_load = wrapped_model_load;
  g_wrapped_api->model_delete = wrapped_model_delete;
  g_wrapped_api->model_clone = wrapped_model_clone;
  g_wrapped_api->model_main_graph = wrapped_model_main_graph;
  g_wrapped_api->model_set_meta_data = wrapped_model_set_meta_data;
  g_wrapped_api->model_get_meta_data = wrapped_model_get_meta_data;
  g_wrapped_api->model_has_meta_data = wrapped_model_has_meta_data;

  g_wrapped_api->graph_get_name = wrapped_graph_get_name;
  g_wrapped_api->graph_get_model = wrapped_graph_get_model;
  g_wrapped_api->graph_nodes_unsafe = wrapped_graph_nodes_unsafe;
  g_wrapped_api->graph_get_inputs_unsafe = wrapped_graph_get_inputs_unsafe;
  g_wrapped_api->graph_get_outputs_unsafe = wrapped_graph_get_outputs_unsafe;
  g_wrapped_api->graph_set_outputs = wrapped_graph_set_outputs;
  g_wrapped_api->graph_get_node = wrapped_graph_get_node;
  g_wrapped_api->graph_producer_node = wrapped_graph_producer_node;
  g_wrapped_api->graph_get_node_arg = wrapped_graph_get_node_arg;
  g_wrapped_api->graph_get_all_initialized_tensors =
      wrapped_graph_get_all_initialized_tensors;
  g_wrapped_api->graph_remove_node = wrapped_graph_remove_node;
  g_wrapped_api->graph_add_node = wrapped_graph_add_node;
  g_wrapped_api->graph_save = wrapped_graph_save;
#if MORPHIZEN_ORT_API_MAJOR >= 18
  g_wrapped_api->graph_save_string = wrapped_graph_save_string;
#endif // MORPHIZEN_ORT_API_MAJOR >= 18
  g_wrapped_api->graph_fuse = wrapped_graph_fuse;
  g_wrapped_api->graph_resolve = wrapped_graph_resolve;
  g_wrapped_api->graph_get_consumer_nodes_unsafe =
      wrapped_graph_get_consumer_nodes_unsafe;
  g_wrapped_api->graph_reverse_dfs_from = wrapped_graph_reverse_dfs_from;

  g_wrapped_api->node_get_name = wrapped_node_get_name;
  g_wrapped_api->node_description = wrapped_node_description;
  g_wrapped_api->node_get_index = wrapped_node_get_index;
  g_wrapped_api->node_op_type = wrapped_node_op_type;
  g_wrapped_api->node_op_domain = wrapped_node_op_domain;
  g_wrapped_api->node_get_inputs_unsafe = wrapped_node_get_inputs_unsafe;
  g_wrapped_api->node_get_output_node_args_unsafe =
      wrapped_node_get_output_node_args_unsafe;
  g_wrapped_api->node_get_attributes = wrapped_node_get_attributes;
  g_wrapped_api->node_get_function_body = wrapped_node_get_function_body;
  g_wrapped_api->node_type_is_fused = wrapped_node_type_is_fused;

  g_wrapped_api->node_arg_get_name_unsafe = wrapped_node_arg_get_name_unsafe;
  g_wrapped_api->node_arg_is_exists = wrapped_node_arg_is_exists;
  g_wrapped_api->node_arg_is_constant = wrapped_node_arg_is_constant;
  g_wrapped_api->node_arg_clone = wrapped_node_arg_clone;
  g_wrapped_api->node_arg_new = wrapped_node_arg_new;
  g_wrapped_api->node_arg_get_shape_i64_unsafe =
      wrapped_node_arg_get_shape_i64_unsafe;
  g_wrapped_api->node_arg_get_denotation_unsafe =
      wrapped_node_arg_get_denotation_unsafe;
  g_wrapped_api->node_arg_set_shape_i64 = wrapped_node_arg_set_shape_i64;
  g_wrapped_api->node_arg_set_denotation = wrapped_node_arg_set_denotation;
  g_wrapped_api->node_arg_get_element_type = wrapped_node_arg_get_element_type;
  g_wrapped_api->node_arg_set_element_type = wrapped_node_arg_set_element_type;
  g_wrapped_api->node_arg_get_const_data_as_tensor =
      wrapped_node_arg_get_const_data_as_tensor;

  g_wrapped_api->node_attributes_new = wrapped_node_attributes_new;
  g_wrapped_api->node_attributes_delete = wrapped_node_attributes_delete;
  g_wrapped_api->node_attributes_add = wrapped_node_attributes_add;
  g_wrapped_api->node_attributes_get = wrapped_node_attributes_get;
  g_wrapped_api->node_attributes_get_keys = wrapped_node_attributes_get_keys;

  g_wrapped_api->attr_proto_delete = wrapped_attr_proto_delete;
  g_wrapped_api->attr_proto_clone = wrapped_attr_proto_clone;
  g_wrapped_api->attr_proto_get_name = wrapped_attr_proto_get_name;
  g_wrapped_api->attr_proto_get_type = wrapped_attr_proto_get_type;
  g_wrapped_api->attr_proto_set_name = wrapped_attr_proto_set_name;
  g_wrapped_api->attr_proto_new_int = wrapped_attr_proto_new_int;
  g_wrapped_api->attr_proto_new_float = wrapped_attr_proto_new_float;
  g_wrapped_api->attr_proto_new_string = wrapped_attr_proto_new_string;
  g_wrapped_api->attr_proto_new_tensor = wrapped_attr_proto_new_tensor;
  g_wrapped_api->attr_proto_new_ints = wrapped_attr_proto_new_ints;
  g_wrapped_api->attr_proto_new_floats = wrapped_attr_proto_new_floats;
  g_wrapped_api->attr_proto_new_strings = wrapped_attr_proto_new_strings;
  g_wrapped_api->attr_proto_get_int = wrapped_attr_proto_get_int;
  g_wrapped_api->attr_proto_get_float = wrapped_attr_proto_get_float;
  g_wrapped_api->attr_proto_get_string = wrapped_attr_proto_get_string;
  g_wrapped_api->attr_proto_get_tensor = wrapped_attr_proto_get_tensor;
  g_wrapped_api->attr_proto_get_ints = wrapped_attr_proto_get_ints;
  g_wrapped_api->attr_proto_get_floats = wrapped_attr_proto_get_floats;
  g_wrapped_api->attr_proto_get_strings = wrapped_attr_proto_get_strings;

  g_wrapped_api->tensor_proto_delete = wrapped_tensor_proto_delete;
  g_wrapped_api->tensor_proto_get_shape_unsafe =
      wrapped_tensor_proto_get_shape_unsafe;
  g_wrapped_api->tensor_proto_data_type = wrapped_tensor_proto_data_type;
  g_wrapped_api->tensor_proto_new_floats = wrapped_tensor_proto_new_floats;
  g_wrapped_api->tensor_proto_new_i64 = wrapped_tensor_proto_new_i64;
  g_wrapped_api->tensor_proto_new_i32 = wrapped_tensor_proto_new_i32;
  g_wrapped_api->tensor_proto_new_i8 = wrapped_tensor_proto_new_i8;
  g_wrapped_api->tensor_proto_get_name = wrapped_tensor_proto_get_name;
  g_wrapped_api->tensor_proto_raw_data_size =
      wrapped_tensor_proto_raw_data_size;
  g_wrapped_api->tensor_proto_as_raw = wrapped_tensor_proto_as_raw;

  g_wrapped_api->get_lib_id = wrapped_get_lib_id;
  g_wrapped_api->get_lib_name = wrapped_get_lib_name;
  g_wrapped_api->graph_add_initialized_tensor =
      wrapped_graph_add_initialized_tensor;
  g_wrapped_api->tensor_proto_new_doubles = wrapped_tensor_proto_new_doubles;
  g_wrapped_api->tensor_proto_new_i16 = wrapped_tensor_proto_new_i16;
  g_wrapped_api->tensor_proto_new_u16 = wrapped_tensor_proto_new_u16;
  g_wrapped_api->tensor_proto_new_u32 = wrapped_tensor_proto_new_u32;
  g_wrapped_api->tensor_proto_new_u8 = wrapped_tensor_proto_new_u8;
  g_wrapped_api->tensor_proto_new_u64 = wrapped_tensor_proto_new_u64;
  g_wrapped_api->tensor_proto_new_fp16 = wrapped_tensor_proto_new_fp16;
  g_wrapped_api->tensor_proto_new_bf16 = wrapped_tensor_proto_new_bf16;
  g_wrapped_api->get_model_path = wrapped_get_model_path;
  g_wrapped_api->create_empty_model = wrapped_create_empty_model;
  g_wrapped_api->graph_set_inputs = wrapped_graph_set_inputs;
  g_wrapped_api->node_arg_external_location =
      wrapped_node_arg_external_location;
  g_wrapped_api->session_option_configuration =
      wrapped_session_option_configuration;
  g_wrapped_api->model_to_proto = wrapped_model_to_proto;
  g_wrapped_api->model_proto_serialize_as_string =
      wrapped_model_proto_serialize_as_string;
  g_wrapped_api->model_proto_delete = wrapped_model_proto_delete;
  g_wrapped_api->attr_proto_release_string = wrapped_attr_proto_release_string;
  g_wrapped_api->is_profiling_enabled = wrapped_is_profiling_enabled;
  g_wrapped_api->tensor_proto_new_i4 = wrapped_tensor_proto_new_i4;
  g_wrapped_api->tensor_proto_new_u4 = wrapped_tensor_proto_new_u4;
  g_wrapped_api->graph_remove_initialized_tensor =
      wrapped_graph_remove_initialized_tensor;
  g_wrapped_api->graph_reverse_dfs_from_preemp =
      wrapped_graph_reverse_dfs_from_preemp;
  g_wrapped_api->graph_set_name = wrapped_graph_set_name;
  g_wrapped_api->graph_infer_shapes_from_filepath =
      wrapped_graph_infer_shapes_from_filepath;
  g_wrapped_api->graph_to_graph_proto = wrapped_graph_to_graph_proto;
  g_wrapped_api->graph_proto_delete = wrapped_graph_proto_delete;
  g_wrapped_api->graph_infer_shapes = wrapped_graph_infer_shapes;
#if MORPHIZEN_ORT_API_MAJOR >= 19
  g_wrapped_api->tensor_proto_new_bool = wrapped_tensor_proto_new_bool;
#endif // MORPHIZEN_ORT_API_MAJOR >= 19

  LOG(INFO) << "Created MorphiZen ORT API test coverage wrapper with "
            << sizeof(morphizen::OrtApiForMorphizen) << " bytes";
  return g_wrapped_api.get();
}

void delete_morphizen_ort_api_coverage_test(
    morphizen::OrtApiForMorphizen* wrapped_api) {
  if (wrapped_api == g_wrapped_api.get()) {
    LOG(INFO) << "Deleting MorphiZen ORT API test coverage wrapper";
    LOG(INFO) << "API call statistics:";
    for (const auto& [api_name, count] : g_api_call_counts) {
      LOG(INFO) << "  " << api_name << ": " << count << " calls";
    }
    g_api_call_counts.clear();
    g_wrapped_api.reset();
    g_original_api = nullptr;
  }
}

std::map<std::string, size_t> get_morphizen_ort_api_call_statistics() {
  return g_api_call_counts;
}

void reset_morphizen_ort_api_call_statistics() { g_api_call_counts.clear(); }

} // namespace test
} // namespace morphizen
