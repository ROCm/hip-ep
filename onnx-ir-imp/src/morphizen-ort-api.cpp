/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// NOTE: it is quite tricky to depends on ONNX Runtime C API and C++ API
// because there are two different versions of ONNX
// 1. morphizen_onnx - which is a custom version of ONNX for Morphizen
// 2. onnx  - which is from ONNX Runtime
#undef ONNX_NAMESPACE
#define ONNX_NAMESPACE onnx

#ifndef ORT_API_MANUAL_INIT
#  define ORT_API_MANUAL_INIT 1
#endif
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT
#include "morphizen/morphizen-ort-api-ext.hpp"
#undef ONNX_NAMESPACE
#define ONNX_NAMESPACE morphizen_onnx
// END
// #include "./graph-shape-infer.hpp"
#include "./graph.hpp"
#include "./model.hpp"
#include "./node-arg-index.hpp"
#include "./node-arg.hpp"
#include "./node-index.hpp"
#include "./node.hpp"
#include "morphizen-utils/morphizen_plugin.hpp"
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <glog/logging.h>
#include <map>
#include <string>
#include <vector>
// String constants
static constexpr const char* AMD_MORPHIZEN_DOMAIN = "com.amd.morphizen";

namespace morphizen {

static constexpr const char* kMemAddrTag = "*/_ORT_MEM_ADDR_/*";

static gsl::span<const char>
tensor_proto_get_mem_addr_span(const morphizen_onnx::TensorProto& t) {
  const char* ptr = nullptr;
  size_t length = 0;
  bool found_location = false;
  bool found_offset = false;
  bool found_length = false;
  for (const auto& entry : t.external_data()) {
    if (entry.key() == "location") {
      CHECK_EQ(entry.value(), kMemAddrTag)
          << "EXTERNAL tensor has unexpected location tag: " << entry.value();
      found_location = true;
    } else if (entry.key() == "offset") {
      ptr = reinterpret_cast<const char*>(
          static_cast<uintptr_t>(std::stoull(entry.value())));
      found_offset = true;
    } else if (entry.key() == "length") {
      length = static_cast<size_t>(std::stoull(entry.value()));
      found_length = true;
    }
  }
  CHECK(found_location)
      << "EXTERNAL tensor missing 'location' in external_data";
  CHECK(found_offset) << "EXTERNAL tensor missing 'offset' in external_data";
  CHECK(found_length) << "EXTERNAL tensor missing 'length' in external_data";
  CHECK(ptr != nullptr) << "EXTERNAL tensor 'offset' decoded to null pointer";
  return gsl::span<const char>(ptr, length);
}

// Forward declarations for dummy objects
// static morphizen::TensorProto& get_dummy_tensor();

// Helper function for creating tensor protos with raw_data (bytes) for
// extensibility ONNX DataType enum values: FLOAT=1, UINT8=2, INT8=3, UINT16=4,
// INT16=5, INT32=6, INT64=7, STRING=8, BOOL=9, FLOAT16=10, DOUBLE=11,
// UINT32=12, UINT64=13, COMPLEX64=14, COMPLEX128=15
morphizen::TensorProto*
tensor_proto_new_with_raw_data(const std::string& name,
                               const std::vector<int64_t>& shape, void* data,
                               size_t size, int data_type) {
  auto ret = std::make_unique<morphizen_onnx::TensorProto>();

  // Set the tensor name
  ret->set_name(name);

  // Set the data type (ONNX data type enum)
  ret->set_data_type(data_type);

  // Set the shape dimensions
  for (int64_t dim : shape) {
    ret->add_dims(dim);
  }

  // Set the raw data as bytes for better extensibility
  const char* data_bytes = reinterpret_cast<const char*>(data);
  ret->set_raw_data(data_bytes, size);

  return reinterpret_cast<morphizen::TensorProto*>(ret.release());
}

// Initialize the global API instance with dummy implementations
MorphizenOrtApiExt the_instance_of_morphizen_ort_api;

// Static initialization function to populate the API structure
static void initialize_dummy_api() {
  static bool initialized = false;
  if (initialized)
    return;

  // Version and magic fields for compatibility checking
  the_instance_of_morphizen_ort_api.magic =
      0x50494156; // Binary format compatibility identifier
  the_instance_of_morphizen_ort_api.major = MORPHIZEN_ORT_API_MAJOR;
  the_instance_of_morphizen_ort_api.minor = MORPHIZEN_ORT_API_MINOR;
  the_instance_of_morphizen_ort_api.patch = MORPHIZEN_ORT_API_PATCH;

  // Core pointers
  the_instance_of_morphizen_ort_api.host_ =
      (onnxruntime::ProviderHost*)(void*)1; // it is not in-use
#ifdef ORT_API_VERSION
  the_instance_of_morphizen_ort_api.ort_api_ = &Ort::GetApi();
#else
  the_instance_of_morphizen_ort_api.ort_api_ = nullptr;
#endif // Model API functions [0-6]
  the_instance_of_morphizen_ort_api.model_load =
      [](const std::string& file) -> morphizen::Model* {
    auto model = morphizen::Model::load(file);
    return reinterpret_cast<morphizen::Model*>(model.release());
  };

  the_instance_of_morphizen_ort_api.model_delete =
      [](morphizen::Model* model) -> void {
    if (model) {
      auto* morphizen_model = reinterpret_cast<morphizen::Model*>(model);
      delete morphizen_model;
    }
  };

  the_instance_of_morphizen_ort_api.model_clone =
      [](const morphizen::Model& model,
         int64_t external_data_threshold) -> morphizen::Model* {
    auto* morphizen_model = reinterpret_cast<const morphizen::Model*>(&model);
    auto cloned_model = morphizen_model->clone(external_data_threshold);
    return reinterpret_cast<morphizen::Model*>(cloned_model.release());
  };
  the_instance_of_morphizen_ort_api.model_main_graph =
      [](morphizen::Model& model) -> morphizen::Graph& {
    auto morphizen_model = reinterpret_cast<morphizen::Model*>(&model);
    return const_cast<morphizen::Graph&>(
        *reinterpret_cast<const morphizen::Graph*>(
            &morphizen_model->main_graph()));
  };
  the_instance_of_morphizen_ort_api.model_set_meta_data =
      [](morphizen::Model& model, const std::string& key,
         const std::string& value) -> void {
    auto* morphizen_model = reinterpret_cast<morphizen::Model*>(&model);
    morphizen_model->set_metadata_prop(key, value);
  };

  the_instance_of_morphizen_ort_api.model_get_meta_data =
      [](const morphizen::Model& model,
         const std::string& key) -> morphizen::DllSafe<std::string> {
    auto* morphizen_model = reinterpret_cast<const morphizen::Model*>(&model);
    std::string value = morphizen_model->get_metadata_prop(key);
    return morphizen::DllSafe<std::string>(new std::string(std::move(value)));
  };

  the_instance_of_morphizen_ort_api.model_has_meta_data =
      [](const morphizen::Model& model, const std::string& key) -> int {
    auto* morphizen_model = reinterpret_cast<const morphizen::Model*>(&model);
    return morphizen_model->has_metadata_prop(key) ? 1 : 0;
  };
  // Graph API functions [7-23]
  the_instance_of_morphizen_ort_api.graph_get_name =
      [](const morphizen::Graph& graph) -> const std::string& {
    auto* morphizen_graph = reinterpret_cast<const morphizen::Graph*>(&graph);
    return morphizen_graph->get_name();
  };

  the_instance_of_morphizen_ort_api.graph_get_model =
      [](const morphizen::Graph& graph) -> const morphizen::Model& {
    return *reinterpret_cast<const morphizen::Model*>(
        &reinterpret_cast<const morphizen::Graph*>(&graph)->get_model());
  };
  the_instance_of_morphizen_ort_api.graph_nodes_unsafe =
      [](const morphizen::Graph& graph)
      -> morphizen::DllSafe<std::vector<const morphizen::Node*>> {
    auto* morphizen_graph = reinterpret_cast<const morphizen::Graph*>(&graph);
    auto node_indices = morphizen_graph->nodes_unsafe();
    auto result = new std::vector<const morphizen::Node*>();
    result->reserve(node_indices.size());
    for (const auto& node_idx : node_indices) {
      // Get the Node from the NodeIndex using the helper method
      const auto* node = static_cast<const morphizen::Node*>(
          node_idx.to_morphizen_core_node_ptr());
      result->push_back(reinterpret_cast<const morphizen::Node*>(node));
    }
    return morphizen::DllSafe<std::vector<const morphizen::Node*>>(result);
  };
  the_instance_of_morphizen_ort_api.graph_get_inputs_unsafe =
      [](const morphizen::Graph& graph)
      -> morphizen::DllSafe<std::vector<const morphizen::NodeArg*>> {
    auto* morphizen_graph = reinterpret_cast<const morphizen::Graph*>(&graph);
    auto inputs = morphizen_graph->get_inputs_unsafe();
    auto result = new std::vector<const morphizen::NodeArg*>();
    result->reserve(inputs.size());
    for (const auto& input_idx : inputs) {
      // Convert NodeArgIndex back to morphizen::NodeArg* using the conversion
      // method
      const auto* node_arg_ptr = static_cast<const morphizen::NodeArg*>(
          input_idx.to_morphizen_core_node_arg_ptr());
      result->push_back(static_cast<const morphizen::NodeArg*>(node_arg_ptr));
    }
    return morphizen::DllSafe<std::vector<const morphizen::NodeArg*>>(result);
  };
  the_instance_of_morphizen_ort_api.graph_get_outputs_unsafe =
      [](const morphizen::Graph& graph)
      -> morphizen::DllSafe<std::vector<const morphizen::NodeArg*>> {
    auto* morphizen_graph = reinterpret_cast<const morphizen::Graph*>(&graph);
    auto outputs = morphizen_graph->get_outputs_unsafe();
    auto result = new std::vector<const morphizen::NodeArg*>();
    result->reserve(outputs.size());
    for (const auto& output_idx : outputs) {
      // Convert NodeArgIndex back to morphizen::NodeArg* using the conversion
      // method
      const auto* node_arg_ptr = static_cast<const morphizen::NodeArg*>(
          output_idx.to_morphizen_core_node_arg_ptr());
      result->push_back(static_cast<const morphizen::NodeArg*>(node_arg_ptr));
    }
    return morphizen::DllSafe<std::vector<const morphizen::NodeArg*>>(result);
  };
  the_instance_of_morphizen_ort_api.graph_set_outputs =
      [](morphizen::Graph& graph,
         gsl::span<const morphizen::NodeArg* const> outputs) -> void {
    auto* morphizen_graph = reinterpret_cast<morphizen::Graph*>(&graph);
    // Convert morphizen::NodeArg* to NodeArgIndex for the morphizen API
    std::vector<NodeArgIndex> converted_outputs;
    converted_outputs.reserve(outputs.size());
    for (auto* output : outputs) {
      auto node_arg = NodeArgIndex::from_morphizen_core_node_arg_ptr(output);
      CHECK(node_arg.is_valid(*morphizen_graph))
          << "NodeArgIndex is not valid in graph_set_outputs"
          << node_arg.to_string()
          << " in graph-id: " << morphizen_graph->get_graph_id().to_string();
      converted_outputs.push_back(node_arg);
    }
    morphizen_graph->set_outputs(
        gsl::span<const NodeArgIndex>(converted_outputs));
  };
  the_instance_of_morphizen_ort_api.graph_get_node =
      [](const morphizen::Graph& graph,
         size_t index) -> const morphizen::Node* {
    auto* morphizen_graph = reinterpret_cast<const morphizen::Graph*>(&graph);
    auto node_index = NodeIndex::from_morphizen_core_node_index(index);
    auto graph_id = morphizen_graph->get_graph_id();
    if (!node_index.is_valid(*morphizen_graph)) {
      LOG(ERROR) << "Invalid NodeIndex: " << node_index.to_string()
                 << " in graph-id: " << graph_id.to_string();
      return nullptr;
    }
    return static_cast<const morphizen::Node*>(
        node_index.to_morphizen_core_node_ptr());
  };
  the_instance_of_morphizen_ort_api.graph_producer_node =
      [](const morphizen::Graph& graph,
         const std::string& node_arg_name) -> const morphizen::Node* {
    auto* morphizen_graph = reinterpret_cast<const morphizen::Graph*>(&graph);
    const auto& result = morphizen_graph->producer_node(node_arg_name);
    return static_cast<const morphizen::Node*>(
        result.to_morphizen_core_node_ptr());
  };
  the_instance_of_morphizen_ort_api.graph_get_node_arg =
      [](const morphizen::Graph& graph,
         const std::string& name) -> const morphizen::NodeArg* {
    auto* morphizen_graph = reinterpret_cast<const morphizen::Graph*>(&graph);
    const auto& result = morphizen_graph->get_node_arg(name);
    return static_cast<const morphizen::NodeArg*>(
        result.to_morphizen_core_node_arg_ptr());
  };
  the_instance_of_morphizen_ort_api.graph_get_all_initialized_tensors =
      [](const morphizen::Graph& graph)
      -> const morphizen::InitializedTensorSet& {
    auto* morphizen_graph = reinterpret_cast<const morphizen::Graph*>(&graph);
    const auto& result = morphizen_graph->get_all_initialized_tensors();
    return reinterpret_cast<const morphizen::InitializedTensorSet&>(result);
  };

  the_instance_of_morphizen_ort_api.graph_remove_node =
      [](morphizen::Graph& graph,
         const morphizen::NodeInput& node_input) -> void {
    auto* morphizen_graph = reinterpret_cast<morphizen::Graph*>(&graph);

    morphizen_graph->remove_node(
        NodeIndex::from_morphizen_core_node_ptr(node_input.node));
  };
  the_instance_of_morphizen_ort_api.graph_add_node =
      [](morphizen::Graph& graph, const std::string& name,
         const std::string& op_type, const std::string& description,
         const std::vector<const morphizen::NodeArg*>& input_args,
         const std::vector<const morphizen::NodeArg*>& output_args,
         const morphizen::NodeAttributes& attributes,
         const std::string& domain) -> morphizen::Node& {
    auto* morphizen_graph = reinterpret_cast<morphizen::Graph*>(&graph);

    // Convert input args
    std::vector<NodeArgIndex> converted_inputs;
    converted_inputs.reserve(input_args.size());
    for (auto* arg : input_args) {
      converted_inputs.push_back(
          NodeArgIndex::from_morphizen_core_node_arg_ptr(arg));
    }

    // Convert output args
    std::vector<NodeArgIndex> converted_outputs;
    converted_outputs.reserve(output_args.size());
    for (auto* arg : output_args) {
      converted_outputs.push_back(
          NodeArgIndex::from_morphizen_core_node_arg_ptr(arg));
    }

    // Convert attributes
    auto* attrs_field = const_cast<
        google::protobuf::RepeatedPtrField<morphizen_onnx::AttributeProto>*>(
        reinterpret_cast<const google::protobuf::RepeatedPtrField<
            morphizen_onnx::AttributeProto>*>(&attributes));

    NodeIndex result =
        morphizen_graph->add_node(name, op_type, description, converted_inputs,
                                  converted_outputs, attrs_field, domain);
    return *const_cast<morphizen::Node*>(static_cast<const morphizen::Node*>(
        result.to_morphizen_core_node_ptr()));
  };

  the_instance_of_morphizen_ort_api.graph_save =
      [](const morphizen::Graph& graph, const std::string& filename,
         const std::string& dat_filename,
         size_t external_data_threshold) -> void {
    auto* morphizen_graph = reinterpret_cast<const morphizen::Graph*>(&graph);
    morphizen_graph->save(filename, dat_filename, external_data_threshold);
  };

#if MORPHIZEN_ORT_API_MAJOR >= 18
  the_instance_of_morphizen_ort_api.graph_save_string =
      [](const morphizen::Graph& graph) -> morphizen::DllSafe<std::string> {
    auto* morphizen_graph = reinterpret_cast<const morphizen::Graph*>(&graph);
    auto model_string = morphizen_graph->save_string();
    return morphizen::DllSafe<std::string>(
        new std::string(std::move(model_string)));
  };
#endif // MORPHIZEN_ORT_API_MAJOR >= 18

  the_instance_of_morphizen_ort_api.graph_fuse =
      [](morphizen::Graph& graph, const std::string& name,
         const std::string& op_type, const std::vector<size_t>& nodes,
         const std::vector<std::string>& inputs,
         const std::vector<std::string>& outputs,
         const std::vector<std::string>& constant_initializers)
      -> morphizen::Node& {
    auto* morphizen_graph = reinterpret_cast<morphizen::Graph*>(&graph);
    auto node_index = morphizen_graph->fuse(name, op_type, nodes, inputs,
                                            outputs, constant_initializers);
    return *const_cast<morphizen::Node*>(static_cast<const morphizen::Node*>(
        node_index.to_morphizen_core_node_ptr()));
  };

  the_instance_of_morphizen_ort_api.graph_resolve = [](morphizen::Graph& graph,
                                                       bool force) -> int {
    auto* morphizen_graph = reinterpret_cast<morphizen::Graph*>(&graph);
    return morphizen_graph->resolve(force);
  };
  the_instance_of_morphizen_ort_api.graph_get_consumer_nodes_unsafe =
      [](const morphizen::Graph& graph, const std::string& node_arg_name)
      -> morphizen::DllSafe<std::vector<const morphizen::Node*>> {
    auto* morphizen_graph = reinterpret_cast<const morphizen::Graph*>(&graph);
    auto node_indices = morphizen_graph->get_consumer_nodes(node_arg_name);
    auto result = new std::vector<const morphizen::Node*>();
    result->reserve(node_indices.size());
    for (const auto& node_idx : node_indices) {
      // Get the Node from the NodeIndex using the helper method
      const auto* node = static_cast<const morphizen::Node*>(
          node_idx.to_morphizen_core_node_ptr());
      result->push_back(reinterpret_cast<const morphizen::Node*>(node));
    }
    return morphizen::DllSafe<std::vector<const morphizen::Node*>>(result);
  };
  the_instance_of_morphizen_ort_api.graph_reverse_dfs_from =
      [](const morphizen::Graph& graph,
         gsl::span<const morphizen::Node* const> from,
         const std::function<void(const morphizen::Node*)>& enter,
         const std::function<void(const morphizen::Node*)>& leave,
         const std::function<bool(const morphizen::Node* /*from*/,
                                  const morphizen::Node* /*to*/)>& stop)
      -> void {
    auto* morphizen_graph = reinterpret_cast<const morphizen::Graph*>(&graph);

    // Convert morphizen::Node* span to NodeIndex vector
    std::vector<NodeIndex> node_indices;
    node_indices.reserve(from.size());
    for (const auto* node : from) {
      CHECK(node != nullptr) << "Node pointer is null in reverse_dfs_from";
      auto node_index = NodeIndex::from_morphizen_core_node_ptr(node);
      CHECK(node_index.is_valid(*morphizen_graph))
          << "NodeIndex is not valid in reverse_dfs_from: "
          << node_index.to_string()
          << " in graph-id: " << morphizen_graph->get_graph_id().to_string();
      node_indices.push_back(node_index);
    }

    // Create wrapper functions that convert NodeIndex back to morphizen::Node*
    auto enter_wrapper = enter ? [&enter](const NodeIndex& node_index) -> bool {
      enter(static_cast<const morphizen::Node*>(
          node_index.to_morphizen_core_node_ptr()));
      return false;
    }
    : std::function<bool(const NodeIndex&)>();

    auto leave_wrapper = leave ? [&leave](const NodeIndex& node_index) -> bool {
      const auto* node_ptr = static_cast<const morphizen::Node*>(
          node_index.to_morphizen_core_node_ptr());
      leave(node_ptr);
      return false;
    }
    : std::function<bool(const NodeIndex&)>{};

    auto stop_wrapper = stop
        ? [&stop](const NodeIndex& from_idx, const NodeIndex& to_idx) -> bool {
      const auto* from_ptr = static_cast<const morphizen::Node*>(
          from_idx.to_morphizen_core_node_ptr());
      const auto* to_ptr = static_cast<const morphizen::Node*>(
          to_idx.to_morphizen_core_node_ptr());
      return stop(from_ptr, to_ptr);
    }
    : std::function<bool(const NodeIndex&, const NodeIndex&)>{};

    // Call the morphizen graph reverse_dfs_from method
    bool include_staging_graph = false; // keep same ORT semantics
    morphizen_graph->reverse_dfs_from_preemp(
        gsl::span<const NodeIndex>(node_indices), enter_wrapper, leave_wrapper,
        nullptr, stop_wrapper, include_staging_graph);
  };
  // Node API functions [24-33]
  the_instance_of_morphizen_ort_api.node_get_name =
      [](const morphizen::Node& node) -> const std::string& {
    // Convert morphizen::Node to NodeIndex and get the node name
    auto node_index = NodeIndex::from_morphizen_core_node_ptr(&node);
    return node_index.get_name();
  };
  the_instance_of_morphizen_ort_api.node_description =
      [](const morphizen::Node& node) -> const std::string& {
    // Convert morphizen::Node to NodeIndex and get the description
    auto node_index = NodeIndex::from_morphizen_core_node_ptr(&node);
    return node_index.get_description();
  };
  the_instance_of_morphizen_ort_api.node_get_index =
      [](const morphizen::Node& node) -> size_t {
    return reinterpret_cast<size_t>(&node);
  };
  the_instance_of_morphizen_ort_api.node_op_type =
      [](const morphizen::Node& node) -> const std::string& {
    // Convert morphizen::Node to NodeIndex and get the op type
    auto node_index = NodeIndex::from_morphizen_core_node_ptr(&node);
    return node_index.get_node_op_type();
  };

  the_instance_of_morphizen_ort_api.node_op_domain =
      [](const morphizen::Node& node) -> const std::string& {
    // Convert morphizen::Node to NodeIndex and get the op domain
    auto node_index = NodeIndex::from_morphizen_core_node_ptr(&node);
    return node_index.get_node_op_domain();
  };
  the_instance_of_morphizen_ort_api.node_get_inputs_unsafe =
      [](const morphizen::Node& node)
      -> morphizen::DllSafe<std::vector<morphizen::NodeInput>> {
    auto node_index = NodeIndex::from_morphizen_core_node_ptr(&node);
    auto input_nodes_args = node_index.get_input_node_args();
    auto result = std::vector<morphizen::NodeInput>();
    result.reserve(input_nodes_args.size());
    for (const auto& input_arg : input_nodes_args) {
      // Convert NodeArgIndex to morphizen::NodeInput
      if (!input_arg.is_valid()) {
        result.push_back(morphizen::NodeInput{nullptr, nullptr});
      } else {
        auto input_node =
            input_arg.get_producer_node(); // Returns NodeIndex by value
        auto morphizen_core_node_ptr =
            input_node.is_valid() ? static_cast<const morphizen::Node*>(
                                        input_node.to_morphizen_core_node_ptr())
                                  : nullptr;
        result.push_back(morphizen::NodeInput{
            morphizen_core_node_ptr,
            static_cast<const morphizen::NodeArg*>(
                input_arg.to_morphizen_core_node_arg_ptr())});
      }
    }
    return morphizen::DllSafe<std::vector<morphizen::NodeInput>>(result);
  };
  the_instance_of_morphizen_ort_api.node_get_implicit_inputs_unsafe =
      [](const morphizen::Node& /*node*/)
      -> morphizen::DllSafe<std::vector<morphizen::NodeInput>> {
    LOG(FATAL) << "node_get_implicit_inputs_unsafe is not implemented in "
                  "the onnx-ir-imp backend; Loop / If / Scan models are "
                  "not supported by this backend yet.";
    return morphizen::DllSafe<std::vector<morphizen::NodeInput>>(
        new std::vector<morphizen::NodeInput>());
  };

  the_instance_of_morphizen_ort_api.node_get_output_node_args_unsafe =
      [](const morphizen::Node& node)
      -> morphizen::DllSafe<std::vector<const morphizen::NodeArg*>> {
    auto node_index = NodeIndex::from_morphizen_core_node_ptr(&node);
    auto& output_node_args =
        node_index.get_output_node_args(); // Get outputs as NodeArgIndex
    // optimize this API impl (ref PR#362)
    // The PSS fuse_transpose pass time : 228ms -> 179ms
    auto ret = std::vector<const morphizen::NodeArg*>(
        *reinterpret_cast<const std::vector<const morphizen::NodeArg*>*>(
            &output_node_args));
    return morphizen::DllSafe<std::vector<const morphizen::NodeArg*>>(
        std::move(ret));
  };

  the_instance_of_morphizen_ort_api.node_get_attributes =
      [](morphizen::Node& node) -> morphizen::NodeAttributes& {
    auto morphizen_node_index = NodeIndex::from_morphizen_core_node_ptr(&node);
    auto result = morphizen_node_index.get_attributes();
    CHECK(result != nullptr)
        << "NodeIndex::node_get_attributes() returned nullptr";
    return const_cast<morphizen::NodeAttributes&>(
        *reinterpret_cast<const morphizen::NodeAttributes*>(result));
  };

  the_instance_of_morphizen_ort_api.node_get_function_body =
      [](const morphizen::Node& node) -> const morphizen::Graph& {
    auto morphizen_node_index = NodeIndex::from_morphizen_core_node_ptr(&node);

    return *reinterpret_cast<const morphizen::Graph*>(
        morphizen_node_index.get_function_body());
  };

  the_instance_of_morphizen_ort_api.node_type_is_fused =
      [](const morphizen::Node& node) -> bool {
    // Convert morphizen::Node to NodeIndex and check if it is fused
    auto node_index = NodeIndex::from_morphizen_core_node_ptr(&node);
    // Use the is_fused() member function to check if the node is fused
    return node_index.is_fused_node();
  };

  // Continue with remaining functions...
  // This is a comprehensive implementation but
  // would be very long For brevity, I'll
  // initialize the remaining function pointers
  // to nullptr and add them as needed

  // NodeArg API functions [34-45]
  the_instance_of_morphizen_ort_api.node_arg_get_name_unsafe =
      [](const morphizen::NodeArg& node_arg) -> const std::string& {
    // Convert the morphizen::NodeArg reference to NodeArgIndex
    auto node_arg_index =
        NodeArgIndex::from_morphizen_core_node_arg_ptr(&node_arg);
    // Use the get_name_unsafe() member function to get the name
    const std::string* name = node_arg_index.get_name_unsafe();
    CHECK(name != nullptr)
        << "NodeArgIndex::get_name_unsafe() returned nullptr";
    return *name;
  };
  the_instance_of_morphizen_ort_api.node_arg_is_exists =
      [](const morphizen::NodeArg& node_arg) -> bool {
    // Convert the morphizen::NodeArg reference to NodeArgIndex
    auto node_arg_index =
        NodeArgIndex::from_morphizen_core_node_arg_ptr(&node_arg);
    // Use the exists() member function to check if the node argument exists
    return node_arg_index.exists();
  };

  the_instance_of_morphizen_ort_api.node_arg_is_constant =
      [](const morphizen::Graph& /*graph*/,
         const morphizen::NodeArg& node_arg) -> bool {
    auto node_arg_index =
        NodeArgIndex::from_morphizen_core_node_arg_ptr(&node_arg);
    return node_arg_index.is_valid_initializer();
  };
  the_instance_of_morphizen_ort_api.node_arg_clone =
      [](morphizen::Graph& graph, const morphizen::NodeArg& node_arg,
         const std::string& name) -> morphizen::NodeArg& {
    auto* morphizen_graph = reinterpret_cast<morphizen::Graph*>(&graph);
    auto* morphizen_node_arg =
        reinterpret_cast<const morphizen::NodeArg*>(&node_arg);
    void* cloned_arg =
        morphizen_graph->node_arg_clone(*morphizen_node_arg, name);
    return *reinterpret_cast<morphizen::NodeArg*>(cloned_arg);
  };

  the_instance_of_morphizen_ort_api.node_arg_new =
      [](morphizen::Graph& graph, const std::string& name,
         const std::vector<int64_t>* shape,
         int element_type) -> morphizen::NodeArg& {
    auto* morphizen_graph = reinterpret_cast<morphizen::Graph*>(&graph);
    auto node_arg_index =
        morphizen_graph->node_arg_new(name, shape, element_type);
    const auto* node_arg_ptr = static_cast<const morphizen::NodeArg*>(
        node_arg_index.to_morphizen_core_node_arg_ptr());
    return *const_cast<morphizen::NodeArg*>(node_arg_ptr);
  };

  the_instance_of_morphizen_ort_api.node_arg_get_shape_i64_unsafe =
      [](const morphizen::NodeArg& node_arg)
      -> morphizen::DllSafe<std::vector<int64_t>> {
    // convert the morphizen::NodeArg reference to NodeArgIndex
    auto node_arg_index =
        NodeArgIndex::from_morphizen_core_node_arg_ptr(&node_arg);
    // Use the get_shape_i64_unsafe() member function to get the shape
    auto shape = node_arg_index.get_shape_i64_unsafe();
    // Return the shape wrapped in DllSafe
    return morphizen::DllSafe<std::vector<int64_t>>(shape);
  };

  the_instance_of_morphizen_ort_api.node_arg_get_denotation_unsafe =
      [](const morphizen::NodeArg& node_arg)
      -> morphizen::DllSafe<std::vector<std::string>> {
    // convert the morphizen::NodeArg reference to NodeArgIndex
    auto node_arg_index =
        NodeArgIndex::from_morphizen_core_node_arg_ptr(&node_arg);
    // Use the get_denotation_unsafe() member function to get the denotation
    auto denotation = node_arg_index.get_denotation_unsafe();
    // Return the denotation wrapped in DllSafe
    if (denotation == nullptr) {
      denotation = node_arg_index.get_denotation_unsafe();
    }
    return morphizen::DllSafe<std::vector<std::string>>(denotation);
  };
  // for graph inputs only (batch size)
  the_instance_of_morphizen_ort_api.node_arg_set_shape_i64 =
      [](const morphizen::NodeArg& node_arg,
         const std::vector<int64_t>& shape) -> void {
    // convert the morphizen::NodeArg reference to NodeArgIndex
    auto node_arg_index =
        NodeArgIndex::from_morphizen_core_node_arg_ptr(&node_arg);
    // Use the set_shape_i64() member function to set the shape
    node_arg_index.set_shape_i64(shape);
  };

  the_instance_of_morphizen_ort_api.node_arg_set_denotation =
      [](const morphizen::NodeArg& node_arg,
         const std::vector<std::string>& denotation) -> void {
    // convert the morphizen::NodeArg reference to NodeArgIndex
    auto node_arg_index =
        NodeArgIndex::from_morphizen_core_node_arg_ptr(&node_arg);
    // Use the set_denotation() member function to set the denotation
    node_arg_index.set_denotation(denotation);
  };

  the_instance_of_morphizen_ort_api.node_arg_get_element_type =
      [](const morphizen::NodeArg& node_arg) -> int {
    auto node_arg_index =
        NodeArgIndex::from_morphizen_core_node_arg_ptr(&node_arg);
    return node_arg_index.get_element_type();
  };

  the_instance_of_morphizen_ort_api.node_arg_set_element_type =
      [](morphizen::NodeArg& /*node_arg*/, int /*data_type*/) -> void {
    // WARNING: this function is not in use.
    LOG(FATAL) << "node_arg_set_element_type is not implemented yet";
  };
  the_instance_of_morphizen_ort_api.node_arg_get_const_data_as_tensor =
      [](const morphizen::Graph& graph,
         const morphizen::NodeArg& node_arg) -> const morphizen::TensorProto& {
    auto* morphizen_graph = reinterpret_cast<const morphizen::Graph*>(&graph);
    auto morphizen_node_arg =
        NodeArgIndex::from_morphizen_core_node_arg_ptr(&node_arg);

    auto tensor_ptr =
        morphizen_node_arg.get_const_data_as_tensor(*morphizen_graph);
    return *reinterpret_cast<const morphizen::TensorProto*>(tensor_ptr);
  };

  // NodeAttributes API functions [46-50]
  the_instance_of_morphizen_ort_api.node_attributes_new =
      []() -> morphizen::NodeAttributes* {
    auto* attrs = new google::protobuf::RepeatedPtrField<
        morphizen_onnx::AttributeProto>();
    return reinterpret_cast<morphizen::NodeAttributes*>(attrs);
  };

  the_instance_of_morphizen_ort_api.node_attributes_delete =
      [](morphizen::NodeAttributes* p) -> void {
    if (p) {
      auto* attrs = reinterpret_cast<
          google::protobuf::RepeatedPtrField<morphizen_onnx::AttributeProto>*>(
          p);
      delete attrs;
    }
  };

  the_instance_of_morphizen_ort_api.node_attributes_add =
      [](morphizen::NodeAttributes& p,
         morphizen::AttributeProto&& attr) -> void {
    auto* attrs = reinterpret_cast<
        google::protobuf::RepeatedPtrField<morphizen_onnx::AttributeProto>*>(
        &p);
    auto* morphizen_attr =
        reinterpret_cast<morphizen_onnx::AttributeProto*>(&attr);
    auto it = std::find_if(
        attrs->begin(), attrs->end(),
        [morphizen_attr](const morphizen_onnx::AttributeProto& existing_attr) {
          return existing_attr.name() == morphizen_attr->name();
        });
    if (it != attrs->end()) {
      // If the attribute already exists, replace it
      *it = std::move(*morphizen_attr);
    } else {
      // If it doesn't exist, add a new attribute
      auto* new_attr = attrs->Add();
      *new_attr = std::move(*morphizen_attr);
    }
  };

  the_instance_of_morphizen_ort_api.node_attributes_get =
      [](const morphizen::NodeAttributes& p,
         const std::string& name) -> const morphizen::AttributeProto* {
    auto* attrs = reinterpret_cast<const google::protobuf::RepeatedPtrField<
        morphizen_onnx::AttributeProto>*>(&p);
    for (const auto& attr : *attrs) {
      if (attr.name() == name) {
        return reinterpret_cast<const morphizen::AttributeProto*>(&attr);
      }
    }
    return nullptr;
  };

  the_instance_of_morphizen_ort_api.node_attributes_get_keys =
      [](morphizen::NodeAttributes& p)
      -> morphizen::DllSafe<std::vector<std::string>> {
    auto* attrs = reinterpret_cast<
        google::protobuf::RepeatedPtrField<morphizen_onnx::AttributeProto>*>(
        &p);
    CHECK(attrs != nullptr) << "NodeAttributes is nullptr";
    auto* keys = new std::vector<std::string>();
    keys->reserve(attrs->size());
    for (const auto& attr : *attrs) {
      keys->push_back(attr.name());
    }
    return morphizen::DllSafe<std::vector<std::string>>(std::move(keys));
  };

  // AttributeProto API functions [51-69]
  the_instance_of_morphizen_ort_api.attr_proto_delete =
      [](morphizen::AttributeProto* attr) -> void {
    if (attr) {
      auto* morphizen_attr =
          reinterpret_cast<morphizen_onnx::AttributeProto*>(attr);
      delete morphizen_attr;
    }
  };

  the_instance_of_morphizen_ort_api.attr_proto_clone =
      [](const morphizen::AttributeProto& attr) -> morphizen::AttributeProto* {
    auto* morphizen_attr =
        reinterpret_cast<const morphizen_onnx::AttributeProto*>(&attr);
    auto* cloned_attr = new morphizen_onnx::AttributeProto(*morphizen_attr);
    return reinterpret_cast<morphizen::AttributeProto*>(cloned_attr);
  };

  the_instance_of_morphizen_ort_api.attr_proto_get_name =
      [](const morphizen::AttributeProto& attr) -> const std::string& {
    auto* morphizen_attr =
        reinterpret_cast<const morphizen_onnx::AttributeProto*>(&attr);
    return morphizen_attr->name();
  };

  the_instance_of_morphizen_ort_api.attr_proto_get_type =
      [](const morphizen::AttributeProto& attr) -> int {
    auto* morphizen_attr =
        reinterpret_cast<const morphizen_onnx::AttributeProto*>(&attr);
    return morphizen_attr->type();
  };

  the_instance_of_morphizen_ort_api.attr_proto_set_name =
      [](morphizen::AttributeProto* attr, const std::string& name) -> void {
    auto* morphizen_attr =
        reinterpret_cast<morphizen_onnx::AttributeProto*>(attr);
    morphizen_attr->set_name(name);
  };

  the_instance_of_morphizen_ort_api.attr_proto_new_int =
      [](const std::string& name, int64_t value) -> morphizen::AttributeProto* {
    auto* attr = new morphizen_onnx::AttributeProto();
    attr->set_name(name);
    attr->set_type(morphizen_onnx::AttributeProto::INT);
    attr->set_i(value);
    return reinterpret_cast<morphizen::AttributeProto*>(attr);
  };

  the_instance_of_morphizen_ort_api.attr_proto_new_float =
      [](const std::string& name, float value) -> morphizen::AttributeProto* {
    auto* attr = new morphizen_onnx::AttributeProto();
    attr->set_name(name);
    attr->set_type(morphizen_onnx::AttributeProto::FLOAT);
    attr->set_f(value);
    return reinterpret_cast<morphizen::AttributeProto*>(attr);
  };

  the_instance_of_morphizen_ort_api.attr_proto_new_string =
      [](const std::string& name,
         const std::string& value) -> morphizen::AttributeProto* {
    auto* attr = new morphizen_onnx::AttributeProto();
    attr->set_name(name);
    attr->set_type(morphizen_onnx::AttributeProto::STRING);
    attr->set_s(value);
    return reinterpret_cast<morphizen::AttributeProto*>(attr);
  };

  the_instance_of_morphizen_ort_api.attr_proto_new_tensor =
      [](const std::string& name,
         const morphizen::TensorProto& value) -> morphizen::AttributeProto* {
    auto* attr = new morphizen_onnx::AttributeProto();
    attr->set_name(name);
    attr->set_type(morphizen_onnx::AttributeProto::TENSOR);
    auto* morphizen_tensor =
        reinterpret_cast<const morphizen_onnx::TensorProto*>(&value);
    *attr->mutable_t() = *morphizen_tensor;
    return reinterpret_cast<morphizen::AttributeProto*>(attr);
  };

  the_instance_of_morphizen_ort_api.attr_proto_new_ints =
      [](const std::string& name,
         const std::vector<int64_t>& value) -> morphizen::AttributeProto* {
    auto* attr = new morphizen_onnx::AttributeProto();
    attr->set_name(name);
    attr->set_type(morphizen_onnx::AttributeProto::INTS);
    for (int64_t v : value) {
      attr->add_ints(v);
    }
    return reinterpret_cast<morphizen::AttributeProto*>(attr);
  };

  the_instance_of_morphizen_ort_api.attr_proto_new_floats =
      [](const std::string& name,
         const std::vector<float>& value) -> morphizen::AttributeProto* {
    auto* attr = new morphizen_onnx::AttributeProto();
    attr->set_name(name);
    attr->set_type(morphizen_onnx::AttributeProto::FLOATS);
    for (float v : value) {
      attr->add_floats(v);
    }
    return reinterpret_cast<morphizen::AttributeProto*>(attr);
  };

  the_instance_of_morphizen_ort_api.attr_proto_new_strings =
      [](const std::string& name,
         const std::vector<std::string>& value) -> morphizen::AttributeProto* {
    auto* attr = new morphizen_onnx::AttributeProto();
    attr->set_name(name);
    attr->set_type(morphizen_onnx::AttributeProto::STRINGS);
    for (const std::string& v : value) {
      attr->add_strings(v);
    }
    return reinterpret_cast<morphizen::AttributeProto*>(attr);
  };

  the_instance_of_morphizen_ort_api.attr_proto_get_int =
      [](const morphizen::AttributeProto& attr) -> int64_t {
    auto* morphizen_attr =
        reinterpret_cast<const morphizen_onnx::AttributeProto*>(&attr);
    CHECK(morphizen_attr != nullptr)
        << "morphizen_attr is nullptr in attr_proto_get_int";
    CHECK_EQ(morphizen_attr->type(), morphizen_onnx::AttributeProto::INT);
    return morphizen_attr->i();
  };

  the_instance_of_morphizen_ort_api.attr_proto_get_float =
      [](const morphizen::AttributeProto& attr) -> float {
    auto* morphizen_attr =
        reinterpret_cast<const morphizen_onnx::AttributeProto*>(&attr);
    CHECK(morphizen_attr != nullptr)
        << "morphizen_attr is nullptr in attr_proto_get_float";
    CHECK_EQ(morphizen_attr->type(), morphizen_onnx::AttributeProto::FLOAT);
    return morphizen_attr->f();
  };

  the_instance_of_morphizen_ort_api.attr_proto_get_string =
      [](const morphizen::AttributeProto& attr) -> const std::string& {
    auto* morphizen_attr =
        reinterpret_cast<const morphizen_onnx::AttributeProto*>(&attr);
    CHECK(morphizen_attr != nullptr)
        << "morphizen_attr is nullptr in attr_proto_get_string";
    CHECK_EQ(morphizen_attr->type(), morphizen_onnx::AttributeProto::STRING);
    return morphizen_attr->s();
  };
  the_instance_of_morphizen_ort_api.attr_proto_get_tensor =
      [](const morphizen::AttributeProto& attr)
      -> const morphizen::TensorProto& {
    auto* morphizen_attr =
        reinterpret_cast<const morphizen_onnx::AttributeProto*>(&attr);
    CHECK(morphizen_attr != nullptr)
        << "morphizen_attr is nullptr in attr_proto_get_tensor";
    CHECK_EQ(morphizen_attr->type(), morphizen_onnx::AttributeProto::TENSOR);
    return *reinterpret_cast<const morphizen::TensorProto*>(
        &morphizen_attr->t());
  };

  the_instance_of_morphizen_ort_api.attr_proto_get_ints =
      [](const morphizen::AttributeProto& attr) -> gsl::span<const int64_t> {
    auto* morphizen_attr =
        reinterpret_cast<const morphizen_onnx::AttributeProto*>(&attr);
    CHECK(morphizen_attr != nullptr)
        << "morphizen_attr is nullptr in attr_proto_get_ints";
    CHECK_EQ(morphizen_attr->type(), morphizen_onnx::AttributeProto::INTS);
    return gsl::span<const int64_t>(morphizen_attr->ints());
  };

  the_instance_of_morphizen_ort_api.attr_proto_get_floats =
      [](const morphizen::AttributeProto& attr) -> gsl::span<const float> {
    auto* morphizen_attr =
        reinterpret_cast<const morphizen_onnx::AttributeProto*>(&attr);
    CHECK(morphizen_attr != nullptr)
        << "morphizen_attr is nullptr in attr_proto_get_floats";
    CHECK_EQ(morphizen_attr->type(), morphizen_onnx::AttributeProto::FLOATS);
    return gsl::span<const float>(morphizen_attr->floats());
  };

  the_instance_of_morphizen_ort_api.attr_proto_get_strings =
      [](const morphizen::AttributeProto& attr) -> std::vector<std::string> {
    auto* morphizen_attr =
        reinterpret_cast<const morphizen_onnx::AttributeProto*>(&attr);
    CHECK(morphizen_attr != nullptr)
        << "morphizen_attr is nullptr in attr_proto_get_strings";
    CHECK_EQ(morphizen_attr->type(), morphizen_onnx::AttributeProto::STRINGS);
    auto& s = morphizen_attr->strings();
    return std::vector<std::string>(s.begin(), s.end());
  };

  // TensorProto API functions [70-89]
  the_instance_of_morphizen_ort_api.tensor_proto_delete =
      [](morphizen::TensorProto* tp) -> void {
    if (tp) {
      auto* morphizen_tensor =
          reinterpret_cast<morphizen_onnx::TensorProto*>(tp);
      delete morphizen_tensor;
    }
  };

  the_instance_of_morphizen_ort_api.tensor_proto_get_shape_unsafe =
      [](const morphizen::TensorProto& tensor_proto)
      -> morphizen::DllSafe<std::vector<int64_t>> {
    auto* morphizen_tensor =
        reinterpret_cast<const morphizen_onnx::TensorProto*>(&tensor_proto);
    auto shape = std::make_unique<std::vector<int64_t>>();
    shape->reserve(morphizen_tensor->dims_size());
    for (int i = 0; i < morphizen_tensor->dims_size(); ++i) {
      shape->push_back(morphizen_tensor->dims(i));
    }
    return morphizen::DllSafe<std::vector<int64_t>>(shape.release());
  };

  the_instance_of_morphizen_ort_api.tensor_proto_data_type =
      [](const morphizen::TensorProto& tensor_proto) -> int {
    auto* morphizen_tensor =
        reinterpret_cast<const morphizen_onnx::TensorProto*>(&tensor_proto);
    return morphizen_tensor->data_type();
  };

  the_instance_of_morphizen_ort_api.tensor_proto_new_floats =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<float>& data) -> morphizen::TensorProto* {
    return tensor_proto_new_with_raw_data(
        name, shape,
        const_cast<void*>(reinterpret_cast<const void*>(data.data())),
        data.size() * sizeof(float),
        1); // FLOAT = 1
  };

  the_instance_of_morphizen_ort_api.tensor_proto_new_i64 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<int64_t>& data) -> morphizen::TensorProto* {
    return tensor_proto_new_with_raw_data(
        name, shape,
        const_cast<void*>(reinterpret_cast<const void*>(data.data())),
        data.size() * sizeof(int64_t),
        7); // INT64 = 7
  };

  the_instance_of_morphizen_ort_api.tensor_proto_new_i32 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<int32_t>& data) -> morphizen::TensorProto* {
    return tensor_proto_new_with_raw_data(
        name, shape,
        const_cast<void*>(reinterpret_cast<const void*>(data.data())),
        data.size() * sizeof(int32_t),
        6); // INT32 = 6
  };

  the_instance_of_morphizen_ort_api.tensor_proto_new_i8 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<int8_t>& data) -> morphizen::TensorProto* {
    return tensor_proto_new_with_raw_data(
        name, shape,
        const_cast<void*>(reinterpret_cast<const void*>(data.data())),
        data.size() * sizeof(int8_t),
        3); // INT8 = 3
  };

  the_instance_of_morphizen_ort_api.tensor_proto_get_name =
      [](const morphizen::TensorProto& tensor_proto) -> const std::string& {
    auto* morphizen_tensor =
        reinterpret_cast<const morphizen_onnx::TensorProto*>(&tensor_proto);
    return morphizen_tensor->name();
  };

  the_instance_of_morphizen_ort_api.tensor_proto_raw_data_size =
      [](const morphizen::TensorProto& tensor_proto) -> size_t {
    auto* t =
        reinterpret_cast<const morphizen_onnx::TensorProto*>(&tensor_proto);
    if (t->data_location() == morphizen_onnx::TensorProto::EXTERNAL) {
      return tensor_proto_get_mem_addr_span(*t).size();
    }
    return t->raw_data().size();
  };

  the_instance_of_morphizen_ort_api.tensor_proto_as_raw =
      [](const morphizen::Graph& /*graph*/,
         const morphizen::TensorProto& tensor_proto) -> gsl::span<const char> {
    auto* t =
        reinterpret_cast<const morphizen_onnx::TensorProto*>(&tensor_proto);
    if (t->data_location() == morphizen_onnx::TensorProto::EXTERNAL) {
      return tensor_proto_get_mem_addr_span(*t);
    }
    const std::string& raw_data = t->raw_data();
    return gsl::span<const char>(raw_data.data(), raw_data.size());
  };

  // Library info functions [80-81]
  the_instance_of_morphizen_ort_api.get_lib_id =
      []() -> morphizen::DllSafe<std::string> {
    return morphizen::DllSafe<std::string>(new std::string("v1.0.0"));
  };

  the_instance_of_morphizen_ort_api.get_lib_name =
      []() -> morphizen::DllSafe<std::string> {
    return morphizen::DllSafe<std::string>(
        new std::string("morphizen-onnx-imp"));
  };
  // Additional new API functions [82+]
  the_instance_of_morphizen_ort_api.graph_add_initialized_tensor =
      [](morphizen::Graph& graph,
         const morphizen::TensorProto& tensor) -> void {
    auto* morphizen_graph = reinterpret_cast<morphizen::Graph*>(&graph);
    morphizen_graph->add_initialized_tensor(
        *reinterpret_cast<const morphizen_onnx::TensorProto*>(&tensor));
  };

  the_instance_of_morphizen_ort_api.tensor_proto_new_doubles =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<double>& data) -> morphizen::TensorProto* {
    return tensor_proto_new_with_raw_data(
        name, shape,
        const_cast<void*>(reinterpret_cast<const void*>(data.data())),
        data.size() * sizeof(double),
        11); // DOUBLE = 11
  };

  the_instance_of_morphizen_ort_api.tensor_proto_new_i16 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<int16_t>& data) -> morphizen::TensorProto* {
    return tensor_proto_new_with_raw_data(
        name, shape,
        const_cast<void*>(reinterpret_cast<const void*>(data.data())),
        data.size() * sizeof(int16_t),
        5); // INT16 = 5
  };

  the_instance_of_morphizen_ort_api.tensor_proto_new_u16 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<uint16_t>& data) -> morphizen::TensorProto* {
    return tensor_proto_new_with_raw_data(
        name, shape,
        const_cast<void*>(reinterpret_cast<const void*>(data.data())),
        data.size() * sizeof(uint16_t),
        4); // UINT16 = 4
  };

  the_instance_of_morphizen_ort_api.tensor_proto_new_u32 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<uint32_t>& data) -> morphizen::TensorProto* {
    return tensor_proto_new_with_raw_data(
        name, shape,
        const_cast<void*>(reinterpret_cast<const void*>(data.data())),
        data.size() * sizeof(uint32_t),
        12); // UINT32 = 12
  };

  the_instance_of_morphizen_ort_api.tensor_proto_new_u8 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<uint8_t>& data) -> morphizen::TensorProto* {
    return tensor_proto_new_with_raw_data(
        name, shape,
        const_cast<void*>(reinterpret_cast<const void*>(data.data())),
        data.size() * sizeof(uint8_t),
        2); // UINT8 = 2
  };

  the_instance_of_morphizen_ort_api.tensor_proto_new_u64 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<uint64_t>& data) -> morphizen::TensorProto* {
    return tensor_proto_new_with_raw_data(
        name, shape,
        const_cast<void*>(reinterpret_cast<const void*>(data.data())),
        data.size() * sizeof(uint64_t),
        13); // UINT64 = 13
  };

  the_instance_of_morphizen_ort_api.tensor_proto_new_fp16 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<int16_t>& data) -> morphizen::TensorProto* {
    return tensor_proto_new_with_raw_data(
        name, shape,
        const_cast<void*>(reinterpret_cast<const void*>(data.data())),
        data.size() * sizeof(int16_t),
        10); // FLOAT16 = 10
  };

  the_instance_of_morphizen_ort_api.tensor_proto_new_bf16 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<int16_t>& data) -> morphizen::TensorProto* {
    return tensor_proto_new_with_raw_data(
        name, shape,
        const_cast<void*>(reinterpret_cast<const void*>(data.data())),
        data.size() * sizeof(int16_t),
        16); // BFLOAT16 = 16
  };

  the_instance_of_morphizen_ort_api.get_model_path =
      [](const morphizen::Graph& graph) -> const std::filesystem::path& {
    auto morphizen_graph = reinterpret_cast<const morphizen::Graph*>(&graph);
    return morphizen_graph->get_model_path();
  };

  the_instance_of_morphizen_ort_api.create_empty_model =
      [](const std::filesystem::path& path,
         const std::vector<std::pair<std::string, int64_t>>& opset)
      -> morphizen::Model* {
    // Create a new empty ModelProto
    auto model_proto = morphizen_onnx::ModelProto();

    // Set basic model information
    model_proto.set_ir_version(8); // ONNX IR version 8 is widely supported
    model_proto.set_producer_name("morphizen-ort-bridge");
    model_proto.set_producer_version("1.0.0");
    model_proto.set_domain(AMD_MORPHIZEN_DOMAIN);
    model_proto.set_model_version(1);
    model_proto.set_doc_string("Empty model created by morphizen");

    // Add opset imports
    for (const auto& [domain, version] : opset) {
      auto* opset_import = model_proto.add_opset_import();
      opset_import->set_domain(domain);
      opset_import->set_version(version);
    }

    // If no opset was provided, add the default ONNX opset
    if (opset.empty()) {
      auto* default_opset = model_proto.add_opset_import();
      default_opset->set_domain(""); // Empty domain means ai.onnx
      default_opset->set_version(
          17); // ONNX opset 17 is stable and widely supported
    }

    // Create an empty graph
    auto* graph_proto = model_proto.mutable_graph();
    graph_proto->set_name("main_graph");

    // Set model path as metadata if provided
    if (!path.empty()) {
      auto* metadata_prop = model_proto.add_metadata_props();
      metadata_prop->set_key("model_path");
      metadata_prop->set_value(path.u8string());
    }

    // Create the Model instance using the existing factory method
    auto model = morphizen::Model::create_model(std::move(model_proto));
    if (!path.empty()) {
      model->set_model_path(path);
    }
    // Release ownership and return as morphizen::Model*
    return reinterpret_cast<morphizen::Model*>(model.release());
  };
  the_instance_of_morphizen_ort_api.graph_set_inputs =
      [](morphizen::Graph& graph,
         gsl::span<const morphizen::NodeArg* const> inputs) -> void {
    auto* morphizen_graph = reinterpret_cast<morphizen::Graph*>(&graph);
    std::vector<morphizen::NodeArgIndex> morphizen_inputs;
    morphizen_inputs.reserve(inputs.size());
    for (const auto* input : inputs) {
      CHECK(input != nullptr)
          << "Input NodeArg pointer is null in graph_set_inputs";
      auto input_index = NodeArgIndex::from_morphizen_core_node_arg_ptr(input);
      morphizen_inputs.push_back(input_index);
    }
    morphizen_graph->set_inputs(morphizen_inputs);
  };

  the_instance_of_morphizen_ort_api.node_arg_external_location =
      [](const morphizen::Graph& /*graph*/, const morphizen::NodeArg& node_arg,
         std::string& file, size_t& offset, size_t& size,
         size_t& checksum) -> int {
    auto node_arg_index =
        NodeArgIndex::from_morphizen_core_node_arg_ptr(&node_arg);
    return node_arg_index.external_location(file, offset, size, checksum);
  };

  the_instance_of_morphizen_ort_api.session_option_configuration =
      [](void* /*mmap*/, void* /*session_option*/,
         void (* /*push*/)(void* /*mmap*/, const char* /*name*/,
                           const char* /*value*/)) -> void {
    // this function is not used in the new ABI flow.
    LOG(WARNING) << "session_option_configuration is not implemented yet";
  };
  the_instance_of_morphizen_ort_api.model_to_proto =
      [](morphizen::Model& model) -> morphizen::ModelProto* {
    auto* morphizen_model = reinterpret_cast<morphizen::Model*>(&model);
    // Return the ModelProto newly created from the morphizen::Model
    // this function is used to implement generic custom op. see #PR/83
    return reinterpret_cast<morphizen::ModelProto*>(
        const_cast<morphizen_onnx::ModelProto*>(
            new morphizen_onnx::ModelProto(morphizen_model->model_proto())));
  };

  the_instance_of_morphizen_ort_api.model_proto_serialize_as_string =
      [](morphizen::ModelProto& model_proto)
      -> morphizen::DllSafe<std::string> {
    auto* morphizen_model_proto =
        reinterpret_cast<morphizen_onnx::ModelProto*>(&model_proto);
    // Serialize the ModelProto to a string
    // this function is used to implement generic custom op. see #PR/83
    std::unique_ptr<std::string> serialized = std::make_unique<std::string>();
    if (!morphizen_model_proto->SerializeToString(serialized.get())) {
      throw std::runtime_error("Failed to serialize ModelProto");
    }
    return morphizen::DllSafe<std::string>(
        serialized.release()); // Transfer ownership to DllSafe
  };

  the_instance_of_morphizen_ort_api.model_proto_delete =
      [](morphizen::ModelProto* p) -> void {
    if (p) {
      auto* morphizen_model_proto =
          reinterpret_cast<morphizen_onnx::ModelProto*>(p);
      delete morphizen_model_proto;
    }
  };

  the_instance_of_morphizen_ort_api.attr_proto_release_string =
      [](morphizen::AttributeProto* attr) -> morphizen::DllSafe<std::string> {
    auto* morphizen_attr =
        reinterpret_cast<morphizen_onnx::AttributeProto*>(attr);
    CHECK_EQ(morphizen_attr->type(), morphizen_onnx::AttributeProto::STRING);
    return morphizen::DllSafe<std::string>(morphizen_attr->release_s());
  };

  the_instance_of_morphizen_ort_api.is_profiling_enabled =
      [](void* /*session_options*/) -> bool {
    // this function is not in use.
    LOG(WARNING) << "is_profiling_enabled is not implemented yet";
    return false;
  };
#if MORPHIZEN_ORT_API_MAJOR >= 19
  the_instance_of_morphizen_ort_api.tensor_proto_new_bool =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<uint8_t>& data) -> morphizen::TensorProto* {
    return tensor_proto_new_with_raw_data(
        name, shape,
        const_cast<void*>(reinterpret_cast<const void*>(data.data())),
        data.size() * sizeof(uint8_t),
        9); // BOOL = 9
  };
#endif      // MORPHIZEN_ORT_API_MAJOR >= 19

  the_instance_of_morphizen_ort_api.tensor_proto_new_i4 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<int8_t>& data) -> morphizen::TensorProto* {
    return tensor_proto_new_with_raw_data(
        name, shape,
        const_cast<void*>(reinterpret_cast<const void*>(data.data())),
        data.size() * sizeof(int8_t),
        22); // INT4 = 22
  };

  the_instance_of_morphizen_ort_api.tensor_proto_new_u4 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<uint8_t>& data) -> morphizen::TensorProto* {
    return tensor_proto_new_with_raw_data(
        name, shape,
        const_cast<void*>(reinterpret_cast<const void*>(data.data())),
        data.size() * sizeof(uint8_t),
        21); // UINT4 = 21
  };
  // it is only used by
  // morphizen_vaiml_common/graph_update_initializer.cpp:56:
  // actually `MORPHIZEN_ORT_API(graph_remove_node)(graph, {nullptr,
  // node_arg});` is the same thing, but it is obscured and to be deprecacted.
  //
  // in morphizen pass, create-const-op, we need to remove original initializer
  // otherwise ORT graph resolver will fail because of duplicated node arg
  // names.
  the_instance_of_morphizen_ort_api.graph_remove_initialized_tensor =
      [](morphizen::Graph& graph, const std::string& tensor_name) -> void {
    auto* morphizen_graph = reinterpret_cast<morphizen::Graph*>(&graph);
    morphizen_graph->remove_initialized_tensor(tensor_name);
  };
  the_instance_of_morphizen_ort_api.graph_reverse_dfs_from_preemp =
      [](const morphizen::Graph& graph,
         gsl::span<const morphizen::Node* const> from,
         const std::function<bool(const morphizen::Node*)>& enter,
         const std::function<bool(const morphizen::Node*)>& leave,
         const std::function<bool(const morphizen::Node*,
                                  const morphizen::Node*)>& comp,
         const std::function<bool(const morphizen::Node* /*from*/,
                                  const morphizen::Node* /*to*/)>& stop)
      -> void {
    auto* morphizen_graph = reinterpret_cast<const morphizen::Graph*>(&graph);

    // Convert morphizen::Node* span to NodeIndex vector
    std::vector<morphizen::NodeIndex> morphizen_from;
    morphizen_from.reserve(from.size());
    for (const auto* node_ptr : from) {
      morphizen_from.push_back(
          morphizen::NodeIndex::from_morphizen_core_node_ptr(node_ptr));
    }

    // Create wrapper functions that convert between NodeIndex and
    // morphizen::Node*
    auto morphizen_enter =
        enter ? [&enter](const morphizen::NodeIndex& node_idx) -> bool {
      const auto* node_ptr = static_cast<const morphizen::Node*>(
          node_idx.to_morphizen_core_node_ptr());
      return enter(node_ptr);
    }
    : std::function<bool(const morphizen::NodeIndex&)>();

    auto morphizen_leave =
        leave ? [&leave](const morphizen::NodeIndex& node_idx) -> bool {
      const auto* node_ptr = static_cast<const morphizen::Node*>(
          node_idx.to_morphizen_core_node_ptr());
      return leave(node_ptr);
    }
    : std::function<bool(const morphizen::NodeIndex&)>();

    auto morphizen_comp =
        comp ? [&comp](const morphizen::NodeIndex& from_idx,
                       const morphizen::NodeIndex& to_idx) -> bool {
      const auto* from_ptr = static_cast<const morphizen::Node*>(
          from_idx.to_morphizen_core_node_ptr());
      const auto* to_ptr = static_cast<const morphizen::Node*>(
          to_idx.to_morphizen_core_node_ptr());
      return comp(from_ptr, to_ptr);
    }
    : std::function<bool(const morphizen::NodeIndex&,
                         const morphizen::NodeIndex&)>();

    auto morphizen_stop =
        stop ? [&stop](const morphizen::NodeIndex& from_idx,
                       const morphizen::NodeIndex& to_idx) -> bool {
      const auto* from_ptr = static_cast<const morphizen::Node*>(
          from_idx.to_morphizen_core_node_ptr());
      const auto* to_ptr = static_cast<const morphizen::Node*>(
          to_idx.to_morphizen_core_node_ptr());
      return stop(from_ptr, to_ptr);
    }
    : std::function<bool(const morphizen::NodeIndex&,
                         const morphizen::NodeIndex&)>();

    // Call the morphizen graph method with converted parameters
    bool include_staging_graph =
        false; // keep same semantic with ORT implementation
    morphizen_graph->reverse_dfs_from_preemp(
        gsl::span<const morphizen::NodeIndex>(morphizen_from), morphizen_enter,
        morphizen_leave, morphizen_comp, morphizen_stop, include_staging_graph);
  };

  the_instance_of_morphizen_ort_api.graph_set_name =
      [](morphizen::Graph& graph, const char* name) -> void {
    auto* morphizen_graph = reinterpret_cast<morphizen::Graph*>(&graph);
    morphizen_graph->set_graph_name(name);
  };

  the_instance_of_morphizen_ort_api.graph_infer_shapes_from_filepath =
      [](const std::string& m, const std::string& save_path) -> void {
    morphizen_onnx::shape_inference::InferShapes(m, save_path);
  };

  the_instance_of_morphizen_ort_api.graph_to_graph_proto =
      [](const morphizen::Graph& graph) -> morphizen::GraphProto* {
    // WARNING: this function is not in use.
    auto* morphizen_graph = reinterpret_cast<const morphizen::Graph*>(&graph);
    return reinterpret_cast<morphizen::GraphProto*>(
        new morphizen_onnx::GraphProto(morphizen_graph->get_graph_proto()));
  };
  the_instance_of_morphizen_ort_api.graph_proto_delete =
      [](morphizen::GraphProto* p) -> void {
    // WARNING: this function is not in use.
    auto morphizen_graph_proto =
        reinterpret_cast<morphizen_onnx::GraphProto*>(p);
    if (morphizen_graph_proto) {
      delete morphizen_graph_proto;
    }
  };

  the_instance_of_morphizen_ort_api.graph_infer_shapes =
      [](morphizen::ModelProto& m) -> void {
    auto* morphizen_model_proto =
        reinterpret_cast<morphizen_onnx::ModelProto*>(&m);
    morphizen_onnx::shape_inference::InferShapes(*morphizen_model_proto);
  };

  // Stubs: onnx-ir-imp does not run control-flow models (Loop/If/Scan) in
  // this project. The MLIR backend is the only path that exercises the
  // variant AttributeProto subgraph machinery; these stubs abort the
  // session if someone tries to run a control-flow model through
  // onnx-ir-imp.
  the_instance_of_morphizen_ort_api.graph_new_subgraph =
      [](morphizen::Graph& parent) -> morphizen::Graph& {
    LOG(FATAL) << "graph_new_subgraph is not implemented in onnx-ir-imp "
                  "backend; switch to mlir-backend to run control-flow "
                  "models";
    return parent; // unreachable; LOG(FATAL) aborts
  };
  the_instance_of_morphizen_ort_api.attr_proto_new_graph =
      [](const std::string& name,
         morphizen::Graph& /*sub*/) -> morphizen::AttributeProto* {
    LOG(FATAL) << "attr_proto_new_graph is not implemented in onnx-ir-imp "
                  "backend (attr='"
               << name
               << "'); switch to mlir-backend to run control-flow "
                  "models";
    return nullptr; // unreachable; LOG(FATAL) aborts
  };

  the_instance_of_morphizen_ort_api.tensor_proto_new_with_external_data =
      [](const std::string& name, const std::vector<int64_t>& shape,
         int element_type, const std::string& external_data_file, size_t size,
         size_t offset) -> morphizen::TensorProto* {
    auto* tensor_proto = new morphizen_onnx::TensorProto();
    tensor_proto->set_name(name);
    tensor_proto->set_data_type(element_type);
    for (int64_t dim : shape) {
      tensor_proto->add_dims(dim);
    }
    tensor_proto->set_data_location(morphizen_onnx::TensorProto::EXTERNAL);
    auto* external_data = tensor_proto->mutable_external_data();
    auto* location_entry = external_data->Add();
    location_entry->set_key("location");
    location_entry->set_value(external_data_file);

    auto* offset_entry = external_data->Add();
    offset_entry->set_key("offset");
    offset_entry->set_value(std::to_string(offset));

    auto* length_entry = external_data->Add();
    length_entry->set_key("length");
    length_entry->set_value(std::to_string(size));
    return reinterpret_cast<morphizen::TensorProto*>(tensor_proto);
  };
  the_instance_of_morphizen_ort_api.tensor_proto_new_raw_data =
      [](const std::string& name,           //
         const std::vector<int64_t>& shape, //
         int element_type,                  //
         const void* data,                  //
         size_t size) -> morphizen::TensorProto* {
    auto* tensor_proto = new morphizen_onnx::TensorProto();
    tensor_proto->set_name(name);
    tensor_proto->set_data_type(element_type);
    for (int64_t dim : shape) {
      tensor_proto->add_dims(dim);
    }

    tensor_proto->mutable_raw_data()->resize(size);
    std::memcpy(tensor_proto->mutable_raw_data()->data(), data, size);
    return reinterpret_cast<morphizen::TensorProto*>(tensor_proto);
  };

  initialized = true;
}

namespace onnx_ir_imp {
const morphizen::OrtApiForMorphizen* get_morphizen_ort_api() {
  initialize_dummy_api();
  return &the_instance_of_morphizen_ort_api;
}
} // namespace onnx_ir_imp
} // namespace morphizen

namespace onnxruntime {
namespace contrib {
void RegisterContribSchemas();
}
} // namespace onnxruntime
namespace {
const morphizen::OrtApiForMorphizen*
morphizen_onnx_ir_imp_get_morphizen_ort_api() {
  onnxruntime::contrib::RegisterContribSchemas();
  return morphizen::onnx_ir_imp::get_morphizen_ort_api();
}

static ::morphizen::StaticPluginRegister
    __register(morphizen::kONNXIRBackend, "morphizen_ort_api_imp",
               (void*)&morphizen_onnx_ir_imp_get_morphizen_ort_api);
} // namespace
