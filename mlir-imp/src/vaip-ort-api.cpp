/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// NOTE: MLIR implementatio
// the_mlir_instance_of_vaip_ort_api.graph_get_inputs_unsafe = This provides an
// MLIR-based implementation of the same API interface used by the ONNX
// implementation, allowing MLIR dialects to be used as an alternative IR
// representation for VAIP passes.
#undef ONNX_NAMESPACE
#define ONNX_NAMESPACE onnx
#include "morphizen/vaip-ort-api-ext.hpp"
#undef ONNX_NAMESPACE
#define ONNX_NAMESPACE morphizen_onnx
// END

// MLIR implementation includes
#include "mlir-context-manager.hpp"
#include "mlir-graph.hpp"
#include "mlir-model.hpp"
#include "mlir-named-attribute.hpp"
#include "mlir-node-arg-index.hpp"
#include "mlir-node-arg.hpp"
#include "mlir-node-attributes.hpp"
#include "mlir-node.hpp"
// #include "mlir-tensor.hpp"  // Replaced by enhanced MLIRNodeArg

// MLIR includes
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

// Plugin registration
#include "morphizen-utils/vaip_plugin.hpp"

// local
#include "./mlir-constants.hpp"
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <glog/logging.h>
#include <string>
#include <unordered_set>
#include <vector>

// String constants
static constexpr const char* AMD_MORPHIZEN_MLIR_DOMAIN =
    "com.amd.morphizen.mlir";

namespace morphizen {

// Static assertion to ensure safe reinterpret_cast between types
static_assert(sizeof(vaip_core::AttributeProto*) ==
                  sizeof(mlir::NamedAttribute*),
              "vaip_core::AttributeProto* and MLIRNamedAttribute* must have "
              "the same size for safe reinterpret_cast");

// Helper function to create MLIRTensor and return as TensorProto*
// Similar to tensor_proto_new_with_raw_data in ONNX implementation
vaip_core::TensorProto* tensor_proto_new_with_raw_data_mlir(
    const std::string& name, const std::vector<int64_t>& shape,
    const void* data, size_t size, int data_type) {
  // Convert std::vector<int64_t> to llvm::SmallVector<int64_t>
  llvm::SmallVector<int64_t> mlir_shape(shape.begin(), shape.end());

  // Create MLIRTensor and return as TensorProto*
  auto* mlir_node_arg =
      new mlir_impl::MLIRNodeArg(name, mlir_shape, data_type, data, size);

  return reinterpret_cast<vaip_core::TensorProto*>(mlir_node_arg);
}

// Initialize the global API instance with MLIR implementations
VaipOrtApiExt the_mlir_instance_of_vaip_ort_api;

// Static initialization function to populate the API structure with MLIR
// implementations
static void initialize_mlir_api() {
  static bool initialized = false;
  if (initialized)
    return;

  // Version and magic fields for compatibility checking
  the_mlir_instance_of_vaip_ort_api.magic =
      0x50494156; // 'VAIP' in little endian
  the_mlir_instance_of_vaip_ort_api.major = VAIP_ORT_API_MAJOR;
  the_mlir_instance_of_vaip_ort_api.minor = VAIP_ORT_API_MINOR;
  the_mlir_instance_of_vaip_ort_api.patch = VAIP_ORT_API_PATCH;

  // Core pointers
  the_mlir_instance_of_vaip_ort_api.host_ =
      (onnxruntime::ProviderHost*)(void*)1; // Not used in MLIR implementation
#ifdef ORT_API_VERSION
  the_mlir_instance_of_vaip_ort_api.ort_api_ = &Ort::GetApi();
#else
  the_mlir_instance_of_vaip_ort_api.ort_api_ = nullptr;
#endif

  // Model API functions [0-6]
  the_mlir_instance_of_vaip_ort_api.model_load =
      [](const std::string& file) -> vaip_core::Model* {
    auto model = mlir_impl::MLIRModel::load(file);
    return reinterpret_cast<vaip_core::Model*>(model.release());
  };

  the_mlir_instance_of_vaip_ort_api.model_delete =
      [](vaip_core::Model* model) -> void {
    if (model) {
      auto* mlir_model = reinterpret_cast<mlir_impl::MLIRModel*>(model);
      delete mlir_model;
    }
  };

  the_mlir_instance_of_vaip_ort_api.model_clone =
      [](const vaip_core::Model& model,
         int64_t external_data_threshold) -> vaip_core::Model* {
    auto* mlir_model = reinterpret_cast<const mlir_impl::MLIRModel*>(&model);
    auto cloned_model = mlir_model->clone(external_data_threshold);
    return reinterpret_cast<vaip_core::Model*>(cloned_model.release());
  };

  the_mlir_instance_of_vaip_ort_api.model_main_graph =
      [](vaip_core::Model& model) -> vaip_core::Graph& {
    auto* mlir_model = reinterpret_cast<mlir_impl::MLIRModel*>(&model);
    return reinterpret_cast<vaip_core::Graph&>(mlir_model->main_graph());
  };

  the_mlir_instance_of_vaip_ort_api.model_set_meta_data =
      [](vaip_core::Model& model, const std::string& key,
         const std::string& value) -> void {
    auto* mlir_model = reinterpret_cast<mlir_impl::MLIRModel*>(&model);
    mlir_model->set_metadata_prop(key, value);
  };

  the_mlir_instance_of_vaip_ort_api.model_get_meta_data =
      [](const vaip_core::Model& model,
         const std::string& key) -> vaip_core::DllSafe<std::string> {
    auto* mlir_model = reinterpret_cast<const mlir_impl::MLIRModel*>(&model);
    std::string value = mlir_model->get_metadata_prop(key);
    return vaip_core::DllSafe<std::string>(new std::string(std::move(value)));
  };

  the_mlir_instance_of_vaip_ort_api.model_has_meta_data =
      [](const vaip_core::Model& model, const std::string& key) -> int {
    auto* mlir_model = reinterpret_cast<const mlir_impl::MLIRModel*>(&model);
    return mlir_model->has_metadata_prop(key) ? 1 : 0;
  };

  // Graph API functions [7-23] - Basic implementations
  the_mlir_instance_of_vaip_ort_api.graph_get_name =
      [](const vaip_core::Graph& graph) -> const std::string& {
    auto* mlir_graph = reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph);
    return mlir_graph->get_name();
  };

  the_mlir_instance_of_vaip_ort_api.graph_set_name =
      [](vaip_core::Graph& graph, const char* name) -> void {
    auto* mlir_graph = reinterpret_cast<mlir_impl::MLIRGraph*>(&graph);
    mlir_graph->set_name(name);
  };

  the_mlir_instance_of_vaip_ort_api.graph_get_model =
      [](const vaip_core::Graph& graph) -> const vaip_core::Model& {
    auto* mlir_graph = reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph);
    return reinterpret_cast<const vaip_core::Model&>(mlir_graph->get_model());
  };

  the_mlir_instance_of_vaip_ort_api.graph_nodes_unsafe =
      [](const vaip_core::Graph& graph)
      -> vaip_core::DllSafe<std::vector<const vaip_core::Node*>> {
    auto* mlir_graph = reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph);
    auto nodes = mlir_graph->nodes_unsafe();
    auto result = new std::vector<const vaip_core::Node*>();
    result->reserve(nodes.size());
    for (const auto* op : nodes) {
      result->push_back(reinterpret_cast<const vaip_core::Node*>(op));
    }
    return vaip_core::DllSafe<std::vector<const vaip_core::Node*>>(result);
  };

  the_mlir_instance_of_vaip_ort_api.graph_get_inputs_unsafe =
      [](const vaip_core::Graph& graph)
      -> vaip_core::DllSafe<std::vector<const vaip_core::NodeArg*>> {
    auto* mlir_graph = reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph);
    auto inputs = mlir_graph->get_inputs();
    CHECK(!inputs.empty()) << "MLIRGraph inputs are empty ";
    auto result = new std::vector<const vaip_core::NodeArg*>();
    result->reserve(inputs.size());
    for (const auto& value : inputs) {
      result->push_back(reinterpret_cast<const vaip_core::NodeArg*>(
          value.to_vaip_core_node_arg_ptr()));
    }
    return vaip_core::DllSafe<std::vector<const vaip_core::NodeArg*>>(result);
  };

  the_mlir_instance_of_vaip_ort_api.graph_get_outputs_unsafe =
      [](const vaip_core::Graph& graph)
      -> vaip_core::DllSafe<std::vector<const vaip_core::NodeArg*>> {
    auto* mlir_graph = reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph);
    auto result = new std::vector<const vaip_core::NodeArg*>();

    auto outputs = mlir_graph->get_outputs();
    result->reserve(outputs.size());
    for (const auto& value : outputs) {
      result->push_back(reinterpret_cast<const vaip_core::NodeArg*>(
          value.to_vaip_core_node_arg_ptr()));
    }
    return vaip_core::DllSafe<std::vector<const vaip_core::NodeArg*>>(result);
  };

  the_mlir_instance_of_vaip_ort_api.graph_set_outputs =
      [](vaip_core::Graph& graph,
         gsl::span<const vaip_core::NodeArg* const> outputs) -> void {
    auto* mlir_graph = reinterpret_cast<mlir_impl::MLIRGraph*>(&graph);

    // Convert from NodeArg* span to MLIRNodeArgIndex span
    llvm::SmallVector<mlir_impl::MLIRNodeArgIndex> mlir_outputs;
    mlir_outputs.reserve(outputs.size());
    for (const auto* nodeArg : outputs) {
      // Convert NodeArg* to MLIRNodeArgIndex using the raw pointer constructor
      auto node_arg_index =
          mlir_impl::MLIRNodeArgIndex::from_vaip_core_node_arg_ptr(nodeArg);
      mlir_outputs.push_back(node_arg_index);
    }
    mlir_graph->set_outputs(mlir_outputs);
  };

  the_mlir_instance_of_vaip_ort_api.graph_set_inputs =
      [](vaip_core::Graph& graph,
         gsl::span<const vaip_core::NodeArg* const> inputs) -> void {
    auto* mlir_graph = reinterpret_cast<mlir_impl::MLIRGraph*>(&graph);

    // Convert from NodeArg* span to mlir::Value span
    llvm::SmallVector<mlir_impl::MLIRNodeArgIndex> mlir_inputs;
    mlir_inputs.reserve(inputs.size());
    for (const auto* nodeArg : inputs) {
      // Convert NodeArg* back to mlir::Value
      auto node_arg_index =
          mlir_impl::MLIRNodeArgIndex::from_vaip_core_node_arg_ptr(nodeArg);
      mlir_inputs.push_back(node_arg_index);
    }
    mlir_graph->set_inputs(mlir_inputs);
  };

  the_mlir_instance_of_vaip_ort_api.create_empty_model =
      [](const std::filesystem::path& path,
         const std::vector<std::pair<std::string, int64_t>>& opset)
      -> vaip_core::Model* {
    auto model = mlir_impl::MLIRModel::create_empty(path, opset);
    return reinterpret_cast<vaip_core::Model*>(model.release());
  };

  // Tensor proto API functions - implemented using MLIRTensor with helper
  // function
  the_mlir_instance_of_vaip_ort_api.tensor_proto_new_floats =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<float>& data) -> vaip_core::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(
        name, shape, data.data(), data.size() * sizeof(float), 1); // FLOAT = 1
  };

  the_mlir_instance_of_vaip_ort_api.tensor_proto_new_i64 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<int64_t>& data) -> vaip_core::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(name, shape, data.data(),
                                               data.size() * sizeof(int64_t),
                                               7); // INT64 = 7
  };

  the_mlir_instance_of_vaip_ort_api.tensor_proto_new_i32 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<int32_t>& data) -> vaip_core::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(name, shape, data.data(),
                                               data.size() * sizeof(int32_t),
                                               6); // INT32 = 6
  };

  the_mlir_instance_of_vaip_ort_api.tensor_proto_new_i8 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<int8_t>& data) -> vaip_core::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(
        name, shape, data.data(), data.size() * sizeof(int8_t), 3); // INT8 = 3
  };

  // Additional tensor functions
  the_mlir_instance_of_vaip_ort_api.tensor_proto_new_doubles =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<double>& data) -> vaip_core::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(name, shape, data.data(),
                                               data.size() * sizeof(double),
                                               11); // DOUBLE = 11
  };

  the_mlir_instance_of_vaip_ort_api.tensor_proto_new_u8 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<uint8_t>& data) -> vaip_core::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(name, shape, data.data(),
                                               data.size() * sizeof(uint8_t),
                                               2); // UINT8 = 2
  };

  the_mlir_instance_of_vaip_ort_api.tensor_proto_new_u32 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<uint32_t>& data) -> vaip_core::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(name, shape, data.data(),
                                               data.size() * sizeof(uint32_t),
                                               12); // UINT32 = 12
  };

  the_mlir_instance_of_vaip_ort_api.tensor_proto_new_u64 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<uint64_t>& data) -> vaip_core::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(name, shape, data.data(),
                                               data.size() * sizeof(uint64_t),
                                               13); // UINT64 = 13
  };

  the_mlir_instance_of_vaip_ort_api.tensor_proto_new_i16 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<int16_t>& data) -> vaip_core::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(name, shape, data.data(),
                                               data.size() * sizeof(int16_t),
                                               5); // INT16 = 5
  };

  the_mlir_instance_of_vaip_ort_api.tensor_proto_new_u16 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<uint16_t>& data) -> vaip_core::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(name, shape, data.data(),
                                               data.size() * sizeof(uint16_t),
                                               4); // UINT16 = 4
  };

  the_mlir_instance_of_vaip_ort_api.tensor_proto_new_fp16 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<int16_t>& data) -> vaip_core::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(name, shape, data.data(),
                                               data.size() * sizeof(int16_t),
                                               10); // FLOAT16 = 10
  };

  the_mlir_instance_of_vaip_ort_api.tensor_proto_new_bf16 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<int16_t>& data) -> vaip_core::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(name, shape, data.data(),
                                               data.size() * sizeof(int16_t),
                                               16); // BFLOAT16 = 16
  };

// Additional tensor proto functions for 4-bit types
#if VAIP_ORT_API_MAJOR >= 19
  the_mlir_instance_of_vaip_ort_api.tensor_proto_new_bool =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<uint8_t>& data) -> vaip_core::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(name, shape, data.data(),
                                               data.size() * sizeof(uint8_t),
                                               9); // BOOL = 9
  };
#endif                                             // VAIP_ORT_API_MAJOR >= 19

  the_mlir_instance_of_vaip_ort_api.tensor_proto_new_i4 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<int8_t>& data) -> vaip_core::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(name, shape, data.data(),
                                               data.size() * sizeof(int8_t),
                                               22); // INT4 = 22
  };

  the_mlir_instance_of_vaip_ort_api.tensor_proto_new_u4 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<uint8_t>& data) -> vaip_core::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(name, shape, data.data(),
                                               data.size() * sizeof(uint8_t),
                                               21); // UINT4 = 21
  };

  // Tensor creation with external data
  the_mlir_instance_of_vaip_ort_api.tensor_proto_new_with_external_data =
      [](const std::string& name, const std::vector<int64_t>& shape,
         int element_type, const std::string& external_data_file, size_t size,
         size_t offset) -> vaip_core::TensorProto* {
    // Convert std::vector<int64_t> to llvm::SmallVector<int64_t>
    llvm::SmallVector<int64_t> mlir_shape(shape.begin(), shape.end());

    // Create MLIRNodeArg without data (tensor argument)
    auto* mlir_node_arg =
        new mlir_impl::MLIRNodeArg(name, mlir_shape, element_type);

    // TODO: Add external data file information to MLIRNodeArg
    // For now, we create a placeholder implementation
    LOG(WARNING)
        << "tensor_proto_new_with_external_data: external data support "
           "not fully implemented in MLIR backend. File: "
        << external_data_file << ", size: " << size << ", offset: " << offset;

    return reinterpret_cast<vaip_core::TensorProto*>(mlir_node_arg);
  };

  // Tensor creation with raw data
  the_mlir_instance_of_vaip_ort_api.tensor_proto_new_raw_data =
      [](const std::string& name, const std::vector<int64_t>& shape,
         int element_type, const void* data,
         size_t size) -> vaip_core::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(name, shape, data, size,
                                               element_type);
  };

  // Tensor info functions - using direct casting
  the_mlir_instance_of_vaip_ort_api.tensor_proto_get_shape_unsafe =
      [](const vaip_core::TensorProto& tensor_proto)
      -> vaip_core::DllSafe<std::vector<int64_t>> {
    // Direct cast TensorProto to MLIRTensor
    auto* mlir_tensor =
        reinterpret_cast<const morphizen::mlir_impl::MLIRTensor*>(
            &tensor_proto);
    const auto& shape = mlir_tensor->getShape();
    auto result =
        std::make_unique<std::vector<int64_t>>(shape.begin(), shape.end());
    return vaip_core::DllSafe<std::vector<int64_t>>(result.release());
  };

  the_mlir_instance_of_vaip_ort_api.tensor_proto_data_type =
      [](const vaip_core::TensorProto& tensor_proto) -> int {
    auto* mlir_tensor =
        reinterpret_cast<const morphizen::mlir_impl::MLIRTensor*>(
            &tensor_proto);
    return mlir_tensor->getElementType();
  };

  the_mlir_instance_of_vaip_ort_api.tensor_proto_get_name =
      [](const vaip_core::TensorProto& tensor_proto) -> const std::string& {
    auto* mlir_tensor =
        reinterpret_cast<const morphizen::mlir_impl::MLIRTensor*>(
            &tensor_proto);
    return mlir_tensor->getName();
  };

  the_mlir_instance_of_vaip_ort_api.tensor_proto_raw_data_size =
      [](const vaip_core::TensorProto& tensor_proto) -> size_t {
    auto* mlir_tensor =
        reinterpret_cast<const morphizen::mlir_impl::MLIRTensor*>(
            &tensor_proto);
    return mlir_tensor->getDataSize();
  };

  the_mlir_instance_of_vaip_ort_api.tensor_proto_as_raw =
      [](const vaip_core::Graph& /*graph*/,
         const vaip_core::TensorProto& tensor_proto) -> gsl::span<const char> {
    auto* mlir_tensor =
        reinterpret_cast<const morphizen::mlir_impl::MLIRTensor*>(
            &tensor_proto);
    const void* data = mlir_tensor->getData();
    size_t size = mlir_tensor->getDataSize();
    return gsl::span<const char>(reinterpret_cast<const char*>(data), size);
  };

  the_mlir_instance_of_vaip_ort_api.tensor_proto_delete =
      [](vaip_core::TensorProto* tp) -> void {
    if (tp) {
      // Direct cast and delete as MLIRTensor
      auto* mlir_tensor =
          reinterpret_cast<morphizen::mlir_impl::MLIRTensor*>(tp);
      delete mlir_tensor;
    }
  };

  // Library info functions
  the_mlir_instance_of_vaip_ort_api.get_lib_id =
      []() -> vaip_core::DllSafe<std::string> {
    return vaip_core::DllSafe<std::string>(new std::string("v1.0.0"));
  };

  the_mlir_instance_of_vaip_ort_api.get_lib_name =
      []() -> vaip_core::DllSafe<std::string> {
    return vaip_core::DllSafe<std::string>(
        new std::string("morphizen-mlir-imp"));
  };

  // Initialize remaining function pointers to nullptr for now
  // These would be implemented as needed for MLIR-specific functionality
  the_mlir_instance_of_vaip_ort_api.graph_get_node =
      [](const vaip_core::Graph& graph,
         size_t index) -> const vaip_core::Node* {
    auto* mlir_graph = reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph);
    auto* op = mlir_graph->get_node(index);
    return reinterpret_cast<const vaip_core::Node*>(op);
  };

  the_mlir_instance_of_vaip_ort_api.graph_producer_node =
      [](const vaip_core::Graph& graph,
         const std::string& node_arg_name) -> const vaip_core::Node* {
    auto* mlir_graph = reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph);
    auto* op = mlir_graph->producer_node(node_arg_name);
    return reinterpret_cast<const vaip_core::Node*>(op);
  };
  the_mlir_instance_of_vaip_ort_api.graph_get_node_arg =
      [](const vaip_core::Graph& graph,
         const std::string& name) -> const vaip_core::NodeArg* {
    auto* mlir_graph = reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph);
    auto node_arg_index = mlir_graph->get_node_arg_index(name);
    if (!node_arg_index.is_valid()) {
      return nullptr;
    }
    return reinterpret_cast<const vaip_core::NodeArg*>(
        node_arg_index.to_vaip_core_node_arg_ptr());
  };

  the_mlir_instance_of_vaip_ort_api.graph_get_all_initialized_tensors =
      [](const vaip_core::Graph& graph)
      -> const vaip_core::InitializedTensorSet& {
    auto* mlir_graph = reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph);
    return reinterpret_cast<const vaip_core::InitializedTensorSet&>(
        mlir_graph->get_all_initialized_tensors());
  };
  the_mlir_instance_of_vaip_ort_api.graph_remove_node =
      [](vaip_core::Graph& graph,
         const vaip_core::NodeInput& node_input) -> void {
    auto* mlir_graph = reinterpret_cast<mlir_impl::MLIRGraph*>(&graph);

    // Convert vaip_core::Node* to mlir::Operation*
    // The node_input.node field contains the Node* that should be removed
    if (!node_input.node) {
      return;
    }

    // node_input.node is actually an mlir::Operation* cast to vaip_core::Node*
    auto* op = reinterpret_cast<mlir::Operation*>(
        const_cast<vaip_core::Node*>(node_input.node));

    mlir_graph->remove_node(op);
  };
  the_mlir_instance_of_vaip_ort_api.graph_add_node =
      [](vaip_core::Graph& graph, const std::string& name,
         const std::string& op_type, const std::string& description,
         const std::vector<const vaip_core::NodeArg*>& input_args,
         const std::vector<const vaip_core::NodeArg*>& output_args,
         const vaip_core::NodeAttributes& attributes,
         const std::string& domain) -> vaip_core::Node& {
    auto* mlir_graph = reinterpret_cast<mlir_impl::MLIRGraph*>(&graph);

    // Convert input NodeArg* to MLIRNodeArgIndex
    std::vector<mlir_impl::MLIRNodeArgIndex> mlir_input_args;
    mlir_input_args.reserve(input_args.size());
    for (const auto* nodeArg : input_args) {
      // Convert NodeArg* to MLIRNodeArgIndex using the raw pointer constructor
      auto node_arg_index =
          mlir_impl::MLIRNodeArgIndex::from_vaip_core_node_arg_ptr(nodeArg);
      mlir_input_args.push_back(node_arg_index);
    }

    // Convert output NodeArg* to MLIRNodeArgIndex
    std::vector<mlir_impl::MLIRNodeArgIndex> mlir_output_args;
    mlir_output_args.reserve(output_args.size());
    for (const auto* nodeArg : output_args) {
      // Convert NodeArg* to MLIRNodeArgIndex using the raw pointer constructor
      auto node_arg_index =
          mlir_impl::MLIRNodeArgIndex::from_vaip_core_node_arg_ptr(nodeArg);
      mlir_output_args.push_back(node_arg_index);
    }

    // Convert vaip_core::NodeAttributes to MLIRNodeAttributes
    auto mlir_node_attrs =
        mlir_impl::MLIRNodeAttributes(const_cast<mlir::Operation*>(
            reinterpret_cast<const mlir::Operation*>(&attributes)));

    // Call the MLIRGraph add_node method
    const mlir::Operation* op =
        mlir_graph->add_node(name, op_type, description, mlir_input_args,
                             mlir_output_args, mlir_node_attrs, domain);

    // Convert to vaip_core::Node* and return as non-const reference
    const vaip_core::Node* node = reinterpret_cast<const vaip_core::Node*>(op);
    return *const_cast<vaip_core::Node*>(node);
  };
  the_mlir_instance_of_vaip_ort_api.graph_save =
      [](const vaip_core::Graph& graph, const std::string& filename,
         const std::string& dat_filename,
         size_t external_data_threshold) -> void {
    auto* mlir_graph = reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph);
    mlir_graph->save(filename, dat_filename, external_data_threshold);
  };
  the_mlir_instance_of_vaip_ort_api.graph_fuse =
      [](vaip_core::Graph& graph, const std::string& name,
         const std::string& op_type, const std::vector<size_t>& nodes,
         const std::vector<std::string>& inputs,
         const std::vector<std::string>& outputs,
         const std::vector<std::string>& constant_initializers)
      -> vaip_core::Node& {
    auto* mlir_graph = reinterpret_cast<mlir_impl::MLIRGraph*>(&graph);

    std::vector<mlir_impl::MLIRNodeArgIndex> input_node_args;
    for (auto& input_name : inputs) {
      input_node_args.push_back(mlir_graph->get_node_arg_index(input_name));
    }
    std::vector<mlir_impl::MLIRNodeArgIndex> output_node_args;
    for (auto& output_name : outputs) {
      output_node_args.push_back(mlir_graph->get_node_arg_index(output_name));
    }
    std::vector<mlir_impl::MLIRNodeArgIndex> constant_initializers_args;
    for (auto& const_name : constant_initializers) {
      constant_initializers_args.push_back(
          mlir_graph->get_node_arg_index(const_name));
    }

    std::vector<const mlir::Operation*> mlir_nodes;
    for (size_t node_id : nodes) {
      mlir_nodes.push_back(mlir_graph->get_node(node_id));
    }
    mlir::Operation* fused_op =
        mlir_graph->fuse(name, op_type, mlir_nodes, input_node_args,
                         output_node_args, constant_initializers_args);
    return *reinterpret_cast<vaip_core::Node*>(fused_op);
  };

  the_mlir_instance_of_vaip_ort_api.graph_resolve = [](vaip_core::Graph& graph,
                                                       bool force) -> int {
    auto* mlir_graph = reinterpret_cast<mlir_impl::MLIRGraph*>(&graph);
    return mlir_graph->resolve(force);
  };

  the_mlir_instance_of_vaip_ort_api.graph_get_consumer_nodes_unsafe =
      [](const vaip_core::Graph& graph, const std::string& node_arg_name)
      -> vaip_core::DllSafe<std::vector<const vaip_core::Node*>> {
    auto* mlir_graph = reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph);
    auto mlir_consumer_ops = mlir_graph->get_consumer_nodes(node_arg_name);
    auto consumer_ops =
        reinterpret_cast<const std::vector<const vaip_core::Node*>&>(
            mlir_consumer_ops);
    return vaip_core::DllSafe<std::vector<const vaip_core::Node*>>(
        std::move(consumer_ops));
  };

  the_mlir_instance_of_vaip_ort_api.graph_reverse_dfs_from =
      [](const vaip_core::Graph& graph,
         gsl::span<const vaip_core::Node* const> from,
         const std::function<void(const vaip_core::Node*)>& enter,
         const std::function<void(const vaip_core::Node*)>& leave,
         const std::function<bool(const vaip_core::Node* /*from*/,
                                  const vaip_core::Node* /*to*/)>& stop)
      -> void {
    auto* mlir_graph = reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph);

    // Convert from vaip_core::Node* to mlir::Operation*
    std::vector<const mlir::Operation*> mlir_from;
    mlir_from.reserve(from.size());
    for (const auto* node_ptr : from) {
      mlir_from.push_back(reinterpret_cast<const mlir::Operation*>(node_ptr));
    }

    // Create wrapper functions that convert void callbacks to bool callbacks
    // The non-preemptive version always returns false (don't stop processing)
    auto mlir_enter = enter ? [&enter](const mlir::Operation* op) -> bool {
      const auto* node_ptr = reinterpret_cast<const vaip_core::Node*>(op);
      enter(node_ptr);
      return false; // Never preempt for non-preemptive version
    }
    : std::function<bool(const mlir::Operation*)>();

    auto mlir_leave = leave ? [&leave](const mlir::Operation* op) -> bool {
      const auto* node_ptr = reinterpret_cast<const vaip_core::Node*>(op);
      leave(node_ptr);
      return false; // Never preempt for non-preemptive version
    }
    : std::function<bool(const mlir::Operation*)>();

    auto mlir_stop = stop ? [&stop](const mlir::Operation* from_op,
                                    const mlir::Operation* to_op) -> bool {
      const auto* from_ptr = reinterpret_cast<const vaip_core::Node*>(from_op);
      const auto* to_ptr = reinterpret_cast<const vaip_core::Node*>(to_op);
      return stop(from_ptr, to_ptr);
    }
    : std::function<bool(const mlir::Operation*, const mlir::Operation*)>();

    // Call the MLIRGraph preemptive method with converted parameters
    // No comparison function needed for the non-preemptive version
    mlir_graph->reverse_dfs_from_preemp(
        gsl::span<const mlir::Operation* const>(mlir_from), mlir_enter,
        mlir_leave, nullptr, mlir_stop);
  };

  // Additional graph API functions
  the_mlir_instance_of_vaip_ort_api.graph_add_initialized_tensor =
      [](vaip_core::Graph& graph,
         const vaip_core::TensorProto& tensor) -> void {
    auto* mlir_graph = reinterpret_cast<mlir_impl::MLIRGraph*>(&graph);
    auto* mlir_tensor = reinterpret_cast<const mlir_impl::MLIRTensor*>(&tensor);
    //
    CHECK(mlir_graph != nullptr)
        << "MLIRGraph pointer is null in add_initialized_tensor";
    CHECK(mlir_tensor != nullptr)
        << "MLIRTensor pointer is null in add_initialized_tensor";
    mlir_graph->add_constant_initialized_tensor(mlir_tensor);
  };

  // Node API functions - to be implemented as needed
  the_mlir_instance_of_vaip_ort_api.node_get_name =
      [](const vaip_core::Node& node) -> const std::string& {
    auto mlir_node = mlir_impl::MLIRNode(reinterpret_cast<mlir::Operation*>(
        const_cast<vaip_core::Node*>(&node)));
    static thread_local std::string name_cache;
    name_cache = mlir_node.getName();
    return name_cache;
  };

  the_mlir_instance_of_vaip_ort_api.node_description =
      [](const vaip_core::Node& node) -> const std::string& {
    auto mlir_node = mlir_impl::MLIRNode(reinterpret_cast<mlir::Operation*>(
        const_cast<vaip_core::Node*>(&node)));
    static thread_local std::string description_cache;
    description_cache = mlir_node.getDescription();
    return description_cache;
  };

  the_mlir_instance_of_vaip_ort_api.node_get_index =
      [](const vaip_core::Node& node) -> size_t {
    return reinterpret_cast<size_t>(&node);
  };

  the_mlir_instance_of_vaip_ort_api.node_op_type =
      [](const vaip_core::Node& node) -> const std::string& {
    auto op = mlir_impl::MLIRNode(reinterpret_cast<mlir::Operation*>(
        const_cast<vaip_core::Node*>(&node)));
    static thread_local std::string op_type_str;
    op_type_str = op.getOpType();
    return op_type_str;
  };

  the_mlir_instance_of_vaip_ort_api.node_op_domain =
      [](const vaip_core::Node& node) -> const std::string& {
    auto op = mlir_impl::MLIRNode(reinterpret_cast<mlir::Operation*>(
        const_cast<vaip_core::Node*>(&node)));
    static thread_local std::string domain_str;
    domain_str = op.getDomain();
    return domain_str;
  };

  the_mlir_instance_of_vaip_ort_api.node_get_inputs_unsafe =
      [](const vaip_core::Node& node)
      -> vaip_core::DllSafe<std::vector<vaip_core::NodeInput>> {
    auto result = new std::vector<vaip_core::NodeInput>();

    // Create MLIRNode wrapper for convenient access to node data
    auto mlir_node = mlir_impl::MLIRNode(reinterpret_cast<mlir::Operation*>(
        const_cast<vaip_core::Node*>(&node)));

    // Get input MLIRNodeArgIndex objects using MLIRNode interface
    auto inputNodeArgs = mlir_node.getInputNodeArgs();
    result->reserve(inputNodeArgs.size());

    for (const auto& MLIRNodeArgIndex : inputNodeArgs) {
      // Convert MLIRNodeArgIndex to NodeInput
      result->push_back({reinterpret_cast<const vaip_core::Node*>(
                             MLIRNodeArgIndex.get_producer_node()),
                         reinterpret_cast<const vaip_core::NodeArg*>(
                             MLIRNodeArgIndex.to_vaip_core_node_arg_ptr())});
    }

    return vaip_core::DllSafe<std::vector<vaip_core::NodeInput>>(result);
  };

  the_mlir_instance_of_vaip_ort_api.node_get_output_node_args_unsafe =
      [](const vaip_core::Node& node)
      -> vaip_core::DllSafe<std::vector<const vaip_core::NodeArg*>> {
    auto result = new std::vector<const vaip_core::NodeArg*>();

    // Create MLIRNode wrapper for convenient access to node data
    auto mlir_node = mlir_impl::MLIRNode(reinterpret_cast<mlir::Operation*>(
        const_cast<vaip_core::Node*>(&node)));

    // Get output MLIRNodeArgIndex objects using MLIRNode interface
    auto outputNodeArgs = mlir_node.getOutputNodeArgs();
    result->reserve(outputNodeArgs.size());

    for (const auto& MLIRNodeArgIndex : outputNodeArgs) {
      // Convert MLIRNodeArgIndex to NodeArg*
      auto nodeArg = reinterpret_cast<const vaip_core::NodeArg*>(
          MLIRNodeArgIndex.to_vaip_core_node_arg_ptr());
      result->push_back(nodeArg);
    }

    return vaip_core::DllSafe<std::vector<const vaip_core::NodeArg*>>(result);
  };

  the_mlir_instance_of_vaip_ort_api.node_get_attributes =
      [](vaip_core::Node& node) -> vaip_core::NodeAttributes& {
    return reinterpret_cast<vaip_core::NodeAttributes&>(node);
  };

  the_mlir_instance_of_vaip_ort_api.node_get_function_body =
      [](const vaip_core::Node& node) -> const vaip_core::Graph& {
    // Create MLIRNode wrapper to access function body functionality
    auto mlir_node = mlir_impl::MLIRNode(reinterpret_cast<mlir::Operation*>(
        const_cast<vaip_core::Node*>(&node)));
    return *reinterpret_cast<const vaip_core::Graph*>(
        mlir_node.getFunctionBody());
  };

  the_mlir_instance_of_vaip_ort_api.node_type_is_fused =
      [](const vaip_core::Node& node) -> bool {
    // Create MLIRNode wrapper for convenient access to fused node checking
    auto mlir_node = mlir_impl::MLIRNode(reinterpret_cast<mlir::Operation*>(
        const_cast<vaip_core::Node*>(&node)));

    // Use MLIRNode's isFused method to check if the node represents a fused
    // operation
    return mlir_node.isFused();
  };

  // NodeAttributes API functions - MLIR implementation
  the_mlir_instance_of_vaip_ort_api.node_attributes_new =
      []() -> vaip_core::NodeAttributes* {
    // Create a new MLIRNodeAttributes with owned context
    auto* mlir_attrs = mlir_impl::MLIRNodeAttributes::Create();
    return reinterpret_cast<vaip_core::NodeAttributes*>(mlir_attrs);
  };

  the_mlir_instance_of_vaip_ort_api.node_attributes_delete =
      [](vaip_core::NodeAttributes* p) -> void {
    if (p) {
      // reinterpret_cast<mlir::Operation*>(p);
    }
  };

  the_mlir_instance_of_vaip_ort_api.node_attributes_add =
      [](vaip_core::NodeAttributes& attrs,
         vaip_core::AttributeProto&& attr_proto) -> void {
    auto mlir_node_attrs = mlir_impl::MLIRNodeAttributes(
        reinterpret_cast<mlir::Operation*>(&attrs));
    auto* mlir_named_attr =
        reinterpret_cast<mlir::NamedAttribute*>(&attr_proto);

    // Use the new add method which includes the replacement logic
    mlir_node_attrs.add(*mlir_named_attr);
  };

  the_mlir_instance_of_vaip_ort_api.node_attributes_get =
      [](const vaip_core::NodeAttributes& p,
         const std::string& name) -> const vaip_core::AttributeProto* {
    auto mlir_attrs =
        mlir_impl::MLIRNodeAttributes(const_cast<mlir::Operation*>(
            reinterpret_cast<const mlir::Operation*>(&p)));
    if (mlir_attrs.has_attribute(name)) {
      return reinterpret_cast<const vaip_core::AttributeProto*>(
          &mlir_attrs.get_mlir_attribute(name));
    }
    return nullptr;
  };

  the_mlir_instance_of_vaip_ort_api.node_attributes_get_keys =
      [](vaip_core::NodeAttributes& p)
      -> vaip_core::DllSafe<std::vector<std::string>> {
    auto mlir_attrs =
        mlir_impl::MLIRNodeAttributes(reinterpret_cast<mlir::Operation*>(&p));
    return vaip_core::DllSafe<std::vector<std::string>>(
        mlir_attrs.get_attribute_names());
  };

  // NodeArg API functions - to be implemented as needed
  the_mlir_instance_of_vaip_ort_api.node_arg_get_name_unsafe =
      [](const vaip_core::NodeArg& node_arg) -> const std::string& {
    return mlir_impl::MLIRNodeArgIndex::from_vaip_core_node_arg_ptr(&node_arg)
        .get_name();
  };

  the_mlir_instance_of_vaip_ort_api.node_arg_is_exists =
      [](const vaip_core::NodeArg& node_arg) -> bool {
    // Convert the vaip_core::NodeArg reference to MLIRNodeArgIndex
    auto node_arg_index =
        mlir_impl::MLIRNodeArgIndex::from_vaip_core_node_arg_ptr(&node_arg);
    // Check if the node argument exists using validity check
    return node_arg_index.is_valid();
  };

  the_mlir_instance_of_vaip_ort_api.node_arg_is_constant =
      [](const vaip_core::Graph& /*graph*/,
         const vaip_core::NodeArg& node_arg) -> bool {
    // Convert the vaip_core::NodeArg reference to MLIRNodeArgIndex
    auto node_arg_index =
        mlir_impl::MLIRNodeArgIndex::from_vaip_core_node_arg_ptr(&node_arg);
    return node_arg_index.is_constant();
  };

  the_mlir_instance_of_vaip_ort_api.node_arg_clone =
      [](vaip_core::Graph& graph, const vaip_core::NodeArg& node_arg,
         const std::string& name) -> vaip_core::NodeArg& {
    auto* mlir_graph = reinterpret_cast<mlir_impl::MLIRGraph*>(&graph);
    // TODO: Implement proper node argument cloning for MLIR
    LOG(WARNING) << "node_arg_clone not implemented in MLIR backend for: "
                 << name;
    (void)mlir_graph; // Suppress unused parameter warning
    (void)node_arg;
    // Return original for now - not safe for production use
    return *const_cast<vaip_core::NodeArg*>(&node_arg);
  };

  the_mlir_instance_of_vaip_ort_api.node_arg_new =
      [](vaip_core::Graph& graph, const std::string& name,
         const std::vector<int64_t>* shape,
         int element_type) -> vaip_core::NodeArg& {
    auto* mlir_graph = reinterpret_cast<mlir_impl::MLIRGraph*>(&graph);

    // Convert std::vector to SmallVector
    llvm::SmallVector<int64_t> small_shape;
    if (shape) {
      small_shape.assign(shape->begin(), shape->end());
    }

    auto MLIRNodeArgIndex = mlir_graph->node_arg_new(
        name, shape ? &small_shape : nullptr, element_type);

    // Convert MLIRNodeArgIndex back to NodeArg* for the API
    const auto* node_arg_ptr = static_cast<const vaip_core::NodeArg*>(
        MLIRNodeArgIndex.to_vaip_core_node_arg_ptr());
    return *const_cast<vaip_core::NodeArg*>(node_arg_ptr);
  };

  the_mlir_instance_of_vaip_ort_api.node_arg_get_shape_i64_unsafe =
      [](const vaip_core::NodeArg& node_arg)
      -> vaip_core::DllSafe<std::vector<int64_t>> {
    // Convert the vaip_core::NodeArg reference to MLIRNodeArgIndex
    auto node_arg_index =
        mlir_impl::MLIRNodeArgIndex::from_vaip_core_node_arg_ptr(&node_arg);

    // Use the get_shape_i64_unsafe() member function to get the shape
    auto& shape = node_arg_index.get_shape_i64();
    auto vec_shape = std::vector<int64_t>(shape.begin(), shape.end());
    // Return the shape wrapped in DllSafe
    return vaip_core::DllSafe<std::vector<int64_t>>(vec_shape);
  };

  the_mlir_instance_of_vaip_ort_api.node_arg_get_denotation_unsafe =
      [](const vaip_core::NodeArg& /*node_arg*/)
      -> vaip_core::DllSafe<std::vector<std::string>> {
    //"node_arg_get_denotation_unsafe not implemented in MLIR backend"
    return vaip_core::DllSafe<std::vector<std::string>>(
        std::vector<std::string>{});
  };

  the_mlir_instance_of_vaip_ort_api.node_arg_set_shape_i64 =
      [](const vaip_core::NodeArg& node_arg,
         const std::vector<int64_t>& shape) -> void {
    // Convert the vaip_core::NodeArg reference to MLIRNodeArgIndex
    auto node_arg_index =
        mlir_impl::MLIRNodeArgIndex::from_vaip_core_node_arg_ptr(&node_arg);
    node_arg_index.set_shape_i64(
        llvm::SmallVector<int64_t, 4>(shape.begin(), shape.end()));
  };

  the_mlir_instance_of_vaip_ort_api.node_arg_set_denotation =
      [](const vaip_core::NodeArg& node_arg,
         const std::vector<std::string>& denotation) -> void {
    // Convert the vaip_core::NodeArg reference to MLIRNodeArgIndex
    auto node_arg_index =
        mlir_impl::MLIRNodeArgIndex::from_vaip_core_node_arg_ptr(&node_arg);
    // TODO: Implement denotation setting for MLIR NodeArg
    LOG(WARNING) << "node_arg_set_denotation not implemented in MLIR backend, "
                    "denotation size: "
                 << denotation.size();
    (void)node_arg_index; // Suppress unused parameter warning
  };

  the_mlir_instance_of_vaip_ort_api.node_arg_get_element_type =
      [](const vaip_core::NodeArg& node_arg) -> int {
    // Convert the vaip_core::NodeArg reference to MLIRNodeArgIndex
    auto node_arg_index =
        mlir_impl::MLIRNodeArgIndex::from_vaip_core_node_arg_ptr(&node_arg);
    // Delegate to MLIRNodeArgIndex implementation which calls
    // MLIRNodeArg::getElementType()
    return node_arg_index.get_element_type();
  };

  the_mlir_instance_of_vaip_ort_api.node_arg_set_element_type =
      [](vaip_core::NodeArg& node_arg, int data_type) -> void {
    // Convert the vaip_core::NodeArg reference to MLIRNodeArgIndex
    auto node_arg_index =
        mlir_impl::MLIRNodeArgIndex::from_vaip_core_node_arg_ptr(&node_arg);
    node_arg_index.set_element_type(data_type);
  };

  the_mlir_instance_of_vaip_ort_api.node_arg_get_const_data_as_tensor =
      [](const vaip_core::Graph& /* graph*/,
         const vaip_core::NodeArg& node_arg) -> const vaip_core::TensorProto& {
    // auto* mlir_graph = reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph);
    auto node_arg_index =
        mlir_impl::MLIRNodeArgIndex::from_vaip_core_node_arg_ptr(&node_arg);
    return reinterpret_cast<const vaip_core::TensorProto&>(
        node_arg_index.get_const_data_as_tensor());
  };

  the_mlir_instance_of_vaip_ort_api.node_arg_external_location =
      [](const vaip_core::Graph& /*graph*/,
         const vaip_core::NodeArg& /*node_arg*/, std::string& /*file*/,
         size_t& /*offset*/, size_t& /*size*/, size_t& /*checksum*/) -> int {
    // TODO: Implement external location retrieval for MLIR NodeArg
    return 0;
  };

  // AttributeProto API functions - implemented using MLIRNamedAttribute
  the_mlir_instance_of_vaip_ort_api.attr_proto_new_ints =
      [](const std::string& name,
         const std::vector<int64_t>& data) -> vaip_core::AttributeProto* {
    // Create MLIRNamedAttribute using factory method
    auto mlir_named_attr =
        mlir_impl::MLIRNamedAttribute::create_int_array(name, data);

    // Release ownership and return as AttributeProto*
    return reinterpret_cast<vaip_core::AttributeProto*>(
        mlir_named_attr.release());
  };

  the_mlir_instance_of_vaip_ort_api.attr_proto_new_floats =
      [](const std::string& name,
         const std::vector<float>& data) -> vaip_core::AttributeProto* {
    auto mlir_named_attr =
        mlir_impl::MLIRNamedAttribute::create_float_array(name, data);
    return reinterpret_cast<vaip_core::AttributeProto*>(
        mlir_named_attr.release());
  };

  the_mlir_instance_of_vaip_ort_api.attr_proto_new_strings =
      [](const std::string& name,
         const std::vector<std::string>& data) -> vaip_core::AttributeProto* {
    auto mlir_named_attr =
        mlir_impl::MLIRNamedAttribute::create_string_array(name, data);
    return reinterpret_cast<vaip_core::AttributeProto*>(
        mlir_named_attr.release());
  };

  the_mlir_instance_of_vaip_ort_api.attr_proto_new_int =
      [](const std::string& name, int64_t value) -> vaip_core::AttributeProto* {
    auto mlir_named_attr =
        mlir_impl::MLIRNamedAttribute::create_int(name, value);
    return reinterpret_cast<vaip_core::AttributeProto*>(
        mlir_named_attr.release());
  };

  the_mlir_instance_of_vaip_ort_api.attr_proto_new_float =
      [](const std::string& name, float value) -> vaip_core::AttributeProto* {
    auto mlir_named_attr =
        mlir_impl::MLIRNamedAttribute::create_float(name, value);
    return reinterpret_cast<vaip_core::AttributeProto*>(
        mlir_named_attr.release());
  };

  the_mlir_instance_of_vaip_ort_api.attr_proto_new_string =
      [](const std::string& name,
         const std::string& value) -> vaip_core::AttributeProto* {
    auto mlir_named_attr =
        mlir_impl::MLIRNamedAttribute::create_string(name, value);
    return reinterpret_cast<vaip_core::AttributeProto*>(
        mlir_named_attr.release());
  };

  the_mlir_instance_of_vaip_ort_api.attr_proto_new_tensor =
      [](const std::string& name,
         const vaip_core::TensorProto& value) -> vaip_core::AttributeProto* {
    // Convert vaip_core::TensorProto to MLIRNodeArg (which is MLIRTensor)
    auto* mlir_tensor = reinterpret_cast<const mlir_impl::MLIRNodeArg*>(&value);

    // Create a tensor attribute using MLIRNamedAttribute factory
    auto mlir_named_attr =
        mlir_impl::MLIRNamedAttribute::create_tensor(name, *mlir_tensor);
    return reinterpret_cast<vaip_core::AttributeProto*>(
        mlir_named_attr.release());
  };

  the_mlir_instance_of_vaip_ort_api.attr_proto_delete =
      [](vaip_core::AttributeProto* attr) -> void {
    if (attr) {
      auto* mlir_named_attr = reinterpret_cast<mlir::NamedAttribute*>(attr);
      delete mlir_named_attr;
    }
  };

  the_mlir_instance_of_vaip_ort_api.attr_proto_get_name =
      [](const vaip_core::AttributeProto& attr) -> const std::string& {
    auto* mlir_named_attr =
        reinterpret_cast<const mlir::NamedAttribute*>(&attr);
    static thread_local std::string attr_name_cache;
    attr_name_cache = mlir_named_attr->getName().str();
    return attr_name_cache;
  };

  the_mlir_instance_of_vaip_ort_api.attr_proto_get_type =
      [](const vaip_core::AttributeProto& attr) -> int {
    // Return the MLIR attribute type mapped to ONNX attribute type constants
    // This is simplified - in a real implementation you'd need proper type
    // mapping
    return reinterpret_cast<const mlir_impl::MLIRNamedAttribute*>(&attr)
        ->get_onnx_type();
  };

  the_mlir_instance_of_vaip_ort_api.attr_proto_set_name =
      [](vaip_core::AttributeProto* attr, const std::string& name) -> void {
    reinterpret_cast<mlir_impl::MLIRNamedAttribute*>(attr)->set_name(name);
  };

  the_mlir_instance_of_vaip_ort_api.attr_proto_clone =
      [](const vaip_core::AttributeProto& attr) -> vaip_core::AttributeProto* {
    auto* p = reinterpret_cast<const mlir::NamedAttribute*>(&attr);
    return reinterpret_cast<vaip_core::AttributeProto*>(
        new mlir::NamedAttribute(p->getName(), p->getValue()));
  };
  the_mlir_instance_of_vaip_ort_api.attr_proto_get_int =
      [](const vaip_core::AttributeProto& attr) -> int64_t {
    auto* p = reinterpret_cast<const mlir_impl::MLIRNamedAttribute*>(&attr);
    return p->get_int();
  };
  the_mlir_instance_of_vaip_ort_api.attr_proto_get_float =
      [](const vaip_core::AttributeProto& attr) -> float {
    auto* p = reinterpret_cast<const mlir_impl::MLIRNamedAttribute*>(&attr);
    return static_cast<float>(p->get_float());
  };
  the_mlir_instance_of_vaip_ort_api.attr_proto_get_string =
      [](const vaip_core::AttributeProto& attr) -> const std::string& {
    auto* p = reinterpret_cast<const mlir_impl::MLIRNamedAttribute*>(&attr);
    return p->get_string();
  };
  the_mlir_instance_of_vaip_ort_api.attr_proto_get_tensor =
      [](const vaip_core::AttributeProto& attr)
      -> const vaip_core::TensorProto& {
    auto* p = reinterpret_cast<const mlir_impl::MLIRNamedAttribute*>(&attr);
    return *reinterpret_cast<const vaip_core::TensorProto*>(p->get_tensor());
  };
  the_mlir_instance_of_vaip_ort_api.attr_proto_get_ints =
      [](const vaip_core::AttributeProto& attr) -> gsl::span<const int64_t> {
    // Static queue to store res with maximum length of 20
    static thread_local std::deque<std::vector<int64_t>> res_queue;
    static const size_t MAX_QUEUE_SIZE = 20;

    if (res_queue.size() >= MAX_QUEUE_SIZE) {
      res_queue.pop_front();
    }
    res_queue.push_back(
        reinterpret_cast<const mlir_impl::MLIRNamedAttribute*>(&attr)
            ->get_ints());
    return res_queue.back();
  };
  the_mlir_instance_of_vaip_ort_api.attr_proto_get_floats =
      [](const vaip_core::AttributeProto& attr) -> gsl::span<const float> {
    return reinterpret_cast<const mlir_impl::MLIRNamedAttribute*>(&attr)
        ->get_floats();
  };
  the_mlir_instance_of_vaip_ort_api.attr_proto_get_strings =
      [](const vaip_core::AttributeProto& attr) -> std::vector<std::string> {
    return reinterpret_cast<const mlir_impl::MLIRNamedAttribute*>(&attr)
        ->get_strings();
  };
  the_mlir_instance_of_vaip_ort_api.attr_proto_release_string =
      [](vaip_core::AttributeProto* attr) -> vaip_core::DllSafe<std::string> {
    return vaip_core::DllSafe<std::string>(
        reinterpret_cast<mlir_impl::MLIRNamedAttribute*>(attr)->get_string());
  };

  the_mlir_instance_of_vaip_ort_api.graph_remove_initialized_tensor =
      [](vaip_core::Graph& graph, const std::string& tensor_name) -> void {
    auto* mlir_graph = reinterpret_cast<mlir_impl::MLIRGraph*>(&graph);
    mlir_graph->remove_initialized_tensor(tensor_name);
  };

  the_mlir_instance_of_vaip_ort_api.graph_reverse_dfs_from_preemp =
      [](const vaip_core::Graph& graph,
         gsl::span<const vaip_core::Node* const> from,
         const std::function<bool(const vaip_core::Node*)>& enter,
         const std::function<bool(const vaip_core::Node*)>& leave,
         const std::function<bool(const vaip_core::Node*,
                                  const vaip_core::Node*)>& comp,
         const std::function<bool(const vaip_core::Node* from,
                                  const vaip_core::Node* to)>& stop) -> void {
    auto* mlir_graph = reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph);

    // Convert from vaip_core::Node* to mlir::Operation*
    std::vector<const mlir::Operation*> mlir_from;
    mlir_from.reserve(from.size());
    for (const auto* node_ptr : from) {
      mlir_from.push_back(reinterpret_cast<const mlir::Operation*>(node_ptr));
    }

    // Create wrapper functions that convert between mlir::Operation* and
    // vaip_core::Node*
    auto mlir_enter = enter ? [&enter](const mlir::Operation* op) -> bool {
      const auto* node_ptr = reinterpret_cast<const vaip_core::Node*>(op);
      return enter(node_ptr);
    }
    : std::function<bool(const mlir::Operation*)>();

    auto mlir_leave = leave ? [&leave](const mlir::Operation* op) -> bool {
      const auto* node_ptr = reinterpret_cast<const vaip_core::Node*>(op);
      return leave(node_ptr);
    }
    : std::function<bool(const mlir::Operation*)>();

    auto mlir_comp = comp ? [&comp](const mlir::Operation* from_op,
                                    const mlir::Operation* to_op) -> bool {
      const auto* from_ptr = reinterpret_cast<const vaip_core::Node*>(from_op);
      const auto* to_ptr = reinterpret_cast<const vaip_core::Node*>(to_op);
      return comp(from_ptr, to_ptr);
    }
    : std::function<bool(const mlir::Operation*, const mlir::Operation*)>();

    auto mlir_stop = stop ? [&stop](const mlir::Operation* from_op,
                                    const mlir::Operation* to_op) -> bool {
      const auto* from_ptr = reinterpret_cast<const vaip_core::Node*>(from_op);
      const auto* to_ptr = reinterpret_cast<const vaip_core::Node*>(to_op);
      return stop(from_ptr, to_ptr);
    }
    : std::function<bool(const mlir::Operation*, const mlir::Operation*)>();

    // Call the MLIRGraph method with converted parameters
    mlir_graph->reverse_dfs_from_preemp(
        gsl::span<const mlir::Operation* const>(mlir_from), mlir_enter,
        mlir_leave, mlir_comp, mlir_stop);
  };

  the_mlir_instance_of_vaip_ort_api.graph_infer_shapes_from_filepath =
      [](const std::string& model_path, const std::string& save_path) -> void {
    // TODO: Implement shape inference from file paths in MLIR backend
    LOG(WARNING)
        << "graph_infer_shapes_from_filepath not implemented in MLIR backend";
    (void)model_path;
    (void)save_path; // Suppress unused parameter warnings
  };

  the_mlir_instance_of_vaip_ort_api.graph_to_graph_proto =
      [](const vaip_core::Graph& graph) -> vaip_core::GraphProto* {
    auto* mlir_graph = const_cast<mlir_impl::MLIRGraph*>(
        reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph));
    return reinterpret_cast<vaip_core::GraphProto*>(mlir_graph);
  };

  the_mlir_instance_of_vaip_ort_api.graph_proto_delete =
      [](vaip_core::GraphProto* /* p */) -> void {
    // do nothing, ~MLIRGraph release itself
  };

  the_mlir_instance_of_vaip_ort_api.graph_infer_shapes =
      [](vaip_core::ModelProto& model_proto) -> void {
    // TODO: Implement shape inference in MLIR backend
    LOG(WARNING) << "graph_infer_shapes not implemented in MLIR backend";
    (void)model_proto; // Suppress unused parameter warning
  };
#if VAIP_ORT_API_MAJOR >= 18
  the_mlir_instance_of_vaip_ort_api.graph_save_string =
      [](const vaip_core::Graph& graph) -> vaip_core::DllSafe<std::string> {
    // TODO: Implement graph string serialization in MLIR backend
    LOG(WARNING) << "graph_save_string not implemented in MLIR backend";
    (void)graph; // Suppress unused parameter warning
    return vaip_core::DllSafe<std::string>(
        new std::string("graph_save_string not implemented in MLIR backend"));
  };
#endif // VAIP_ORT_API_MAJOR >= 18

  the_mlir_instance_of_vaip_ort_api.get_model_path =
      [](const vaip_core::Graph& graph) -> const std::filesystem::path& {
    auto mlir_graph = reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph);
    auto& model = mlir_graph->get_model();
    static thread_local std::filesystem::path model_path;
    model_path = model.has_metadata_prop("model_path")
                     ? model.get_metadata_prop("model_path")
                     : "";
    return model_path;
  };

  the_mlir_instance_of_vaip_ort_api.session_option_configuration =
      [](void* mmap, void* session_option,
         void (*push)(void* mmap, const char* name,
                      const char* value)) -> void {
    // TODO: Implement session option configuration in MLIR backend
    LOG(WARNING)
        << "session_option_configuration not implemented in MLIR backend";
    (void)mmap;
    (void)session_option;
    (void)push; // Suppress unused parameter warnings
  };

  the_mlir_instance_of_vaip_ort_api.model_to_proto =
      [](vaip_core::Model& model) -> vaip_core::ModelProto* {
    auto* mlir_model = reinterpret_cast<mlir_impl::MLIRModel*>(&model);
    return reinterpret_cast<vaip_core::ModelProto*>(mlir_model);
  };

  the_mlir_instance_of_vaip_ort_api.model_proto_serialize_as_string =
      [](vaip_core::ModelProto& model_proto)
      -> vaip_core::DllSafe<std::string> {
    auto* mlir_model = reinterpret_cast<mlir_impl::MLIRModel*>(&model_proto);
    return vaip_core::DllSafe<std::string>(mlir_model->serialize_as_string());
  };

  the_mlir_instance_of_vaip_ort_api.model_proto_delete =
      [](vaip_core::ModelProto* /* p*/) -> void {
    // do nothing, ~MLIRModel release itself
  };

  the_mlir_instance_of_vaip_ort_api.is_profiling_enabled =
      [](void* session_options) -> bool {
    // TODO: Implement profiling status check in MLIR backend
    LOG(WARNING) << "is_profiling_enabled not implemented in MLIR backend";
    (void)session_options; // Suppress unused parameter warning
    return false;
  };

  // Mark as initialized
  initialized = true;
}

// Public API to get the MLIR implementation
extern "C" VaipOrtApiExt* get_vaip_ort_api_mlir() {
  initialize_mlir_api();
  return &the_mlir_instance_of_vaip_ort_api;
}

} // namespace morphizen

// Function for plugin registration with proper naming
namespace {
const vaip_core::OrtApiForVaip* morphizen_mlir_imp_get_vaip_ort_api() {
  return reinterpret_cast<const vaip_core::OrtApiForVaip*>(
      morphizen::get_vaip_ort_api_mlir());
}

static ::vaip_core::StaticPluginRegister
    __register(morphizen::kMLIRBackend, "vaip_ort_api_imp",
               (void*)&morphizen_mlir_imp_get_vaip_ort_api);
} // namespace

// Factory function for creating MLIR-based models
namespace onnxruntime {
// Placeholder for integration with ONNXRuntime if needed
}
