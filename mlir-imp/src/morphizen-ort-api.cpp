/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// NOTE: MLIR implementation
// the_mlir_instance_of_morphizen_ort_api.graph_get_inputs_unsafe = This
// provides an MLIR-based implementation of the same API interface used by the
// ONNX implementation, allowing MLIR dialects to be used as an alternative IR
// representation for MorphiZen passes.
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
#include "morphizen-utils/morphizen_plugin.hpp"

// local
#include "./mlir-constants.hpp"
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <glog/logging.h>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

// String constants
static constexpr const char* AMD_MORPHIZEN_MLIR_DOMAIN =
    "com.amd.morphizen.mlir";

namespace morphizen {

// Static assertion to ensure safe reinterpret_cast between types
static_assert(sizeof(morphizen::AttributeProto*) ==
                  sizeof(mlir::NamedAttribute*),
              "morphizen::AttributeProto* and MLIRNamedAttribute* must have "
              "the same size for safe reinterpret_cast");

// The variant AttributeProto encoding embeds a raw MLIRGraph* in an
// IntegerAttr as int64 (see MLIRNamedAttribute::create_subgraph_ref).
// Replicate the invariant at the api boundary so the surface itself
// documents the pointer-width assumption.
static_assert(sizeof(void*) <= sizeof(int64_t),
              "raw MLIRGraph* must fit in int64 for the variant "
              "AttributeProto encoding");

// ONNX ↔ MLIR shape sentinel translation. ONNX uses -1 (and occasionally
// other negatives) for unknown / dynamic dims; MLIR's RankedTensorType
// verifier only accepts mlir::ShapedType::kDynamic (= INT64_MIN). These
// helpers live here at the api boundary; MLIR-internal code never sees raw
// ONNX negatives and ONNX-external code never sees kDynamic.
namespace mlir_impl {
namespace {
llvm::SmallVector<int64_t> to_mlir_dims(llvm::ArrayRef<int64_t> onnx_dims) {
  llvm::SmallVector<int64_t> dims;
  dims.reserve(onnx_dims.size());
  for (int64_t d : onnx_dims) {
    dims.push_back(d < 0 ? mlir::ShapedType::kDynamic : d);
  }
  return dims;
}
std::vector<int64_t> to_onnx_dims(llvm::ArrayRef<int64_t> mlir_dims) {
  std::vector<int64_t> dims;
  dims.reserve(mlir_dims.size());
  for (int64_t d : mlir_dims) {
    dims.push_back(d == mlir::ShapedType::kDynamic ? -1 : d);
  }
  return dims;
}
} // namespace
} // namespace mlir_impl

// Helper function to create MLIRTensor and return as TensorProto*
// Similar to tensor_proto_new_with_raw_data in ONNX implementation
morphizen::TensorProto* tensor_proto_new_with_raw_data_mlir(
    const std::string& name, const std::vector<int64_t>& shape,
    const void* data, size_t size, int data_type) {
  // ONNX→MLIR boundary: translate negative (dynamic) dims to kDynamic once.
  auto mlir_shape = mlir_impl::to_mlir_dims(shape);

  // Create MLIRTensor and return as TensorProto*
  auto* mlir_node_arg =
      new mlir_impl::MLIRNodeArg(name, mlir_shape, data_type, data, size);

  return reinterpret_cast<morphizen::TensorProto*>(mlir_node_arg);
}

// Initialize the global API instance with MLIR implementations
MorphizenOrtApiExt the_mlir_instance_of_morphizen_ort_api;

// Static initialization function to populate the API structure with MLIR
// implementations
static void initialize_mlir_api() {
  static bool initialized = false;
  if (initialized)
    return;

  // Version and magic fields for compatibility checking
  the_mlir_instance_of_morphizen_ort_api.magic =
      0x50494156; // Binary format compatibility identifier
  the_mlir_instance_of_morphizen_ort_api.major = MORPHIZEN_ORT_API_MAJOR;
  the_mlir_instance_of_morphizen_ort_api.minor = MORPHIZEN_ORT_API_MINOR;
  the_mlir_instance_of_morphizen_ort_api.patch = MORPHIZEN_ORT_API_PATCH;

  // Core pointers
  the_mlir_instance_of_morphizen_ort_api.host_ =
      (onnxruntime::ProviderHost*)(void*)1; // Not used in MLIR implementation
#ifdef ORT_API_VERSION
  the_mlir_instance_of_morphizen_ort_api.ort_api_ = &Ort::GetApi();
#else
  the_mlir_instance_of_morphizen_ort_api.ort_api_ = nullptr;
#endif

  // Model API functions [0-6]
  the_mlir_instance_of_morphizen_ort_api.model_load =
      [](const std::string& file) -> morphizen::Model* {
    auto model = mlir_impl::MLIRModel::load(file);
    return reinterpret_cast<morphizen::Model*>(model.release());
  };

  the_mlir_instance_of_morphizen_ort_api.model_delete =
      [](morphizen::Model* model) -> void {
    if (model) {
      auto* mlir_model = reinterpret_cast<mlir_impl::MLIRModel*>(model);
      delete mlir_model;
    }
  };

  the_mlir_instance_of_morphizen_ort_api.model_clone =
      [](const morphizen::Model& model,
         int64_t external_data_threshold) -> morphizen::Model* {
    auto* mlir_model = reinterpret_cast<const mlir_impl::MLIRModel*>(&model);
    auto cloned_model = mlir_model->clone(external_data_threshold);
    return reinterpret_cast<morphizen::Model*>(cloned_model.release());
  };

  the_mlir_instance_of_morphizen_ort_api.model_main_graph =
      [](morphizen::Model& model) -> morphizen::Graph& {
    auto* mlir_model = reinterpret_cast<mlir_impl::MLIRModel*>(&model);
    return reinterpret_cast<morphizen::Graph&>(mlir_model->main_graph());
  };

  the_mlir_instance_of_morphizen_ort_api.model_set_meta_data =
      [](morphizen::Model& model, const std::string& key,
         const std::string& value) -> void {
    auto* mlir_model = reinterpret_cast<mlir_impl::MLIRModel*>(&model);
    mlir_model->set_metadata_prop(key, value);
  };

  the_mlir_instance_of_morphizen_ort_api.model_get_meta_data =
      [](const morphizen::Model& model,
         const std::string& key) -> morphizen::DllSafe<std::string> {
    auto* mlir_model = reinterpret_cast<const mlir_impl::MLIRModel*>(&model);
    std::string value = mlir_model->get_metadata_prop(key);
    return morphizen::DllSafe<std::string>(new std::string(std::move(value)));
  };

  the_mlir_instance_of_morphizen_ort_api.model_has_meta_data =
      [](const morphizen::Model& model, const std::string& key) -> int {
    auto* mlir_model = reinterpret_cast<const mlir_impl::MLIRModel*>(&model);
    return mlir_model->has_metadata_prop(key) ? 1 : 0;
  };

  // Graph API functions [7-23] - Basic implementations
  the_mlir_instance_of_morphizen_ort_api.graph_get_name =
      [](const morphizen::Graph& graph) -> const std::string& {
    auto* mlir_graph = reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph);
    return mlir_graph->get_name();
  };

  the_mlir_instance_of_morphizen_ort_api.graph_set_name =
      [](morphizen::Graph& graph, const char* name) -> void {
    auto* mlir_graph = reinterpret_cast<mlir_impl::MLIRGraph*>(&graph);
    mlir_graph->set_name(name);
  };

  the_mlir_instance_of_morphizen_ort_api.graph_get_model =
      [](const morphizen::Graph& graph) -> const morphizen::Model& {
    auto* mlir_graph = reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph);
    return reinterpret_cast<const morphizen::Model&>(mlir_graph->get_model());
  };

  the_mlir_instance_of_morphizen_ort_api.graph_nodes_unsafe =
      [](const morphizen::Graph& graph)
      -> morphizen::DllSafe<std::vector<const morphizen::Node*>> {
    auto* mlir_graph = reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph);
    auto nodes = mlir_graph->nodes_unsafe();
    auto result = new std::vector<const morphizen::Node*>();
    result->reserve(nodes.size());
    for (const auto* op : nodes) {
      result->push_back(reinterpret_cast<const morphizen::Node*>(op));
    }
    return morphizen::DllSafe<std::vector<const morphizen::Node*>>(result);
  };

  the_mlir_instance_of_morphizen_ort_api.graph_get_inputs_unsafe =
      [](const morphizen::Graph& graph)
      -> morphizen::DllSafe<std::vector<const morphizen::NodeArg*>> {
    auto* mlir_graph = reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph);
    auto inputs = mlir_graph->get_inputs();
    CHECK(!inputs.empty()) << "MLIRGraph inputs are empty ";
    auto result = new std::vector<const morphizen::NodeArg*>();
    result->reserve(inputs.size());
    for (const auto& value : inputs) {
      result->push_back(reinterpret_cast<const morphizen::NodeArg*>(
          value.to_morphizen_core_node_arg_ptr()));
    }
    return morphizen::DllSafe<std::vector<const morphizen::NodeArg*>>(result);
  };

  the_mlir_instance_of_morphizen_ort_api.graph_get_outputs_unsafe =
      [](const morphizen::Graph& graph)
      -> morphizen::DllSafe<std::vector<const morphizen::NodeArg*>> {
    auto* mlir_graph = reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph);
    auto result = new std::vector<const morphizen::NodeArg*>();

    auto outputs = mlir_graph->get_outputs();
    result->reserve(outputs.size());
    for (const auto& value : outputs) {
      result->push_back(reinterpret_cast<const morphizen::NodeArg*>(
          value.to_morphizen_core_node_arg_ptr()));
    }
    return morphizen::DllSafe<std::vector<const morphizen::NodeArg*>>(result);
  };

  the_mlir_instance_of_morphizen_ort_api.graph_set_outputs =
      [](morphizen::Graph& graph,
         gsl::span<const morphizen::NodeArg* const> outputs) -> void {
    auto* mlir_graph = reinterpret_cast<mlir_impl::MLIRGraph*>(&graph);

    // Convert from NodeArg* span to MLIRNodeArgIndex span
    llvm::SmallVector<mlir_impl::MLIRNodeArgIndex> mlir_outputs;
    mlir_outputs.reserve(outputs.size());
    for (const auto* nodeArg : outputs) {
      // Convert NodeArg* to MLIRNodeArgIndex using the raw pointer constructor
      auto node_arg_index =
          mlir_impl::MLIRNodeArgIndex::from_morphizen_core_node_arg_ptr(
              nodeArg);
      mlir_outputs.push_back(node_arg_index);
    }
    mlir_graph->set_outputs(mlir_outputs);
  };

  the_mlir_instance_of_morphizen_ort_api.graph_set_inputs =
      [](morphizen::Graph& graph,
         gsl::span<const morphizen::NodeArg* const> inputs) -> void {
    auto* mlir_graph = reinterpret_cast<mlir_impl::MLIRGraph*>(&graph);

    // Convert from NodeArg* span to mlir::Value span
    llvm::SmallVector<mlir_impl::MLIRNodeArgIndex> mlir_inputs;
    mlir_inputs.reserve(inputs.size());
    for (const auto* nodeArg : inputs) {
      // Convert NodeArg* back to mlir::Value
      auto node_arg_index =
          mlir_impl::MLIRNodeArgIndex::from_morphizen_core_node_arg_ptr(
              nodeArg);
      mlir_inputs.push_back(node_arg_index);
    }
    mlir_graph->set_inputs(mlir_inputs);
  };

  the_mlir_instance_of_morphizen_ort_api.create_empty_model =
      [](const std::filesystem::path& path,
         const std::vector<std::pair<std::string, int64_t>>& opset)
      -> morphizen::Model* {
    auto model = mlir_impl::MLIRModel::create_empty(path, opset);
    return reinterpret_cast<morphizen::Model*>(model.release());
  };

  // Tensor proto API functions - implemented using MLIRTensor with helper
  // function
  the_mlir_instance_of_morphizen_ort_api.tensor_proto_new_floats =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<float>& data) -> morphizen::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(
        name, shape, data.data(), data.size() * sizeof(float), 1); // FLOAT = 1
  };

  the_mlir_instance_of_morphizen_ort_api.tensor_proto_new_i64 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<int64_t>& data) -> morphizen::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(name, shape, data.data(),
                                               data.size() * sizeof(int64_t),
                                               7); // INT64 = 7
  };

  the_mlir_instance_of_morphizen_ort_api.tensor_proto_new_i32 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<int32_t>& data) -> morphizen::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(name, shape, data.data(),
                                               data.size() * sizeof(int32_t),
                                               6); // INT32 = 6
  };

  the_mlir_instance_of_morphizen_ort_api.tensor_proto_new_i8 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<int8_t>& data) -> morphizen::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(
        name, shape, data.data(), data.size() * sizeof(int8_t), 3); // INT8 = 3
  };

  // Additional tensor functions
  the_mlir_instance_of_morphizen_ort_api.tensor_proto_new_doubles =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<double>& data) -> morphizen::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(name, shape, data.data(),
                                               data.size() * sizeof(double),
                                               11); // DOUBLE = 11
  };

  the_mlir_instance_of_morphizen_ort_api.tensor_proto_new_u8 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<uint8_t>& data) -> morphizen::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(name, shape, data.data(),
                                               data.size() * sizeof(uint8_t),
                                               2); // UINT8 = 2
  };

  the_mlir_instance_of_morphizen_ort_api.tensor_proto_new_u32 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<uint32_t>& data) -> morphizen::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(name, shape, data.data(),
                                               data.size() * sizeof(uint32_t),
                                               12); // UINT32 = 12
  };

  the_mlir_instance_of_morphizen_ort_api.tensor_proto_new_u64 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<uint64_t>& data) -> morphizen::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(name, shape, data.data(),
                                               data.size() * sizeof(uint64_t),
                                               13); // UINT64 = 13
  };

  the_mlir_instance_of_morphizen_ort_api.tensor_proto_new_i16 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<int16_t>& data) -> morphizen::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(name, shape, data.data(),
                                               data.size() * sizeof(int16_t),
                                               5); // INT16 = 5
  };

  the_mlir_instance_of_morphizen_ort_api.tensor_proto_new_u16 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<uint16_t>& data) -> morphizen::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(name, shape, data.data(),
                                               data.size() * sizeof(uint16_t),
                                               4); // UINT16 = 4
  };

  the_mlir_instance_of_morphizen_ort_api.tensor_proto_new_fp16 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<int16_t>& data) -> morphizen::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(name, shape, data.data(),
                                               data.size() * sizeof(int16_t),
                                               10); // FLOAT16 = 10
  };

  the_mlir_instance_of_morphizen_ort_api.tensor_proto_new_bf16 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<int16_t>& data) -> morphizen::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(name, shape, data.data(),
                                               data.size() * sizeof(int16_t),
                                               16); // BFLOAT16 = 16
  };

// Additional tensor proto functions for 4-bit types
#if MORPHIZEN_ORT_API_MAJOR >= 19
  the_mlir_instance_of_morphizen_ort_api.tensor_proto_new_bool =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<uint8_t>& data) -> morphizen::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(name, shape, data.data(),
                                               data.size() * sizeof(uint8_t),
                                               9); // BOOL = 9
  };
#endif // MORPHIZEN_ORT_API_MAJOR >= 19

  the_mlir_instance_of_morphizen_ort_api.tensor_proto_new_i4 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<int8_t>& data) -> morphizen::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(name, shape, data.data(),
                                               data.size() * sizeof(int8_t),
                                               22); // INT4 = 22
  };

  the_mlir_instance_of_morphizen_ort_api.tensor_proto_new_u4 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<uint8_t>& data) -> morphizen::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(name, shape, data.data(),
                                               data.size() * sizeof(uint8_t),
                                               21); // UINT4 = 21
  };

  // Tensor creation with external data
  the_mlir_instance_of_morphizen_ort_api.tensor_proto_new_with_external_data =
      [](const std::string& name, const std::vector<int64_t>& shape,
         int element_type, const std::string& external_data_file, size_t size,
         size_t offset) -> morphizen::TensorProto* {
    // ONNX→MLIR boundary: translate negative (dynamic) dims to kDynamic once.
    auto mlir_shape = mlir_impl::to_mlir_dims(shape);

    auto* mlir_node_arg = new mlir_impl::MLIRNodeArg(
        name, mlir_shape, element_type, external_data_file, offset, size);

    return reinterpret_cast<morphizen::TensorProto*>(mlir_node_arg);
  };

  // Tensor creation with raw data
  the_mlir_instance_of_morphizen_ort_api.tensor_proto_new_raw_data =
      [](const std::string& name, const std::vector<int64_t>& shape,
         int element_type, const void* data,
         size_t size) -> morphizen::TensorProto* {
    return tensor_proto_new_with_raw_data_mlir(name, shape, data, size,
                                               element_type);
  };

  // Tensor info functions - using direct casting
  the_mlir_instance_of_morphizen_ort_api.tensor_proto_get_shape_unsafe =
      [](const morphizen::TensorProto& tensor_proto)
      -> morphizen::DllSafe<std::vector<int64_t>> {
    // Direct cast TensorProto to MLIRTensor
    auto* mlir_tensor =
        reinterpret_cast<const morphizen::mlir_impl::MLIRTensor*>(
            &tensor_proto);
    // TensorProto always backs a concrete initializer with a known rank;
    // unranked storage is only possible at the ORT NodeArg boundary.
    auto shape = mlir_tensor->getShape();
    CHECK(shape.has_value())
        << "tensor_proto_get_shape_unsafe: tensor has no rank: "
        << mlir_tensor->getName();
    // MLIR→ONNX boundary: translate kDynamic back to -1.
    auto result =
        std::make_unique<std::vector<int64_t>>(mlir_impl::to_onnx_dims(*shape));
    return morphizen::DllSafe<std::vector<int64_t>>(result.release());
  };

  the_mlir_instance_of_morphizen_ort_api.tensor_proto_data_type =
      [](const morphizen::TensorProto& tensor_proto) -> int {
    auto* mlir_tensor =
        reinterpret_cast<const morphizen::mlir_impl::MLIRTensor*>(
            &tensor_proto);
    return mlir_tensor->getElementType();
  };

  the_mlir_instance_of_morphizen_ort_api.tensor_proto_get_name =
      [](const morphizen::TensorProto& tensor_proto) -> const std::string& {
    auto* mlir_tensor =
        reinterpret_cast<const morphizen::mlir_impl::MLIRTensor*>(
            &tensor_proto);
    return mlir_tensor->getName();
  };

  the_mlir_instance_of_morphizen_ort_api.tensor_proto_raw_data_size =
      [](const morphizen::TensorProto& tensor_proto) -> size_t {
    auto* mlir_tensor =
        reinterpret_cast<const morphizen::mlir_impl::MLIRTensor*>(
            &tensor_proto);
    return mlir_tensor->getDataSize();
  };

  the_mlir_instance_of_morphizen_ort_api.tensor_proto_as_raw =
      [](const morphizen::Graph& /*graph*/,
         const morphizen::TensorProto& tensor_proto) -> gsl::span<const char> {
    auto* mlir_tensor =
        reinterpret_cast<const morphizen::mlir_impl::MLIRTensor*>(
            &tensor_proto);
    const void* data = mlir_tensor->getData();
    size_t size = mlir_tensor->getDataSize();
    return gsl::span<const char>(reinterpret_cast<const char*>(data), size);
  };

  the_mlir_instance_of_morphizen_ort_api.tensor_proto_delete =
      [](morphizen::TensorProto* tp) -> void {
    if (tp) {
      // Direct cast and delete as MLIRTensor
      auto* mlir_tensor =
          reinterpret_cast<morphizen::mlir_impl::MLIRTensor*>(tp);
      delete mlir_tensor;
    }
  };

  // Library info functions
  the_mlir_instance_of_morphizen_ort_api.get_lib_id =
      []() -> morphizen::DllSafe<std::string> {
    return morphizen::DllSafe<std::string>(new std::string("v1.0.0"));
  };

  the_mlir_instance_of_morphizen_ort_api.get_lib_name =
      []() -> morphizen::DllSafe<std::string> {
    return morphizen::DllSafe<std::string>(
        new std::string("morphizen-mlir-imp"));
  };

  // Initialize remaining function pointers to nullptr for now
  // These would be implemented as needed for MLIR-specific functionality
  the_mlir_instance_of_morphizen_ort_api.graph_get_node =
      [](const morphizen::Graph& graph,
         size_t index) -> const morphizen::Node* {
    auto* mlir_graph = reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph);
    auto* op = mlir_graph->get_node(index);
    return reinterpret_cast<const morphizen::Node*>(op);
  };

  the_mlir_instance_of_morphizen_ort_api.graph_producer_node =
      [](const morphizen::Graph& graph,
         const std::string& node_arg_name) -> const morphizen::Node* {
    auto* mlir_graph = reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph);
    auto* op = mlir_graph->producer_node(node_arg_name);
    return reinterpret_cast<const morphizen::Node*>(op);
  };
  the_mlir_instance_of_morphizen_ort_api.graph_get_node_arg =
      [](const morphizen::Graph& graph,
         const std::string& name) -> const morphizen::NodeArg* {
    auto* mlir_graph = reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph);
    auto node_arg_index = mlir_graph->get_node_arg_index(name);
    if (!node_arg_index.is_valid()) {
      return nullptr;
    }
    return reinterpret_cast<const morphizen::NodeArg*>(
        node_arg_index.to_morphizen_core_node_arg_ptr());
  };

  the_mlir_instance_of_morphizen_ort_api.graph_get_all_initialized_tensors =
      [](const morphizen::Graph& graph)
      -> const morphizen::InitializedTensorSet& {
    auto* mlir_graph = reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph);
    return reinterpret_cast<const morphizen::InitializedTensorSet&>(
        mlir_graph->get_all_initialized_tensors());
  };
  the_mlir_instance_of_morphizen_ort_api.graph_remove_node =
      [](morphizen::Graph& graph,
         const morphizen::NodeInput& node_input) -> void {
    auto* mlir_graph = reinterpret_cast<mlir_impl::MLIRGraph*>(&graph);

    // Convert morphizen::Node* to mlir::Operation*
    // The node_input.node field contains the Node* that should be removed
    if (!node_input.node) {
      return;
    }

    // node_input.node is actually an mlir::Operation* cast to morphizen::Node*
    auto* op = reinterpret_cast<mlir::Operation*>(
        const_cast<morphizen::Node*>(node_input.node));

    mlir_graph->remove_node(op);
  };
  the_mlir_instance_of_morphizen_ort_api.graph_add_node =
      [](morphizen::Graph& graph, const std::string& name,
         const std::string& op_type, const std::string& description,
         const std::vector<const morphizen::NodeArg*>& input_args,
         const std::vector<const morphizen::NodeArg*>& output_args,
         const morphizen::NodeAttributes& attributes,
         const std::string& domain) -> morphizen::Node& {
    auto* mlir_graph = reinterpret_cast<mlir_impl::MLIRGraph*>(&graph);

    // Convert input NodeArg* to MLIRNodeArgIndex
    std::vector<mlir_impl::MLIRNodeArgIndex> mlir_input_args;
    mlir_input_args.reserve(input_args.size());
    for (const auto* nodeArg : input_args) {
      // Convert NodeArg* to MLIRNodeArgIndex using the raw pointer constructor
      auto node_arg_index =
          mlir_impl::MLIRNodeArgIndex::from_morphizen_core_node_arg_ptr(
              nodeArg);
      mlir_input_args.push_back(node_arg_index);
    }

    // Convert output NodeArg* to MLIRNodeArgIndex
    std::vector<mlir_impl::MLIRNodeArgIndex> mlir_output_args;
    mlir_output_args.reserve(output_args.size());
    for (const auto* nodeArg : output_args) {
      // Convert NodeArg* to MLIRNodeArgIndex using the raw pointer constructor
      auto node_arg_index =
          mlir_impl::MLIRNodeArgIndex::from_morphizen_core_node_arg_ptr(
              nodeArg);
      mlir_output_args.push_back(node_arg_index);
    }

    // Convert morphizen::NodeAttributes to MLIRNodeAttributes
    auto mlir_node_attrs =
        mlir_impl::MLIRNodeAttributes(const_cast<mlir::Operation*>(
            reinterpret_cast<const mlir::Operation*>(&attributes)));

    // Call the MLIRGraph add_node method
    const mlir::Operation* op =
        mlir_graph->add_node(name, op_type, description, mlir_input_args,
                             mlir_output_args, mlir_node_attrs, domain);

    // Convert to morphizen::Node* and return as non-const reference
    const morphizen::Node* node = reinterpret_cast<const morphizen::Node*>(op);
    return *const_cast<morphizen::Node*>(node);
  };
  the_mlir_instance_of_morphizen_ort_api.graph_save =
      [](const morphizen::Graph& graph, const std::string& filename,
         const std::string& dat_filename,
         size_t external_data_threshold) -> void {
    auto* mlir_graph = reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph);
    mlir_graph->save(filename, dat_filename, external_data_threshold);
  };
  the_mlir_instance_of_morphizen_ort_api.graph_fuse =
      [](morphizen::Graph& graph, const std::string& name,
         const std::string& op_type, const std::vector<size_t>& nodes,
         const std::vector<std::string>& inputs,
         const std::vector<std::string>& outputs,
         const std::vector<std::string>& constant_initializers)
      -> morphizen::Node& {
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
    return *reinterpret_cast<morphizen::Node*>(fused_op);
  };

  the_mlir_instance_of_morphizen_ort_api.graph_resolve =
      [](morphizen::Graph& graph, bool force) -> int {
    auto* mlir_graph = reinterpret_cast<mlir_impl::MLIRGraph*>(&graph);
    return mlir_graph->resolve(force);
  };

  the_mlir_instance_of_morphizen_ort_api.graph_get_consumer_nodes_unsafe =
      [](const morphizen::Graph& graph, const std::string& node_arg_name)
      -> morphizen::DllSafe<std::vector<const morphizen::Node*>> {
    auto* mlir_graph = reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph);
    auto mlir_consumer_ops = mlir_graph->get_consumer_nodes(node_arg_name);

    // Convert vector<mlir::Operation*> to vector<morphizen::Node*>
    //
    // IMPORTANT: We cannot reinterpret_cast the entire vector because:
    // 1. vector<T*> and vector<U*> have different memory layouts (even if T*
    // and U* are compatible)
    // 2. Doing so violates strict-aliasing rules and causes GCC
    // -Werror=strict-aliasing
    // 3. It's undefined behavior even though pointers have same size
    //
    // Performance: Element-wise conversion is efficient:
    // - Single allocation with reserve() - no reallocations
    // - Typical consumer count: 1-20 nodes → negligible cost (< 1μs)
    // - Cost is O(n) pointer copies, dominated by graph traversal itself
    // - Correctness and portability are more important than micro-optimization
    std::vector<const morphizen::Node*> consumer_ops;
    consumer_ops.reserve(mlir_consumer_ops.size());
    for (auto* op : mlir_consumer_ops) {
      consumer_ops.push_back(reinterpret_cast<const morphizen::Node*>(op));
    }
    return morphizen::DllSafe<std::vector<const morphizen::Node*>>(
        std::move(consumer_ops));
  };

  the_mlir_instance_of_morphizen_ort_api.graph_reverse_dfs_from =
      [](const morphizen::Graph& graph,
         gsl::span<const morphizen::Node* const> from,
         const std::function<void(const morphizen::Node*)>& enter,
         const std::function<void(const morphizen::Node*)>& leave,
         const std::function<bool(const morphizen::Node* /*from*/,
                                  const morphizen::Node* /*to*/)>& stop)
      -> void {
    auto* mlir_graph = reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph);

    // Convert from morphizen::Node* to mlir::Operation*
    std::vector<const mlir::Operation*> mlir_from;
    mlir_from.reserve(from.size());
    for (const auto* node_ptr : from) {
      mlir_from.push_back(reinterpret_cast<const mlir::Operation*>(node_ptr));
    }

    // Create wrapper functions that convert void callbacks to bool callbacks
    // The non-preemptive version always returns false (don't stop processing)
    auto mlir_enter = enter ? [&enter](const mlir::Operation* op) -> bool {
      const auto* node_ptr = reinterpret_cast<const morphizen::Node*>(op);
      enter(node_ptr);
      return false; // Never preempt for non-preemptive version
    }
    : std::function<bool(const mlir::Operation*)>();

    auto mlir_leave = leave ? [&leave](const mlir::Operation* op) -> bool {
      const auto* node_ptr = reinterpret_cast<const morphizen::Node*>(op);
      leave(node_ptr);
      return false; // Never preempt for non-preemptive version
    }
    : std::function<bool(const mlir::Operation*)>();

    auto mlir_stop = stop ? [&stop](const mlir::Operation* from_op,
                                    const mlir::Operation* to_op) -> bool {
      const auto* from_ptr = reinterpret_cast<const morphizen::Node*>(from_op);
      const auto* to_ptr = reinterpret_cast<const morphizen::Node*>(to_op);
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
  the_mlir_instance_of_morphizen_ort_api.graph_add_initialized_tensor =
      [](morphizen::Graph& graph,
         const morphizen::TensorProto& tensor) -> void {
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
  the_mlir_instance_of_morphizen_ort_api.node_get_name =
      [](const morphizen::Node& node) -> const std::string& {
    auto mlir_node = mlir_impl::MLIRNode(reinterpret_cast<mlir::Operation*>(
        const_cast<morphizen::Node*>(&node)));
    static thread_local std::string name_cache;
    name_cache = mlir_node.getName();
    return name_cache;
  };

  the_mlir_instance_of_morphizen_ort_api.node_description =
      [](const morphizen::Node& node) -> const std::string& {
    auto mlir_node = mlir_impl::MLIRNode(reinterpret_cast<mlir::Operation*>(
        const_cast<morphizen::Node*>(&node)));
    static thread_local std::string description_cache;
    description_cache = mlir_node.getDescription();
    return description_cache;
  };

  the_mlir_instance_of_morphizen_ort_api.node_get_index =
      [](const morphizen::Node& node) -> size_t {
    return reinterpret_cast<size_t>(&node);
  };

  the_mlir_instance_of_morphizen_ort_api.node_op_type =
      [](const morphizen::Node& node) -> const std::string& {
    auto op = mlir_impl::MLIRNode(reinterpret_cast<mlir::Operation*>(
        const_cast<morphizen::Node*>(&node)));
    static thread_local std::string op_type_str;
    op_type_str = op.getOpType();
    return op_type_str;
  };

  the_mlir_instance_of_morphizen_ort_api.node_op_domain =
      [](const morphizen::Node& node) -> const std::string& {
    auto op = mlir_impl::MLIRNode(reinterpret_cast<mlir::Operation*>(
        const_cast<morphizen::Node*>(&node)));
    static thread_local std::string domain_str;
    domain_str = op.getDomain();
    return domain_str;
  };

  the_mlir_instance_of_morphizen_ort_api.node_get_inputs_unsafe =
      [](const morphizen::Node& node)
      -> morphizen::DllSafe<std::vector<morphizen::NodeInput>> {
    auto result = new std::vector<morphizen::NodeInput>();

    // Create MLIRNode wrapper for convenient access to node data
    auto mlir_node = mlir_impl::MLIRNode(reinterpret_cast<mlir::Operation*>(
        const_cast<morphizen::Node*>(&node)));

    // Get input MLIRNodeArgIndex objects using MLIRNode interface
    auto inputNodeArgs = mlir_node.getInputNodeArgs();
    result->reserve(inputNodeArgs.size());

    for (const auto& MLIRNodeArgIndex : inputNodeArgs) {
      // Convert MLIRNodeArgIndex to NodeInput
      result->push_back(
          {reinterpret_cast<const morphizen::Node*>(
               MLIRNodeArgIndex.get_producer_node()),
           reinterpret_cast<const morphizen::NodeArg*>(
               MLIRNodeArgIndex.to_morphizen_core_node_arg_ptr())});
    }

    return morphizen::DllSafe<std::vector<morphizen::NodeInput>>(result);
  };

  the_mlir_instance_of_morphizen_ort_api.node_get_implicit_inputs_unsafe =
      [](const morphizen::Node& node)
      -> morphizen::DllSafe<std::vector<morphizen::NodeInput>> {
    auto result = new std::vector<morphizen::NodeInput>();

    auto mlir_node = mlir_impl::MLIRNode(reinterpret_cast<mlir::Operation*>(
        const_cast<morphizen::Node*>(&node)));
    auto implicit_args = mlir_node.getImplicitInputNodeArgs();
    result->reserve(implicit_args.size());

    for (const auto& arg_index : implicit_args) {
      result->push_back({reinterpret_cast<const morphizen::Node*>(
                             arg_index.get_producer_node()),
                         reinterpret_cast<const morphizen::NodeArg*>(
                             arg_index.to_morphizen_core_node_arg_ptr())});
    }

    return morphizen::DllSafe<std::vector<morphizen::NodeInput>>(result);
  };

  the_mlir_instance_of_morphizen_ort_api.node_get_output_node_args_unsafe =
      [](const morphizen::Node& node)
      -> morphizen::DllSafe<std::vector<const morphizen::NodeArg*>> {
    auto result = new std::vector<const morphizen::NodeArg*>();

    // Create MLIRNode wrapper for convenient access to node data
    auto mlir_node = mlir_impl::MLIRNode(reinterpret_cast<mlir::Operation*>(
        const_cast<morphizen::Node*>(&node)));

    // Get output MLIRNodeArgIndex objects using MLIRNode interface
    auto outputNodeArgs = mlir_node.getOutputNodeArgs();
    result->reserve(outputNodeArgs.size());

    for (const auto& MLIRNodeArgIndex : outputNodeArgs) {
      // Convert MLIRNodeArgIndex to NodeArg*
      auto nodeArg = reinterpret_cast<const morphizen::NodeArg*>(
          MLIRNodeArgIndex.to_morphizen_core_node_arg_ptr());
      result->push_back(nodeArg);
    }

    return morphizen::DllSafe<std::vector<const morphizen::NodeArg*>>(result);
  };

  the_mlir_instance_of_morphizen_ort_api.node_get_attributes =
      [](morphizen::Node& node) -> morphizen::NodeAttributes& {
    return reinterpret_cast<morphizen::NodeAttributes&>(node);
  };

  the_mlir_instance_of_morphizen_ort_api.node_get_function_body =
      [](const morphizen::Node& node) -> const morphizen::Graph& {
    // Create MLIRNode wrapper to access function body functionality
    auto mlir_node = mlir_impl::MLIRNode(reinterpret_cast<mlir::Operation*>(
        const_cast<morphizen::Node*>(&node)));
    return *reinterpret_cast<const morphizen::Graph*>(
        mlir_node.getFunctionBody());
  };

  the_mlir_instance_of_morphizen_ort_api.node_type_is_fused =
      [](const morphizen::Node& node) -> bool {
    // Create MLIRNode wrapper for convenient access to fused node checking
    auto mlir_node = mlir_impl::MLIRNode(reinterpret_cast<mlir::Operation*>(
        const_cast<morphizen::Node*>(&node)));

    // Use MLIRNode's isFused method to check if the node represents a fused
    // operation
    return mlir_node.isFused();
  };

  // NodeAttributes API functions - MLIR implementation
  the_mlir_instance_of_morphizen_ort_api.node_attributes_new =
      []() -> morphizen::NodeAttributes* {
    // Create a new MLIRNodeAttributes with owned context
    auto* mlir_attrs = mlir_impl::MLIRNodeAttributes::Create();
    return reinterpret_cast<morphizen::NodeAttributes*>(mlir_attrs);
  };

  the_mlir_instance_of_morphizen_ort_api.node_attributes_delete =
      [](morphizen::NodeAttributes* p) -> void {
    if (p) {
      // reinterpret_cast<mlir::Operation*>(p);
    }
  };

  the_mlir_instance_of_morphizen_ort_api.node_attributes_add =
      [](morphizen::NodeAttributes& attrs,
         morphizen::AttributeProto&& attr_proto) -> void {
    auto mlir_node_attrs = mlir_impl::MLIRNodeAttributes(
        reinterpret_cast<mlir::Operation*>(&attrs));
    auto* mlir_named_attr =
        reinterpret_cast<mlir::NamedAttribute*>(&attr_proto);

    // Use the new add method which includes the replacement logic
    mlir_node_attrs.add(*mlir_named_attr);
  };

  the_mlir_instance_of_morphizen_ort_api.node_attributes_get =
      [](const morphizen::NodeAttributes& p,
         const std::string& name) -> const morphizen::AttributeProto* {
    auto mlir_attrs =
        mlir_impl::MLIRNodeAttributes(const_cast<mlir::Operation*>(
            reinterpret_cast<const mlir::Operation*>(&p)));
    if (mlir_attrs.has_attribute(name)) {
      return reinterpret_cast<const morphizen::AttributeProto*>(
          &mlir_attrs.get_mlir_attribute(name));
    }
    return nullptr;
  };

  the_mlir_instance_of_morphizen_ort_api.node_attributes_get_keys =
      [](morphizen::NodeAttributes& p)
      -> morphizen::DllSafe<std::vector<std::string>> {
    auto mlir_attrs =
        mlir_impl::MLIRNodeAttributes(reinterpret_cast<mlir::Operation*>(&p));
    return morphizen::DllSafe<std::vector<std::string>>(
        mlir_attrs.get_attribute_names());
  };

  // NodeArg API functions - to be implemented as needed
  the_mlir_instance_of_morphizen_ort_api.node_arg_get_name_unsafe =
      [](const morphizen::NodeArg& node_arg) -> const std::string& {
    return mlir_impl::MLIRNodeArgIndex::from_morphizen_core_node_arg_ptr(
               &node_arg)
        .get_name();
  };

  the_mlir_instance_of_morphizen_ort_api.node_arg_is_exists =
      [](const morphizen::NodeArg& node_arg) -> bool {
    // Convert the morphizen::NodeArg reference to MLIRNodeArgIndex
    auto node_arg_index =
        mlir_impl::MLIRNodeArgIndex::from_morphizen_core_node_arg_ptr(
            &node_arg);
    // Check if the node argument exists using validity check
    return node_arg_index.is_valid();
  };

  the_mlir_instance_of_morphizen_ort_api.node_arg_is_constant =
      [](const morphizen::Graph& /*graph*/,
         const morphizen::NodeArg& node_arg) -> bool {
    // Convert the morphizen::NodeArg reference to MLIRNodeArgIndex
    auto node_arg_index =
        mlir_impl::MLIRNodeArgIndex::from_morphizen_core_node_arg_ptr(
            &node_arg);
    return node_arg_index.is_constant();
  };

  the_mlir_instance_of_morphizen_ort_api.node_arg_clone =
      [](morphizen::Graph& graph, const morphizen::NodeArg& node_arg,
         const std::string& name) -> morphizen::NodeArg& {
    auto* mlir_graph = reinterpret_cast<mlir_impl::MLIRGraph*>(&graph);
    // TODO: Implement proper node argument cloning for MLIR
    LOG(WARNING) << "node_arg_clone not implemented in MLIR backend for: "
                 << name;
    (void)mlir_graph; // Suppress unused parameter warning
    (void)node_arg;
    // Return original for now - not safe for production use
    return *const_cast<morphizen::NodeArg*>(&node_arg);
  };

  the_mlir_instance_of_morphizen_ort_api.node_arg_new =
      [](morphizen::Graph& graph, const std::string& name,
         const std::vector<int64_t>* shape,
         int element_type) -> morphizen::NodeArg& {
    auto* mlir_graph = reinterpret_cast<mlir_impl::MLIRGraph*>(&graph);

    // ONNX→MLIR boundary: translate negative (dynamic) dims to kDynamic once.
    llvm::SmallVector<int64_t> small_shape;
    if (shape) {
      small_shape = mlir_impl::to_mlir_dims(*shape);
    }

    auto MLIRNodeArgIndex = mlir_graph->node_arg_new(
        name, shape ? &small_shape : nullptr, element_type);

    // Convert MLIRNodeArgIndex back to NodeArg* for the API
    const auto* node_arg_ptr = static_cast<const morphizen::NodeArg*>(
        MLIRNodeArgIndex.to_morphizen_core_node_arg_ptr());
    return *const_cast<morphizen::NodeArg*>(node_arg_ptr);
  };

  the_mlir_instance_of_morphizen_ort_api.node_arg_get_shape_i64_unsafe =
      [](const morphizen::NodeArg& node_arg)
      -> morphizen::DllSafe<std::vector<int64_t>> {
    auto node_arg_index =
        mlir_impl::MLIRNodeArgIndex::from_morphizen_core_node_arg_ptr(
            &node_arg);
    auto mlir_shape = node_arg_index.get_shape_i64();
    // Unranked: a DllSafe with get()==nullptr is the cross-backend signal that
    // morphizen-core's wrapper translates into a null shape, which is what
    // node_arg_is_unknown_shape() checks. Matches the ONNX-IR backend's
    // NodeArgIndex::get_shape_i64_unsafe() returning a nullptr vector.
    if (!mlir_shape) {
      return morphizen::DllSafe<std::vector<int64_t>>{};
    }
    // MLIR→ONNX boundary: translate kDynamic back to -1.
    auto vec_shape = mlir_impl::to_onnx_dims(*mlir_shape);
    return morphizen::DllSafe<std::vector<int64_t>>(vec_shape);
  };

  the_mlir_instance_of_morphizen_ort_api.node_arg_get_denotation_unsafe =
      [](const morphizen::NodeArg& /*node_arg*/)
      -> morphizen::DllSafe<std::vector<std::string>> {
    //"node_arg_get_denotation_unsafe not implemented in MLIR backend"
    return morphizen::DllSafe<std::vector<std::string>>(
        std::vector<std::string>{});
  };

  the_mlir_instance_of_morphizen_ort_api.node_arg_set_shape_i64 =
      [](const morphizen::NodeArg& node_arg,
         const std::vector<int64_t>& shape) -> void {
    // Convert the morphizen::NodeArg reference to MLIRNodeArgIndex
    auto node_arg_index =
        mlir_impl::MLIRNodeArgIndex::from_morphizen_core_node_arg_ptr(
            &node_arg);
    // ONNX→MLIR boundary: translate negative (dynamic) dims to kDynamic once.
    node_arg_index.set_shape_i64(mlir_impl::to_mlir_dims(shape));
  };

  the_mlir_instance_of_morphizen_ort_api.node_arg_set_denotation =
      [](const morphizen::NodeArg& node_arg,
         const std::vector<std::string>& denotation) -> void {
    // Convert the morphizen::NodeArg reference to MLIRNodeArgIndex
    auto node_arg_index =
        mlir_impl::MLIRNodeArgIndex::from_morphizen_core_node_arg_ptr(
            &node_arg);
    // TODO: Implement denotation setting for MLIR NodeArg
    LOG(WARNING) << "node_arg_set_denotation not implemented in MLIR backend, "
                    "denotation size: "
                 << denotation.size();
    (void)node_arg_index; // Suppress unused parameter warning
  };

  the_mlir_instance_of_morphizen_ort_api.node_arg_get_element_type =
      [](const morphizen::NodeArg& node_arg) -> int {
    // Convert the morphizen::NodeArg reference to MLIRNodeArgIndex
    auto node_arg_index =
        mlir_impl::MLIRNodeArgIndex::from_morphizen_core_node_arg_ptr(
            &node_arg);
    // Delegate to MLIRNodeArgIndex implementation which calls
    // MLIRNodeArg::getElementType()
    return node_arg_index.get_element_type();
  };

  the_mlir_instance_of_morphizen_ort_api.node_arg_set_element_type =
      [](morphizen::NodeArg& node_arg, int data_type) -> void {
    // Convert the morphizen::NodeArg reference to MLIRNodeArgIndex
    auto node_arg_index =
        mlir_impl::MLIRNodeArgIndex::from_morphizen_core_node_arg_ptr(
            &node_arg);
    node_arg_index.set_element_type(data_type);
  };

  the_mlir_instance_of_morphizen_ort_api.node_arg_get_const_data_as_tensor =
      [](const morphizen::Graph& /* graph*/,
         const morphizen::NodeArg& node_arg) -> const morphizen::TensorProto& {
    // auto* mlir_graph = reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph);
    auto node_arg_index =
        mlir_impl::MLIRNodeArgIndex::from_morphizen_core_node_arg_ptr(
            &node_arg);
    return reinterpret_cast<const morphizen::TensorProto&>(
        node_arg_index.get_const_data_as_tensor());
  };

  the_mlir_instance_of_morphizen_ort_api.node_arg_external_location =
      [](const morphizen::Graph& /*graph*/, const morphizen::NodeArg& node_arg,
         std::string& file, size_t& offset, size_t& size,
         size_t& checksum) -> int {
    auto node_arg_index =
        mlir_impl::MLIRNodeArgIndex::from_morphizen_core_node_arg_ptr(
            &node_arg);
    return node_arg_index.external_location(file, offset, size, checksum);
  };

  // AttributeProto API functions - implemented using MLIRNamedAttribute
  the_mlir_instance_of_morphizen_ort_api.attr_proto_new_ints =
      [](const std::string& name,
         const std::vector<int64_t>& data) -> morphizen::AttributeProto* {
    // Create MLIRNamedAttribute using factory method
    auto mlir_named_attr =
        mlir_impl::MLIRNamedAttribute::create_int_array(name, data);

    // Release ownership and return as AttributeProto*
    return reinterpret_cast<morphizen::AttributeProto*>(
        mlir_named_attr.release());
  };

  the_mlir_instance_of_morphizen_ort_api.attr_proto_new_floats =
      [](const std::string& name,
         const std::vector<float>& data) -> morphizen::AttributeProto* {
    auto mlir_named_attr =
        mlir_impl::MLIRNamedAttribute::create_float_array(name, data);
    return reinterpret_cast<morphizen::AttributeProto*>(
        mlir_named_attr.release());
  };

  the_mlir_instance_of_morphizen_ort_api.attr_proto_new_strings =
      [](const std::string& name,
         const std::vector<std::string>& data) -> morphizen::AttributeProto* {
    auto mlir_named_attr =
        mlir_impl::MLIRNamedAttribute::create_string_array(name, data);
    return reinterpret_cast<morphizen::AttributeProto*>(
        mlir_named_attr.release());
  };

  the_mlir_instance_of_morphizen_ort_api.attr_proto_new_int =
      [](const std::string& name, int64_t value) -> morphizen::AttributeProto* {
    auto mlir_named_attr =
        mlir_impl::MLIRNamedAttribute::create_int(name, value);
    return reinterpret_cast<morphizen::AttributeProto*>(
        mlir_named_attr.release());
  };

  the_mlir_instance_of_morphizen_ort_api.attr_proto_new_float =
      [](const std::string& name, float value) -> morphizen::AttributeProto* {
    auto mlir_named_attr =
        mlir_impl::MLIRNamedAttribute::create_float(name, value);
    return reinterpret_cast<morphizen::AttributeProto*>(
        mlir_named_attr.release());
  };

  the_mlir_instance_of_morphizen_ort_api.attr_proto_new_string =
      [](const std::string& name,
         const std::string& value) -> morphizen::AttributeProto* {
    auto mlir_named_attr =
        mlir_impl::MLIRNamedAttribute::create_string(name, value);
    return reinterpret_cast<morphizen::AttributeProto*>(
        mlir_named_attr.release());
  };

  the_mlir_instance_of_morphizen_ort_api.attr_proto_new_tensor =
      [](const std::string& name,
         const morphizen::TensorProto& value) -> morphizen::AttributeProto* {
    // Convert morphizen::TensorProto to MLIRNodeArg (which is MLIRTensor)
    auto* mlir_tensor = reinterpret_cast<const mlir_impl::MLIRNodeArg*>(&value);

    // Create a tensor attribute using MLIRNamedAttribute factory
    auto mlir_named_attr =
        mlir_impl::MLIRNamedAttribute::create_tensor(name, *mlir_tensor);
    return reinterpret_cast<morphizen::AttributeProto*>(
        mlir_named_attr.release());
  };

  // Business logic lives in MLIRGraph::new_subgraph and
  // MLIRNamedAttribute::create_subgraph_ref; these are boundary
  // translators only.
  the_mlir_instance_of_morphizen_ort_api.graph_new_subgraph =
      [](morphizen::Graph& parent_graph) -> morphizen::Graph& {
    auto& mlir_parent = *reinterpret_cast<mlir_impl::MLIRGraph*>(&parent_graph);
    return reinterpret_cast<morphizen::Graph&>(mlir_parent.new_subgraph());
  };

  the_mlir_instance_of_morphizen_ort_api.attr_proto_new_graph =
      [](const std::string& name,
         morphizen::Graph& sub) -> morphizen::AttributeProto* {
    auto& mlir_sub = *reinterpret_cast<mlir_impl::MLIRGraph*>(&sub);
    auto attr =
        mlir_impl::MLIRNamedAttribute::create_subgraph_ref(name, mlir_sub);
    return reinterpret_cast<morphizen::AttributeProto*>(attr.release());
  };

  the_mlir_instance_of_morphizen_ort_api.attr_proto_delete =
      [](morphizen::AttributeProto* attr) -> void {
    if (attr) {
      auto* mlir_named_attr = reinterpret_cast<mlir::NamedAttribute*>(attr);
      delete mlir_named_attr;
    }
  };

  the_mlir_instance_of_morphizen_ort_api.attr_proto_get_name =
      [](const morphizen::AttributeProto& attr) -> const std::string& {
    auto* mlir_named_attr =
        reinterpret_cast<const mlir::NamedAttribute*>(&attr);
    static thread_local std::string attr_name_cache;
    attr_name_cache = mlir_named_attr->getName().str();
    return attr_name_cache;
  };

  the_mlir_instance_of_morphizen_ort_api.attr_proto_get_type =
      [](const morphizen::AttributeProto& attr) -> int {
    // Return the MLIR attribute type mapped to ONNX attribute type constants
    // This is simplified - in a real implementation you'd need proper type
    // mapping
    return reinterpret_cast<const mlir_impl::MLIRNamedAttribute*>(&attr)
        ->get_onnx_type();
  };

  the_mlir_instance_of_morphizen_ort_api.attr_proto_set_name =
      [](morphizen::AttributeProto* attr, const std::string& name) -> void {
    reinterpret_cast<mlir_impl::MLIRNamedAttribute*>(attr)->set_name(name);
  };

  the_mlir_instance_of_morphizen_ort_api.attr_proto_clone =
      [](const morphizen::AttributeProto& attr) -> morphizen::AttributeProto* {
    auto* p = reinterpret_cast<const mlir::NamedAttribute*>(&attr);
    return reinterpret_cast<morphizen::AttributeProto*>(
        new mlir::NamedAttribute(p->getName(), p->getValue()));
  };
  the_mlir_instance_of_morphizen_ort_api.attr_proto_get_int =
      [](const morphizen::AttributeProto& attr) -> int64_t {
    auto* p = reinterpret_cast<const mlir_impl::MLIRNamedAttribute*>(&attr);
    return p->get_int();
  };
  the_mlir_instance_of_morphizen_ort_api.attr_proto_get_float =
      [](const morphizen::AttributeProto& attr) -> float {
    auto* p = reinterpret_cast<const mlir_impl::MLIRNamedAttribute*>(&attr);
    return static_cast<float>(p->get_float());
  };
  the_mlir_instance_of_morphizen_ort_api.attr_proto_get_string =
      [](const morphizen::AttributeProto& attr) -> const std::string& {
    auto* p = reinterpret_cast<const mlir_impl::MLIRNamedAttribute*>(&attr);
    return p->get_string();
  };
  the_mlir_instance_of_morphizen_ort_api.attr_proto_get_tensor =
      [](const morphizen::AttributeProto& attr)
      -> const morphizen::TensorProto& {
    auto* p = reinterpret_cast<const mlir_impl::MLIRNamedAttribute*>(&attr);
    return *reinterpret_cast<const morphizen::TensorProto*>(p->get_tensor());
  };
  the_mlir_instance_of_morphizen_ort_api.attr_proto_get_ints =
      [](const morphizen::AttributeProto& attr) -> gsl::span<const int64_t> {
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
  the_mlir_instance_of_morphizen_ort_api.attr_proto_get_floats =
      [](const morphizen::AttributeProto& attr) -> gsl::span<const float> {
    return reinterpret_cast<const mlir_impl::MLIRNamedAttribute*>(&attr)
        ->get_floats();
  };
  the_mlir_instance_of_morphizen_ort_api.attr_proto_get_strings =
      [](const morphizen::AttributeProto& attr) -> std::vector<std::string> {
    return reinterpret_cast<const mlir_impl::MLIRNamedAttribute*>(&attr)
        ->get_strings();
  };
  the_mlir_instance_of_morphizen_ort_api.attr_proto_release_string =
      [](morphizen::AttributeProto* attr) -> morphizen::DllSafe<std::string> {
    return morphizen::DllSafe<std::string>(
        reinterpret_cast<mlir_impl::MLIRNamedAttribute*>(attr)->get_string());
  };

  the_mlir_instance_of_morphizen_ort_api.graph_remove_initialized_tensor =
      [](morphizen::Graph& graph, const std::string& tensor_name) -> void {
    auto* mlir_graph = reinterpret_cast<mlir_impl::MLIRGraph*>(&graph);
    mlir_graph->remove_initialized_tensor(tensor_name);
  };

  the_mlir_instance_of_morphizen_ort_api.graph_reverse_dfs_from_preemp =
      [](const morphizen::Graph& graph,
         gsl::span<const morphizen::Node* const> from,
         const std::function<bool(const morphizen::Node*)>& enter,
         const std::function<bool(const morphizen::Node*)>& leave,
         const std::function<bool(const morphizen::Node*,
                                  const morphizen::Node*)>& comp,
         const std::function<bool(const morphizen::Node* from,
                                  const morphizen::Node* to)>& stop) -> void {
    auto* mlir_graph = reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph);

    // Convert from morphizen::Node* to mlir::Operation*
    std::vector<const mlir::Operation*> mlir_from;
    mlir_from.reserve(from.size());
    for (const auto* node_ptr : from) {
      mlir_from.push_back(reinterpret_cast<const mlir::Operation*>(node_ptr));
    }

    // Create wrapper functions that convert between mlir::Operation* and
    // morphizen::Node*
    auto mlir_enter = enter ? [&enter](const mlir::Operation* op) -> bool {
      const auto* node_ptr = reinterpret_cast<const morphizen::Node*>(op);
      return enter(node_ptr);
    }
    : std::function<bool(const mlir::Operation*)>();

    auto mlir_leave = leave ? [&leave](const mlir::Operation* op) -> bool {
      const auto* node_ptr = reinterpret_cast<const morphizen::Node*>(op);
      return leave(node_ptr);
    }
    : std::function<bool(const mlir::Operation*)>();

    auto mlir_comp = comp ? [&comp](const mlir::Operation* from_op,
                                    const mlir::Operation* to_op) -> bool {
      const auto* from_ptr = reinterpret_cast<const morphizen::Node*>(from_op);
      const auto* to_ptr = reinterpret_cast<const morphizen::Node*>(to_op);
      return comp(from_ptr, to_ptr);
    }
    : std::function<bool(const mlir::Operation*, const mlir::Operation*)>();

    auto mlir_stop = stop ? [&stop](const mlir::Operation* from_op,
                                    const mlir::Operation* to_op) -> bool {
      const auto* from_ptr = reinterpret_cast<const morphizen::Node*>(from_op);
      const auto* to_ptr = reinterpret_cast<const morphizen::Node*>(to_op);
      return stop(from_ptr, to_ptr);
    }
    : std::function<bool(const mlir::Operation*, const mlir::Operation*)>();

    // Call the MLIRGraph method with converted parameters
    mlir_graph->reverse_dfs_from_preemp(
        gsl::span<const mlir::Operation* const>(mlir_from), mlir_enter,
        mlir_leave, mlir_comp, mlir_stop);
  };

  the_mlir_instance_of_morphizen_ort_api.graph_infer_shapes_from_filepath =
      [](const std::string& model_path, const std::string& save_path) -> void {
    // TODO: Implement shape inference from file paths in MLIR backend
    LOG(WARNING)
        << "graph_infer_shapes_from_filepath not implemented in MLIR backend";
    (void)model_path;
    (void)save_path; // Suppress unused parameter warnings
  };

  the_mlir_instance_of_morphizen_ort_api.graph_to_graph_proto =
      [](const morphizen::Graph& graph) -> morphizen::GraphProto* {
    auto* mlir_graph = const_cast<mlir_impl::MLIRGraph*>(
        reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph));
    return reinterpret_cast<morphizen::GraphProto*>(mlir_graph);
  };

  the_mlir_instance_of_morphizen_ort_api.graph_proto_delete =
      [](morphizen::GraphProto* /* p */) -> void {
    // do nothing, ~MLIRGraph release itself
  };

  the_mlir_instance_of_morphizen_ort_api.graph_infer_shapes =
      [](morphizen::ModelProto& model_proto) -> void {
    // TODO: Implement shape inference in MLIR backend
    LOG(WARNING) << "graph_infer_shapes not implemented in MLIR backend";
    (void)model_proto; // Suppress unused parameter warning
  };
#if MORPHIZEN_ORT_API_MAJOR >= 18
  the_mlir_instance_of_morphizen_ort_api.graph_save_string =
      [](const morphizen::Graph& graph) -> morphizen::DllSafe<std::string> {
    auto* mlir_graph = reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph);
    auto graph_string = mlir_graph->save_string();
    return morphizen::DllSafe<std::string>(
        new std::string(std::move(graph_string)));
  };
#endif // MORPHIZEN_ORT_API_MAJOR >= 18

  the_mlir_instance_of_morphizen_ort_api.get_model_path =
      [](const morphizen::Graph& graph) -> const std::filesystem::path& {
    auto mlir_graph = reinterpret_cast<const mlir_impl::MLIRGraph*>(&graph);
    auto& model = mlir_graph->get_model();
    static thread_local std::filesystem::path model_path;
    model_path = model.has_metadata_prop("model_path")
                     ? model.get_metadata_prop("model_path")
                     : "";
    return model_path;
  };

  the_mlir_instance_of_morphizen_ort_api.session_option_configuration =
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

  the_mlir_instance_of_morphizen_ort_api.model_to_proto =
      [](morphizen::Model& model) -> morphizen::ModelProto* {
    auto* mlir_model = reinterpret_cast<mlir_impl::MLIRModel*>(&model);
    return reinterpret_cast<morphizen::ModelProto*>(mlir_model);
  };

  the_mlir_instance_of_morphizen_ort_api.model_proto_serialize_as_string =
      [](morphizen::ModelProto& model_proto)
      -> morphizen::DllSafe<std::string> {
    auto* mlir_model = reinterpret_cast<mlir_impl::MLIRModel*>(&model_proto);
    return morphizen::DllSafe<std::string>(mlir_model->serialize_as_string());
  };

  the_mlir_instance_of_morphizen_ort_api.model_proto_delete =
      [](morphizen::ModelProto* /* p*/) -> void {
    // do nothing, ~MLIRModel release itself
  };

  the_mlir_instance_of_morphizen_ort_api.is_profiling_enabled =
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
extern "C" MorphizenOrtApiExt* get_morphizen_ort_api_mlir() {
  initialize_mlir_api();
  return &the_mlir_instance_of_morphizen_ort_api;
}

} // namespace morphizen

// Function for plugin registration with proper naming
namespace {
const morphizen::OrtApiForMorphizen*
morphizen_mlir_imp_get_morphizen_ort_api() {
  return static_cast<const morphizen::OrtApiForMorphizen*>(
      morphizen::get_morphizen_ort_api_mlir());
}

static ::morphizen::StaticPluginRegister
    __register(morphizen::kMLIRBackend, "morphizen_ort_api_imp",
               (void*)&morphizen_mlir_imp_get_morphizen_ort_api);
} // namespace

// Factory function for creating MLIR-based models
namespace onnxruntime {
// Placeholder for integration with ONNXRuntime if needed
}
