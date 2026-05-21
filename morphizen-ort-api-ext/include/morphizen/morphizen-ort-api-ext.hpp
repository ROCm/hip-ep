/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include <morphizen/morphizen_ort_api.h>
#include <string>
#include <vector>

namespace morphizen {

static constexpr const char* kONNXIRBackend = "onnx-ir-imp";
static constexpr const char* kMLIRBackend = "mlir-backend";

// New morphizen-prefixed API
uint32_t get_morphizen_version_major();
uint32_t get_morphizen_version_minor();
uint32_t get_morphizen_version_patch();

struct MorphizenOrtApiExt : public morphizen::OrtApiForMorphizen {
  // Create a sub Graph owned by `parent_graph`. The returned reference is
  // valid for the parent's lifetime. The sub Graph is unattached until
  // graph_add_node consumes the corresponding attr_proto_new_graph.
  morphizen::Graph& (*graph_new_subgraph)(morphizen::Graph& parent_graph);

  // Wrap a sub Graph (from graph_new_subgraph) as a GRAPH-typed
  // AttributeProto. graph_add_node uses it as the body of the resulting
  // node (e.g. onnx.If then_branch / else_branch, onnx.Loop / Scan body).
  morphizen::AttributeProto* (*attr_proto_new_graph)(
      const std::string& name, //
      morphizen::Graph& sub_graph);

  morphizen::TensorProto* (*tensor_proto_new_with_external_data)(
      const std::string& name,           //
      const std::vector<int64_t>& shape, //
      int element_type,                  //
      const std::string& location_file,  //
      size_t location_size,              //
      size_t location_offset             //
  );
  morphizen::TensorProto* (*tensor_proto_new_raw_data)(
      const std::string& name,           //
      const std::vector<int64_t>& shape, //
      int element_type,                  //
      const void* data,                  //
      size_t size);

  morphizen::DllSafe<std::vector<morphizen::NodeInput>> (
      *node_get_implicit_inputs_unsafe)(const morphizen::Node& node);
};

/**
 * @brief Create and initialize the MorphiZen ORT API
 * @return an opaque pointer which restore api.
 */
std::shared_ptr<void> setup_global_morphizen_ort_api(const char* backend_ir);

const morphizen::OrtApiForMorphizen*
get_global_morphizen_ort_api(const char* ir_backend_name);

const morphizen::OrtApiForMorphizen* get_the_global_api_unsafe();

MORPHIZEN_DLL_SPEC void set_the_global_api(morphizen::OrtApiForMorphizen* api);
MORPHIZEN_DLL_SPEC const morphizen::OrtApiForMorphizen* api();

} // namespace morphizen

#define MORPHIZEN_ORT_API_EXT(name)                                            \
  (static_cast<const ::morphizen::MorphizenOrtApiExt*>(::morphizen::api())     \
               ->name != nullptr                                               \
       ? static_cast<const ::morphizen::MorphizenOrtApiExt*>(                  \
             ::morphizen::api())                                               \
             ->name                                                            \
       : (assert(false && #name " is not set"), nullptr))
