/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./graph.hpp"

namespace morphizen {

Graph::Graph(const ApiPtrs& api_ptrs, const OrtGraph& graph)
    : ApiPtrs(api_ptrs), B(&graph) {}

const char* Graph::name() const {
  const char* ret = nullptr;
  throw_if_error(ort_api.Graph_GetName(p_, &ret));
  return ret;
}

int64_t Graph::ir_version() const {
  int64_t ret = 0;
  throw_if_error(ort_api.Graph_GetOnnxIRVersion(p_, &ret));
  return ret;
}

// RAII-managed array access methods
OrtArraySpan<const OrtNode* const> Graph::nodes_managed() const {
  OrtArrayOfConstObjects* nodes_array = nullptr;
  throw_if_error(ort_api.Graph_GetNodes(p_, &nodes_array));
  return make_array_span<const OrtNode* const>(nodes_array);
}

OrtArraySpan<const OrtValueInfo* const> Graph::inputs_managed() const {
  OrtArrayOfConstObjects* inputs_array = nullptr;
  throw_if_error(ort_api.Graph_GetInputs(p_, &inputs_array));
  return make_array_span<const OrtValueInfo* const>(inputs_array);
}

OrtArraySpan<const OrtValueInfo* const> Graph::outputs_managed() const {
  OrtArrayOfConstObjects* outputs_array = nullptr;
  throw_if_error(ort_api.Graph_GetOutputs(p_, &outputs_array));
  return make_array_span<const OrtValueInfo* const>(outputs_array);
}

OrtArraySpan<const OrtValueInfo* const> Graph::initializers_managed() const {
  OrtArrayOfConstObjects* initializers_array = nullptr;
  throw_if_error(ort_api.Graph_GetInitializers(p_, &initializers_array));
  return make_array_span<const OrtValueInfo* const>(initializers_array);
}

// Convenience methods that copy to vector
std::vector<const OrtNode*> Graph::nodes() const {
  auto managed = nodes_managed();
  return std::vector<const OrtNode*>(managed.begin(), managed.end());
}

std::vector<const OrtValueInfo*> Graph::inputs() const {
  auto managed = inputs_managed();
  return std::vector<const OrtValueInfo*>(managed.begin(), managed.end());
}

std::vector<const OrtValueInfo*> Graph::outputs() const {
  auto managed = outputs_managed();
  return std::vector<const OrtValueInfo*>(managed.begin(), managed.end());
}

std::vector<const OrtValueInfo*> Graph::initializers() const {
  auto managed = initializers_managed();
  return std::vector<const OrtValueInfo*>(managed.begin(), managed.end());
}

std::map<std::string, int> Graph::guess_opset() const {
  auto opset = std::map<std::string, int>{};
  for (const OrtNode* node_in_c : nodes()) {
    const char* domain = nullptr;
    int since_version = 0;
    throw_if_error(ort_api.Node_GetDomain(node_in_c, &domain));
    throw_if_error(ort_api.Node_GetSinceVersion(node_in_c, &since_version));
    if (domain == nullptr || domain[0] == '\0') {
      // Default domain is empty string
      domain = "ai.onnx";
    }
    auto it = opset.find(domain);
    if (it == opset.end()) {
      opset[domain] = since_version;
    } else {
      if (it->second < since_version) {
        it->second = since_version;
      }
    }
  }
  return opset;
}

} // namespace morphizen
