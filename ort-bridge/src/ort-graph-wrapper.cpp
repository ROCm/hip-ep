/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./ort-graph-wrapper.hpp"

namespace morphizen {

OrtGraphWrapper::OrtGraphWrapper(const ApiPtrs& api_ptrs, const OrtGraph& graph)
    : ApiPtrs(api_ptrs), B(&graph) {}
const OrtGraph& OrtGraphWrapper::get() const {
  // Return the underlying graph pointer
  return *static_cast<const OrtGraph*>(this->p_);
}
bool OrtGraphWrapper::is_subgraph() const {
  return Ort::ConstGraph(p_).GetParentNode() != nullptr;
}
const char* OrtGraphWrapper::name() const {
  const char* ret = nullptr;
  throw_if_error(ort_api.Graph_GetName(p_, &ret));
  return ret;
}

int64_t OrtGraphWrapper::ir_version() const {
  int64_t ret = 0;
  throw_if_error(ort_api.Graph_GetOnnxIRVersion(p_, &ret));
  return ret;
}
Ort::ModelMetadata OrtGraphWrapper::get_model_metadata() const {
  return Ort::ConstGraph(this->p_).GetModelMetadata();
}
// Convenience methods that copy to vector
std::vector<const OrtNode*> OrtGraphWrapper::nodes() const {
  auto ret = std::vector<const OrtNode*>{};
  size_t num_nodes = 0;
  throw_if_error(ort_api.Graph_GetNumNodes(p_, &num_nodes));
  ret.resize(num_nodes);
  throw_if_error(ort_api.Graph_GetNodes(p_, ret.data(), ret.size()));
  return ret;
}

std::vector<const OrtValueInfo*> OrtGraphWrapper::inputs() const {
  auto ret = std::vector<const OrtValueInfo*>{};
  size_t num_inputs = 0;
  throw_if_error(ort_api.Graph_GetNumInputs(p_, &num_inputs));
  ret.resize(num_inputs);
  throw_if_error(ort_api.Graph_GetInputs(p_, ret.data(), ret.size()));
  return ret;
}

std::vector<const OrtValueInfo*> OrtGraphWrapper::outputs() const {
  auto ret = std::vector<const OrtValueInfo*>{};
  size_t num_outputs = 0;
  throw_if_error(ort_api.Graph_GetNumOutputs(p_, &num_outputs));
  ret.resize(num_outputs);
  throw_if_error(ort_api.Graph_GetOutputs(p_, ret.data(), ret.size()));
  return ret;
}

std::vector<const OrtValueInfo*> OrtGraphWrapper::initializers() const {
  auto ret = std::vector<const OrtValueInfo*>{};
  size_t num_initializers = 0;
  throw_if_error(ort_api.Graph_GetNumInitializers(p_, &num_initializers));
  ret.resize(num_initializers);
  throw_if_error(ort_api.Graph_GetInitializers(p_, ret.data(), ret.size()));
  return ret;
}

std::map<std::string, int> OrtGraphWrapper::guess_opset() const {
  auto opset = std::map<std::string, int>{};
  for (const OrtNode* node_in_c : nodes()) {
    const char* domain = nullptr;
    int since_version = 0;
    throw_if_error(ort_api.Node_GetDomain(node_in_c, &domain));
    throw_if_error(ort_api.Node_GetSinceVersion(node_in_c, &since_version));
    if (domain == nullptr || domain[0] == '\0') {
      // Default domain is empty string
      domain = "";
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
