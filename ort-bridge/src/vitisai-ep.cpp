/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./vitisai-ep.hpp"
#include "./graph.hpp"
#include "./ir-converter.hpp"
#include "glog/logging.h"
#include "morphizen-utils/morphizen-utils.hpp"
DEF_ENV_PARAM(MORPHIZEN_DEBUG_VITISAI_EP, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_VITISAI_EP) >= n)
namespace morphizen {

VitisAIEP::VitisAIEP(ApiPtrs apis, const std::string& name,
                     const OrtSessionOptions& session_options,
                     const OrtLogger& logger)
    : ApiPtrs(apis), name_{name}, logger_{logger} {
  // Initialize the execution provider.
  auto status = ort_api.Logger_LogMessage(
      &logger_, OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
      ("ExampleEp has been created with name " + name_).c_str(), ORT_FILE,
      __LINE__, __FUNCTION__);
  // ignore status for now
  (void)status;

  // Can get configurations from the session options.
  // Note: should not store a direct reference to the session options object
  // as its lifespan is not guaranteed. EP should copy any configurations or
  // settings it needs.
  (void)session_options;

  ort_version_supported =
      ORT_API_VERSION; // set to the ORT version we were compiled with.
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  Compile = CompileImpl;
  ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
  ir_converter = morphizen::IRConverter::to_onnx_model;
}
VitisAIEP::~VitisAIEP() {}

const char* ORT_API_CALL VitisAIEP::GetNameImpl(const OrtEp* this_ptr) {
  const auto* ep = static_cast<const VitisAIEP*>(this_ptr);
  return ep->name_.c_str();
}

OrtStatus* ORT_API_CALL
VitisAIEP::GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
                             OrtEpGraphSupportInfo* /*graph_support_info*/) {
  VitisAIEP* self = static_cast<VitisAIEP*>(this_ptr);

  if (self->ir_converter == nullptr) {
    MY_LOG(1) << " no available IR converter.";
    return nullptr;
  }

  // Use the Graph class instead of manual API calls
  auto graph_wrapper = morphizen::Graph(*self, *graph);
  auto ir_graph = self->ir_converter(*self, *graph);

  // Get nodes using the Graph class
  auto nodes_span = graph_wrapper.nodes();

  if (nodes_span.empty()) {
    return nullptr; // No nodes to process
  }

  std::vector<const OrtNode*> supported_nodes;

  for (const OrtNode* node_in_c : nodes_span) {
    const char* node_name = nullptr;
    self->throw_if_error(self->ort_api.Node_GetName(node_in_c, &node_name));
    MY_LOG(1) << node_name;
  }
  /*if (!supported_nodes.empty()) {
    RETURN_IF_ERROR(self->ep_api.EpGraphSupportInfo_AddNodesToFuse(
        graph_support_info, supported_nodes.data(), supported_nodes.size()));
  }*/
  return nullptr;
}
OrtStatus* ORT_API_CALL
VitisAIEP::CompileImpl(OrtEp* /*this_ptr*/, const OrtGraph** /*graphs*/,
                       const OrtNode** /*fused_nodes*/, size_t /*count*/,
                       OrtNodeComputeInfo** /*node_compute_infos*/) {
  // VitisAIEP* ep = static_cast<VitisAIEP*>(this_ptr);

  return nullptr;
}
void ORT_API_CALL VitisAIEP::ReleaseNodeComputeInfosImpl(
    OrtEp* /*this_ptr*/, OrtNodeComputeInfo** /*node_compute_infos*/,
    size_t /*num_node_compute_infos*/) {}

} // namespace morphizen
