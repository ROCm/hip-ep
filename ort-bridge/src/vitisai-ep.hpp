/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "./api-ptrs.hpp"
namespace onnxruntime {
struct Graph;
}
namespace morphizen {

class VitisAIEP : public OrtEp, ApiPtrs {
public:
  VitisAIEP(ApiPtrs apis, const std::string& name,
            const OrtSessionOptions& session_options, const OrtLogger& logger);
  ~VitisAIEP();
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr);

  static OrtStatus* ORT_API_CALL
  GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
                    OrtEpGraphSupportInfo* graph_support_info);

  static OrtStatus* ORT_API_CALL CompileImpl(
      OrtEp* this_ptr, const OrtGraph** graphs, const OrtNode** fused_nodes,
      size_t count, OrtNodeComputeInfo** node_compute_infos);

  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(
      OrtEp* this_ptr, OrtNodeComputeInfo** node_compute_infos,
      size_t num_node_compute_infos);

private:
  std::string name_;
  const OrtLogger& logger_;
  ModelUniquePtr (*ir_converter)(const ApiPtrs& api_ptrs,
                                 const OrtGraph& graph) = nullptr;
};
} // namespace morphizen
