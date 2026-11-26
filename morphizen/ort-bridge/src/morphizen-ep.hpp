/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "./api-ptrs.hpp"
#include "./backend-routing.hpp"
#include "./ir-converter.hpp"
#include "./migraphx-backend.hpp"
#include "morphizen/custom_op.h" // in onnxruntime/core/providers/morphizen/include
#include "morphizen/morphizen-ort-api-ext.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace onnxruntime {
struct OrtGraphWrapper;
}
namespace morphizen {

class MorphiZenEP : public OrtEp, ApiPtrs {
public:
  MorphiZenEP(ApiPtrs apis, const std::string &name,
              const OrtKeyValuePairs *const *ep_metadata,
              const OrtSessionOptions &session_options,
              const OrtLogger &logger);
  ~MorphiZenEP();
  static const char *ORT_API_CALL GetNameImpl(const OrtEp *this_ptr) noexcept;

  static OrtStatus *ORT_API_CALL
  GetCapabilityImpl(OrtEp *this_ptr, const OrtGraph *graph,
                    OrtEpGraphSupportInfo *graph_support_info) noexcept;

  static OrtStatus *ORT_API_CALL CompileImpl(
      OrtEp *this_ptr, const OrtGraph **graphs, const OrtNode **fused_nodes,
      size_t count, OrtNodeComputeInfo **node_compute_infos,
      OrtNode **ep_context_nodes) noexcept;

  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(
      OrtEp *this_ptr, OrtNodeComputeInfo **node_compute_infos,
      size_t num_node_compute_infos) noexcept;
  static const char *ORT_API_CALL GetCompiledModelCompatibilityInfoImpl(
      OrtEp *this_ptr, const OrtGraph *graph) noexcept;

  static OrtStatus *ORT_API_CALL
  OnRunStartImpl(OrtEp *this_ptr, const OrtRunOptions *run_options) noexcept;

  static OrtStatus *ORT_API_CALL OnRunEndImpl(OrtEp *this_ptr,
                                              const OrtRunOptions *run_options,
                                              bool sync_stream) noexcept;

private:
  OrtStatus *GetCapability(OrtGraphWrapper &graph_wrapper,
                           OrtEpGraphSupportInfo &graph_support_info);
  OrtStatus *Compile(const OrtGraph **graphs, const OrtNode **fused_nodes,
                     size_t count, OrtNodeComputeInfo **node_compute_infos,
                     OrtNode **ep_context_nodes);
  OrtStatus *ReleaseNodeComputeInfos(OrtNodeComputeInfo **node_compute_infos,
                                     size_t num_node_compute_infos);
  /**
   * @brief Compile a single subgraph for a specific execution provider
   * @param ep The execution provider to compile for
   * @param graph The graph containing the subgraph
   * @param fused_node The fused node representing the subgraph
   * @param node_compute_info Output parameter for node compute information
   * @param ep_context_node Output parameter for EP context node
   * @return OrtStatus indicating success or failure
   */
  OrtStatus *CompileSubgraph(const morphizen::ExecutionProvider &ep,
                             const OrtGraph *graph, const OrtNode *fused_node,
                             OrtNodeComputeInfo *&node_compute_info,
                             OrtNode *&ep_context_node);
  OrtStatus *CreateEpContextNodes(const OrtNode **fused_nodes,
                                  OrtNode **ep_context_nodes,
                                  size_t count) const;
  OrtNode *
  convert_morphizen_node_to_ort_node(const OrtModelEditorApi *model_editor_api,
                                     const onnxruntime::Node *morphizen_node,
                                     const OrtNode *ort_node) const;
  /**
   * @brief Update provider options from ORT session configuration
   * @param session_options The ORT session options containing configuration
   * entries
   */
  void update_provider_options_from_session_config(
      const OrtSessionOptions &session_options);

  /**
   * @brief Update provider options from EP metadata key-value pairs
   * @param ep_metadata The EP metadata containing key-value pairs, can be null
   */
  void update_provider_options_from_ep_metadata(
      const OrtKeyValuePairs *const *ep_metadata);

  OrtStatus *
  GetSessionConfigEntryOrDefault(const OrtSessionOptions &session_options,
                                 const char *config_key,
                                 const std::string &default_val,
                                 /*out*/ std::string &config_val);
  void update_input_output_argument_indice(morphizen::ExecutionProvider &ep,
                                           const OrtNode *fused_node);
  // Whole-graph MIGraphX path (docs/design/migraphx-backend.md).
  OrtStatus *ClaimForMigraphx(OrtGraphWrapper &graph_wrapper,
                              OrtEpGraphSupportInfo &graph_support_info,
                              const std::string &why);
  OrtStatus *CompileMigraphx(const OrtNode *fused_node,
                             OrtNodeComputeInfo *&node_compute_info);
  void log_routing(const char *backend, const std::string &reason) const;

private:
  std::string name_;
  const OrtLogger &logger_;
  ModelUniquePtr (*ir_converter)(const ApiPtrs &api_ptrs, const OrtGraph &graph,
                                 const IRConverterConfig &config) = nullptr;
  // copied from
  // onnxruntime/core/providers/morphizen/morphizen_execution_provider.h
  using my_ep_t = morphizen::DllSafe<
      std::vector<std::unique_ptr<morphizen::ExecutionProvider>>>;
  using my_ep_uptr_t = std::shared_ptr<my_ep_t>;
  // we have to hide the implementation by forward declaration.
  mutable my_ep_uptr_t execution_providers_;
  std::unordered_map<std::string, std::string> provider_options_;
  std::map<std::string, std::string> session_configs_;
  bool enable_ep_context_;
  GpuBackend gpu_backend_ = GpuBackend::Auto;
  // Set when the graph was routed to MIGraphX; execution_providers_ is then
  // null. Graph output names in graph order, matching the program outputs.
  std::shared_ptr<MigraphxProgram> mgx_program_;
  std::vector<std::string> mgx_output_names_;
};
struct MorphiZenEP_ComputeInfo : public OrtNodeComputeInfo {
  morphizen::ExecutionProvider *morphizen_ep =
      nullptr; // Pointer to the MorphiZenEP instance
  // "hip_ep.graph.<model> sub=<subgraph>" ROCTx range around each Compute;
  // empty when ROCTx is off.
  std::string roctx_label;
  // MIGraphX path only: the shared program and, per fused-node input/output
  // position, the program input index / graph output index.
  std::shared_ptr<MigraphxProgram> mgx_program;
  std::vector<size_t> mgx_inputs;
  std::vector<size_t> mgx_outputs;
};
} // namespace morphizen
