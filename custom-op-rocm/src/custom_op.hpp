// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>
#include <miopen/miopen.h>
#include <hipblaslt/hipblaslt.h>
#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <chrono>
#include <thread>
#include <glog/logging.h>

#include "morphizen/vaip.hpp"
#include "morphizen/env_config.hpp"
#include "rocm.pb.h"

namespace rocm_ep {

// Environment variables for timeout configuration are defined in custom_op.cpp
// They are declared here for reference:
//   MORPHIZEN_GPU_TIMEOUT_MS - Default 5 second timeout (5000)
//   MORPHIZEN_GPU_WATCHDOG_ENABLED - Enable watchdog by default (1)
//   MORPHIZEN_DEBUG_ROCM - Debug logging level (0)

/**
 * GPU Operation Timeout Result
 */
enum class TimeoutStatus {
  SUCCESS,           // Operation completed successfully
  TIMEOUT,          // Operation timed out
  ERROR             // Error occurred
};

/**
 * Helper function to wait for HIP stream with timeout
 * 
 * @param stream HIP stream to wait for
 * @param timeout_ms Timeout in milliseconds
 * @return TimeoutStatus indicating success, timeout, or error
 */
inline TimeoutStatus WaitStreamWithTimeout(hipStream_t stream, int timeout_ms) {
  const int poll_interval_ms = 10;  // Poll every 10ms
  auto start = std::chrono::steady_clock::now();
  
  while (true) {
    // Query stream status (non-blocking)
    hipError_t err = hipStreamQuery(stream);
    
    if (err == hipSuccess) {
      // All operations completed
      return TimeoutStatus::SUCCESS;
    } else if (err == hipErrorNotReady) {
      // Operations still in progress
      auto elapsed = std::chrono::steady_clock::now() - start;
      auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
      
      if (elapsed_ms >= timeout_ms) {
        LOG(ERROR) << "[ROCm Timeout] GPU operation timed out after " << elapsed_ms << "ms";
        return TimeoutStatus::TIMEOUT;
      }
      
      // Sleep briefly to avoid busy-waiting
      std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms));
    } else {
      // Error occurred
      LOG(ERROR) << "[ROCm Timeout] hipStreamQuery failed: " 
                 << hipGetErrorString(err) << " (" << err << ")";
      return TimeoutStatus::ERROR;
    }
  }
}

/**
 * Shared HIP Context Singleton
 * 
 * Provides a single shared HIP stream for all ROCm operations.
 * This enables implicit fusion between Conv and Gemm operations.
 * Includes timeout handling to prevent indefinite GPU waits.
 */
class HipContext {
public:
  static HipContext& instance() {
    static HipContext ctx;
    return ctx;
  }

  HipContext(const HipContext&) = delete;
  HipContext& operator=(const HipContext&) = delete;

  hipStream_t stream() {
    ensure_initialized();
    return stream_;
  }

  miopenHandle_t miopen_handle() {
    ensure_initialized();
    return miopen_handle_;
  }

  hipblasLtHandle_t hipblaslt_handle() {
    ensure_initialized();
    return hipblaslt_handle_;
  }

  bool is_initialized() {
    ensure_initialized();
    return initialized_;
  }

  /**
   * Synchronize stream with timeout protection
   * 
   * @param timeout_ms Timeout in milliseconds (0 = use default from env var)
   * @return TimeoutStatus indicating success, timeout, or error
   */
  TimeoutStatus sync_stream_with_timeout(int timeout_ms = 0);

private:
  HipContext() = default;

  ~HipContext() {
    if (initialized_) {
      hipblasLtDestroy(hipblaslt_handle_);
      miopenDestroy(miopen_handle_);
      hipStreamDestroy(stream_);
    }
  }

  void ensure_initialized();

  std::once_flag init_flag_;
  bool initialized_ = false;
  hipStream_t stream_ = nullptr;
  miopenHandle_t miopen_handle_ = nullptr;
  hipblasLtHandle_t hipblaslt_handle_ = nullptr;
};

/**
 * Per-node runtime data for subgraph execution
 * Stores device buffers and descriptors for a single operation
 */
struct NodeRuntimeData {
  // Output buffers for this node (indexed by output_index)
  std::vector<float*> output_buffers;
  std::vector<size_t> output_sizes;
  
  // Weight buffers (cached on GPU)
  float* d_weight = nullptr;
  float* d_bias = nullptr;
  
  // Host-side cached weight/bias data
  std::vector<float> host_weight;
  std::vector<float> host_bias;
  
  // Workspace for this node
  void* workspace = nullptr;
  size_t workspace_size = 0;
  
  ~NodeRuntimeData() {
    for (auto* buf : output_buffers) {
      if (buf) hipFree(buf);
    }
    if (d_weight) hipFree(d_weight);
    if (d_bias) hipFree(d_bias);
    if (workspace) hipFree(workspace);
  }
};

/**
 * Unified ROCm Custom Op for Subgraph Execution
 * 
 * Executes a RocmSubgraphProto containing multiple operations
 * (Conv, Gemm) on a shared HIP stream. Intermediate tensors
 * stay on GPU, only external inputs/outputs are transferred.
 * 
 * Supports:
 * - Multi-node subgraph execution
 * - Topology-aware tensor routing (TensorRefProto)
 * - Async overlapped D2H transfers (ExternalOutputProto)
 * - Cached weights per node
 */
class RocmCustomOp : public vaip_core::CustomOpImp {
public:
  RocmCustomOp(std::shared_ptr<const vaip_core::PassContext> context,
               const std::shared_ptr<vaip_core::MetaDefProto>& meta_def,
               onnxruntime::Model* model);

  virtual ~RocmCustomOp();

private:
  void Compute(const OrtApi* api, OrtKernelContext* context) const override;

  // Subgraph execution phases
  void UploadExternalInputs(const OrtApi* api, OrtKernelContext* context) const;
  void ExecuteSubgraph(const OrtApi* api, OrtKernelContext* context) const;
  void DownloadExternalOutputs(const OrtApi* api, OrtKernelContext* context) const;

  // Node execution
  void ExecuteNode(const rocm::RocmNodeProto& node) const;
  void ExecuteConvNode(const rocm::ConvParamProto& params, 
                       const std::vector<float*>& inputs,
                       float* output) const;
  void ExecuteGemmNode(const rocm::GemmParamProto& params,
                       const std::vector<float*>& inputs,
                       float* output) const;

  // Tensor resolution
  float* ResolveTensorRef(const rocm::TensorRefProto& ref) const;
  size_t GetOutputSize(int32_t node_id, int32_t output_index) const;

  // Initialization
  void LoadAllWeights();
  void LoadNodeWeights(int32_t node_id, const rocm::RocmParamProto& params);
  void AllocateIntermediateBuffers();

private:
  // Subgraph definition (either single-node or multi-node)
  rocm::RocmSubgraphProto subgraph_;
  
  // Fallback for single-node case (backward compatibility)
  bool is_single_node_ = false;
  rocm::RocmParamProto single_node_proto_;

  // Per-node runtime data
  mutable std::vector<std::unique_ptr<NodeRuntimeData>> node_data_;

  // External input buffers (name -> device pointer)
  mutable std::unordered_map<std::string, float*> external_input_buffers_;
  mutable std::unordered_map<std::string, size_t> external_input_sizes_;

  // Output index mapping for ORT (output_name -> ORT output index)
  std::unordered_map<std::string, size_t> output_name_to_index_;

  // Cached weights loaded flag
  mutable bool weights_loaded_ = false;
  mutable bool buffers_allocated_ = false;
};

} // namespace rocm_ep
