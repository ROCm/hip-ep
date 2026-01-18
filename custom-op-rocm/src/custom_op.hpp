// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>
#include <miopen/miopen.h>
#include <hipblaslt/hipblaslt.h>
#include <memory>
#include <vector>
#include <mutex>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <atomic>
#include <glog/logging.h>

#include "morphizen/vaip.hpp"
#include "morphizen/env_config.hpp"
#include "rocm.pb.h"

// Direct stdout debug logging macro (checks env var directly)
#define HIP_DEBUG_LOG(msg) do { \
  const char* debug_env = std::getenv("MORPHIZEN_DEBUG_ROCM"); \
  int debug_level = debug_env ? std::atoi(debug_env) : 0; \
  if (debug_level >= 1) { \
    std::cout << "[HipContext] DEBUG: " << msg << std::endl << std::flush; \
  } \
} while(0)

namespace hip_ep {

// Environment variables for timeout configuration
DEF_ENV_PARAM(MORPHIZEN_GPU_TIMEOUT_MS, "5000")      // Default 5 second timeout
DEF_ENV_PARAM(MORPHIZEN_GPU_WATCHDOG_ENABLED, "1")   // Enable watchdog by default

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
  TimeoutStatus sync_stream_with_timeout(int timeout_ms = 0) {
    ensure_initialized();
    if (!initialized_) {
      return TimeoutStatus::ERROR;
    }
    
    // Use environment variable default if not specified
    if (timeout_ms <= 0) {
      timeout_ms = ENV_PARAM(MORPHIZEN_GPU_TIMEOUT_MS);
    }
    
    LOG(INFO) << "[HipContext] Synchronizing stream with " << timeout_ms << "ms timeout...";
    return WaitStreamWithTimeout(stream_, timeout_ms);
  }

private:
  HipContext() = default;

  ~HipContext() {
    if (initialized_) {
      hipblasLtDestroy(hipblaslt_handle_);
      miopenDestroy(miopen_handle_);
      hipStreamDestroy(stream_);
    }
  }

  void ensure_initialized() {
    std::call_once(init_flag_, [this]() {
      HIP_DEBUG_LOG("ensure_initialized() starting...");
      VLOG(2) << "[HipContext] ensure_initialized() starting...";
      
      // Check if HIP runtime is available
      HIP_DEBUG_LOG("Calling hipGetDeviceCount...");
      int device_count = 0;
      hipError_t hip_err = hipGetDeviceCount(&device_count);
      HIP_DEBUG_LOG("hipGetDeviceCount returned: " << hip_err << ", device_count=" << device_count);
      VLOG(2) << "[HipContext] hipGetDeviceCount returned: " << hip_err 
              << " (" << hipGetErrorString(hip_err) << "), device_count=" << device_count;
      
      if (hip_err != hipSuccess || device_count == 0) {
        HIP_DEBUG_LOG("No AMD GPU detected!");
        LOG(ERROR) << "[HipContext] No AMD GPU detected! HIP error: " 
                   << hipGetErrorString(hip_err);
        initialized_ = false;
        return;
      }
      
      // Get device properties
      HIP_DEBUG_LOG("Calling hipGetDeviceProperties...");
      hipDeviceProp_t props;
      hip_err = hipGetDeviceProperties(&props, 0);
      HIP_DEBUG_LOG("hipGetDeviceProperties returned: " << hip_err);
      VLOG(2) << "[HipContext] hipGetDeviceProperties returned: " << hip_err;
      if (hip_err == hipSuccess) {
        HIP_DEBUG_LOG("GPU name: " << props.name << ", gcnArchName: " << props.gcnArchName);
        LOG(INFO) << "[HipContext] GPU name: " << props.name 
                  << ", gcnArchName: " << props.gcnArchName;
      }
      
      // Create HIP stream
      HIP_DEBUG_LOG("Creating HIP stream...");
      VLOG(2) << "[HipContext] Creating HIP stream...";
      hip_err = hipStreamCreate(&stream_);
      HIP_DEBUG_LOG("hipStreamCreate returned: " << hip_err << ", stream=" << stream_);
      VLOG(2) << "[HipContext] hipStreamCreate returned: " << hip_err 
              << ", stream=" << stream_;
      if (hip_err != hipSuccess) {
        HIP_DEBUG_LOG("hipStreamCreate failed!");
        LOG(ERROR) << "[HipContext] hipStreamCreate failed!";
        initialized_ = false;
        return;
      }
      
      // Create MIOpen handle
      HIP_DEBUG_LOG("Creating MIOpen handle...");
      VLOG(2) << "[HipContext] Creating MIOpen handle...";
      miopenStatus_t miopen_status = miopenCreate(&miopen_handle_);
      HIP_DEBUG_LOG("miopenCreate returned: " << miopen_status << ", handle=" << miopen_handle_);
      VLOG(2) << "[HipContext] miopenCreate returned: " << miopen_status 
              << ", handle=" << miopen_handle_;
      if (miopen_status != miopenStatusSuccess) {
        HIP_DEBUG_LOG("miopenCreate failed!");
        LOG(ERROR) << "[HipContext] miopenCreate failed!";
        hipStreamDestroy(stream_);
        stream_ = nullptr;
        initialized_ = false;
        return;
      }
      
      // Set MIOpen stream
      HIP_DEBUG_LOG("Setting MIOpen stream...");
      VLOG(2) << "[HipContext] Setting MIOpen stream...";
      miopen_status = miopenSetStream(miopen_handle_, stream_);
      HIP_DEBUG_LOG("miopenSetStream returned: " << miopen_status);
      VLOG(2) << "[HipContext] miopenSetStream returned: " << miopen_status;
      
      // Create hipBLASLt handle
      HIP_DEBUG_LOG("Creating hipBLASLt handle...");
      VLOG(2) << "[HipContext] Creating hipBLASLt handle...";
      hipblasStatus_t blaslt_status = hipblasLtCreate(&hipblaslt_handle_);
      HIP_DEBUG_LOG("hipblasLtCreate returned: " << blaslt_status << ", handle=" << hipblaslt_handle_);
      VLOG(2) << "[HipContext] hipblasLtCreate returned: " << blaslt_status 
              << ", handle=" << hipblaslt_handle_;
      if (blaslt_status != HIPBLAS_STATUS_SUCCESS) {
        HIP_DEBUG_LOG("hipblasLtCreate failed!");
        LOG(ERROR) << "[HipContext] hipblasLtCreate failed!";
        miopenDestroy(miopen_handle_);
        miopen_handle_ = nullptr;
        hipStreamDestroy(stream_);
        stream_ = nullptr;
        initialized_ = false;
        return;
      }
      
      initialized_ = true;
      HIP_DEBUG_LOG("HIP context initialized successfully!");
      LOG(INFO) << "[HipContext] HIP context initialized successfully!";
    });
  }

  std::once_flag init_flag_;
  bool initialized_ = false;
  hipStream_t stream_ = nullptr;
  miopenHandle_t miopen_handle_ = nullptr;
  hipblasLtHandle_t hipblaslt_handle_ = nullptr;
};

/**
 * Unified ROCm Custom Op
 * 
 * Handles both Conv (MIOpen) and Gemm (hipBLASLt) operations
 * based on the op_type field in RocmParamProto.
 */
class RocmCustomOp : public vaip_core::CustomOpImp {
public:
  RocmCustomOp(std::shared_ptr<const vaip_core::PassContext> context,
               const std::shared_ptr<vaip_core::MetaDefProto>& meta_def,
               onnxruntime::Model* model);

  virtual ~RocmCustomOp();

private:
  void Compute(const OrtApi* api, OrtKernelContext* context) const override;

  void ExecuteConv(const OrtApi* api, OrtKernelContext* context) const;
  void ExecuteGemm(const OrtApi* api, OrtKernelContext* context) const;

private:
  void LoadCachedWeights();

private:
  rocm::RocmParamProto rocm_proto_;

  // Host-side cached weight/bias data (loaded from pass context)
  std::vector<float> host_weight_;
  std::vector<float> host_bias_;

  // Cached device weights
  mutable float* d_weight_ = nullptr;
  mutable float* d_bias_ = nullptr;
  mutable size_t weight_size_ = 0;
  mutable size_t bias_size_ = 0;
  mutable bool weights_cached_ = false;

  // MIOpen descriptors (lazy initialized)
  mutable miopenTensorDescriptor_t input_desc_ = nullptr;
  mutable miopenTensorDescriptor_t weight_desc_ = nullptr;
  mutable miopenTensorDescriptor_t output_desc_ = nullptr;
  mutable miopenConvolutionDescriptor_t conv_desc_ = nullptr;

  // hipBLASLt descriptors (lazy initialized)
  mutable hipblasLtMatrixLayout_t layout_a_ = nullptr;
  mutable hipblasLtMatrixLayout_t layout_b_ = nullptr;
  mutable hipblasLtMatrixLayout_t layout_c_ = nullptr;
  mutable hipblasLtMatmulDesc_t matmul_desc_ = nullptr;

  // Workspace
  mutable void* workspace_ = nullptr;
  mutable size_t workspace_size_ = 0;
};

} // namespace hip_ep
