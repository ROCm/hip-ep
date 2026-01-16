// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>
#include <miopen/miopen.h>
#include <hipblaslt/hipblaslt.h>
#include <memory>
#include <vector>
#include <mutex>
#include <glog/logging.h>

#include "morphizen/vaip.hpp"
#include "rocm.pb.h"

namespace rocm_ep {

/**
 * Shared HIP Context Singleton
 * 
 * Provides a single shared HIP stream for all ROCm operations.
 * This enables implicit fusion between Conv and Gemm operations.
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
      VLOG(2) << "[HipContext] ensure_initialized() starting...";
      
      // Check if HIP runtime is available
      int device_count = 0;
      hipError_t hip_err = hipGetDeviceCount(&device_count);
      VLOG(2) << "[HipContext] hipGetDeviceCount returned: " << hip_err 
              << " (" << hipGetErrorString(hip_err) << "), device_count=" << device_count;
      
      if (hip_err != hipSuccess || device_count == 0) {
        LOG(ERROR) << "[HipContext] No AMD GPU detected! HIP error: " 
                   << hipGetErrorString(hip_err);
        initialized_ = false;
        return;
      }
      
      // Get device properties
      hipDeviceProp_t props;
      hip_err = hipGetDeviceProperties(&props, 0);
      VLOG(2) << "[HipContext] hipGetDeviceProperties returned: " << hip_err;
      if (hip_err == hipSuccess) {
        LOG(INFO) << "[HipContext] GPU name: " << props.name 
                  << ", gcnArchName: " << props.gcnArchName;
      }
      
      // Create HIP stream
      VLOG(2) << "[HipContext] Creating HIP stream...";
      hip_err = hipStreamCreate(&stream_);
      VLOG(2) << "[HipContext] hipStreamCreate returned: " << hip_err 
              << ", stream=" << stream_;
      if (hip_err != hipSuccess) {
        LOG(ERROR) << "[HipContext] hipStreamCreate failed!";
        initialized_ = false;
        return;
      }
      
      // Create MIOpen handle
      VLOG(2) << "[HipContext] Creating MIOpen handle...";
      miopenStatus_t miopen_status = miopenCreate(&miopen_handle_);
      VLOG(2) << "[HipContext] miopenCreate returned: " << miopen_status 
              << ", handle=" << miopen_handle_;
      if (miopen_status != miopenStatusSuccess) {
        LOG(ERROR) << "[HipContext] miopenCreate failed!";
        hipStreamDestroy(stream_);
        stream_ = nullptr;
        initialized_ = false;
        return;
      }
      
      // Set MIOpen stream
      VLOG(2) << "[HipContext] Setting MIOpen stream...";
      miopen_status = miopenSetStream(miopen_handle_, stream_);
      VLOG(2) << "[HipContext] miopenSetStream returned: " << miopen_status;
      
      // Create hipBLASLt handle
      VLOG(2) << "[HipContext] Creating hipBLASLt handle...";
      hipblasStatus_t blaslt_status = hipblasLtCreate(&hipblaslt_handle_);
      VLOG(2) << "[HipContext] hipblasLtCreate returned: " << blaslt_status 
              << ", handle=" << hipblaslt_handle_;
      if (blaslt_status != HIPBLAS_STATUS_SUCCESS) {
        LOG(ERROR) << "[HipContext] hipblasLtCreate failed!";
        miopenDestroy(miopen_handle_);
        miopen_handle_ = nullptr;
        hipStreamDestroy(stream_);
        stream_ = nullptr;
        initialized_ = false;
        return;
      }
      
      initialized_ = true;
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
  rocm::RocmParamProto rocm_proto_;

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

} // namespace rocm_ep
