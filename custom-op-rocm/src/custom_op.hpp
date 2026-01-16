// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>
#include <miopen/miopen.h>
#include <hipblaslt/hipblaslt.h>
#include <memory>
#include <vector>
#include <mutex>

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
      hipStreamCreate(&stream_);
      miopenCreate(&miopen_handle_);
      miopenSetStream(miopen_handle_, stream_);
      hipblasLtCreate(&hipblaslt_handle_);
      initialized_ = true;
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
