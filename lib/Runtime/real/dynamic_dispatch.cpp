/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "dynamic_dispatch.h"
#include "../hipdnn_ep_runtime.h"
#include "../runtime_state_internal.h"
#include "../op_state.h"
#include "../hipdnn_ep_errors.h"

#include <any>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

// DynamicDispatch headers
// Note: Include paths are configured in CMakeLists.txt to work with both
// install layout (ryzenai/dynamic_dispatch/...) and source layout
#include <ops/dmacompiler/combined_gemm/combined_gemm.hpp>
#include <ops/dmacompiler/iconv/iconv.hpp>
#include <ops/op_interface.hpp>
#include <xrt_context/xrt_context.hpp>

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

// Convert HIPDNN_EP_DATATYPE_* to DynamicDispatch dtype string
static const char *hipdnn_datatype_to_dd_string(int64_t data_type) {
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return "float32";
  case HIPDNN_EP_DATATYPE_HALF:
    return "float16";
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return "bfloat16";
  case HIPDNN_EP_DATATYPE_INT8:
    return "int8";
  case HIPDNN_EP_DATATYPE_UINT8:
    return "uint8";
  case HIPDNN_EP_DATATYPE_INT16:
    return "int16";
  case HIPDNN_EP_DATATYPE_INT32:
    return "int32";
  case HIPDNN_EP_DATATYPE_INT64:
    return "int64";
  default:
    return nullptr;
  }
}

// Get or create XRT context from RuntimeState
// Lazily initialized on first DynamicDispatch operation call
static std::shared_ptr<ryzenai::dynamic_dispatch::xrt_context>
get_xrt_context(RuntimeState *state) {
  if (!state) {
    fprintf(stderr, "get_xrt_context: null RuntimeState\n");
    return nullptr;
  }

  // If XRT context already exists, return it
  if (state->xrt_context) {
    auto *ctx_ptr = static_cast<
        std::shared_ptr<ryzenai::dynamic_dispatch::xrt_context> *>(
        state->xrt_context);
    return *ctx_ptr;
  }

  // Lazy initialization: create XRT context on first use
  // XRT context requires an XRT device. For now, we'll use the default device.
  // In a multi-device setup, this could be configured via environment variable
  // or passed through RuntimeState initialization.
  try {
    auto ctx = std::make_shared<ryzenai::dynamic_dispatch::xrt_context>();

    // Store the shared_ptr in RuntimeState for reuse
    // Allocate on heap so it persists beyond this function
    auto *ctx_ptr =
        new std::shared_ptr<ryzenai::dynamic_dispatch::xrt_context>(ctx);
    state->xrt_context = static_cast<void *>(ctx_ptr);

    fprintf(stderr,
            "[DynamicDispatch] Initialized XRT context for NPU/IPU backend\n");
    return ctx;
  } catch (const std::exception &e) {
    fprintf(stderr, "get_xrt_context: failed to create XRT context: %s\n",
            e.what());
    return nullptr;
  }
}

//===----------------------------------------------------------------------===//
// Op-State Management for DynamicDispatch Operators
//===----------------------------------------------------------------------===//
//
// DynamicDispatch operators are C++ objects with initialization overhead
// (XRT context binding, transaction binary loading, etc.). We cache them
// in RuntimeState's op_state slots to avoid recreating them per inference.
//
// Each DD operator uses the OpStateT<T> CRTP base for lifecycle management.
//===----------------------------------------------------------------------===//

// GEMM operator state
struct DDGemmState : OpStateT<DDGemmState> {
  using GemmOp = ryzenai::combined_gemm<uint16_t, uint8_t, uint16_t>;
  std::unique_ptr<GemmOp> op;

  DDGemmState(std::unique_ptr<GemmOp> &&gemm_op) : op(std::move(gemm_op)) {}
};

// Conv2D operator state
struct DDConvState : OpStateT<DDConvState> {
  using ConvOp = ryzenai::iconv<uint16_t, uint8_t, uint16_t>;
  std::unique_ptr<ConvOp> op;

  DDConvState(std::unique_ptr<ConvOp> &&conv_op) : op(std::move(conv_op)) {}
};

//===----------------------------------------------------------------------===//
// wrap_dd_matmul - GEMM/MatMul via DynamicDispatch
//===----------------------------------------------------------------------===//

int wrap_dd_matmul(RuntimeState *state, int32_t op_state_slot,
                   const void *input_a, const void *input_b, const void *bias,
                   void *output, int64_t M, int64_t N, int64_t K, double alpha,
                   double beta, int64_t transA, int64_t transB,
                   int64_t data_type) {
  if (!state) {
    fprintf(stderr, "wrap_dd_matmul: null RuntimeState\n");
    return HIPDNN_EP_ERR_NULL_POINTER;
  }

  // Convert data type
  const char *dtype_str = hipdnn_datatype_to_dd_string(data_type);
  if (!dtype_str) {
    fprintf(stderr, "wrap_dd_matmul: unsupported data type %lld\n",
            (long long)data_type);
    return HIPDNN_EP_ERR_INVALID_DIMENSION;
  }

  // Get XRT context
  auto xrt_ctx = get_xrt_context(state);
  if (!xrt_ctx) {
    fprintf(stderr, "wrap_dd_matmul: XRT context not initialized\n");
    fprintf(stderr, "  (XRT context management not yet implemented in RuntimeState)\n");
    return -1;
  }

  DDGemmState *gemm_state = DDGemmState::get_op_state(state, op_state_slot);

  if (!gemm_state) {
    // First call: create operator instance
    try {
      // Create DynamicDispatch combined_gemm operator
      // Constructor: combined_gemm(a_dtype, b_dtype, c_dtype, load_xrt, attr={})
      std::map<std::string, std::any> attr;
      auto gemm_op = std::make_unique<DDGemmState::GemmOp>(
          dtype_str, dtype_str, dtype_str, true, attr);

      // Create state and store in slot
      auto state_ptr = DDGemmState::create(std::move(gemm_op));
      gemm_state = state_ptr.get();
      hipdnn_ep_op_state_set(state, op_state_slot, state_ptr.release());

    } catch (const std::exception &e) {
      fprintf(stderr, "wrap_dd_matmul: failed to create operator: %s\n",
                  e.what());
      return -1;
    }
  }

  // Prepare input/output tensors
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;

  // Input A tensor
  Tensor tensor_a;
  tensor_a.data = const_cast<void *>(input_a);
  tensor_a.shape = {static_cast<size_t>(M), static_cast<size_t>(K)};
  tensor_a.dtype = dtype_str;
  inputs.push_back(tensor_a);

  // Input B tensor (weights)
  Tensor tensor_b;
  tensor_b.data = const_cast<void *>(input_b);
  tensor_b.shape = {static_cast<size_t>(K), static_cast<size_t>(N)};
  tensor_b.dtype = dtype_str;
  inputs.push_back(tensor_b);

  // Bias tensor (if provided)
  if (bias) {
    Tensor tensor_bias;
    tensor_bias.data = const_cast<void *>(bias);
    tensor_bias.shape = {static_cast<size_t>(N)};
    tensor_bias.dtype = dtype_str;
    inputs.push_back(tensor_bias);
  }

  // Output tensor
  Tensor tensor_out;
  tensor_out.data = output;
  tensor_out.shape = {static_cast<size_t>(M), static_cast<size_t>(N)};
  tensor_out.dtype = dtype_str;
  outputs.push_back(tensor_out);

  // Execute the operator
  try {
    gemm_state->op->execute(inputs, outputs);
  } catch (const std::exception &e) {
    fprintf(stderr, "wrap_dd_matmul: execution failed: %s\n", e.what());
    fprintf(stderr, "  M=%lld, N=%lld, K=%lld, dtype=%s\n", (long long)M,
            (long long)N, (long long)K, dtype_str);
    return -1;
  }

  (void)alpha; (void)beta; (void)transA; (void)transB;  // TODO: Handle these parameters
  return 0;
}

//===----------------------------------------------------------------------===//
// wrap_dd_conv2d - Convolution via DynamicDispatch
//===----------------------------------------------------------------------===//

int wrap_dd_conv2d(RuntimeState *state, int32_t op_state_slot,
                   const void *input, int64_t n, int64_t c, int64_t h,
                   int64_t w, const void *weights, int64_t k, const void *bias,
                   void *output, int64_t out_h, int64_t out_w,
                   int64_t kernel_h, int64_t kernel_w, int64_t stride_h,
                   int64_t stride_w, int64_t pad_top, int64_t pad_left,
                   int64_t pad_bottom, int64_t pad_right, int64_t dilation_h,
                   int64_t dilation_w, int64_t group, int64_t data_type) {
  if (!state) {
    fprintf(stderr, "wrap_dd_conv2d: null RuntimeState\n");
    return HIPDNN_EP_ERR_NULL_POINTER;
  }

  // Convert data type
  const char *dtype_str = hipdnn_datatype_to_dd_string(data_type);
  if (!dtype_str) {
    fprintf(stderr, "wrap_dd_conv2d: unsupported data type %lld\n",
            (long long)data_type);
    return HIPDNN_EP_ERR_INVALID_DIMENSION;
  }

  // Get XRT context
  auto xrt_ctx = get_xrt_context(state);
  if (!xrt_ctx) {
    fprintf(stderr, "wrap_dd_conv2d: XRT context not initialized\n");
    fprintf(stderr, "  (XRT context management not yet implemented in RuntimeState)\n");
    return -1;
  }

  DDConvState *conv_state = DDConvState::get_op_state(state, op_state_slot);

  if (!conv_state) {
    // First call: create operator instance
    try {
      // Create DynamicDispatch iconv operator
      // Constructor: iconv(a_dtype, b_dtype, c_dtype, load_xrt, attr)
      std::map<std::string, std::any> attr;
      auto conv_op = std::make_unique<DDConvState::ConvOp>(
              dtype_str, dtype_str, dtype_str, true, attr);

      // Create state and store in slot
      auto state_ptr = DDConvState::create(std::move(conv_op));
      conv_state = state_ptr.get();
      hipdnn_ep_op_state_set(state, op_state_slot, state_ptr.release());

    } catch (const std::exception &e) {
      fprintf(stderr, "wrap_dd_conv2d: failed to create operator: %s\n",
                  e.what());
      return -1;
    }
  }

  // TODO: Implement tensor preparation and execution for Conv2D
  // This requires understanding the DynamicDispatch iconv tensor format
  (void)conv_state;
  (void)input; (void)n; (void)c; (void)h; (void)w;
  (void)weights; (void)k; (void)bias; (void)output;
  (void)out_h; (void)out_w; (void)kernel_h; (void)kernel_w;
  (void)stride_h; (void)stride_w; (void)pad_top; (void)pad_left;
  (void)pad_bottom; (void)pad_right; (void)dilation_h; (void)dilation_w;
  (void)group;

  fprintf(stderr, "wrap_dd_conv2d: tensor preparation not yet implemented\n");
  return -1;
}

//===----------------------------------------------------------------------===//
// XRT Context Accessors
//===----------------------------------------------------------------------===//

extern "C" {

// Get XRT device from RuntimeState
// Currently returns nullptr as device selection is handled by xrt_context
void *hipdnn_ep_state_get_xrt_device(RuntimeState *state) {
  (void)state;
  // XRT device selection is handled internally by xrt_context
  // In a multi-device setup, this could return the selected device ID
  return nullptr;
}

// Get XRT context from RuntimeState
// Returns the cached context or creates it lazily on first call
void *hipdnn_ep_state_get_xrt_context(RuntimeState *state) {
  if (!state) {
    return nullptr;
  }

  // Lazy initialization via get_xrt_context helper
  auto ctx = get_xrt_context(state);
  return ctx ? ctx.get() : nullptr;
}

} // extern "C"
