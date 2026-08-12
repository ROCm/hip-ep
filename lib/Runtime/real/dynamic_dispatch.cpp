/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "dynamic_dispatch.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_state.h"
#include "../runtime_state_internal.h"
#include "runtime_types.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// DynamicDispatch headers
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
// Each DD operator instance is stored as:
//   OpState { void *payload = std::unique_ptr<OpInterface>* }
//
// The cleanup callback is set to delete the unique_ptr when the session ends.
//===----------------------------------------------------------------------===//

// Base cleanup function for DynamicDispatch operators
template <typename T>
static void dd_op_cleanup(void *payload) {
  if (payload) {
    auto *op_ptr = static_cast<std::unique_ptr<T> *>(payload);
    delete op_ptr;
  }
}

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
    return HIPDNN_STATUS_BAD_PARAM;
  }

  // Convert data type
  const char *dtype_str = hipdnn_datatype_to_dd_string(data_type);
  if (!dtype_str) {
    fprintf(stderr, "wrap_dd_matmul: unsupported data type %lld\n",
            (long long)data_type);
    return HIPDNN_STATUS_BAD_PARAM;
  }

  // Get XRT context
  auto xrt_ctx = get_xrt_context(state);
  if (!xrt_ctx) {
    fprintf(stderr, "wrap_dd_matmul: XRT context not initialized\n");
    fprintf(stderr, "  (XRT context management not yet implemented in RuntimeState)\n");
    return HIPDNN_STATUS_INTERNAL_ERROR;
  }

  // Get or create the DD operator instance from op_state slot
  using GemmOp = ryzenai::combined_gemm<uint16_t, uint8_t, uint16_t>;
  OpState *op_slot = hipdnn_ep_get_op_state(state, op_state_slot);

  std::unique_ptr<GemmOp> *gemm_op_ptr = nullptr;

  if (!op_slot || !op_slot->payload) {
    // First call: create operator instance
    try {
      auto gemm_op = std::make_unique<GemmOp>(dtype_str, dtype_str, dtype_str,
                                               true /* load_xrt */);
      gemm_op->register_xrt_context(xrt_ctx);

      // Store in op_state slot
      gemm_op_ptr = new std::unique_ptr<GemmOp>(std::move(gemm_op));

      if (!op_slot) {
        op_slot = hipdnn_ep_alloc_op_state(state, op_state_slot);
        if (!op_slot) {
          delete gemm_op_ptr;
          fprintf(stderr, "wrap_dd_matmul: failed to allocate op_state slot\n");
          return HIPDNN_STATUS_ALLOC_FAILED;
        }
      }

      op_slot->payload = gemm_op_ptr;
      op_slot->cleanup = dd_op_cleanup<GemmOp>;

    } catch (const std::exception &e) {
      fprintf(stderr, "wrap_dd_matmul: failed to create operator: %s\n",
              e.what());
      return HIPDNN_STATUS_EXECUTION_FAILED;
    }
  } else {
    // Reuse cached operator
    gemm_op_ptr = static_cast<std::unique_ptr<GemmOp> *>(op_slot->payload);
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
    (*gemm_op_ptr)->execute(inputs, outputs);
  } catch (const std::exception &e) {
    fprintf(stderr, "wrap_dd_matmul: execution failed: %s\n", e.what());
    fprintf(stderr, "  M=%lld, N=%lld, K=%lld, dtype=%s\n", (long long)M,
            (long long)N, (long long)K, dtype_str);
    return HIPDNN_STATUS_EXECUTION_FAILED;
  }

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
    return HIPDNN_STATUS_BAD_PARAM;
  }

  // Convert data type
  const char *dtype_str = hipdnn_datatype_to_dd_string(data_type);
  if (!dtype_str) {
    fprintf(stderr, "wrap_dd_conv2d: unsupported data type %lld\n",
            (long long)data_type);
    return HIPDNN_STATUS_BAD_PARAM;
  }

  // Get XRT context
  auto xrt_ctx = get_xrt_context(state);
  if (!xrt_ctx) {
    fprintf(stderr, "wrap_dd_conv2d: XRT context not initialized\n");
    fprintf(stderr, "  (XRT context management not yet implemented in RuntimeState)\n");
    return HIPDNN_STATUS_INTERNAL_ERROR;
  }

  // Get or create the DD operator instance
  using ConvOp = ryzenai::iconv<uint16_t, uint8_t, uint16_t>;
  OpState *op_slot = hipdnn_ep_get_op_state(state, op_state_slot);

  std::unique_ptr<ConvOp> *conv_op_ptr = nullptr;

  if (!op_slot || !op_slot->payload) {
    // First call: create operator instance
    try {
      auto conv_op = std::make_unique<ConvOp>(dtype_str, dtype_str, dtype_str,
                                               true /* load_xrt */);
      conv_op->register_xrt_context(xrt_ctx);

      // Store in op_state slot
      conv_op_ptr = new std::unique_ptr<ConvOp>(std::move(conv_op));

      if (!op_slot) {
        op_slot = hipdnn_ep_alloc_op_state(state, op_state_slot);
        if (!op_slot) {
          delete conv_op_ptr;
          fprintf(stderr, "wrap_dd_conv2d: failed to allocate op_state slot\n");
          return HIPDNN_STATUS_ALLOC_FAILED;
        }
      }

      op_slot->payload = conv_op_ptr;
      op_slot->cleanup = dd_op_cleanup<ConvOp>;

    } catch (const std::exception &e) {
      fprintf(stderr, "wrap_dd_conv2d: failed to create operator: %s\n",
              e.what());
      return HIPDNN_STATUS_EXECUTION_FAILED;
    }
  } else {
    // Reuse cached operator
    conv_op_ptr = static_cast<std::unique_ptr<ConvOp> *>(op_slot->payload);
  }

  // Prepare input/output tensors
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;

  // Input tensor (N x C x H x W)
  Tensor tensor_in;
  tensor_in.data = const_cast<void *>(input);
  tensor_in.shape = {static_cast<size_t>(n), static_cast<size_t>(c),
                     static_cast<size_t>(h), static_cast<size_t>(w)};
  tensor_in.dtype = dtype_str;
  inputs.push_back(tensor_in);

  // Weights tensor (K x C/group x kH x kW)
  Tensor tensor_w;
  tensor_w.data = const_cast<void *>(weights);
  tensor_w.shape = {static_cast<size_t>(k), static_cast<size_t>(c / group),
                    static_cast<size_t>(kernel_h),
                    static_cast<size_t>(kernel_w)};
  tensor_w.dtype = dtype_str;
  inputs.push_back(tensor_w);

  // Bias tensor (if provided)
  if (bias) {
    Tensor tensor_bias;
    tensor_bias.data = const_cast<void *>(bias);
    tensor_bias.shape = {static_cast<size_t>(k)};
    tensor_bias.dtype = dtype_str;
    inputs.push_back(tensor_bias);
  }

  // Output tensor (N x K x outH x outW)
  Tensor tensor_out;
  tensor_out.data = output;
  tensor_out.shape = {static_cast<size_t>(n), static_cast<size_t>(k),
                      static_cast<size_t>(out_h), static_cast<size_t>(out_w)};
  tensor_out.dtype = dtype_str;
  outputs.push_back(tensor_out);

  // Execute the operator
  try {
    (*conv_op_ptr)->execute(inputs, outputs);
  } catch (const std::exception &e) {
    fprintf(stderr, "wrap_dd_conv2d: execution failed: %s\n", e.what());
    fprintf(stderr, "  N=%lld, C=%lld, H=%lld, W=%lld, K=%lld\n", (long long)n,
            (long long)c, (long long)h, (long long)w, (long long)k);
    return HIPDNN_STATUS_EXECUTION_FAILED;
  }

  return 0;
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
