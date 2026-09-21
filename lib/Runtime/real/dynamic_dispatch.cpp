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

// DynamicDispatch C API header
// This provides stable C ABI functions that avoid vtable/template issues across DLL boundaries
#include <ops/dd_c_api.h>

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

// Convert HIPDNN_EP_DATATYPE_* to DynamicDispatch dtype string
// NOTE: DynamicDispatch combined_gemm only supports specific dtypes:
//   - Activation/Output: "uint16" (float16)
//   - Weights: "uint8" or "int8"
static const char *hipdnn_datatype_to_dd_string(int64_t data_type) {
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    // DynamicDispatch doesn't support float32 for combined_gemm
    // Return nullptr to trigger an error
    return nullptr;
  case HIPDNN_EP_DATATYPE_HALF:
    return "uint16";  // DynamicDispatch uses uint16 for float16
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return nullptr;  // Not supported by combined_gemm
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

// Note: XRT context initialization is now handled internally by the DD C API
// when creating operators with load_xrt=true

//===----------------------------------------------------------------------===//
// Op-State Management for DynamicDispatch Operators
//===----------------------------------------------------------------------===//
//
// DynamicDispatch operators are opaque C handles with initialization overhead
// (XRT context binding, transaction binary loading, etc.). We cache them
// in RuntimeState's op_state slots to avoid recreating them per inference.
//
// Each DD operator uses the OpStateT<T> CRTP base for lifecycle management.
// We use the C API (dd_c_api.h) to avoid C++ ABI/vtable issues across DLL boundaries.
//===----------------------------------------------------------------------===//

// GEMM operator state - stores opaque C API handle
struct DDGemmState : OpStateT<DDGemmState> {
  dd_gemm_handle_t handle;

  DDGemmState(dd_gemm_handle_t h) : handle(h) {}

  ~DDGemmState() {
    if (handle) {
      dd_combined_gemm_destroy(handle);
    }
  }
};

// Conv2D operator state - stores opaque C API handle
struct DDConvState : OpStateT<DDConvState> {
  dd_conv_handle_t handle;

  DDConvState(dd_conv_handle_t h) : handle(h) {}

  ~DDConvState() {
    if (handle) {
      dd_iconv_destroy(handle);
    }
  }
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
    fprintf(stderr, "  DynamicDispatch combined_gemm only supports:\n");
    fprintf(stderr, "    - Activation/Output: float16 (HIPDNN_EP_DATATYPE_HALF)\n");
    fprintf(stderr, "    - Weights: int8 or uint8\n");
    return HIPDNN_EP_ERR_INVALID_DIMENSION;
  }

#if 0
  // HACK to always create cached operator handle
  DDGemmState *gemm_state = nullptr;
#else
  // Get or create cached operator handle
  DDGemmState *gemm_state = DDGemmState::get_op_state(state, op_state_slot);
#endif

  if (!gemm_state) {
    // First call: create operator instance via C API
    // For combined_gemm, we use: a_dtype="uint16", b_dtype="uint8", c_dtype="uint16"
    const char *weight_dtype = "uint8";

    fprintf(stderr, "[DD] Creating combined_gemm: a=%s, b=%s, c=%s\n",  dtype_str, weight_dtype, dtype_str);
    dd_gemm_handle_t handle = dd_combined_gemm_create(                  dtype_str, weight_dtype, dtype_str, true);  // load_xrt=true
    if (!handle) {
      fprintf(stderr, "[DD] wrap_dd_matmul: dd_combined_gemm_create failed\n");
      return -1;
    }
    fprintf(stderr, "[DD] wrap_dd_matmul: dd_combined_gemm_create: handle= %p\n", (void*) handle);

    // Create state and store in slot
    auto state_ptr = DDGemmState::create(handle);

    gemm_state = state_ptr.get();
    hipdnn_ep_op_state_set(state, op_state_slot, state_ptr.release());

    fprintf(stderr, "[DD] combined_gemm created  successfully, gemm_state handle=%p  handle=%p"     "\n", (void*) gemm_state->handle, (void*) handle);
  } else {
    fprintf(stderr, "[DD] combined_gemm obtained successfully, gemm_state handle=%p"                "\n", (void*) gemm_state->handle);
  }

  // Initialize weights if this is the first call or weights changed
  // For now, we'll initialize weights on every call (DD caches internally)
  dd_tensor_t weight_tensor;
  weight_tensor.data = const_cast<void*>(input_b);
  weight_tensor.shape[0] = static_cast<size_t>(K);
  weight_tensor.shape[1] = static_cast<size_t>(N);
  weight_tensor.shape[2] = 0;
  weight_tensor.shape[3] = 0;
  weight_tensor.ndim = 2;
  weight_tensor.dtype = "uint8";  // Weights are uint8

  fprintf(stderr, "[DD] initializing weights combined_gemm"                                         "\n");
  int ret = dd_combined_gemm_initialize_weights(gemm_state->handle, &weight_tensor);
  if (ret != 0) {
    fprintf(stderr, "wrap_dd_matmul: dd_combined_gemm_initialize_weights failed: %d\n", ret);
    return -1;
  }
  fprintf(stderr, "[DD] wrap_dd_matmul: dd_combined_gemm_initialize_weights: ret= %d\n", ret);

  // Setup input tensor (activation)
  dd_tensor_t input_tensor;
  input_tensor.data = const_cast<void*>(input_a);
  input_tensor.shape[0] = static_cast<size_t>(M);
  input_tensor.shape[1] = static_cast<size_t>(K);
  input_tensor.shape[2] = 0;
  input_tensor.shape[3] = 0;
  input_tensor.ndim = 2;
  input_tensor.dtype = dtype_str;

  // Setup output tensor
  dd_tensor_t output_tensor;
  output_tensor.data = output;
  output_tensor.shape[0] = static_cast<size_t>(M);
  output_tensor.shape[1] = static_cast<size_t>(N);
  output_tensor.shape[2] = 0;
  output_tensor.shape[3] = 0;
  output_tensor.ndim = 2;
  output_tensor.dtype = dtype_str;

  // Execute via C API
  fprintf(stderr, "[DD] executing combined_gemm"                                         "\n");
  ret = dd_combined_gemm_execute(gemm_state->handle, &input_tensor, &output_tensor);
  if (ret != 0) {
    fprintf(stderr, "wrap_dd_matmul: dd_combined_gemm_execute failed: %d\n", ret);
    fprintf(stderr, "  M=%lld, N=%lld, K=%lld, dtype=%s\n",         (long long)M, (long long)N, (long long)K, dtype_str);
    return -1;
  }
  fprintf(stderr, "[DD] wrap_dd_matmul: execute: ret= %d\n", ret);

  // TODO: Handle bias, alpha, beta, transA, transB parameters
  (void)bias; (void)alpha; (void)beta; (void)transA; (void)transB;

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

#if 1
  // HACK to always create cached operator handle
  DDConvState *conv_state = nullptr;
#else
  // Get or create cached operator handle
  DDConvState *conv_state = DDConvState::get_op_state(state, op_state_slot);
#endif

  if (!conv_state) {
    // First call: create operator instance via C API
    const char *weight_dtype = "uint8";

    fprintf(stderr, "[DD] Creating iconv: a=%s, b=%s, c=%s\n",
            dtype_str, weight_dtype, dtype_str);

    dd_conv_handle_t handle = dd_iconv_create(
        dtype_str, weight_dtype, dtype_str, true);  // load_xrt=true

    if (!handle) {
      fprintf(stderr, "wrap_dd_conv2d: dd_iconv_create failed\n");
      return -1;
    }

    // Create state and store in slot
    auto state_ptr = DDConvState::create(handle);
    conv_state = state_ptr.get();
    hipdnn_ep_op_state_set(state, op_state_slot, state_ptr.release());

    fprintf(stderr, "[DD] iconv created successfully, handle=%p\n", handle);
  }

  // Initialize weights
  dd_tensor_t weight_tensor;
  weight_tensor.data = const_cast<void*>(weights);
  weight_tensor.shape[0] = static_cast<size_t>(k);
  weight_tensor.shape[1] = static_cast<size_t>(c);
  weight_tensor.shape[2] = static_cast<size_t>(kernel_h);
  weight_tensor.shape[3] = static_cast<size_t>(kernel_w);
  weight_tensor.ndim = 4;
  weight_tensor.dtype = "uint8";

  int ret = dd_iconv_initialize_weights(conv_state->handle, &weight_tensor);
  if (ret != 0) {
    fprintf(stderr, "wrap_dd_conv2d: dd_iconv_initialize_weights failed: %d\n", ret);
    return -1;
  }

  // Setup input tensor (NCHW format)
  dd_tensor_t input_tensor;
  input_tensor.data = const_cast<void*>(input);
  input_tensor.shape[0] = static_cast<size_t>(n);
  input_tensor.shape[1] = static_cast<size_t>(c);
  input_tensor.shape[2] = static_cast<size_t>(h);
  input_tensor.shape[3] = static_cast<size_t>(w);
  input_tensor.ndim = 4;
  input_tensor.dtype = dtype_str;

  // Setup output tensor
  dd_tensor_t output_tensor;
  output_tensor.data = output;
  output_tensor.shape[0] = static_cast<size_t>(n);
  output_tensor.shape[1] = static_cast<size_t>(k);
  output_tensor.shape[2] = static_cast<size_t>(out_h);
  output_tensor.shape[3] = static_cast<size_t>(out_w);
  output_tensor.ndim = 4;
  output_tensor.dtype = dtype_str;

  // Execute via C API
  ret = dd_iconv_execute(conv_state->handle, &input_tensor, &output_tensor);
  if (ret != 0) {
    fprintf(stderr, "wrap_dd_conv2d: dd_iconv_execute failed: %d\n", ret);
    return -1;
  }

  // TODO: Handle bias and other conv parameters in DD C API
  (void)bias; (void)stride_h; (void)stride_w;
  (void)pad_top; (void)pad_left; (void)pad_bottom; (void)pad_right;
  (void)dilation_h; (void)dilation_w; (void)group;

  return 0;
}

//===----------------------------------------------------------------------===//
// XRT Context Accessors (Optional - for future use)
//===----------------------------------------------------------------------===//
//
// XRT context is now managed internally by DynamicDispatch C API.
// These functions are kept for potential future extensions.
//===----------------------------------------------------------------------===//

extern "C" {

void *hipdnn_ep_state_get_xrt_device(RuntimeState *state) {
  (void)state;
  return nullptr;  // XRT device managed by DD C API
}

void *hipdnn_ep_state_get_xrt_context(RuntimeState *state) {
  (void)state;
  return nullptr;  // XRT context managed by DD C API
}

} // extern "C"
