/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "runtime_types.h"
#include "hip_custom_kernels.h"

#include <cstdio>

#define MIOPEN_CHECK(cmd)                                                      \
  do {                                                                         \
    miopenStatus_t status = (cmd);                                             \
    if (status != miopenStatusSuccess) {                                       \
      RUNTIME_DEBUG_LOG("[REAL] MIOpen error %d at %s:%d\n", status, __FILE__, \
                        __LINE__);                                             \
      return -1;                                                               \
    }                                                                          \
  } while (0)

// Explicit mapping from backend-independent HIPDNN_EP_DATATYPE_* enum to
// MIOpen-specific miopenDataType_t. No static_cast -- our enum values are
// independent of any library.
static miopenDataType_t hipdnn_ep_to_miopen_type(int64_t data_type) {
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return miopenFloat;
  case HIPDNN_EP_DATATYPE_HALF:
    return miopenHalf;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return miopenBFloat16;
  default:
    fprintf(stderr, "[REAL] unsupported data_type %lld for MIOpen\n",
            (long long)data_type);
    return miopenFloat;
  }
}

// Explicit mapping from backend-independent HIPDNN_EP_TENSOR_OP_* enum to
// MIOpen-specific miopenTensorOp_t.
static miopenTensorOp_t hipdnn_ep_to_miopen_op(int64_t tensor_op) {
  switch (tensor_op) {
  case HIPDNN_EP_TENSOR_OP_MUL:
    return miopenTensorOpMul;
  case HIPDNN_EP_TENSOR_OP_ADD:
    return miopenTensorOpAdd;
  case HIPDNN_EP_TENSOR_OP_MIN:
    return miopenTensorOpMin;
  case HIPDNN_EP_TENSOR_OP_MAX:
    return miopenTensorOpMax;
  default:
    fprintf(stderr, "[REAL] unsupported tensor_op %lld for MIOpen\n",
            (long long)tensor_op);
    return miopenTensorOpMul;
  }
}

// =============================================================================
// Generic Element-wise Tensor Operation via MIOpen
// =============================================================================
//
// Uses miopenOpTensor to compute:
//   output = alpha1 * op(lhs, alpha2 * rhs) + beta * output
// With alpha1=1, alpha2=1, beta=0 this gives: output = op(lhs, rhs)
//
// The tensor is represented as a flat 1D tensor [1, 1, 1, num_elements]
// to satisfy miopenSet4dTensorDescriptor's 4D requirement.
// =============================================================================

int wrap_miopenOpTensor(RuntimeState *state, void *lhs, void *rhs, void *output,
                        int64_t num_elements, int64_t data_type,
                        int64_t tensor_op) {
  if (!state || !lhs || !rhs || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_miopenOpTensor: null argument\n");
    return -1;
  }

  const char *type_name = hipdnn_ep_datatype_name(data_type);
  const char *op_name = hipdnn_ep_tensor_op_name(tensor_op);
  int64_t elem_size = hipdnn_ep_datatype_size(data_type);
  RUNTIME_DEBUG_LOG("[REAL] wrap_miopenOpTensor: op=%s, num_elements=%lld, "
                    "data_type=%s(%lld), element_size=%lld bytes, "
                    "total_size=%lld bytes\n",
                    op_name, (long long)num_elements, type_name,
                    (long long)data_type, (long long)elem_size,
                    (long long)(num_elements * elem_size));

  miopenHandle_t handle =
      static_cast<miopenHandle_t>(hipdnn_ep_state_get_miopen_handle(state));
  if (!handle) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_miopenOpTensor: null MIOpen handle\n");
    return -1;
  }

  miopenDataType_t miopen_type = hipdnn_ep_to_miopen_type(data_type);
  miopenTensorOp_t miopen_op = hipdnn_ep_to_miopen_op(tensor_op);

  miopenTensorDescriptor_t aDesc, bDesc, cDesc;
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&aDesc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&bDesc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&cDesc));

  int n = static_cast<int>(num_elements);
  RUNTIME_DEBUG_LOG("[REAL] wrap_miopenOpTensor: creating tensor descriptors "
                    "[1,1,1,%d] with type %s\n",
                    n, type_name);

  MIOPEN_CHECK(miopenSet4dTensorDescriptor(aDesc, miopen_type, 1, 1, 1, n));
  MIOPEN_CHECK(miopenSet4dTensorDescriptor(bDesc, miopen_type, 1, 1, 1, n));
  MIOPEN_CHECK(miopenSet4dTensorDescriptor(cDesc, miopen_type, 1, 1, 1, n));

  float alpha1 = 1.0f, alpha2 = 1.0f, beta = 0.0f;
  RUNTIME_DEBUG_LOG("[REAL] wrap_miopenOpTensor: calling miopenOpTensor"
                    "(op=%s, alpha1=%.1f, alpha2=%.1f, beta=%.1f)\n",
                    op_name, alpha1, alpha2, beta);

  MIOPEN_CHECK(miopenOpTensor(handle, miopen_op, &alpha1, aDesc, lhs, &alpha2,
                              bDesc, rhs, &beta, cDesc, output));

  RUNTIME_DEBUG_LOG("[REAL] wrap_miopenOpTensor: completed successfully\n");

  miopenDestroyTensorDescriptor(aDesc);
  miopenDestroyTensorDescriptor(bDesc);
  miopenDestroyTensorDescriptor(cDesc);
  return 0;
}

// =============================================================================
// Element-wise Subtraction via Custom HIP Kernel
// =============================================================================
//
// For types unsupported by MIOpen (e.g. int64), dispatches to
// hip_elementwise_sub from the custom kernels library. The caller passes
// element_size_bytes; we map it to the corresponding hip_dtype_t.
// =============================================================================

int wrap_elementwise_sub(RuntimeState *state, void *lhs, void *rhs,
                         void *output, int64_t num_elements,
                         int64_t element_size_bytes) {
  if (!state || !lhs || !rhs || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_elementwise_sub: null argument\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  int hip_dtype;
  switch (element_size_bytes) {
  case 8:
    hip_dtype = HIP_DTYPE_INT64;
    break;
  default:
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_elementwise_sub: unsupported element_size=%lld, "
        "only int64 (8 bytes) is currently supported via custom kernel\n",
        (long long)element_size_bytes);
    return -1;
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_elementwise_sub: num_elements=%lld, "
      "element_size=%lld, dtype=%d -> calling hip_elementwise_sub\n",
      (long long)num_elements, (long long)element_size_bytes, hip_dtype);

  return hip_elementwise_sub(stream, lhs, rhs, output, num_elements,
                              hip_dtype);
}
