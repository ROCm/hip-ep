/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "cache_utils.h"
#include "error_check_macros.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>
#include <functional>
#include <unordered_map>

// Explicit mapping from backend-independent HIPDNN_EP_DATATYPE_* enum to
// MIOpen-specific miopenDataType_t. No static_cast -- our enum values are
// independent of any library.
static miopenDataType_t hipdnn_ep_to_miopen_type(int64_t data_type, bool *ok) {
  *ok = true;
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
    *ok = false;
    return miopenFloat;
  }
}

// Explicit mapping from backend-independent HIPDNN_EP_TENSOR_OP_* enum to
// MIOpen-specific miopenTensorOp_t.
static miopenTensorOp_t hipdnn_ep_to_miopen_op(int64_t tensor_op, bool *ok) {
  *ok = true;
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
    *ok = false;
    return miopenTensorOpMul;
  }
}

//===----------------------------------------------------------------------===//
// MIOpen OpTensor descriptor cache
//===----------------------------------------------------------------------===//
//
// Three MIOpen tensor descriptors (lhs, rhs, output) are created once per
// unique (lhs_shape, rhs_shape, out_shape, data_type) combination and reused
// for the process lifetime.  Avoids repeated miopenCreate/Set/Destroy on
// every element-wise inference call.

struct OpTensorCacheKey {
  int64_t lhs_n, lhs_c, lhs_h, lhs_w; // lhs 4D shape
  int64_t rhs_n, rhs_c, rhs_h, rhs_w; // rhs 4D shape
  int64_t out_n, out_c, out_h, out_w; // output 4D shape
  int64_t data_type;                  // HIPDNN_EP_DATATYPE_* enum value
  bool operator==(const OpTensorCacheKey &o) const {
    return lhs_n == o.lhs_n && lhs_c == o.lhs_c && lhs_h == o.lhs_h &&
           lhs_w == o.lhs_w && rhs_n == o.rhs_n && rhs_c == o.rhs_c &&
           rhs_h == o.rhs_h && rhs_w == o.rhs_w && out_n == o.out_n &&
           out_c == o.out_c && out_h == o.out_h && out_w == o.out_w &&
           data_type == o.data_type;
  }
};

struct OpTensorCacheKeyHash {
  size_t operator()(const OpTensorCacheKey &k) const {
    size_t h = 0;
    hash_combine_val(h, k.lhs_n);
    hash_combine_val(h, k.lhs_c);
    hash_combine_val(h, k.lhs_h);
    hash_combine_val(h, k.lhs_w);
    hash_combine_val(h, k.rhs_n);
    hash_combine_val(h, k.rhs_c);
    hash_combine_val(h, k.rhs_h);
    hash_combine_val(h, k.rhs_w);
    hash_combine_val(h, k.out_n);
    hash_combine_val(h, k.out_c);
    hash_combine_val(h, k.out_h);
    hash_combine_val(h, k.out_w);
    hash_combine_val(h, k.data_type);
    return h;
  }
};

/// Cached MIOpen tensor descriptors for a single OpTensor shape.
/// Ownership: descriptors are created in queryOrCreateOpTensor() and live
/// for the process lifetime (never destroyed individually).
struct OpTensorCacheEntry {
  miopenTensorDescriptor_t aDesc, bDesc, cDesc; // lhs, rhs, output
};

static std::unordered_map<OpTensorCacheKey, OpTensorCacheEntry,
                          OpTensorCacheKeyHash>
    g_optensor_cache;

/// Look up or create cached MIOpen tensor descriptors for an OpTensor shape.
/// Returns nullptr on any MIOpen API failure (partially created descriptors
/// are cleaned up before returning).
static const OpTensorCacheEntry *
queryOrCreateOpTensor(const OpTensorCacheKey &key) {
  auto it = g_optensor_cache.find(key);
  if (it != g_optensor_cache.end())
    return &it->second;

  bool type_ok;
  miopenDataType_t dt = hipdnn_ep_to_miopen_type(key.data_type, &type_ok);
  if (!type_ok)
    return nullptr;
  OpTensorCacheEntry e{};
  miopenStatus_t st;

#define OP_CACHE_CHECK(call)                                                   \
  do {                                                                         \
    st = (call);                                                               \
    if (st != miopenStatusSuccess)                                             \
      goto cache_fail;                                                         \
  } while (0)

  OP_CACHE_CHECK(miopenCreateTensorDescriptor(&e.aDesc));
  OP_CACHE_CHECK(miopenCreateTensorDescriptor(&e.bDesc));
  OP_CACHE_CHECK(miopenCreateTensorDescriptor(&e.cDesc));

  OP_CACHE_CHECK(miopenSet4dTensorDescriptor(
      e.aDesc, dt, static_cast<int>(key.lhs_n), static_cast<int>(key.lhs_c),
      static_cast<int>(key.lhs_h), static_cast<int>(key.lhs_w)));
  OP_CACHE_CHECK(miopenSet4dTensorDescriptor(
      e.bDesc, dt, static_cast<int>(key.rhs_n), static_cast<int>(key.rhs_c),
      static_cast<int>(key.rhs_h), static_cast<int>(key.rhs_w)));
  OP_CACHE_CHECK(miopenSet4dTensorDescriptor(
      e.cDesc, dt, static_cast<int>(key.out_n), static_cast<int>(key.out_c),
      static_cast<int>(key.out_h), static_cast<int>(key.out_w)));

#undef OP_CACHE_CHECK
  goto cache_done;

cache_fail:
  if (e.cDesc)
    miopenDestroyTensorDescriptor(e.cDesc);
  if (e.bDesc)
    miopenDestroyTensorDescriptor(e.bDesc);
  if (e.aDesc)
    miopenDestroyTensorDescriptor(e.aDesc);
  return nullptr;

cache_done:
  auto [ins, _] = g_optensor_cache.emplace(key, e);
  return &ins->second;
}

//===----------------------------------------------------------------------===//
// Generic Element-wise Tensor Operation via MIOpen
//===----------------------------------------------------------------------===//
//
// Uses miopenOpTensor to compute:
//   output = alpha1 * op(lhs, alpha2 * rhs) + beta * output
// With alpha1=1, alpha2=1, beta=0 this gives: output = op(lhs, rhs)
//
// Each operand's shape is passed as 4D (N, C, H, W) to allow MIOpen-native
// broadcasting.  When a dimension is 1 in one operand but >1 in the other,
// MIOpen broadcasts automatically (e.g. bias addition).
// The compiler (HipToLLVM) left-pads shapes with 1 for rank < 4.
//===----------------------------------------------------------------------===//

int wrap_miopenOpTensor(RuntimeState *state, void *lhs, void *rhs, void *output,
                        int64_t lhs_n, int64_t lhs_c, int64_t lhs_h,
                        int64_t lhs_w, int64_t rhs_n, int64_t rhs_c,
                        int64_t rhs_h, int64_t rhs_w, int64_t out_n,
                        int64_t out_c, int64_t out_h, int64_t out_w,
                        int64_t data_type, int64_t tensor_op) {
  if (!state || !lhs || !rhs || !output) {
    fprintf(stderr, "wrap_miopenOpTensor: null argument\n");
    return -1;
  }

  const char *type_name = hipdnn_ep_datatype_name(data_type);
  const char *op_name = hipdnn_ep_tensor_op_name(tensor_op);
  RUNTIME_DEBUG_LOG("[REAL] wrap_miopenOpTensor: op=%s, "
                    "lhs=[%lld,%lld,%lld,%lld], "
                    "rhs=[%lld,%lld,%lld,%lld], "
                    "out=[%lld,%lld,%lld,%lld], "
                    "data_type=%s(%lld)\n",
                    op_name, (long long)lhs_n, (long long)lhs_c,
                    (long long)lhs_h, (long long)lhs_w, (long long)rhs_n,
                    (long long)rhs_c, (long long)rhs_h, (long long)rhs_w,
                    (long long)out_n, (long long)out_c, (long long)out_h,
                    (long long)out_w, type_name, (long long)data_type);

  miopenHandle_t handle =
      static_cast<miopenHandle_t>(hipdnn_ep_state_get_miopen_handle(state));
  if (!handle) {
    fprintf(stderr, "wrap_miopenOpTensor: null MIOpen handle\n");
    return -1;
  }

  bool op_ok;
  miopenTensorOp_t miopen_op = hipdnn_ep_to_miopen_op(tensor_op, &op_ok);
  if (!op_ok) {
    fprintf(stderr, "wrap_miopenOpTensor: unsupported tensor_op %lld\n",
            (long long)tensor_op);
    return -1;
  }

  OpTensorCacheKey key{lhs_n, lhs_c, lhs_h, lhs_w, rhs_n, rhs_c,    rhs_h,
                       rhs_w, out_n, out_c, out_h, out_w, data_type};
  const OpTensorCacheEntry *c = queryOrCreateOpTensor(key);
  if (!c) {
    fprintf(stderr, "wrap_miopenOpTensor: descriptor cache creation failed\n");
    return -1;
  }

  float alpha1 = 1.0f, alpha2 = 1.0f, beta = 0.0f;

  RUNTIME_DEBUG_LOG("[REAL] wrap_miopenOpTensor: calling miopenOpTensor"
                    "(op=%s, alpha1=%.1f, alpha2=%.1f, beta=%.1f)\n",
                    op_name, alpha1, alpha2, beta);

  miopenStatus_t st =
      miopenOpTensor(handle, miopen_op, &alpha1, c->aDesc, lhs, &alpha2,
                     c->bDesc, rhs, &beta, c->cDesc, output);
  if (st != miopenStatusSuccess) {
    fprintf(stderr, "wrap_miopenOpTensor: miopenOpTensor failed (%d)\n", st);
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_miopenOpTensor: completed successfully\n");
  return 0;
}

//===----------------------------------------------------------------------===//
// Element-wise Subtraction via Custom HIP Kernel
//===----------------------------------------------------------------------===//
//
// For types unsupported by MIOpen (e.g. int64), dispatches to
// hip_elementwise_sub from the custom kernels library. The caller passes
// element_size_bytes; we map it to the corresponding hip_dtype_t.
//===----------------------------------------------------------------------===//

int wrap_elementwise_sub(RuntimeState *state, void *lhs, void *rhs,
                         void *output, int64_t num_elements,
                         int64_t element_size_bytes) {
  if (!state || !lhs || !rhs || !output) {
    fprintf(stderr, "wrap_elementwise_sub: null argument\n");
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

  return hip_elementwise_sub(stream, lhs, rhs, output, num_elements, hip_dtype);
}
