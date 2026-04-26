/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "cache_utils.h"
#include "error_check_macros.h"
#include "nan_check.h"
#include "runtime_types.h"

#include <cstdio>
#include <functional>
#include <unordered_map>

static miopenDataType_t hipdnn_ep_to_miopen_type(int64_t data_type, bool &ok) {
  ok = true;
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
    ok = false;
    return miopenFloat;
  }
}

static miopenActivationMode_t hipdnn_ep_to_miopen_activation(int64_t mode,
                                                             bool &ok) {
  ok = true;
  switch (mode) {
  case HIPDNN_EP_ACTIVATION_SIGMOID:
    return miopenActivationLOGISTIC;
  case HIPDNN_EP_ACTIVATION_RELU:
    return miopenActivationRELU;
  case HIPDNN_EP_ACTIVATION_TANH:
    return miopenActivationTANH;
  case HIPDNN_EP_ACTIVATION_SOFTPLUS:
    return miopenActivationSOFTRELU;
  default:
    fprintf(stderr, "[REAL] unsupported activation_mode %lld for MIOpen\n",
            (long long)mode);
    ok = false;
    return miopenActivationLOGISTIC;
  }
}

//===----------------------------------------------------------------------===//
// Descriptor cache: 2 tensor descriptors + 1 activation descriptor created
// once per unique (num_elements, data_type, activation_mode) triple and
// reused for the process lifetime.
//===----------------------------------------------------------------------===//

struct ActivationCacheKey {
  int64_t num_elements, data_type, activation_mode;
  bool operator==(const ActivationCacheKey &o) const {
    return num_elements == o.num_elements && data_type == o.data_type &&
           activation_mode == o.activation_mode;
  }
};

struct ActivationCacheKeyHash {
  size_t operator()(const ActivationCacheKey &k) const {
    size_t h = 0;
    hash_combine_val(h, k.num_elements);
    hash_combine_val(h, k.data_type);
    hash_combine_val(h, k.activation_mode);
    return h;
  }
};

/// Cached MIOpen descriptors for a single activation shape.
/// Ownership: descriptors are created in queryOrCreateActivation() and live
/// for the process lifetime (never destroyed individually).
struct ActivationCacheEntry {
  miopenTensorDescriptor_t inDesc, outDesc;
  miopenActivationDescriptor_t actDesc;
};

static std::unordered_map<ActivationCacheKey, ActivationCacheEntry,
                          ActivationCacheKeyHash>
    g_activation_cache;

static const ActivationCacheEntry *
queryOrCreateActivation(const ActivationCacheKey &key) {
  auto it = g_activation_cache.find(key);
  if (it != g_activation_cache.end())
    return &it->second;

  bool type_ok, act_ok;
  miopenDataType_t dt = hipdnn_ep_to_miopen_type(key.data_type, type_ok);
  miopenActivationMode_t act =
      hipdnn_ep_to_miopen_activation(key.activation_mode, act_ok);
  if (!type_ok || !act_ok)
    return nullptr;
  int n = static_cast<int>(key.num_elements);

  ActivationCacheEntry e{};
  int result = 0;

  MIOPEN_CHECK_GOTO(miopenCreateTensorDescriptor(&e.inDesc), cache_fail);
  MIOPEN_CHECK_GOTO(miopenCreateTensorDescriptor(&e.outDesc), cache_fail);
  {
    int dims[] = {1, 1, 1, n};
    MIOPEN_CHECK_GOTO(miopenSetNdTensorDescriptorWithLayout(
                          e.inDesc, dt, miopenTensorNCHW, dims, 4),
                      cache_fail);
    MIOPEN_CHECK_GOTO(miopenSetNdTensorDescriptorWithLayout(
                          e.outDesc, dt, miopenTensorNCHW, dims, 4),
                      cache_fail);
  }
  MIOPEN_CHECK_GOTO(miopenCreateActivationDescriptor(&e.actDesc), cache_fail);
  MIOPEN_CHECK_GOTO(
      miopenSetActivationDescriptor(e.actDesc, act, 0.0, 0.0, 0.0), cache_fail);
  goto cache_done;

cache_fail:
  if (e.actDesc)
    miopenDestroyActivationDescriptor(e.actDesc);
  if (e.outDesc)
    miopenDestroyTensorDescriptor(e.outDesc);
  if (e.inDesc)
    miopenDestroyTensorDescriptor(e.inDesc);
  return nullptr;

cache_done:
  auto [ins, _] = g_activation_cache.emplace(key, e);

  RUNTIME_DEBUG_LOG("[REAL] queryOrCreateActivation: cached 2 tensor + 1 act "
                    "desc for num_elements=%lld data_type=%lld mode=%lld\n",
                    (long long)key.num_elements, (long long)key.data_type,
                    (long long)key.activation_mode);

  return &ins->second;
}

//===----------------------------------------------------------------------===//
// Generic MIOpen Activation Forward
//===----------------------------------------------------------------------===//
//
// Applies activation_mode element-wise using miopenActivationForward.
// Tensor is represented as flat 1D [1, 1, 1, num_elements] to satisfy
// MIOpen's 4D tensor descriptor requirement.
//===----------------------------------------------------------------------===//

int wrap_miopenActivationForward(RuntimeState *state, void *input, void *output,
                                 int64_t num_elements, int64_t data_type,
                                 int64_t activation_mode) {
  if (!state || !input || !output) {
    fprintf(stderr, "wrap_miopenActivationForward: null argument\n");
    return -1;
  }

  const char *act_name = hipdnn_ep_activation_name(activation_mode);
  const char *type_name = hipdnn_ep_datatype_name(data_type);
  int64_t elem_size = hipdnn_ep_datatype_size(data_type);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_miopenActivationForward: activation=%s, "
      "num_elements=%lld, data_type=%s(%lld), element_size=%lld bytes, "
      "total_size=%lld bytes\n",
      act_name, (long long)num_elements, type_name, (long long)data_type,
      (long long)elem_size, (long long)(num_elements * elem_size));

  miopenHandle_t handle =
      static_cast<miopenHandle_t>(hipdnn_ep_state_get_miopen_handle(state));
  if (!handle) {
    fprintf(stderr, "wrap_miopenActivationForward: null MIOpen handle\n");
    return -1;
  }

  ActivationCacheKey key{num_elements, data_type, activation_mode};
  const ActivationCacheEntry *c = queryOrCreateActivation(key);
  if (!c) {
    fprintf(stderr, "[REAL] wrap_miopenActivationForward: descriptor cache "
                    "creation failed\n");
    return -1;
  }

  float alpha = 1.0f, beta = 0.0f;
  miopenStatus_t st = miopenActivationForward(
      handle, c->actDesc, &alpha, c->inDesc, input, &beta, c->outDesc, output);
  if (st != miopenStatusSuccess) {
    fprintf(stderr,
            "[REAL] wrap_miopenActivationForward: "
            "miopenActivationForward failed (%d)\n",
            st);
    return -1;
  }

  nan_trace_check("activation", output, num_elements);

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_miopenActivationForward: completed successfully\n");
  return 0;
}
