/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "error_check_macros.h"
#include "runtime_types.h"

#include <cstdio>
#include <functional>
#include <unordered_map>

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

// Maps HIPDNN_EP_ACTIVATION_* to miopenActivationMode_t.
// MIOpen calls sigmoid "logistic" (miopenActivationLOGISTIC).
static miopenActivationMode_t hipdnn_ep_to_miopen_activation(int64_t mode) {
  switch (mode) {
  case HIPDNN_EP_ACTIVATION_SIGMOID:
    return miopenActivationLOGISTIC;
  case HIPDNN_EP_ACTIVATION_RELU:
    return miopenActivationRELU;
  case HIPDNN_EP_ACTIVATION_TANH:
    return miopenActivationTANH;
  default:
    fprintf(stderr, "[REAL] unsupported activation_mode %lld for MIOpen\n",
            (long long)mode);
    return miopenActivationLOGISTIC;
  }
}

// =============================================================================
// Descriptor cache: 2 tensor descriptors + 1 activation descriptor created
// once per unique (num_elements, data_type, activation_mode) triple and
// reused for the process lifetime.
// =============================================================================

struct ActivationCacheKey {
  int64_t num_elements, data_type, activation_mode;
  bool operator==(const ActivationCacheKey &o) const {
    return num_elements == o.num_elements && data_type == o.data_type &&
           activation_mode == o.activation_mode;
  }
};

struct ActivationCacheKeyHash {
  size_t operator()(const ActivationCacheKey &k) const {
    size_t h = std::hash<int64_t>{}(k.num_elements);
    h ^= std::hash<int64_t>{}(k.data_type) + 0x9e3779b9 + (h << 6) +
         (h >> 2);
    h ^= std::hash<int64_t>{}(k.activation_mode) + 0x9e3779b9 + (h << 6) +
         (h >> 2);
    return h;
  }
};

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

  miopenDataType_t dt = hipdnn_ep_to_miopen_type(key.data_type);
  miopenActivationMode_t act =
      hipdnn_ep_to_miopen_activation(key.activation_mode);
  int n = static_cast<int>(key.num_elements);

  ActivationCacheEntry e{};

  if (miopenCreateTensorDescriptor(&e.inDesc) != miopenStatusSuccess)
    return nullptr;
  if (miopenCreateTensorDescriptor(&e.outDesc) != miopenStatusSuccess) {
    miopenDestroyTensorDescriptor(e.inDesc);
    return nullptr;
  }

  miopenSet4dTensorDescriptor(e.inDesc, dt, 1, 1, 1, n);
  miopenSet4dTensorDescriptor(e.outDesc, dt, 1, 1, 1, n);

  if (miopenCreateActivationDescriptor(&e.actDesc) != miopenStatusSuccess) {
    miopenDestroyTensorDescriptor(e.outDesc);
    miopenDestroyTensorDescriptor(e.inDesc);
    return nullptr;
  }
  miopenSetActivationDescriptor(e.actDesc, act, 0.0, 0.0, 0.0);

  auto [ins, _] = g_activation_cache.emplace(key, e);

  RUNTIME_DEBUG_LOG("[REAL] queryOrCreateActivation: cached 2 tensor + 1 act "
                    "desc for num_elements=%lld data_type=%lld mode=%lld\n",
                    (long long)key.num_elements, (long long)key.data_type,
                    (long long)key.activation_mode);

  return &ins->second;
}

// =============================================================================
// Generic MIOpen Activation Forward
// =============================================================================
//
// Applies activation_mode element-wise using miopenActivationForward.
// Tensor is represented as flat 1D [1, 1, 1, num_elements] to satisfy
// miopenSet4dTensorDescriptor's 4D requirement.
// =============================================================================

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
    RUNTIME_DEBUG_LOG("[REAL] wrap_miopenActivationForward: descriptor cache "
                      "creation failed\n");
    return -1;
  }

  float alpha = 1.0f, beta = 0.0f;
  miopenStatus_t st =
      miopenActivationForward(handle, c->actDesc, &alpha, c->inDesc, input,
                              &beta, c->outDesc, output);
  if (st != miopenStatusSuccess) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_miopenActivationForward: "
                      "miopenActivationForward failed (%d)\n",
                      st);
    return -1;
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_miopenActivationForward: completed successfully\n");
  return 0;
}
