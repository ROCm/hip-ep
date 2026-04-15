/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "cache_utils.h"
#include "error_check_macros.h"
#include "runtime_types.h"

#include <cstdio>
#include <functional>
#include <unordered_map>

// Explicit mapping from backend-independent HIPDNN_EP_DATATYPE_* enum to
// MIOpen-specific miopenDataType_t. No static_cast -- our enum values are
// independent of any library.
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

//===----------------------------------------------------------------------===//
// MIOpen Power Activation descriptor cache
//===----------------------------------------------------------------------===//
//
// Three MIOpen descriptors (input, output, activation) are created once per
// unique (num_elements, data_type, alpha, beta, gamma) combination and reused
// for the process lifetime. Avoids repeated miopenCreate/Set/Destroy on every
// power operation call.
//
// Uses miopenActivationPOWER mode with formula: (alpha + beta * x)^gamma
// - Reciprocal: alpha=0, beta=1, gamma=-1.0 → (0 + 1*x)^(-1) = 1/x
// - Sqrt:       alpha=0, beta=1, gamma=0.5  → (0 + 1*x)^(0.5) = √x
// - Square:     alpha=0, beta=1, gamma=2.0  → (0 + 1*x)^2 = x^2
// - Cube:       alpha=0, beta=1, gamma=3.0  → (0 + 1*x)^3 = x^3
// - General:    any alpha, beta, gamma      → (alpha + beta*x)^gamma

struct PowerActivationCacheKey {
  int64_t num_elements, data_type;
  miopenActivationMode_t mode; // Always miopenActivationPOWER
  double alpha, beta, gamma;   // MIOpen Power activation parameters

  bool operator==(const PowerActivationCacheKey &o) const {
    return num_elements == o.num_elements && data_type == o.data_type &&
           mode == o.mode && alpha == o.alpha && beta == o.beta &&
           gamma == o.gamma;
  }
};

struct PowerActivationCacheKeyHash {
  size_t operator()(const PowerActivationCacheKey &k) const {
    size_t h = 0;
    hash_combine_val(h, k.num_elements);
    hash_combine_val(h, k.data_type);
    hash_combine_val(h, static_cast<int>(k.mode));
    // Use fixed-point representation for floating-point params to ensure stable
    // hashing
    hash_combine_val(h, static_cast<int64_t>(k.alpha * 1000000));
    hash_combine_val(h, static_cast<int64_t>(k.beta * 1000000));
    hash_combine_val(h, static_cast<int64_t>(k.gamma * 1000000));
    return h;
  }
};

struct PowerActivationCacheEntry {
  miopenTensorDescriptor_t inDesc, outDesc;
  miopenActivationDescriptor_t actDesc;
};

static std::unordered_map<PowerActivationCacheKey, PowerActivationCacheEntry,
                          PowerActivationCacheKeyHash>
    g_power_activation_cache;

/// Look up or create cached MIOpen descriptors for power activation.
/// Returns nullptr on any MIOpen API failure (partially created descriptors
/// are cleaned up before returning).
static const PowerActivationCacheEntry *
queryOrCreatePowerActivation(const PowerActivationCacheKey &key) {
  // 1. Check cache
  auto it = g_power_activation_cache.find(key);
  if (it != g_power_activation_cache.end())
    return &it->second;

  // 2. Type validation
  bool type_ok;
  miopenDataType_t dt = hipdnn_ep_to_miopen_type(key.data_type, type_ok);
  if (!type_ok)
    return nullptr;

  int n = static_cast<int>(key.num_elements);
  PowerActivationCacheEntry e{};
  int result = 0;

  // 3. Create descriptors
  MIOPEN_CHECK_GOTO(miopenCreateTensorDescriptor(&e.inDesc), cache_fail);
  MIOPEN_CHECK_GOTO(miopenCreateTensorDescriptor(&e.outDesc), cache_fail);
  MIOPEN_CHECK_GOTO(miopenCreateActivationDescriptor(&e.actDesc), cache_fail);

  // 4. Configure tensor descriptors (flatten to [1,1,1,N])
  MIOPEN_CHECK_GOTO(miopenSet4dTensorDescriptor(e.inDesc, dt, n, 1, 1, 1),
                    cache_fail);
  MIOPEN_CHECK_GOTO(miopenSet4dTensorDescriptor(e.outDesc, dt, n, 1, 1, 1),
                    cache_fail);

  // 5. Configure activation descriptor
  // miopenActivationPOWER: (alpha + beta * x)^gamma
  MIOPEN_CHECK_GOTO(miopenSetActivationDescriptor(
                        e.actDesc, key.mode, key.alpha, key.beta, key.gamma),
                    cache_fail);
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
  auto [ins, _] = g_power_activation_cache.emplace(key, e);
  return &ins->second;
}

//===----------------------------------------------------------------------===//
// Generic Power Operation via MIOpen Activation Power
//===----------------------------------------------------------------------===//
//
// Unified implementation for all power operations: y = (alpha + beta * x)^gamma
// Fully supports miopenActivationPOWER formula with all three parameters.
//
// Common operations:
// - Reciprocal: alpha=0, beta=1, gamma=-1.0  → (0 + 1*x)^(-1) = 1/x
// - Sqrt:       alpha=0, beta=1, gamma=0.5   → (0 + 1*x)^(0.5) = √x
// - Square:     alpha=0, beta=1, gamma=2.0   → (0 + 1*x)^2 = x^2
// - Cube:       alpha=0, beta=1, gamma=3.0   → (0 + 1*x)^3 = x^3
// - General:    any alpha, beta, gamma        → (alpha + beta*x)^gamma
//
// This implementation leverages MIOpen's hardware-optimized power activation,
// providing better cross-architecture performance and production stability.
//===----------------------------------------------------------------------===//

int wrap_power(RuntimeState *state, void *input, void *output,
               int64_t num_elements, int64_t data_type, double alpha,
               double beta, double gamma) {
  if (!state || !input || !output) {
    fprintf(stderr, "wrap_power: null argument\n");
    return -1;
  }

  const char *type_name = hipdnn_ep_datatype_name(data_type);
  RUNTIME_DEBUG_LOG("[REAL] wrap_power: num_elements=%lld, "
                    "data_type=%s(%lld), alpha=%.2f, beta=%.2f, gamma=%.2f\n",
                    (long long)num_elements, type_name, (long long)data_type,
                    alpha, beta, gamma);

  miopenHandle_t handle =
      static_cast<miopenHandle_t>(hipdnn_ep_state_get_miopen_handle(state));
  if (!handle) {
    fprintf(stderr, "wrap_power: null MIOpen handle\n");
    return -1;
  }

  PowerActivationCacheKey key{num_elements, data_type, miopenActivationPOWER,
                              alpha,        beta,      gamma};
  const PowerActivationCacheEntry *c = queryOrCreatePowerActivation(key);
  if (!c) {
    fprintf(stderr, "wrap_power: descriptor cache creation failed\n");
    return -1;
  }

  float scale_alpha = 1.0f, scale_beta = 0.0f;
  miopenStatus_t st =
      miopenActivationForward(handle, c->actDesc, &scale_alpha, c->inDesc,
                              input, &scale_beta, c->outDesc, output);
  if (st != miopenStatusSuccess) {
    fprintf(stderr, "wrap_power: miopenActivationForward failed (%d)\n", st);
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_power: completed successfully\n");
  return 0;
}

//===----------------------------------------------------------------------===//
// Convenience wrappers for common power operations
//===----------------------------------------------------------------------===//

int wrap_reciprocal(RuntimeState *state, void *input, void *output,
                    int64_t num_elements, int64_t data_type) {
  // Reciprocal: (0 + 1*x)^(-1) = 1/x
  return wrap_power(state, input, output, num_elements, data_type, 0.0, 1.0,
                    -1.0);
}

int wrap_sqrt(RuntimeState *state, void *input, void *output,
              int64_t num_elements, int64_t data_type) {
  // Sqrt: (0 + 1*x)^(0.5) = √x
  return wrap_power(state, input, output, num_elements, data_type, 0.0, 1.0,
                    0.5);
}
