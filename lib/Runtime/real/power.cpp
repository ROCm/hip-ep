/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "cache_utils.h"
#include "error_check_macros.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>
#include <functional>
#include <unordered_map>

// Map HIPDNN_EP_DATATYPE_* to hip_dtype_t for elementwise HIP kernels
// (reciprocal, sqrt). Values match hip_dtype_t in hip_custom_kernels.h.
static int hipdnn_ep_to_hip_dtype_elementwise_unary(int64_t data_type) {
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return HIP_DTYPE_BFLOAT16;
  default:
    return -1;
  }
}

// Map execution-provider tensor dtype (HIPDNN_EP_DATATYPE_*) to MIOpen's
// miopenDataType_t for tensor/activation descriptors. On success sets *ok true
// and returns the matching MIOpen enum; on unsupported *data_type* sets *ok
// false, logs, and returns miopenFloat as a safe placeholder (callers must
// check *ok* before using the result).
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
    hipdnn_ep_log_emit("[REAL] unsupported data_type %lld for MIOpen\n",
                       (long long)data_type);
    ok = false;
    return miopenFloat;
  }
}

//===----------------------------------------------------------------------===//
//  MIOpen Power Activation descriptor cache
//===----------------------------------------------------------------------===//
//
// Three MIOpen descriptors (input, output, activation) are created once per
// unique (num_elements, data_type, alpha, beta, gamma) combination and reused
// for the process lifetime. Avoids repeated miopenCreate/Set/Destroy on every
// power operation call.
//
// Uses miopenActivationPOWER mode with formula: (alpha + beta * x)^gamma
// Reciprocal and Sqrt are not executed through this cache (HIP elementwise).
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

  // 4. Configure tensor descriptors: MIOpen NCHW layout (N,C,H,W) =
  //    (num_elements, 1, 1, 1) so N*C*H*W equals element count.
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
// HipToLLVM lowers hip.reciprocal and hip.sqrt to @wrap_power with
// (alpha, beta, gamma). Parameters match MIOpen's POWER activation formula
// y = (alpha + beta*x)^gamma for the MIOpen-backed cases below.
//
// - Reciprocal (0, 1, -1): ONNX 1/x via hip_elementwise_reciprocal.
// - Sqrt (0, 1, 0.5): ONNX sqrt via hip_elementwise_sqrt (negative → NaN).
// - Other (alpha, beta, gamma): MIOpen miopenActivationPOWER path.
//===----------------------------------------------------------------------===//

namespace {

//===----------------------------------------------------------------------===//
// why use HIP kernel for Reciprocal instead of MIOpen miopenActivationPOWER
//===----------------------------------------------------------------------===//
//
// ONNX Reciprocal is IEEE-style element-wise y = 1/x over the full signed
// domain (negative x → negative y; x = 0 → ±inf/NaN per IEEE).
//
// miopenActivationForward with miopenActivationPOWER is an activation-layer
// primitive. Its GPU implementation is aligned with CNN activation use cases:
// the effective operand is often treated as non-negative before applying
// (alpha + beta*x)^gamma (e.g. paths equivalent to clamping or log-pow on a
// non-negative base). For gamma = -1 that breaks ONNX semantics: negative
// inputs can incorrectly become 0 (or otherwise not match 1/x).
//
// Therefore wrap_power(alpha=0, beta=1, gamma=-1) dispatches to
// hip_elementwise_reciprocal instead of MIOpen.
//
// Sqrt (gamma=0.5) has the same activation-shape issue for negative inputs
// (ONNX requires NaN). wrap_power(alpha=0, beta=1, gamma=0.5) dispatches to
// hip_elementwise_sqrt. Other (alpha, beta, gamma) use miopenActivationPOWER.

int launchReciprocalHip(RuntimeState *state, void *input, void *output,
                        int64_t num_elements, int64_t data_type) {
  if (!state || !input || !output) {
    hipdnn_ep_log_emit("launchReciprocalHip: null argument\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  int hip_dtype = hipdnn_ep_to_hip_dtype_elementwise_unary(data_type);
  if (hip_dtype < 0) {
    hipdnn_ep_log_emit(
        "[REAL] wrap_power (reciprocal HIP): unsupported data_type %lld "
        "(%s)\n",
        (long long)data_type, hipdnn_ep_datatype_name(data_type));
    return -1;
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_power (reciprocal HIP 1/x): num_elements=%lld, dtype=%s\n",
      (long long)num_elements, hipdnn_ep_datatype_name(data_type));

  return hip_elementwise_reciprocal(stream, input, output, num_elements,
                                    hip_dtype);
}

int launchSqrtHip(RuntimeState *state, void *input, void *output,
                  int64_t num_elements, int64_t data_type) {
  if (!state || !input || !output) {
    hipdnn_ep_log_emit("launchSqrtHip: null argument\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  int hip_dtype = hipdnn_ep_to_hip_dtype_elementwise_unary(data_type);
  if (hip_dtype < 0) {
    hipdnn_ep_log_emit(
        "[REAL] wrap_power (sqrt HIP): unsupported data_type %lld (%s)\n",
        (long long)data_type, hipdnn_ep_datatype_name(data_type));
    return -1;
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_power (sqrt HIP): num_elements=%lld, dtype=%s\n",
      (long long)num_elements, hipdnn_ep_datatype_name(data_type));

  return hip_elementwise_sqrt(stream, input, output, num_elements, hip_dtype);
}

} // namespace

int wrap_power(RuntimeState *state, void *input, void *output,
               int64_t num_elements, int64_t data_type, double alpha,
               double beta, double gamma) {
  OP_PROFILE(
      "power",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "n=%lld", (long long)num_elements);
        return std::string(b);
      },
      state);
  if (!state || !input || !output) {
    hipdnn_ep_log_emit("wrap_power: null argument\n");
    return -1;
  }

  // hip.reciprocal lowers to wrap_power(…, 0, 1, -1).
  if (alpha == 0.0 && beta == 1.0 && gamma == -1.0)
    return launchReciprocalHip(state, input, output, num_elements, data_type);

  // hip.sqrt lowers to wrap_power(…, 0, 1, 0.5).
  if (alpha == 0.0 && beta == 1.0 && gamma == 0.5)
    return launchSqrtHip(state, input, output, num_elements, data_type);

  const char *type_name = hipdnn_ep_datatype_name(data_type);
  RUNTIME_DEBUG_LOG("[REAL] wrap_power: num_elements=%lld, "
                    "data_type=%s(%lld), alpha=%.2f, beta=%.2f, gamma=%.2f\n",
                    (long long)num_elements, type_name, (long long)data_type,
                    alpha, beta, gamma);

  miopenHandle_t handle =
      static_cast<miopenHandle_t>(hipdnn_ep_state_get_miopen_handle(state));
  if (!handle) {
    hipdnn_ep_log_emit("wrap_power: null MIOpen handle\n");
    return -1;
  }

  PowerActivationCacheKey key{num_elements, data_type, miopenActivationPOWER,
                              alpha,        beta,      gamma};
  const PowerActivationCacheEntry *c = queryOrCreatePowerActivation(key);
  if (!c) {
    hipdnn_ep_log_emit("wrap_power: descriptor cache creation failed\n");
    return -1;
  }

  float scale_alpha = 1.0f, scale_beta = 0.0f;
  miopenStatus_t st =
      miopenActivationForward(handle, c->actDesc, &scale_alpha, c->inDesc,
                              input, &scale_beta, c->outDesc, output);
  if (st != miopenStatusSuccess) {
    hipdnn_ep_log_emit("wrap_power: miopenActivationForward failed (%d)\n", st);
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_power: completed successfully\n");
  return 0;
}
