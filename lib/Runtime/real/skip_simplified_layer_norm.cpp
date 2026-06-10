/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Enable MIOpen beta APIs (miopenT5LayerNormForward, miopenNormMode_t)
#define MIOPEN_BETA_API

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "cache_utils.h"
#include "error_check_macros.h"
#include "runtime_types.h"

#include <cstdio>
#include <functional>
#include <unordered_map>

// Use the shared MIOPEN_CHECK_GOTO macro for goto cleanup pattern
#undef MIOPEN_CHECK
#define MIOPEN_CHECK(cmd) MIOPEN_CHECK_GOTO(cmd, cleanup)

static constexpr size_t kScratchAlignment = 256;

// ck_dsl_skip_simplified_layer_norm() (the fused f32/N=4096 fast path) is
// declared in hipdnn_ep_runtime.h alongside the other runtime ops.

//===----------------------------------------------------------------------===//
// Descriptor cache: T5LayerNorm + OpTensor descriptors created once per unique
// (num_rows, hidden_dim, data_type) triple, reused for process lifetime.
// Follows the same pattern as activation.cpp (commit 2f560bb).
//===----------------------------------------------------------------------===//

struct SkipT5NormCacheKey {
  int64_t num_rows, hidden_dim;
  miopenDataType_t data_type;
  bool operator==(const SkipT5NormCacheKey &o) const {
    return num_rows == o.num_rows && hidden_dim == o.hidden_dim &&
           data_type == o.data_type;
  }
};

struct SkipT5NormCacheKeyHash {
  size_t operator()(const SkipT5NormCacheKey &k) const {
    size_t h = 0;
    hash_combine_val(h, k.num_rows);
    hash_combine_val(h, k.hidden_dim);
    hash_combine_val(h, static_cast<int>(k.data_type));
    return h;
  }
};

/// Cached MIOpen descriptors for a single SkipSimplifiedLayerNorm shape.
/// Ownership: descriptors are created in queryOrCreateSkipT5Norm() and live
/// for the process lifetime (never destroyed individually).
struct SkipT5NormCacheEntry {
  // T5LayerNorm descriptors
  miopenTensorDescriptor_t xDesc, yDesc, weightDesc, rstdDesc;
  // OpTensor descriptors (input + skip [+ bias] add)
  miopenTensorDescriptor_t addADesc, addBDesc, addCDesc, biasDesc;
};

static std::unordered_map<SkipT5NormCacheKey, SkipT5NormCacheEntry,
                          SkipT5NormCacheKeyHash>
    g_skip_t5norm_cache;

static const SkipT5NormCacheEntry *
queryOrCreateSkipT5Norm(const SkipT5NormCacheKey &key) {
  auto it = g_skip_t5norm_cache.find(key);
  if (it != g_skip_t5norm_cache.end())
    return &it->second;

  SkipT5NormCacheEntry e{};
  constexpr int NUM_DESCS = 8;
  miopenTensorDescriptor_t *descs[NUM_DESCS] = {
      &e.xDesc,    &e.yDesc,    &e.weightDesc, &e.rstdDesc,
      &e.addADesc, &e.addBDesc, &e.addCDesc,   &e.biasDesc};
  int created = 0;
  miopenStatus_t st;

#define SKIP_CACHE_CHECK(call)                                                 \
  do {                                                                         \
    st = (call);                                                               \
    if (st != miopenStatusSuccess)                                             \
      goto cache_fail;                                                         \
  } while (0)

  for (int i = 0; i < NUM_DESCS; i++) {
    SKIP_CACHE_CHECK(miopenCreateTensorDescriptor(descs[i]));
    created++;
  }

  {
    // T5LayerNorm: 2D [num_rows, hidden_dim]
    int x_dims[] = {static_cast<int>(key.num_rows),
                    static_cast<int>(key.hidden_dim)};
    int x_strides[] = {static_cast<int>(key.hidden_dim), 1};
    int w_dims[] = {static_cast<int>(key.hidden_dim)};
    int w_strides[] = {1};
    int rstd_dims[] = {static_cast<int>(key.num_rows)};
    int rstd_strides[] = {1};
    int bias_dims[] = {1, static_cast<int>(key.hidden_dim)};
    int bias_strides[] = {static_cast<int>(key.hidden_dim), 1};

    SKIP_CACHE_CHECK(miopenSetTensorDescriptor(e.xDesc, key.data_type, 2,
                                               x_dims, x_strides));
    SKIP_CACHE_CHECK(miopenSetTensorDescriptor(e.yDesc, key.data_type, 2,
                                               x_dims, x_strides));
    SKIP_CACHE_CHECK(miopenSetTensorDescriptor(e.weightDesc, key.data_type, 1,
                                               w_dims, w_strides));
    SKIP_CACHE_CHECK(miopenSetTensorDescriptor(e.rstdDesc, miopenFloat, 1,
                                               rstd_dims, rstd_strides));
    SKIP_CACHE_CHECK(miopenSetTensorDescriptor(e.addADesc, key.data_type, 2,
                                               x_dims, x_strides));
    SKIP_CACHE_CHECK(miopenSetTensorDescriptor(e.addBDesc, key.data_type, 2,
                                               x_dims, x_strides));
    SKIP_CACHE_CHECK(miopenSetTensorDescriptor(e.addCDesc, key.data_type, 2,
                                               x_dims, x_strides));
    SKIP_CACHE_CHECK(miopenSetTensorDescriptor(e.biasDesc, key.data_type, 2,
                                               bias_dims, bias_strides));
  }

#undef SKIP_CACHE_CHECK
  goto cache_done;

cache_fail:
  for (int j = 0; j < created; j++)
    miopenDestroyTensorDescriptor(*descs[j]);
  return nullptr;

cache_done:
  auto [ins, _] = g_skip_t5norm_cache.emplace(key, e);

  RUNTIME_DEBUG_LOG("[REAL] queryOrCreateSkipT5Norm: cached 8 descriptors for "
                    "num_rows=%lld hidden_dim=%lld data_type=%d\n",
                    (long long)key.num_rows, (long long)key.hidden_dim,
                    static_cast<int>(key.data_type));

  return &ins->second;
}

//===----------------------------------------------------------------------===//
// SkipSimplifiedLayerNormalization via MIOpen (Full MS spec)
//===----------------------------------------------------------------------===//
//
// ONNX SkipSimplifiedLayerNormalization (com.microsoft):
//   input_skip_bias_sum = input + skip [+ bias]
//   output              = RMSNorm(input_skip_bias_sum) * gamma
//
// MIOpen has no fused "add + T5 norm" API, so we compose calls:
//   1. miopenOpTensor(ADD):           tmp = input + skip
//   2. miopenOpTensor(ADD):           tmp = tmp + bias   (if bias != nullptr)
//   3. miopenT5LayerNormForward:      output = RMSNorm(tmp) * gamma
//
// All execute on the same GPU stream via the MIOpen handle -- no host-device
// round trips.
//
// If input_skip_bias_sum is nullptr (optional output not requested), a
// scratch region from the shared workspace is used for the intermediate result.
//
// Tensor layout:
//   input / skip / input_skip_bias_sum:  [num_rows, hidden_dim]
//   gamma / bias:                        [hidden_dim]
//   output:                              [num_rows, hidden_dim]
//   rstd (scratch):                      [num_rows] (f32)
//===----------------------------------------------------------------------===//

int wrap_skip_simplified_layer_norm(RuntimeState *state, void *input,
                                    void *skip, void *gamma, void *bias,
                                    void *output, void *input_skip_bias_sum,
                                    int64_t input_num_elements,
                                    int64_t gamma_num_elements,
                                    int64_t element_size_bytes, float epsilon) {
  OP_PROFILE(
      "skip_layernorm",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "%lldx%lld",
                 (long long)(gamma_num_elements > 0
                                 ? input_num_elements / gamma_num_elements
                                 : 0),
                 (long long)gamma_num_elements);
        return std::string(b);
      },
      state);
  if (!state || !input || !skip || !gamma || !output) {
    fprintf(stderr,
            "wrap_skip_simplified_layer_norm: null required argument\n");
    return -1;
  }

  miopenHandle_t handle =
      static_cast<miopenHandle_t>(hipdnn_ep_state_get_miopen_handle(state));
  if (!handle) {
    fprintf(stderr, "wrap_skip_simplified_layer_norm: null MIOpen handle\n");
    return -1;
  }

  int64_t hidden_dim = gamma_num_elements;
  int64_t num_rows = input_num_elements / hidden_dim;

  const char *type_name = (element_size_bytes == 2)   ? "f16"
                          : (element_size_bytes == 4) ? "f32"
                                                      : "?";
  RUNTIME_DEBUG_LOG("[REAL] wrap_skip_simplified_layer_norm: num_rows=%lld, "
                    "hidden_dim=%lld, data_type=%s, epsilon=%e, "
                    "bias=%s, input_skip_bias_sum=%s\n",
                    (long long)num_rows, (long long)hidden_dim, type_name,
                    (double)epsilon, bias ? "yes" : "no",
                    input_skip_bias_sum ? "yes" : "no");

  miopenDataType_t data_type;
  if (element_size_bytes == 2)
    data_type = miopenHalf;
  else if (element_size_bytes == 4)
    data_type = miopenFloat;
  else {
    fprintf(stderr,
            "wrap_skip_simplified_layer_norm: unsupported element_size %lld\n",
            (long long)element_size_bytes);
    return -1;
  }

  // Look up or create cached descriptors (T5LayerNorm + OpTensor)
  SkipT5NormCacheKey key{num_rows, hidden_dim, data_type};
  const SkipT5NormCacheEntry *c = queryOrCreateSkipT5Norm(key);
  if (!c) {
    fprintf(
        stderr,
        "wrap_skip_simplified_layer_norm: descriptor cache creation failed\n");
    return -1;
  }

  // Shared workspace: pack [tmp_skip_buf (if needed) | rstd_buf]
  // Align rstd_buf to 256 bytes for GPU memory access efficiency.
  size_t skip_bytes = 0;
  if (!input_skip_bias_sum)
    skip_bytes = static_cast<size_t>(input_num_elements) * element_size_bytes;
  size_t skip_aligned =
      (skip_bytes + kScratchAlignment - 1) & ~(kScratchAlignment - 1);
  size_t rstd_bytes = static_cast<size_t>(num_rows) * sizeof(float);
  size_t total_ws = skip_aligned + rstd_bytes;

  if (hipdnn_ep_state_ensure_workspace(state, total_ws) != 0) {
    fprintf(stderr,
            "wrap_skip_simplified_layer_norm: workspace allocation failed\n");
    return -1;
  }
  char *ws = static_cast<char *>(hipdnn_ep_state_get_workspace(state));

  void *skip_buf = input_skip_bias_sum ? input_skip_bias_sum : ws;
  void *rstd_buf = ws + skip_aligned;

  int result = 0;

  //===--------------------------------------------------------------------===//
  // Fast path: ck_dsl-generated fused (Add + RMSNorm * Gamma) kernel.
  // Only fires for the exact shape the HSACO was compiled for (f32,
  // hidden_dim=4096, no bias). Anything else falls through to the MIOpen
  // baseline below.
  //===--------------------------------------------------------------------===//
  if (data_type == miopenFloat && !bias && hidden_dim == 4096) {
    void *stream = hipdnn_ep_state_get_stream(state);
    if (stream) {
      int ck_rc = ck_dsl_skip_simplified_layer_norm(stream, input, skip, gamma,
                                                    output, skip_buf, num_rows,
                                                    hidden_dim, epsilon);
      if (ck_rc == 0) {
        RUNTIME_DEBUG_LOG("[REAL] wrap_skip_simplified_layer_norm: ck_dsl "
                          "fused fast path used (M=%lld, N=%lld)\n",
                          (long long)num_rows, (long long)hidden_dim);
        return 0;
      }
      // ck_rc == -2 -> RejectFallback (shape mismatch or load failure);
      // any other non-zero -> HIP launch error already logged. Either
      // way fall through to the MIOpen baseline below.
    }
  }

  //===--------------------------------------------------------------------===//
  // Step 1: Element-wise add -- skip_buf = input + skip
  //===--------------------------------------------------------------------===//
  RUNTIME_DEBUG_LOG("[REAL] wrap_skip_simplified_layer_norm: step 1 -- "
                    "miopenOpTensor(ADD) for %lld elements\n",
                    (long long)input_num_elements);

  {
    float alpha1 = 1.0f, alpha2 = 1.0f, beta = 0.0f;
    MIOPEN_CHECK(miopenOpTensor(handle, miopenTensorOpAdd, &alpha1, c->addADesc,
                                input, &alpha2, c->addBDesc, skip, &beta,
                                c->addCDesc, skip_buf));
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_skip_simplified_layer_norm: step 1 completed "
                    "(add)\n");

  //===--------------------------------------------------------------------===//
  // Step 1b (optional): Add bias -- skip_buf = skip_buf + bias
  //===--------------------------------------------------------------------===//
  if (bias) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_skip_simplified_layer_norm: step 1b -- "
                      "adding bias (%lld elements, broadcast over %lld rows)\n",
                      (long long)hidden_dim, (long long)num_rows);

    {
      float alpha1 = 1.0f, alpha2 = 1.0f, beta = 0.0f;
      MIOPEN_CHECK(miopenOpTensor(handle, miopenTensorOpAdd, &alpha1,
                                  c->addCDesc, skip_buf, &alpha2, c->biasDesc,
                                  bias, &beta, c->addCDesc, skip_buf));
    }

    RUNTIME_DEBUG_LOG("[REAL] wrap_skip_simplified_layer_norm: step 1b "
                      "completed (bias add)\n");
  }

  //===--------------------------------------------------------------------===//
  // Step 2: T5 RMS norm -- output = RMSNorm(skip_buf) * gamma
  //===--------------------------------------------------------------------===//
  RUNTIME_DEBUG_LOG("[REAL] wrap_skip_simplified_layer_norm: step 2 -- "
                    "miopenT5LayerNormForward(eps=%e)\n",
                    (double)epsilon);

  MIOPEN_CHECK(miopenT5LayerNormForward(
      handle, MIOPEN_ELEMENTWISE_AFFINE_T5, c->xDesc, skip_buf, c->weightDesc,
      gamma, epsilon, c->yDesc, output, c->rstdDesc, rstd_buf));

  RUNTIME_DEBUG_LOG("[REAL] wrap_skip_simplified_layer_norm: completed "
                    "successfully\n");
  return 0;

cleanup:
  return result;
}
