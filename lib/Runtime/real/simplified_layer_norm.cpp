/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Enable MIOpen beta APIs (miopenT5LayerNormForward, miopenNormMode_t)
#define MIOPEN_BETA_API

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "../op_state.h"
#include "cache_utils.h"
#include "error_check_macros.h"
#include "runtime_types.h"

#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

// ck_dsl_simplified_layer_norm() (the f32/N=4096 fast path) is declared in
// hipdnn_ep_runtime.h alongside the other runtime ops.

// Use the shared macros from error_check_macros.h with goto cleanup pattern
#define MIOPEN_CHECK(cmd) MIOPEN_CHECK_GOTO(cmd, cleanup)

//===----------------------------------------------------------------------===//
// Descriptor cache: 4 tensor descriptors created once per unique
// (num_rows, hidden_dim, data_type) triple and reused for process lifetime.
// Follows the same pattern as activation.cpp (commit 2f560bb).
//===----------------------------------------------------------------------===//

struct T5NormCacheKey {
  int64_t num_rows, hidden_dim;
  miopenDataType_t data_type;
  bool operator==(const T5NormCacheKey &o) const {
    return num_rows == o.num_rows && hidden_dim == o.hidden_dim &&
           data_type == o.data_type;
  }
};

struct T5NormCacheKeyHash {
  size_t operator()(const T5NormCacheKey &k) const {
    size_t h = 0;
    hash_combine_val(h, k.num_rows);
    hash_combine_val(h, k.hidden_dim);
    hash_combine_val(h, static_cast<int>(k.data_type));
    return h;
  }
};

/// Cached MIOpen tensor descriptors for a single T5LayerNorm shape. Owned by
/// the T5NormTable, which destroys them when the last session sharing it is
/// torn down (previously these leaked for the process lifetime).
struct T5NormCacheEntry {
  miopenTensorDescriptor_t xDesc, yDesc, weightDesc, rstdDesc;
};

// One descriptor table shared across every session in the process and freed
// when the last session holding it is destroyed. MIOpen tensor descriptors are
// pure shape/dtype metadata (no device binding), so a single table is correct
// for all sessions; the mutex guards find/insert because sessions run
// Compute() on independent threads.
struct T5NormTable {
  std::mutex mu;
  std::unordered_map<T5NormCacheKey, T5NormCacheEntry, T5NormCacheKeyHash> map;
  ~T5NormTable() {
    for (auto &kv : map) {
      T5NormCacheEntry &e = kv.second;
      if (e.rstdDesc)
        miopenDestroyTensorDescriptor(e.rstdDesc);
      if (e.weightDesc)
        miopenDestroyTensorDescriptor(e.weightDesc);
      if (e.yDesc)
        miopenDestroyTensorDescriptor(e.yDesc);
      if (e.xDesc)
        miopenDestroyTensorDescriptor(e.xDesc);
    }
  }
};

// Per-instance op state for hip.rms_norm: the slot holds a shared_ptr to the
// one shared descriptor table, reached through a global WeakStore keyed by
// device (see op_state.h). The store is weak_ptr-backed, so the table lives
// only while some session's T5NormState holds a shared_ptr to it.
struct T5NormState : OpStateT<T5NormState> {
  std::shared_ptr<T5NormTable> table;
  T5NormState() {
    int dev = 0;
    hipGetDevice(&dev);
    table = WeakStore<int, T5NormTable>::get_or_create(
        dev, [] { return std::make_shared<T5NormTable>(); });
  }
};

extern "C" int8_t hipdnn_ep_op_state_construct_t5norm(RuntimeState *state,
                                                      int32_t slot) {
  hipdnn_ep_op_state_set(state, slot, T5NormState::create().release());
  return 0;
}

static const T5NormCacheEntry *queryOrCreateT5Norm(T5NormTable &table,
                                                   const T5NormCacheKey &key) {
  // The table is shared across sessions, so guard find/insert. Entries are
  // never erased, so the returned pointer stays valid after the lock drops.
  std::lock_guard<std::mutex> guard(table.mu);
  auto it = table.map.find(key);
  if (it != table.map.end())
    return &it->second;

  T5NormCacheEntry e{};
  miopenStatus_t st;

#define T5_CACHE_CHECK(call)                                                   \
  do {                                                                         \
    st = (call);                                                               \
    if (st != miopenStatusSuccess)                                             \
      goto cache_fail;                                                         \
  } while (0)

  T5_CACHE_CHECK(miopenCreateTensorDescriptor(&e.xDesc));
  T5_CACHE_CHECK(miopenCreateTensorDescriptor(&e.yDesc));
  T5_CACHE_CHECK(miopenCreateTensorDescriptor(&e.weightDesc));
  T5_CACHE_CHECK(miopenCreateTensorDescriptor(&e.rstdDesc));

  {
    int x_dims[] = {static_cast<int>(key.num_rows),
                    static_cast<int>(key.hidden_dim)};
    int x_strides[] = {static_cast<int>(key.hidden_dim), 1};
    int w_dims[] = {static_cast<int>(key.hidden_dim)};
    int w_strides[] = {1};
    int rstd_dims[] = {static_cast<int>(key.num_rows)};
    int rstd_strides[] = {1};

    T5_CACHE_CHECK(miopenSetTensorDescriptor(e.xDesc, key.data_type, 2, x_dims,
                                             x_strides));
    T5_CACHE_CHECK(miopenSetTensorDescriptor(e.yDesc, key.data_type, 2, x_dims,
                                             x_strides));
    T5_CACHE_CHECK(miopenSetTensorDescriptor(e.weightDesc, key.data_type, 1,
                                             w_dims, w_strides));
    T5_CACHE_CHECK(miopenSetTensorDescriptor(e.rstdDesc, miopenFloat, 1,
                                             rstd_dims, rstd_strides));
  }

#undef T5_CACHE_CHECK
  goto cache_done;

cache_fail:
  if (e.rstdDesc)
    miopenDestroyTensorDescriptor(e.rstdDesc);
  if (e.weightDesc)
    miopenDestroyTensorDescriptor(e.weightDesc);
  if (e.yDesc)
    miopenDestroyTensorDescriptor(e.yDesc);
  if (e.xDesc)
    miopenDestroyTensorDescriptor(e.xDesc);
  return nullptr;

cache_done:
  auto [ins, _] = table.map.emplace(key, e);

  RUNTIME_DEBUG_LOG("[REAL] queryOrCreateT5Norm: cached 4 descriptors for "
                    "num_rows=%lld hidden_dim=%lld data_type=%d\n",
                    (long long)key.num_rows, (long long)key.hidden_dim,
                    static_cast<int>(key.data_type));

  return &ins->second;
}

//===----------------------------------------------------------------------===//
// SimplifiedLayerNormalization via MIOpen T5LayerNorm
//===----------------------------------------------------------------------===//
//
// ONNX SimplifiedLayerNormalization (RMS Norm):
//   rms   = sqrt(mean(input^2, axis) + epsilon)
//   output = (input / rms) * scale
//
// This maps directly to MIOpen's T5LayerNorm with MIOPEN_ELEMENTWISE_AFFINE_T5.
//
// Tensor layout (row-major):
//   input:  [num_rows, hidden_dim]  where num_rows = input_num_elements /
//   scale_num_elements scale:  [hidden_dim] output: [num_rows, hidden_dim]
//   rstd:   [num_rows]              (scratch -- not exposed to caller)
//===----------------------------------------------------------------------===//

int wrap_miopenT5LayerNormForward(RuntimeState *state, int op_state_slot,
                                  void *input, void *scale, void *output,
                                  int64_t input_num_elements,
                                  int64_t scale_num_elements,
                                  int64_t element_size_bytes, int64_t axis,
                                  float epsilon, int64_t stash_type) {
  OP_PROFILE(
      "layernorm",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "%lldx%lld",
                 (long long)(scale_num_elements > 0
                                 ? input_num_elements / scale_num_elements
                                 : 0),
                 (long long)scale_num_elements);
        return std::string(b);
      },
      state);
  if (!state || !input || !scale || !output) {
    fprintf(stderr, "Invalid arguments to wrap_miopenT5LayerNormForward\n");
    return -1;
  }

  miopenHandle_t handle =
      static_cast<miopenHandle_t>(hipdnn_ep_state_get_miopen_handle(state));
  if (!handle) {
    fprintf(stderr, "wrap_miopenT5LayerNormForward: null MIOpen handle\n");
    return -1;
  }

  int64_t hidden_dim = scale_num_elements;
  int64_t num_rows = input_num_elements / hidden_dim;

  const char *type_name = (element_size_bytes == 2)   ? "f16"
                          : (element_size_bytes == 4) ? "f32"
                                                      : "?";
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_miopenT5LayerNormForward: num_rows=%lld, hidden_dim=%lld, "
      "data_type=%s, epsilon=%e, "
      "total_bytes=%lld\n",
      (long long)num_rows, (long long)hidden_dim, type_name, (double)epsilon,
      (long long)(input_num_elements * element_size_bytes));

  miopenDataType_t data_type;
  if (element_size_bytes == 2)
    data_type = miopenHalf;
  else if (element_size_bytes == 4)
    data_type = miopenFloat;
  else {
    fprintf(stderr,
            "wrap_miopenT5LayerNormForward: unsupported element_size %lld\n",
            (long long)element_size_bytes);
    return -1;
  }

  T5NormState *ns = T5NormState::get_op_state(state, op_state_slot);
  if (!ns || !ns->table) {
    fprintf(stderr,
            "wrap_miopenT5LayerNormForward: missing op-state for slot %d\n",
            op_state_slot);
    return -1;
  }

  // Look up or create cached descriptors
  T5NormCacheKey key{num_rows, hidden_dim, data_type};
  const T5NormCacheEntry *c = queryOrCreateT5Norm(*ns->table, key);
  if (!c) {
    fprintf(
        stderr,
        "wrap_miopenT5LayerNormForward: descriptor cache creation failed\n");
    return -1;
  }

  // Use shared workspace for rstd scratch buffer (always f32)
  size_t rstd_bytes = static_cast<size_t>(num_rows) * sizeof(float);
  if (hipdnn_ep_state_ensure_workspace(state, rstd_bytes) != 0) {
    fprintf(stderr,
            "wrap_miopenT5LayerNormForward: workspace allocation failed\n");
    return -1;
  }
  void *rstd_buf = hipdnn_ep_state_get_workspace(state);

  // Fast path: ck_dsl-generated RMSNorm kernel. The shim itself decides
  // whether the (dtype, hidden_dim) tuple matches its compiled HSACO
  // (f32/N=4096); anything else returns -2 and we fall through to MIOpen.
  // Sibling of the SkipSimplifiedLayerNorm fast path.
  {
    void *stream = hipdnn_ep_state_get_stream(state);
    if (stream) {
      int ck_rc =
          ck_dsl_simplified_layer_norm(stream, input, scale, output, num_rows,
                                       hidden_dim, epsilon, element_size_bytes);
      if (ck_rc == 0) {
        return 0;
      }
    }
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_miopenT5LayerNormForward: calling miopenT5LayerNormForward"
      "(mode=ELEMENTWISE_AFFINE_T5, eps=%e)\n",
      (double)epsilon);

  int result = 0;
  MIOPEN_CHECK(miopenT5LayerNormForward(
      handle, MIOPEN_ELEMENTWISE_AFFINE_T5, c->xDesc, input, c->weightDesc,
      scale, epsilon, c->yDesc, output, c->rstdDesc, rstd_buf));

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_miopenT5LayerNormForward: completed successfully\n");
  return 0;

cleanup:
  return result;
}
