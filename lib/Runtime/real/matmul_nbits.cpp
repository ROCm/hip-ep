/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../module_registry.h"
#include "../op_profile.h"
#include "../runtime_state_internal.h"
#include "error_check_macros.h"
#include "hip_custom_kernels.h"
#include "zp_unpack_cache.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <mutex>
#include <unordered_map>
#include <utility>

#define HIP_CHECK(cmd) HIP_CHECK_GOTO(cmd, cleanup)

// ---------------------------------------------------------------------------
// Asym MatMulNBits zero_points unpack cache.
//
// For each unique zero_points input pointer (which is stable for the lifetime
// of the model.dll — the pointer comes from the constants blob), cache the
// unpacked uint8 buffer used by GEMV/naive paths and the converted fp16
// buffer used by the WMMA / col-major-GEMV (M>1) paths. This avoids the
// per-call unpack/convert kernel launches that were the dominant per-call
// overhead for asym 8B decode (~225 launches per Compute()).
//
// Lifecycle: registered as the "zp_unpack" op-module. The single slot is
// shared between wrap_matmul_nbits (this TU) and wrap_qmoe (qmoe.cpp), both
// reaching it through the lookup_or_*_zp_* helpers below. Created lazily on
// first call; freed via ZpUnpackState::~ZpUnpackState from
// module_registry_destroy at session cleanup.
// ---------------------------------------------------------------------------

namespace hipdnn_ep_real {

struct ZpUnpackState {
  // Map keyed on zero_points GPU pointer. Value = (device buffer, byte size).
  std::unordered_map<const void *, std::pair<void *, size_t>> u8;
  std::unordered_map<const void *, std::pair<void *, size_t>> fp16;
  // Concurrent inferences on the same RuntimeState don't run today (single
  // stream, sequential Compute() calls) but the lock is cheap and lets us
  // remain correct if that ever changes.
  std::mutex mu;

  // The op-module spec uses make_op_module_spec<ZpUnpackState>, which expects
  // a (RuntimeState*) constructor. The runtime state itself isn't needed --
  // all per-call info arrives through lookup_or_*_zp_*'s arguments.
  explicit ZpUnpackState(RuntimeState *) {}

  // RAII teardown: free every cached device buffer. The shared cleanup path
  // synchronizes the stream before tearing down the ModuleRegistry, so any
  // in-flight unpack/convert kernel has finished by the time we free.
  ~ZpUnpackState() {
    for (auto &kv : u8)
      hipFree(kv.second.first);
    for (auto &kv : fp16)
      hipFree(kv.second.first);
  }

  // HIPDNN_EP_DUMP_STATE hook: report combined cached-unpacked-buffer
  // footprint. Stable across inferences once warmup has populated every
  // unique zero_points pointer in the model.
  size_t mem_bytes() const {
    size_t total = 0;
    for (auto &kv : u8)
      total += kv.second.second;
    for (auto &kv : fp16)
      total += kv.second.second;
    return total;
  }
};

} // namespace hipdnn_ep_real

namespace {
// Slot accessor for the asym zero_points unpack cache. Single slot shared
// between matmul_nbits and qmoe paths -- both wrap_* entry points reach this
// state through the lookup_or_*_zp_* helpers below.
HIPDNN_OP_MODULE(zp_unpack_module, "zp_unpack", hipdnn_ep_real::ZpUnpackState);
} // namespace

namespace hipdnn_ep_real {

// Returns the cached u8 buffer for `zp_packed`, or unpacks into a freshly
// allocated buffer on miss. Returns nullptr only on hipMalloc failure.
const void *lookup_or_unpack_zp_u8(RuntimeState *state, void *stream,
                                   const void *zp_packed, int N, int groups_k) {
  ZpUnpackState *st = zp_unpack_module(state);
  if (!st) {
    fprintf(stderr, "matmul_nbits: failed to obtain ZpUnpackState\n");
    return nullptr;
  }
  const size_t need = static_cast<size_t>(N) * static_cast<size_t>(groups_k);

  std::lock_guard<std::mutex> lock(st->mu);
  auto it = st->u8.find(zp_packed);
  if (it != st->u8.end() && it->second.second >= need)
    return it->second.first;

  // Miss (or cached buffer too small for an unexpected re-shape on the same
  // pointer — shouldn't happen for stable model constants, but guard it).
  void *dst = nullptr;
  if (hipMalloc(&dst, need) != hipSuccess) {
    fprintf(stderr, "matmul_nbits: hipMalloc(%zu) for zp_u8 cache failed\n",
            need);
    return nullptr;
  }
  hip_matmul_nbits_unpack_zp_u8(stream, zp_packed, dst, N, groups_k);

  if (it != st->u8.end()) {
    // Replace the undersized entry. Free the stale buffer.
    hipFree(it->second.first);
    it->second = {dst, need};
  } else {
    st->u8.emplace(zp_packed, std::make_pair(dst, need));
  }
  return dst;
}

const void *lookup_or_convert_zp_fp16(RuntimeState *state, void *stream,
                                      const void *zp_packed, int N,
                                      int groups_k) {
  ZpUnpackState *st = zp_unpack_module(state);
  if (!st) {
    fprintf(stderr, "matmul_nbits: failed to obtain ZpUnpackState\n");
    return nullptr;
  }
  const size_t need =
      static_cast<size_t>(N) * static_cast<size_t>(groups_k) * sizeof(__fp16);

  std::lock_guard<std::mutex> lock(st->mu);
  auto it = st->fp16.find(zp_packed);
  if (it != st->fp16.end() && it->second.second >= need)
    return it->second.first;

  void *dst = nullptr;
  if (hipMalloc(&dst, need) != hipSuccess) {
    fprintf(stderr, "matmul_nbits: hipMalloc(%zu) for zp_fp16 cache failed\n",
            need);
    return nullptr;
  }
  hip_matmul_nbits_convert_zp_fp16(stream, zp_packed, dst, N, groups_k);

  if (it != st->fp16.end()) {
    hipFree(it->second.first);
    it->second = {dst, need};
  } else {
    st->fp16.emplace(zp_packed, std::make_pair(dst, need));
  }
  return dst;
}

} // namespace hipdnn_ep_real

int wrap_matmul_nbits(RuntimeState *state, const void *A, const void *B,
                      const void *scales, const void *zero_points,
                      const void *g_idx, const void *bias, void *output,
                      int64_t M, int64_t N, int64_t K, int64_t batch_count,
                      int64_t bits, int64_t block_size, int64_t elem_size,
                      int64_t zp_elem_size) {
  OP_PROFILE(
      "matmul_nbits",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "m=%lld,n=%lld,k=%lld", (long long)M,
                 (long long)N, (long long)K);
        return std::string(b);
      },
      state);
  if (!state || !A || !B || !scales || !output) {
    fprintf(stderr, "wrap_matmul_nbits: null argument\n");
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_matmul_nbits(M=%lld, N=%lld, K=%lld, "
                    "batch=%lld, bits=%lld, block_size=%lld, elem_size=%lld, "
                    "zp_elem_size=%lld, zero_points=%s, g_idx=%s, bias=%s)\n",
                    (long long)M, (long long)N, (long long)K,
                    (long long)batch_count, (long long)bits,
                    (long long)block_size, (long long)elem_size,
                    (long long)zp_elem_size, zero_points ? "yes" : "null",
                    g_idx ? "yes" : "null", bias ? "yes" : "null");

  void *stream = hipdnn_ep_state_get_stream(state);
  if (!stream) {
    fprintf(stderr, "wrap_matmul_nbits: null stream\n");
    return -1;
  }

  if (g_idx) {
    fprintf(stderr, "wrap_matmul_nbits: g_idx not supported\n");
    return -1;
  }

  // Pre-unpack zero_points (asym path) using the per-state pointer-keyed
  // cache. The kernel itself no longer launches its own unpack/convert.
  const void *pre_zp_u8 = nullptr;
  const void *pre_zp_fp16 = nullptr;
  if (zero_points && zp_elem_size == 1 && bits == 4 && block_size > 0) {
    int ngk = static_cast<int>((K + block_size - 1) / block_size);
    pre_zp_u8 = hipdnn_ep_real::lookup_or_unpack_zp_u8(
        state, stream, zero_points, static_cast<int>(N), ngk);
    if (!pre_zp_u8)
      return -1;
    // The fp16 buffer is consumed only by WMMA (batch==1 && K%32==0 && M>=16)
    // and the col-major GEMV M>1 fallback (same predicate on K, M>1). Build
    // it eagerly when those preconditions are met — the cache makes the cost
    // a one-time hit per zero_points pointer.
    bool wmma_data_format = (batch_count == 1) && (K % 32 == 0);
    if (wmma_data_format && M > 1) {
      pre_zp_fp16 = hipdnn_ep_real::lookup_or_convert_zp_fp16(
          state, stream, zero_points, static_cast<int>(N), ngk);
      if (!pre_zp_fp16)
        return -1;
    }
  }

  int result = 0;
  HIP_CHECK(hip_matmul_nbits(stream, A, B, scales, zero_points, bias, output, M,
                             N, K, batch_count, bits, block_size, elem_size,
                             zp_elem_size, pre_zp_u8, pre_zp_fp16));

cleanup:
  return result;
}
