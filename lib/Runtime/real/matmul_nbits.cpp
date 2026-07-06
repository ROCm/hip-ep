/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "../op_state.h"
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
// Asym MatMulNBits zero_points unpack cache (ZpUnpackCache).
//
// For each unique zero_points input pointer (which is stable for the
// lifetime of the JITted per-model session — the pointer comes from the
// constants blob), cache the
// unpacked uint8 buffer used by GEMV/naive paths and the converted fp16
// buffer used by the WMMA / col-major-GEMV (M>1) paths. This avoids the
// per-call unpack/convert kernel launches that were the dominant per-call
// overhead for asym 8B decode (~225 launches per Compute()).
//
// Ownership (see docs/design/op-state-slots-design.md): matmul_nbits owns a
// per-op-instance cache in its MatmulNbitsState op-state slot, while qmoe uses
// the per-session RuntimeState::zp_unpack_cache (reached via
// get_or_create_zp_cache below). The struct + lookup helpers are defined here
// (HIP lives here); the struct definition + helper decls are in
// zp_unpack_cache.h so qmoe can reach a cache too.
// ---------------------------------------------------------------------------

namespace hipdnn_ep_real {

// Out-of-line so zp_unpack_cache.h needs no HIP. Frees every cached device
// buffer when the owning cache (op-state slot or RuntimeState field) is torn
// down.
ZpUnpackCache::~ZpUnpackCache() {
  for (auto &[k, v] : u8)
    hipFree(v.first);
  for (auto &[k, v] : fp16)
    hipFree(v.first);
}

// Lazily create the per-session ZpUnpackCache owned by RuntimeState (used by
// wrap_qmoe). matmul_nbits itself uses a per-instance cache in its op-state
// slot; this RuntimeState-owned cache is qmoe's home for the same data.
ZpUnpackCache *get_or_create_zp_cache(RuntimeState *state) {
  if (!state->zp_unpack_cache)
    state->zp_unpack_cache = new ZpUnpackCache();
  return static_cast<ZpUnpackCache *>(state->zp_unpack_cache);
}

// Returns the cached u8 buffer for `zp_packed`, or unpacks into a freshly
// allocated buffer on miss. Returns nullptr only on hipMalloc failure.
const void *lookup_or_unpack_zp_u8(ZpUnpackCache &cache, void *stream,
                                   const void *zp_packed, int N, int groups_k) {
  const size_t need = static_cast<size_t>(N) * static_cast<size_t>(groups_k);

  std::lock_guard<std::mutex> lock(cache.mu);
  auto it = cache.u8.find(zp_packed);
  if (it != cache.u8.end() && it->second.second >= need)
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

  if (it != cache.u8.end()) {
    // Replace the undersized entry. Free the stale buffer.
    hipFree(it->second.first);
    it->second = {dst, need};
  } else {
    cache.u8.emplace(zp_packed, std::make_pair(dst, need));
  }
  return dst;
}

const void *lookup_or_convert_zp_fp16(ZpUnpackCache &cache, void *stream,
                                      const void *zp_packed, int N,
                                      int groups_k) {
  const size_t need =
      static_cast<size_t>(N) * static_cast<size_t>(groups_k) * sizeof(__fp16);

  std::lock_guard<std::mutex> lock(cache.mu);
  auto it = cache.fp16.find(zp_packed);
  if (it != cache.fp16.end() && it->second.second >= need)
    return it->second.first;

  void *dst = nullptr;
  if (hipMalloc(&dst, need) != hipSuccess) {
    fprintf(stderr, "matmul_nbits: hipMalloc(%zu) for zp_fp16 cache failed\n",
            need);
    return nullptr;
  }
  hip_matmul_nbits_convert_zp_fp16(stream, zp_packed, dst, N, groups_k);

  if (it != cache.fp16.end()) {
    hipFree(it->second.first);
    it->second = {dst, need};
  } else {
    cache.fp16.emplace(zp_packed, std::make_pair(dst, need));
  }
  return dst;
}

} // namespace hipdnn_ep_real

// Teardown shim for the qmoe-owned RuntimeState::zp_unpack_cache. Called from
// hipdnn_ep_state_cleanup; delete invokes ~ZpUnpackCache which hipFree's every
// cached device buffer.
extern "C" void hipdnn_ep_zp_unpack_cache_destroy(void *cache_ptr) {
  delete static_cast<hipdnn_ep_real::ZpUnpackCache *>(cache_ptr);
}

// Per-instance MatMulNBits op-state (see docs/design/op-state-slots-design.md):
// owns this instance's zero_points unpack cache. Replaces the former shared
// RuntimeState::zp_unpack_cache, so concurrent matmul_nbits sessions no longer
// share it.
struct MatmulNbitsState : OpStateT<MatmulNbitsState> {
  hipdnn_ep_real::ZpUnpackCache zp;
};

extern "C" int8_t hipdnn_ep_op_state_construct_matmul_nbits(RuntimeState *state,
                                                            int32_t slot) {
  hipdnn_ep_op_state_set(state, slot, MatmulNbitsState::create().release());
  return 0;
}

// ---------------------------------------------------------------------------
// Fused shared-expert gate+up MatMulNBits.
//
// The shared-expert MLP issues gate_proj and up_proj as TWO separate int4
// MatMulNBits at N=inter (e.g. 512) reading the same activation. Small N =
// grid-under-utilized = ~4% of peak BW (measured). This entry merges them into
// ONE matmul at N=2*inter by concatenating the two weight/scale/zp/bias blobs
// along N ONCE (cached, keyed by the gate weight pointer -- stable model
// constant) into persistent device buffers, then calling the existing autotuned
// hip_matmul_nbits at the larger, more efficient N. Output is [M, 2*inter];
// the graph slices it back to gate=[:,:inter], up=[:,inter:] for SwiGLU.
//
// Concatenation is a pure dim-0 (row) append for the row-major [N, kb, blob]
// layout -> just two D2D copies per blob. One-time; amortized over the session.
// ---------------------------------------------------------------------------
namespace {
struct FusedGateUp {
  void *B = nullptr, *S = nullptr, *Z = nullptr, *bias = nullptr;
  hipdnn_ep_real::ZpUnpackCache zpc; // pre-unpack cache for the fused zp
};
std::mutex g_gateup_mu;
std::unordered_map<const void *, FusedGateUp *> g_gateup_cache; // key: gate B
} // namespace

extern "C" int wrap_matmul_nbits_gateup(
    RuntimeState *state, const void *A, const void *gate_B,
    const void *gate_scales, const void *gate_zp, const void *gate_bias,
    const void *up_B, const void *up_scales, const void *up_zp,
    const void *up_bias, void *output, int64_t M, int64_t N_half, int64_t K,
    int64_t bits, int64_t block_size, int64_t elem_size, int64_t zp_elem_size) {
  OP_PROFILE_BYTES(
      "matmul_nbits_gateup",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "m=%lld,n=%lld,k=%lld", (long long)M,
                 (long long)(2 * N_half), (long long)K);
        return std::string(b);
      },
      [&] {
        int64_t es = elem_size > 0 ? elem_size : 2;
        int64_t kb = block_size > 0 ? (K + block_size - 1) / block_size : 0;
        int64_t w = 2 * N_half * K * bits / 8;
        int64_t s = 2 * N_half * kb * es;
        return w + s + M * K * es + M * 2 * N_half * es;
      },
      state);
  if (!state || !A || !gate_B || !gate_scales || !up_B || !up_scales ||
      !output) {
    fprintf(stderr, "wrap_matmul_nbits_gateup: null argument\n");
    return -1;
  }
  void *stream = hipdnn_ep_state_get_stream(state);
  if (!stream)
    return -1;

  const int64_t kb = (K + block_size - 1) / block_size;
  const int64_t blob = block_size * bits / 8;
  const size_t bBytes = static_cast<size_t>(N_half) * kb * blob;
  const size_t sBytes = static_cast<size_t>(N_half) * kb * elem_size;
  const bool has_zp = (gate_zp != nullptr && up_zp != nullptr);
  const size_t zBytes =
      has_zp ? (zp_elem_size == 1
                    ? static_cast<size_t>(N_half) * ((kb + 1) / 2)
                    : static_cast<size_t>(N_half) * kb * 2)
             : 0;
  const bool has_bias = (gate_bias != nullptr && up_bias != nullptr);
  const size_t biasBytes =
      has_bias ? static_cast<size_t>(N_half) * elem_size : 0;

  // Look up / build the fused (concatenated) weights, keyed by gate_B.
  FusedGateUp *fg = nullptr;
  {
    std::lock_guard<std::mutex> lk(g_gateup_mu);
    auto it = g_gateup_cache.find(gate_B);
    if (it != g_gateup_cache.end()) {
      fg = it->second;
    } else {
      fg = new FusedGateUp();
      auto d2d = [&](void *dst, const void *src, size_t n) {
        (void)hipMemcpyAsync(dst, src, n, hipMemcpyDeviceToDevice,
                             static_cast<hipStream_t>(stream));
      };
      bool ok = true;
      ok &= (hipMalloc(&fg->B, 2 * bBytes) == hipSuccess);
      ok &= (hipMalloc(&fg->S, 2 * sBytes) == hipSuccess);
      if (has_zp)
        ok &= (hipMalloc(&fg->Z, 2 * zBytes) == hipSuccess);
      if (has_bias)
        ok &= (hipMalloc(&fg->bias, 2 * biasBytes) == hipSuccess);
      if (!ok) {
        fprintf(stderr, "wrap_matmul_nbits_gateup: hipMalloc failed\n");
        return -1;
      }
      d2d(fg->B, gate_B, bBytes);
      d2d(static_cast<char *>(fg->B) + bBytes, up_B, bBytes);
      d2d(fg->S, gate_scales, sBytes);
      d2d(static_cast<char *>(fg->S) + sBytes, up_scales, sBytes);
      if (has_zp) {
        d2d(fg->Z, gate_zp, zBytes);
        d2d(static_cast<char *>(fg->Z) + zBytes, up_zp, zBytes);
      }
      if (has_bias) {
        d2d(fg->bias, gate_bias, biasBytes);
        d2d(static_cast<char *>(fg->bias) + biasBytes, up_bias, biasBytes);
      }
      g_gateup_cache.emplace(gate_B, fg);
    }
  }

  const int64_t N = 2 * N_half;
  const void *pre_zp_u8 = nullptr, *pre_zp_fp16 = nullptr;
  if (fg->Z && zp_elem_size == 1 && bits == 4 && block_size > 0) {
    int ngk = static_cast<int>(kb);
    pre_zp_u8 = hipdnn_ep_real::lookup_or_unpack_zp_u8(
        fg->zpc, stream, fg->Z, static_cast<int>(N), ngk);
    if (!pre_zp_u8)
      return -1;
    if ((K % 32 == 0) && M > 1) {
      pre_zp_fp16 = hipdnn_ep_real::lookup_or_convert_zp_fp16(
          fg->zpc, stream, fg->Z, static_cast<int>(N), ngk);
      if (!pre_zp_fp16)
        return -1;
    }
  }

  int result = 0;
  HIP_CHECK(hip_matmul_nbits(stream, A, fg->B, fg->S, fg->Z, fg->bias, output, M,
                             N, K, 1, bits, block_size, elem_size, zp_elem_size,
                             pre_zp_u8, pre_zp_fp16));
cleanup:
  return result;
}

int wrap_matmul_nbits(RuntimeState *state, int op_state_slot, const void *A,
                      const void *B, const void *scales,
                      const void *zero_points, const void *g_idx,
                      const void *bias, void *output, int64_t M, int64_t N,
                      int64_t K, int64_t batch_count, int64_t bits,
                      int64_t block_size, int64_t elem_size,
                      int64_t zp_elem_size) {
  OP_PROFILE_BYTES(
      "matmul_nbits",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "m=%lld,n=%lld,k=%lld", (long long)M,
                 (long long)N, (long long)K);
        return std::string(b);
      },
      [&] {
        // GEMV footprint: quantized weights + scales + activations + output.
        int64_t es = elem_size > 0 ? elem_size : 2;
        int64_t weights = N * K * bits / 8;
        int64_t scales =
            block_size > 0 ? N * ((K + block_size - 1) / block_size) * es : 0;
        int64_t act = M * K * es;
        int64_t out = M * N * es;
        return (weights + scales + act + out) * (batch_count > 0 ? batch_count : 1);
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

  // Pre-unpack zero_points (asym path) using this instance's pointer-keyed
  // cache (owned by its op-state slot). The kernel itself no longer launches
  // its own unpack/convert.
  const void *pre_zp_u8 = nullptr;
  const void *pre_zp_fp16 = nullptr;
  if (zero_points && zp_elem_size == 1 && bits == 4 && block_size > 0) {
    MatmulNbitsState *mst =
        MatmulNbitsState::get_op_state(state, op_state_slot);
    if (!mst) {
      fprintf(stderr, "wrap_matmul_nbits: no MatmulNbitsState at slot %d\n",
              op_state_slot);
      return -1;
    }
    int ngk = static_cast<int>((K + block_size - 1) / block_size);
    pre_zp_u8 = hipdnn_ep_real::lookup_or_unpack_zp_u8(
        mst->zp, stream, zero_points, static_cast<int>(N), ngk);
    if (!pre_zp_u8)
      return -1;
    // The fp16 buffer is consumed only by WMMA (batch==1 && K%32==0 && M>=16)
    // and the col-major GEMV M>1 fallback (same predicate on K, M>1). Build
    // it eagerly when those preconditions are met — the cache makes the cost
    // a one-time hit per zero_points pointer.
    bool wmma_data_format = (batch_count == 1) && (K % 32 == 0);
    if (wmma_data_format && M > 1) {
      pre_zp_fp16 = hipdnn_ep_real::lookup_or_convert_zp_fp16(
          mst->zp, stream, zero_points, static_cast<int>(N), ngk);
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
