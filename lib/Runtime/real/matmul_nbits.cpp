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

// bits=2 variant: unpacks the 4-per-byte packed 2-bit zero_points stream to
// one uint8 per group. Same pointer-keyed cache as the nibble path — a given
// zero_points pointer is either 2-bit or 4-bit packed for the life of the
// session, so keying by pointer keeps the two unambiguous.
const void *lookup_or_unpack_zp_u8_2bit(ZpUnpackCache &cache, void *stream,
                                        const void *zp_packed, int N,
                                        int groups_k) {
  const size_t need = static_cast<size_t>(N) * static_cast<size_t>(groups_k);

  std::lock_guard<std::mutex> lock(cache.mu);
  auto it = cache.u8.find(zp_packed);
  if (it != cache.u8.end() && it->second.second >= need)
    return it->second.first;

  void *dst = nullptr;
  if (hipMalloc(&dst, need) != hipSuccess) {
    fprintf(stderr, "matmul_nbits: hipMalloc(%zu) for zp_u8 (2-bit) cache "
                    "failed\n",
            need);
    return nullptr;
  }
  hip_matmul_nbits_unpack_zp_u8_2bit(stream, zp_packed, dst, N, groups_k);

  if (it != cache.u8.end()) {
    hipFree(it->second.first);
    it->second = {dst, need};
  } else {
    cache.u8.emplace(zp_packed, std::make_pair(dst, need));
  }
  return dst;
}

// bits=3 variant: unpacks the continuous per-row 3-bit packed zero_points
// stream to one uint8 per group. Shares the pointer-keyed cache with the
// nibble/2-bit paths (a given zero_points pointer has a single packing for
// the life of the session).
const void *lookup_or_unpack_zp_u8_3bit(ZpUnpackCache &cache, void *stream,
                                        const void *zp_packed, int N,
                                        int groups_k) {
  const size_t need = static_cast<size_t>(N) * static_cast<size_t>(groups_k);

  std::lock_guard<std::mutex> lock(cache.mu);
  auto it = cache.u8.find(zp_packed);
  if (it != cache.u8.end() && it->second.second >= need)
    return it->second.first;

  void *dst = nullptr;
  if (hipMalloc(&dst, need) != hipSuccess) {
    fprintf(stderr, "matmul_nbits: hipMalloc(%zu) for zp_u8 (3-bit) cache "
                    "failed\n",
            need);
    return nullptr;
  }
  hip_matmul_nbits_unpack_zp_u8_3bit(stream, zp_packed, dst, N, groups_k);

  if (it != cache.u8.end()) {
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

int wrap_matmul_nbits(RuntimeState *state, int op_state_slot, const void *A,
                      const void *B, const void *scales,
                      const void *zero_points, const void *g_idx,
                      const void *bias, void *output, int64_t M, int64_t N,
                      int64_t K, int64_t batch_count, int64_t bits,
                      int64_t block_size, int64_t elem_size,
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

  // Pre-unpack zero_points (asym path) using this instance's pointer-keyed
  // cache (owned by its op-state slot). The kernel itself no longer launches
  // its own unpack/convert.
  const void *pre_zp_u8 = nullptr;
  const void *pre_zp_fp16 = nullptr;
  if (zero_points && zp_elem_size == 1 &&
      (bits == 4 || bits == 3 || bits == 2) && block_size > 0) {
    MatmulNbitsState *mst =
        MatmulNbitsState::get_op_state(state, op_state_slot);
    if (!mst) {
      fprintf(stderr, "wrap_matmul_nbits: no MatmulNbitsState at slot %d\n",
              op_state_slot);
      return -1;
    }
    int ngk = static_cast<int>((K + block_size - 1) / block_size);
    // ONNX MatMulNBits packs zero_points at `bits` bits: 2-per-byte nibbles
    // for bits=4, 4-per-byte for bits=2, and a continuous per-row 3-bit
    // stream for bits=3. Unpack to one-byte-per-group so the kernel's
    // GEMV/naive/WMMA paths can index zp[n*ngk + grp] directly.
    pre_zp_u8 =
        (bits == 2)
            ? hipdnn_ep_real::lookup_or_unpack_zp_u8_2bit(
                  mst->zp, stream, zero_points, static_cast<int>(N), ngk)
        : (bits == 3)
            ? hipdnn_ep_real::lookup_or_unpack_zp_u8_3bit(
                  mst->zp, stream, zero_points, static_cast<int>(N), ngk)
            : hipdnn_ep_real::lookup_or_unpack_zp_u8(
                  mst->zp, stream, zero_points, static_cast<int>(N), ngk);
    if (!pre_zp_u8)
      return -1;
    // The fp16 buffer is consumed only by the bits=4 WMMA (batch==1 &&
    // K%32==0 && M>=16) and col-major GEMV M>1 fallback (same predicate on K,
    // M>1). Build it eagerly when those preconditions are met — the cache
    // makes the cost a one-time hit per zero_points pointer. The u2 WMMA path
    // consumes uint8 zp directly, so bits=2 never needs the fp16 buffer (and
    // the converter is nibble-specific anyway).
    bool wmma_data_format = (batch_count == 1) && (K % 32 == 0);
    if (bits == 4 && wmma_data_format && M > 1) {
      pre_zp_fp16 = hipdnn_ep_real::lookup_or_convert_zp_fp16(
          mst->zp, stream, zero_points, static_cast<int>(N), ngk);
      if (!pre_zp_fp16)
        return -1;
    }
  }

  int result = 0;

  // Opt-in W4A8 dp4a fast path for single-row decode GEMV. Eligibility:
  //   - HIPDNN_EP_MATMUL_DP4A=1
  //   - M==1 && batch_count==1 (single decode token)
  //   - bits==4, K%32==0, block_size>0, elem_size==2 (fp16 activation)
  //   - symmetric (no zero_points) OR asym with pre_zp_u8 already unpacked
  //   above
  // The kernel dynamically quantizes the activation row into per-session
  // scratch (a_qb: K int8 bytes, a_scale: ceil(K/block_size) floats) sized once
  // and grown on demand, then runs an integer-dot GEMV. Quantizing once (here)
  // and reusing the compact int8 across all N/TILE_N GEMV blocks is measurably
  // faster than fusing the quant into the GEMV (which would re-read fp16 A +
  // re-convert in every block); the extra quant launch it saves is dwarfed by
  // that redundant work. All other shapes fall through to hip_matmul_nbits.
  if (hipdnn_ep_matmul_dp4a_enabled() && M == 1 && batch_count == 1 &&
      bits == 4 && block_size > 0 && (K % 32 == 0) && elem_size == 2 &&
      (!zero_points || (zp_elem_size == 1 && pre_zp_u8))) {
    const size_t k_blocks =
        static_cast<size_t>((K + block_size - 1) / block_size);
    // a_qb at offset 0 (K bytes, hipMalloc base is over-aligned); a_scale after
    // a 64-byte-rounded gap so its float reads stay naturally aligned.
    const size_t a_qb_bytes = static_cast<size_t>(K);
    const size_t scale_off = (a_qb_bytes + 63) & ~static_cast<size_t>(63);
    const size_t need = scale_off + k_blocks * sizeof(float);
    if (hipdnn_ep_state_ensure_matmul_dp4a_scratch(state, need) == 0) {
      char *base =
          static_cast<char *>(hipdnn_ep_state_get_matmul_dp4a_scratch(state));
      if (base) {
        void *a_qb = base;
        void *a_scale = base + scale_off;
        int rc = hip_matmul_nbits_dp4a(stream, A, B, scales, pre_zp_u8, bias,
                                       output, N, K, block_size, a_qb, a_scale);
        if (rc == 0) {
          result = 0;
          goto cleanup;
        }
        // Non-zero rc (e.g. unsupported geometry): fall through to fp path.
        RUNTIME_DEBUG_LOG(
            "[REAL] matmul_nbits dp4a rc=%d, falling back to fp GEMV\n", rc);
      }
    }
  }

  HIP_CHECK(hip_matmul_nbits(stream, A, B, scales, zero_points, bias, output, M,
                             N, K, batch_count, bits, block_size, elem_size,
                             zp_elem_size, pre_zp_u8, pre_zp_fp16));

cleanup:
  return result;
}
