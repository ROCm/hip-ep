/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "../op_state.h"
#include "cache_utils.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <unordered_map>

//===----------------------------------------------------------------------===//
// Per-shape descriptor cache with multi-algorithm auto-tune.
//
// On the first call for each unique (M, N, K, batch, elem_size) shape we
// create descriptors, request up to MAX_ALGO_CANDIDATES algorithms from the
// heuristic, and store them. On the first actual matmul call (when real GPU
// pointers are available), we benchmark each candidate and cache the winner.
// Subsequent calls reuse the tuned algorithm with zero overhead.
//
// Auto-tune is enabled by default. Set HIPDNN_EP_AUTOTUNE=0 to disable.
//===----------------------------------------------------------------------===//

static constexpr int MAX_ALGO_CANDIDATES = 60;
static constexpr int AUTOTUNE_TIMING_ITERS = 3;

static bool autotune_enabled() {
  static const bool enabled = [] {
    const char *env = std::getenv("HIPDNN_EP_AUTOTUNE");
    return env && std::strcmp(env, "0") != 0;
  }();
  return enabled;
}

struct MatmulCacheKey {
  int64_t M, N, K, batch_count, elem_size;
  // hipBLASLt's STRIDED_BATCH_OFFSET on layA, in elements. Two distinct
  // values reach this site at the same (M,N,K,batch,elem_size):
  //   * 0   — B is a broadcast weight (rank-2 [K,N], or rank-N
  //           [1,...,1,K,N] whose leading-dim product is 1).
  //   * K*N — B is per-batch (leading-dim product > 1; the buffer holds
  //           multiple [K,N] matrices laid out contiguously).
  // Part of the cache key because the layout descriptor is parameterised
  // by the stride: mixing the two would silently route one path through
  // the other's stride and read past the end of a broadcast weight buffer.
  // Keyed on the actual int stride (not a 0/1 bool) so any future site
  // that legitimately uses a stride other than {0, K*N} also gets its
  // own cache entry rather than aliasing one of these two.
  int64_t b_batch_stride;
  bool operator==(const MatmulCacheKey &o) const {
    return M == o.M && N == o.N && K == o.K && batch_count == o.batch_count &&
           elem_size == o.elem_size && b_batch_stride == o.b_batch_stride;
  }
};

struct MatmulCacheKeyHash {
  size_t operator()(const MatmulCacheKey &k) const {
    size_t h = 0;
    hash_combine_val(h, k.M);
    hash_combine_val(h, k.N);
    hash_combine_val(h, k.K);
    hash_combine_val(h, k.batch_count);
    hash_combine_val(h, k.elem_size);
    hash_combine_val(h, k.b_batch_stride);
    return h;
  }
};

/// Cached hipBLASLt descriptors + multi-algorithm auto-tune state for a
/// single (M, N, K, batch, elem_size) shape. Descriptors are created in
/// queryOrCreateMatmul() and owned by the MatmulAlgoTable, which frees them
/// when the last session sharing it is destroyed.
struct MatmulCacheEntry {
  hipblasLtMatmulDesc_t desc;
  hipblasLtMatrixLayout_t layA, layB, layC;
  hipblasLtMatmulAlgo_t algo;
  size_t workspace_size;
  // Set true (with release ordering) once autotune has chosen `algo`. Atomic so
  // sessions sharing this entry can read it lock-free on the steady-state path.
  std::atomic<bool> tuned;
  int num_candidates;
  size_t max_candidate_workspace;
  // True if no heuristic algorithm could be found and we should fall back to
  // hipblasLtMatmul(..., algo=nullptr, ws=nullptr, ws_size=0). Lets gfx1151
  // outliers (e.g. lm_head M=128 N=201088 K=2880) use hipBLASLt's internal
  // default kernel when its Tensile library has no matching tile.
  bool use_default_algo;
  hipblasLtMatmulHeuristicResult_t candidates[MAX_ALGO_CANDIDATES];
  // Serialises the one-time autotune of this entry across all sessions sharing
  // it; pairs with the atomic `tuned` for a lock-free steady state.
  std::mutex tune_mu;
};

// One hipBLASLt algo table per device, shared across every session in the
// process and freed when the last session holding it is destroyed. The table
// owns the descriptors and frees them in its destructor, fixing the previous
// process-lifetime leak. Entries are stored by pointer so they keep a stable
// address (raw MatmulCacheEntry* handed out below) and can hold a non-movable
// std::once_flag.
struct MatmulAlgoTable {
  std::mutex mu;
  std::unordered_map<MatmulCacheKey, std::unique_ptr<MatmulCacheEntry>,
                     MatmulCacheKeyHash>
      map;
  ~MatmulAlgoTable() {
    for (auto &kv : map) {
      MatmulCacheEntry &e = *kv.second;
      if (e.layC)
        hipblasLtMatrixLayoutDestroy(e.layC);
      if (e.layB)
        hipblasLtMatrixLayoutDestroy(e.layB);
      if (e.layA)
        hipblasLtMatrixLayoutDestroy(e.layA);
      if (e.desc)
        hipblasLtMatmulDescDestroy(e.desc);
    }
  }
};

// Per-instance op state: each hip.matmul slot holds a shared_ptr to its
// device's shared table, keeping it alive for the session's lifetime. The table
// is reached through a global WeakStore keyed by device (see op_state.h); it is
// weak_ptr-backed, so it lives only while some session's MatmulState holds a
// shared_ptr to it.
struct MatmulState : OpStateT<MatmulState> {
  std::shared_ptr<MatmulAlgoTable> table;
  MatmulState() {
    int dev = 0;
    hipGetDevice(&dev);
    table = WeakStore<int, MatmulAlgoTable>::get_or_create(
        dev, [] { return std::make_shared<MatmulAlgoTable>(); });
  }
};

extern "C" int8_t hipdnn_ep_op_state_construct_matmul(RuntimeState *state,
                                                      int32_t slot) {
  hipdnn_ep_op_state_set(state, slot, MatmulState::create().release());
  return 0;
}

static MatmulCacheEntry *queryOrCreateMatmul(MatmulAlgoTable &table,
                                             hipblasLtHandle_t handle,
                                             const MatmulCacheKey &key) {
  assert(handle && "queryOrCreateMatmul: null handle");
  // The table is shared across sessions via WeakStore, so guard find/insert.
  // Entries are never erased, so the returned raw pointer stays valid for
  // unlocked read/tune after this returns.
  std::lock_guard<std::mutex> tableGuard(table.mu);
  auto it = table.map.find(key);
  if (it != table.map.end())
    return it->second.get();

  hipDataType dt = (key.elem_size == 2) ? HIP_R_16F : HIP_R_32F;
  int64_t M = key.M, N = key.N, K = key.K;

  auto entryPtr = std::make_unique<MatmulCacheEntry>();
  MatmulCacheEntry &entry = *entryPtr;
  entry.tuned = false;
  entry.num_candidates = 0;
  entry.max_candidate_workspace = 0;

  hipblasLtMatmulPreference_t pref = nullptr;
  hipblasStatus_t st;

#define MATMUL_CACHE_CHECK(call)                                               \
  do {                                                                         \
    st = (call);                                                               \
    if (st != HIPBLAS_STATUS_SUCCESS)                                          \
      goto cache_fail;                                                         \
  } while (0)

  MATMUL_CACHE_CHECK(
      hipblasLtMatmulDescCreate(&entry.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));

  {
    hipblasOperation_t opN = HIPBLAS_OP_N;
    MATMUL_CACHE_CHECK(hipblasLtMatmulDescSetAttribute(
        entry.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
    MATMUL_CACHE_CHECK(hipblasLtMatmulDescSetAttribute(
        entry.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
  }

  // Row-major -> col-major trick: BLAS sees m=N, k=K, n=M with ld = first dim
  MATMUL_CACHE_CHECK(hipblasLtMatrixLayoutCreate(&entry.layA, dt, N, K, N));
  MATMUL_CACHE_CHECK(hipblasLtMatrixLayoutCreate(&entry.layB, dt, K, M, K));
  MATMUL_CACHE_CHECK(hipblasLtMatrixLayoutCreate(&entry.layC, dt, N, M, N));

  if (key.batch_count > 1) {
    int64_t bc = key.batch_count;
    // layA → user's B. The stride is whatever the compiler computed —
    // 0 for broadcast B (rank-2 [K,N] or rank-N [1,...,1,K,N]), K*N for
    // per-batch B (rank-N with leading-dim product > 1). Setting sA = K*N
    // for a broadcast weight reads K*N elements PAST the end of the
    // buffer on batch 1+ and feeds garbage into the GEMM — typical symptom
    // on vision models is image-0 correct, image-1+ NaN (the OOB read
    // often lands in a fp16-NaN pattern from adjacent pool slots /
    // constants). Mis-setting sA = 0 for a per-batch B does the opposite:
    // every batch reads matrix 0 instead of its own.
    // layB → user's A and layC → output are always per-batch (the BATCH
    // partition comes from A's leading dim by construction).
    int64_t sA = key.b_batch_stride;
    int64_t sB = M * K, sC = M * N;
    MATMUL_CACHE_CHECK(hipblasLtMatrixLayoutSetAttribute(
        entry.layA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)));
    MATMUL_CACHE_CHECK(hipblasLtMatrixLayoutSetAttribute(
        entry.layA, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sA,
        sizeof(sA)));
    MATMUL_CACHE_CHECK(hipblasLtMatrixLayoutSetAttribute(
        entry.layB, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)));
    MATMUL_CACHE_CHECK(hipblasLtMatrixLayoutSetAttribute(
        entry.layB, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sB,
        sizeof(sB)));
    MATMUL_CACHE_CHECK(hipblasLtMatrixLayoutSetAttribute(
        entry.layC, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)));
    MATMUL_CACHE_CHECK(hipblasLtMatrixLayoutSetAttribute(
        entry.layC, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sC,
        sizeof(sC)));
  }

  // Algorithm selection with escalating fallback.
  //
  // hipblasLtMatmulAlgoGetHeuristic is sensitive to (workspace_limit,
  // request_count): for outlier shapes such as the gpt-oss-120b lm_head
  // (M=128 N=201088 K=2880, ld=N) it returns 0 algorithms / status 3 with
  // the default (256MB, 1) on gfx1151's Tensile library because no tiled
  // algorithm satisfies the leading-dimension constraint. Try a small
  // escalation ladder so common outlier shapes pick up a non-tiled or
  // larger-candidate-set algorithm:
  //   1. (256MB, requested)   original behaviour
  //   2. (0,     requested)   force non-workspace (often unblocks huge ld)
  //   3. (256MB, MAX)         broader candidate sweep
  //   4. (0,     MAX)         non-workspace + broader sweep
  {
    bool do_autotune = autotune_enabled();
    int request_count = do_autotune ? MAX_ALGO_CANDIDATES : 1;

    int returned = 0;
    auto try_heuristic = [&](size_t ws_bytes, int req_count) -> bool {
      hipblasLtMatmulPreference_t local_pref = nullptr;
      if (hipblasLtMatmulPreferenceCreate(&local_pref) !=
          HIPBLAS_STATUS_SUCCESS) {
        return false;
      }
      hipblasLtMatmulPreferenceSetAttribute(
          local_pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes,
          sizeof(ws_bytes));
      returned = 0;
      hipblasStatus_t s = hipblasLtMatmulAlgoGetHeuristic(
          handle, entry.desc, entry.layA, entry.layB, entry.layC, entry.layC,
          local_pref, req_count, entry.candidates, &returned);
      hipblasLtMatmulPreferenceDestroy(local_pref);
      return s == HIPBLAS_STATUS_SUCCESS && returned > 0;
    };

    bool ok = try_heuristic(kMaxWorkspaceBytes, request_count) ||
              try_heuristic(0, request_count) ||
              try_heuristic(kMaxWorkspaceBytes, MAX_ALGO_CANDIDATES) ||
              try_heuristic(0, MAX_ALGO_CANDIDATES);

    if (!ok) {
      // Final fallback: let hipBLASLt pick its internal default kernel by
      // passing algo=nullptr at call time. This keeps the cached layout/desc
      // so we don't pay create cost again, but avoids the heuristic on a
      // shape gfx1151's Tensile library has no tile for.
      fprintf(stderr,
              "queryOrCreateMatmul: no algo from heuristic for M=%lld "
              "N=%lld K=%lld batch=%lld; falling back to default algo "
              "(algo=nullptr, ws=0)\n",
              (long long)M, (long long)N, (long long)K,
              (long long)key.batch_count);
      entry.num_candidates = 0;
      entry.workspace_size = 0;
      entry.max_candidate_workspace = 0;
      entry.use_default_algo = true;
      entry.tuned = true;
    } else {
      entry.num_candidates = returned;
      entry.algo = entry.candidates[0].algo;
      entry.workspace_size = entry.candidates[0].workspaceSize;
      entry.use_default_algo = false;

      if (do_autotune) {
        for (int i = 0; i < returned; i++)
          entry.max_candidate_workspace = std::max(
              entry.max_candidate_workspace, entry.candidates[i].workspaceSize);
        entry.tuned = (returned <= 1);
      } else {
        entry.tuned = true;
      }
    }
  }

#undef MATMUL_CACHE_CHECK
  goto cache_done;

cache_fail:
  if (pref)
    hipblasLtMatmulPreferenceDestroy(pref);
  if (entry.layC)
    hipblasLtMatrixLayoutDestroy(entry.layC);
  if (entry.layB)
    hipblasLtMatrixLayoutDestroy(entry.layB);
  if (entry.layA)
    hipblasLtMatrixLayoutDestroy(entry.layA);
  if (entry.desc)
    hipblasLtMatmulDescDestroy(entry.desc);
  return nullptr;

cache_done:
  auto [ins, _] = table.map.emplace(key, std::move(entryPtr));

  RUNTIME_DEBUG_LOG("[MATMUL] cached M=%lld N=%lld K=%lld batch=%lld: "
                    "%d algo(s), autotune=%s\n",
                    (long long)M, (long long)N, (long long)K,
                    (long long)key.batch_count, entry.num_candidates,
                    entry.tuned ? "skipped" : "pending");

  return ins->second.get();
}

//===----------------------------------------------------------------------===//
// Auto-tune: benchmark all candidate algorithms and select the fastest.
// Called once per shape on the first matmul invocation with real GPU pointers.
//===----------------------------------------------------------------------===//

static void autotuneMatmul(hipblasLtHandle_t handle, hipStream_t stream,
                           MatmulCacheEntry *entry, const void *blas_A,
                           const void *blas_B, void *blas_C, void *ws_ptr,
                           size_t ws_size, const MatmulCacheKey &key) {
  float alpha = 1.0f, beta = 0.0f;
  hipEvent_t ev_start = nullptr, ev_stop = nullptr;
  if (hipEventCreate(&ev_start) != hipSuccess ||
      hipEventCreate(&ev_stop) != hipSuccess) {
    if (ev_start)
      hipEventDestroy(ev_start);
    fprintf(stderr, "[AUTOTUNE] WARNING: hipEventCreate failed, skipping\n");
    entry->tuned = true;
    return;
  }

  float best_ms = std::numeric_limits<float>::max();
  int best_idx = 0;
  int tested = 0;

  for (int i = 0; i < entry->num_candidates; i++) {
    auto &cand = entry->candidates[i];

    if (cand.workspaceSize > ws_size)
      continue;

    // Warm-up
    hipblasStatus_t st =
        hipblasLtMatmul(handle, entry->desc, &alpha, blas_A, entry->layA,
                        blas_B, entry->layB, &beta, blas_C, entry->layC, blas_C,
                        entry->layC, &cand.algo, ws_ptr, ws_size, stream);
    if (st != HIPBLAS_STATUS_SUCCESS)
      continue;

    // Timed iterations
    hipEventRecord(ev_start, stream);
    for (int t = 0; t < AUTOTUNE_TIMING_ITERS; t++) {
      hipblasLtMatmul(handle, entry->desc, &alpha, blas_A, entry->layA, blas_B,
                      entry->layB, &beta, blas_C, entry->layC, blas_C,
                      entry->layC, &cand.algo, ws_ptr, ws_size, stream);
    }
    hipEventRecord(ev_stop, stream);
    hipEventSynchronize(ev_stop);

    float ms = 0.0f;
    hipEventElapsedTime(&ms, ev_start, ev_stop);
    tested++;

    if (ms < best_ms) {
      best_ms = ms;
      best_idx = i;
    }
  }

  hipEventDestroy(ev_start);
  hipEventDestroy(ev_stop);

  if (tested == 0) {
    fprintf(stderr,
            "[AUTOTUNE] WARNING: M=%lld N=%lld K=%lld batch=%lld: "
            "0/%d candidates passed, keeping heuristic #0\n",
            (long long)key.M, (long long)key.N, (long long)key.K,
            (long long)key.batch_count, entry->num_candidates);
    return;
  }

  entry->algo = entry->candidates[best_idx].algo;
  entry->workspace_size = entry->candidates[best_idx].workspaceSize;
  entry->tuned = true;

  RUNTIME_DEBUG_LOG("[AUTOTUNE] M=%lld N=%lld K=%lld batch=%lld: "
                    "tested %d/%d algos, best=#%d (%.3f ms/%d iters)\n",
                    (long long)key.M, (long long)key.N, (long long)key.K,
                    (long long)key.batch_count, tested, entry->num_candidates,
                    best_idx, best_ms, AUTOTUNE_TIMING_ITERS);
}

//===----------------------------------------------------------------------===//
// Batched MatMul via hipBLASLt
//===----------------------------------------------------------------------===//
//
// ONNX MatMul semantics: output = A @ B (row-major)
//   A: [batch_count x M x K]
//   B: [batch_count x K x N]  (or [K x N] with broadcast)
//   output: [batch_count x M x N]
//
// hipBLASLt expects column-major storage. To avoid explicit transposition we
// use the identity:  C_row = (B^T * A^T)^T
//
// In column-major terms, row-major matrix A(M,K) looks like A^T(K,M).
// So we tell hipBLASLt:
//   m = N, n = M, k = K
//   A_ptr = B (column-major view is B^T: N rows, K cols, ld = N)
//   B_ptr = A (column-major view is A^T: K rows, M cols, ld = K)
//   C_ptr = output (column-major view is C^T: N rows, M cols, ld = N)
//
// Both fp16 and fp32 use HIPBLAS_COMPUTE_32F for accumulation precision.
//===----------------------------------------------------------------------===//

int wrap_hipblasLtMatmul(RuntimeState *state, int op_state_slot, const void *A,
                         const void *B, void *output, int64_t M, int64_t N,
                         int64_t K, int64_t batch_count, int64_t elem_size,
                         int64_t b_batch_stride) {
  OP_PROFILE(
      "matmul",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "m=%lld,n=%lld,k=%lld", (long long)M,
                 (long long)N, (long long)K);
        return std::string(b);
      },
      state);
  if (!state || !A || !B || !output) {
    fprintf(stderr, "Invalid arguments to wrap_hipblasLtMatmul\n");
    return -1;
  }

  hipblasLtHandle_t handle =
      static_cast<hipblasLtHandle_t>(hipdnn_ep_state_get_hipblas_handle(state));
  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));

  if (!handle || !stream) {
    fprintf(stderr, "wrap_hipblasLtMatmul: null handle or stream\n");
    return -1;
  }

  if (elem_size != 2 && elem_size != 4) {
    fprintf(stderr, "wrap_hipblasLtMatmul: unsupported elem_size %lld\n",
            (long long)elem_size);
    return -1;
  }

  const char *type_name = (elem_size == 2) ? "f16" : "f32";
  RUNTIME_DEBUG_LOG("[REAL] wrap_hipblasLtMatmul: M=%lld, N=%lld, K=%lld, "
                    "batch=%lld, b_batch_stride=%lld, elem_size=%lld (%s), "
                    "total_bytes=%lld\n",
                    (long long)M, (long long)N, (long long)K,
                    (long long)batch_count, (long long)b_batch_stride,
                    (long long)elem_size, type_name,
                    (long long)(batch_count * M * N * elem_size));

  MatmulState *ms = MatmulState::get_op_state(state, op_state_slot);
  if (!ms || !ms->table) {
    fprintf(stderr, "wrap_hipblasLtMatmul: missing op-state for slot %d\n",
            op_state_slot);
    return -1;
  }

  MatmulCacheKey key{M, N, K, batch_count, elem_size, b_batch_stride};
  MatmulCacheEntry *cached = queryOrCreateMatmul(*ms->table, handle, key);
  if (!cached) {
    fprintf(stderr,
            "wrap_hipblasLtMatmul: failed to create/find cached "
            "descriptors for M=%lld N=%lld K=%lld batch=%lld\n",
            (long long)M, (long long)N, (long long)K, (long long)batch_count);
    return -1;
  }

  // Workspace for auto-tune candidates (first call) or the tuned algo.
  // Uses ensure_workspace (not scratch_alloc) because the autotune loop
  // needs the raw buffer + its full size for hipBLASLt benchmarking.
  size_t needed_ws =
      cached->tuned ? cached->workspace_size : cached->max_candidate_workspace;
  if (needed_ws > 0) {
    if (hipdnn_ep_state_ensure_workspace(state, needed_ws) != 0)
      return -1;
  }

  void *ws_ptr = hipdnn_ep_state_get_workspace(state);
  size_t ws_size = hipdnn_ep_state_get_workspace_size(state);

  // Auto-tune once per shape across all sessions sharing this entry: the
  // winning algo is device-specific and identical for every session, so the
  // first caller benchmarks (with its own stream/workspace/data) and the rest
  // reuse the result. Double-checked locking on the per-entry mutex keeps the
  // steady state a lock-free atomic read; only concurrent first-touch of the
  // same shape blocks. (std::call_once is deliberately avoided: its MSVC
  // __std_init_once_* support symbols are unresolvable in the JIT-linked
  // runtime bitcode.)
  if (!cached->tuned.load(std::memory_order_acquire)) {
    std::lock_guard<std::mutex> tuneGuard(cached->tune_mu);
    if (!cached->tuned.load(std::memory_order_relaxed))
      autotuneMatmul(handle, stream, cached, B, A, output, ws_ptr, ws_size,
                     key);
  }

  float alpha = 1.0f;
  float beta = 0.0f;

  // For shapes the heuristic could not satisfy (use_default_algo), pass
  // algo=nullptr and zero workspace to let hipBLASLt use its internal default.
  hipblasLtMatmulAlgo_t *algo_ptr =
      cached->use_default_algo
          ? nullptr
          : const_cast<hipblasLtMatmulAlgo_t *>(&cached->algo);
  void *call_ws_ptr = cached->use_default_algo ? nullptr : ws_ptr;
  size_t call_ws_size = cached->use_default_algo ? 0 : ws_size;

  hipblasStatus_t st =
      hipblasLtMatmul(handle, cached->desc, &alpha, B,
                      cached->layA,    // "A" = B (row->col trick)
                      A, cached->layB, // "B" = A (row->col trick)
                      &beta, output, cached->layC, output, cached->layC,
                      algo_ptr, call_ws_ptr, call_ws_size, stream);

  if (st != HIPBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "wrap_hipblasLtMatmul: hipblasLtMatmul failed (%d)\n", st);
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_hipblasLtMatmul: completed successfully\n");
  return 0;
}
