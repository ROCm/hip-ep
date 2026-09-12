/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "../op_state.h"
#include "cache_utils.h"
#include "ck_gemm_select.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <unordered_map>

//===----------------------------------------------------------------------===//
// Per-shape routing cache. On the first call for each unique
// (M, N, K, batch, elem_size, trans) shape we time the CK instances that
// accept it and cache the winner (or -1 for the reference fallback);
// subsequent calls reuse the decision with zero overhead.
//===----------------------------------------------------------------------===//

struct MatmulCacheKey {
  int64_t M, N, K, batch_count, elem_size, b_batch_stride, transA, transB;
  bool operator==(const MatmulCacheKey &o) const {
    return M == o.M && N == o.N && K == o.K && batch_count == o.batch_count &&
           elem_size == o.elem_size && b_batch_stride == o.b_batch_stride &&
           transA == o.transA && transB == o.transB;
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
    hash_combine_val(h, k.transA);
    hash_combine_val(h, k.transB);
    return h;
  }
};

/// Resolved routing for a single shape. Held by pointer in the
/// MatmulAlgoTable so the per-entry mutex has a stable address.
struct MatmulCacheEntry {
  // >= 0: Composable Kernel serves this shape with that instance. -1: the CK
  // instances do not serve it, so the reference GEMM fallback is used. Resolved
  // once per shape under `mu`; `resolved` gates the lock-free steady state.
  int ck_instance = -1;
  std::atomic<bool> resolved{false};
  std::mutex mu;
};

// One routing table per device, shared across every session in the process and
// freed when the last session holding it is destroyed. Entries are stored by
// pointer so they keep a stable address and can hold a non-movable mutex.
struct MatmulAlgoTable {
  std::mutex mu;
  std::unordered_map<MatmulCacheKey, std::unique_ptr<MatmulCacheEntry>,
                     MatmulCacheKeyHash>
      map;
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

//===----------------------------------------------------------------------===//
// Batched MatMul via Composable Kernel (tuned instances + reference fallback)
//===----------------------------------------------------------------------===//
//
// ONNX MatMul semantics: output = A @ B (row-major)
//   A: [batch_count x M x K]
//   B: [batch_count x K x N]  (or [K x N] with broadcast)
//   output: [batch_count x M x N]
//
// The kernels take the column-major convention. To avoid explicit
// transposition we use the identity  C_row = (B^T * A^T)^T, i.e. a row-major
// A(M,K) is a column-major A^T(K,M). So we swap A/B with m=N, n=M, k=K:
//   "A" = B buffer (column-major B^T: N rows, K cols, ld=N)
//   "B" = A buffer (column-major A^T: K rows, M cols, ld=K)
//   "C" = output   (column-major C^T: N rows, M cols, ld=N)
//===----------------------------------------------------------------------===//

int wrap_hipblasLtMatmul(RuntimeState *state, int op_state_slot, const void *A,
                         const void *B, void *output, int64_t M, int64_t N,
                         int64_t K, int64_t batch_count, int64_t elem_size,
                         int64_t b_batch_stride, int64_t transA,
                         int64_t transB) {
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

  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
  if (!stream) {
    fprintf(stderr, "wrap_hipblasLtMatmul: null stream\n");
    return -1;
  }

  if (elem_size != 2 && elem_size != 4) {
    fprintf(stderr, "wrap_hipblasLtMatmul: unsupported elem_size %lld\n",
            (long long)elem_size);
    return -1;
  }
  const int abDtype = (elem_size == 2) ? HIP_DTYPE_FLOAT16 : HIP_DTYPE_FLOAT32;

  RUNTIME_DEBUG_LOG("[REAL] wrap_hipblasLtMatmul: M=%lld, N=%lld, K=%lld, "
                    "batch=%lld, b_batch_stride=%lld, transA=%lld, "
                    "transB=%lld, elem_size=%lld\n",
                    (long long)M, (long long)N, (long long)K,
                    (long long)batch_count, (long long)b_batch_stride,
                    (long long)transA, (long long)transB, (long long)elem_size);

  MatmulState *ms = MatmulState::get_op_state(state, op_state_slot);
  if (!ms || !ms->table) {
    fprintf(stderr, "wrap_hipblasLtMatmul: missing op-state for slot %d\n",
            op_state_slot);
    return -1;
  }
  MatmulAlgoTable &table = *ms->table;

  MatmulCacheKey key{M,      N,     K, batch_count, elem_size, b_batch_stride,
                     transA, transB};
  MatmulCacheEntry *entry;
  {
    std::lock_guard<std::mutex> lk(table.mu);
    auto it = table.map.find(key);
    if (it == table.map.end()) {
      it = table.map.emplace(key, std::make_unique<MatmulCacheEntry>()).first;
    }
    entry = it->second.get();
  }

  // CK's f16 instances apply neither alpha nor a bias, matching this call's
  // fixed alpha=1 / beta=0 / no-C form, and are instantiated only for
  // untransposed operands. With transA and transB both N the column-major
  // leading dimensions collapse to (lda, ldb, ldd) = (N, K, N).
  const bool ck_eligible = elem_size == 2 && transA == 0 && transB == 0;
  const int64_t hblA_ld = transB ? K : N; // "A" = B buffer
  const int64_t hblB_ld = transA ? M : K; // "B" = A buffer

  // Resolve once per shape across all sessions sharing this entry; the CK
  // decision is device-specific and identical for every session. Double-checked
  // locking on the per-entry mutex keeps the steady state a lock-free read.
  if (!entry->resolved.load(std::memory_order_acquire)) {
    std::lock_guard<std::mutex> probeGuard(entry->mu);
    if (!entry->resolved.load(std::memory_order_relaxed)) {
      if (ck_eligible) {
        entry->ck_instance = ckSelectGemmInstance(
            stream, B, A, /*bias=*/nullptr, output, N, M, K, batch_count,
            /*transA=*/0, /*transB=*/0, HIP_DTYPE_FLOAT16, HIP_DTYPE_FLOAT16,
            /*alpha=*/1.0f, /*lda=*/N, /*ldb=*/K, /*ldd=*/N,
            /*strideA=*/b_batch_stride, /*strideB=*/M * K, /*strideD=*/M * N);
      }
      entry->resolved.store(true, std::memory_order_release);
      RUNTIME_DEBUG_LOG("[MATMUL] resolved M=%lld N=%lld K=%lld batch=%lld -> "
                        "ck_instance=%d\n",
                        (long long)M, (long long)N, (long long)K,
                        (long long)batch_count, entry->ck_instance);
    }
  }

  if (entry->ck_instance >= 0) {
    if (hip_ck_gemm_run(stream, entry->ck_instance, B, A, /*bias=*/nullptr,
                        output, N, M, K, batch_count, /*transA=*/0,
                        /*transB=*/0, HIP_DTYPE_FLOAT16, HIP_DTYPE_FLOAT16,
                        /*alpha=*/1.0f, /*lda=*/N, /*ldb=*/K, /*ldd=*/N,
                        /*strideA=*/b_batch_stride, /*strideB=*/M * K,
                        /*strideD=*/M * N) != 0) {
      // The instance was chosen by running this same geometry, so a refusal
      // here means the ABI contract is broken rather than the shape changing.
      fprintf(stderr,
              "wrap_hipblasLtMatmul: CK instance %d refused M=%lld N=%lld "
              "K=%lld batch=%lld\n",
              entry->ck_instance, (long long)M, (long long)N, (long long)K,
              (long long)batch_count);
      return -1;
    }
    return 0;
  }

  // Reference fallback: A @ B with alpha=1, no bias, over all transpose combos.
  int rc = hip_ref_gemm_run(stream, B, A, output, N, M, K, batch_count,
                            static_cast<int>(transB), static_cast<int>(transA),
                            abDtype, abDtype, /*alpha=*/1.0f, hblA_ld, hblB_ld,
                            N, /*strideA=*/b_batch_stride, /*strideB=*/M * K,
                            /*strideD=*/M * N);
  if (rc != 0) {
    fprintf(stderr,
            "wrap_hipblasLtMatmul: reference GEMM unsupported for M=%lld "
            "N=%lld K=%lld elem_size=%lld\n",
            (long long)M, (long long)N, (long long)K, (long long)elem_size);
    return -1;
  }
  return 0;
}
