/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "../op_state.h"
#include "error_check_macros.h"
#include "runtime_types.h"

#include <hipblaslt/hipblaslt-ext.hpp>

#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#define HIPBLAS_CHECK(cmd) HIPBLAS_CHECK_GOTO(cmd, cleanup)

// Type codes — must match the lowering in HipToLLVM.cpp GemmOpLowering
static constexpr int64_t kTypeFloat16 = 0;
static constexpr int64_t kTypeFloat32 = 1;
static constexpr int64_t kTypeFloat64 = 2;
static constexpr int64_t kTypeBFloat16 = 3;

static bool resolveGemmTypes(int64_t typeCode, hipDataType &dataType,
                             hipblasComputeType_t &computeType,
                             hipDataType &scaleType) {
  switch (typeCode) {
  case kTypeFloat16:
    dataType = HIP_R_16F;
    computeType = HIPBLAS_COMPUTE_32F;
    scaleType = HIP_R_32F;
    return true;
  case kTypeFloat32:
    dataType = HIP_R_32F;
    computeType = HIPBLAS_COMPUTE_32F;
    scaleType = HIP_R_32F;
    return true;
  case kTypeFloat64:
    dataType = HIP_R_64F;
    computeType = HIPBLAS_COMPUTE_64F;
    scaleType = HIP_R_64F;
    return true;
  case kTypeBFloat16:
    dataType = HIP_R_16BF;
    computeType = HIPBLAS_COMPUTE_32F;
    scaleType = HIP_R_32F;
    return true;
  default:
    return false;
  }
}

// =============================================================================
// Algorithm cache: query heuristic once per unique problem shape, reuse after.
// =============================================================================

struct GemmCacheKey {
  int64_t M, N, K, transA, transB, typeCode;
  bool bias_epilogue; // distinct algo for the fused-bias-epilogue problem
  bool operator==(const GemmCacheKey &o) const {
    return M == o.M && N == o.N && K == o.K && transA == o.transA &&
           transB == o.transB && typeCode == o.typeCode &&
           bias_epilogue == o.bias_epilogue;
  }
};

struct GemmCacheKeyHash {
  size_t operator()(const GemmCacheKey &k) const {
    size_t h = std::hash<int64_t>{}(k.M);
    h ^= std::hash<int64_t>{}(k.N) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int64_t>{}(k.K) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int64_t>{}(k.transA) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int64_t>{}(k.transB) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int64_t>{}(k.typeCode) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<bool>{}(k.bias_epilogue) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

struct GemmCacheEntry {
  hipblasLtMatmulAlgo_t algo;
  size_t workspace_size;
  // True if no heuristic algorithm could be found and we should fall back to
  // hipblasLtMatmul(..., algo=nullptr, ws=nullptr, ws_size=0). This unblocks
  // outlier shapes where gfx1151's Tensile library has no tile (e.g. router
  // M=128 N=128 K=2880 and lm_head M=128 N=201088 K=2880) but the default
  // internal kernel still works.
  bool use_default_algo = false;
};

// One hipBLASLt algo table shared across every session in the process and
// freed when the last session holding it is destroyed. Entries are POD (a
// selected hipblasLtMatmulAlgo_t value + workspace size); the per-call
// descriptors/layouts are created and destroyed in wrap_gemm, so the table
// owns no GPU resources and the destructor is implicit. The mutex serialises
// find/insert; concurrent cold-misses may each benchmark and the last writer
// wins -- wasteful but correct.
struct GemmAlgoTable {
  std::mutex mu;
  std::unordered_map<GemmCacheKey, GemmCacheEntry, GemmCacheKeyHash> map;
};

// Per-instance op state for hip.gemm: the slot holds a shared_ptr to the one
// shared algo table, reached through a global WeakStore keyed by device (see
// op_state.h). The store is weak_ptr-backed, so the table lives only while some
// session's GemmState holds a shared_ptr to it.
struct GemmState : OpStateT<GemmState> {
  std::shared_ptr<GemmAlgoTable> table;
  GemmState() {
    int dev = 0;
    hipGetDevice(&dev);
    table = WeakStore<int, GemmAlgoTable>::get_or_create(
        dev, [] { return std::make_shared<GemmAlgoTable>(); });
  }
};

extern "C" int8_t hipdnn_ep_op_state_construct_gemm(RuntimeState *state,
                                                    int32_t slot) {
  hipdnn_ep_op_state_set(state, slot, GemmState::create().release());
  return 0;
}

// =============================================================================
// Cold-path algorithm selection (factored out of wrap_gemm)
// =============================================================================

// Fill `heurs` with up to `maxCandidates` supported algorithm candidates for
// this problem, fast-first. Primary source is full solution enumeration via
// the ext API (getAllAlgos), ordered by estimated time: it exposes the fast
// (~4 ms) solution hipblaslt-bench uses, whereas
// hipblasLtMatmulAlgoGetHeuristic alone returns only a curated subset and can
// put a ~104 ms algo at [0] for the small-N vision/proj shapes (e.g. M=7296
// N=1152 K=4304). We copy the first supported candidates (filling workspaceSize
// via matmulIsAlgoSupported) and the benchmark below picks the measured
// fastest.
//
// Falls back to hipblasLtMatmulAlgoGetHeuristic with an escalating
// (workspace_limit, request_count) ladder when enumeration is unavailable or
// empty -- outlier shapes such as the gpt-oss-120b MoE router (M=128 N=128
// K=2880) and lm_head (M=128 N=201088 K=2880) get HIPBLAS_STATUS_INVALID_VALUE
// from the default (256MB, 1) on gfx1151's Tensile library because no tiled
// algorithm satisfies the leading-dimension / small-N constraints:
//   1. (2GB,   16)   broad candidate sweep, large workspace
//   2. (256MB, 16)   original workspace, broad sweep
//   3. (0,     16)   force non-workspace algo (often unblocks narrow N / huge
//   ld)
//   4. (256MB, 1)    original behaviour
//   5. (0,     1)    non-workspace, single candidate
// Returns the candidate count (0 if none found).
static int enumerateGemmAlgos(
    hipblasLtHandle_t handle, hipblasLtMatmulDesc_t matmul_desc,
    hipblasLtMatrixLayout_t matA_layout, hipblasLtMatrixLayout_t matB_layout,
    hipblasLtMatrixLayout_t matC_layout, float alpha, int64_t transA,
    int64_t transB, hipDataType dataType, hipblasComputeType_t computeType,
    hipblasLtMatmulHeuristicResult_t *heurs, int maxCandidates) {
  int returned = 0;

  auto try_heuristic = [&](size_t ws_bytes, int req_count) -> bool {
    hipblasLtMatmulPreference_t local_pref = nullptr;
    if (hipblasLtMatmulPreferenceCreate(&local_pref) != HIPBLAS_STATUS_SUCCESS)
      return false;
    hipblasLtMatmulPreferenceSetAttribute(
        local_pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes,
        sizeof(ws_bytes));
    returned = 0;
    hipblasStatus_t st = hipblasLtMatmulAlgoGetHeuristic(
        handle, matmul_desc, matA_layout, matB_layout, matC_layout, matC_layout,
        local_pref, req_count, heurs, &returned);
    hipblasLtMatmulPreferenceDestroy(local_pref);
    return st == HIPBLAS_STATUS_SUCCESS && returned > 0;
  };

  {
    std::vector<hipblasLtMatmulHeuristicResult_t> all_algos;
    hipblasOperation_t ga_opA = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t ga_opB = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    float zero_chk = 0.0f;
    if (hipblaslt_ext::getAllAlgos(
            handle, hipblaslt_ext::GemmType::HIPBLASLT_GEMM, ga_opA, ga_opB,
            dataType, dataType, dataType, dataType, computeType,
            all_algos) == HIPBLAS_STATUS_SUCCESS) {
      for (auto &r : all_algos) {
        if (returned >= maxCandidates)
          break;
        size_t need = 0;
        if (hipblaslt_ext::matmulIsAlgoSupported(
                handle, matmul_desc, &alpha, matA_layout, matB_layout,
                &zero_chk, matC_layout, matC_layout, r.algo,
                need) != HIPBLAS_STATUS_SUCCESS)
          continue;
        heurs[returned] = r;
        heurs[returned].workspaceSize = need;
        returned++;
      }
    }
  }

  bool ok = returned > 0;
  // Fallback to the heuristic API if enumeration is unavailable/empty.
  if (!ok) {
    ok = try_heuristic(2ULL << 30, maxCandidates) ||
         try_heuristic(256ULL << 20, maxCandidates) ||
         try_heuristic(0, maxCandidates) || try_heuristic(256ULL << 20, 1) ||
         try_heuristic(0, 1);
  }

  return ok ? returned : 0;
}

// Benchmark `returned` candidates in `heurs` and return the index of the
// measured fastest (defaults to 0 if timing setup fails). Times into a
// throwaway scratch buffer with beta=0 so the caller's real output is never
// disturbed.
static int benchmarkGemmAlgos(
    RuntimeState *state, hipblasLtHandle_t handle,
    hipblasLtMatmulDesc_t matmul_desc, hipblasLtMatrixLayout_t matA_layout,
    hipblasLtMatrixLayout_t matB_layout, hipblasLtMatrixLayout_t matC_layout,
    const void *A, const void *B, float alpha, int64_t M, int64_t N,
    hipDataType dataType, hipStream_t stream,
    hipblasLtMatmulHeuristicResult_t *heurs, int returned) {
  int best_idx = 0;
  size_t elemSize = (dataType == HIP_R_64F)   ? 8
                    : (dataType == HIP_R_32F) ? 4
                                              : 2;
  size_t bench_bytes = static_cast<size_t>(M) * N * elemSize;
  void *bench_out = nullptr;
  size_t maxws = 0;
  for (int i = 0; i < returned; ++i)
    if (heurs[i].workspaceSize > maxws)
      maxws = heurs[i].workspaceSize;
  void *bench_ws = nullptr;
  size_t bench_ws_size = 0;
  if (maxws > 0 && hipdnn_ep_state_ensure_workspace(state, maxws) == 0) {
    bench_ws = hipdnn_ep_state_get_workspace(state);
    bench_ws_size = hipdnn_ep_state_get_workspace_size(state);
  }
  hipEvent_t bs = nullptr, be = nullptr;
  if (bench_bytes > 0 && hipMalloc(&bench_out, bench_bytes) == hipSuccess &&
      hipEventCreate(&bs) == hipSuccess && hipEventCreate(&be) == hipSuccess) {
    float zero = 0.0f;
    auto run_cand = [&](int i) -> hipblasStatus_t {
      size_t wss = heurs[i].workspaceSize;
      void *wsp = (wss > 0) ? bench_ws : nullptr;
      return hipblasLtMatmul(handle, matmul_desc, &alpha, B, matA_layout, A,
                             matB_layout, &zero, bench_out, matC_layout,
                             bench_out, matC_layout, &heurs[i].algo, wsp, wss,
                             stream);
    };
    double best_ms = 1e30;
    for (int i = 0; i < returned; ++i) {
      // Skip candidates that need more workspace than we allocated --
      // running them with a smaller/null workspace just errors out.
      if (heurs[i].workspaceSize > bench_ws_size)
        continue;
      if (run_cand(i) != HIPBLAS_STATUS_SUCCESS) // warmup
        continue;
      if (hipEventRecord(bs, stream) != hipSuccess)
        continue;
      for (int r = 0; r < 3; ++r)
        run_cand(i);
      if (hipEventRecord(be, stream) != hipSuccess)
        continue;
      if (hipEventSynchronize(be) != hipSuccess)
        continue;
      float ms = 0.0f;
      if (hipEventElapsedTime(&ms, bs, be) != hipSuccess)
        continue;
      if (ms < best_ms) {
        best_ms = ms;
        best_idx = i;
      }
    }
  }
  if (bs)
    hipEventDestroy(bs);
  if (be)
    hipEventDestroy(be);
  if (bench_out)
    hipFree(bench_out);
  return best_idx;
}

// Cold-path selection for a unique problem shape: enumerate candidates,
// benchmark them (when more than one), and build the cache entry. Falls back to
// hipBLASLt's internal default kernel (use_default_algo, algo=nullptr at call
// time) when no heuristic algorithm can be found.
static GemmCacheEntry selectGemmAlgo(
    RuntimeState *state, hipblasLtHandle_t handle,
    hipblasLtMatmulDesc_t matmul_desc, hipblasLtMatrixLayout_t matA_layout,
    hipblasLtMatrixLayout_t matB_layout, hipblasLtMatrixLayout_t matC_layout,
    const void *A, const void *B, float alpha, int64_t M, int64_t N, int64_t K,
    int64_t transA, int64_t transB, int64_t typeCode, hipDataType dataType,
    hipblasComputeType_t computeType, hipStream_t stream) {
  constexpr int kMaxCandidates = 16;
  hipblasLtMatmulHeuristicResult_t heurs[kMaxCandidates];

  int returned = enumerateGemmAlgos(
      handle, matmul_desc, matA_layout, matB_layout, matC_layout, alpha, transA,
      transB, dataType, computeType, heurs, kMaxCandidates);

  GemmCacheEntry entry;
  if (returned > 0) {
    int best_idx =
        (returned > 1)
            ? benchmarkGemmAlgos(state, handle, matmul_desc, matA_layout,
                                 matB_layout, matC_layout, A, B, alpha, M, N,
                                 dataType, stream, heurs, returned)
            : 0;
    entry.algo = heurs[best_idx].algo;
    entry.workspace_size = heurs[best_idx].workspaceSize;
    entry.use_default_algo = false;
  } else {
    fprintf(stderr,
            "wrap_gemm: no algorithm from heuristic for M=%lld N=%lld "
            "K=%lld transA=%lld transB=%lld typeCode=%lld; falling back "
            "to default algo (algo=nullptr, ws=0)\n",
            (long long)M, (long long)N, (long long)K, (long long)transA,
            (long long)transB, (long long)typeCode);
    entry.workspace_size = 0;
    entry.use_default_algo = true;
  }
  return entry;
}

// =============================================================================
// ONNX Gemm via hipBLASLt
// =============================================================================
//
// ONNX Gemm semantics (row-major):
//   Y = alpha * op(A) * op(B) + beta * C
//   op(A) = A^T if transA else A  →  always [M, K] after op
//   op(B) = B^T if transB else B  →  always [K, N] after op
//   C is optional; supported shapes:
//     absent, [M, N], or [1, N]/[N] with beta==1 (fused bias epilogue)
//   Other C shapes (scalar, [M, 1], [1, N] with beta!=1, etc.) are unsupported.
//   Y has shape [M, N]
//
// hipBLASLt uses column-major. Using the transpose identity:
//   Y^T = alpha * op(B)^T * op(A)^T + beta * C^T
//
// So we swap A↔B in the hipBLASLt call with m=N, n=M, k=K:
//   hipBLASLt "A" = B buffer, TRANSA = transB ? OP_T : OP_N
//   hipBLASLt "B" = A buffer, TRANSB = transA ? OP_T : OP_N
//
// Matrix layouts (col-major view of row-major data):
//   transB=0: B_rm[K,N] → col-major [N,K] ld=N
//   transB=1: B_rm[N,K] → col-major [K,N] ld=K
//   transA=0: A_rm[M,K] → col-major [K,M] ld=K
//   transA=1: A_rm[K,M] → col-major [M,K] ld=M
//   C/Y:      [M,N] rm  → col-major [N,M] ld=N
// =============================================================================

int wrap_gemm(RuntimeState *state, int op_state_slot, const void *A,
              const void *B, const void *C, void *output, int64_t M, int64_t N,
              int64_t K, float alpha, float beta, int64_t transA,
              int64_t transB, int64_t typeCode, int64_t cDim0, int64_t cDim1) {
  OP_PROFILE(
      "gemm",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "m=%lld,n=%lld,k=%lld", (long long)M,
                 (long long)N, (long long)K);
        return std::string(b);
      },
      state);
  if (!state || !A || !B || !output) {
    fprintf(stderr, "wrap_gemm: invalid arguments\n");
    return -1;
  }

  hipblasLtHandle_t handle =
      static_cast<hipblasLtHandle_t>(hipdnn_ep_state_get_hipblas_handle(state));
  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));

  if (!handle || !stream) {
    fprintf(stderr, "wrap_gemm: null handle or stream\n");
    return -1;
  }

  GemmState *gs = GemmState::get_op_state(state, op_state_slot);
  if (!gs || !gs->table) {
    fprintf(stderr, "wrap_gemm: missing op-state for slot %d\n", op_state_slot);
    return -1;
  }
  GemmAlgoTable &table = *gs->table;

  hipDataType dataType;
  hipblasComputeType_t computeType;
  hipDataType scaleType;
  if (!resolveGemmTypes(typeCode, dataType, computeType, scaleType)) {
    fprintf(stderr, "wrap_gemm: unsupported typeCode %lld\n",
            (long long)typeCode);
    return -1;
  }

  // Fused-bias epilogue eligibility: a per-output-feature [N] / [1,N] bias
  // (cDim0==1, cDim1==N) with beta==1 maps exactly to hipBLASLt's
  // HIPBLASLT_EPILOGUE_BIAS, whose bias length must equal the rows of D.
  // hipBLASLt's D = C^T = [N, M], so the ONNX [N] bias is a per-row-of-D
  // vector -> direct match. Using the epilogue lets us run with beta=0 (D
  // written fresh, no C==output in-place read), which keeps the fast split-K
  // algorithms eligible -- the in-place beta=1 path was selecting a ~14x
  // slower kernel for the vision GEMM shapes. (bench: m=1152 n=7296 k=4304 TN
  // is ~3.9 ms with this config vs ~56 ms for beta=1 C==output.)
  bool use_bias_epilogue =
      C && beta == 1.0f && cDim0 == 1 && cDim1 == N &&
      (typeCode == kTypeFloat16 || typeCode == kTypeFloat32 ||
       typeCode == kTypeBFloat16);

  // C must already be [M, N], or [1, N]/[N] via the fused-bias epilogue.
  bool needsBroadcast = C && !(cDim0 == M && cDim1 == N) && !use_bias_epilogue;

  RUNTIME_DEBUG_LOG("[REAL] wrap_gemm: M=%lld, N=%lld, K=%lld, transA=%lld, "
                    "transB=%lld, alpha=%f, beta=%f, typeCode=%lld, C=%p, "
                    "cDim0=%lld, cDim1=%lld, needsBroadcast=%d\n",
                    (long long)M, (long long)N, (long long)K, (long long)transA,
                    (long long)transB, alpha, beta, (long long)typeCode, C,
                    (long long)cDim0, (long long)cDim1, (int)needsBroadcast);

  if (needsBroadcast) {
    fprintf(stderr,
            "wrap_gemm: unsupported C broadcast C[%lld,%lld] -> [%lld,%lld]\n",
            (long long)cDim0, (long long)cDim1, (long long)M, (long long)N);
    return -1;
  }

  // Select effective beta and C pointer for hipblasLtMatmul.
  //   C absent:      beta=0, C_ptr=output (placeholder, content irrelevant)
  //   C is [M,N]:    beta=beta, C_ptr=C
  //   bias epilogue: beta=0, C_ptr=output (bias applied in the epilogue)
  float effective_beta;
  const void *effective_C;
  if (use_bias_epilogue) {
    // Bias applied in the matmul epilogue; D written fresh (no C read).
    effective_beta = 0.0f;
    effective_C = output;
  } else if (!C) {
    effective_beta = 0.0f;
    effective_C = output;
  } else {
    effective_beta = beta;
    effective_C = C;
  }

  hipblasLtMatrixLayout_t matA_layout = nullptr;
  hipblasLtMatrixLayout_t matB_layout = nullptr;
  hipblasLtMatrixLayout_t matC_layout = nullptr;
  hipblasLtMatmulDesc_t matmul_desc = nullptr;
  int result = 0;

  // Cached algo for this problem, copied out by value so no map iterator is
  // held across the (long, unlocked) cold path -- a concurrent insert could
  // otherwise rehash and invalidate it.
  GemmCacheEntry cached{};
  bool have_cached = false;

  GemmCacheKey key{M, N, K, transA, transB, typeCode, use_bias_epilogue};
  {
    std::lock_guard<std::mutex> lk(table.mu);
    auto it = table.map.find(key);
    if (it != table.map.end()) {
      cached = it->second;
      have_cached = true;
    }
  }

  // hipBLASLt "A" = B buffer
  int64_t hblA_rows, hblA_cols, hblA_ld;
  if (!transB) {
    hblA_rows = N;
    hblA_cols = K;
    hblA_ld = N;
  } else {
    hblA_rows = K;
    hblA_cols = N;
    hblA_ld = K;
  }

  // hipBLASLt "B" = A buffer
  int64_t hblB_rows, hblB_cols, hblB_ld;
  if (!transA) {
    hblB_rows = K;
    hblB_cols = M;
    hblB_ld = K;
  } else {
    hblB_rows = M;
    hblB_cols = K;
    hblB_ld = M;
  }

  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&matA_layout, dataType, hblA_rows,
                                            hblA_cols, hblA_ld));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&matB_layout, dataType, hblB_rows,
                                            hblB_cols, hblB_ld));
  // C and output: [M,N] row-major → col-major [N,M] ld=N
  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&matC_layout, dataType, N, M, N));

  HIPBLAS_CHECK(
      hipblasLtMatmulDescCreate(&matmul_desc, computeType, scaleType));

  {
    hipblasOperation_t opA = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t opB = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    HIPBLAS_CHECK(hipblasLtMatmulDescSetAttribute(
        matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    HIPBLAS_CHECK(hipblasLtMatmulDescSetAttribute(
        matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));
  }

  // Fused-bias epilogue: add the [N] bias (= per-row-of-D vector) in the
  // matmul epilogue with beta=0. Set BEFORE algo selection so
  // matmulIsAlgoSupported and the benchmark below evaluate the actual
  // (epilogue) problem.
  if (use_bias_epilogue) {
    hipblasLtEpilogue_t epi = HIPBLASLT_EPILOGUE_BIAS;
    const void *bias_ptr = C;
    hipDataType bias_dtype = dataType;
    HIPBLAS_CHECK(hipblasLtMatmulDescSetAttribute(
        matmul_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
    HIPBLAS_CHECK(hipblasLtMatmulDescSetAttribute(
        matmul_desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr,
        sizeof(bias_ptr)));
    HIPBLAS_CHECK(hipblasLtMatmulDescSetAttribute(
        matmul_desc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_dtype,
        sizeof(bias_dtype)));
  }

  // Algorithm selection with caching. On a cold miss, select (enumerate +
  // benchmark) the algo for this problem shape, then publish it to the
  // process-wide cache. selectGemmAlgo never throws and always yields a usable
  // entry (a concrete algo, or use_default_algo for shapes with no heuristic).
  if (!have_cached) {
    GemmCacheEntry entry =
        selectGemmAlgo(state, handle, matmul_desc, matA_layout, matB_layout,
                       matC_layout, A, B, alpha, M, N, K, transA, transB,
                       typeCode, dataType, computeType, stream);
    {
      std::lock_guard<std::mutex> lk(table.mu);
      // Another thread may have inserted this key meanwhile; try_emplace keeps
      // the existing entry in that case. Either way `cached` ends up holding a
      // valid algo for this problem.
      cached = table.map.try_emplace(key, entry).first->second;
    }
    have_cached = true;

    RUNTIME_DEBUG_LOG("[REAL] wrap_gemm: cached algo for M=%lld N=%lld K=%lld "
                      "transA=%lld transB=%lld (ws=%zu)\n",
                      (long long)M, (long long)N, (long long)K,
                      (long long)transA, (long long)transB,
                      cached.workspace_size);
  }

  {
    if (cached.workspace_size > 0) {
      if (hipdnn_ep_state_ensure_workspace(state, cached.workspace_size) != 0) {
        result = -1;
        goto cleanup;
      }
    }

    void *ws_ptr = cached.use_default_algo
                       ? nullptr
                       : hipdnn_ep_state_get_workspace(state);
    size_t ws_size =
        cached.use_default_algo ? 0 : hipdnn_ep_state_get_workspace_size(state);
    hipblasLtMatmulAlgo_t *algo_ptr =
        cached.use_default_algo
            ? nullptr
            : const_cast<hipblasLtMatmulAlgo_t *>(&cached.algo);

    if (typeCode == kTypeFloat64) {
      double alpha_d = static_cast<double>(alpha);
      double beta_d = static_cast<double>(effective_beta);
      HIPBLAS_CHECK(hipblasLtMatmul(
          handle, matmul_desc, &alpha_d, B, matA_layout, A, matB_layout,
          &beta_d, effective_C, matC_layout, output, matC_layout, algo_ptr,
          ws_ptr, ws_size, stream));
    } else {
      HIPBLAS_CHECK(hipblasLtMatmul(
          handle, matmul_desc, &alpha, B, matA_layout, A, matB_layout,
          &effective_beta, effective_C, matC_layout, output, matC_layout,
          algo_ptr, ws_ptr, ws_size, stream));
    }
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_gemm: completed successfully\n");

cleanup:
  if (matA_layout)
    hipblasLtMatrixLayoutDestroy(matA_layout);
  if (matB_layout)
    hipblasLtMatrixLayoutDestroy(matB_layout);
  if (matC_layout)
    hipblasLtMatrixLayoutDestroy(matC_layout);
  if (matmul_desc)
    hipblasLtMatmulDescDestroy(matmul_desc);

  return result;
}
