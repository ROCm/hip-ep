/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "error_check_macros.h"
#include "runtime_types.h"

#include <hipblaslt/hipblaslt-ext.hpp>

#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>

#define HIPBLAS_CHECK(cmd) HIPBLAS_CHECK_GOTO(cmd, cleanup)
#define MIOPEN_CHECK(cmd) MIOPEN_CHECK_GOTO(cmd, cleanup)

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

static std::unordered_map<GemmCacheKey, GemmCacheEntry, GemmCacheKeyHash>
    g_gemm_algo_cache;
// Serializes access to g_gemm_algo_cache. Concurrent cold-misses may each run
// the benchmark and the last writer wins; that is wasteful but correct.
static std::mutex g_gemm_algo_cache_mutex;

// =============================================================================
// Broadcast helper: write beta * C_broadcast into output using MIOpen
// =============================================================================
// Uses miopenOpTensor(Add) with broadcasting:
//   output = beta * C_broadcast
// C is normalized to 2D [cDim0, cDim1], broadcastable to [M, N].
// After this, hipblasLtMatmul accumulates with effective_beta=1.0:
//   output = alpha * A * B + 1.0 * output  (= alpha*A*B + beta*C_broadcast)

static bool resolveGemmMiopenType(int64_t typeCode, miopenDataType_t &dt) {
  switch (typeCode) {
  case kTypeFloat16:
    dt = miopenHalf;
    return true;
  case kTypeFloat32:
    dt = miopenFloat;
    return true;
  case kTypeFloat64:
    dt = miopenDouble;
    return true;
  case kTypeBFloat16:
    dt = miopenBFloat16;
    return true;
  default:
    return false;
  }
}

// MIOpen contract for miopenOpTensor: the A-operand descriptor and the
// destination descriptor must have *identical* shapes; only the B operand may
// broadcast. The previous implementation passed the small bias descriptor
// (cDesc) as both A and B while the destination was the full-output descriptor,
// which MIOpen rejected with "A and C Tensors do not match" (status 7) any time
// the bias actually needed broadcasting (e.g. a [1,N] bias onto [M,N]).
//
// Fix: zero the output buffer, then compute
//     output = 1.0 * output + beta * broadcast(C)
// so the A operand and destination both use outDesc and only B (the bias)
// broadcasts. Pre-zeroing avoids dependency on uninitialized output (which
// could contain NaNs that 1.0*NaN would propagate).
static int broadcastBiasToOutput(RuntimeState *state, const void *C,
                                 void *output, int64_t M, int64_t N,
                                 int64_t cDim0, int64_t cDim1, float beta,
                                 int64_t typeCode) {
  miopenHandle_t handle =
      static_cast<miopenHandle_t>(hipdnn_ep_state_get_miopen_handle(state));
  if (!handle) {
    fprintf(stderr, "wrap_gemm: broadcastBiasToOutput: null MIOpen handle\n");
    return -1;
  }

  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
  if (!stream) {
    fprintf(stderr, "wrap_gemm: broadcastBiasToOutput: null stream\n");
    return -1;
  }

  miopenDataType_t dt;
  if (!resolveGemmMiopenType(typeCode, dt)) {
    fprintf(stderr,
            "wrap_gemm: broadcastBiasToOutput: unsupported typeCode %lld\n",
            (long long)typeCode);
    return -1;
  }

  size_t elemSize = (typeCode == kTypeFloat64)   ? 8
                    : (typeCode == kTypeFloat32) ? 4
                                                 : 2; // fp16 / bf16

  RUNTIME_DEBUG_LOG("[REAL] wrap_gemm: broadcastBiasToOutput C[%lld,%lld] -> "
                    "[%lld,%lld], beta=%f\n",
                    (long long)cDim0, (long long)cDim1, (long long)M,
                    (long long)N, beta);

  // Pre-zero the destination so 1.0*output contributes 0 in the op below.
  size_t outBytes = static_cast<size_t>(M) * static_cast<size_t>(N) * elemSize;
  hipError_t hipErr = hipMemsetAsync(output, 0, outBytes, stream);
  if (hipErr != hipSuccess) {
    fprintf(stderr,
            "wrap_gemm: broadcastBiasToOutput: hipMemsetAsync(%zu bytes) "
            "failed (%d): %s\n",
            outBytes, (int)hipErr, hipGetErrorString(hipErr));
    return -1;
  }

  miopenTensorDescriptor_t cDesc = nullptr;
  miopenTensorDescriptor_t outDesc = nullptr;
  int result = 0;

  MIOPEN_CHECK(miopenCreateTensorDescriptor(&cDesc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&outDesc));

  MIOPEN_CHECK(miopenSet4dTensorDescriptor(
      cDesc, dt, 1, 1, static_cast<int>(cDim0), static_cast<int>(cDim1)));
  MIOPEN_CHECK(miopenSet4dTensorDescriptor(
      outDesc, dt, 1, 1, static_cast<int>(M), static_cast<int>(N)));

  // output = 1.0 * output + beta * broadcast(C) + 0 * output
  //        = beta * broadcast(C)              (since output was zeroed)
  if (typeCode == kTypeFloat64) {
    double alpha1 = 1.0, alpha2 = static_cast<double>(beta), beta_c = 0.0;
    MIOPEN_CHECK(miopenOpTensor(handle, miopenTensorOpAdd, &alpha1, outDesc,
                                output, &alpha2, cDesc, C, &beta_c, outDesc,
                                output));
  } else {
    float alpha1 = 1.0f, alpha2 = beta, beta_c = 0.0f;
    MIOPEN_CHECK(miopenOpTensor(handle, miopenTensorOpAdd, &alpha1, outDesc,
                                output, &alpha2, cDesc, C, &beta_c, outDesc,
                                output));
  }

cleanup:
  if (cDesc)
    miopenDestroyTensorDescriptor(cDesc);
  if (outDesc)
    miopenDestroyTensorDescriptor(outDesc);
  return result;
}

// =============================================================================
// ONNX Gemm via hipBLASLt
// =============================================================================
//
// ONNX Gemm semantics (row-major):
//   Y = alpha * op(A) * op(B) + beta * C
//   op(A) = A^T if transA else A  →  always [M, K] after op
//   op(B) = B^T if transB else B  →  always [K, N] after op
//   C is optional, broadcastable to [M, N]
//   Y has shape [M, N]
//
// C broadcast shapes (ONNX unidirectional broadcastable to [M, N]):
//   []      → scalar      → cDim0=1, cDim1=1
//   [N]     → row vector   → cDim0=1, cDim1=N   (most common: FC bias)
//   [1, N]  → row vector   → cDim0=1, cDim1=N
//   [M, 1]  → col vector   → cDim0=M, cDim1=1
//   [M, N]  → no broadcast  → cDim0=M, cDim1=N
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

int wrap_gemm(RuntimeState *state, const void *A, const void *B, const void *C,
              void *output, int64_t M, int64_t N, int64_t K, float alpha,
              float beta, int64_t transA, int64_t transB, int64_t typeCode,
              int64_t cDim0, int64_t cDim1) {
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
  // slower kernel for the vision GEMM shapes. It also drops the separate
  // MIOpen broadcast op. (bench: m=1152 n=7296 k=4304 TN is ~3.9 ms with this
  // config vs ~56 ms for broadcast+beta=1 C==output.)
  bool use_bias_epilogue =
      C && beta == 1.0f && cDim0 == 1 && cDim1 == N &&
      (typeCode == kTypeFloat16 || typeCode == kTypeFloat32 ||
       typeCode == kTypeBFloat16);

  // Determine if C needs broadcasting.
  // When C is [M, N], hipblasLtMatmul handles it directly (single-pass).
  // Otherwise, we first broadcast beta*C into output via MIOpen, then let
  // hipblasLtMatmul accumulate with effective_beta=1.0 on top of it.
  bool needsBroadcast = C && !(cDim0 == M && cDim1 == N) && !use_bias_epilogue;

  RUNTIME_DEBUG_LOG("[REAL] wrap_gemm: M=%lld, N=%lld, K=%lld, transA=%lld, "
                    "transB=%lld, alpha=%f, beta=%f, typeCode=%lld, C=%p, "
                    "cDim0=%lld, cDim1=%lld, needsBroadcast=%d\n",
                    (long long)M, (long long)N, (long long)K, (long long)transA,
                    (long long)transB, alpha, beta, (long long)typeCode, C,
                    (long long)cDim0, (long long)cDim1, (int)needsBroadcast);

  // Pre-broadcast: write beta * C_broadcast into output before matmul.
  if (needsBroadcast) {
    int bc_result = broadcastBiasToOutput(state, C, output, M, N, cDim0, cDim1,
                                          beta, typeCode);
    if (bc_result != 0)
      return bc_result;
  }

  // Select effective beta and C pointer for hipblasLtMatmul.
  //   C absent:     beta=0, C_ptr=output (placeholder, content irrelevant)
  //   C is [M,N]:   beta=beta, C_ptr=C  (direct single-pass, no broadcast)
  //   C broadcast:  beta=1.0, C_ptr=output (already holds beta*C_broadcast)
  float effective_beta;
  const void *effective_C;
  if (use_bias_epilogue) {
    // Bias applied in the matmul epilogue; D written fresh (no C read).
    effective_beta = 0.0f;
    effective_C = output;
  } else if (!C) {
    effective_beta = 0.0f;
    effective_C = output;
  } else if (!needsBroadcast) {
    effective_beta = beta;
    effective_C = C;
  } else {
    effective_beta = 1.0f;
    effective_C = output;
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
    std::lock_guard<std::mutex> lk(g_gemm_algo_cache_mutex);
    auto it = g_gemm_algo_cache.find(key);
    if (it != g_gemm_algo_cache.end()) {
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
  // matmul epilogue with beta=0, instead of a separate broadcast + beta=1
  // in-place accumulate. Set BEFORE algo selection so matmulIsAlgoSupported
  // and the benchmark below evaluate the actual (epilogue) problem.
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

  // Algorithm selection with caching.
  //
  // hipblasLtMatmulAlgoGetHeuristic is sensitive to (workspace_limit,
  // request_count): for outlier shapes such as the gpt-oss-120b MoE router
  // (M=128 N=128 K=2880) and lm_head (M=128 N=201088 K=2880) the default
  // (256MB, 1) returns HIPBLAS_STATUS_INVALID_VALUE on gfx1151's Tensile
  // library because no tiled algorithm satisfies the leading-dimension /
  // small-N constraints. Try a small escalation ladder so common outlier
  // shapes pick up a non-tiled or larger-candidate-set algorithm:
  //   1. (256MB, 1)         original behaviour
  //   2. (0,     1)         force non-workspace algo
  //                          (often unblocks narrow N or huge ld)
  //   3. (256MB, 16)        broader candidate sweep
  //   4. (0,     16)        non-workspace + broader sweep
  if (!have_cached) {
    constexpr int kMaxCandidates = 16;
    hipblasLtMatmulHeuristicResult_t heurs[kMaxCandidates];
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
      hipblasStatus_t st = hipblasLtMatmulAlgoGetHeuristic(
          handle, matmul_desc, matA_layout, matB_layout, matC_layout,
          matC_layout, local_pref, req_count, heurs, &returned);
      hipblasLtMatmulPreferenceDestroy(local_pref);
      return st == HIPBLAS_STATUS_SUCCESS && returned > 0;
    };

    // Primary candidate source: full solution enumeration via the ext API,
    // ordered by estimated time. hipblasLtMatmulAlgoGetHeuristic returns only a
    // curated subset and puts a ~104 ms algo at [0] for the small-N vision/proj
    // shapes (e.g. M=7296 N=1152 K=4304); getAllAlgos exposes the fast (~4 ms)
    // solution that hipblaslt-bench uses. We copy the first supported
    // candidates (sorted, so the fast one is early) into heurs[] -- filling
    // workspaceSize via matmulIsAlgoSupported -- and the benchmark loop below
    // picks the measured fastest.
    bool ok = false;
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
          if (returned >= kMaxCandidates)
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
      ok = returned > 0;
    }
    // Fallback to the heuristic API if enumeration is unavailable/empty.
    if (!ok) {
      ok = try_heuristic(2ULL << 30, kMaxCandidates) ||
           try_heuristic(256ULL << 20, kMaxCandidates) ||
           try_heuristic(0, kMaxCandidates) || try_heuristic(256ULL << 20, 1) ||
           try_heuristic(0, 1);
    }

    GemmCacheEntry entry;
    if (ok) {
      // Benchmark the returned candidates and pick the measured fastest.
      // Time into a throwaway scratch output buffer with beta=0 so we never
      // disturb the real `output` (which, on the broadcast path, already
      // holds beta*C and would accumulate across timing iterations).
      int best_idx = 0;
      if (returned > 1) {
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
        if (bench_bytes > 0 &&
            hipMalloc(&bench_out, bench_bytes) == hipSuccess &&
            hipEventCreate(&bs) == hipSuccess &&
            hipEventCreate(&be) == hipSuccess) {
          float zero = 0.0f;
          auto run_cand = [&](int i) -> hipblasStatus_t {
            size_t wss = heurs[i].workspaceSize;
            void *wsp = (wss > 0) ? bench_ws : nullptr;
            return hipblasLtMatmul(handle, matmul_desc, &alpha, B, matA_layout,
                                   A, matB_layout, &zero, bench_out,
                                   matC_layout, bench_out, matC_layout,
                                   &heurs[i].algo, wsp, wss, stream);
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
      }
      entry.algo = heurs[best_idx].algo;
      entry.workspace_size = heurs[best_idx].workspaceSize;
      entry.use_default_algo = false;
    } else {
      // Final fallback: let hipBLASLt pick its internal default kernel by
      // passing algo=nullptr at call time. This is what hipblas.cpp's fp32
      // GEMM already does unconditionally and it's the documented escape
      // hatch when the heuristic can't satisfy the layout constraints.
      fprintf(stderr,
              "wrap_gemm: no algorithm from heuristic for M=%lld N=%lld "
              "K=%lld transA=%lld transB=%lld typeCode=%lld; falling back "
              "to default algo (algo=nullptr, ws=0)\n",
              (long long)M, (long long)N, (long long)K, (long long)transA,
              (long long)transB, (long long)typeCode);
      entry.workspace_size = 0;
      entry.use_default_algo = true;
    }
    {
      std::lock_guard<std::mutex> lk(g_gemm_algo_cache_mutex);
      // Another thread may have inserted this key meanwhile; try_emplace keeps
      // the existing entry in that case. Either way `cached` ends up holding a
      // valid algo for this problem.
      cached = g_gemm_algo_cache.try_emplace(key, entry).first->second;
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
