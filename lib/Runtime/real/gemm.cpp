/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "../op_state.h"
#include "ck_gemm_select.h"
#include "runtime_types.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

// Type codes — must match the lowering in HipToLLVM.cpp GemmOpLowering
static constexpr int64_t kTypeFloat16 = 0;
static constexpr int64_t kTypeFloat32 = 1;
static constexpr int64_t kTypeFloat64 = 2;
static constexpr int64_t kTypeBFloat16 = 3;

static int ckDtypeForTypeCode(int64_t typeCode) {
  switch (typeCode) {
  case kTypeFloat16:
    return HIP_DTYPE_FLOAT16;
  case kTypeFloat32:
    return HIP_DTYPE_FLOAT32;
  case kTypeFloat64:
    return HIP_DTYPE_FLOAT64;
  case kTypeBFloat16:
    return HIP_DTYPE_BFLOAT16;
  default:
    return -1;
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
  // >= 0: Composable Kernel serves this problem with that instance. -1: the CK
  // instances do not serve it, so the reference GEMM fallback is used. Resolved
  // once per shape and cached, so the instance sweep runs only on a cold miss.
  int ck_instance = -1;
};

// One instance-selection table shared across every session in the process and
// freed when the last session holding it is destroyed. Entries are POD (the
// resolved CK instance index). The mutex serialises find/insert; concurrent
// cold-misses may each sweep and the last writer wins -- wasteful but correct.
struct GemmAlgoTable {
  std::mutex mu;
  std::unordered_map<GemmCacheKey, GemmCacheEntry, GemmCacheKeyHash> map;
};

// Per-instance op state for hip.gemm: the slot holds a shared_ptr to the one
// shared table, reached through a global WeakStore keyed by device (see
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
// Broadcast helper: write beta * broadcast(C) into dst[M, N]
// =============================================================================
// C is [cDim0, cDim1] and must be unidirectional-broadcastable to [M, N]:
// each cDim is 1 or equals the corresponding output extent. The reference
// fallback then adds this into the GEMM output elementwise.
//
// Device copies double a filled prefix (rows or columns) so a scalar, a
// row, or a column becomes a full [M, N] buffer without a library op.

static size_t gemmElemSize(int64_t typeCode) {
  if (typeCode == kTypeFloat64)
    return 8;
  if (typeCode == kTypeFloat32)
    return 4;
  return 2;
}

static float f16ToFloat(uint16_t h) {
  const uint32_t sign = (static_cast<uint32_t>(h & 0x8000u) << 16);
  const uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x3ffu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      int32_t e = 127 - 15 + 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        --e;
      }
      mant &= 0x3ffu;
      bits = sign | (static_cast<uint32_t>(e) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

static uint16_t floatToF16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const uint32_t absBits = bits & 0x7fffffffu;
  if (absBits > 0x7f800000u)
    return static_cast<uint16_t>(sign | 0x7e00u | ((absBits >> 13) & 0x3ffu));
  int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127;
  uint32_t mant = bits & 0x7fffffu;
  if (exp > 15)
    return static_cast<uint16_t>(sign | 0x7c00u);
  if (exp >= -14) {
    uint32_t half = (static_cast<uint32_t>(exp + 15) << 10) | (mant >> 13);
    const uint32_t remainder = mant & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u)))
      ++half;
    return static_cast<uint16_t>(sign | half);
  }
  if (exp < -24)
    return static_cast<uint16_t>(sign);
  mant |= 0x800000u;
  const int shift = -14 - exp;
  uint32_t half = mant >> (13 + shift);
  const uint32_t halfBit = 1u << (12 + shift);
  const uint32_t rem = mant & ((1u << (13 + shift)) - 1u);
  if ((rem & halfBit) && ((rem & (halfBit - 1u)) || (half & 1u)))
    ++half;
  return static_cast<uint16_t>(sign | half);
}

static float bf16ToFloat(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

static uint16_t floatToBf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

static void scaleHostC(unsigned char *data, size_t count, size_t elemSize,
                       int64_t typeCode, float beta) {
  for (size_t i = 0; i < count; ++i) {
    unsigned char *elem = data + i * elemSize;
    switch (typeCode) {
    case kTypeFloat32: {
      float value;
      std::memcpy(&value, elem, sizeof(value));
      value *= beta;
      std::memcpy(elem, &value, sizeof(value));
      break;
    }
    case kTypeFloat64: {
      double value;
      std::memcpy(&value, elem, sizeof(value));
      value *= static_cast<double>(beta);
      std::memcpy(elem, &value, sizeof(value));
      break;
    }
    case kTypeFloat16: {
      uint16_t bits;
      std::memcpy(&bits, elem, sizeof(bits));
      bits = floatToF16(f16ToFloat(bits) * beta);
      std::memcpy(elem, &bits, sizeof(bits));
      break;
    }
    case kTypeBFloat16: {
      uint16_t bits;
      std::memcpy(&bits, elem, sizeof(bits));
      bits = floatToBf16(bf16ToFloat(bits) * beta);
      std::memcpy(elem, &bits, sizeof(bits));
      break;
    }
    default:
      break;
    }
  }
}

static int expandFilledPrefix(void *dst, size_t blockBytes, size_t count,
                              hipStream_t stream) {
  if (count <= 1 || blockBytes == 0)
    return 0;
  auto *base = static_cast<char *>(dst);
  size_t filled = 1;
  while (filled < count) {
    const size_t n = std::min(filled, count - filled);
    hipError_t err =
        hipMemcpyAsync(base + filled * blockBytes, base, n * blockBytes,
                       hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
      fprintf(stderr, "wrap_gemm: expandFilledPrefix failed (%s)\n",
              hipGetErrorString(err));
      return -1;
    }
    filled += n;
  }
  return 0;
}

static int expandFilledColumns(void *output, size_t elemSize, int64_t M,
                               int64_t N, hipStream_t stream) {
  if (N <= 1)
    return 0;
  auto *base = static_cast<char *>(output);
  const size_t pitch = static_cast<size_t>(N) * elemSize;
  size_t filled = 1;
  while (filled < static_cast<size_t>(N)) {
    const size_t n = std::min(filled, static_cast<size_t>(N) - filled);
    hipError_t err = hipMemcpy2DAsync(
        base + filled * elemSize, pitch, base, pitch, n * elemSize,
        static_cast<size_t>(M), hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
      fprintf(stderr, "wrap_gemm: expandFilledColumns failed (%s)\n",
              hipGetErrorString(err));
      return -1;
    }
    filled += n;
  }
  return 0;
}

static int writeBroadcastC(RuntimeState *state, const void *C, void *output,
                           int64_t M, int64_t N, int64_t cDim0, int64_t cDim1,
                           float beta, int64_t typeCode) {
  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
  if (!stream) {
    fprintf(stderr, "wrap_gemm: writeBroadcastC: null stream\n");
    return -1;
  }
  if (M <= 0 || N <= 0 || cDim0 <= 0 || cDim1 <= 0 ||
      (cDim0 != 1 && cDim0 != M) || (cDim1 != 1 && cDim1 != N)) {
    fprintf(stderr,
            "wrap_gemm: writeBroadcastC: C[%lld,%lld] is not broadcastable "
            "to [%lld,%lld]\n",
            (long long)cDim0, (long long)cDim1, (long long)M, (long long)N);
    return -1;
  }

  const size_t elemSize = gemmElemSize(typeCode);
  const size_t outBytes =
      static_cast<size_t>(M) * static_cast<size_t>(N) * elemSize;

  RUNTIME_DEBUG_LOG("[REAL] wrap_gemm: writeBroadcastC C[%lld,%lld] -> "
                    "[%lld,%lld], beta=%f\n",
                    (long long)cDim0, (long long)cDim1, (long long)M,
                    (long long)N, beta);

  if (beta == 0.0f) {
    hipError_t err = hipMemsetAsync(output, 0, outBytes, stream);
    if (err != hipSuccess) {
      fprintf(stderr, "wrap_gemm: writeBroadcastC hipMemsetAsync failed (%s)\n",
              hipGetErrorString(err));
      return -1;
    }
    return 0;
  }

  const void *src = C;
  hipMemcpyKind kind = hipMemcpyDeviceToDevice;
  std::vector<unsigned char> host;
  if (beta != 1.0f) {
    const size_t count =
        static_cast<size_t>(cDim0) * static_cast<size_t>(cDim1);
    const size_t srcBytes = count * elemSize;
    host.resize(srcBytes);
    hipError_t err = hipMemcpy(host.data(), C, srcBytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
      fprintf(stderr, "wrap_gemm: writeBroadcastC D2H failed (%s)\n",
              hipGetErrorString(err));
      return -1;
    }
    scaleHostC(host.data(), count, elemSize, typeCode, beta);
    src = host.data();
    kind = hipMemcpyHostToDevice;
  }

  const bool broadcastRows = (cDim0 == 1);
  const bool broadcastCols = (cDim1 == 1);
  hipError_t err;
  const bool fromHost = (kind == hipMemcpyHostToDevice);
  auto copySeed = [&](void *dst, const void *s, size_t bytes) -> hipError_t {
    if (fromHost)
      return hipMemcpy(dst, s, bytes, kind);
    return hipMemcpyAsync(dst, s, bytes, kind, stream);
  };
  auto copySeed2D = [&](void *dst, size_t dpitch, const void *s, size_t spitch,
                        size_t width, size_t height) -> hipError_t {
    if (fromHost)
      return hipMemcpy2D(dst, dpitch, s, spitch, width, height, kind);
    return hipMemcpy2DAsync(dst, dpitch, s, spitch, width, height, kind,
                            stream);
  };

  if (broadcastRows && broadcastCols) {
    err = copySeed(output, src, elemSize);
    if (err != hipSuccess) {
      fprintf(stderr, "wrap_gemm: writeBroadcastC scalar copy failed (%s)\n",
              hipGetErrorString(err));
      return -1;
    }
    if (expandFilledPrefix(output, elemSize, static_cast<size_t>(N), stream) !=
        0)
      return -1;
    if (expandFilledPrefix(output, static_cast<size_t>(N) * elemSize,
                           static_cast<size_t>(M), stream) != 0)
      return -1;
    return 0;
  }
  if (broadcastRows) {
    const size_t rowBytes = static_cast<size_t>(N) * elemSize;
    err = copySeed(output, src, rowBytes);
    if (err != hipSuccess) {
      fprintf(stderr, "wrap_gemm: writeBroadcastC row copy failed (%s)\n",
              hipGetErrorString(err));
      return -1;
    }
    return expandFilledPrefix(output, rowBytes, static_cast<size_t>(M), stream);
  }
  if (broadcastCols) {
    const size_t pitch = static_cast<size_t>(N) * elemSize;
    err = copySeed2D(output, pitch, src, elemSize, elemSize,
                     static_cast<size_t>(M));
    if (err != hipSuccess) {
      fprintf(stderr, "wrap_gemm: writeBroadcastC col copy failed (%s)\n",
              hipGetErrorString(err));
      return -1;
    }
    return expandFilledColumns(output, elemSize, M, N, stream);
  }

  err = copySeed(output, src, outBytes);
  if (err != hipSuccess) {
    fprintf(stderr, "wrap_gemm: writeBroadcastC full copy failed (%s)\n",
            hipGetErrorString(err));
    return -1;
  }
  return 0;
}

// =============================================================================
// ONNX Gemm via Composable Kernel (tuned instances + reference fallback)
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
// The CK/reference kernels take the column-major convention. Using the
// transpose identity:
//   Y^T = alpha * op(B)^T * op(A)^T + beta * C^T
//
// So we swap A↔B with m=N, n=M, k=K:
//   "A" = B buffer, TRANSA = transB ? OP_T : OP_N
//   "B" = A buffer, TRANSB = transA ? OP_T : OP_N
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

  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
  if (!stream) {
    fprintf(stderr, "wrap_gemm: null stream\n");
    return -1;
  }

  GemmState *gs = GemmState::get_op_state(state, op_state_slot);
  if (!gs || !gs->table) {
    fprintf(stderr, "wrap_gemm: missing op-state for slot %d\n", op_state_slot);
    return -1;
  }
  GemmAlgoTable &table = *gs->table;

  const int abDtype = ckDtypeForTypeCode(typeCode);
  if (abDtype < 0) {
    fprintf(stderr, "wrap_gemm: unsupported typeCode %lld\n",
            (long long)typeCode);
    return -1;
  }

  // A per-output-feature [N] / [1,N] bias with beta==1 is the fused-bias
  // problem CK serves through its Add epilogue; every other C shape routes
  // through the post-add below the reference fallback.
  const bool use_bias_epilogue =
      C && beta == 1.0f && cDim0 == 1 && cDim1 == N &&
      (typeCode == kTypeFloat16 || typeCode == kTypeFloat32 ||
       typeCode == kTypeBFloat16);

  RUNTIME_DEBUG_LOG("[REAL] wrap_gemm: M=%lld, N=%lld, K=%lld, transA=%lld, "
                    "transB=%lld, alpha=%f, beta=%f, typeCode=%lld, C=%p, "
                    "cDim0=%lld, cDim1=%lld\n",
                    (long long)M, (long long)N, (long long)K, (long long)transA,
                    (long long)transB, alpha, beta, (long long)typeCode, C,
                    (long long)cDim0, (long long)cDim1);

  // Column-major leading dimensions of the swapped call (see banner): the CK
  // and reference kernels take these as lda/ldb ("A" = B buffer, "B" = A).
  const int64_t hblA_ld = transB ? K : N;
  const int64_t hblB_ld = transA ? M : K;

  GemmCacheKey key{M, N, K, transA, transB, typeCode, use_bias_epilogue};
  GemmCacheEntry cached{};
  bool have_cached = false;
  {
    std::lock_guard<std::mutex> lk(table.mu);
    auto it = table.map.find(key);
    if (it != table.map.end()) {
      cached = it->second;
      have_cached = true;
    }
  }

  // CK eligibility mirrors the tuned f16 instances: no alpha, ONNX transA==0,
  // and the A-side transpose (ONNX transB) only together with a bias. `!C ||
  // use_bias_epilogue` also keeps `output` free of a pre-seeded beta*C, which
  // ckSelectGemmInstance relies on when it times into `output`.
  const bool ck_eligible = typeCode == kTypeFloat16 && alpha == 1.0f &&
                           transA == 0 && (!C || use_bias_epilogue) &&
                           (transB == 0 || use_bias_epilogue);
  const void *ck_bias = use_bias_epilogue ? C : nullptr;

  // Resolve once per shape: try the CK instances when eligible, else mark the
  // entry for the reference fallback (ck_instance == -1). Cached either way so
  // the instance sweep runs only on a cold miss.
  if (!have_cached) {
    GemmCacheEntry entry;
    if (ck_eligible) {
      entry.ck_instance = ckSelectGemmInstance(
          stream, B, A, ck_bias, output, N, M, K, /*batch=*/1,
          static_cast<int>(transB), static_cast<int>(transA), HIP_DTYPE_FLOAT16,
          HIP_DTYPE_FLOAT16, alpha, hblA_ld, hblB_ld, N,
          /*strideA=*/0, /*strideB=*/0, /*strideD=*/0);
    }
    {
      std::lock_guard<std::mutex> lk(table.mu);
      cached = table.map.try_emplace(key, entry).first->second;
    }
    have_cached = true;
    RUNTIME_DEBUG_LOG("[REAL] wrap_gemm: resolved M=%lld N=%lld K=%lld "
                      "transA=%lld transB=%lld -> ck_instance=%d\n",
                      (long long)M, (long long)N, (long long)K,
                      (long long)transA, (long long)transB, cached.ck_instance);
  }

  if (cached.ck_instance >= 0) {
    int result = hip_ck_gemm_run(
        stream, cached.ck_instance, B, A, ck_bias, output, N, M, K,
        /*batch=*/1, static_cast<int>(transB), static_cast<int>(transA),
        HIP_DTYPE_FLOAT16, HIP_DTYPE_FLOAT16, alpha, hblA_ld, hblB_ld, N,
        /*strideA=*/0, /*strideB=*/0, /*strideD=*/0);
    if (result != 0) {
      // The instance was chosen by running this same geometry, so a refusal
      // here means the ABI contract is broken rather than the shape changing.
      fprintf(stderr,
              "wrap_gemm: CK instance %d refused M=%lld N=%lld K=%lld "
              "transA=%lld transB=%lld\n",
              cached.ck_instance, (long long)M, (long long)N, (long long)K,
              (long long)transA, (long long)transB);
    }
    return result;
  }

  // Reference fallback: alpha * op(A) op(B) into output, then + beta*C.
  int rc = hip_ref_gemm_run(stream, B, A, output, N, M, K, /*batch=*/1,
                            static_cast<int>(transB), static_cast<int>(transA),
                            abDtype, abDtype, alpha, hblA_ld, hblB_ld, N,
                            /*strideA=*/0, /*strideB=*/0, /*strideD=*/0);
  if (rc != 0) {
    fprintf(stderr,
            "wrap_gemm: reference GEMM unsupported for typeCode=%lld M=%lld "
            "N=%lld K=%lld\n",
            (long long)typeCode, (long long)M, (long long)N, (long long)K);
    return -1;
  }

  if (C) {
    // output += beta * C_broadcast. hip_elementwise_add serves f16/f32; a C
    // term with fp64 or bf16 appears in no supported model and errors.
    if (abDtype != HIP_DTYPE_FLOAT16 && abDtype != HIP_DTYPE_FLOAT32) {
      fprintf(stderr, "wrap_gemm: C/bias add unsupported for typeCode=%lld\n",
              (long long)typeCode);
      return -1;
    }
    const size_t elemSize = gemmElemSize(typeCode);
    const size_t outBytes =
        static_cast<size_t>(M) * static_cast<size_t>(N) * elemSize;
    void *scratch = nullptr;
    if (hipMalloc(&scratch, outBytes) != hipSuccess) {
      fprintf(stderr, "wrap_gemm: C-add scratch alloc failed\n");
      return -1;
    }
    int bc =
        writeBroadcastC(state, C, scratch, M, N, cDim0, cDim1, beta, typeCode);
    if (bc == 0) {
      bc = hip_elementwise_add(stream, output, scratch, output,
                               static_cast<int64_t>(M) * N, abDtype);
    }
    hipStreamSynchronize(stream);
    hipFree(scratch);
    if (bc != 0) {
      fprintf(stderr, "wrap_gemm: C/bias add failed (%d)\n", bc);
      return -1;
    }
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_gemm: completed successfully\n");
  return 0;
}
