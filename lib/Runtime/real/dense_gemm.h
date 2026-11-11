/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_DENSE_GEMM_H
#define HIPDNN_EP_DENSE_GEMM_H

#include "../debug_log.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace hipdnn_ep::dense_gemm {

// Preserve the existing MatMul ABI's f16/f32 byte-size values. BF16 uses a
// distinct marker, because its two-byte representation is not IEEE fp16.
inline int typeCode(int64_t elementKind) {
  switch (elementKind) {
  case 2:
    return 0;
  case 4:
    return 1;
  case 8:
    return 2;
  case -2:
    return 3;
  default:
    return -1;
  }
}

inline size_t elementBytes(int code) {
  return code == 2 ? 8 : code == 1 ? 4 : 2;
}

inline bool multiply(int64_t a, int64_t b, int64_t &result) {
  if (a < 0 || b < 0 || (b && a > INT64_MAX / b))
    return false;
  result = a * b;
  return true;
}

inline bool add(int64_t a, int64_t b, int64_t &result) {
  if (a < 0 || b < 0 || a > INT64_MAX - b)
    return false;
  result = a + b;
  return true;
}

inline bool byteSize(int64_t elements, size_t bytes, size_t &result) {
  if (elements < 0 || static_cast<uint64_t>(elements) >
                          static_cast<uint64_t>(PTRDIFF_MAX) / bytes)
    return false;
  result = static_cast<size_t>(elements) * bytes;
  return true;
}

inline bool geometry(int64_t m, int64_t n, int64_t k, int64_t ta, int64_t tb,
                     int code) {
  return m >= 0 && m <= INT_MAX && n >= 0 && n <= INT_MAX && k >= 0 &&
         k <= INT_MAX && (ta == 0 || ta == 1) && (tb == 0 || tb == 1) &&
         code >= 0 && code <= 3;
}

// All dense runtime entries converge here. Unsupported requests and launch
// failures are returned to the caller; there is no vendor GEMM retry.
inline int launch(void *stream, const void *a, const void *b, const void *c,
                  void *y, int64_t m, int64_t n, int64_t k, float alpha,
                  float beta, int64_t ta, int64_t tb, int code,
                  int64_t cRows = 1, int64_t cCols = 1) {
  if (!geometry(m, n, k, ta, tb, code) ||
      (c && ((cRows != 1 && cRows != m) || (cCols != 1 && cCols != n)))) {
    fprintf(stderr,
            "custom GEMM: unsupported dimensions, dtype or C broadcast\n");
    return -1;
  }
  if (!m || !n)
    return 0;
  if (!stream || !a || !b || !y)
    return -1;
  int64_t aCount, bCount, yCount, cCount;
  size_t checkedBytes;
  if (!multiply(m, k, aCount) || !multiply(k, n, bCount) ||
      !multiply(m, n, yCount) ||
      !byteSize(aCount, elementBytes(code), checkedBytes) ||
      !byteSize(bCount, elementBytes(code), checkedBytes) ||
      !byteSize(yCount, elementBytes(code), checkedBytes) ||
      (c && (!multiply(cRows, cCols, cCount) ||
             !byteSize(cCount, elementBytes(code), checkedBytes))))
    return -1;

  RUNTIME_DEBUG_LOG("[REAL] custom GEMM M=%lld N=%lld K=%lld dtype=%d "
                    "ta=%lld tb=%lld\n",
                    (long long)m, (long long)n, (long long)k, code,
                    (long long)ta, (long long)tb);
  int rc =
      hip_gemm(stream, a, b, c, y, static_cast<int>(m), static_cast<int>(n),
               static_cast<int>(k), alpha, beta, static_cast<int>(ta),
               static_cast<int>(tb), code, static_cast<int>(c ? cRows : 1),
               static_cast<int>(c ? cCols : 1));
  if (rc)
    fprintf(stderr, "custom GEMM failed (%d); M=%lld N=%lld K=%lld dtype=%d\n",
            rc, (long long)m, (long long)n, (long long)k, code);
  return rc;
}

// ONNX leading axes broadcast independently. A flat B stride cannot represent
// [2,1,M,K] @ [1,3,K,N], or one unbatched A shared by batched B.
// This host-only map preserves tight matrix storage without expanding tensors.
struct BatchLayout {
  int rank = 0;
  int64_t count = 1;
  int64_t aElements = 0, bElements = 0;
  std::vector<int64_t> sizes, aStrides, bStrides;

  bool initialize(int aRank, const int64_t *aDims, int bRank,
                  const int64_t *bDims, int64_t aMatrix, int64_t bMatrix) {
    if (aRank < 0 || bRank < 0 || (aRank && !aDims) || (bRank && !bDims))
      return false;
    rank = aRank > bRank ? aRank : bRank;
    sizes.resize(rank);
    aStrides.resize(rank);
    bStrides.resize(rank);
    count = 1;
    int64_t aStride = aMatrix, bStride = bMatrix;
    for (int i = rank - 1; i >= 0; --i) {
      const int ai = i - (rank - aRank), bi = i - (rank - bRank);
      const int64_t ad = ai < 0 ? 1 : aDims[ai];
      const int64_t bd = bi < 0 ? 1 : bDims[bi];
      if (ad < 0 || bd < 0 || (ad != bd && ad != 1 && bd != 1))
        return false;
      sizes[i] = ad == 1 ? bd : ad;
      aStrides[i] = ad == 1 ? 0 : aStride;
      bStrides[i] = bd == 1 ? 0 : bStride;
      if (!multiply(aStride, ad, aStride) || !multiply(bStride, bd, bStride) ||
          !multiply(count, sizes[i], count))
        return false;
    }
    aElements = aStride;
    bElements = bStride;
    return true;
  }

  void offsets(int64_t batch, int64_t &a, int64_t &b) const {
    a = b = 0;
    for (int i = rank - 1; i >= 0; --i) {
      const int64_t coordinate = batch % sizes[i];
      batch /= sizes[i];
      a += coordinate * aStrides[i];
      b += coordinate * bStrides[i];
    }
  }
};

} // namespace hipdnn_ep::dense_gemm

#endif // HIPDNN_EP_DENSE_GEMM_H
