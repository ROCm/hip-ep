/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_ATTENTION_GEMM_H
#define HIPDNN_EP_ATTENTION_GEMM_H

#include "dense_gemm.h"

namespace hipdnn_ep::dense_gemm {

// The decomposed attention entries historically describe a column-major
// operation op(A)[m,k] * B[k,n]. Swapping the physical operands gives the
// equivalent row-major operation B^T[n,k] * op(A)^T[k,m], without tensor
// expansion or a new kernel. Explicit strides preserve chunk/page layouts.
struct ColumnMajorBatch {
  int64_t m = 0, n = 0, k = 0, batches = 0;
  int64_t strideA = 0, strideB = 0, strideY = 0;
  int64_t aCount = 0, bCount = 0;
  int inputCode = 0, outputCode = 0;
  bool transA = false;
  size_t workspace_size = 0, scratchBOffset = 0;

  bool initialize(int64_t rows, int64_t cols, int64_t reduction,
                  int64_t batchCount, bool transposeA, int inputType,
                  int outputType, int64_t aStride, int64_t bStride,
                  int64_t yStride) {
    if (!geometry(cols, rows, reduction, 0, transposeA, outputType) ||
        batchCount < 0 || (inputType != 0 && inputType != 1) ||
        (outputType != inputType && !(inputType == 0 && outputType == 1)) ||
        aStride < 0 || bStride < 0 || yStride < 0)
      return false;
    m = rows;
    n = cols;
    k = reduction;
    batches = batchCount;
    transA = transposeA;
    inputCode = inputType;
    outputCode = outputType;
    int64_t yCount;
    if (!multiply(m, k, aCount) || !multiply(n, k, bCount) ||
        !multiply(m, n, yCount))
      return false;
    strideA = aStride ? aStride : aCount;
    strideB = bStride ? bStride : bCount;
    strideY = yStride ? yStride : yCount;
    auto extentFits = [&](int64_t stride, int64_t matrix, int code) {
      int64_t last, extent;
      size_t bytes;
      return multiply(batches ? batches - 1 : 0, stride, last) &&
             add(last, batches ? matrix : 0, extent) &&
             byteSize(extent, elementBytes(code), bytes);
    };
    if (!extentFits(strideA, aCount, inputCode) ||
        !extentFits(strideB, bCount, inputCode) ||
        !extentFits(strideY, yCount, outputCode))
      return false;

    // hip_gemm's input/output dtype is uniform. Exact fp16->fp32 casts before
    // a score GEMM retain fp32 accumulation/output for the following softmax.
    workspace_size = scratchBOffset = 0;
    if (inputCode != outputCode) {
      size_t aBytes, bBytes;
      if (!byteSize(aCount, sizeof(float), aBytes) ||
          !byteSize(bCount, sizeof(float), bBytes) || aBytes > SIZE_MAX - 63)
        return false;
      scratchBOffset = (aBytes + 63) & ~size_t(63);
      if (bBytes > SIZE_MAX - scratchBOffset - 63)
        return false;
      workspace_size = scratchBOffset + bBytes + 63;
    }
    return true;
  }

  int run(void *stream, const void *a, const void *b, void *y, float alpha,
          void *workspace, size_t workspaceBytes) const {
    if (!batches || !m || !n)
      return 0;
    if (!a || !b || !y ||
        (workspace_size && (!workspace || workspaceBytes < workspace_size)))
      return -1;
    char *scratch = workspace_size
                        ? reinterpret_cast<char *>(
                              (reinterpret_cast<uintptr_t>(workspace) + 63) &
                              ~uintptr_t(63))
                        : nullptr;
    for (int64_t batch = 0; batch < batches; ++batch) {
      const void *pa =
          static_cast<const char *>(a) +
          static_cast<size_t>(batch * strideA) * elementBytes(inputCode);
      const void *pb =
          static_cast<const char *>(b) +
          static_cast<size_t>(batch * strideB) * elementBytes(inputCode);
      if (scratch) {
        int rc = hip_cast(stream, pa, scratch, aCount, HIP_DTYPE_FLOAT16,
                          HIP_DTYPE_FLOAT32);
        if (rc)
          return rc;
        rc = hip_cast(stream, pb, scratch + scratchBOffset, bCount,
                      HIP_DTYPE_FLOAT16, HIP_DTYPE_FLOAT32);
        if (rc)
          return rc;
        pa = scratch;
        pb = scratch + scratchBOffset;
      }
      void *py = static_cast<char *>(y) + static_cast<size_t>(batch * strideY) *
                                              elementBytes(outputCode);
      int rc = launch(stream, pb, pa, nullptr, py, n, m, k, alpha, 0.0f, 0,
                      transA, outputCode);
      if (rc)
        return rc;
    }
    return 0;
  }
};

} // namespace hipdnn_ep::dense_gemm

#endif // HIPDNN_EP_ATTENTION_GEMM_H
