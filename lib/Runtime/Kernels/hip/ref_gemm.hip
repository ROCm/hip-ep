/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Composable Kernel naive (reference) GEMM behind a narrow C ABI, the universal
// fallback for shapes and dtypes the tuned WMMA/DL instances in ck_gemm.hip do
// not serve. It is correct for any m/n/k and covers fp16/bf16/fp32/fp64, so the
// callers that used to fall back to hipBLASLt reach this instead.
//
// Arguments arrive in hipBLASLt's column-major convention and are remapped the
// same way as ck_gemm.hip: a column-major D = op(A) op(B) becomes CK's
// row-major E = A' B' with M'=n, N'=m, K'=k and the operands swapped
// (A' = B buffer, B' = A buffer). transB must be 0.
//
// ck::ReferenceGemm assumes packed operands (it ignores leading dimensions and
// has no batch), so this ABI's lda/ldb/ldd are unused and the caller must pass
// packed buffers; batch is handled here by looping and offsetting pointers by
// the batch strides. alpha enters as a Scale C-element-op.

#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cstdint>

#include "ck/ck.hpp"
#include "ck/stream_config.hpp"
#include "ck/library/reference_tensor_operation/gpu/reference_gemm.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace {

using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using F32  = float;
using F64  = double;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Scale       = ck::tensor_operation::element_wise::Scale;

template <typename ABData, typename CData, typename AccData, typename ApLayout,
          typename BpLayout>
void runRefBatched(hipStream_t stream, const void *aprime, const void *bprime,
                   void *e, int64_t mp, int64_t np, int64_t k, int64_t batch,
                   int64_t batchStrideA, int64_t batchStrideB,
                   int64_t batchStrideE, float alpha) {
  using RefOp = ck::tensor_operation::device::ReferenceGemm<
      ApLayout, BpLayout, Row, ABData, ABData, CData, AccData, PassThrough,
      PassThrough, Scale>;
  const auto *a = static_cast<const ABData *>(aprime);
  const auto *b = static_cast<const ABData *>(bprime);
  auto *c = static_cast<CData *>(e);
  auto inv = RefOp::MakeInvoker();
  for (int64_t i = 0; i < batch; ++i) {
    auto arg = RefOp::MakeArgument(
        a + i * batchStrideA, b + i * batchStrideB, c + i * batchStrideE,
        (ck::index_t)mp, (ck::index_t)np, (ck::index_t)k, PassThrough{},
        PassThrough{}, Scale{alpha});
    inv.Run(arg, StreamConfig{stream, false});
  }
}

// After the column-major -> row-major swap (A' = B buffer, B' = A buffer,
// M'=n, N'=m), A' is transposed exactly when the ABI's op(B) is (transB) and
// B' when op(A) is (transA); E is always Row. Dispatch lives here so the
// Row/Col aliases stay inside the anonymous namespace.
template <typename ABData, typename CData, typename AccData>
void runRefDispatch(hipStream_t stream, const void *aprime, const void *bprime,
                    void *e, int64_t mp, int64_t np, int64_t k, int64_t batch,
                    int transA, int transB, int64_t batchStrideA,
                    int64_t batchStrideB, int64_t batchStrideE, float alpha) {
  if (transB && transA) {
    runRefBatched<ABData, CData, AccData, Col, Col>(
        stream, aprime, bprime, e, mp, np, k, batch, batchStrideA, batchStrideB,
        batchStrideE, alpha);
  } else if (transB) {
    runRefBatched<ABData, CData, AccData, Col, Row>(
        stream, aprime, bprime, e, mp, np, k, batch, batchStrideA, batchStrideB,
        batchStrideE, alpha);
  } else if (transA) {
    runRefBatched<ABData, CData, AccData, Row, Col>(
        stream, aprime, bprime, e, mp, np, k, batch, batchStrideA, batchStrideB,
        batchStrideE, alpha);
  } else {
    runRefBatched<ABData, CData, AccData, Row, Row>(
        stream, aprime, bprime, e, mp, np, k, batch, batchStrideA, batchStrideB,
        batchStrideE, alpha);
  }
}

int runRef(hipStream_t stream, const void *A, const void *B, void *D, int64_t m,
           int64_t n, int64_t k, int64_t batch, int transA, int transB,
           int abDtype, int dDtype, float alpha, int64_t strideA,
           int64_t strideB, int64_t strideD) {
  const void *ap = B; // A' = B buffer
  const void *bp = A; // B' = A buffer
  const int64_t mp = n; // M' = n
  const int64_t np = m; // N' = m
  const int64_t bstA = strideB;
  const int64_t bstB = strideA;
  const int64_t bstE = strideD;

#define REF_DISPATCH(ABD, CD, ACC)                                             \
  runRefDispatch<ABD, CD, ACC>(stream, ap, bp, D, mp, np, k, batch, transA,    \
                               transB, bstA, bstB, bstE, alpha)

  try {
    if (abDtype == HIP_DTYPE_FLOAT16 && dDtype == HIP_DTYPE_FLOAT16) {
      REF_DISPATCH(F16, F16, float);
    } else if (abDtype == HIP_DTYPE_FLOAT16 && dDtype == HIP_DTYPE_FLOAT32) {
      REF_DISPATCH(F16, F32, float);
    } else if (abDtype == HIP_DTYPE_BFLOAT16 && dDtype == HIP_DTYPE_BFLOAT16) {
      REF_DISPATCH(BF16, BF16, float);
    } else if (abDtype == HIP_DTYPE_BFLOAT16 && dDtype == HIP_DTYPE_FLOAT32) {
      REF_DISPATCH(BF16, F32, float);
    } else if (abDtype == HIP_DTYPE_FLOAT32 && dDtype == HIP_DTYPE_FLOAT32) {
      REF_DISPATCH(F32, F32, float);
    } else if (abDtype == HIP_DTYPE_FLOAT64 && dDtype == HIP_DTYPE_FLOAT64) {
      REF_DISPATCH(F64, F64, double);
    } else {
      return 1;
    }
  } catch (...) {
    return 1;
  }

#undef REF_DISPATCH
  return 0;
}

} // namespace

extern "C" int hip_ref_gemm_run(void *stream, const void *A, const void *B,
                                void *D, int64_t m, int64_t n, int64_t k,
                                int64_t batch, int transA, int transB,
                                int abDtype, int dDtype, float alpha,
                                int64_t lda, int64_t ldb, int64_t ldd,
                                int64_t strideA, int64_t strideB,
                                int64_t strideD) {
  (void)lda;
  (void)ldb;
  (void)ldd;
  return runRef(static_cast<hipStream_t>(stream), A, B, D, m, n, k, batch,
                transA, transB, abDtype, dDtype, alpha, strideA, strideB,
                strideD);
}
