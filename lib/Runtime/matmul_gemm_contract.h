/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_MATMUL_GEMM_CONTRACT_H
#define HIPDNN_EP_MATMUL_GEMM_CONTRACT_H

#include <cstddef>
#include <cstdint>
#include <limits>

namespace hipdnn_ep {
namespace blas_contract {

struct OutputSize {
  size_t elements = 0;
  size_t bytes = 0;
};

inline bool checkedMul(size_t lhs, size_t rhs, size_t &result) {
  if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs)
    return false;
  result = lhs * rhs;
  return true;
}

inline bool checkedI64Mul(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0)
    return false;
  if (rhs != 0 && lhs > std::numeric_limits<int64_t>::max() / rhs)
    return false;
  result = lhs * rhs;
  return true;
}

inline bool computeOutputSize(int64_t batch, int64_t m, int64_t n,
                              int64_t elementBytes, OutputSize &result) {
  if (batch < 0 || m < 0 || n < 0 || elementBytes <= 0)
    return false;

  int64_t elementsI64 = 0;
  if (!checkedI64Mul(batch, m, elementsI64) ||
      !checkedI64Mul(elementsI64, n, elementsI64))
    return false;
  size_t elements = static_cast<size_t>(elementsI64);

  size_t bytes = 0;
  if (!checkedMul(elements, static_cast<size_t>(elementBytes), bytes))
    return false;
  result = {elements, bytes};
  return true;
}

inline bool validateMatrixProducts(int64_t m, int64_t n, int64_t kA,
                                   int64_t kB) {
  int64_t ignored = 0;
  return kA >= 0 && kB >= 0 && checkedI64Mul(m, n, ignored) &&
         checkedI64Mul(m, kA, ignored) && checkedI64Mul(kB, n, ignored);
}

inline int64_t gemmElementBytes(int64_t typeCode) {
  switch (typeCode) {
  case 0: // f16
  case 3: // bf16
    return 2;
  case 1: // f32
    return 4;
  case 2: // f64
    return 8;
  default:
    return -1;
  }
}

enum class ValidationStatus {
  Success,
  EmptyOutput,
  Failure,
};

struct ValidationResult {
  ValidationStatus status = ValidationStatus::Success;
  OutputSize outputSize;
  bool outputSizeKnown = false;
  const char *errorMessage = nullptr;
};

inline ValidationResult
validateMatmul(bool batchAxesValid, int64_t m, int64_t n, int64_t kA,
               int64_t kB, int64_t batchCount, int64_t elementBytes,
               int64_t aBatchCount, int64_t bBatchCount, int64_t aBatchStride,
               int64_t bBatchStride, bool hasA, bool hasB, bool hasOutput) {
  ValidationResult result;
  bool elementSizeSupported = elementBytes == 2 || elementBytes == 4;
  result.outputSizeKnown =
      elementSizeSupported &&
      computeOutputSize(batchCount, m, n, elementBytes, result.outputSize);
  auto fail = [&](const char *message) {
    result.status = ValidationStatus::Failure;
    result.errorMessage = message;
    return result;
  };

  if (!batchAxesValid)
    return fail("wrap_hipblasLtMatmul: invalid runtime batch axes");
  if (!elementSizeSupported)
    return fail("wrap_hipblasLtMatmul: unsupported element size");
  if (!result.outputSizeKnown || !validateMatrixProducts(m, n, kA, kB) ||
      aBatchCount < 0 || bBatchCount < 0 || aBatchStride < 0 ||
      bBatchStride < 0)
    return fail("wrap_hipblasLtMatmul: invalid or overflowing dimensions");
  if (kA != kB)
    return fail("wrap_hipblasLtMatmul: contraction dimensions do not match");
  if ((aBatchCount != 1 && aBatchCount != batchCount) ||
      (bBatchCount != 1 && bBatchCount != batchCount))
    return fail("wrap_hipblasLtMatmul: runtime batch layout is not "
                "representable by one stride per operand");
  if (result.outputSize.elements == 0) {
    result.status = ValidationStatus::EmptyOutput;
    return result;
  }
  if (!hasA || !hasB || !hasOutput)
    return fail("Invalid tensor pointer in wrap_hipblasLtMatmul");
  return result;
}

inline ValidationResult validateGemm(int64_t m, int64_t n, int64_t kA,
                                     int64_t kB, int64_t transA, int64_t transB,
                                     int64_t typeCode, bool hasC, int64_t cDim0,
                                     int64_t cDim1, bool hasA, bool hasB,
                                     bool hasOutput) {
  ValidationResult result;
  result.outputSizeKnown = computeOutputSize(
      /*batch=*/1, m, n, gemmElementBytes(typeCode), result.outputSize);
  bool cShapeValid = !hasC ? cDim0 == 0 && cDim1 == 0
                           : cDim0 >= 0 && cDim1 >= 0 &&
                                 (cDim0 == 1 || cDim0 == m) &&
                                 (cDim1 == 1 || cDim1 == n);
  auto fail = [&](const char *message) {
    result.status = ValidationStatus::Failure;
    result.errorMessage = message;
    return result;
  };

  if (!result.outputSizeKnown || !validateMatrixProducts(m, n, kA, kB) ||
      (transA != 0 && transA != 1) || (transB != 0 && transB != 1) ||
      !cShapeValid)
    return fail("wrap_gemm: invalid or overflowing dimensions");
  if (kA != kB)
    return fail("wrap_gemm: contraction dimensions do not match");
  if (result.outputSize.elements == 0) {
    result.status = ValidationStatus::EmptyOutput;
    return result;
  }
  if (!hasA || !hasB || !hasOutput)
    return fail("wrap_gemm: invalid tensor pointer");
  return result;
}

} // namespace blas_contract
} // namespace hipdnn_ep

#endif // HIPDNN_EP_MATMUL_GEMM_CONTRACT_H
