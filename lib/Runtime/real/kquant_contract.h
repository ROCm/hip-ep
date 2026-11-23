/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPDNN_EP_KQUANT_CONTRACT_H
#define HIPDNN_EP_KQUANT_CONTRACT_H

#include "hip/kquant_abi.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace hipdnn_ep::kquant {

constexpr int64_t kQ8BlockElements = HIP_Q8_1_BLOCK_ELEMENTS;
constexpr int64_t kQ8BlockBytes = HIP_Q8_1_BLOCK_SIZE;
constexpr int64_t kKQuantBlockElements = HIP_KQUANT_BLOCK_ELEMENTS;
constexpr uintptr_t kQ8BlockAlignment = HIP_Q8_1_BLOCK_ALIGNMENT;

constexpr int64_t kQ4KFormatId = HIP_KQUANT_FORMAT_Q4_K;
constexpr int64_t kQ5KFormatId = HIP_KQUANT_FORMAT_Q5_K;
constexpr int64_t kQ6KFormatId = HIP_KQUANT_FORMAT_Q6_K;

struct PrefixShape {
  int64_t dim0;
  int64_t dim1;
  int64_t dim2;

  bool operator==(const PrefixShape &other) const {
    return dim0 == other.dim0 && dim1 == other.dim1 && dim2 == other.dim2;
  }
};

inline bool checkedMultiply(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs <= 0 || rhs <= 0 || lhs > std::numeric_limits<int64_t>::max() / rhs)
    return false;
  result = lhs * rhs;
  return true;
}

inline bool rangesOverlap(const void *lhs, size_t lhsBytes, const void *rhs,
                          size_t rhsBytes) {
  const uintptr_t lhsBegin = reinterpret_cast<uintptr_t>(lhs);
  const uintptr_t rhsBegin = reinterpret_cast<uintptr_t>(rhs);
  if (lhsBytes > std::numeric_limits<uintptr_t>::max() - lhsBegin ||
      rhsBytes > std::numeric_limits<uintptr_t>::max() - rhsBegin) {
    return true;
  }
  const uintptr_t lhsEnd = lhsBegin + lhsBytes;
  const uintptr_t rhsEnd = rhsBegin + rhsBytes;
  return lhsBegin < rhsEnd && rhsBegin < lhsEnd;
}

inline bool hasQ8Alignment(const void *pointer) {
  return reinterpret_cast<uintptr_t>(pointer) % kQ8BlockAlignment == 0;
}

inline bool getKQuantBlockBytes(int64_t format, int64_t &bytes) {
  if (format == kQ4KFormatId)
    bytes = HIP_KQUANT_Q4_K_BLOCK_SIZE;
  else if (format == kQ5KFormatId)
    bytes = HIP_KQUANT_Q5_K_BLOCK_SIZE;
  else if (format == kQ6KFormatId)
    bytes = HIP_KQUANT_Q6_K_BLOCK_SIZE;
  else
    return false;
  return true;
}

inline bool getKQuantBlockElements(int64_t format, int64_t &elements) {
  if (format != kQ4KFormatId && format != kQ5KFormatId &&
      format != kQ6KFormatId)
    return false;
  elements = kKQuantBlockElements;
  return true;
}

inline bool hasKQuantWeightAlignment(const void *pointer, int64_t format) {
  uintptr_t alignment = 0;
  if (format == kQ4KFormatId)
    alignment = HIP_KQUANT_Q4_K_BLOCK_ALIGNMENT;
  else if (format == kQ6KFormatId)
    alignment = HIP_KQUANT_Q6_K_BLOCK_ALIGNMENT;
  else if (format == kQ5KFormatId)
    alignment = HIP_KQUANT_Q5_K_BLOCK_ALIGNMENT;
  else
    return false;
  return reinterpret_cast<uintptr_t>(pointer) % alignment == 0;
}

inline bool hasF32Alignment(const void *pointer) {
  return reinterpret_cast<uintptr_t>(pointer) % alignof(float) == 0;
}

struct AddRmsNormQ8Layout {
  int64_t rowCount;
  size_t tensorBytes;
  size_t scaleBytes;
  size_t q8Bytes;
};

inline bool validateAddRmsNormQ8Layout(PrefixShape lhs, PrefixShape rhs,
                                       PrefixShape residual, PrefixShape q8,
                                       int64_t K, AddRmsNormQ8Layout &layout) {
  if (!(lhs == rhs) || !(lhs == residual) || !(lhs == q8) || lhs.dim0 <= 0 ||
      lhs.dim1 <= 0 || lhs.dim2 <= 0 || K <= 0 || K % kKQuantBlockElements != 0)
    return false;

  int64_t rows01 = 0;
  int64_t rows = 0;
  int64_t tensorElements = 0;
  int64_t q8RowBytes = 0;
  int64_t q8Bytes = 0;
  int64_t scaleBytes = 0;
  if (!checkedMultiply(lhs.dim0, lhs.dim1, rows01) ||
      !checkedMultiply(rows01, lhs.dim2, rows) ||
      !checkedMultiply(rows, K, tensorElements) ||
      !checkedMultiply(tensorElements, int64_t{sizeof(float)},
                       tensorElements) ||
      !checkedMultiply(K / kQ8BlockElements, kQ8BlockBytes, q8RowBytes) ||
      !checkedMultiply(rows, q8RowBytes, q8Bytes) ||
      !checkedMultiply(K, int64_t{sizeof(float)}, scaleBytes)) {
    return false;
  }
  if (static_cast<uint64_t>(tensorElements) >
          std::numeric_limits<size_t>::max() ||
      static_cast<uint64_t>(scaleBytes) > std::numeric_limits<size_t>::max() ||
      static_cast<uint64_t>(q8Bytes) > std::numeric_limits<size_t>::max()) {
    return false;
  }

  layout = {rows, static_cast<size_t>(tensorElements),
            static_cast<size_t>(scaleBytes), static_cast<size_t>(q8Bytes)};
  return true;
}

inline bool hasDisjointAddRmsNormQ8Outputs(const float *lhs, const float *rhs,
                                           const float *scale, float *residual,
                                           const void *quantized,
                                           const AddRmsNormQ8Layout &layout) {
  const auto overlapsResidual = [&](const void *input, size_t bytes) {
    return rangesOverlap(residual, layout.tensorBytes, input, bytes);
  };
  const auto overlapsQ8 = [&](const void *input, size_t bytes) {
    return rangesOverlap(quantized, layout.q8Bytes, input, bytes);
  };
  return !overlapsResidual(lhs, layout.tensorBytes) &&
         !overlapsResidual(rhs, layout.tensorBytes) &&
         !overlapsResidual(scale, layout.scaleBytes) &&
         !rangesOverlap(residual, layout.tensorBytes, quantized,
                        layout.q8Bytes) &&
         !overlapsQ8(lhs, layout.tensorBytes) &&
         !overlapsQ8(rhs, layout.tensorBytes) &&
         !overlapsQ8(scale, layout.scaleBytes);
}

struct KQuantSwiGluQ8Layout {
  int64_t batchCount;
  int64_t M;
  int64_t outputElements;
  size_t q8Bytes;
  size_t gateWeightBytes;
  size_t upWeightBytes;
  size_t outputBytes;
};

inline bool isKQuantSwiGluFormat(int64_t format) {
  return format == kQ4KFormatId || format == kQ6KFormatId;
}

inline bool
validateKQuantSwiGluQ8Layout(PrefixShape q8, int64_t gateWeightBytes,
                             int64_t upWeightBytes, PrefixShape output,
                             int64_t K, int64_t N, int64_t gateFormat,
                             int64_t upFormat, KQuantSwiGluQ8Layout &layout) {
  if (!(q8 == output) || q8.dim0 <= 0 || q8.dim1 <= 0 || q8.dim2 <= 0 ||
      K <= 0 || K % kQ8BlockElements != 0 || N <= 0 || gateWeightBytes <= 0 ||
      upWeightBytes <= 0 || !isKQuantSwiGluFormat(gateFormat) ||
      !isKQuantSwiGluFormat(upFormat))
    return false;

  int64_t gateBlockBytes = 0;
  int64_t upBlockBytes = 0;
  int64_t gateBlockElements = 0;
  int64_t upBlockElements = 0;
  int64_t gateBlocksPerWeightRow = 0;
  int64_t upBlocksPerWeightRow = 0;
  int64_t expectedGateBytes = 0;
  int64_t expectedUpBytes = 0;
  int64_t batchCount = 0;
  int64_t rowCount = 0;
  int64_t outputElements = 0;
  int64_t q8RowBytes = 0;
  int64_t q8Bytes = 0;
  int64_t outputBytes = 0;
  if (!getKQuantBlockElements(gateFormat, gateBlockElements) ||
      !getKQuantBlockElements(upFormat, upBlockElements) ||
      K % gateBlockElements != 0 || K % upBlockElements != 0 ||
      !getKQuantBlockBytes(gateFormat, gateBlockBytes) ||
      !getKQuantBlockBytes(upFormat, upBlockBytes) ||
      !checkedMultiply(N, K / gateBlockElements, gateBlocksPerWeightRow) ||
      !checkedMultiply(gateBlocksPerWeightRow, gateBlockBytes,
                       expectedGateBytes) ||
      !checkedMultiply(N, K / upBlockElements, upBlocksPerWeightRow) ||
      !checkedMultiply(upBlocksPerWeightRow, upBlockBytes, expectedUpBytes) ||
      expectedGateBytes != gateWeightBytes ||
      expectedUpBytes != upWeightBytes ||
      !checkedMultiply(q8.dim0, q8.dim1, batchCount) ||
      !checkedMultiply(batchCount, q8.dim2, rowCount) ||
      !checkedMultiply(rowCount, N, outputElements) ||
      !checkedMultiply(K / kQ8BlockElements, kQ8BlockBytes, q8RowBytes) ||
      !checkedMultiply(rowCount, q8RowBytes, q8Bytes) ||
      !checkedMultiply(outputElements, int64_t{sizeof(float)}, outputBytes)) {
    return false;
  }
  if (static_cast<uint64_t>(q8Bytes) > std::numeric_limits<size_t>::max() ||
      static_cast<uint64_t>(expectedGateBytes) >
          std::numeric_limits<size_t>::max() ||
      static_cast<uint64_t>(expectedUpBytes) >
          std::numeric_limits<size_t>::max() ||
      static_cast<uint64_t>(outputBytes) > std::numeric_limits<size_t>::max()) {
    return false;
  }

  layout = {batchCount,
            q8.dim2,
            outputElements,
            static_cast<size_t>(q8Bytes),
            static_cast<size_t>(expectedGateBytes),
            static_cast<size_t>(expectedUpBytes),
            static_cast<size_t>(outputBytes)};
  return true;
}

inline bool
hasDisjointKQuantSwiGluQ8Output(const void *q8, const void *gateWeights,
                                const void *upWeights, const float *output,
                                const void *scratch,
                                const KQuantSwiGluQ8Layout &layout) {
  if (rangesOverlap(output, layout.outputBytes, q8, layout.q8Bytes) ||
      rangesOverlap(output, layout.outputBytes, gateWeights,
                    layout.gateWeightBytes) ||
      rangesOverlap(output, layout.outputBytes, upWeights,
                    layout.upWeightBytes)) {
    return false;
  }
  if (!scratch)
    return true;
  return !rangesOverlap(scratch, layout.outputBytes, output,
                        layout.outputBytes) &&
         !rangesOverlap(scratch, layout.outputBytes, q8, layout.q8Bytes) &&
         !rangesOverlap(scratch, layout.outputBytes, gateWeights,
                        layout.gateWeightBytes) &&
         !rangesOverlap(scratch, layout.outputBytes, upWeights,
                        layout.upWeightBytes);
}

} // namespace hipdnn_ep::kquant

#endif // HIPDNN_EP_KQUANT_CONTRACT_H
