/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// The K-quant weight and Q8_1 activation layouts the hip.kquant_* operations
// read, as the compiler sees them. The kernels' view of the same layouts is
// lib/Runtime/Kernels/include/kquant_abi.h; the two must agree.

#ifndef HIP_DIALECT_IR_HIPKQUANT_H
#define HIP_DIALECT_IR_HIPKQUANT_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace mlir::hip {
namespace kquant {

enum class Format { Q4K, Q5K, Q6K };

/// Values a K-quant block holds.
inline constexpr int64_t kBlockElements = 256;
inline constexpr int64_t kQ8_1BlockElements = 32;
inline constexpr int64_t kQ8_1BlockBytes = 36;

/// What a Q8_1 block's sum field holds (`sum_mode`).
inline constexpr int64_t kSumPrequantization = 0;
inline constexpr int64_t kSumScaleTimesQuantized = 1;
inline constexpr int64_t kSumMatrixScaledVectorInteger = 2;

inline std::optional<Format> parseFormat(llvm::StringRef name) {
  if (name == "q4_k")
    return Format::Q4K;
  if (name == "q5_k")
    return Format::Q5K;
  if (name == "q6_k")
    return Format::Q6K;
  return std::nullopt;
}

inline constexpr llvm::StringLiteral getFormatName(Format format) {
  switch (format) {
  case Format::Q4K:
    return "q4_k";
  case Format::Q5K:
    return "q5_k";
  case Format::Q6K:
    return "q6_k";
  }
  return "";
}

inline constexpr int64_t getBlockBytes(Format format) {
  switch (format) {
  case Format::Q4K:
    return 144;
  case Format::Q5K:
    return 176;
  case Format::Q6K:
    return 210;
  }
  return 0;
}

/// The format's identifier in the kernels' ABI (kquant_abi.h).
inline constexpr int64_t getKernelFormatId(Format format) {
  switch (format) {
  case Format::Q4K:
    return 0;
  case Format::Q6K:
    return 1;
  case Format::Q5K:
    return 2;
  }
  return -1;
}

} // namespace kquant

/// The head dimensions hip.token_major_gqa has a kernel for.
inline constexpr bool isTokenMajorGqaHeadDimension(int64_t headDimension) {
  return headDimension == 40 || headDimension == 64 || headDimension == 72 ||
         headDimension == 80 || headDimension == 96 || headDimension == 112 ||
         headDimension == 128 || headDimension == 256;
}

} // namespace mlir::hip

#endif // HIP_DIALECT_IR_HIPKQUANT_H
