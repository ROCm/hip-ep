/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// The K-quant weight and Q8_1 activation layouts the hip.kquant_* operations
// read, as the compiler sees them (hip/kquant_abi.h).

#ifndef HIP_DIALECT_IR_HIPKQUANT_H
#define HIP_DIALECT_IR_HIPKQUANT_H

#include "hip/kquant_abi.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace mlir::hip {
namespace kquant {

enum class Format { Q4K, Q5K, Q6K };

/// Values a K-quant block holds.
inline constexpr int64_t kBlockElements = HIP_KQUANT_BLOCK_ELEMENTS;
inline constexpr int64_t kQ8_1BlockElements = HIP_Q8_1_BLOCK_ELEMENTS;
inline constexpr int64_t kQ8_1BlockBytes = HIP_Q8_1_BLOCK_SIZE;

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
    return HIP_KQUANT_Q4_K_BLOCK_SIZE;
  case Format::Q5K:
    return HIP_KQUANT_Q5_K_BLOCK_SIZE;
  case Format::Q6K:
    return HIP_KQUANT_Q6_K_BLOCK_SIZE;
  }
  return 0;
}

/// The format's identifier in the kernels' ABI.
inline constexpr int64_t getKernelFormatId(Format format) {
  switch (format) {
  case Format::Q4K:
    return HIP_KQUANT_FORMAT_Q4_K;
  case Format::Q5K:
    return HIP_KQUANT_FORMAT_Q5_K;
  case Format::Q6K:
    return HIP_KQUANT_FORMAT_Q6_K;
  }
  return -1;
}

} // namespace kquant

/// The head dimensions hip.token_major_gqa has a kernel for
/// (hip_token_major_gqa_head_dimension_valid in hip_custom_kernels.h).
inline constexpr bool isTokenMajorGqaHeadDimension(int64_t headDimension) {
  return headDimension == 40 || headDimension == 64 || headDimension == 72 ||
         headDimension == 80 || headDimension == 96 || headDimension == 112 ||
         headDimension == 128 || headDimension == 256;
}

} // namespace mlir::hip

#endif // HIP_DIALECT_IR_HIPKQUANT_H
