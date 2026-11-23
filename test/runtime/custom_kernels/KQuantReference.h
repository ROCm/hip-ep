/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Q4_K, Q5_K, Q6_K, and Q8_1 storage layouts are adapted from
// ggml-common.h in llama.cpp at commit
// b21e4de74567f5eef213765c9476a843c2e43f0d. llama.cpp is Copyright (c)
// 2023-2026 The ggml authors and is distributed under the MIT License; see
// lib/Runtime/Kernels/licenses/llama.cpp/LICENSE.

#ifndef HIP_TEST_RUNTIME_KQUANT_KQUANTREFERENCE_H
#define HIP_TEST_RUNTIME_KQUANT_KQUANTREFERENCE_H

#include "kquant_abi.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace hip::kquant {

inline constexpr std::size_t kKQuantBlockElements = HIP_KQUANT_BLOCK_ELEMENTS;
inline constexpr std::size_t kQ4KScaleBytes = HIP_KQUANT_Q4_K_SCALE_BYTES;
inline constexpr std::size_t kQ8_1BlockElements = HIP_Q8_1_BLOCK_ELEMENTS;
inline constexpr std::size_t kQ8_1BlockBytes = HIP_Q8_1_BLOCK_SIZE;
inline constexpr std::size_t kQ8_1BlockAlignment = HIP_Q8_1_BLOCK_ALIGNMENT;

// Raw IEEE binary16 bits, in the host's (little-endian) byte order.
using Fp16Bits = std::uint16_t;

// Q4_K represents 256 values as eight 32-value sub-blocks. Each sub-block has
// a six-bit scale and minimum encoded in `scales`; `d` and `dmin` scale those
// metadata values. `qs` contains the two four-bit values in each byte.
using Q4KBlock = hip_kquant_q4_k_block_t;

// Q5_K adds one high bit per value to the Q4_K-style low nibbles while using
// the same embedded six-bit scale/minimum metadata.
using Q5KBlock = hip_kquant_q5_k_block_t;

// Q6_K represents 256 values as sixteen 16-value sub-blocks. `ql` stores four
// low bits per value, `qh` stores two high bits, and `scales` contains a signed
// scale for each sub-block. `d` is the common binary16 scale.
using Q6KBlock = hip_kquant_q6_k_block_t;
using Q8_1Block = hip_q8_1_block_t;

static_assert(std::is_standard_layout_v<Q4KBlock>);
static_assert(std::is_trivially_copyable_v<Q4KBlock>);
static_assert(std::is_standard_layout_v<Q6KBlock>);
static_assert(std::is_trivially_copyable_v<Q6KBlock>);
static_assert(std::is_standard_layout_v<Q5KBlock>);
static_assert(std::is_trivially_copyable_v<Q5KBlock>);
static_assert(std::is_standard_layout_v<Q8_1Block>);
static_assert(std::is_trivially_copyable_v<Q8_1Block>);

// Converts raw IEEE binary16 bits to binary32 without relying on a host half
// type. This is part of the scalar reference contract, not a fast path.
[[nodiscard]] float fp16BitsToFloat(Fp16Bits bits) noexcept;

// Scalar correctness oracles for one packed block. `output` must provide 256
// floats. The operations intentionally mirror ggml's decoding order.
void dequantizeQ4KBlock(const Q4KBlock &block, float *output) noexcept;
void dequantizeQ5KBlock(const Q5KBlock &block, float *output) noexcept;
void dequantizeQ6KBlock(const Q6KBlock &block, float *output) noexcept;

// Row helpers. `output` must provide blockCount times 256 floats.
void dequantizeQ4K(const Q4KBlock *blocks, std::size_t blockCount,
                   float *output) noexcept;
void dequantizeQ5K(const Q5KBlock *blocks, std::size_t blockCount,
                   float *output) noexcept;
void dequantizeQ6K(const Q6KBlock *blocks, std::size_t blockCount,
                   float *output) noexcept;

// Deliberately scalar dequantize-then-F32-dot references for validating basic
// packed-byte decoding and indexing. These do not model llama.cpp's Q8_K CPU
// activation path or Q8_1 GPU MMVQ activation path and are not intended for
// model execution.
[[nodiscard]] float referenceDotQ4K(const Q4KBlock *blocks,
                                    std::size_t blockCount,
                                    const float *rhs) noexcept;
[[nodiscard]] float referenceDotQ5K(const Q5KBlock *blocks,
                                    std::size_t blockCount,
                                    const float *rhs) noexcept;
[[nodiscard]] float referenceDotQ6K(const Q6KBlock *blocks,
                                    std::size_t blockCount,
                                    const float *rhs) noexcept;

} // namespace hip::kquant

#endif // HIP_TEST_RUNTIME_KQUANT_KQUANTREFERENCE_H
