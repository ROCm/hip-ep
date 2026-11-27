/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// The scalar Q4_K/Q5_K/Q6_K decoding formulas are adapted from ggml-quants.c
// and ggml-impl.h in llama.cpp at commit
// b21e4de74567f5eef213765c9476a843c2e43f0d. llama.cpp is Copyright (c)
// 2023-2026 The ggml authors and is distributed under the MIT License; see
// lib/Runtime/Kernels/licenses/llama.cpp/LICENSE.

#include "KQuantReference.h"

#include <array>
#include <cstring>

namespace hip::kquant {
namespace {

float floatFromBits(std::uint32_t bits) noexcept {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::uint32_t floatToBits(float value) noexcept {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

void getQ4KScaleAndMin(std::size_t index, const std::uint8_t *packed,
                       std::uint8_t &scale, std::uint8_t &minimum) noexcept {
  if (index < 4) {
    scale = packed[index] & 0x3f;
    minimum = packed[index + 4] & 0x3f;
    return;
  }

  scale = (packed[index + 4] & 0x0f) |
          static_cast<std::uint8_t>((packed[index - 4] >> 6) << 4);
  minimum = (packed[index + 4] >> 4) |
            static_cast<std::uint8_t>((packed[index] >> 6) << 4);
}

template <typename Block, typename Decode>
float referenceDot(const Block *blocks, std::size_t blockCount,
                   const float *rhs, Decode decode) noexcept {
  std::array<float, kKQuantBlockElements> decoded;
  float result = 0.0F;
  for (std::size_t blockIndex = 0; blockIndex < blockCount; ++blockIndex) {
    decode(blocks[blockIndex], decoded.data());
    const std::size_t elementBase = blockIndex * kKQuantBlockElements;
    for (std::size_t i = 0; i < kKQuantBlockElements; ++i)
      result += decoded[i] * rhs[elementBase + i];
  }
  return result;
}

} // namespace

float fp16BitsToFloat(Fp16Bits bits) noexcept {
  // Keep this operation order identical to ggml_compute_fp16_to_fp32. Besides
  // normal and subnormal numbers, that also gives the same quieting behavior
  // for every binary16 NaN payload.
  const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16;
  const std::uint32_t sign = word & 0x80000000U;
  const std::uint32_t twiceWord = word + word;

  const std::uint32_t exponentOffset = 0xe0U << 23;
  const float exponentScale = 0x1.0p-112F;
  const float normalized =
      floatFromBits((twiceWord >> 4) + exponentOffset) * exponentScale;

  const std::uint32_t magicMask = 126U << 23;
  const float denormalized =
      floatFromBits((twiceWord >> 17) | magicMask) - 0.5F;

  const std::uint32_t denormalizedCutoff = 1U << 27;
  return floatFromBits(sign | (twiceWord < denormalizedCutoff
                                   ? floatToBits(denormalized)
                                   : floatToBits(normalized)));
}

void dequantizeQ4KBlock(const Q4KBlock &block, float *output) noexcept {
  const float d = fp16BitsToFloat(block.d);
  const float dmin = fp16BitsToFloat(block.dmin);

  std::size_t scaleIndex = 0;
  for (std::size_t group = 0; group < kKQuantBlockElements; group += 64) {
    std::uint8_t scale;
    std::uint8_t minimum;
    getQ4KScaleAndMin(scaleIndex++, block.scales, scale, minimum);
    const float lowScale = d * scale;
    const float lowMinimum = dmin * minimum;

    getQ4KScaleAndMin(scaleIndex++, block.scales, scale, minimum);
    const float highScale = d * scale;
    const float highMinimum = dmin * minimum;

    const std::uint8_t *quants = block.qs + group / 2;
    for (std::size_t i = 0; i < 32; ++i)
      output[group + i] = lowScale * (quants[i] & 0x0fU) - lowMinimum;
    for (std::size_t i = 0; i < 32; ++i)
      output[group + 32 + i] = highScale * (quants[i] >> 4) - highMinimum;
  }
}

void dequantizeQ5KBlock(const Q5KBlock &block, float *output) noexcept {
  const float d = fp16BitsToFloat(block.d);
  const float dmin = fp16BitsToFloat(block.dmin);

  for (std::size_t group = 0; group < 8; ++group) {
    std::uint8_t scale;
    std::uint8_t minimum;
    getQ4KScaleAndMin(group, block.scales, scale, minimum);
    const float groupScale = d * scale;
    const float groupMinimum = dmin * minimum;
    const std::size_t packedBase = (group / 2) * 32;
    const std::uint8_t highMask = static_cast<std::uint8_t>(1U << group);
    for (std::size_t index = 0; index < 32; ++index) {
      const std::uint8_t packed = block.qs[packedBase + index];
      const int low = (group & 1) != 0 ? packed >> 4 : packed & 0x0fU;
      const int quant = low + ((block.qh[index] & highMask) != 0 ? 16 : 0);
      output[group * 32 + index] =
          groupScale * static_cast<float>(quant) - groupMinimum;
    }
  }
}

void dequantizeQ6KBlock(const Q6KBlock &block, float *output) noexcept {
  const float d = fp16BitsToFloat(block.d);

  for (std::size_t group = 0; group < kKQuantBlockElements; group += 128) {
    const std::uint8_t *low = block.ql + group / 2;
    const std::uint8_t *high = block.qh + group / 4;
    const std::int8_t *scales = block.scales + group / 16;

    for (std::size_t i = 0; i < 32; ++i) {
      const std::size_t scaleIndex = i / 16;
      const std::int8_t q0 = static_cast<std::int8_t>(
          (low[i] & 0x0fU) | (((high[i] >> 0) & 0x03U) << 4));
      const std::int8_t q1 = static_cast<std::int8_t>(
          (low[i + 32] & 0x0fU) | (((high[i] >> 2) & 0x03U) << 4));
      const std::int8_t q2 = static_cast<std::int8_t>(
          (low[i] >> 4) | (((high[i] >> 4) & 0x03U) << 4));
      const std::int8_t q3 = static_cast<std::int8_t>(
          (low[i + 32] >> 4) | (((high[i] >> 6) & 0x03U) << 4));

      output[group + i] =
          d * scales[scaleIndex + 0] * static_cast<std::int8_t>(q0 - 32);
      output[group + 32 + i] =
          d * scales[scaleIndex + 2] * static_cast<std::int8_t>(q1 - 32);
      output[group + 64 + i] =
          d * scales[scaleIndex + 4] * static_cast<std::int8_t>(q2 - 32);
      output[group + 96 + i] =
          d * scales[scaleIndex + 6] * static_cast<std::int8_t>(q3 - 32);
    }
  }
}

void dequantizeQ4K(const Q4KBlock *blocks, std::size_t blockCount,
                   float *output) noexcept {
  for (std::size_t i = 0; i < blockCount; ++i)
    dequantizeQ4KBlock(blocks[i], output + i * kKQuantBlockElements);
}

void dequantizeQ5K(const Q5KBlock *blocks, std::size_t blockCount,
                   float *output) noexcept {
  for (std::size_t i = 0; i < blockCount; ++i)
    dequantizeQ5KBlock(blocks[i], output + i * kKQuantBlockElements);
}

void dequantizeQ6K(const Q6KBlock *blocks, std::size_t blockCount,
                   float *output) noexcept {
  for (std::size_t i = 0; i < blockCount; ++i)
    dequantizeQ6KBlock(blocks[i], output + i * kKQuantBlockElements);
}

float referenceDotQ4K(const Q4KBlock *blocks, std::size_t blockCount,
                      const float *rhs) noexcept {
  return referenceDot(blocks, blockCount, rhs, dequantizeQ4KBlock);
}

float referenceDotQ5K(const Q5KBlock *blocks, std::size_t blockCount,
                      const float *rhs) noexcept {
  return referenceDot(blocks, blockCount, rhs, dequantizeQ5KBlock);
}

float referenceDotQ6K(const Q6KBlock *blocks, std::size_t blockCount,
                      const float *rhs) noexcept {
  return referenceDot(blocks, blockCount, rhs, dequantizeQ6KBlock);
}

} // namespace hip::kquant
