/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- KQuantTestSupport.h - K-quant test helpers ---------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_TEST_RUNTIME_KQUANT_KQUANTTESTSUPPORT_H
#define HIP_TEST_RUNTIME_KQUANT_KQUANTTESTSUPPORT_H

#include "KQuantReference.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace hipdnn_ep::test::kquant {

using hip::kquant::Q4KBlock;
using hip::kquant::Q5KBlock;
using hip::kquant::Q6KBlock;
using hip::kquant::Q8_1Block;

inline std::uint16_t floatToHalfBits(float value) {
  const _Float16 half = static_cast<_Float16>(value);
  std::uint16_t bits;
  std::memcpy(&bits, &half, sizeof(bits));
  return bits;
}

inline void unpackQ4KScaleMin(std::size_t group, const std::uint8_t *packed,
                              std::uint8_t &scale, std::uint8_t &minimum) {
  if (group < 4) {
    scale = packed[group] & 0x3f;
    minimum = packed[group + 4] & 0x3f;
    return;
  }
  scale = (packed[group + 4] & 0x0f) | ((packed[group - 4] >> 6) << 4);
  minimum = (packed[group + 4] >> 4) | ((packed[group] >> 6) << 4);
}

inline int unpackQ6K(const Q6KBlock &block, int index) {
  const int half = index / 128;
  const int indexIn128 = index % 128;
  const int quarter = indexIn128 / 32;
  const int lane = indexIn128 % 32;
  const int lowIndex = half * 64 + lane + ((quarter & 1) ? 32 : 0);
  const std::uint8_t lowByte = block.ql[lowIndex];
  const int low = quarter < 2 ? lowByte & 0x0f : lowByte >> 4;
  const int high = (block.qh[half * 32 + lane] >> (2 * quarter)) & 0x03;
  return (low | (high << 4)) - 32;
}

inline int unpackQ5K(const Q5KBlock &block, int index) {
  const int group = index / 32;
  const int lane = index % 32;
  const std::uint8_t packed = block.qs[(group / 2) * 32 + lane];
  const int low = (group & 1) != 0 ? packed >> 4 : packed & 0x0f;
  return low + ((block.qh[lane] & (1U << group)) != 0 ? 16 : 0);
}

template <typename Block>
inline std::vector<Block> makeKQuantWeights(std::int64_t count,
                                            std::uint32_t seedOffset) {
  static_assert(std::is_same_v<Block, Q4KBlock> ||
                std::is_same_v<Block, Q5KBlock> ||
                std::is_same_v<Block, Q6KBlock>);

  std::vector<Block> blocks(static_cast<std::size_t>(count));
  for (std::int64_t index = 0; index < count; ++index) {
    Block &block = blocks[static_cast<std::size_t>(index)];
    const std::uint32_t seed = static_cast<std::uint32_t>(index) + seedOffset;
    if constexpr (std::is_same_v<Block, Q4KBlock>) {
      constexpr std::array<std::uint16_t, 4> deltas = {0x3800, 0x3555, 0x3c00,
                                                       0x2e66};
      constexpr std::array<std::uint16_t, 4> minimums = {0x3400, 0x3000, 0x399a,
                                                         0x2800};
      block.d = deltas[seed % deltas.size()];
      block.dmin = minimums[(seed * 3) % minimums.size()];
      for (std::size_t i = 0; i < hip::kquant::kQ4KScaleBytes; ++i)
        block.scales[i] = static_cast<std::uint8_t>(37 * i + 53 * seed + 11);
      for (std::size_t i = 0; i < hip::kquant::kKQuantBlockElements / 2; ++i)
        block.qs[i] = static_cast<std::uint8_t>(29 * i + 17 * seed + 7);
    } else if constexpr (std::is_same_v<Block, Q5KBlock>) {
      constexpr std::array<std::uint16_t, 4> deltas = {0x3800, 0x3555, 0x3c00,
                                                       0x2e66};
      constexpr std::array<std::uint16_t, 4> minimums = {0x3400, 0x3000, 0x399a,
                                                         0x2800};
      block.d = deltas[seed % deltas.size()];
      block.dmin = minimums[(seed * 3) % minimums.size()];
      for (std::size_t i = 0; i < hip::kquant::kQ4KScaleBytes; ++i)
        block.scales[i] = static_cast<std::uint8_t>(37 * i + 53 * seed + 11);
      for (std::size_t i = 0; i < hip::kquant::kKQuantBlockElements / 8; ++i)
        block.qh[i] = static_cast<std::uint8_t>(41 * i + 23 * seed + 5);
      for (std::size_t i = 0; i < hip::kquant::kKQuantBlockElements / 2; ++i)
        block.qs[i] = static_cast<std::uint8_t>(29 * i + 17 * seed + 7);
    } else {
      constexpr std::array<std::uint16_t, 4> deltas = {0x3400, 0xb555, 0x3a00,
                                                       0x2c00};
      block.d = deltas[seed % deltas.size()];
      for (std::size_t i = 0; i < hip::kquant::kKQuantBlockElements / 2; ++i)
        block.ql[i] = static_cast<std::uint8_t>(43 * i + 19 * seed + 3);
      for (std::size_t i = 0; i < hip::kquant::kKQuantBlockElements / 4; ++i)
        block.qh[i] = static_cast<std::uint8_t>(71 * i + 31 * seed + 5);
      for (std::size_t i = 0; i < hip::kquant::kKQuantBlockElements / 16; ++i) {
        const int scale = static_cast<int>((29 * i + 47 * seed) % 255) - 127;
        block.scales[i] = static_cast<std::int8_t>(scale);
      }
    }
  }
  return blocks;
}

template <typename Block>
inline float referenceDotQ8_1(const Block *weights, std::size_t blockCount,
                              const Q8_1Block *activation,
                              bool integerSum = false) {
  static_assert(std::is_same_v<Block, Q4KBlock> ||
                std::is_same_v<Block, Q5KBlock> ||
                std::is_same_v<Block, Q6KBlock>);

  float result = 0.0F;
  for (std::size_t blockIndex = 0; blockIndex < blockCount; ++blockIndex) {
    const Block &weight = weights[blockIndex];
    for (std::size_t group = 0; group < 8; ++group) {
      const Q8_1Block &q8 = activation[blockIndex * 8 + group];
      float activationSum = hip::kquant::fp16BitsToFloat(q8.s);
      if (integerSum) {
        // The quants' own sum, whatever the block stores.
        int quantSum = 0;
        for (int8_t quant : q8.qs)
          quantSum += quant;
        activationSum =
            hip::kquant::fp16BitsToFloat(q8.d) * static_cast<float>(quantSum);
      }
      if constexpr (std::is_same_v<Block, Q4KBlock>) {
        const std::size_t pair = group / 2;
        const bool highNibble = (group & 1) != 0;
        int dot = 0;
        for (std::size_t index = 0; index < 32; ++index) {
          const std::uint8_t packed = weight.qs[pair * 32 + index];
          const int quant = highNibble ? packed >> 4 : packed & 0x0f;
          dot += static_cast<int>(q8.qs[index]) * quant;
        }
        std::uint8_t scale;
        std::uint8_t minimum;
        unpackQ4KScaleMin(group, weight.scales, scale, minimum);
        result += hip::kquant::fp16BitsToFloat(weight.d) *
                      hip::kquant::fp16BitsToFloat(q8.d) *
                      static_cast<float>(scale * dot) -
                  hip::kquant::fp16BitsToFloat(weight.dmin) *
                      static_cast<float>(minimum) * activationSum;
      } else if constexpr (std::is_same_v<Block, Q5KBlock>) {
        int dot = 0;
        for (int index = 0; index < 32; ++index) {
          dot += static_cast<int>(q8.qs[index]) *
                 unpackQ5K(weight, static_cast<int>(group * 32) + index);
        }
        std::uint8_t scale;
        std::uint8_t minimum;
        unpackQ4KScaleMin(group, weight.scales, scale, minimum);
        result += hip::kquant::fp16BitsToFloat(weight.d) *
                      hip::kquant::fp16BitsToFloat(q8.d) *
                      static_cast<float>(scale * dot) -
                  hip::kquant::fp16BitsToFloat(weight.dmin) *
                      static_cast<float>(minimum) * activationSum;
      } else {
        int lowDot = 0;
        int highDot = 0;
        for (int index = 0; index < 16; ++index) {
          lowDot += static_cast<int>(q8.qs[index]) *
                    unpackQ6K(weight, static_cast<int>(group * 32) + index);
          highDot +=
              static_cast<int>(q8.qs[index + 16]) *
              unpackQ6K(weight, static_cast<int>(group * 32) + index + 16);
        }
        const int scaleIndex = static_cast<int>(group * 2);
        const int scaledDot = weight.scales[scaleIndex] * lowDot +
                              weight.scales[scaleIndex + 1] * highDot;
        result += hip::kquant::fp16BitsToFloat(weight.d) *
                  hip::kquant::fp16BitsToFloat(q8.d) *
                  static_cast<float>(scaledDot);
      }
    }
  }
  return result;
}

} // namespace hipdnn_ep::test::kquant

#endif // HIP_TEST_RUNTIME_KQUANT_KQUANTTESTSUPPORT_H
