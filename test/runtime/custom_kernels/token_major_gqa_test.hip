/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipTestSupport.h"
#include "KQuantReference.h"
#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr char kTestName[] = "token_major_gqa_test";
using hipdnn_ep::test::DeviceBuffer;

constexpr float kGuard = -9876.5F;

bool checkHip(hipError_t error, const char *operation) {
  return hipdnn_ep::test::checkHip(error, operation, kTestName);
}

uint16_t floatToHalfBits(float value) {
  const _Float16 half = static_cast<_Float16>(value);
  uint16_t bits;
  std::memcpy(&bits, &half, sizeof(bits));
  return bits;
}

float halfBitsToFloat(uint16_t bits) {
  return hip::kquant::fp16BitsToFloat(bits);
}

// Token-major: [batch, capacity, kvHeads, headDimension].
size_t cacheOffset(int64_t batch, int64_t token, int64_t head, int64_t channel,
                   int64_t cacheCapacity, int64_t kvHeads,
                   int64_t headDimension) {
  return static_cast<size_t>(
      ((batch * cacheCapacity + token) * kvHeads + head) * headDimension +
      channel);
}

std::vector<float> referenceAttention(
    const std::vector<float> &query, const std::vector<uint16_t> &key,
    const std::vector<uint16_t> &value, const std::vector<uint16_t> &mask,
    int64_t batchCount, int64_t queryTokens, int64_t queryHeads,
    int64_t kvHeads, int64_t cacheCapacity, int64_t headDimension,
    int64_t maskBatchCount, float scale) {
  std::vector<float> output(static_cast<size_t>(batchCount * queryTokens *
                                                queryHeads * headDimension));
  const int64_t headsPerGroup = queryHeads / kvHeads;
  std::vector<float> scores(static_cast<size_t>(cacheCapacity));

  for (int64_t batch = 0; batch < batchCount; ++batch) {
    const int64_t maskBatch = maskBatchCount == 1 ? 0 : batch;
    for (int64_t token = 0; token < queryTokens; ++token) {
      for (int64_t head = 0; head < queryHeads; ++head) {
        const int64_t kvHead = head / headsPerGroup;
        const size_t outputBase = static_cast<size_t>(
            ((batch * queryTokens + token) * queryHeads + head) *
            headDimension);
        float maximum = -std::numeric_limits<float>::infinity();
        bool hasScore = false;
        for (int64_t cacheToken = 0; cacheToken < cacheCapacity; ++cacheToken) {
          float dot = 0.0F;
          for (int64_t channel = 0; channel < headDimension; ++channel) {
            const size_t queryIndex = static_cast<size_t>(
                ((batch * queryTokens + token) * queryHeads + head) *
                    headDimension +
                channel);
            const size_t keyIndex =
                cacheOffset(batch, cacheToken, kvHead, channel,
                            cacheCapacity, kvHeads, headDimension);
            dot = std::fma(query[queryIndex], halfBitsToFloat(key[keyIndex]),
                           dot);
          }
          const size_t maskIndex = static_cast<size_t>(
              (maskBatch * queryTokens + token) * cacheCapacity + cacheToken);
          const float score =
              scale * dot + halfBitsToFloat(mask[maskIndex]);
          scores[static_cast<size_t>(cacheToken)] = score;
          if (!(std::isinf(score) && score < 0.0F)) {
            maximum = std::max(maximum, score);
            hasScore = true;
          }
        }

        if (!hasScore) {
          std::fill_n(output.data() + outputBase,
                      static_cast<size_t>(headDimension), 0.0F);
          continue;
        }

        float denominator = 0.0F;
        std::vector<float> accumulator(static_cast<size_t>(headDimension),
                                       0.0F);
        for (int64_t cacheToken = 0; cacheToken < cacheCapacity; ++cacheToken) {
          const float score = scores[static_cast<size_t>(cacheToken)];
          if (std::isinf(score) && score < 0.0F)
            continue;
          const float probability = std::exp(score - maximum);
          denominator += probability;
          for (int64_t channel = 0; channel < headDimension; ++channel) {
            const size_t valueIndex =
                cacheOffset(batch, cacheToken, kvHead, channel,
                            cacheCapacity, kvHeads, headDimension);
            accumulator[static_cast<size_t>(channel)] =
                std::fma(probability, halfBitsToFloat(value[valueIndex]),
                         accumulator[static_cast<size_t>(channel)]);
          }
        }
        for (int64_t channel = 0; channel < headDimension; ++channel) {
          output[outputBase + static_cast<size_t>(channel)] =
              accumulator[static_cast<size_t>(channel)] / denominator;
        }
      }
    }
  }
  return output;
}

// The F32 reference against llama.cpp's tile kernel, which rounds the scaled
// query and the running weighted-V sum to F16 as llama.cpp does: an F16 ulp
// at these magnitudes, not the F32 kernels' 2e-4.
bool closeEnough(float actual, float expected) {
  return std::isfinite(actual) && std::fabs(actual - expected) <=
                                      1.0e-3F + 1.0e-3F * std::fabs(expected);
}

bool runCase(int64_t batchCount, int64_t queryTokens, int64_t queryHeads,
             int64_t kvHeads, int64_t cacheCapacity, int64_t headDimension,
             int64_t maskBatchCount, bool includeFullyMaskedRow,
             const char *name, std::vector<int64_t> writtenRows = {}) {
  const size_t queryElements = static_cast<size_t>(batchCount * queryTokens *
                                                   queryHeads * headDimension);
  const size_t cacheElements =
      static_cast<size_t>(batchCount * cacheCapacity * kvHeads * headDimension);
  const size_t maskElements =
      static_cast<size_t>(maskBatchCount * queryTokens * cacheCapacity);
  const size_t outputElements = queryElements;

  std::vector<float> query(queryElements);
  for (size_t index = 0; index < query.size(); ++index) {
    query[index] = 0.11F * std::sin(static_cast<float>(index + 3) * 0.017F) +
                   0.03F * static_cast<float>(static_cast<int>(index % 9) - 4);
  }

  std::vector<uint16_t> key(cacheElements);
  std::vector<uint16_t> value(cacheElements);
  for (int64_t batch = 0; batch < batchCount; ++batch) {
    for (int64_t token = 0; token < cacheCapacity; ++token) {
      for (int64_t head = 0; head < kvHeads; ++head) {
        for (int64_t channel = 0; channel < headDimension; ++channel) {
          const size_t index =
              cacheOffset(batch, token, head, channel, cacheCapacity, kvHeads,
                          headDimension);
          const float phase = static_cast<float>(13 * batch + 17 * token +
                                                 23 * head + 3 * channel);
          key[index] = floatToHalfBits(0.15F * std::cos(phase * 0.013F));
          value[index] = floatToHalfBits(0.2F * std::sin(phase * 0.011F) +
                                         0.01F * (channel % 5));
        }
      }
    }
  }

  std::vector<uint16_t> mask(maskElements);
  for (int64_t batch = 0; batch < maskBatchCount; ++batch) {
    for (int64_t token = 0; token < queryTokens; ++token) {
      const bool fullyMasked = includeFullyMaskedRow &&
                               batch + 1 == maskBatchCount &&
                               token + 1 == queryTokens;
      const int64_t visibleThrough =
          std::min<int64_t>(cacheCapacity - 1, 3 + batch + 2 * token);
      for (int64_t cacheToken = 0; cacheToken < cacheCapacity; ++cacheToken) {
        float maskValue = fullyMasked || cacheToken > visibleThrough
                              ? -std::numeric_limits<float>::infinity()
                              : (cacheToken == 1 ? -0.25F : 0.0F);
        const size_t index = static_cast<size_t>(
            (batch * queryTokens + token) * cacheCapacity + cacheToken);
        mask[index] = floatToHalfBits(maskValue);
      }
    }
  }

  const float scale = 1.0F / std::sqrt(static_cast<float>(headDimension));
  const std::vector<float> expected = referenceAttention(
      query, key, value, mask, batchCount, queryTokens, queryHeads, kvHeads,
      cacheCapacity, headDimension, maskBatchCount, scale);
  constexpr size_t guardElements = 11;
  std::vector<float> actual(outputElements + guardElements, kGuard);

  const size_t scratchBytes = std::max<size_t>(
      1, hip_token_major_gqa_scratch_size(
             batchCount, queryTokens, queryHeads, kvHeads, cacheCapacity,
             headDimension));
  DeviceBuffer queryDevice(query.size() * sizeof(float));
  DeviceBuffer keyDevice(key.size() * sizeof(uint16_t));
  DeviceBuffer valueDevice(value.size() * sizeof(uint16_t));
  DeviceBuffer maskDevice(mask.size() * sizeof(uint16_t));
  DeviceBuffer outputDevice(actual.size() * sizeof(float));
  DeviceBuffer scratchDevice(scratchBytes);
  DeviceBuffer writtenRowsDevice(writtenRows.empty()
                                     ? sizeof(int64_t)
                                     : writtenRows.size() * sizeof(int64_t));
  if (!queryDevice.get() || !keyDevice.get() || !valueDevice.get() ||
      !maskDevice.get() || !outputDevice.get() || !scratchDevice.get() ||
      !writtenRowsDevice.get()) {
    std::fprintf(stderr, "ggml_flash_attention_test: %s allocation failed\n",
                 name);
    return false;
  }
  if (!checkHip(hipMemcpy(queryDevice.get(), query.data(),
                          query.size() * sizeof(float), hipMemcpyHostToDevice),
                "query H2D") ||
      !checkHip(hipMemcpy(keyDevice.get(), key.data(),
                          key.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
                "key H2D") ||
      !checkHip(hipMemcpy(valueDevice.get(), value.data(),
                          value.size() * sizeof(uint16_t),
                          hipMemcpyHostToDevice),
                "value H2D") ||
      !checkHip(hipMemcpy(maskDevice.get(), mask.data(),
                          mask.size() * sizeof(uint16_t),
                          hipMemcpyHostToDevice),
                "mask H2D") ||
      !checkHip(hipMemcpy(outputDevice.get(), actual.data(),
                          actual.size() * sizeof(float), hipMemcpyHostToDevice),
                "output H2D")) {
    return false;
  }
  if (!writtenRows.empty() &&
      !checkHip(hipMemcpy(writtenRowsDevice.get(), writtenRows.data(),
                          writtenRows.size() * sizeof(int64_t),
                          hipMemcpyHostToDevice),
                "written rows H2D")) {
    return false;
  }

  const int status = hip_token_major_gqa(
      nullptr, static_cast<const float *>(queryDevice.get()), keyDevice.get(),
      valueDevice.get(), maskDevice.get(),
      static_cast<float *>(outputDevice.get()), batchCount, queryTokens,
      queryHeads, kvHeads, cacheCapacity,
      writtenRows.empty()
          ? nullptr
          : static_cast<const int64_t *>(writtenRowsDevice.get()),
      static_cast<int64_t>(writtenRows.size()), headDimension, maskBatchCount,
      scale, scratchDevice.get(), scratchBytes);
  if (status != hipSuccess ||
      !checkHip(hipMemcpy(actual.data(), outputDevice.get(),
                          actual.size() * sizeof(float), hipMemcpyDeviceToHost),
                "output D2H")) {
    std::fprintf(stderr, "ggml_flash_attention_test: %s launch status=%d\n",
                 name, status);
    return false;
  }

  float maximumDifference = 0.0F;
  for (size_t index = 0; index < outputElements; ++index) {
    maximumDifference =
        std::max(maximumDifference, std::fabs(actual[index] - expected[index]));
    if (!closeEnough(actual[index], expected[index])) {
      std::fprintf(stderr,
                   "ggml_flash_attention_test: %s mismatch at %zu: actual=%g "
                   "expected=%g\n",
                   name, index, static_cast<double>(actual[index]),
                   static_cast<double>(expected[index]));
      return false;
    }
  }
  for (size_t index = outputElements; index < actual.size(); ++index) {
    if (actual[index] != kGuard) {
      std::fprintf(stderr,
                   "ggml_flash_attention_test: %s wrote output guard %zu\n",
                   name, index - outputElements);
      return false;
    }
  }

  std::printf("%s passed; max_abs=%g\n", name,
              static_cast<double>(maximumDifference));
  return true;
}

// Calls no tile kernel takes, and malformed ones, fail rather than run.
bool testRejectedArguments() {
  DeviceBuffer buffer(1 << 16);
  if (!buffer.get())
    return false;
  const auto *f32 = static_cast<const float *>(buffer.get());
  const int64_t row = 3;
  const auto invoke = [&](int64_t heads, int64_t kvHeads, int64_t headDimension,
                          float scale, size_t scratchBytes,
                          const void *mask, const int64_t *writtenRows,
                          int64_t writtenRowCount) {
    return hip_token_major_gqa(
        nullptr, f32, buffer.get(), buffer.get(), mask,
        static_cast<float *>(buffer.get()), /*B=*/1, /*T=*/1, heads, kvHeads,
        /*C=*/8, writtenRows, writtenRowCount, headDimension, /*maskB=*/1,
        scale, buffer.get(), scratchBytes);
  };
  const void *mask = buffer.get();
  return invoke(/*H=*/7, /*G=*/2, /*D=*/64, 1.0F, 1 << 16, mask, nullptr, 0) !=
             0 &&
         invoke(/*H=*/8, /*G=*/2, /*D=*/48, 1.0F, 1 << 16, mask, nullptr, 0) !=
             0 &&
         invoke(/*H=*/8, /*G=*/2, /*D=*/512, 1.0F, 1 << 16, mask, nullptr,
                0) != 0 &&
         invoke(/*H=*/8, /*G=*/2, /*D=*/64, 0.0F, 1 << 16, mask, nullptr, 0) !=
             0 &&
         invoke(/*H=*/8, /*G=*/2, /*D=*/64, 1.0F, 1 << 16, nullptr, nullptr,
                0) != 0 &&
         invoke(/*H=*/8, /*G=*/2, /*D=*/64, 1.0F, 1 << 16, mask, nullptr, 1) !=
             0 &&
         invoke(/*H=*/8, /*G=*/2, /*D=*/64, 1.0F, 1 << 16, mask, &row, 0) != 0;
}

} // namespace

int main() {
  const int device = hipdnn_ep::test::findGfx1151Device(kTestName);
  if (device == -1) {
    std::fprintf(stderr,
                 "ggml_flash_attention_test: no gfx1151 device; skipping\n");
    return 77;
  }
  if (device < 0 || !checkHip(hipSetDevice(device), "hipSetDevice"))
    return 1;

  const bool decode =
      runCase(/*B=*/1, /*T=*/1, /*H=*/8, /*G=*/2, /*C=*/33, /*D=*/128,
              /*maskB=*/1, /*fullyMasked=*/false, "decode");
  const bool fullyMaskedDecode =
      runCase(/*B=*/1, /*T=*/1, /*H=*/4, /*G=*/2, /*C=*/7, /*D=*/64,
              /*maskB=*/1, /*fullyMasked=*/true, "fully-masked decode");
  const bool prefill =
      runCase(/*B=*/2, /*T=*/3, /*H=*/8, /*G=*/2, /*C=*/19, /*D=*/64,
              /*maskB=*/2, /*fullyMasked=*/true, "prefill");
  const bool tiledHeads =
      runCase(/*B=*/1, /*T=*/2, /*H=*/24, /*G=*/1, /*C=*/11, /*D=*/96,
              /*maskB=*/1, /*fullyMasked=*/false, "multi-tile grouped heads");
  const bool llamaDecode =
      runCase(/*B=*/1, /*T=*/1, /*H=*/24, /*G=*/8, /*C=*/256, /*D=*/128,
              /*maskB=*/1, /*fullyMasked=*/false, "Llama-sized decode");
  const bool llamaPrefill =
      runCase(/*B=*/1, /*T=*/40, /*H=*/24, /*G=*/8, /*C=*/256, /*D=*/128,
              /*maskB=*/1, /*fullyMasked=*/false, "Llama-sized prefill");
  // Three query heads a KV head: padded to groups of four for a short step.
  const bool paddedGroupDecode =
      runCase(/*B=*/2, /*T=*/2, /*H=*/24, /*G=*/8, /*C=*/256, /*D=*/128,
              /*maskB=*/2, /*fullyMasked=*/false, "padded-group decode");
  const bool paddedWideGroupDecode =
      runCase(/*B=*/1, /*T=*/1, /*H=*/12, /*G=*/2, /*C=*/512, /*D=*/64,
              /*maskB=*/1, /*fullyMasked=*/false, "padded six-head groups");
  const bool wideDecode =
      runCase(/*B=*/1, /*T=*/1, /*H=*/16, /*G=*/2, /*C=*/33, /*D=*/256,
              /*maskB=*/1, /*fullyMasked=*/false, "D=256 decode");
  // The rows a cache update just wrote bound the keys the kernel reads; the
  // mask lets the single decode token see through row 3, and the last prefill
  // token through row 7.
  const bool writtenRowsDecode =
      runCase(/*B=*/1, /*T=*/1, /*H=*/24, /*G=*/8, /*C=*/256, /*D=*/128,
              /*maskB=*/1, /*fullyMasked=*/false, "written-rows decode",
              /*writtenRows=*/{3});
  const bool writtenRowsPrefill =
      runCase(/*B=*/1, /*T=*/3, /*H=*/4, /*G=*/2, /*C=*/64, /*D=*/64,
              /*maskB=*/1, /*fullyMasked=*/false, "written-rows prefill",
              /*writtenRows=*/{5, 6, 7});
  const bool rejected = testRejectedArguments();
  return decode && fullyMaskedDecode && prefill && tiledHeads && llamaDecode &&
                 llamaPrefill && paddedGroupDecode && paddedWideGroupDecode &&
                 wideDecode && writtenRowsDecode &&
                 writtenRowsPrefill && rejected
             ? 0
             : 1;
}
