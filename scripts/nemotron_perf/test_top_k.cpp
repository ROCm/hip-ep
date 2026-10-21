// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "hip_custom_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <hip/hip_runtime.h>
#include <limits>
#include <stdexcept>
#include <vector>

extern "C" int hip_top_k_baseline(void *, const void *, void *, void *, int64_t,
                                  int64_t, int64_t, int64_t, const int64_t *,
                                  int64_t, int);
static void check(hipError_t error) {
  if (error != hipSuccess)
    throw std::runtime_error(hipGetErrorString(error));
}
struct Buffer {
  void *data = nullptr;
  explicit Buffer(size_t bytes) { check(hipMalloc(&data, bytes)); }
  ~Buffer() {
    if (data)
      (void)hipFree(data);
  }
};
static void test(int rows, int width, int k, bool largest, bool special) {
  std::vector<float> input(rows * width), result(rows * k), baseline(rows * k);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = (int((i * 17 + 13) % 53) - 26) * 0.125f;
  if (special) {
    for (int row = 0; row < rows; ++row) {
      input[row * width] = std::numeric_limits<float>::quiet_NaN();
      input[row * width + width / 2] = std::numeric_limits<float>::quiet_NaN();
      input[row * width + 1] = std::numeric_limits<float>::infinity();
      input[row * width + 2] = -std::numeric_limits<float>::infinity();
      input[row * width + 3] = -0.0f;
      input[row * width + 4] = 0.0f;
    }
  }
  Buffer data(input.size() * 4), a(result.size() * 4), b(result.size() * 4);
  Buffer ai(result.size() * 8), bi(result.size() * 8);
  check(hipMemcpy(data.data, input.data(), input.size() * 4,
                  hipMemcpyHostToDevice));
  int64_t shape[] = {1, rows, width};
  hipStream_t stream;
  check(hipStreamCreate(&stream));
  auto run = [&](bool old) {
    int rc = old ? hip_top_k_baseline(stream, data.data, a.data, ai.data, 2,
                                      largest, 1, 3, shape, k, 4)
                 : hip_top_k(stream, data.data, b.data, bi.data, 2, largest, 1,
                             3, shape, k, 4);
    check(static_cast<hipError_t>(rc));
  };
  run(true);
  run(false);
  check(hipStreamSynchronize(stream));
  check(hipMemcpy(baseline.data(), a.data, baseline.size() * 4,
                  hipMemcpyDeviceToHost));
  check(hipMemcpy(result.data(), b.data, result.size() * 4,
                  hipMemcpyDeviceToHost));
  std::vector<int64_t> oldIds(result.size()), newIds(result.size());
  check(hipMemcpy(oldIds.data(), ai.data, oldIds.size() * 8,
                  hipMemcpyDeviceToHost));
  check(hipMemcpy(newIds.data(), bi.data, newIds.size() * 8,
                  hipMemcpyDeviceToHost));
  if (oldIds != newIds ||
      std::memcmp(baseline.data(), result.data(), result.size() * 4))
    throw std::runtime_error("TopK differs from the original kernel");
  // A separate serial oracle checks tie and NaN behavior.
  for (int row = 0; row < rows; ++row) {
    std::vector<bool> used(width, false);
    for (int pick = 0; pick < k; ++pick) {
      int best = -1;
      for (int i = 0; i < width; ++i)
        if (!used[i] &&
            (best < 0 ||
             (largest ? input[row * width + i] > input[row * width + best]
                      : input[row * width + i] < input[row * width + best])))
          best = i;
      if (newIds[row * k + pick] != best)
        throw std::runtime_error("TopK differs from oracle");
      used[best] = true;
    }
  }
  float times[2];
  hipEvent_t start, end;
  check(hipEventCreate(&start));
  check(hipEventCreate(&end));
  for (int version = 0; version < 2; ++version) {
    check(hipEventRecord(start, stream));
    for (int repeat = 0; repeat < 20; ++repeat)
      run(version == 0);
    check(hipEventRecord(end, stream));
    check(hipEventSynchronize(end));
    check(hipEventElapsedTime(&times[version], start, end));
  }
  std::printf("PASS rows=%d width=%d k=%d largest=%d special=%d "
              "baseline_ms=%.6f candidate_ms=%.6f\n",
              rows, width, k, largest, special, times[0] / 20, times[1] / 20);
  check(hipEventDestroy(start));
  check(hipEventDestroy(end));
  check(hipStreamDestroy(stream));
}
int main() {
  try {
    for (bool largest : {false, true})
      for (bool special : {false, true}) {
        test(1, 128, 6, largest, special);
        test(5, 255, 16, largest, special);
        test(3, 256, 16, largest, special);
        test(2, 257, 17, largest, special);
      }
    test(512, 128, 6, true, false);
    test(1, 8, 8, true, true);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
