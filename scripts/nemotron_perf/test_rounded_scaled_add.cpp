/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "rounded_scaled_add.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <limits>
#include <stdexcept>
#include <vector>

using Shape = std::array<int64_t, 4>;
void check(int result) {
  if (result != hipSuccess)
    throw std::runtime_error(hipGetErrorString(hipError_t(result)));
}
template <class T> struct Buffer {
  T *p = nullptr;
  explicit Buffer(size_t n) { check(hipMalloc(&p, n * sizeof(T))); }
  ~Buffer() {
    if (p)
      (void)hipFree(p);
  }
};
int64_t count(Shape s) {
  int64_t n = 1;
  for (auto d : s)
    n *= d;
  return n;
}
void test(Shape bs, Shape xs, Shape ys, Shape os, int lhs, bool special,
          int inputPadding = 0, int outputPadding = 0) {
  size_t n = count(os);
  std::array<std::vector<float>, 3> raw;
  Shape shapes[] = {bs, xs, ys};
  Shape ps;
  for (int axis = 0; axis < 4; ++axis)
    ps[axis] = xs[axis] == 1 ? ys[axis] : xs[axis];
  for (int a = 0; a < 3; ++a) {
    raw[a].resize(count(shapes[a]));
    for (size_t i = 0; i < raw[a].size(); ++i) {
      uint16_t bits = uint16_t(i * (a * 6 + 1) + a * 17);
      __half h;
      std::memcpy(&h, &bits, 2);
      raw[a][i] = special ? __half2float(h) : float(std::sin(i * .13 + a)) * 2;
      if (special && i % 19 == 0) {
        // Include FP32 midpoints, subnormals, infinities and signed zeros.
        const float edges[] = {
            0.f,    -0.f,    65504.f,   65520.f,  1.00048828125f,
            1.e38f, -1.e38f, 0x1p-149f, INFINITY, -INFINITY,
            NAN};
        raw[a][i] = edges[(i / 19 + a) % 11];
      }
    }
  }
  Buffer<float> b(count(bs) + inputPadding), x(count(xs) + inputPadding),
      y(count(ys) + inputPadding), product(count(ps));
  Buffer<__half> bh(count(bs)), ph(count(ps)), reference(n),
      actual(n + outputPadding + 1);
  hipStream_t stream;
  check(hipStreamCreate(&stream));
  float *inputs[] = {b.p + inputPadding, x.p + inputPadding,
                     y.p + inputPadding};
  __half *output = actual.p + outputPadding;
  for (int a = 0; a < 3; ++a) {
    check(hipMemcpy(inputs[a], raw[a].data(), raw[a].size() * 4,
                    hipMemcpyHostToDevice));
  }
  check(hipMemset(actual.p, 0x5a, (n + outputPadding + 1) * sizeof(__half)));
  auto original = [&] {
    // Use the original broadcast geometry for all four GPU operations.
    check(hip_elementwise_binary_bcast(stream, inputs[1], inputs[2], product.p,
                                       xs.data(), ys.data(), ps.data(), 1,
                                       HIP_DTYPE_FLOAT32));
    check(hip_cast(stream, product.p, ph.p, count(ps), HIP_DTYPE_FLOAT32,
                   HIP_DTYPE_FLOAT16));
    check(hip_cast(stream, inputs[0], bh.p, count(bs), HIP_DTYPE_FLOAT32,
                   HIP_DTYPE_FLOAT16));
    check(hip_elementwise_binary_bcast(
        stream, lhs ? ph.p : bh.p, lhs ? bh.p : ph.p, reference.p,
        lhs ? ps.data() : bs.data(), lhs ? bs.data() : ps.data(), os.data(), 0,
        HIP_DTYPE_FLOAT16));
  };
  auto fused = [&] {
    check(hip_rounded_scaled_add_f16(stream, inputs[0], inputs[1], inputs[2],
                                     output, bs.data(), xs.data(), ys.data(),
                                     os.data(), lhs));
  };
  original();
  fused();
  check(hipStreamSynchronize(stream));
  std::vector<uint16_t> a(n), r(n);
  check(hipMemcpy(a.data(), output, n * 2, hipMemcpyDeviceToHost));
  check(hipMemcpy(r.data(), reference.p, n * 2, hipMemcpyDeviceToHost));
  uint16_t guard;
  check(hipMemcpy(&guard, output + n, 2, hipMemcpyDeviceToHost));
  if (guard != 0x5a5a)
    throw std::runtime_error("write beyond output");
  if (outputPadding) {
    check(hipMemcpy(&guard, output - 1, 2, hipMemcpyDeviceToHost));
    if (guard != 0x5a5a)
      throw std::runtime_error("write before output");
  }
  size_t payloads = 0;
  auto nan = [](uint16_t v) { return (v & 0x7c00) == 0x7c00 && (v & 0x3ff); };
  for (size_t i = 0; i < n; ++i)
    if (a[i] != r[i]) {
      if (nan(a[i]) && nan(r[i]))
        ++payloads;
      else
        throw std::runtime_error("value mismatch at " + std::to_string(i));
    }
  if (hip_rounded_scaled_add_f16(stream, inputs[0], inputs[1], inputs[2],
                                 inputs[0], bs.data(), xs.data(), ys.data(),
                                 os.data(), lhs) != hipErrorInvalidValue)
    throw std::runtime_error("unsafe alias accepted");
  float oldMs = 0, newMs = 0;
  if (!special) {
    auto timing = [&](auto launch) {
      hipEvent_t begin, end;
      check(hipEventCreate(&begin));
      check(hipEventCreate(&end));
      for (int i = 0; i < 10; ++i)
        launch();
      check(hipEventRecord(begin, stream));
      for (int i = 0; i < 100; ++i)
        launch();
      check(hipEventRecord(end, stream));
      check(hipEventSynchronize(end));
      float ms;
      check(hipEventElapsedTime(&ms, begin, end));
      check(hipEventDestroy(begin));
      check(hipEventDestroy(end));
      return ms / 100;
    };
    oldMs = timing(original);
    newMs = timing(fused);
  }
  std::printf("PASS n=%zu product_on_lhs=%d special=%d input_padding=%d "
              "output_padding=%d nan_payload_changes=%zu "
              "old_ms=%.6f fused_ms=%.6f\n",
              n, lhs, special, inputPadding, outputPadding, payloads, oldMs,
              newMs);
  check(hipStreamDestroy(stream));
}
int main() {
  try {
    for (int lhs : {0, 1}) {
      for (int64_t n : {1, 4096, 65536, 53 * 4096, 512 * 4096}) {
        Shape s{1, 1, 1, n};
        test(s, s, s, s, lhs, true);
        if (n == 4096 || n == 512 * 4096)
          test(s, s, s, s, lhs, false);
      }
      test({2, 3, 17, 13}, {1, 3, 1, 13}, {2, 1, 17, 1}, {2, 3, 17, 13}, lhs,
           true);
      test({1, 3, 17, 1}, {2, 1, 17, 13}, {1, 1, 1, 1}, {2, 3, 17, 13}, lhs,
           true);
      // Test the model's geometry, including a partial prefill chunk.
      for (int64_t tokens : {1, 53, 512}) {
        Shape s{1, tokens, 64, 64};
        test(s, {1, 1, 64, 1}, s, s, lhs, true);
        test(s, {1, 1, 64, 1}, s, s, lhs, false);
      }
      // Test each indexing mode and reject unaligned vector loads/stores.
      for (int ip : {0, 1})
        for (int op : {0, 1}) {
          test({1, 1, 1, 1}, {2, 4, 1, 8}, {1, 1, 16, 1}, {2, 4, 16, 8}, lhs,
               true, ip, op);
          test({2, 3, 17, 8}, {1, 3, 1, 8}, {2, 1, 17, 1}, {2, 3, 17, 8}, lhs,
               true, ip, op);
          test({1, 53, 64, 64}, {1, 1, 64, 1}, {1, 53, 64, 64}, {1, 53, 64, 64},
               lhs, true, ip, op);
        }
    }
    Shape zero{1, 0, 3, 4}, one{1, 1, 1, 1}, bad{1, 2, 3, 4};
    check(hip_rounded_scaled_add_f16(nullptr, nullptr, nullptr, nullptr,
                                     nullptr, zero.data(), one.data(),
                                     zero.data(), zero.data(), 0));
    if (hip_rounded_scaled_add_f16(nullptr, nullptr, nullptr, nullptr, nullptr,
                                   bad.data(), one.data(), zero.data(),
                                   bad.data(), 0) != hipErrorInvalidValue)
      throw std::runtime_error("invalid broadcast accepted");
    std::puts("PASS zero extents and invalid broadcast rejection");
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
