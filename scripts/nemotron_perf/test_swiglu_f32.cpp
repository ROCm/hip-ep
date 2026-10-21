/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "hip_custom_kernels.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <vector>
#ifdef NEMOTRON_SWIGLU_BCAST
#include "swiglu_f32.h"
#include <array>
static int hip_swiglu_f16_f32(void *stream, const void *g, const void *u,
                              void *out, int64_t n) {
  const int64_t shape[] = {1, 1, 1, n};
  return hip_swiglu_f16_f32_bcast(stream, g, u, out, shape, shape, shape);
}
#else

extern "C" int hip_swiglu_f16_f32(void *, const void *, const void *, void *,
                                  int64_t);
#endif

static void check(int result) {
  if (result != hipSuccess)
    throw std::runtime_error(
        hipGetErrorString(static_cast<hipError_t>(result)));
}

#ifdef NEMOTRON_SWIGLU_BCAST
static void testBroadcast(hipStream_t stream) {
  using Shape = std::array<int64_t, 4>;
  const Shape full{2, 3, 17, 13};
  auto count = [](Shape shape) {
    int64_t n = 1;
    for (int64_t d : shape)
      n *= d;
    return n;
  };
  auto offset = [&](int64_t i, Shape shape) {
    int64_t result = 0, stride = 1;
    for (int axis = 3; axis >= 0; --axis) {
      int64_t coordinate = i % full[axis];
      i /= full[axis];
      result += (shape[axis] == 1 ? 0 : coordinate) * stride;
      stride *= shape[axis];
    }
    return result;
  };
  for (auto shapes :
       {std::array<Shape, 2>{Shape{1, 3, 1, 13}, Shape{2, 1, 17, 1}},
        std::array<Shape, 2>{Shape{1, 1, 1, 1}, full},
        std::array<Shape, 2>{full, Shape{1, 3, 1, 13}}}) {
    auto gs = shapes[0], us = shapes[1];
    int64_t n = count(full), ng = count(gs), nu = count(us);
    std::vector<uint16_t> g(ng), u(nu), eg(n), eu(n), expected(n), actual(n);
    for (int64_t i = 0; i < ng; ++i)
      g[i] = uint16_t(i * 47 + 0x3600);
    for (int64_t i = 0; i < nu; ++i)
      u[i] = uint16_t(i * 13 + 0x9100);
    for (int64_t i = 0; i < n; ++i) {
      eg[i] = g[offset(i, gs)];
      eu[i] = u[offset(i, us)];
    }
    uint16_t *dg, *du, *out, *egpu, *ugpu, *ref;
    float *gf, *uf, *sf, *mf;
    for (auto ptr : {&dg, &du, &out, &egpu, &ugpu, &ref})
      check(hipMalloc(ptr, n * 2));
    for (auto ptr : {&gf, &uf, &sf, &mf})
      check(hipMalloc(ptr, n * 4));
    check(
        hipMemcpyAsync(egpu, eg.data(), n * 2, hipMemcpyHostToDevice, stream));
    check(
        hipMemcpyAsync(ugpu, eu.data(), n * 2, hipMemcpyHostToDevice, stream));
    check(hip_cast(stream, egpu, gf, n, HIP_DTYPE_FLOAT16, HIP_DTYPE_FLOAT32));
    check(hip_cast(stream, ugpu, uf, n, HIP_DTYPE_FLOAT16, HIP_DTYPE_FLOAT32));
    check(hip_swish(stream, gf, sf, n, HIP_DTYPE_FLOAT32, 1.0));
    check(hip_elementwise_mul(stream, uf, sf, mf, n, HIP_DTYPE_FLOAT32));
    check(hip_cast(stream, mf, ref, n, HIP_DTYPE_FLOAT32, HIP_DTYPE_FLOAT16));
    check(hipMemcpyAsync(expected.data(), ref, n * 2, hipMemcpyDeviceToHost,
                         stream));
    for (auto destination : {out, dg, du}) {
      check(
          hipMemcpyAsync(dg, g.data(), ng * 2, hipMemcpyHostToDevice, stream));
      check(
          hipMemcpyAsync(du, u.data(), nu * 2, hipMemcpyHostToDevice, stream));
      int status = hip_swiglu_f16_f32_bcast(stream, dg, du, destination,
                                            gs.data(), us.data(), full.data());
      if ((destination == dg && gs != full) ||
          (destination == du && us != full)) {
        if (status != hipErrorInvalidValue)
          throw std::runtime_error("unsafe broadcast alias accepted");
        continue;
      }
      check(status);
      check(hipMemcpyAsync(actual.data(), destination, n * 2,
                           hipMemcpyDeviceToHost, stream));
      check(hipStreamSynchronize(stream));
      for (int64_t i = 0; i < n; ++i) {
        auto nan = [](uint16_t x) {
          return (x & 0x7c00) == 0x7c00 && (x & 0x3ff);
        };
        if (actual[i] != expected[i] && !(nan(actual[i]) && nan(expected[i])))
          throw std::runtime_error("broadcast value mismatch at " +
                                   std::to_string(i));
      }
    }
    for (void *p : {static_cast<void *>(dg), static_cast<void *>(du),
                    static_cast<void *>(out), static_cast<void *>(egpu),
                    static_cast<void *>(ugpu), static_cast<void *>(ref),
                    static_cast<void *>(gf), static_cast<void *>(uf),
                    static_cast<void *>(sf), static_cast<void *>(mf)})
      check(hipFree(p));
  }
  Shape zero{1, 0, 17, 13}, ones{1, 1, 1, 1}, wrong{1, 2, 17, 13};
  check(hip_swiglu_f16_f32_bcast(stream, nullptr, nullptr, nullptr, zero.data(),
                                 ones.data(), zero.data()));
  if (hip_swiglu_f16_f32_bcast(stream, nullptr, nullptr, nullptr, zero.data(),
                               wrong.data(),
                               wrong.data()) != hipErrorInvalidValue)
    throw std::runtime_error("invalid broadcast accepted");
  std::puts("PASS broadcast: multi-axis, scalar, tails, valid aliases, "
            "unsafe-alias rejection, zero extent");
}
#endif

int main() {
  hipStream_t stream;
  check(hipStreamCreate(&stream));
#ifdef NEMOTRON_SWIGLU_BCAST
  testBroadcast(stream);
#endif
  for (int n : {1, 4096, 65536, 53 * 4096, 512 * 4096}) {
    // Cover every FP16 bit pattern, including signed zero and non-finite data.
    std::vector<uint16_t> gate(n), up(n), expected(n), actual(n);
    for (int i = 0; i < n; ++i) {
      gate[i] = static_cast<uint16_t>(i);
      up[i] = static_cast<uint16_t>(i * 13 + 113);
    }
    uint16_t *g, *u, *out, *ref;
    float *gf, *uf, *sf, *mf;
    check(hipMalloc(&g, n * 2));
    check(hipMalloc(&u, n * 2));
    check(hipMalloc(&out, n * 2));
    check(hipMalloc(&ref, n * 2));
    check(hipMalloc(&gf, n * 4));
    check(hipMalloc(&uf, n * 4));
    check(hipMalloc(&sf, n * 4));
    check(hipMalloc(&mf, n * 4));
    auto restore = [&] {
      check(
          hipMemcpyAsync(g, gate.data(), n * 2, hipMemcpyHostToDevice, stream));
      check(hipMemcpyAsync(u, up.data(), n * 2, hipMemcpyHostToDevice, stream));
    };
    auto reference = [&] {
      check(hip_cast(stream, g, gf, n, HIP_DTYPE_FLOAT16, HIP_DTYPE_FLOAT32));
      check(hip_cast(stream, u, uf, n, HIP_DTYPE_FLOAT16, HIP_DTYPE_FLOAT32));
      check(hip_swish(stream, gf, sf, n, HIP_DTYPE_FLOAT32, 1.0));
      check(hip_elementwise_mul(stream, uf, sf, mf, n, HIP_DTYPE_FLOAT32));
      check(hip_cast(stream, mf, ref, n, HIP_DTYPE_FLOAT32, HIP_DTYPE_FLOAT16));
    };
    restore();
    reference();
    check(hipMemcpyAsync(expected.data(), ref, n * 2, hipMemcpyDeviceToHost,
                         stream));
    check(hipStreamSynchronize(stream));
    size_t nanPayloadDifferences = 0;
    auto nan = [](uint16_t bits) {
      return (bits & 0x7c00) == 0x7c00 && (bits & 0x03ff) != 0;
    };
    for (uint16_t *destination : {out, g, u}) {
      restore();
      check(hip_swiglu_f16_f32(stream, g, u, destination, n));
      check(hipMemcpyAsync(actual.data(), destination, n * 2,
                           hipMemcpyDeviceToHost, stream));
      check(hipStreamSynchronize(stream));
      for (int i = 0; i < n; ++i) {
        if (actual[i] == expected[i])
          continue;
        if (nan(actual[i]) && nan(expected[i])) {
          ++nanPayloadDifferences;
          continue;
        }
        std::fprintf(stderr, "Mismatch n=%d i=%d expected=%04x actual=%04x\n",
                     n, i, expected[i], actual[i]);
        float swish, value, product;
        check(hipMemcpy(&swish, sf + i, 4, hipMemcpyDeviceToHost));
        check(hipMemcpy(&value, uf + i, 4, hipMemcpyDeviceToHost));
        check(hipMemcpy(&product, mf + i, 4, hipMemcpyDeviceToHost));
        _Float16 direct = static_cast<_Float16>(double(swish) * double(value));
        _Float16 staged = static_cast<_Float16>(product);
        uint16_t directBits, stagedBits;
        std::memcpy(&directBits, &direct, 2);
        std::memcpy(&stagedBits, &staged, 2);
        std::fprintf(stderr,
                     "Reference swish=%.17g up=%.17g product_f32=%.17g "
                     "direct_half=%04x staged_half=%04x\n",
                     swish, value, product, directBits, stagedBits);
        return 1;
      }
    }
    restore();
    hipEvent_t start, stop;
    check(hipEventCreate(&start));
    check(hipEventCreate(&stop));
    auto timed = [&](bool fused) {
      auto run = [&] {
        if (fused)
          check(hip_swiglu_f16_f32(stream, g, u, out, n));
        else
          reference();
      };
      for (int i = 0; i < 10; ++i)
        run();
      check(hipEventRecord(start, stream));
      for (int i = 0; i < 100; ++i)
        run();
      check(hipEventRecord(stop, stream));
      check(hipEventSynchronize(stop));
      float ms;
      check(hipEventElapsedTime(&ms, start, stop));
      return ms / 100;
    };
    float refMs = timed(false), fusedMs = timed(true);
    std::printf("PASS n=%d exact_finite_and_inf=1 aliases=3 "
                "nan_payload_differences=%zu reference_ms=%.6f fused_ms=%.6f\n",
                n, nanPayloadDifferences, refMs, fusedMs);
    check(hipEventDestroy(start));
    check(hipEventDestroy(stop));
    for (void *p : {static_cast<void *>(g), static_cast<void *>(u),
                    static_cast<void *>(out), static_cast<void *>(ref),
                    static_cast<void *>(gf), static_cast<void *>(uf),
                    static_cast<void *>(sf), static_cast<void *>(mf)})
      check(hipFree(p));
  }
  check(hipStreamDestroy(stream));
}
