/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * hip_cumsum against a CPU reference, across both of its launch strategies.
 *
 * The interesting shapes are the ones with FEW slices and a LONG axis. That is
 * the decoder attention mask -- [1, context] scanned on axis 1, which collapses
 * to outer = inner = 1 -- and it is the case the slice-parallel kernel handles
 * with a single thread. Nothing in test/numeric reaches it: the longest axis
 * there is 128 and every case has outer*inner > 1.
 *
 * The cases below therefore straddle the selection boundary in both directions,
 * cover the ragged final tile, and keep a vision-shaped case (many slices,
 * short axis) as a regression guard on the path that was already there.
 *
 * Build and run:
 *   hipcc --offload-arch=gfx1151 -O3 -std=c++17 \
 *       test_cumsum_scan.cpp ../../../hip/cumsum_kernel.hip \
 *       -I../../../include -o test_cumsum_scan.exe
 *   ./test_cumsum_scan.exe
 */

#include "hip_custom_kernels.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    hipError_t e_ = (cmd);                                                     \
    if (e_ != hipSuccess) {                                                    \
      std::printf("HIP error %s at line %d\n", hipGetErrorString(e_),          \
                  __LINE__);                                                   \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)

static int g_failures = 0;

// Deterministic, so a failure reproduces exactly.
static uint32_t rng_state = 0x2545F491u;
static uint32_t next_rand() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  return rng_state;
}

struct Shape {
  const char *name;
  int64_t outer, axis, inner;
};

// Mirrors the kernel's own selection so the report can say which path ran. Kept
// deliberately as a copy: if the launcher's rule changes without this one, the
// coverage claim these cases make silently stops being true.
static bool expects_cooperative(const Shape &s) { return s.axis >= 512; }

template <typename T, typename ACC>
static void cpu_reference(const std::vector<T> &x, std::vector<T> &y,
                          int64_t outer, int64_t axis, int64_t inner, bool excl,
                          bool rev) {
  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t i = 0; i < inner; ++i) {
      const int64_t base = o * axis * inner + i;
      ACC acc = ACC(0);
      for (int64_t n = 0; n < axis; ++n) {
        const int64_t k = rev ? (axis - 1 - n) : n;
        const int64_t idx = base + k * inner;
        if (excl) {
          y[idx] = static_cast<T>(acc);
          acc += static_cast<ACC>(x[idx]);
        } else {
          acc += static_cast<ACC>(x[idx]);
          y[idx] = static_cast<T>(acc);
        }
      }
    }
  }
}

template <typename T> static double to_double(T v) {
  return static_cast<double>(v);
}
template <> double to_double<__half>(__half v) {
  return static_cast<double>(__half2float(v));
}

template <typename T, typename ACC>
static void run_case(const Shape &s, int hip_dtype, const char *dtype_name,
                     bool excl, bool rev, double tol) {
  const int64_t n = s.outer * s.axis * s.inner;
  std::vector<T> h_x(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    // 0/1 payload: it is what a mask carries, it keeps the fp16 sums exact for
    // these axis lengths, and it makes an off-by-one in the scan obvious in the
    // output rather than lost in rounding.
    h_x[static_cast<size_t>(i)] = static_cast<T>((next_rand() & 3u) ? 1 : 0);
  }
  std::vector<T> h_ref(static_cast<size_t>(n));
  cpu_reference<T, ACC>(h_x, h_ref, s.outer, s.axis, s.inner, excl, rev);

  T *d_x = nullptr, *d_y = nullptr;
  HIP_CHECK(hipMalloc(&d_x, sizeof(T) * static_cast<size_t>(n)));
  HIP_CHECK(hipMalloc(&d_y, sizeof(T) * static_cast<size_t>(n)));
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), sizeof(T) * static_cast<size_t>(n),
                      hipMemcpyHostToDevice));
  // Poison the output so a kernel that silently writes nothing fails loudly
  // instead of inheriting a zeroed buffer that happens to match.
  HIP_CHECK(hipMemset(d_y, 0x5A, sizeof(T) * static_cast<size_t>(n)));

  int rc = hip_cumsum(nullptr, d_x, d_y, s.outer, s.axis, s.inner, hip_dtype,
                      excl ? 1 : 0, rev ? 1 : 0);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<T> h_y(static_cast<size_t>(n));
  HIP_CHECK(hipMemcpy(h_y.data(), d_y, sizeof(T) * static_cast<size_t>(n),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_y));

  double worst = 0.0;
  int64_t worst_at = -1;
  for (int64_t i = 0; i < n; ++i) {
    const double a = to_double<T>(h_y[static_cast<size_t>(i)]);
    const double b = to_double<T>(h_ref[static_cast<size_t>(i)]);
    const double d =
        std::fabs(a - b) / (std::fabs(b) > 1.0 ? std::fabs(b) : 1.0);
    if (d > worst) {
      worst = d;
      worst_at = i;
    }
  }

  const bool ok = (rc == 0) && (worst <= tol);
  if (!ok)
    ++g_failures;
  std::printf("  %-24s %-6s %-9s %-5s  %-14s  worst %.3e  %s\n", s.name,
              dtype_name, excl ? "excl" : "incl", rev ? "rev" : "fwd",
              expects_cooperative(s) ? "cooperative" : "slice-parallel", worst,
              ok ? "PASS" : "FAIL");
  if (!ok && worst_at >= 0) {
    std::printf("      rc=%d first-worst index %lld: got %.6f want %.6f\n", rc,
                (long long)worst_at,
                to_double<T>(h_y[static_cast<size_t>(worst_at)]),
                to_double<T>(h_ref[static_cast<size_t>(worst_at)]));
  }
}

// Wall time for one shape, i64, the decode mask's own dtype. Reported so the
// selection boundary can be read as a cliff: 511 and 512 do near-identical work
// and differ only in which kernel runs.
static void bench_case(const Shape &s, int iters) {
  const int64_t n = s.outer * s.axis * s.inner;
  std::vector<int64_t> h_x(static_cast<size_t>(n), 1);
  int64_t *d_x = nullptr, *d_y = nullptr;
  HIP_CHECK(hipMalloc(&d_x, sizeof(int64_t) * static_cast<size_t>(n)));
  HIP_CHECK(hipMalloc(&d_y, sizeof(int64_t) * static_cast<size_t>(n)));
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), sizeof(int64_t) * static_cast<size_t>(n),
                      hipMemcpyHostToDevice));

  for (int i = 0; i < 20; ++i)
    hip_cumsum(nullptr, d_x, d_y, s.outer, s.axis, s.inner, HIP_DTYPE_INT64, 0,
               0);
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t t0, t1;
  HIP_CHECK(hipEventCreate(&t0));
  HIP_CHECK(hipEventCreate(&t1));
  HIP_CHECK(hipEventRecord(t0));
  for (int i = 0; i < iters; ++i)
    hip_cumsum(nullptr, d_x, d_y, s.outer, s.axis, s.inner, HIP_DTYPE_INT64, 0,
               0);
  HIP_CHECK(hipEventRecord(t1));
  HIP_CHECK(hipEventSynchronize(t1));
  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
  HIP_CHECK(hipEventDestroy(t0));
  HIP_CHECK(hipEventDestroy(t1));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_y));

  std::printf("  %-24s slices %-8lld axis %-8lld %-14s %9.2f us\n", s.name,
              (long long)(s.outer * s.inner), (long long)s.axis,
              expects_cooperative(s) ? "cooperative" : "slice-parallel",
              1000.0 * ms / iters);
}

int main(int argc, char **argv) {
  bool bench = false;
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--bench")
      bench = true;

  const Shape shapes[] = {
      // The decode attention mask, at both ends of the context range measured.
      {"mask 1x2240x1", 1, 2240, 1},
      {"mask 1x16241x1", 1, 16241, 1},
      // Selection boundary, one either side, single slice.
      {"boundary axis 511", 1, 511, 1},
      {"boundary axis 512", 1, 512, 1},
      // Selection boundary on slice count, with a long axis.
      {"boundary slices 1023", 1023, 600, 1},
      {"boundary slices 1024", 1024, 600, 1},
      // Tile edges: exactly one tile, one past it, and a ragged tail.
      {"tile exact 256", 1, 256, 1},
      {"tile 257", 1, 257, 1},
      {"tile ragged 1000", 1, 1000, 1},
      // Strided: inner > 1 exercises the slice stride on the cooperative path.
      {"strided 2x1000x3", 2, 1000, 3},
      // Vision-shaped regression guard for the pre-existing path.
      {"vision 4096x16x1", 4096, 16, 1},
      // Many slices AND a long axis: the case that decides whether the
      // selection needs a slice-count term at all.
      {"wide+long 8192x1024", 8192, 1024, 1},
  };

  if (bench) {
    std::printf("hip_cumsum: wall time per call, i64, inclusive forward\n");
    for (const Shape &s : shapes)
      bench_case(s, 200);
    return 0;
  }

  std::printf(
      "hip_cumsum: CPU-reference check across both launch strategies\n");
  for (const Shape &s : shapes) {
    for (int e = 0; e < 2; ++e) {
      for (int r = 0; r < 2; ++r) {
        const bool excl = (e != 0), rev = (r != 0);
        // int64 is the decode mask's own dtype; the scan is exact, so any
        // difference at all is a bug.
        run_case<int64_t, int64_t>(s, HIP_DTYPE_INT64, "i64", excl, rev, 0.0);
        run_case<int32_t, int32_t>(s, HIP_DTYPE_INT32, "i32", excl, rev, 0.0);
        run_case<float, double>(s, HIP_DTYPE_FLOAT32, "f32", excl, rev, 1e-6);
        // fp16 accumulates in float in the kernel; with a 0/1 payload every
        // partial sum here is integral and under 2048, so it is exact too.
        if (s.axis <= 2048)
          run_case<__half, float>(s, HIP_DTYPE_FLOAT16, "f16", excl, rev, 0.0);
      }
    }
  }

  std::printf("\n%s (%d failing case(s))\n",
              g_failures == 0 ? "ALL PASS" : "FAILURES", g_failures);
  return g_failures == 0 ? 0 : 1;
}
