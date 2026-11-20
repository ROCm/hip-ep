// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Ground-truth calibration workload for tools/perf-harness.
//
// Every kernel's cost is known by construction, so a profiler's numbers can be
// scored rather than trusted:
//   spin   busy-waits on wall_clock64(), a constant-rate counter whose rate the
//          device reports (hipDeviceAttributeWallClockRate), for an exact time.
//   read   reads exactly --mb MiB (grid-stride 16-byte loads, folded into one
//          word so the loads cannot be eliminated).
//   write  writes exactly --mb MiB.
//   copy   reads and writes exactly --mb MiB each.
// Each launch is also bracketed by its own hipEvent pair, a second independent
// timing source. With --shim <hipgpu.dll> the profiling build's trace shim is
// loaded into this process and each phase is a ROCTx range.
//
// Output: one JSON document on stdout describing what ran and what it should
// have cost.

#include <hip/hip_runtime.h>

#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(x)                                                               \
  do {                                                                         \
    hipError_t e_ = (x);                                                       \
    if (e_ != hipSuccess) {                                                    \
      std::fprintf(stderr, "%s failed: %s\n", #x, hipGetErrorString(e_));      \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

extern "C" __global__ void calib_spin(unsigned long long ticks,
                                      unsigned long long *out) {
  const unsigned long long t0 = wall_clock64();
  unsigned long long t = t0;
  while (t - t0 < ticks)
    t = wall_clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0)
    *out = t - t0;
}

extern "C" __global__ void calib_read(const uint4 *__restrict__ src, size_t n,
                                      unsigned int *__restrict__ sink) {
  unsigned int acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    const uint4 v = src[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  // Never true for the zero-filled buffer, but the compiler cannot know that.
  if (acc == 0x9e3779b9u)
    *sink = acc;
}

extern "C" __global__ void calib_write(uint4 *__restrict__ dst, size_t n,
                                       unsigned int value) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    dst[i] = make_uint4(value, value, value, value);
}

extern "C" __global__ void calib_copy(const uint4 *__restrict__ src,
                                      uint4 *__restrict__ dst, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    dst[i] = src[i];
}

using RangePushFn = int (*)(const char *);
using RangePopFn = int (*)();
static RangePushFn g_push = nullptr;
static RangePopFn g_pop = nullptr;

struct Range {
  explicit Range(const char *name) {
    if (g_push)
      g_push(name);
  }
  ~Range() {
    if (g_pop)
      g_pop();
  }
};

static double ms(hipEvent_t a, hipEvent_t b) {
  float v = 0.0f;
  CHECK(hipEventElapsedTime(&v, a, b));
  return v;
}

int main(int argc, char **argv) {
  size_t mb = 256;
  int iters = 20;
  std::vector<double> spinUs = {10.0, 100.0, 1000.0, 10000.0};
  const char *shim = nullptr;
  // RGP streams a capture out of the live process, so a capture of this
  // workload needs it to stay alive after the last kernel.
  int holdMs = 0;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--mb") && i + 1 < argc)
      mb = std::strtoull(argv[++i], nullptr, 10);
    else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc)
      iters = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--shim") && i + 1 < argc)
      shim = argv[++i];
    else if (!std::strcmp(argv[i], "--hold-ms") && i + 1 < argc)
      holdMs = std::atoi(argv[++i]);
  }

  CHECK(hipFree(nullptr)); // create the context before the shim attaches
  if (shim) {
    HMODULE m = LoadLibraryA(shim);
    if (!m) {
      std::fprintf(stderr, "cannot load %s (error %lu)\n", shim, GetLastError());
      return 1;
    }
    g_push = reinterpret_cast<RangePushFn>(GetProcAddress(m, "roctxRangePushA"));
    g_pop = reinterpret_cast<RangePopFn>(GetProcAddress(m, "roctxRangePop"));
  }

  int dev = 0;
  CHECK(hipGetDevice(&dev));
  hipDeviceProp_t props{};
  CHECK(hipGetDeviceProperties(&props, dev));
  int wallKhz = 0;
  CHECK(hipDeviceGetAttribute(&wallKhz, hipDeviceAttributeWallClockRate, dev));

  const size_t bytes = mb << 20;
  const size_t n = bytes / sizeof(uint4);
  uint4 *a = nullptr, *b = nullptr;
  unsigned int *sink = nullptr;
  unsigned long long *spinOut = nullptr;
  CHECK(hipMalloc(&a, bytes));
  CHECK(hipMalloc(&b, bytes));
  CHECK(hipMalloc(&sink, sizeof(unsigned int)));
  CHECK(hipMalloc(&spinOut, sizeof(unsigned long long)));
  CHECK(hipMemset(a, 0, bytes));
  CHECK(hipMemset(b, 0, bytes));
  CHECK(hipDeviceSynchronize());

  const int threads = 256;
  const int blocks = props.multiProcessorCount * 32;
  hipStream_t s;
  CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));

  std::string js = "{\n";
  char buf[512];
  std::snprintf(buf, sizeof(buf),
                "  \"device\": \"%s\", \"arch\": \"%s\", \"wall_clock_khz\": "
                "%d, \"cu_or_wgp\": %d, \"mb\": %zu, \"bytes\": %zu, "
                "\"iters\": %d, \"blocks\": %d, \"threads\": %d,\n",
                props.name, props.gcnArchName, wallKhz,
                props.multiProcessorCount, mb, bytes, iters, blocks, threads);
  js += buf;
  js += "  \"launches\": [\n";
  bool first = true;
  auto record = [&](const char *kernel, const char *phase, double expectUs,
                    double expectBytes, double eventMs, unsigned long long t) {
    std::snprintf(buf, sizeof(buf),
                  "%s    {\"kernel\": \"%s\", \"phase\": \"%s\", "
                  "\"expect_us\": %.3f, \"expect_bytes\": %.0f, \"event_ms\": "
                  "%.6f, \"wall_ticks\": %llu}",
                  first ? "" : ",\n", kernel, phase, expectUs, expectBytes,
                  eventMs, t);
    js += buf;
    first = false;
  };

  {
    Range warm("calib.warmup");
    for (int i = 0; i < 3; ++i) {
      calib_read<<<blocks, threads, 0, s>>>(a, n, sink);
      calib_write<<<blocks, threads, 0, s>>>(b, n, 1u);
      calib_copy<<<blocks, threads, 0, s>>>(a, b, n);
    }
    CHECK(hipStreamSynchronize(s));
  }

  char phase[64];
  for (double us : spinUs) {
    std::snprintf(phase, sizeof(phase), "calib.spin %.0fus", us);
    Range r(phase);
    const unsigned long long ticks =
        static_cast<unsigned long long>(us * wallKhz / 1000.0);
    for (int i = 0; i < iters; ++i) {
      CHECK(hipEventRecord(e0, s));
      calib_spin<<<1, 1, 0, s>>>(ticks, spinOut);
      CHECK(hipGetLastError());
      CHECK(hipEventRecord(e1, s));
      CHECK(hipStreamSynchronize(s));
      unsigned long long got = 0;
      CHECK(hipMemcpy(&got, spinOut, sizeof(got), hipMemcpyDeviceToHost));
      record("calib_spin", phase, ticks * 1000.0 / wallKhz, 0, ms(e0, e1), got);
    }
  }

  struct Mem {
    const char *kernel;
    double bytes;
  } mems[] = {{"calib_read", (double)bytes},
              {"calib_write", (double)bytes},
              {"calib_copy", 2.0 * bytes}};
  for (const Mem &m : mems) {
    std::snprintf(phase, sizeof(phase), "calib.%s", m.kernel + 6);
    Range r(phase);
    for (int i = 0; i < iters; ++i) {
      CHECK(hipEventRecord(e0, s));
      if (m.kernel[6] == 'r')
        calib_read<<<blocks, threads, 0, s>>>(a, n, sink);
      else if (m.kernel[6] == 'w')
        calib_write<<<blocks, threads, 0, s>>>(b, n, (unsigned)i);
      else
        calib_copy<<<blocks, threads, 0, s>>>(a, b, n);
      CHECK(hipGetLastError());
      CHECK(hipEventRecord(e1, s));
      CHECK(hipStreamSynchronize(s));
      record(m.kernel, phase, -1.0, m.bytes, ms(e0, e1), 0);
    }
  }
  js += "\n  ]\n}\n";
  std::fputs(js.c_str(), stdout);
  std::fflush(stdout);
  if (holdMs > 0)
    Sleep(static_cast<DWORD>(holdMs));

  CHECK(hipStreamDestroy(s));
  CHECK(hipFree(a));
  CHECK(hipFree(b));
  CHECK(hipFree(sink));
  CHECK(hipFree(spinOut));
  return 0;
}
