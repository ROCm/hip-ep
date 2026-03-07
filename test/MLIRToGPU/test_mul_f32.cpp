/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <hip/hip_runtime_api.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    if (hipError_t e = (call))                                                 \
      return fprintf(stderr, "HIP error %d at %s:%d\n", e, __FILE__,           \
                     __LINE__),                                                \
             1;                                                                \
  } while (0)

// Unpacked memref ABI: 3 x (ptr, ptr, i64, i64, i64) = 15 args.
typedef void (*MulFn)(void *, void *, int64_t, int64_t, int64_t, void *, void *,
                      int64_t, int64_t, int64_t, void *, void *, int64_t,
                      int64_t, int64_t);

static void *gpuAlloc(const void *src, size_t bytes) {
  void *d = nullptr;
  hipMalloc(&d, bytes);
  if (src)
    hipMemcpy(d, src, bytes, hipMemcpyHostToDevice);
  else
    hipMemset(d, 0, bytes);
  return d;
}

static constexpr int N = 8;

int main(int argc, char **argv) {
  if (argc < 3)
    return fprintf(stderr, "Usage: %s <dll> <sym> [--const]\n", argv[0]), 1;

  const char *dllPath = argv[1], *sym = argv[2];
  bool isConst = argc > 3 && strcmp(argv[3], "--const") == 0;

  // Load DLL
#ifdef _WIN32
  HMODULE lib = LoadLibraryA(dllPath);
  auto fn = lib ? (MulFn)GetProcAddress(lib, sym) : nullptr;
#else
  void *lib = dlopen(dllPath, RTLD_NOW);
  auto fn = lib ? (MulFn)dlsym(lib, sym) : nullptr;
#endif
  if (!fn)
    return fprintf(stderr, "Cannot load %s::%s\n", dllPath, sym), 1;

  // Host data: A = {1..8}, B = {2,2,...} (runtime) or {1..8} (const)
  float hA[N], hB[N], expected[N];
  for (int i = 0; i < N; i++) {
    hA[i] = (float)(i + 1);
    hB[i] = isConst ? (float)(i + 1) : 2.0f;
    expected[i] = hA[i] * hB[i];
  }

  // GPU buffers
  void *dA = gpuAlloc(hA, sizeof(hA));
  void *dC = gpuAlloc(nullptr, sizeof(hA));

  if (isConst) {
    int64_t blob = N * (int64_t)sizeof(float);
    void *dBlob = gpuAlloc(hB, blob);
    fn(dA, dA, 0, N, 1, dC, dC, 0, N, 1, dBlob, dBlob, 0, blob, 1);
    hipFree(dBlob);
  } else {
    void *dB = gpuAlloc(hB, sizeof(hB));
    fn(dA, dA, 0, N, 1, dB, dB, 0, N, 1, dC, dC, 0, N, 1);
    hipFree(dB);
  }

  float hC[N];
  HIP_CHECK(hipMemcpy(hC, dC, sizeof(hC), hipMemcpyDeviceToHost));
  hipFree(dA);
  hipFree(dC);
#ifdef _WIN32
  FreeLibrary(lib);
#endif

  int fail = 0;
  for (int i = 0; i < N; i++)
    if (std::fabs(hC[i] - expected[i]) > 1e-5f)
      fprintf(stderr, "MISMATCH [%d]: %f != %f\n", i, hC[i], expected[i]),
          fail++;

  printf("%s: %d elements (%s)\n", fail ? "FAIL" : "PASS", N,
         isConst ? "const" : "runtime");
  return fail ? 1 : 0;
}
