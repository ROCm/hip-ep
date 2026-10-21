// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "hip_custom_kernels.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <vector>

#define CHECK(call)                                                            \
  do {                                                                         \
    if ((call) != 0) {                                                         \
      std::fprintf(stderr, "Failed: %s at %d\n", #call, __LINE__);             \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

decltype(&hip_linear_attention_decode) referenceDecode;
decltype(&hip_linear_attention_prefill_chunked) referencePrefill;
decltype(&hip_linear_attention_prefill_scratch_bytes) referenceScratchBytes;
decltype(&hip_linear_attention_decode_f16_state) referenceF16;

template <class T> struct Buffer {
  T *ptr = nullptr;
  void *host = nullptr;
  size_t size;
  explicit Buffer(size_t n, bool mapped = false) : size(n) {
    if (mapped) {
      CHECK(hipHostMalloc(&host, n * sizeof(T),
                          hipHostMallocMapped | hipHostMallocCoherent));
      CHECK(hipHostGetDevicePointer(reinterpret_cast<void **>(&ptr), host, 0));
    } else {
      CHECK(hipMalloc(&ptr, n * sizeof(T)));
    }
  }
  ~Buffer() {
    if (host) {
      CHECK(hipHostFree(host));
    } else {
      CHECK(hipFree(ptr));
    }
  }
  void set(const std::vector<T> &v) {
    CHECK(hipMemcpy(ptr, v.data(), size * sizeof(T), hipMemcpyHostToDevice));
  }
  std::vector<T> get() {
    std::vector<T> v(size);
    CHECK(hipMemcpy(v.data(), ptr, size * sizeof(T), hipMemcpyDeviceToHost));
    return v;
  }
};

std::vector<float> data(size_t size, int seed, float scale = 0.15f) {
  std::vector<float> result(size);
  for (size_t i = 0; i < size; ++i)
    result[i] = scale * std::sin(float((i * 17 + seed) % 997));
  return result;
}

void test(int B, int Hq, int Hkv, int Nk, int dk, int dv, int rule, int per_key,
          int per_head, bool alias, bool mapped = false) {
  const size_t elements = size_t(B) * Hkv * dk * dv;
  Buffer<float> q(size_t(B) * Hq * dk), k(size_t(B) * Nk * dk);
  Buffer<float> v(size_t(B) * Hkv * dv),
      d(size_t(B) * Hkv * (per_key ? dk : 1));
  Buffer<float> beta(size_t(B) * (per_head ? Hkv : 1));
  Buffer<float> state32(elements), out1(size_t(B) * std::max(Hq, Hkv) * dv);
  Buffer<float> out2(out1.size);
  Buffer<__half> past(elements, mapped), state16(elements, mapped),
      ref16(elements, mapped);
  q.set(data(q.size, 1));
  k.set(data(k.size, 2));
  v.set(data(v.size, 3));
  auto gates = data(d.size, 4);
  for (float &x : gates)
    x = -std::abs(x);
  d.set(gates);
  beta.set(data(beta.size, 5));
  auto raw = data(elements, 6, 0.5f);
  std::vector<__half> initial(elements);
  for (size_t i = 0; i < elements; ++i)
    initial[i] = __float2half(raw[i]);
  past.set(initial);
  hipStream_t stream;
  CHECK(hipStreamCreate(&stream));
  auto reference = [&] {
    CHECK(hip_cast(stream, past.ptr, state32.ptr, elements, 1, 0));
    CHECK(referenceDecode(
        stream, q.ptr, k.ptr, v.ptr, rule == 1 || rule == 3 ? d.ptr : nullptr,
        rule >= 2 ? beta.ptr : nullptr, state32.ptr, out1.ptr, B, 1, Hq, Hkv,
        Nk, dk, dv, 0.125f, rule, per_key, per_head, 0));
    CHECK(hip_cast(stream, state32.ptr, ref16.ptr, elements, 0, 1));
  };
  auto fused = [&] {
    CHECK(hip_linear_attention_decode_f16_state(
        stream, q.ptr, k.ptr, v.ptr, rule == 1 || rule == 3 ? d.ptr : nullptr,
        rule >= 2 ? beta.ptr : nullptr, past.ptr,
        alias ? past.ptr : state16.ptr, out2.ptr, B, Hq, Hkv, Nk, dk, dv,
        0.125f, rule, per_key, per_head));
  };
  for (int step = 0; step < 8; ++step) {
    reference();
    fused();
    CHECK(hipStreamSynchronize(stream));
    auto a = out1.get(), b = out2.get();
    auto sa = ref16.get(), sb = alias ? past.get() : state16.get();
    if (std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) ||
        std::memcmp(sa.data(), sb.data(), sa.size() * sizeof(__half))) {
      size_t outChanges = 0, stateChanges = 0;
      double maxOut = 0, maxState = 0;
      for (size_t i = 0; i < a.size(); ++i) {
        outChanges += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
        maxOut = std::max(maxOut, std::abs(double(a[i]) - b[i]));
      }
      for (size_t i = 0; i < sa.size(); ++i) {
        stateChanges += std::memcmp(&sa[i], &sb[i], sizeof(__half)) != 0;
        maxState = std::max(maxState, std::abs(double(__half2float(sa[i])) -
                                               __half2float(sb[i])));
      }
      std::fprintf(stderr,
                   "Mismatch: Hq=%d rule=%d layout=%d/%d alias=%d step=%d "
                   "output_changes=%zu max_output=%.9g state_changes=%zu "
                   "max_state=%.9g\n",
                   Hq, rule, per_key, per_head, alias, step, outChanges, maxOut,
                   stateChanges, maxState);
      std::exit(1);
    }
    past.set(sa);
  }
  auto timing = [&](auto launch) {
    hipEvent_t start, end;
    CHECK(hipEventCreate(&start));
    CHECK(hipEventCreate(&end));
    for (int i = 0; i < 20; ++i)
      launch();
    CHECK(hipEventRecord(start, stream));
    for (int i = 0; i < 200; ++i)
      launch();
    CHECK(hipEventRecord(end, stream));
    CHECK(hipEventSynchronize(end));
    float ms;
    CHECK(hipEventElapsedTime(&ms, start, end));
    CHECK(hipEventDestroy(start));
    CHECK(hipEventDestroy(end));
    return ms * 1000 / 200;
  };
  // Timing uses disjoint states, so every invocation gets the same input.
  float base_us = 0, fused_us = 0;
  if (!alias && B == 1 && Hkv == 64 && rule == 1) {
    base_us = timing(reference);
    fused_us = timing(fused);
  }
  std::printf("PASS B=%d Hq=%d Hkv=%d Nk=%d dk=%d dv=%d rule=%d layout=%d/%d "
              "alias=%d mapped=%d base_us=%.3f fused_us=%.3f\n",
              B, Hq, Hkv, Nk, dk, dv, rule, per_key, per_head, alias, mapped,
              base_us, fused_us);
  CHECK(hipStreamDestroy(stream));
}

void testHeadBroadcast(int B, int tokens) {
  constexpr int heads = 64, compact = 8, dk = 128, dv = 64;
  const size_t rows = size_t(B) * tokens;
  auto smallQ = data(rows * compact * dk, 17);
  auto smallK = data(rows * compact * dk, 23);
  std::vector<float> largeQ(rows * heads * dk), largeK(largeQ.size());
  for (size_t row = 0; row < rows; ++row)
    for (int h = 0; h < heads; ++h)
      for (int d = 0; d < dk; ++d) {
        size_t source = (row * compact + h / 8) * dk + d;
        size_t dest = (row * heads + h) * dk + d;
        largeQ[dest] = smallQ[source];
        largeK[dest] = smallK[source];
      }
  Buffer<float> q(largeQ.size()), k(largeK.size());
  Buffer<float> qc(smallQ.size()), kc(smallK.size());
  Buffer<float> v(rows * heads * dv), decay(rows * heads);
  Buffer<float> state(size_t(B) * heads * dk * dv), stateCompact(state.size);
  Buffer<float> out(rows * heads * dv), outCompact(out.size);
  q.set(largeQ);
  k.set(largeK);
  qc.set(smallQ);
  kc.set(smallK);
  v.set(data(v.size, 31));
  auto gate = data(decay.size, 41);
  for (float &x : gate)
    x = -std::abs(x);
  decay.set(gate);
  auto initial = data(state.size, 51, 0.5f);
  state.set(initial);
  stateCompact.set(initial);
  size_t bytes =
      tokens > 1 ? referenceScratchBytes(B, tokens, heads, dk, dv, 1) : 1;
  Buffer<unsigned char> scratch(std::max(bytes, size_t(1)));
  hipStream_t stream;
  CHECK(hipStreamCreate(&stream));
  auto run = [&](int h, float *qp, float *kp, float *sp, float *op) {
    if (tokens == 1)
      CHECK(referenceDecode(stream, qp, kp, v.ptr, decay.ptr, nullptr, sp, op,
                            B, tokens, h, heads, h, dk, dv, 1.0f, 1, 0, 0, 0));
    else
      CHECK(referencePrefill(stream, qp, kp, v.ptr, decay.ptr, nullptr, sp, op,
                             B, tokens, h, heads, h, dk, dv, 1.0f, 1, 0, 0, 0,
                             scratch.ptr, bytes));
  };
  run(heads, q.ptr, k.ptr, state.ptr, out.ptr);
  run(compact, qc.ptr, kc.ptr, stateCompact.ptr, outCompact.ptr);
  CHECK(hipStreamSynchronize(stream));
  auto a = out.get(), b = outCompact.get();
  auto sa = state.get(), sb = stateCompact.get();
  if (std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) ||
      std::memcmp(sa.data(), sb.data(), sa.size() * sizeof(float))) {
    std::fprintf(stderr, "Head broadcast mismatch: B=%d tokens=%d\n", B,
                 tokens);
    std::exit(1);
  }
  std::printf("PASS repeated-vs-native heads B=%d tokens=%d: outputs and state "
              "bit-exact\n",
              B, tokens);
  CHECK(hipStreamDestroy(stream));
}

void testColdState(bool mapped) {
  constexpr int rotations = 64, elements = 64 * 128 * 64;
  Buffer<float> q(8 * 128), k(8 * 128), v(64 * 64), decay(64), out(64 * 64);
  Buffer<__half> past(size_t(elements) * rotations, mapped);
  Buffer<__half> state(past.size, mapped);
  q.set(data(q.size, 1));
  k.set(data(k.size, 2));
  v.set(data(v.size, 3));
  auto gate = data(decay.size, 4);
  for (float &x : gate)
    x = -std::abs(x);
  decay.set(gate);
  auto raw = data(elements, 6, 0.5f);
  std::vector<__half> initial(elements), expected(elements), actual(elements);
  for (int i = 0; i < elements; ++i)
    initial[i] = __float2half(raw[i]);
  hipStream_t stream;
  CHECK(hipStreamCreate(&stream));
  for (int r = 0; r < rotations; ++r)
    CHECK(hipMemcpyAsync(past.ptr + size_t(r) * elements, initial.data(),
                         elements * 2, hipMemcpyHostToDevice, stream));
  CHECK(hipStreamSynchronize(stream));
  auto run = [&](bool fixed, int r) {
    auto fn = fixed ? &hip_linear_attention_decode_f16_state : referenceF16;
    CHECK(fn(stream, q.ptr, k.ptr, v.ptr, decay.ptr, nullptr,
             past.ptr + size_t(r) * elements, state.ptr + size_t(r) * elements,
             out.ptr, 1, 8, 64, 8, 128, 64, 0.125f, 1, 0, 0));
  };
  for (int r : {0, 17, 63}) {
    run(false, r);
    CHECK(hipStreamSynchronize(stream));
    auto refOut = out.get();
    CHECK(hipMemcpy(expected.data(), state.ptr + size_t(r) * elements,
                    elements * 2, hipMemcpyDeviceToHost));
    run(true, r);
    CHECK(hipStreamSynchronize(stream));
    auto newOut = out.get();
    CHECK(hipMemcpy(actual.data(), state.ptr + size_t(r) * elements,
                    elements * 2, hipMemcpyDeviceToHost));
    if (std::memcmp(expected.data(), actual.data(), elements * 2) ||
        std::memcmp(refOut.data(), newOut.data(), refOut.size() * 4)) {
      std::fprintf(stderr, "Cold-state mismatch mapped=%d rotation=%d\n",
                   mapped, r);
      std::exit(1);
    }
  }
  hipEvent_t start, stop;
  CHECK(hipEventCreate(&start));
  CHECK(hipEventCreate(&stop));
  for (int round = 0; round < 4; ++round) {
    for (int order = 0; order < 2; ++order) {
      bool fixed = (round + order) % 2;
      for (int r = 0; r < rotations; ++r)
        run(fixed, r);
      CHECK(hipEventRecord(start, stream));
      for (int i = 0; i < rotations * 4; ++i)
        run(fixed, i % rotations);
      CHECK(hipEventRecord(stop, stream));
      CHECK(hipEventSynchronize(stop));
      float ms;
      CHECK(hipEventElapsedTime(&ms, start, stop));
      std::printf(
          "COLD mapped=%d round=%d fixed=%d us=%.4f rotating_state_MiB=128\n",
          mapped, round, fixed, ms * 1000 / (rotations * 4));
    }
  }
  CHECK(hipEventDestroy(start));
  CHECK(hipEventDestroy(stop));
  CHECK(hipStreamDestroy(stream));
}

int main() {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  // Use the frozen pre-change kernel, not the refactored implementation.
  const char *library = std::getenv("NEMOTRON_LA_REFERENCE_LIB");
  if (!library)
    return 1;
  void *handle = dlopen(library, RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
  if (!handle) {
    std::fprintf(stderr, "%s\n", dlerror());
    return 1;
  }
  referenceDecode = reinterpret_cast<decltype(referenceDecode)>(
      dlsym(handle, "hip_linear_attention_decode"));
  referencePrefill = reinterpret_cast<decltype(referencePrefill)>(
      dlsym(handle, "hip_linear_attention_prefill_chunked"));
  referenceScratchBytes = reinterpret_cast<decltype(referenceScratchBytes)>(
      dlsym(handle, "hip_linear_attention_prefill_scratch_bytes"));
  referenceF16 = reinterpret_cast<decltype(referenceF16)>(
      dlsym(handle, "hip_linear_attention_decode_f16_state"));
  if (!referenceDecode || !referencePrefill || !referenceScratchBytes ||
      !referenceF16)
    return 1;
  std::printf("Frozen reference: %s\n", library);
  test(1, 8, 64, 8, 128, 64, 1, 0, 0, false);
  test(1, 8, 64, 8, 128, 64, 1, 0, 0, true);
  test(1, 8, 64, 8, 128, 64, 1, 0, 0, false, true);
  test(1, 8, 64, 8, 128, 64, 1, 0, 0, true, true);
  test(2, 8, 64, 8, 128, 64, 1, 0, 1, false);
  test(1, 64, 64, 64, 128, 64, 1, 0, 0, false);
  test(1, 64, 64, 64, 128, 64, 1, 0, 0, true);
  test(1, 64, 64, 64, 128, 64, 1, 0, 0, false, true);
  test(1, 64, 64, 64, 128, 64, 1, 0, 0, true, true);
  for (int rule = 0; rule < 4; ++rule)
    for (bool alias : {false, true})
      test(1, 32, 64, 8, 128, 64, rule, 0, 1, alias);
  test(2, 8, 4, 2, 64, 32, 3, 1, 0, false);
  test(2, 8, 4, 2, 64, 32, 3, 1, 0, true);
  testHeadBroadcast(1, 1);
  testHeadBroadcast(1, 53);
  testHeadBroadcast(1, 512);
  testHeadBroadcast(2, 8);
  testColdState(false);
  testColdState(true);
  dlclose(handle);
  return 0;
}
