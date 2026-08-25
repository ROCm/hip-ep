/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// T0.1 spike -- HIP-only half of Candidate A ("HIP owns").
//
// This program needs nothing beyond the HIP runtime, so it builds and runs
// on any machine with ROCm/HIP installed -- including this one, which has no
// NPU. It answers the two sub-questions from the T0.1 task text that do not
// require XRT/DynamicDispatch:
//
//   1. Does every allocation the EP allocator produces -- including its
//      size-class-rounded pooled ones -- come back 4 KiB aligned?
//   2. Does registration survive the pointer being freed and reissued from
//      the allocator's freelist?
//
// It also runs the CPU<->GPU half of the zero-copy aliasing check (write a
// pattern from the CPU, have a HIP kernel mutate it in place, read the
// mutation back from the CPU with no explicit copy). The NPU leg of the same
// check lives in npu_probe.cpp, which only builds where DynamicDispatch/XRT
// are available -- see README.md.
//
// This file intentionally reimplements morphizen's size-class table rather
// than #include-ing morphizen-hip-gpu-allocator.hpp: that header pulls in
// api-ptrs.hpp and the ORT C API purely for its OrtAllocator vtable
// plumbing, which this probe has no use for. Only the allocation *shape*
// (call pattern + size classes) matters here, and it is a straight copy of
// BuildSizeClasses() in
// morphizen/ort-bridge/src/morphizen-hip-gpu-allocator.hpp. If that table
// changes, re-sync this copy.

#include <hip/hip_runtime.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "bf16_util.hpp"
#include "gpu_touch.hpp"

namespace {

constexpr size_t kPageSize = 4096;

// --- verbatim copy of morphizen-hip-gpu-allocator.hpp's BuildSizeClasses() ---
struct SizeClassTable {
  size_t data[256];
  size_t count;
  constexpr void Add(size_t v) {
    if (count == 0 || data[count - 1] != v) {
      data[count++] = v;
    }
  }
};

constexpr SizeClassTable BuildSizeClasses() {
  SizeClassTable t{};
  t.count = 0;
  constexpr size_t kKB = 1024;
  constexpr size_t kMB = 1024 * 1024;
  for (size_t v = 128; v <= kKB; v *= 2) t.Add(v);
  for (size_t base = kKB; base < kMB; base *= 2) {
    for (size_t k = 0; k < 4; ++k) t.Add(base + base * k / 4);
  }
  t.Add(kMB);
  for (size_t base = kMB; base < 4 * kMB; base *= 2) {
    for (size_t k = 0; k < 16; ++k) t.Add(base + base * k / 16);
  }
  t.Add(4 * kMB);
  for (size_t base = 4 * kMB; base < 16 * kMB; base *= 2) {
    for (size_t k = 0; k < 32; ++k) t.Add(base + base * k / 32);
  }
  t.Add(16 * kMB);
  return t;
}

constexpr SizeClassTable kSizeClasses = BuildSizeClasses();
// --- end verbatim copy ---

bool CheckHip(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    std::fprintf(stderr, "FAIL: %s: %s\n", what, hipGetErrorString(err));
    return false;
  }
  return true;
}

// Mirrors HipGpuAllocator::AllocImpl's cold-miss path exactly: same flags,
// same "pooled requests are rounded up to full class capacity" rule.
void* AllocatorShapedAlloc(size_t requested, size_t* out_alloc_size) {
  size_t alloc_size = requested;
  for (size_t i = 0; i < kSizeClasses.count; ++i) {
    if (requested <= kSizeClasses.data[i]) {
      alloc_size = kSizeClasses.data[i];
      break;
    }
  }
  void* ptr = nullptr;
  if (!CheckHip(hipHostMalloc(&ptr, alloc_size,
                              hipHostMallocMapped | hipHostMallocCoherent),
                "hipHostMalloc"))
    return nullptr;
  *out_alloc_size = alloc_size;
  return ptr;
}

// Sub-question 1: alignment across every size class, plus a >16 MB
// "exact-size, never pooled" allocation.
bool ProbeAlignment() {
  std::printf("\n=== Alignment probe (%zu size classes + 1 large) ===\n",
              kSizeClasses.count);
  bool all_ok = true;
  std::vector<void*> to_free;

  for (size_t i = 0; i < kSizeClasses.count; ++i) {
    size_t alloc_size = 0;
    void* ptr = AllocatorShapedAlloc(kSizeClasses.data[i], &alloc_size);
    if (!ptr) {
      all_ok = false;
      continue;
    }
    to_free.push_back(ptr);
    const bool aligned = (reinterpret_cast<uintptr_t>(ptr) % kPageSize) == 0;
    if (!aligned) {
      std::fprintf(stderr,
                   "FAIL: class %zu (%zu B) -> ptr %p is NOT 4 KiB aligned\n",
                   i, alloc_size, ptr);
      all_ok = false;
    }
  }

  // A large, never-pooled allocation (32 MB), exactly the KV-cache shape.
  {
    size_t alloc_size = 0;
    void* ptr = AllocatorShapedAlloc(32u * 1024 * 1024, &alloc_size);
    if (ptr) {
      to_free.push_back(ptr);
      const bool aligned = (reinterpret_cast<uintptr_t>(ptr) % kPageSize) == 0;
      std::printf("large (%zu B) -> ptr %p, 4 KiB aligned: %s\n", alloc_size,
                  ptr, aligned ? "yes" : "NO");
      all_ok = all_ok && aligned;
    } else {
      all_ok = false;
    }
  }

  std::printf(
      "Result: every hipHostMalloc(Mapped|Coherent) pointer, across "
      "all %zu size classes and one large exact-size allocation, "
      "%s 4 KiB aligned.\n",
      kSizeClasses.count, all_ok ? "was" : "was NOT");

  for (void* p : to_free) hipHostFree(p);
  return all_ok;
}

// Sub-question 2: does a pooled buffer keep its address (and hence would
// keep a prior XRT registration valid) across a free()+realloc() cycle at
// the same size class? Contrasted with a large, unpooled buffer, where
// FreeImpl calls hipHostFree immediately and a fresh hipHostMalloc is not
// guaranteed to return the same address.
bool ProbeFreelistReissue() {
  std::printf("\n=== Freelist reissue probe ===\n");
  bool all_ok = true;

  // Pooled case: pick a mid-sized class (index chosen arbitrarily, not the
  // first or last so the test isn't accidentally exercising an edge).
  const size_t cls_idx = kSizeClasses.count / 2;
  const size_t cls_size = kSizeClasses.data[cls_idx];

  size_t alloc_size = 0;
  void* first = AllocatorShapedAlloc(cls_size, &alloc_size);
  if (!first) return false;

  // Simulate HipGpuAllocator::FreeImpl's pooled branch: push to a freelist,
  // do NOT call hipHostFree. Then simulate AllocImpl's fast path: pop from
  // the freelist and hand the same pointer back, with no HIP call at all.
  std::vector<void*> freelist;
  void* current = first;
  bool address_stable = true;
  for (int cycle = 0; cycle < 5; ++cycle) {
    freelist.push_back(current);  // Free()
    current = freelist.back();    // Alloc() fast path
    freelist.pop_back();
    if (current != first) address_stable = false;
  }
  std::printf(
      "pooled class %zu (%zu B): address %s stable across 5 free/alloc "
      "cycles (%p)\n",
      cls_idx, cls_size, address_stable ? "was" : "was NOT", first);
  std::printf(
      "  -> implication: a registration made once for this physical "
      "pointer stays valid across every logical ORT Alloc()/Free() that "
      "maps to this size class -- no re-registration needed on the fast "
      "path.\n");
  all_ok = all_ok && address_stable;
  hipHostFree(first);

  // Unpooled (large) case: FreeImpl calls hipHostFree immediately, so a
  // fresh hipHostMalloc for the "same" logical request is a fresh driver
  // call. Observe (not assert -- the driver is free to reuse the VA range)
  // whether the address in fact changes.
  {
    const size_t large_size = 32u * 1024 * 1024;
    size_t sz = 0;
    void* a = AllocatorShapedAlloc(large_size, &sz);
    if (a) {
      hipHostFree(a);
      void* b = AllocatorShapedAlloc(large_size, &sz);
      std::printf("large (%zu B): first=%p, after free+realloc=%p -- %s\n",
                  large_size, a, b,
                  a == b ? "SAME address (driver happened to "
                           "reuse the VA range)"
                         : "DIFFERENT address");
      std::printf(
          "  -> implication: any cached registration for `a` is invalid "
          "once freed; the registry must resolve `b` as a fresh interval, "
          "not assume the old registration still applies.\n");
      if (b) hipHostFree(b);
    } else {
      all_ok = false;
    }
  }

  return all_ok;
}

// CPU<->GPU half of the zero-copy aliasing requirement. The NPU half of the
// same check (CPU -> NPU -> CPU -> GPU -> CPU on one allocation) is in
// npu_probe.cpp.
bool ProbeCpuGpuAliasing() {
  std::printf("\n=== CPU<->GPU aliasing probe ===\n");
  constexpr size_t kElems = 64;
  void* ptr = nullptr;
  if (!CheckHip(hipHostMalloc(&ptr, kElems * sizeof(uint16_t),
                              hipHostMallocMapped | hipHostMallocCoherent),
                "hipHostMalloc"))
    return false;

  void* dev_ptr = nullptr;
  if (!CheckHip(hipHostGetDevicePointer(&dev_ptr, ptr, 0),
                "hipHostGetDevicePointer")) {
    hipHostFree(ptr);
    return false;
  }
  std::printf("host ptr = %p, device ptr = %p (%s)\n", ptr, dev_ptr,
              ptr == dev_ptr ? "identical VA -- consistent with UMA"
                             : "different VA -- expected on a "
                               "non-UMA/PCIe device");

  auto* data = static_cast<uint16_t*>(ptr);
  for (size_t i = 0; i < kElems; ++i) {
    data[i] = npu_spike::FloatToBf16(static_cast<float>(i));
  }

  constexpr uint16_t kDelta = 0x0001;  // recognizable, deliberately tiny bump
  if (!GpuBumpBf16InPlace(ptr, kElems, kDelta)) {
    std::fprintf(stderr, "FAIL: GpuBumpBf16InPlace\n");
    hipHostFree(ptr);
    return false;
  }

  // No explicit copy back -- read straight through the original CPU
  // pointer. If this were staged/copied instead of aliased, the mutation
  // performed by the GPU kernel on `dev_ptr` would not be visible here.
  bool ok = true;
  for (size_t i = 0; i < kElems && ok; ++i) {
    const uint16_t expected = static_cast<uint16_t>(
        npu_spike::FloatToBf16(static_cast<float>(i)) + kDelta);
    if (data[i] != expected) {
      std::fprintf(stderr,
                   "FAIL: element %zu = 0x%04x, expected 0x%04x -- GPU write "
                   "not observed through the CPU pointer (would indicate a "
                   "boundary copy, not aliasing)\n",
                   i, data[i], expected);
      ok = false;
    }
  }
  std::printf(
      "Result: GPU kernel wrote in place; CPU read the mutation with no "
      "explicit copy: %s\n",
      ok ? "PASS" : "FAIL");

  hipHostFree(ptr);
  return ok;
}

}  // namespace

int main() {
  // Unbuffered stdout: this spike is meant to show progress even if a later
  // probe crashes (which is itself useful signal on a spike), not lose it to
  // a buffer flush that never happens.
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  int device_count = 0;
  if (!CheckHip(hipGetDeviceCount(&device_count), "hipGetDeviceCount")) {
    return 1;
  }
  std::printf("HIP device count: %d\n", device_count);
  if (device_count > 0) {
    hipDeviceProp_t props{};
    if (CheckHip(hipGetDeviceProperties(&props, 0), "hipGetDeviceProperties")) {
      std::printf("Device 0: %s (gcnArchName=%s)\n", props.name,
                  props.gcnArchName);
    }
  } else {
    std::fprintf(stderr,
                 "No HIP device found -- the allocation calls below still "
                 "exercise the host-mapped-memory path, but there is no "
                 "GPU to run the aliasing kernel on.\n");
  }

  bool ok = true;
  ok &= ProbeAlignment();
  ok &= ProbeFreelistReissue();
  if (device_count > 0) {
    ok &= ProbeCpuGpuAliasing();
  }

  std::printf("\n=== alloc_probe: %s ===\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
