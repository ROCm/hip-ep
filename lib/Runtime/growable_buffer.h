/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_GROWABLE_BUFFER_H
#define HIPDNN_EP_GROWABLE_BUFFER_H

// Move-only RAII helper for the "grow-on-demand, never shrink" buffer
// pattern. One class template, two policies:
//   GrowableDeviceBuffer  - hipMalloc     / hipFree
//   GrowablePinnedBuffer  - hipHostMalloc / hipHostFree
//
// grow() returns 0 (already large enough OR grown successfully) or -1
// (alloc failure). On grow we free the old pointer and allocate a new
// one; we do NOT call hipStreamSynchronize ourselves because the HIP
// runtime API specifies that both hipFree and hipHostFree "perform an
// implicit hipDeviceSynchronize() call" before reclaiming the buffer
// (see the HIP 7.0 reference: Memory management). Synchronizing on top
// of that would be redundant and slightly broader (device vs stream is
// implementation-defined). The destructor uses the same property.
//
// HIP calls outside of grow()'s alloc step are best-effort: a failure
// in hipFree / hipHostFree is logged via HIP_CLEANUP but does not abort
// the surrounding op. The alloc step is the only one that returns -1
// to the caller, because that is the only failure mode the caller can
// usefully react to.

// runtime_types.h supplies hipError_t, hipMalloc, hipFree, hipHostMalloc,
// hipHostFree, hipGetErrorString, hipSuccess, and hipHostMallocDefault.
// It resolves via the active build's -I path to either real/runtime_types.h
// (which pulls in <hip/hip_runtime.h>) or mock/runtime_types.h (which
// provides shim typedefs and extern "C" mock function declarations). Do NOT
// include <hip/hip_runtime.h> directly here -- that header is absent from
// the BUILD_MOCK_RUNTIME=ON bitcode compile and would break CI's mock job.
#include "hip_cleanup.h"
#include "runtime_types.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>

namespace hipdnn_ep {

namespace detail {

// 1.5x growth amortizes monotonic per-inference deltas to O(log N)
// reallocations. Cold-start sizes to exactly `requested` so a large warmup
// allocation isn't doubled.
inline size_t pick_grow_size(size_t current_size, size_t requested) {
  if (current_size == 0)
    return requested;
  size_t grown = current_size + current_size / 2; // 1.5x
  return grown > requested ? grown : requested;
}

// Backend policies for GrowableBuffer. Each member is a tiny static
// wrapper around the corresponding HIP call; the bitcode compiler
// inlines them away (template static dispatch, no function pointers,
// no virtuals).
struct DevicePolicy {
  static hipError_t alloc(void **p, size_t n) { return hipMalloc(p, n); }
  static hipError_t free(void *p) { return hipFree(p); }
  static constexpr const char *tag() { return "GrowableDeviceBuffer"; }
  static constexpr const char *alloc_name() { return "hipMalloc"; }
};

struct PinnedPolicy {
  static hipError_t alloc(void **p, size_t n) {
    return hipHostMalloc(p, n, hipHostMallocDefault);
  }
  static hipError_t free(void *p) { return hipHostFree(p); }
  static constexpr const char *tag() { return "GrowablePinnedBuffer"; }
  static constexpr const char *alloc_name() { return "hipHostMalloc"; }
};

} // namespace detail

template <class Policy> class GrowableBuffer {
public:
  GrowableBuffer() = default;

  GrowableBuffer(const GrowableBuffer &) = delete;
  GrowableBuffer &operator=(const GrowableBuffer &) = delete;

  GrowableBuffer(GrowableBuffer &&other) noexcept
      : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
  }
  GrowableBuffer &operator=(GrowableBuffer &&other) noexcept {
    if (this != &other) {
      free_now();
      ptr_ = other.ptr_;
      size_ = other.size_;
      other.ptr_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  ~GrowableBuffer() { free_now(); }

  // Stable across calls until the next grow() that actually enlarges it.
  void *data() const noexcept { return ptr_; }
  size_t size() const noexcept { return size_; }

  int grow(size_t needed) {
    if (needed == 0)
      return 0;
    if (size_ >= needed)
      return 0;

    size_t alloc_size = detail::pick_grow_size(size_, needed);

    if (ptr_) {
      // hipFree / hipHostFree both perform an implicit
      // hipDeviceSynchronize() per the HIP runtime API, so we do NOT
      // need an explicit hipStreamSynchronize here. The free will
      // block until in-flight kernels referencing this buffer drain.
      HIP_CLEANUP(Policy::free(ptr_));
      ptr_ = nullptr;
      size_ = 0;
    }

    hipError_t alloc_err = Policy::alloc(&ptr_, alloc_size);
    if (alloc_err != hipSuccess || !ptr_) {
      fprintf(stderr, "%s: %s(%zu) failed: %s\n", Policy::tag(),
              Policy::alloc_name(), alloc_size, hipGetErrorString(alloc_err));
      ptr_ = nullptr;
      return -1;
    }
    size_ = alloc_size;
    return 0;
  }

private:
  void free_now() noexcept {
    if (ptr_) {
      HIP_CLEANUP(Policy::free(ptr_));
      ptr_ = nullptr;
      size_ = 0;
    }
  }

  void *ptr_ = nullptr;
  size_t size_ = 0;
};

// Concrete aliases preserve the names callers already use.
using GrowableDeviceBuffer = GrowableBuffer<detail::DevicePolicy>;
using GrowablePinnedBuffer = GrowableBuffer<detail::PinnedPolicy>;

} // namespace hipdnn_ep

#endif // HIPDNN_EP_GROWABLE_BUFFER_H
