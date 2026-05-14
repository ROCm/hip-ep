/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_GROWABLE_BUFFER_H
#define HIPDNN_EP_GROWABLE_BUFFER_H

// Move-only RAII helpers for the "grow-on-demand, never shrink" buffer
// pattern. Two concrete types (no templates) so the bitcode compiler can
// inline the alloc / free pair into module destructors. grow() returns
// 0 / -1; sync-then-realloc on grow; destructor frees without pre-sync
// (relies on hipdnn_ep_state_cleanup having already synced the stream).

#include <hip/hip_runtime.h>

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

} // namespace detail

// Device buffer (hipMalloc / hipFree). `stream` is used only to sync
// before freeing the old pointer on grow.
class GrowableDeviceBuffer {
public:
  explicit GrowableDeviceBuffer(hipStream_t stream = nullptr)
      : stream_(stream) {}

  GrowableDeviceBuffer(const GrowableDeviceBuffer &) = delete;
  GrowableDeviceBuffer &operator=(const GrowableDeviceBuffer &) = delete;

  GrowableDeviceBuffer(GrowableDeviceBuffer &&other) noexcept
      : stream_(other.stream_), ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
  }
  GrowableDeviceBuffer &operator=(GrowableDeviceBuffer &&other) noexcept {
    if (this != &other) {
      free_now();
      stream_ = other.stream_;
      ptr_ = other.ptr_;
      size_ = other.size_;
      other.ptr_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  ~GrowableDeviceBuffer() { free_now(); }

  void set_stream(hipStream_t stream) { stream_ = stream; }

  // Stable across calls until the next grow() that actually enlarges it.
  void *data() const noexcept { return ptr_; }
  size_t size() const noexcept { return size_; }

  // Returns 0 (already large enough OR grown successfully) or -1 (alloc
  // failure). Syncs the bound stream before freeing the old pointer.
  int grow(size_t needed) {
    if (needed == 0)
      return 0;
    if (size_ >= needed)
      return 0;

    size_t alloc_size = detail::pick_grow_size(size_, needed);

    if (ptr_) {
      if (stream_) {
        hipStreamSynchronize(stream_);
      }
      hipFree(ptr_);
      ptr_ = nullptr;
      size_ = 0;
    }

    if (hipMalloc(&ptr_, alloc_size) != hipSuccess) {
      fprintf(stderr, "GrowableDeviceBuffer: hipMalloc failed for %zu bytes\n",
              alloc_size);
      ptr_ = nullptr;
      return -1;
    }
    size_ = alloc_size;
    return 0;
  }

private:
  void free_now() noexcept {
    if (ptr_) {
      hipFree(ptr_);
      ptr_ = nullptr;
      size_ = 0;
    }
  }

  hipStream_t stream_ = nullptr;
  void *ptr_ = nullptr;
  size_t size_ = 0;
};

// Pinned host buffer (hipHostMalloc / hipHostFree). Same shape as
// GrowableDeviceBuffer. Required for D2H staging on Windows where
// pageable host memory silently falls back to sync-staging.
class GrowablePinnedBuffer {
public:
  explicit GrowablePinnedBuffer(hipStream_t stream = nullptr)
      : stream_(stream) {}

  GrowablePinnedBuffer(const GrowablePinnedBuffer &) = delete;
  GrowablePinnedBuffer &operator=(const GrowablePinnedBuffer &) = delete;

  GrowablePinnedBuffer(GrowablePinnedBuffer &&other) noexcept
      : stream_(other.stream_), ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
  }
  GrowablePinnedBuffer &operator=(GrowablePinnedBuffer &&other) noexcept {
    if (this != &other) {
      free_now();
      stream_ = other.stream_;
      ptr_ = other.ptr_;
      size_ = other.size_;
      other.ptr_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  ~GrowablePinnedBuffer() { free_now(); }

  void set_stream(hipStream_t stream) { stream_ = stream; }
  void *data() const noexcept { return ptr_; }
  size_t size() const noexcept { return size_; }

  int grow(size_t needed) {
    if (needed == 0)
      return 0;
    if (size_ >= needed)
      return 0;

    size_t alloc_size = detail::pick_grow_size(size_, needed);

    if (ptr_) {
      if (stream_) {
        // Any in-flight hipMemcpyAsync targeting this pinned buffer must
        // complete before we free it.
        hipStreamSynchronize(stream_);
      }
      hipHostFree(ptr_);
      ptr_ = nullptr;
      size_ = 0;
    }

    if (hipHostMalloc(&ptr_, alloc_size, hipHostMallocDefault) != hipSuccess) {
      fprintf(stderr,
              "GrowablePinnedBuffer: hipHostMalloc failed for %zu bytes\n",
              alloc_size);
      ptr_ = nullptr;
      return -1;
    }
    size_ = alloc_size;
    return 0;
  }

private:
  void free_now() noexcept {
    if (ptr_) {
      hipHostFree(ptr_);
      ptr_ = nullptr;
      size_ = 0;
    }
  }

  hipStream_t stream_ = nullptr;
  void *ptr_ = nullptr;
  size_t size_ = 0;
};

} // namespace hipdnn_ep

#endif // HIPDNN_EP_GROWABLE_BUFFER_H
