/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_GROWABLE_BUFFER_H
#define HIPDNN_EP_GROWABLE_BUFFER_H

// growable_buffer: small RAII helpers for the "lazy grow-on-demand, never
// shrink" buffer pattern that we already use for the shared workspace, the
// qmoe device scratch, and the qmoe pinned-host scratch. Before this header
// each call site hand-rolled the same logic (size check, 1.5x growth,
// stream-sync on grow, free-then-realloc, error logging) — see
// `hipdnn_ep_state_ensure_workspace` / `_ensure_qmoe_scratch` /
// `_ensure_qmoe_host_scratch` in hipdnn_ep_runtime_state.cpp for the
// pre-modularization template. Op modules built with this header inherit:
//
//   * grow(): one entry point, returns 0 on success / -1 on failure, ensures
//     `data()` has at least the requested byte count.
//   * 1.5x exponential growth amortization (cold-start sizes exactly to the
//     request so we don't double a large warmup allocation).
//   * Stream sync before each free (any in-flight async kernel must have
//     completed before the old buffer pointer is invalidated).
//   * One-shot constructor: bind the owning stream once at the call site,
//     then never repeat the stream argument.
//   * RAII destructor: hooks into the module registry's destroy_fn path so
//     no per-op cleanup C symbol leaks into the public runtime ABI.
//
// All allocation is via the HIP runtime (`hipMalloc` for device,
// `hipHostMalloc(hipHostMallocDefault)` for pinned host). The two flavors
// share enough code that they'd be tempting to template, but keeping them
// as concrete types lets the bitcode compiler trivially inline the alloc /
// free pair into the module destructor with no llvm.dbg / type-info bloat.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>

namespace hipdnn_ep {

namespace detail {

// Compute the post-grow allocation size. Returns the larger of `requested`
// and `1.5 * current_size` so that callers whose request increments
// monotonically by a small delta per inference (the GQA / KV-cache decode
// path is the canonical example) get O(log N) reallocations instead of one
// per step. Cold-start (current_size == 0) sizes to exactly `requested`.
inline size_t pick_grow_size(size_t current_size, size_t requested) {
  if (current_size == 0)
    return requested;
  size_t grown = current_size + current_size / 2; // 1.5x
  return grown > requested ? grown : requested;
}

} // namespace detail

// Device buffer (hipMalloc / hipFree). Stream is provided at construction
// and used only for the pre-free synchronization on grow. Use nullptr for
// the stream argument if the buffer is constructed before the runtime
// stream is known; sync_on_grow_only() then becomes a no-op (callers that
// know the buffer is dead need to ensure no kernel still references it).
class GrowableDeviceBuffer {
public:
  // Construct an empty buffer bound to `stream`. The buffer holds no
  // device memory until grow() is called.
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

  // Bind / rebind the stream. Safe to call before any grow() has run.
  void set_stream(hipStream_t stream) { stream_ = stream; }

  // Pointer to the underlying device buffer. Stable across calls UNTIL the
  // next successful grow() that actually enlarges the buffer.
  void *data() const noexcept { return ptr_; }

  size_t size() const noexcept { return size_; }

  // Ensure the buffer is at least `needed` bytes. Returns 0 on success
  // (already large enough OR successfully grown), -1 on allocation failure.
  // On grow we sync the bound stream first so any in-flight kernel using
  // the old buffer completes before we free it.
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
      // The caller (typically an op-module destructor) is responsible for
      // having already synchronized the stream — module destruction runs
      // inside hipdnn_ep_state_cleanup AFTER the shared stream sync.
      hipFree(ptr_);
      ptr_ = nullptr;
      size_ = 0;
    }
  }

  hipStream_t stream_ = nullptr;
  void *ptr_ = nullptr;
  size_t size_ = 0;
};

// Pinned host buffer (hipHostMalloc / hipHostFree). Same semantics as
// GrowableDeviceBuffer; used for D2H readback staging when the host side
// is read with hipMemcpyAsync (pageable host memory silently falls back
// to sync-staging on Windows).
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
        // Sync first: any in-flight hipMemcpyAsync(D2H/H2D) targeting this
        // pinned buffer must complete before we free it.
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
