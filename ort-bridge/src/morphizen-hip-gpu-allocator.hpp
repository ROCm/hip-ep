/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

// HIP GPU allocator + data transfer implementation for the MorphiZen EP.
// Only compiled when morphizen_ENABLE_HIP_GPU_ALLOCATOR=ON (the build system
// will additionally link the HIP runtime in that case). Without the flag the
// factory keeps returning the legacy "CreateAllocator should not be called"
// status and stays a CPU-only EP, so existing users that do not use the AMD
// HIP-based GPU backend (hipdnn-ep) are not affected.

#include "./api-ptrs.hpp"

#include <array>
#include <cstddef>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace morphizen {

// Fixed size-class boundaries (bytes). A request of N bytes is served from the
// first class whose capacity is >= N, and the buffer is allocated at the full
// class capacity so any later request mapping to the same class can reuse it.
// Requests larger than the last class are allocated at their exact size and
// pooled best-fit in a SEPARATE large free list that is capped at
// kMaxLargeFreeBuffers entries; on a cold miss the smallest pooled large
// buffers are evicted (freed to the driver) before a new one is allocated, so
// peak memory stays bounded even when a model grows its largest transient
// incrementally — see AllocImpl in the .cpp.
inline constexpr size_t kSizeClasses[] = {
    1ull * 1024,         // 1 KB
    2ull * 1024,         // 2 KB
    4ull * 1024,         // 4 KB
    6ull * 1024,         // 6 KB
    8ull * 1024,         // 8 KB
    12ull * 1024,        // 12 KB
    16ull * 1024,        // 16 KB
    24ull * 1024,        // 24 KB
    32ull * 1024,        // 32 KB
    48ull * 1024,        // 48 KB
    64ull * 1024,        // 64 KB
    96ull * 1024,        // 96 KB
    128ull * 1024,       // 128 KB
    192ull * 1024,       // 192 KB
    256ull * 1024,       // 256 KB
    384ull * 1024,       // 384 KB
    512ull * 1024,       // 512 KB
    768ull * 1024,       // 768 KB
    1ull * 1024 * 1024,  // 1 MB
    1536ull * 1024,      // 1.5 MB
    2ull * 1024 * 1024,  // 2 MB
    3ull * 1024 * 1024,  // 3 MB
    4ull * 1024 * 1024,  // 4 MB
    6ull * 1024 * 1024,  // 6 MB
    8ull * 1024 * 1024,  // 8 MB
    12ull * 1024 * 1024, // 12 MB
    16ull * 1024 * 1024, // 16 MB
};
inline constexpr size_t kNumSizeClasses =
    sizeof(kSizeClasses) / sizeof(kSizeClasses[0]);

// Upper bound on the number of buffers kept in the large free list (those
// bigger than the largest size class). Large buffers are expensive to keep
// pinned, so when a cold miss would grow the driver-side footprint we evict the
// smallest pooled large buffers down to this cap first. Small enough to keep
// peak memory bounded under an incrementally-growing transient, large enough to
// retain useful reuse across the few distinct large shapes a model exercises.
inline constexpr size_t kMaxLargeFreeBuffers = 4;

// hipHostMalloc(Mapped|Coherent) backed OrtAllocator. One instance is created
// per OrtMemoryInfo registered with OrtEpDevice (typically one DEFAULT GPU
// info and one HOST_ACCESSIBLE info — both currently route to the same
// mapped pinned allocator, since AMD APU iGPU shares physical memory with
// the host). ORT keeps the allocator alive for the matching session.
struct HipGpuAllocator : OrtAllocator {
  // memory_info ownership stays with the factory. The OrtMemoryInfo's
  // device_id (extracted via OrtApi::MemoryInfoGetId at construction time)
  // is what the allocator passes to hipSetDevice; this is how a single
  // factory serving multiple AMD GPUs keeps each allocator pinned to the
  // GPU that the OrtEpDevice actually represents (instead of always hitting
  // device 0).
  HipGpuAllocator(const OrtMemoryInfo* memory_info, const OrtApi& api);
  // Frees every pinned buffer this allocator ever handed out (both the
  // currently-free pool entries and any still checked out). Called by the
  // factory's ReleaseAllocator at session teardown.
  ~HipGpuAllocator();

private:
  static void* ORT_API_CALL AllocImpl(OrtAllocator* this_, size_t size);
  static void ORT_API_CALL FreeImpl(OrtAllocator* this_, void* p);
  static const OrtMemoryInfo* ORT_API_CALL InfoImpl(const OrtAllocator* this_);
  static void* ORT_API_CALL ReserveImpl(OrtAllocator* this_, size_t size);

  // Fixed size-class caching allocator. hipHostMalloc is a heavyweight
  // (page-pinning) call: ORT re-allocates the per-Run input device-copy
  // buffers and (allocator mode) the graph-output buffers on EVERY inference,
  // so without caching the EP pays dozens of hipHostMalloc + hipHostFree per
  // Compute (the dominant non-compute cost on small fixed-shape graphs).
  //
  // A request is rounded up to one of kSizeClasses and served from that
  // class's free list; the buffer is allocated at the full class capacity, so
  // every buffer within a class is interchangeable and any later request
  // mapping to the same class reuses it (100% hit rate after warmup, with at
  // most a few distinct class sizes regardless of how many dynamic shapes the
  // model sees). Requests larger than the last class are allocated at their
  // exact size and recycled best-fit (the smallest free large buffer that is
  // >= the new request is reused). Free never returns memory to the driver
  // (except large-buffer eviction below); everything is released wholesale in
  // the destructor — matching the project's per-session "grow-on-demand, never
  // shrink, free at cleanup" memory contract.
  //
  // The large free list is the one exception to "never shrink": it is capped at
  // kMaxLargeFreeBuffers, and on a cold miss for a large request AllocImpl
  // evicts the smallest pooled large buffers (freeing them to the driver) down
  // to the cap before allocating the new one. This keeps reuse performance for
  // the handful of distinct large shapes a model actually uses while preventing
  // the working set from ballooning when a model grows its largest transient a
  // little each Run (otherwise each slightly-larger request would allocate
  // fresh while every prior near-size buffer stayed pinned in the free list).
  //
  // Reuse needs no per-handout stream sync: ORT only calls Free after Run
  // returns, and allocator-mode inference_compute ends with a full
  // hipdnn_ep_stream_sync, so any GPU work touching a freed buffer has
  // already drained before it can be handed back out (or evicted).
  std::mutex pool_mutex_;
  // Index i holds reusable buffers each exactly kSizeClasses[i] bytes.
  std::array<std::vector<void*>, kNumSizeClasses> free_lists_;
  // Freed buffers larger than the largest size class, keyed by their exact
  // allocated byte size. Reuse is best-fit via lower_bound(request); eviction
  // takes the smallest via begin(). Capped at kMaxLargeFreeBuffers.
  std::multimap<size_t, void*> large_free_;
  // Every pointer hipHostMalloc'd by this allocator that is still outstanding
  // -> its allocated byte size (the rounded-up class capacity for pooled
  // buffers, the exact size for large ones). Pooled buffers stay tracked for
  // the allocator's lifetime (Free only moves them onto a free list) so the
  // destructor can release them and AllocImpl can recover an evicted large
  // buffer's size; an evicted large buffer is erased from this map in AllocImpl
  // when it is freed to the driver.
  std::unordered_map<void*, size_t> ptr_to_size_;

  const OrtMemoryInfo* memory_info_;
  // Cached at construction time. -1 means "couldn't read it from memory_info"
  // (e.g. degenerate / fake OrtMemoryInfo); AllocImpl falls back to the
  // current HIP device in that case rather than failing the allocation.
  int device_id_;
};

// hipMemcpy / hipMemcpyAsync based OrtDataTransferImpl. A single shared
// instance lives in the factory for the whole process lifetime.
struct HipDataTransferImpl : OrtDataTransferImpl {
  explicit HipDataTransferImpl(const OrtApi& ort_api_in);

private:
  static bool CanCopyImpl(const OrtDataTransferImpl* this_ptr,
                          const OrtMemoryDevice* src_memory_device,
                          const OrtMemoryDevice* dst_memory_device) noexcept;

  static OrtStatus* CopyTensorsImpl(OrtDataTransferImpl* this_ptr,
                                    const OrtValue** src_tensors,
                                    OrtValue** dst_tensors,
                                    OrtSyncStream** streams,
                                    size_t num_tensors) noexcept;

  static void ReleaseImpl(OrtDataTransferImpl* this_ptr) noexcept;

  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
};

} // namespace morphizen
