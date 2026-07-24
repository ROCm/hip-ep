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
#include <mutex>
#include <unordered_map>
#include <vector>

namespace morphizen {

// Fixed size-class boundaries (bytes). A request of N bytes is served from the
// first class whose capacity is >= N, and the buffer is allocated at the full
// class capacity so any later request mapping to the same class can reuse it.
// Requests larger than the last class (> 16 MB) are NOT pooled: they are
// allocated at their exact size and released straight back to the driver in
// FreeImpl, so a one-off huge transient can never pin memory in a free list.
//
// The class boundaries are generated at compile time in four tiers, with the
// tier edges de-duplicated where they meet:
//   [128 B, 1 KB] : powers of two                   -> 128, 256, 512, 1024
//   (1 KB, 1 MB]  : 4 steps per octave (3 inserts)  ->
//   base*{1, 1.25, 1.5, 1.75} (1 MB, 4 MB]  : 16 steps per octave (15 inserts)
//   -> base*{1 + k/16} (4 MB, 16 MB] : 32 steps per octave (31 inserts) ->
//   base*{1 + k/32}
// This gives fine granularity for the small/medium transients a model churns
// every Run while keeping the total class count modest.
namespace detail {

// Compile-time-built table of size classes. Capacity (256) is comfortably
// above the ~140 classes the four tiers below actually produce; `count` is the
// number of valid leading entries in `data`.
struct SizeClassTable {
  size_t data[256];
  size_t count;
  // De-duplicating append (tier edges 1 KB and 1 MB are produced twice). A
  // constexpr member function is used instead of a local lambda because
  // defining a lambda variable inside a constexpr function is not allowed
  // before C++23.
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
  // Tier 1: 128 B .. 1 KB, doubling.
  for (size_t v = 128; v <= kKB; v *= 2) {
    t.Add(v);
  }
  // Tier 2: 1 KB .. 1 MB, quarter steps within each octave (3 inserts/octave).
  for (size_t base = kKB; base < kMB; base *= 2) {
    for (size_t k = 0; k < 4; ++k) {
      t.Add(base + base * k / 4);
    }
  }
  t.Add(kMB);
  // Tier 3: 1 MB .. 4 MB, sixteenth steps per octave (15 inserts/octave).
  for (size_t base = kMB; base < 4 * kMB; base *= 2) {
    for (size_t k = 0; k < 16; ++k) {
      t.Add(base + base * k / 16);
    }
  }
  t.Add(4 * kMB);
  // Tier4: 4 MB .. 16 MB, thirty-second steps per octave (31 inserts/octave).
  for (size_t base = 4 * kMB; base < 16 * kMB; base *= 2) {
    for (size_t k = 0; k < 32; ++k) {
      t.Add(base + base * k / 32);
    }
  }
  t.Add(16 * kMB);
  return t;
}

inline constexpr SizeClassTable kSizeClassTable = BuildSizeClasses();

} // namespace detail

inline constexpr const size_t *kSizeClasses = detail::kSizeClassTable.data;
inline constexpr size_t kNumSizeClasses = detail::kSizeClassTable.count;

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
  HipGpuAllocator(const OrtMemoryInfo *memory_info, const OrtApi &api);
  // Frees every pinned buffer this allocator ever handed out (both the
  // currently-free pool entries and any still checked out). Called by the
  // factory's ReleaseAllocator at session teardown.
  ~HipGpuAllocator();

private:
  static void *ORT_API_CALL AllocImpl(OrtAllocator *this_, size_t size);
  static void ORT_API_CALL FreeImpl(OrtAllocator *this_, void *p);
  static const OrtMemoryInfo *ORT_API_CALL InfoImpl(const OrtAllocator *this_);
  static void *ORT_API_CALL ReserveImpl(OrtAllocator *this_, size_t size);

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
  // model sees). Pooled (<= 16 MB) buffers are never returned to the driver in
  // Free; they are released wholesale in the destructor — matching the
  // project's per-session "grow-on-demand, never shrink, free at cleanup"
  // memory contract.
  //
  // Requests larger than the largest size class (> 16 MB) are NOT pooled: they
  // are allocated at their exact size and released straight back to the driver
  // in FreeImpl. A model's largest transients are few and shape-specific, so
  // pooling them buys little reuse while risking an unbounded pinned working
  // set when a model grows its largest transient a little each Run. Treating
  // them as one-shot allocations keeps peak memory bounded with no eviction
  // bookkeeping.
  //
  // Reuse needs no per-handout stream sync: ORT only calls Free after Run
  // returns, and allocator-mode inference_compute ends with a full
  // hipdnn_ep_stream_sync, so any GPU work touching a freed buffer has
  // already drained before it can be handed back out.
  std::mutex pool_mutex_;
  // Index i holds reusable buffers each exactly kSizeClasses[i] bytes.
  std::array<std::vector<void *>, kNumSizeClasses> free_lists_;
  // Every pointer hipHostMalloc'd by this allocator that is still outstanding
  // -> its allocated byte size (the rounded-up class capacity for pooled
  // buffers, the exact size for large ones). Pooled buffers stay tracked for
  // the allocator's lifetime (Free only moves them onto a free list) so the
  // destructor can release them; a large buffer is erased from this map in
  // FreeImpl when it is freed back to the driver.
  std::unordered_map<void *, size_t> ptr_to_size_;

  const OrtMemoryInfo *memory_info_;
  // Cached at construction time. -1 means "couldn't read it from memory_info"
  // (e.g. degenerate / fake OrtMemoryInfo); AllocImpl falls back to the
  // current HIP device in that case rather than failing the allocation.
  int device_id_;

  // Optional MemoryManager reference for KV cache buffer tracking (Phase 4a).
  // Set after session init via set_memory_manager(). When non-null, large
  // (>16MB) allocations are registered as KV cache buffers so the EP can
  // track KV memory usage. Typed as void* to avoid cross-module include of
  // mm/memory_manager.h.
  void *memory_manager_ = nullptr;

public:
  // Connect this allocator to a MemoryManager for KV cache tracking.
  // `mm` is a MemoryManager* cast to void*. Must be called after the
  // session's inference_init creates the MM. Safe to call with nullptr
  // (disables tracking). Not thread-safe with concurrent Alloc/Free calls
  // (but ORT serializes session init vs Run).
  void set_memory_manager(void *mm) { memory_manager_ = mm; }
};

// hipMemcpy / hipMemcpyAsync based OrtDataTransferImpl. A single shared
// instance lives in the factory for the whole process lifetime.
struct HipDataTransferImpl : OrtDataTransferImpl {
  explicit HipDataTransferImpl(const OrtApi &ort_api_in);

private:
  static bool CanCopyImpl(const OrtDataTransferImpl *this_ptr,
                          const OrtMemoryDevice *src_memory_device,
                          const OrtMemoryDevice *dst_memory_device) noexcept;

  static OrtStatus *CopyTensorsImpl(OrtDataTransferImpl *this_ptr,
                                    const OrtValue **src_tensors,
                                    OrtValue **dst_tensors,
                                    OrtSyncStream **streams,
                                    size_t num_tensors) noexcept;

  static void ReleaseImpl(OrtDataTransferImpl *this_ptr) noexcept;

  const OrtApi &ort_api;
  const OrtEpApi &ep_api;
};

} // namespace morphizen
