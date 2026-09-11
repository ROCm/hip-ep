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
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace morphizen {

// Fixed size-class boundaries (bytes). A request of N bytes is served from the
// first class whose capacity is >= N, and the buffer is allocated at the full
// class capacity so any later request mapping to the same class can reuse it.
// Requests larger than the last class (> 16 MB) are served at their exact size
// until a capacity has been observed to be reused; see the large-pool comment
// on HipGpuAllocator's members.
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

// Ceiling on pinned host bytes held idle in the large free lists, shared by
// every allocator one factory creates.
//
// The budget is common because ORT makes a HipGpuAllocator per OrtMemoryInfo
// and all of them pin from the same physical pool. It is owned by the factory
// rather than being a module-scoped static because the factory provably
// outlives the allocators -- it is what constructs and deletes them -- so the
// counter cannot be reached after its storage is gone, and a reload starts from
// a known state instead of relying on the loader unmapping a data segment.
struct LargePoolBudget {
  LargePoolBudget() noexcept;

  bool TryReserve(size_t n) noexcept;
  // Floored, because an unbalanced release must not wrap the counter: from
  // there the pool either refuses every reservation or stops bounding itself,
  // and both are silent.
  void Release(size_t n) noexcept;

  size_t cap() const noexcept { return cap_; }

private:
  std::atomic<size_t> used_{0};
  size_t cap_;
};

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
  //
  // `budget` is the factory's; it must outlive this allocator, which it does
  // because the factory is what deletes it.
  HipGpuAllocator(const OrtMemoryInfo *memory_info, const OrtApi &api,
                  LargePoolBudget &budget);
  // Frees every pinned buffer this allocator ever handed out (both the
  // currently-free pool entries and any still checked out). Called by the
  // factory's ReleaseAllocator at session teardown.
  ~HipGpuAllocator();

private:
  static void *ORT_API_CALL AllocImpl(OrtAllocator *this_, size_t size);
  static void ORT_API_CALL FreeImpl(OrtAllocator *this_, void *p);
  static const OrtMemoryInfo *ORT_API_CALL InfoImpl(const OrtAllocator *this_);
  static void *ORT_API_CALL ReserveImpl(OrtAllocator *this_, size_t size);

  // All three run under pool_mutex_ and append the pointers they drop to `out`
  // instead of calling hipHostFree, so the driver call happens after the
  // caller has released the lock.
  //
  // Releases capacities that have gone untouched for two decode steps' worth
  // of allocator traffic. A growing tensor eventually outgrows a capacity and
  // never asks for it again; without this those buffers are held for the
  // session.
  void TrimStaleLarge(std::vector<void *> &out);
  // Backstop for the retention cap: drops one buffer from the
  // least-recently-used capacity other than `keep`. False when nothing is
  // left to evict.
  bool EvictLruLarge(size_t keep, std::vector<void *> &out);
  void DropRetained(void *ptr, std::vector<void *> &out);

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
  // Requests above the largest size class (> 16 MB) cannot use the fixed
  // table, whose last class ends at 16 MB. They are grouped by a rounded
  // capacity (LargeCapacity in the .cpp) instead -- but only those that need
  // to be, because the two patterns a decoder produces want opposite things
  // and look identical at the first request:
  //
  //   past_present_share_buffer=false -- the decoder reallocates every
  //   past/present KV tensor every step, one token larger each time. No two
  //   requests are ever the same size, so only rounding can pool them. Left
  //   unpooled, each step hands the driver ~60-96 hipHostMalloc/hipHostFree
  //   pairs and several GiB of freshly pinned pages: at 16 K that was 856 ms
  //   of an 890 ms first generation and still 7 ms/token warm.
  //
  //   past_present_share_buffer=true -- the KV cache is allocated once at
  //   max_length and freed only when the generator goes away. Every request
  //   is the same size forever, so exact sizes pool perfectly and rounding
  //   buys nothing while costing its waste (1/8 + 1/64) on the whole KV cache
  //   in pinned pages. Rounding these unconditionally cost 6-13% of peak host
  //   memory on 12 of 15 models in a CI sweep for zero throughput.
  //
  // So retention is by exact allocated size, and a capacity only starts
  // rounding once a second, different size maps to it. That is a direct test
  // of the thing that matters -- whether this capacity serves a moving size
  // or a fixed one -- rather than a proxy, and it settles within two decode
  // steps for a growing tensor and never for a fixed one.
  //
  // Serving at exact size until then also avoids a transition that is not
  // obvious: a class that switches from size S to capacity K while S buffers
  // are still live leaves the S pages in the working set alongside the new K
  // ones. Triggering on anything a fixed-size model can do -- a free, a
  // generator teardown -- pays that once per capacity and doubled measured
  // peak memory on a share_buffer=true model.
  //
  // Retention needs two bounds once a class is rounded. LargeCapacity rounds
  // with a 1/64 growth headroom so a whole generation of growth shares one
  // capacity, and TrimStaleLarge releases capacities the model has grown out
  // of -- without it a monotonically growing tensor strands a full set of
  // buffers at every capacity it leaves behind.
  //
  // Reuse needs no per-handout stream sync: ORT only calls Free after Run
  // returns, and allocator-mode inference_compute ends with a full
  // hipdnn_ep_stream_sync, so any GPU work touching a freed buffer has
  // already drained before it can be handed back out.
  std::mutex pool_mutex_;
  // Index i holds reusable buffers each exactly kSizeClasses[i] bytes.
  std::array<std::vector<void *>, kNumSizeClasses> free_lists_;

  struct LargeClass {
    // Retained buffers, all of one size: `rounded` decides which, and the
    // buffers left over from before that flip are dropped rather than kept.
    std::vector<void *> free;
    // large_ops_ at this class's last use. Touched on both take and return, so
    // a class in rotation is refreshed twice per step per buffer while one the
    // model has outgrown stops being touched at all.
    uint64_t last_used = 0;
    // First exact size seen for this capacity, and whether a second, different
    // one has since arrived. Both sticky for the session: TrimStaleLarge
    // empties `free` but keeps the entry, so a class does not have to relearn
    // that it is moving and flip sizes underneath live buffers a second time.
    size_t witness_size = 0;
    bool rounded = false;
  };
  std::map<size_t, LargeClass> large_classes_;
  // Large allocations plus large frees. The clock last_used is measured
  // against -- not a wall clock, because what distinguishes a dead capacity
  // from an idle session is how much allocator traffic went by without it
  // being asked for.
  uint64_t large_ops_ = 0;
  // Large buffers currently checked out. A step allocates and frees each live
  // buffer once, so a step is about twice this many large ops -- the scale
  // TrimStaleLarge measures staleness in.
  size_t outstanding_large_ = 0;
  // This instance's share of the factory-wide budget, so the destructor can
  // hand back exactly what it holds.
  size_t retained_large_bytes_ = 0;
  LargePoolBudget *budget_;
  bool warned_cap_ = false;

  struct Block {
    // What was handed to hipHostMalloc: a fixed class capacity, a rounded
    // large capacity, or the request's exact size.
    size_t bytes;
    // Large capacity class, or 0 for the fixed table.
    size_t klass;
  };
  // Every pointer this allocator has outstanding. Pooled buffers stay here for
  // the allocator's lifetime (Free only moves them onto a free list) so the
  // destructor can release them; a block is erased when it goes back to the
  // driver.
  std::unordered_map<void *, Block> blocks_;

  const OrtMemoryInfo *memory_info_;
  // Cached at construction time. -1 means "couldn't read it from memory_info"
  // (e.g. degenerate / fake OrtMemoryInfo); AllocImpl falls back to the
  // current HIP device in that case rather than failing the allocation.
  int device_id_;
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
