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

#include <cstddef>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace morphizen {

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

  // Caching free-list. hipHostMalloc is a heavyweight (page-pinning) call:
  // ORT re-allocates the per-Run input device-copy buffers and (allocator
  // mode) the graph-output buffers on EVERY inference, so without caching the
  // EP pays dozens of hipHostMalloc + hipHostFree per Compute (the dominant
  // non-compute cost on small fixed-shape graphs). Instead Free returns the
  // buffer to a size-keyed free list and Alloc reuses it; the driver is only
  // touched on a cold miss (first inference / a newly-seen size). The pool
  // grows on demand and is released wholesale in the destructor — matching
  // the project's per-session "grow-on-demand, never shrink, free at cleanup"
  // memory contract. Keyed on exact byte size so fixed-shape models reuse at
  // 100% hit rate; distinct dynamic-shape sizes simply pool separately.
  //
  // Reuse needs no per-handout stream sync: ORT only calls Free after Run
  // returns, and allocator-mode inference_compute ends with a full
  // hipdnn_ep_stream_sync, so any GPU work touching a freed buffer has
  // already drained before it can be handed back out.
  std::mutex pool_mutex_;
  std::unordered_map<size_t, std::vector<void*>> free_lists_;
  // Every pointer hipHostMalloc'd by this allocator -> its byte size. Entries
  // persist for the allocator's lifetime (Free does not erase them; it only
  // moves the pointer onto free_lists_) so the destructor can free them all
  // and FreeImpl can recover a pointer's bucket size.
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
