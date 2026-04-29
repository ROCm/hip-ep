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

private:
  static void* ORT_API_CALL AllocImpl(OrtAllocator* this_, size_t size);
  static void ORT_API_CALL FreeImpl(OrtAllocator* this_, void* p);
  static const OrtMemoryInfo* ORT_API_CALL InfoImpl(const OrtAllocator* this_);
  static void* ORT_API_CALL ReserveImpl(OrtAllocator* this_, size_t size);

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
