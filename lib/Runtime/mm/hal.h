/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- hal.h - Hardware Abstraction Layer for memory management -----------===//
//
// Defines HalAllocator — the polymorphic interface between MemoryManager and
// the HIP runtime. Two concrete backends:
//   ApuHalAllocator  — integrated GPU / UMA (hipHostMalloc Mapped+NonCoherent;
//                      gpu_ptr == cpu_ptr; eviction/restore are fence-only)
//   DiscreteHalAllocator — discrete dGPU (hipMalloc VRAM + hipHostMalloc
//                          pinned; both allocated at alloc() time;
//                          eviction/restore are async D2H/H2D memcpy)
//
// When compiled with HIPDNN_EP_MM_MOCK_HAL, both backends replace HIP calls
// with malloc/free/memcpy — no HIP SDK required (used by unit tests).
//
//===----------------------------------------------------------------------===//

#ifndef HIPDNN_EP_RUNTIME_MM_HAL_H
#define HIPDNN_EP_RUNTIME_MM_HAL_H

#include <cstddef>
#include <cstdint>

// Forward-declare hipStream_t without pulling in hip_runtime.h; real builds
// include runtime_types.h which provides the real definition; mock builds
// define it as void* in mock_types.h.
struct ihipStream_t;
typedef ihipStream_t *hipStream_t_alias;

//===----------------------------------------------------------------------===//
// Memory class tags — passed at call sites, not embedded in the IR.
//===----------------------------------------------------------------------===//

// Values match HIPDNN_MM_CLASS_* constants in hipdnn_ep_runtime.h.
enum class MemClass : int {
  Weight = 0,      // Model constants blob (session-lifetime, GPU VRAM)
  Activation = 1,  // Intermediate tensors (inference-scoped, pool-managed)
  KVCache = 2,     // KV cache blocks (sequence-scoped, eviction-eligible)
  Scratch = 3,     // Temporary workspace (op-scoped, bump-ptr recycled)
  HostScalar = 4,  // Tiny scalars requiring host-write + GPU-read access
};

//===----------------------------------------------------------------------===//
// Memory tier
//===----------------------------------------------------------------------===//

enum class MemTier : int {
  GPU = 0, // Tier 0 — primary compute tier (VRAM or UMA-GPU)
  CPU = 1, // Tier 1 — pinned host RAM (slower but larger)
};

//===----------------------------------------------------------------------===//
// HalBlock — one physical allocation made by the HAL
//===----------------------------------------------------------------------===//
//
// On APU (UMA): gpu_ptr == cpu_ptr (same VA, no copy ever needed).
//   alloc() calls hipHostMalloc(Mapped|NonCoherent); hipHostGetDevicePointer
//   returns the same VA as the host ptr on current AMD APU hardware.
//
// On discrete GPU: gpu_ptr is VRAM VA; cpu_ptr is pinned host VA.
//   Both are allocated up-front (no deferred host alloc at eviction time)
//   to avoid hipHostMalloc latency spikes under memory pressure.
//
// Invariant: gpu_ptr is always non-null after a successful alloc().
//            cpu_ptr is always non-null after a successful alloc().
//            (For APU they are the same pointer.)
//
struct HalBlock {
  void *gpu_ptr = nullptr; // GPU-accessible VA (always valid after alloc)
  void *cpu_ptr = nullptr; // Host-accessible VA (always valid after alloc)
  size_t size = 0;
  MemTier resident = MemTier::GPU; // which tier holds the authoritative data
  bool gpu_valid = false;          // data in gpu_ptr is up-to-date
  bool cpu_valid = false;          // data in cpu_ptr is up-to-date
};

//===----------------------------------------------------------------------===//
// HalAllocator — abstract interface
//===----------------------------------------------------------------------===//

class HalAllocator {
public:
  // Allocate `bytes` bytes preferring `preferred` tier.
  // On failure: returns a HalBlock with gpu_ptr == nullptr.
  virtual HalBlock alloc(size_t bytes, MemTier preferred) = 0;

  // Free a block returned by alloc(). No-op if block.gpu_ptr == nullptr.
  virtual void free(HalBlock &block) = 0;

  // Async D2H: copy gpu_ptr → cpu_ptr on `stream`.
  // Sets block.cpu_valid = true; block.resident = CPU after the stream
  // work completes. On APU: no data copy, only a fence event is recorded.
  // `stream` may be nullptr in tests (mock: synchronous).
  virtual void evict_to_cpu(HalBlock &block, void *stream) = 0;

  // Async H2D: copy cpu_ptr → gpu_ptr on `stream`.
  // Sets block.gpu_valid = true; block.resident = GPU after stream work.
  // On APU: no data copy, only a wait-event is issued.
  // `stream` may be nullptr in tests (mock: synchronous).
  virtual void restore_to_gpu(HalBlock &block, void *stream) = 0;

  // Returns true for integrated GPU / UMA hardware.
  virtual bool is_integrated() const = 0;

  virtual ~HalAllocator() = default;
};

//===----------------------------------------------------------------------===//
// Factory — returns the correct backend for the current device
//===----------------------------------------------------------------------===//

// Detects hipDeviceAttributeIntegrated (or uses the mock path under
// HIPDNN_EP_MM_MOCK_HAL) and constructs the matching allocator.
// Caller owns the returned pointer; delete when the session ends.
HalAllocator *hal_create_for_device(int device_id);

//===----------------------------------------------------------------------===//
// Concrete backends (declared here, implemented in hal_apu.cpp /
// hal_discrete.cpp). Exposed so unit tests can construct either one directly.
//===----------------------------------------------------------------------===//

class ApuHalAllocator : public HalAllocator {
public:
  HalBlock alloc(size_t bytes, MemTier preferred) override;
  void free(HalBlock &block) override;
  void evict_to_cpu(HalBlock &block, void *stream) override;
  void restore_to_gpu(HalBlock &block, void *stream) override;
  bool is_integrated() const override { return true; }
};

class DiscreteHalAllocator : public HalAllocator {
public:
  HalBlock alloc(size_t bytes, MemTier preferred) override;
  void free(HalBlock &block) override;
  void evict_to_cpu(HalBlock &block, void *stream) override;
  void restore_to_gpu(HalBlock &block, void *stream) override;
  bool is_integrated() const override { return false; }
};

#endif // HIPDNN_EP_RUNTIME_MM_HAL_H
