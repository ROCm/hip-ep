/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- memory_manager.h - Unified Memory Manager --------------------------===//
//
// MemoryManager is the single owner of all session-scoped GPU and host memory.
// It replaces the six independent `ensure_*` patterns previously scattered
// across RuntimeState (workspace, host_scratch, qmoe_scratch,
// qmoe_host_scratch, conv_scratch, and the per-domain pool arrays).
//
// Phase 1: wrapped all six existing patterns behind a uniform typed API with
//   1.5× amortized growth, seqlens_k cache, begin_compute()/end_compute()
//   lifecycle hooks, and APU + discrete GPU backends.
//
// Phase 2: removed legacy RuntimeState alias fields (pool_base, pool_size,
//   workspace, host_scratch, qmoe_scratch, conv_scratch, seqlens_k_cached_*).
//   All callers now go through MemoryManager exclusively; fallback code paths
//   deleted.
//
// Phase 3: bump-pointer scratch arena backed by the shared workspace buffer.
//   scratch_alloc(N) replaces manual offset arithmetic; begin_compute() resets
//   the bump pointer. All callers migrated from ensure_workspace + manual
//   offsets to scratch_alloc().
//
// Future phases add: KvCacheManager (Phase 4), Tier-1 CPU offload (Phase 5).
// The public API surface is kept stable across phases.
//
//===----------------------------------------------------------------------===//

#ifndef HIPDNN_EP_RUNTIME_MM_MEMORY_MANAGER_H
#define HIPDNN_EP_RUNTIME_MM_MEMORY_MANAGER_H

#include "hal.h"

#include <cstddef>
#include <cstdint>

//===----------------------------------------------------------------------===//
// MemoryManager
//===----------------------------------------------------------------------===//

class MemoryManager {
public:
  // Takes ownership of `hal` (deleted in destructor).
  explicit MemoryManager(HalAllocator *hal);
  ~MemoryManager();

  // Non-copyable, non-movable (owns raw HAL allocations).
  MemoryManager(const MemoryManager &) = delete;
  MemoryManager &operator=(const MemoryManager &) = delete;

  //-------------------------------------------------------------------
  // Pool management (replaces hipdnn_ep_pool_init / hipdnn_ep_get_pool_base)
  //-------------------------------------------------------------------

  // Called from hipdnn_ep_pool_init (generated inference_init).
  // Records the compile-time static pool size and buffer offsets for domain 0.
  // Does NOT allocate yet; the actual allocation happens on the first
  // get_pool_base(0, ...) call (lazy, matches the multi-domain contract).
  void load_pool_plan(size_t pool_size, const size_t *buffer_offsets,
                      size_t num_buffers);

  // Get (and if needed, grow) the GPU pool base for `domain_id`.
  // Growth is 1.5× amortized to avoid per-inference sync+free+malloc cycles.
  // Returns nullptr on allocation failure.
  void *get_pool_base(int domain_id, size_t needed_size);

  // Legacy: return a pointer into domain 0 pool at `buffer_offsets[index]`.
  void *get_buffer_from_pool(size_t index);

  //-------------------------------------------------------------------
  // Host-scalar scratch (replaces hipdnn_ep_get_host_scratch_base)
  //-------------------------------------------------------------------

  // hipHostMalloc(Mapped|NonCoherent) backed buffer; host-writable AND
  // GPU-readable. Grows on demand (exact size, no growth factor — rarely
  // called and size is compile-time-determined by MaterializeHostScalars).
  // Returns nullptr on failure.
  void *get_host_scratch(size_t needed_size);

  //-------------------------------------------------------------------
  // Scratch arena (bump-pointer allocator over the shared workspace)
  //-------------------------------------------------------------------

  // Bump-allocate `size` bytes (64-byte aligned) from the scratch arena.
  // Backed by the shared workspace buffer; grows it on demand (1.5×
  // amortized). Returns nullptr on allocation failure. The bump pointer
  // is reset by begin_compute() at the start of each Compute().
  void *scratch_alloc(size_t size);

  // Reset the bump pointer to 0 (called by begin_compute()).
  void scratch_reset();

  // Current bump pointer offset (for diagnostics and tests).
  size_t scratch_offset() const { return scratch_offset_; }

  //-------------------------------------------------------------------
  // Shared workspace (low-level; prefer scratch_alloc for new code)
  //-------------------------------------------------------------------

  // Ensure the workspace buffer is at least `needed_size` bytes. Resets
  // the scratch bump pointer (callers that use ensure_workspace want the
  // whole buffer). Kept for gemm/matmul autotune which needs the raw
  // buffer + its full size. Returns nullptr on failure.
  void *ensure_workspace(size_t needed_size);
  void *get_workspace() const;
  size_t get_workspace_size() const;

  // QMoE pinned-host mirror for async D2H readback (distinct from GPU workspace
  // because it must be host-accessible, not GPU-accessible).
  // Growth is 1.5× amortized using hipHostMalloc(Default).
  void *ensure_qmoe_host_scratch(size_t needed_size);
  void *get_qmoe_host_scratch() const;

  //-------------------------------------------------------------------
  // Per-inference lifecycle
  //-------------------------------------------------------------------

  // Called at the start of every Compute():
  //   - Resets the scratch arena bump pointer.
  //   - Invalidates the seqlens_k cache.
  void begin_compute();

  // Called at the end of every Compute(). Currently a no-op; reserved for
  // future phases that need end-of-inference cleanup.
  void end_compute();

  //-------------------------------------------------------------------
  // seqlens_k cache (moved from RuntimeState; same semantics)
  //-------------------------------------------------------------------

  // Returns true if the cached value is valid for the given device pointer.
  bool seqlens_k_cache_valid(const void *ptr) const;

  // Read the cached seqlens_k value.
  int32_t seqlens_k_cached_val() const;

  // Populate the cache.
  void seqlens_k_cache_set(const void *ptr, int32_t val);

  //-------------------------------------------------------------------
  // Stats (for testing and diagnostics)
  //-------------------------------------------------------------------

  size_t gpu_bytes_used() const;
  size_t cpu_bytes_used() const;

  // Number of live pool domains (for tests).
  int num_pool_domains() const { return num_pool_domains_; }

  // For direct field access in unit tests (public by design in test builds).
  // In production builds these are read through the accessors above.
  bool seqlens_k_valid_ = false;
  int32_t seqlens_k_val_ = 0;
  const void *seqlens_k_ptr_ = nullptr;

private:
  //-------------------------------------------------------------------
  // Pool domain storage (replaces RuntimeState::pool_base/pool_size arrays)
  //-------------------------------------------------------------------
  struct PoolDomain {
    void *base = nullptr;
    size_t size = 0;
  };

  PoolDomain *domains_ = nullptr; // heap array, grown on demand
  int num_pool_domains_ = 0;

  // Static-pool metadata for domain 0 (from load_pool_plan).
  size_t static_pool_size_ = 0;
  size_t *buffer_offsets_ = nullptr;
  size_t num_buffers_ = 0;

  //-------------------------------------------------------------------
  // Shared GPU workspace + scratch arena bump pointer
  //-------------------------------------------------------------------
  void *workspace_ = nullptr;
  size_t workspace_size_ = 0;
  size_t scratch_offset_ = 0; // bump pointer into workspace_

  //-------------------------------------------------------------------
  // Host-scalar scratch (hipHostMalloc Mapped+NonCoherent)
  //-------------------------------------------------------------------
  void *host_scratch_ = nullptr;
  size_t host_scratch_size_ = 0;

  //-------------------------------------------------------------------
  // QMoE pinned-host mirror (hipHostMalloc Default — async DMA staging)
  //-------------------------------------------------------------------
  void *qmoe_host_scratch_ = nullptr;
  size_t qmoe_host_scratch_size_ = 0;

  //-------------------------------------------------------------------
  // Hardware abstraction backend
  //-------------------------------------------------------------------
  HalAllocator *hal_ = nullptr;

  //-------------------------------------------------------------------
  // HIP stream (borrowed from RuntimeState; used for sync-before-free)
  //-------------------------------------------------------------------
  void *stream_ = nullptr; // hipStream_t as void*; set by set_stream()

  //-------------------------------------------------------------------
  // Helpers
  //-------------------------------------------------------------------

  // Ensure domains_[domain_id] exists; zero-fills new slots.
  bool ensure_domain(int domain_id);

  // Grow a GPU buffer to at least `needed` bytes with 1.5× amortization.
  // On success, writes new ptr/size to *ptr_out/*size_out. Returns false on
  // failure (leaves old allocation intact or sets ptr_out to nullptr on fresh).
  bool grow_gpu_buffer(void **ptr_out, size_t *size_out, size_t needed,
                       const char *debug_name);

  // Grow a pinned host buffer with 1.5× amortization.
  bool grow_host_buffer(void **ptr_out, size_t *size_out, size_t needed,
                        const char *debug_name);

public:
  // Set the HIP stream (borrowed; not owned). Called by
  // initialize_state_handles after the stream is created so grow_gpu_buffer can
  // sync before realloc.
  void set_stream(void *stream) { stream_ = stream; }
};

#endif // HIPDNN_EP_RUNTIME_MM_MEMORY_MANAGER_H
