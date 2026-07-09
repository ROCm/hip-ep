/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- memory_manager.cpp - Unified Memory Manager implementation ---------===//
//
// See memory_manager.h for the design.
//
// Wraps all six existing ensure_* patterns behind a typed API with uniform
// 1.5× amortized growth. All callers go through MM exclusively; the legacy
// RuntimeState fields were removed in Phase 2.
//
//===----------------------------------------------------------------------===//

#include "memory_manager.h"
#include "hal.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// HIP calls for GPU buffers that are NOT managed through the HAL (pool domains,
// shared workspace). These are pure VRAM allocations with no CPU backing
// needed.
#ifndef HIPDNN_EP_MM_MOCK_HAL
#include "runtime_types.h"

// Convenience wrappers used only within this file.
static inline bool hip_sync_stream(void *stream) {
  if (!stream)
    return true;
  return hipStreamSynchronize(static_cast<hipStream_t>(stream)) == hipSuccess;
}
static inline void *hip_malloc_gpu(size_t size) {
  void *ptr = nullptr;
  if (hipMalloc(&ptr, size) != hipSuccess)
    return nullptr;
  return ptr;
}
static inline void hip_free_gpu(void *ptr) {
  if (ptr)
    hipFree(ptr);
}
static inline void *hip_malloc_host(size_t size) {
  void *ptr = nullptr;
  if (hipHostMalloc(&ptr, size, hipHostMallocDefault) != hipSuccess)
    return nullptr;
  return ptr;
}
static inline void hip_free_host(void *ptr) {
  if (ptr)
    hipHostFree(ptr);
}
#else
// Mock stubs — unit-test path (no HIP SDK).
static inline bool hip_sync_stream(void * /*stream*/) { return true; }
static inline void *hip_malloc_gpu(size_t size) { return malloc(size); }
static inline void hip_free_gpu(void *ptr) { ::free(ptr); }
static inline void *hip_malloc_host(size_t size) { return malloc(size); }
static inline void hip_free_host(void *ptr) { ::free(ptr); }
#endif

//===----------------------------------------------------------------------===//
// Construction / destruction
//===----------------------------------------------------------------------===//

MemoryManager::MemoryManager(HalAllocator *hal) : hal_(hal) {}

MemoryManager::~MemoryManager() {
  // Free all pool domains.
  if (domains_) {
    for (int i = 0; i < num_pool_domains_; ++i) {
      if (domains_[i].base)
        hip_free_gpu(domains_[i].base);
    }
    ::free(domains_);
    domains_ = nullptr;
    num_pool_domains_ = 0;
  }
  ::free(buffer_offsets_);
  buffer_offsets_ = nullptr;

  // Free shared workspace.
  hip_free_gpu(workspace_);
  workspace_ = nullptr;
  workspace_size_ = 0;

  // Free host-scalar scratch (HAL-allocated — Mapped+NonCoherent).
  if (host_scratch_) {
    HalBlock b;
    b.cpu_ptr = host_scratch_;
    b.gpu_ptr = host_scratch_; // APU: same ptr; discrete path not used here
    b.size = host_scratch_size_;
    if (hal_)
      hal_->free(b);
    else
      hip_free_host(host_scratch_);
    host_scratch_ = nullptr;
    host_scratch_size_ = 0;
  }

  // Free QMoE pinned-host scratch.
  hip_free_host(qmoe_host_scratch_);
  qmoe_host_scratch_ = nullptr;
  qmoe_host_scratch_size_ = 0;

  delete hal_;
  hal_ = nullptr;
}

//===----------------------------------------------------------------------===//
// Pool management
//===----------------------------------------------------------------------===//

bool MemoryManager::ensure_domain(int domain_id) {
  if (domain_id < num_pool_domains_)
    return true;
  int new_count = domain_id + 1;
  auto *new_domains = static_cast<PoolDomain *>(
      realloc(domains_, sizeof(PoolDomain) * new_count));
  if (!new_domains) {
    fprintf(stderr,
            "MemoryManager::ensure_domain: realloc failed (%d domains)\n",
            new_count);
    return false;
  }
  domains_ = new_domains;
  for (int i = num_pool_domains_; i < new_count; ++i)
    domains_[i] = PoolDomain{};
  num_pool_domains_ = new_count;
  return true;
}

bool MemoryManager::grow_gpu_buffer(void **ptr_out, size_t *size_out,
                                    size_t needed, const char *debug_name) {
  if (*size_out >= needed)
    return true;

  // 1.5× amortized growth: avoids per-inference realloc cycles when shapes
  // grow by a small increment each token (e.g. GQA decode total_seq).
  // Cold-start (no existing allocation) keeps the exact requested size.
  size_t new_size = needed;
  if (*size_out > 0)
    new_size = std::max(needed, *size_out + *size_out / 2);

  if (*ptr_out) {
    hip_sync_stream(stream_);
    fprintf(stderr,
            "MemoryManager: growing %s %zu -> %zu bytes "
            "(rare; new input shape)\n",
            debug_name, *size_out, new_size);
    fflush(stderr);
    hip_free_gpu(*ptr_out);
  }

  void *new_ptr = hip_malloc_gpu(new_size);
  if (!new_ptr) {
    fprintf(stderr, "MemoryManager: hipMalloc failed for %s (%zu bytes)\n",
            debug_name, new_size);
    *ptr_out = nullptr;
    *size_out = 0;
    return false;
  }
  *ptr_out = new_ptr;
  *size_out = new_size;
  return true;
}

bool MemoryManager::grow_host_buffer(void **ptr_out, size_t *size_out,
                                     size_t needed, const char *debug_name) {
  if (*size_out >= needed)
    return true;

  size_t new_size = needed;
  if (*size_out > 0)
    new_size = std::max(needed, *size_out + *size_out / 2);

  if (*ptr_out) {
    hip_sync_stream(stream_);
    fprintf(stderr, "MemoryManager: growing host buffer %s %zu -> %zu bytes\n",
            debug_name, *size_out, new_size);
    fflush(stderr);
    hip_free_host(*ptr_out);
  }

  void *new_ptr = hip_malloc_host(new_size);
  if (!new_ptr) {
    fprintf(stderr, "MemoryManager: hipHostMalloc failed for %s (%zu bytes)\n",
            debug_name, new_size);
    *ptr_out = nullptr;
    *size_out = 0;
    return false;
  }
  *ptr_out = new_ptr;
  *size_out = new_size;
  return true;
}

void MemoryManager::load_pool_plan(size_t pool_size,
                                   const size_t *buffer_offsets,
                                   size_t num_buffers) {
  static_pool_size_ = pool_size;
  num_buffers_ = num_buffers;

  ::free(buffer_offsets_);
  buffer_offsets_ = nullptr;
  if (num_buffers > 0 && buffer_offsets) {
    buffer_offsets_ =
        static_cast<size_t *>(malloc(sizeof(size_t) * num_buffers));
    if (buffer_offsets_)
      memcpy(buffer_offsets_, buffer_offsets, sizeof(size_t) * num_buffers);
  }

  // Eagerly size domain 0 to match static_pool_size_ (matches the old
  // hipdnn_ep_pool_init contract: domain 0 is pre-allocated, domains 1..N
  // grow lazily). The actual hipMalloc is deferred to the first get_pool_base
  // call so that zero-pool models don't allocate anything.
}

void *MemoryManager::get_pool_base(int domain_id, size_t needed_size) {
  if (domain_id < 0) {
    fprintf(stderr,
            "MemoryManager::get_pool_base: negative domain_id %d "
            "(compiler bug)\n",
            domain_id);
    return nullptr;
  }
  if (!ensure_domain(domain_id))
    return nullptr;

  if (needed_size == 0)
    needed_size = 1; // hipMalloc(0) is undefined

  char name[64];
  snprintf(name, sizeof(name), "pool[%d]", domain_id);
  if (!grow_gpu_buffer(&domains_[domain_id].base, &domains_[domain_id].size,
                       needed_size, name))
    return nullptr;

  return domains_[domain_id].base;
}

void *MemoryManager::get_buffer_from_pool(size_t index) {
  if (num_pool_domains_ < 1 || !domains_[0].base) {
    fprintf(stderr,
            "MemoryManager::get_buffer_from_pool: pool not initialized\n");
    return nullptr;
  }
  if (index >= num_buffers_) {
    fprintf(stderr,
            "MemoryManager::get_buffer_from_pool: index %zu out of range\n",
            index);
    return nullptr;
  }
  if (!buffer_offsets_)
    return domains_[0].base;
  return static_cast<char *>(domains_[0].base) + buffer_offsets_[index];
}

//===----------------------------------------------------------------------===//
// Host-scalar scratch
//===----------------------------------------------------------------------===//

void *MemoryManager::get_host_scratch(size_t needed_size) {
  if (needed_size == 0)
    needed_size = 1;

  if (needed_size <= host_scratch_size_)
    return host_scratch_;

  // Grow: use the HAL allocator so that the correct hipHostMalloc flags are
  // used (Mapped+NonCoherent on APU; plain host on discrete). We allocate a
  // fresh block and copy nothing (host scratch is re-initialized every
  // inference by MaterializeHostScalars-emitted stores).
  if (host_scratch_) {
    hip_sync_stream(stream_);
    HalBlock old_block;
    old_block.cpu_ptr = host_scratch_;
    old_block.gpu_ptr = host_scratch_;
    old_block.size = host_scratch_size_;
    if (hal_)
      hal_->free(old_block);
    else
      hip_free_host(host_scratch_);
    host_scratch_ = nullptr;
    host_scratch_size_ = 0;
  }

  HalBlock block;
  if (hal_) {
    block = hal_->alloc(needed_size, MemTier::GPU);
  } else {
    block.cpu_ptr = hip_malloc_host(needed_size);
    block.gpu_ptr = block.cpu_ptr;
  }

  if (!block.cpu_ptr) {
    fprintf(stderr,
            "MemoryManager::get_host_scratch: alloc failed (%zu bytes)\n",
            needed_size);
    return nullptr;
  }
  host_scratch_ = block.cpu_ptr;
  host_scratch_size_ = needed_size;
  return host_scratch_;
}

//===----------------------------------------------------------------------===//
// Scratch arena (bump-pointer allocator over the shared workspace)
//===----------------------------------------------------------------------===//

static constexpr size_t kScratchAlignment = 64;

void *MemoryManager::scratch_alloc(size_t size) {
  if (size == 0)
    return workspace_ ? static_cast<char *>(workspace_) + scratch_offset_
                      : nullptr;

  size_t aligned = (size + kScratchAlignment - 1) & ~(kScratchAlignment - 1);
  size_t needed = scratch_offset_ + aligned;
  if (needed > workspace_size_) {
    // Growth frees the old buffer and allocates a new one. Any pointers
    // returned by prior scratch_alloc calls in this sequence become invalid.
    // Callers must call scratch_reserve(total) up front to avoid mid-sequence
    // growth, or re-derive all pointers after any alloc that might grow.
    if (!grow_gpu_buffer(&workspace_, &workspace_size_, needed, "workspace"))
      return nullptr;
  }

  void *ptr = static_cast<char *>(workspace_) + scratch_offset_;
  scratch_offset_ += aligned;
  return ptr;
}

bool MemoryManager::scratch_reserve(size_t total) {
  if (total <= workspace_size_)
    return true;
  return grow_gpu_buffer(&workspace_, &workspace_size_, total, "workspace");
}

void MemoryManager::scratch_reset() { scratch_offset_ = 0; }

//===----------------------------------------------------------------------===//
// Shared GPU workspace (low-level; prefer scratch_alloc for new code)
//===----------------------------------------------------------------------===//

void *MemoryManager::ensure_workspace(size_t needed_size) {
  if (needed_size == 0)
    return workspace_;
  if (!grow_gpu_buffer(&workspace_, &workspace_size_, needed_size, "workspace"))
    return nullptr;
  return workspace_;
}

void *MemoryManager::get_workspace() const { return workspace_; }
size_t MemoryManager::get_workspace_size() const { return workspace_size_; }

//===----------------------------------------------------------------------===//
// QMoE pinned-host scratch
//===----------------------------------------------------------------------===//

void *MemoryManager::ensure_qmoe_host_scratch(size_t needed_size) {
  if (needed_size == 0)
    return qmoe_host_scratch_;
  if (!grow_host_buffer(&qmoe_host_scratch_, &qmoe_host_scratch_size_,
                        needed_size, "qmoe_host_scratch"))
    return nullptr;
  return qmoe_host_scratch_;
}

void *MemoryManager::get_qmoe_host_scratch() const {
  return qmoe_host_scratch_;
}

//===----------------------------------------------------------------------===//
// Per-inference lifecycle
//===----------------------------------------------------------------------===//

void MemoryManager::begin_compute() {
  scratch_reset();
  seqlens_k_valid_ = false;
  seqlens_k_ptr_ = nullptr;
}

void MemoryManager::end_compute() {
  // Reserved for future phases that need end-of-inference cleanup.
}

//===----------------------------------------------------------------------===//
// seqlens_k cache
//===----------------------------------------------------------------------===//

bool MemoryManager::seqlens_k_cache_valid(const void *ptr) const {
  return seqlens_k_valid_ && (seqlens_k_ptr_ == ptr);
}

int32_t MemoryManager::seqlens_k_cached_val() const { return seqlens_k_val_; }

void MemoryManager::seqlens_k_cache_set(const void *ptr, int32_t val) {
  seqlens_k_ptr_ = ptr;
  seqlens_k_val_ = val;
  seqlens_k_valid_ = true;
}

//===----------------------------------------------------------------------===//
// KV cache buffer tracking
//===----------------------------------------------------------------------===//

void MemoryManager::register_kv_buffer(void *ptr, size_t size) {
  if (!ptr || size == 0)
    return;
  for (int i = 0; i < kv_buffer_count_; ++i) {
    if (kv_entries_[i].ptr == ptr)
      return;
  }
  if (kv_buffer_count_ >= kMaxKvBuffers) {
    fprintf(stderr,
            "MemoryManager::register_kv_buffer: capacity exceeded (%d)\n",
            kMaxKvBuffers);
    return;
  }
  kv_entries_[kv_buffer_count_++] = {ptr, size};
  kv_bytes_total_ += size;
}

// C-linkage wrappers so the allocator module (morphizen/ort-bridge) can call
// register/unregister without including mm/memory_manager.h.
extern "C" void hipdnn_ep_mm_register_kv_buffer(void *mm, void *ptr,
                                                size_t size) {
  if (mm)
    static_cast<MemoryManager *>(mm)->register_kv_buffer(ptr, size);
}

extern "C" void hipdnn_ep_mm_unregister_kv_buffer(void *mm, void *ptr) {
  if (mm)
    static_cast<MemoryManager *>(mm)->unregister_kv_buffer(ptr);
}

void MemoryManager::unregister_kv_buffer(void *ptr) {
  if (!ptr)
    return;
  for (int i = 0; i < kv_buffer_count_; ++i) {
    if (kv_entries_[i].ptr == ptr) {
      kv_bytes_total_ -= kv_entries_[i].size;
      kv_entries_[i] = kv_entries_[--kv_buffer_count_];
      return;
    }
  }
}

//===----------------------------------------------------------------------===//
// Stats
//===----------------------------------------------------------------------===//

size_t MemoryManager::gpu_bytes_used() const {
  size_t total = 0;
  for (int i = 0; i < num_pool_domains_; ++i)
    total += domains_[i].size;
  total += workspace_size_;
  total += kv_bytes_total_;
  return total;
}

size_t MemoryManager::cpu_bytes_used() const {
  return host_scratch_size_ + qmoe_host_scratch_size_;
}
